"""Tensor-parallel (Megatron-style) sharding for the Mobile-VTON Attention module.

column-parallel to_q/to_k/to_v (each rank keeps its head slice; no comm) +
row-parallel to_out (each rank computes a partial output; one all_reduce sums
them). Per attention: exactly one all_reduce of [B, S, out_dim].

    shard_attention_(attn, rank, world)       # slice weights in place
    attn.set_processor(TPAttnProcessor(group)) # processor that all_reduces to_out

MVP limits (see PROGRESS.md): standard MHA only (kv_heads == heads), no rope.
"""
import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers.models.embeddings import apply_rotary_emb


def _slice_rows(linear, start, end):
    """Keep output features [start:end] of an nn.Linear (column-parallel)."""
    linear.weight = torch.nn.Parameter(linear.weight.data[start:end].clone())
    if linear.bias is not None:
        linear.bias = torch.nn.Parameter(linear.bias.data[start:end].clone())
    linear.out_features = end - start


def _slice_cols(linear, start, end, keep_bias):
    """Keep input features [start:end] of an nn.Linear (row-parallel).

    Bias is applied to the FULL output, so it must be counted once across ranks:
    keep it on rank 0, zero it elsewhere -> after all_reduce the bias is correct.
    """
    linear.weight = torch.nn.Parameter(linear.weight.data[:, start:end].clone())
    if linear.bias is not None and not keep_bias:
        linear.bias = torch.nn.Parameter(torch.zeros_like(linear.bias.data))
    linear.in_features = end - start


def shard_attention_(attn, rank, world):
    """In-place TP shard of one Attention module across `world` ranks.

    Query heads are always split (column-parallel to_q + row-parallel to_out).
    KV heads (G): split if G % world == 0, otherwise REPLICATED on every rank
    (handles MQA G=1 / GQA G<world; kv projections are tiny so replication is
    cheap). The processor repeat_interleaves kv to match local query heads.
    """
    H = attn.heads
    assert H % world == 0, f"query heads {H} not divisible by world {world}"
    head_dim = attn.inner_dim // H
    G = attn.inner_kv_dim // head_dim
    localH = H // world
    qs, qe = rank * localH * head_dim, (rank + 1) * localH * head_dim
    _slice_rows(attn.to_q, qs, qe)
    if G % world == 0 and G >= world:
        localG = G // world
        ks, ke = rank * localG * head_dim, (rank + 1) * localG * head_dim
        _slice_rows(attn.to_k, ks, ke)
        _slice_rows(attn.to_v, ks, ke)
        attn.inner_kv_dim = localG * head_dim
    # else: G < world (e.g. MQA G=1) -> replicate kv (leave to_k/to_v full)
    _slice_cols(attn.to_out[0], qs, qe, keep_bias=(rank == 0))
    attn.heads = localH
    attn.inner_dim = localH * head_dim
    return attn


def _is_attention(m):
    return (all(hasattr(m, a) for a in
                ("to_q", "to_k", "to_v", "to_out", "heads", "inner_dim", "inner_kv_dim"))
            and getattr(m, "to_k", None) is not None
            and getattr(m, "to_out", None) is not None)


def iter_attention_modules(model):
    """Yield (name, module) for every Attention-like submodule."""
    for name, m in model.named_modules():
        if _is_attention(m):
            yield name, m


def tp_supported(attn, world):
    """True if this Attention can be TP-sharded across `world` ranks.

    Only requirement: query heads divisible by world. KV (MHA/GQA/MQA) is handled
    by shard-or-replicate.
    """
    return attn.heads % world == 0


def shard_denoiser_attention_(model, rank, world):
    """TP-shard every supported Attention in `model`. Returns (sharded, skipped) names."""
    sharded, skipped = [], []
    for name, attn in iter_attention_modules(model):
        if not tp_supported(attn, world):
            skipped.append(name)
            continue
        shard_attention_(attn, rank, world)
        attn.set_processor(TPAttnProcessor())
        sharded.append(name)
    return sharded, skipped


class TPAttnProcessor:
    """AttnProcessor2_0 + row-parallel to_out all_reduce. Use after shard_attention_."""

    def __init__(self, group=None):
        self.group = group  # gloo process group (None -> default WORLD)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, image_rotary_emb=None,
                 *args, **kwargs):
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)

        batch_size, seq, _ = (
            hidden_states.shape if encoder_hidden_states is None
            else encoder_hidden_states.shape
        )
        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, seq, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1,
                                                 attention_mask.shape[-1])
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        head_dim = query.shape[-1] // attn.heads
        kv_heads = key.shape[-1] // head_dim
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        if kv_heads != attn.heads:  # GQA/MQA: replicate kv to match local query heads
            rep = attn.heads // kv_heads
            key = torch.repeat_interleave(key, rep, dim=1)
            value = torch.repeat_interleave(value, rep, dim=1)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # rope: applied per-head on head_dim (never mixes heads), so running it on
        # each rank's LOCAL query/key heads is identical to the full-model result.
        # Logic mirrors the real Attn processor (prefix/suffix split when the rope
        # table is shorter than the sequence; key only for self-attention).
        if image_rotary_emb is not None:
            S_rope = image_rotary_emb[0].shape[0]
            S = query.shape[2]
            if S_rope < S:
                q_prefix = apply_rotary_emb(query[:, :, :S_rope, :], image_rotary_emb)
                q_suffix = apply_rotary_emb(query[:, :, S_rope:, :], image_rotary_emb)
                query = torch.cat([q_prefix, q_suffix], dim=2)
            else:
                query = apply_rotary_emb(query, image_rotary_emb)
            if not attn.is_cross_attention:
                if S_rope < S:
                    k_prefix = apply_rotary_emb(key[:, :, :S_rope, :], image_rotary_emb)
                    k_suffix = apply_rotary_emb(key[:, :, S_rope:, :], image_rotary_emb)
                    key = torch.cat([k_prefix, k_suffix], dim=2)
                else:
                    key = apply_rotary_emb(key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim).to(query.dtype)

        # row-parallel: each rank produces a PARTIAL output, sum across ranks.
        hidden_states = attn.to_out[0](hidden_states)
        dist.all_reduce(hidden_states, op=dist.ReduceOp.SUM, group=self.group)
        hidden_states = attn.to_out[1](hidden_states)  # dropout (eval: no-op)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(b, c, h, w)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor
