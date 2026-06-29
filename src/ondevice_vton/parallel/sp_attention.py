"""Spatial (H-band / sequence) parallel attention for the Mobile-VTON denoiser.

Weights stay FULL/replicated; the image token sequence is split by H-band (each
rank owns a contiguous block of the row-major H*W tokens). Per attention site:

  cross-attention (attn2, encoder_hidden_states given): query = this rank's person
    band, key/value = text/image embeds (replicated). Pure query-parallel -> NO
    comm. Only the query's RoPE must use the band's absolute position slice.

  self-attention (attn1, encoder_hidden_states is None): the block feeds
    cat[person_band, garment] (garment replicated, appended along seq). Each rank
    must attend its band queries against ALL person keys + garment keys, so the
    PERSON key/value are all_gathered across ranks (MQA -> tiny), garment kv kept
    once. Output is per-query -> sharded by band, NO all_reduce. RoPE: band-offset
    slice for the person queries, full table for the gathered person keys, none for
    garment.

Geometry (Lp = band person tokens, offset, Pfull = full person tokens, W) is stashed
by the patched Transformer2DModel.forward into SP_CTX just before its blocks run.

    shard_denoiser_spatial_attention_(model, group)  # patch transformers + attn procs
"""
import torch
import torch.nn.functional as F
import torch.distributed as dist
from diffusers.models.embeddings import apply_rotary_emb

from parallel.sp_common import band_bounds

# Geometry for the transformer currently executing (set by the patched
# Transformer2DModel.forward; blocks run synchronously so a module global is safe).
SP_CTX = {"Lp": None, "offset": 0, "Pfull": None, "W": None, "group": None}


def _rw(group):
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(group), dist.get_world_size(group)
    return 0, 1


def _gather_seq(x, group):
    """all_gather a [B, L, D] tensor along dim=1 (L may differ per rank). Lean int
    all_gather for sizes (no pickle)."""
    rank, world = _rw(group)
    if world == 1:
        return x
    t = torch.tensor([x.shape[1]], dtype=torch.int64)
    so = [torch.zeros(1, dtype=torch.int64) for _ in range(world)]
    dist.all_gather(so, t, group=group)
    sizes = [int(o.item()) for o in so]
    parts = []
    for r in range(world):
        if r == rank:
            buf = x.contiguous()
        else:
            shp = list(x.shape)
            shp[1] = sizes[r]
            buf = torch.empty(shp, dtype=x.dtype, device=x.device)
        parts.append(buf)
    for r in range(world):
        dist.broadcast(parts[r], src=r, group=group)
    return torch.cat(parts, dim=1)


def _apply_rope_prefix(t, cos_sin, n):
    """Apply RoPE to the first n tokens of t [B, H, S, hd] using cos_sin sliced to
    n rows; leave the rest (e.g. garment) untouched."""
    if cos_sin is None or n <= 0:
        return t
    pre = apply_rotary_emb(t[:, :, :n, :], cos_sin)
    if n < t.shape[2]:
        return torch.cat([pre, t[:, :, n:, :]], dim=2)
    return pre


def _apply_rope_pre_suf(t, cos_sin, n):
    """Original processor's prefix/suffix RoPE: apply the table to the first n rows
    (person) AND, if the sequence is longer (garment suffix), apply it again to the
    rest. Mirrors AttnProcessor2_0 lines 890-893."""
    cos, sin = cos_sin
    S = t.shape[2]
    if n < S:
        pre = apply_rotary_emb(t[:, :, :n, :], (cos, sin))
        suf = apply_rotary_emb(t[:, :, n:, :], (cos, sin))
        return torch.cat([pre, suf], dim=2)
    return apply_rotary_emb(t, (cos, sin))


class SPAttnProcessor:
    """AttnProcessor2_0 with H-band spatial parallelism. Mirrors the model's
    AttnProcessor2_0 exactly, but: the image queries are this rank's band only, and
    for SELF-attention the person key/value are all_gathered to the full sequence so
    each band query attends to every person token (+ replicated garment). Garment
    QUERIES are dropped (their outputs are discarded by the block) -> query stays
    band-sized, so SDPA keeps the memory-efficient kernel."""

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, image_rotary_emb=None,
                 *args, **kwargs):
        ctx = SP_CTX
        group = ctx["group"]
        is_self = encoder_hidden_states is None

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)

        batch_size = hidden_states.shape[0]
        if attention_mask is not None:
            seq_len = hidden_states.shape[1] if encoder_hidden_states is None \
                else encoder_hidden_states.shape[1]
            attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        Lp = ctx["Lp"]                       # person tokens in this band
        offset = ctx["offset"]               # absolute token offset of the band
        Pfull = ctx["Pfull"]                 # full person token count

        # QUERY: this rank's person band only (drop garment queries -> outputs unused)
        q_in = hidden_states[:, :Lp, :] if is_self else hidden_states
        query = attn.to_q(q_in)

        if encoder_hidden_states is None:
            kv_in = hidden_states                       # self: [person_band + garment]
        else:
            kv_in = attn.norm_encoder_hidden_states(encoder_hidden_states) \
                if attn.norm_cross else encoder_hidden_states
        key = attn.to_k(kv_in)
        value = attn.to_v(kv_in)

        head_dim = query.shape[-1] // attn.heads
        kv_heads = key.shape[-1] // head_dim
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)

        if is_self:
            # all_gather the PERSON band key/value (kv_heads small -> cheap) to the
            # full person sequence, then re-append the replicated garment kv. The
            # result is byte-identical in layout to the single-Pi [person+garment] kv.
            kp, kg = key[:, :, :Lp, :], key[:, :, Lp:, :]
            vp, vg = value[:, :, :Lp, :], value[:, :, Lp:, :]
            kp = kp.transpose(1, 2).reshape(batch_size, Lp, kv_heads * head_dim)
            vp = vp.transpose(1, 2).reshape(batch_size, Lp, kv_heads * head_dim)
            kp = _gather_seq(kp, group).view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
            vp = _gather_seq(vp, group).view(batch_size, -1, kv_heads, head_dim).transpose(1, 2)
            key = torch.cat([kp, kg], dim=2)
            value = torch.cat([vp, vg], dim=2)

        if kv_heads != attn.heads:  # MQA/GQA: expand kv to query heads (as in original)
            rep = attn.heads // kv_heads
            key = torch.repeat_interleave(key, rep, dim=1)
            value = torch.repeat_interleave(value, rep, dim=1)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if image_rotary_emb is not None:
            cos, sin = image_rotary_emb
            # query: this band's absolute positions
            query = apply_rotary_emb(query, (cos[offset:offset + Lp], sin[offset:offset + Lp]))
            if is_self:
                # key now == single-Pi [person_full + garment]; apply the original
                # prefix/suffix table (person prefix + garment suffix).
                key = _apply_rope_pre_suf(key, (cos, sin), Pfull)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim).to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(b, c, h, w)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


def make_sp_transformer_forward(tfm, group=None):
    """Wrap a Transformer2DModel.forward to stash band geometry into SP_CTX, then
    call the original forward. Geometry comes from the (banded) input shape + the
    full RoPE table length."""
    orig = type(tfm).forward

    def forward(hidden_states, *args, **kwargs):
        rope = kwargs.get("image_rotary_emb", None)
        if rope is None and len(args) >= 4:
            rope = args[3]
        localH, W = hidden_states.shape[-2], hidden_states.shape[-1]
        rank, world = _rw(group)
        if rope is not None:
            Pfull = rope[0].shape[0]
            fullH = Pfull // W
            off_rows = band_bounds(fullH, rank, world)[0]
            SP_CTX.update(Lp=localH * W, offset=off_rows * W, Pfull=Pfull,
                          W=W, group=group)
        else:
            SP_CTX.update(Lp=localH * W, offset=0, Pfull=localH * W, W=W, group=group)
        return orig(tfm, hidden_states, *args, **kwargs)
    return forward


def _iter_transformers(model):
    for m in model.modules():
        if type(m).__name__ == "Transformer2DModel":
            yield m


def _is_attention(m):
    return all(hasattr(m, a) for a in ("to_q", "to_k", "to_v", "to_out", "heads")) \
        and getattr(m, "to_k", None) is not None


def shard_denoiser_spatial_attention_(model, group=None):
    """Patch every denoiser Transformer2DModel (geometry stash) and set the SP
    processor on its attentions. Attentions outside transformers (e.g. the image
    Resampler) are left replicated. Returns (n_transformers, n_attn)."""
    n_t = n_a = 0
    for tfm in _iter_transformers(model):
        tfm.forward = make_sp_transformer_forward(tfm, group=group)
        n_t += 1
        for sub in tfm.modules():
            if _is_attention(sub):
                sub.set_processor(SPAttnProcessor())
                n_a += 1
    return n_t, n_a
