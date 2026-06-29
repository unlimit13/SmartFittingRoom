"""Stage 2: tensor-parallel (Megatron-MLP) sharding for the transformer FFN.

The feed-forward is ~48% of the denoiser params and was fully REPLICATED by
Stages 1/1b -> the dominant Amdahl floor. FeedForward.net = [GEGLU, Dropout,
Linear(inner->dim)]. GEGLU.proj is Linear(dim -> 2*inner) and forward does
`proj(x).chunk(2)` -> (hidden, gate) -> hidden * gelu(gate).

Megatron-MLP shard, splitting the `inner` dimension:
  GEGLU.proj : COLUMN-parallel, but the output is [hidden(inner) | gate(inner)],
               so each rank takes a slice of BOTH halves (interleaved) -> its
               local activation is exactly the matching slice of hidden*gelu(gate).
  out Linear : ROW-parallel (slice inner in) -> partial dim out -> 1 all_reduce.

Every inner-dim intermediate stays consistently sliced, so the original
FeedForward.forward runs unchanged; the all_reduce is a forward-hook on the
output Linear. One all_reduce per FFN.

    shard_ffn_(ff, rank, world)
    shard_denoiser_ffn_(model, rank, world)
"""
import torch
import torch.nn as nn
import torch.distributed as dist


def _is_geglu(m):
    return hasattr(m, "proj") and isinstance(getattr(m, "proj", None), nn.Linear) \
        and "GEGLU" in type(m).__name__


def _ffn_parts(ff):
    """Return (geglu, out_linear) or (None, None) if not a shardable GEGLU FFN."""
    if not hasattr(ff, "net") or len(ff.net) < 1 or not _is_geglu(ff.net[0]):
        return None, None
    linears = [m for m in ff.net if isinstance(m, nn.Linear)]  # only the out proj
    if len(linears) != 1:
        return None, None
    return ff.net[0], linears[0]


def tp_ffn_supported(ff, world):
    geglu, out = _ffn_parts(ff)
    if geglu is None:
        return False
    inner = geglu.proj.out_features // 2
    return inner % world == 0 and out.in_features == inner


def shard_ffn_(ff, rank, world):
    """In-place Megatron-MLP shard of one GEGLU FeedForward across `world` ranks."""
    geglu, out = _ffn_parts(ff)
    inner = geglu.proj.out_features // 2
    local = inner // world
    s, e = rank * local, (rank + 1) * local

    # column-parallel proj: take the rank's slice from BOTH the hidden [0:inner]
    # and gate [inner:2*inner] halves so chunk(2) yields the matching local slices.
    w = geglu.proj.weight.data
    rows = torch.cat([w[s:e], w[inner + s:inner + e]], dim=0)
    geglu.proj.weight = nn.Parameter(rows.clone())
    if geglu.proj.bias is not None:
        b = geglu.proj.bias.data
        geglu.proj.bias = nn.Parameter(torch.cat([b[s:e], b[inner + s:inner + e]]).clone())
    geglu.proj.out_features = 2 * local

    # row-parallel out Linear: slice input cols; keep bias on rank 0 only so the
    # all_reduce counts it once.
    out.weight = nn.Parameter(out.weight.data[:, s:e].clone())
    if out.bias is not None and rank != 0:
        out.bias = nn.Parameter(torch.zeros_like(out.bias.data))
    out.in_features = local

    def _all_reduce_hook(_m, _i, o):
        dist.all_reduce(o, op=dist.ReduceOp.SUM)
        return o

    out.register_forward_hook(_all_reduce_hook)
    return ff


def iter_ffn_modules(model):
    from Mobile_VTON.models.attention import FeedForward
    for name, m in model.named_modules():
        if isinstance(m, FeedForward):
            yield name, m


def shard_denoiser_ffn_(model, rank, world):
    """Shard every supported GEGLU FFN. Returns (done, skipped) names."""
    done, skip = [], []
    for name, ff in iter_ffn_modules(model):
        if tp_ffn_supported(ff, world):
            shard_ffn_(ff, rank, world)
            done.append(name)
        else:
            skip.append(name)
    return done, skip
