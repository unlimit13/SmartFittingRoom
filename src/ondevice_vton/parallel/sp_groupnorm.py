"""Spatial-parallel GroupNorm: sync per-group statistics across the H-band split.

GroupNorm normalizes each (sample, group) over its (C/G channels x H x W) elements.
When we split H across ranks, each rank holds ALL channels but only a slice of H,
so its per-group sum/sqsum/count are PARTIAL along the spatial axis. The channel-TP
"whole-group local stats" trick does NOT apply here (that split channels, leaving
spatial whole). We must all_reduce (sum, sqsum, count) per (sample, group) to get
the global mean/var, then normalize the local rows and apply affine locally.

Matches torch.nn.GroupNorm exactly (biased variance, same eps).

    shard_denoiser_groupnorm_(model, group)   # patch every GroupNorm in place
"""
import torch
import torch.distributed as dist


def _sp_groupnorm_forward(gn, x, group=None):
    # x: [B, C, h, W]  (h = this rank's row band)
    B, C, h, W = x.shape
    G = gn.num_groups
    cg = C // G
    xg = x.view(B, G, cg, h, W)
    local_sum = xg.sum(dim=(2, 3, 4))            # [B, G]
    local_sqsum = (xg * xg).sum(dim=(2, 3, 4))   # [B, G]
    local_count = x.new_full((B, G), float(cg * h * W))
    stats = torch.stack([local_sum, local_sqsum, local_count], dim=-1)  # [B,G,3]
    if dist.is_available() and dist.is_initialized() and dist.get_world_size(group) > 1:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM, group=group)
    tot = stats[..., 2]                            # [B,G] global element count
    mean = stats[..., 0] / tot                     # [B,G]
    var = stats[..., 1] / tot - mean * mean        # [B,G] biased, == torch GroupNorm
    mean = mean.view(B, G, 1, 1, 1)
    inv = torch.rsqrt(var.view(B, G, 1, 1, 1) + gn.eps)
    xhat = (xg - mean) * inv
    out = xhat.view(B, C, h, W)
    if gn.affine:
        out = out * gn.weight.view(1, C, 1, 1) + gn.bias.view(1, C, 1, 1)
    return out


def make_sp_groupnorm_forward(gn, group=None):
    def forward(x):
        return _sp_groupnorm_forward(gn, x, group=group)
    return forward


def shard_denoiser_groupnorm_(model, group=None):
    """Replace every GroupNorm.forward with the spatial-parallel version. Returns
    the number of GroupNorm modules patched."""
    n = 0
    for m in model.modules():
        if isinstance(m, torch.nn.GroupNorm):
            m.forward = make_sp_groupnorm_forward(m, group=group)
            n += 1
    return n
