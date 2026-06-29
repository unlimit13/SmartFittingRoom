"""Spatial (height-axis) parallelism primitives for the Mobile-VTON denoiser.

DIFFERENT axis from the channel-TP code (tp_*.py): here the WEIGHTS stay full and
replicated on every rank, and we split the ACTIVATION feature maps along H (rows).
Each rank owns a contiguous band of rows [h0:h1] of every [B,C,H,W] tensor. Only
ops whose math crosses the H boundary need communication:
  - kernel>1 convs   -> halo exchange of (k-1)//2 border rows  (sp_conv.py)
  - GroupNorm        -> all_reduce of per-group (sum, sqsum, count) (sp_groupnorm.py)
  - self-attention   -> all_gather of the image K/V over the band (sp_attention.py)
Everything else (1x1 conv, Linear/FFN, SiLU, add, channel-concat skips, time embed,
cross-attention to garment/text) is row-independent and runs unchanged per band.

Rank order is TOP->BOTTOM: rank 0 owns the top rows, rank world-1 the bottom rows.
Input is scattered by rows once at the denoiser entry; output gathered once at exit.
"""
import torch
import torch.distributed as dist


import os


def band_bounds(H, rank, world):
    """Contiguous row band [h0, h1) this rank owns. Splits H as evenly as possible;
    earlier ranks get the +1 when H is not divisible (kept deterministic so every
    rank agrees on every other rank's band size).

    For world==2 an uneven split can be requested via SP_BAND_FRAC0 (rank 0's
    fraction of H, default 0.5) -- useful when one Pi has less free RAM. The split
    stays deterministic and contiguous, so halo/offset math is unaffected."""
    if world == 2:
        frac0 = float(os.environ.get("SP_BAND_FRAC0", "0.5"))
        split = max(1, min(H - 1, int(round(H * frac0))))
        return (0, split) if rank == 0 else (split, H)
    base, rem = divmod(H, world)
    h0 = rank * base + min(rank, rem)
    h1 = h0 + base + (1 if rank < rem else 0)
    return h0, h1


def band_size(H, rank, world):
    h0, h1 = band_bounds(H, rank, world)
    return h1 - h0


def scatter_rows(x_full, rank, world, group=None):
    """Take the full [B,C,H,W] tensor (identical on every rank) and return this
    rank's row band. No comm: every rank already holds x_full, just slices."""
    h0, h1 = band_bounds(x_full.shape[-2], rank, world)
    return x_full[:, :, h0:h1, :].contiguous()


def _all_gather_sizes(local_size, world, group=None):
    """Gather each rank's size along the split dim with a tiny int all_gather
    (no pickle / all_gather_object churn)."""
    t = torch.tensor([local_size], dtype=torch.int64)
    out = [torch.zeros(1, dtype=torch.int64) for _ in range(world)]
    dist.all_gather(out, t, group=group)
    return [int(o.item()) for o in out]


def gather_rows(x_local, rank, world, group=None):
    """Concatenate every rank's row band back into the full [B,C,H,W] tensor on
    every rank (all_gather along H). Bands may differ in H."""
    if world == 1:
        return x_local
    sizes = _all_gather_sizes(x_local.shape[-2], world, group)
    parts = []
    for r in range(world):
        if r == rank:
            buf = x_local.contiguous()
        else:
            shp = list(x_local.shape)
            shp[-2] = sizes[r]
            buf = torch.empty(shp, dtype=x_local.dtype, device=x_local.device)
        parts.append(buf)
    for r in range(world):
        dist.broadcast(parts[r], src=r, group=group)
    return torch.cat(parts, dim=-2)


def exchange_halo(x, pad, group=None):
    """Return x extended along H with `pad` border rows on top and bottom.

    Interior boundaries get the neighbor's real rows; the global top (rank 0) and
    global bottom (rank world-1) get ZERO rows so a following kernel>1 conv with
    its H-padding removed reproduces the single-Pi zero-padded receptive field
    exactly. x: [B,C,h,W] -> [B,C,h+2*pad,W].
    """
    if not (dist.is_available() and dist.is_initialized()):
        # single process: just zero-pad both ends (matches conv padding)
        z = torch.zeros_like(x[:, :, :pad, :])
        return torch.cat([z, x, z], dim=-2)
    rank = dist.get_rank(group)
    world = dist.get_world_size(group)
    if world == 1:
        z = torch.zeros_like(x[:, :, :pad, :])
        return torch.cat([z, x, z], dim=-2)

    top = torch.zeros_like(x[:, :, :pad, :])   # filled from rank-1 if interior
    bot = torch.zeros_like(x[:, :, :pad, :])   # filled from rank+1 if interior
    send_top = x[:, :, :pad, :].contiguous()   # my first rows -> rank-1's bottom halo
    send_bot = x[:, :, -pad:, :].contiguous()  # my last rows  -> rank+1's top halo

    reqs = []
    if rank > 0:
        reqs.append(dist.isend(send_top, rank - 1, group=group, tag=10))
        reqs.append(dist.irecv(top, rank - 1, group=group, tag=20))
    if rank < world - 1:
        reqs.append(dist.isend(send_bot, rank + 1, group=group, tag=20))
        reqs.append(dist.irecv(bot, rank + 1, group=group, tag=10))
    for r in reqs:
        r.wait()
    # top/bot are zero at the global edges (rank 0 top, rank world-1 bottom),
    # which is exactly the zero padding single-Pi conv applies there.
    return torch.cat([top, x, bot], dim=-2)
