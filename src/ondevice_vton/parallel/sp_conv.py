"""Spatial-parallel convolutions + samplers for the Mobile-VTON denoiser.

Two mechanisms:
  1. kernel>1, stride==1 convs (conv_in, conv_out, all 3x3 depthwise dw_conv):
     HALO. Exchange (k-1)//2 border rows with the neighbor, then run the conv with
     its H-padding removed. Reproduces the single-Pi receptive field exactly. 1x1
     convs need no halo (row-independent) and are left untouched.
  2. Downsample2D / Upsample2D (stride-2 conv / interpolate + conv): GATHER-FULL.
     There are only 4 of them; the stride-2 seam + nearest-upsample boundary math
     is fiddly, so gather the band to the full map, run the ORIGINAL sampler, and
     slice this rank's band at the NEW resolution. Correct by construction; the
     redundant per-rank sampler compute is cheap (samplers are a small fraction).

    shard_denoiser_spatial_conv_(model, group)  # returns (n_halo, n_sampler)
"""
import torch
import torch.nn.functional as F
import torch.distributed as dist

from parallel.sp_common import exchange_halo, gather_rows, scatter_rows


def _rw(group):
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(group), dist.get_world_size(group)
    return 0, 1


def make_sp_halo_conv_forward(conv, group=None):
    """Halo forward for a stride-1, kernel>1 Conv2d (any groups: depthwise or not)."""
    padH, padW = conv.padding if isinstance(conv.padding, tuple) else (conv.padding, conv.padding)

    def forward(x):
        x_ext = exchange_halo(x, padH, group=group)  # +padH real/zero rows each end
        return F.conv2d(x_ext, conv.weight, conv.bias, stride=conv.stride,
                        padding=(0, padW), dilation=conv.dilation, groups=conv.groups)
    return forward


def make_sp_sampler_forward(module, group=None):
    """Gather-full forward for Downsample2D / Upsample2D. Runs the module's ORIGINAL
    (class) forward on the full map, then re-slices this rank's band at the output
    resolution. Inner convs of the module must NOT be halo-patched."""
    orig_forward = type(module).forward  # unbound class forward

    def forward(x, *args, **kwargs):
        rank, world = _rw(group)
        full = gather_rows(x, rank, world, group=group)
        out_full = orig_forward(module, full, *args, **kwargs)
        return scatter_rows(out_full, rank, world, group=group)
    return forward


def _sampler_modules(model):
    return [m for m in model.modules()
            if type(m).__name__ in ("Downsample2D", "Upsample2D")]


def shard_denoiser_spatial_conv_(model, group=None):
    """Patch convs (halo) + samplers (gather-full) for H-band spatial parallelism.

    Returns (n_halo, n_sampler). Convs living INSIDE a sampler are skipped (the
    sampler handles them via gather-full)."""
    samplers = _sampler_modules(model)
    # collect every conv that belongs to a sampler, to exclude from halo patching
    inside_sampler = set()
    for s in samplers:
        for sub in s.modules():
            if isinstance(sub, torch.nn.Conv2d):
                inside_sampler.add(id(sub))

    n_halo = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d) and id(m) not in inside_sampler:
            k = m.kernel_size
            s = m.stride
            if k[0] > 1 and s[0] == 1 and s[1] == 1:  # kernel>1, stride 1 -> halo
                m.forward = make_sp_halo_conv_forward(m, group=group)
                n_halo += 1
            # 1x1 (or any stride-1 1x1) convs are row-independent -> leave as is.
            # stride>1 convs only exist inside samplers (excluded above).

    for s in samplers:
        s.forward = make_sp_sampler_forward(s, group=group)

    return n_halo, len(samplers)
