"""Verify spatial (H-band) GroupNorm + halo conv + samplers == full, on real
denoiser modules, across 2 localhost gloo ranks.

    .venv/bin/python parallel/sp_test_conv.py
"""
import os
import sys
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from Mobile_VTON.models.unets.unet_2d_condition_tryon import (  # noqa: E402
    UNet2DConditionModel as Unet_Tryon,
)
from parallel.sp_common import scatter_rows  # noqa: E402
from parallel.sp_groupnorm import shard_denoiser_groupnorm_  # noqa: E402
from parallel.sp_conv import shard_denoiser_spatial_conv_  # noqa: E402
from parallel.tp_conv import iter_resnet_blocks  # noqa: E402

CKPT = os.path.join(ROOT, "checkpoint")


def patch_spatial_(mod, group=None):
    shard_denoiser_groupnorm_(mod, group=group)
    return shard_denoiser_spatial_conv_(mod, group=group)


def check(name, ref_full, out_band, rank, world, atol, log):
    ref_band = scatter_rows(ref_full, rank, world)
    if rank == 0:
        d = (out_band - ref_band).abs().max().item()
        ok = torch.allclose(out_band, ref_band, atol=atol)
        log.append((name, ok, d))
        if not ok:
            print(f"  FAIL {name} max|diff|={d:.2e}")


def worker(rank, world):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29587")
    dist.init_process_group("gloo", rank=rank, world_size=world)
    torch.set_grad_enabled(False)

    model = Unet_Tryon.from_pretrained(
        CKPT, subfolder="denoiser", torch_dtype=torch.float32).eval()
    log = []

    # 1) GroupNorm in isolation (a few real ones)
    gns = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.GroupNorm)]
    for i, (name, gn) in enumerate(gns[:4]):
        C = gn.num_channels
        torch.manual_seed(100 + i)
        x = torch.randn(2, C, 8, 6)
        ref = gn(x)
        spgn = deepcopy(gn)
        shard_denoiser_groupnorm_(torch.nn.ModuleList([spgn]))
        out = spgn(scatter_rows(x, rank, world))
        check(f"groupnorm/{name}", ref, out, rank, world, 1e-4, log)

    # 2) ResnetBlock2D (GN + dw halo conv + 1x1 local)
    blocks = list(iter_resnet_blocks(model))
    for i, (name, block) in enumerate(blocks[:6]):
        cin = block.conv1.dw_conv.in_channels
        temb_c = block.time_emb_proj.in_features
        torch.manual_seed(300 + i)
        x = torch.randn(2, cin, 8, 6)
        temb = torch.randn(2, temb_c)
        ref = block(x, temb)
        sp = deepcopy(block)
        patch_spatial_(sp)
        out = sp(scatter_rows(x, rank, world), temb)
        check(f"resnet/{name}", ref, out, rank, world, 1e-3, log)

    # 3) Samplers (Downsample2D / Upsample2D) gather-full
    samps = [(n, m) for n, m in model.named_modules()
             if type(m).__name__ in ("Downsample2D", "Upsample2D")]
    for i, (name, s) in enumerate(samps):
        C = s.channels
        torch.manual_seed(500 + i)
        x = torch.randn(2, C, 8, 6)
        ref = s(x)
        sp = deepcopy(s)
        shard_denoiser_spatial_conv_(torch.nn.ModuleList([sp]))
        out = sp(scatter_rows(x, rank, world))
        check(f"sampler/{name}[{type(s).__name__}]", ref, out, rank, world, 1e-4, log)

    if rank == 0:
        npass = sum(1 for _, ok, _ in log if ok)
        nfail = sum(1 for _, ok, _ in log if not ok)
        mx = max((d for _, _, d in log), default=0.0)
        print(f"[result] spatial conv/gn/sampler == full: pass={npass} fail={nfail} "
              f"| max|diff|={mx:.2e}")
        print("ALL PASS" if nfail == 0 else "SOME FAILED")
    dist.destroy_process_group()


if __name__ == "__main__":
    world = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    print(f"[sp_test_conv] world={world}")
    mp.spawn(worker, args=(world,), nprocs=world)
