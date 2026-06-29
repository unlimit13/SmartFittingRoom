"""Verify Megatron-MLP TP == full on every real FeedForward (GEGLU) module.

    .venv/bin/python parallel/tp_test_ffn.py
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
from parallel.tp_ffn import (  # noqa: E402
    iter_ffn_modules, tp_ffn_supported, shard_ffn_,
)

CKPT = os.path.join(ROOT, "checkpoint")


def worker(rank, world):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29585")
    dist.init_process_group("gloo", rank=rank, world_size=world)

    model = Unet_Tryon.from_pretrained(
        CKPT, subfolder="denoiser", torch_dtype=torch.float32).eval()
    ffns = list(iter_ffn_modules(model))
    if rank == 0:
        sup = sum(tp_ffn_supported(f, world) for _, f in ffns)
        print(f"[inventory] {len(ffns)} FeedForward, TP-able={sup}")

    npass = nfail = nskip = 0
    maxdiff = 0.0
    for i, (name, ff) in enumerate(ffns):
        if not tp_ffn_supported(ff, world):
            nskip += 1
            continue
        dim = ff.net[0].proj.in_features
        torch.manual_seed(3000 + i)
        x = torch.randn(1, 16, dim)
        with torch.no_grad():
            ref = ff(x)
            tp = deepcopy(ff)
            shard_ffn_(tp, rank, world)
            out = tp(x)
        if rank == 0:
            d = (out - ref).abs().max().item()
            maxdiff = max(maxdiff, d)
            if torch.allclose(out, ref, atol=1e-3):
                npass += 1
            else:
                nfail += 1
                print(f"  FAIL {name} max|diff|={d:.2e}")

    if rank == 0:
        print(f"[result] ffn TP==full: pass={npass} fail={nfail} skip={nskip} "
              f"| max|diff|={maxdiff:.2e}")
        print("ALL PASS" if nfail == 0 else "SOME FAILED")
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(2,), nprocs=2)
