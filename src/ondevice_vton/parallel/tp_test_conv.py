"""Verify channel-parallel TP == full on every real ResnetBlock2D.

Loads the real denoiser on 2 localhost gloo ranks, and for each ResnetBlock2D
checks shard_resnet_ (channel-parallel pw1/pw2 + aligned norm2 + depthwise slice
+ all_reduce) reproduces the full block output, using the real trained weights.

    .venv/bin/python parallel/tp_test_conv.py
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
from parallel.tp_conv import (  # noqa: E402
    iter_resnet_blocks, tp_resnet_supported, shard_resnet_,
)

CKPT = os.path.join(ROOT, "checkpoint")


def worker(rank, world):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29583")
    dist.init_process_group("gloo", rank=rank, world_size=world)

    model = Unet_Tryon.from_pretrained(
        CKPT, subfolder="denoiser", torch_dtype=torch.float32).eval()
    blocks = list(iter_resnet_blocks(model))
    if rank == 0:
        sup = sum(tp_resnet_supported(b, world) for _, b in blocks)
        print(f"[inventory] {len(blocks)} ResnetBlock2D, TP-able={sup}")

    npass = nfail = nskip = 0
    maxdiff = 0.0
    for i, (name, block) in enumerate(blocks):
        if not tp_resnet_supported(block, world):
            nskip += 1
            continue
        cin = block.conv1.dw_conv.in_channels
        temb_c = block.time_emb_proj.in_features
        torch.manual_seed(2000 + i)
        x = torch.randn(1, cin, 8, 8)
        temb = torch.randn(1, temb_c)
        with torch.no_grad():
            ref = block(x, temb)
            tp = deepcopy(block)
            shard_resnet_(tp, rank, world)
            out = tp(x, temb)
        if rank == 0:
            d = (out - ref).abs().max().item()
            maxdiff = max(maxdiff, d)
            if torch.allclose(out, ref, atol=1e-3):
                npass += 1
            else:
                nfail += 1
                print(f"  FAIL {name} max|diff|={d:.2e}")

    if rank == 0:
        print(f"[result] resnet TP==full: pass={npass} fail={nfail} skip={nskip} "
              f"| max|diff|={maxdiff:.2e}")
        print("ALL PASS" if nfail == 0 else "SOME FAILED")
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(2,), nprocs=2)
