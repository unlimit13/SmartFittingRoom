"""Verify TP attention sharding on the REAL denoiser, per Attention module.

Loads the actual denoiser (fp32) on 2 local gloo ranks, walks every Attention
module, and checks the head-parallel TP output == the full output for each, using
the real trained weights. Also prints an inventory (head counts, cross/self, any
modules unsupported by the MVP). Same machine -> correctness only, not speed.

    .venv/bin/python parallel/tp_test_denoiser.py
"""
import os
import sys
from collections import Counter
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
from parallel.tp_attention import (  # noqa: E402
    iter_attention_modules, tp_supported, shard_attention_, TPAttnProcessor,
)

CKPT = os.path.join(ROOT, "checkpoint")


def worker(rank, world):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29578")
    dist.init_process_group("gloo", rank=rank, world_size=world)

    if rank == 0:
        print("[load] denoiser (fp32) ...", flush=True)
    model = Unet_Tryon.from_pretrained(
        CKPT, subfolder="denoiser", torch_dtype=torch.float32).eval()
    mods = list(iter_attention_modules(model))

    if rank == 0:
        heads = Counter(a.heads for _, a in mods)
        cross = sum(1 for _, a in mods if a.is_cross_attention)
        skip = [n for n, a in mods if not tp_supported(a, world)]
        print(f"[inventory] {len(mods)} Attention modules "
              f"(cross={cross}, self={len(mods) - cross})")
        print(f"[inventory] heads histogram: {dict(heads)}")
        print(f"[inventory] unsupported for world={world}: {len(skip)} {skip[:5]}")

    npass = nfail = nskip = 0
    maxdiff = 0.0
    for i, (name, attn) in enumerate(mods):
        if not tp_supported(attn, world):
            nskip += 1
            continue
        torch.manual_seed(1000 + i)  # identical input on both ranks
        x = torch.randn(1, 32, attn.to_q.in_features)
        enc = (torch.randn(1, 32, attn.to_k.in_features)
               if attn.is_cross_attention else None)
        try:
            with torch.no_grad():
                ref = attn(x, encoder_hidden_states=enc)
                tp = deepcopy(attn)
                shard_attention_(tp, rank, world)
                tp.set_processor(TPAttnProcessor())
                out = tp(x, encoder_hidden_states=enc)
        except Exception as e:
            if rank == 0:
                print(f"  ERROR {name}: {type(e).__name__}: {e}")
            nfail += 1
            continue
        if rank == 0:
            d = (out - ref).abs().max().item()
            maxdiff = max(maxdiff, d)
            if torch.allclose(out, ref, atol=1e-3):
                npass += 1
            else:
                nfail += 1
                print(f"  FAIL {name} max|diff|={d:.2e}")

    if rank == 0:
        print(f"\n[result] pass={npass} fail={nfail} skip={nskip} "
              f"| max|diff|={maxdiff:.2e}")
        print("ALL PASS" if nfail == 0 else "SOME FAILED")
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(2,), nprocs=2)
