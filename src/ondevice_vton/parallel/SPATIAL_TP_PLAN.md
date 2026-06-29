# Spatial (Sequence) Parallelism — Design Plan

**Why:** channel-TP (attention + conv + FFN) is tapped out at **1.36x/step**, with a
hard Amdahl floor of ~2.11x at N=inf because ~47% of the step (depthwise convs +
GroupNorm + up/down samplers) is non-matmul work channel-TP cannot split. Spatial
parallelism splits the H×W feature map across ranks, so it splits the **entire**
step uniformly — including that 47% floor. On 2 Pis it is the only lever with a
ceiling well above 1.36x (ideal ~2x; realistic ~1.6–1.8x, comm-eroded).

**Relationship to channel-TP:** this is a DIFFERENT axis, not an add-on. On 2 Pis
you pick ONE axis per layer; mixing axes mid-network needs an all-to-all reshard
(expensive). Recommendation: build spatial-TP as a STANDALONE scheme replacing
channel-TP for the 2-Pi case. Reuse only the infra: `tp_bootstrap.py` (gloo init),
`run_tp_pair.sh`, `compare_outputs.py`. The matmul-sharding files
(`tp_attention/conv/ffn.py`) are NOT reused (different axis). For 4 Pis later:
2-way spatial × 2-way channel hybrid.

**Stacks with:** overclock (~1.15x) and step-count distillation (linear in steps) —
all orthogonal. Spatial does NOT stack with channel-TP on the same 2 Pis.

## Split scheme
- Split along **H** (rows): each rank owns a contiguous row band `[h0:h1]` of every
  feature map. Contiguous → halo is just the top/bottom border rows; ownership
  stays contiguous across resolution changes (up/down just scales h0/h1).
- rank0 = top band, rank1 = bottom band. Input scattered by rank0, output gathered
  to rank0 (gate writes to rank0, same as current bootstrap).
- W kept whole (split only one axis to keep halo/all-gather logic simple on 2 Pis).

## Per-op treatment (what changes vs single-Pi)
| op | spatial behavior | comm |
|----|------------------|------|
| pointwise conv 1×1 (`pw_conv`), shortcut 1×1, FFN, time_emb | row-independent | **none** |
| depthwise conv 3×3 (`dw_conv`), pad=1 | needs 1 neighbor row each side | **halo exchange** (1 row, [B,C,1,W]) |
| GroupNorm (`norm1`/`norm2`, eps=1e-6) | stats span full H | **all_reduce** per-group (sum, sumsq, count) |
| self-attention | queries local, needs all K/V | **all_gather K/V** (MQA → tiny: [B,S,64]) |
| cross-attention (garment) | keys = garment (replicated/cached) | **none** |
| rope (`image_rotary_emb`) | per-position table | slice table to local token range (no comm) |
| downsample (stride-2 conv, pad=1, +asymmetric `F.pad(0,1,0,1)`) | halo + boundary index math | **halo** |
| upsample (`F.interpolate` nearest ×2, then conv) | nearest is local; conv needs halo | **halo** (after upsample) |
| SiLU/dropout/add/scale | elementwise | **none** |

Per step the new collectives ≈ ~40 halos (1-row, latency-bound ~283us each) +
~34 GroupNorm stat all_reduces (tiny) + self-attention K/V all_gathers (MQA-small,
biggest single cost at top res S=12288: ~3MB → ~25ms). Rough comm budget < ~0.5s
on the direct cable vs ~27s of split compute → comm is small but NOT zero (many
latency-bound syncs at low-res stages erode the ideal 2x).

## Correctness-critical details (get these exactly right)
1. **GroupNorm stat sync** — the channel-TP "whole-group local stats" trick does
   NOT apply here. Each rank computes partial per-group sum/sumsq/count over its
   rows, all_reduce SUM, then mean/var = global. Apply affine locally. eps=1e-6.
   This is the #1 equivalence risk; unit-test a GroupNorm in isolation first.
2. **Depthwise halo** — for 3×3 pad=1, exchange exactly 1 border row with the
   neighbor before `dw_conv`; rank0 has no top neighbor (zero-pad), rank1 no bottom
   (zero-pad) — must reproduce the single-Pi zero padding at the OUTER edges only.
3. **Downsample stride-2** — `Downsample2D` uses pad=1 conv (and an asymmetric
   `F.pad(0,1,0,1)` only when padding==0; here padding=1 so that branch is off).
   Stride-2 over a row band: the output row a rank owns depends on input rows
   `2*r-1 .. 2*r+1`; need 1 halo row and care that the global stride grid matches
   the single-Pi grid (no double-count / off-by-one at the band seam). Keep H even
   per band where possible.
4. **Upsample** — `F.interpolate(scale_factor=2, nearest)` is purely local per row
   (row r -> rows 2r,2r+1), no halo; the FOLLOWING conv (if use_conv) needs the
   halo. Verify the global row indexing after ×2 still tiles correctly.
5. **rope table slicing** — real denoiser feeds `image_rotary_emb` keyed by absolute
   token position. Each rank applies rope to its LOCAL query rows using the matching
   slice of the table. Mirror the prefix/suffix logic already in `TPAttnProcessor`.
6. **Odd / tiny H at deep stages** — bottleneck H may get small (128→…→16→8). 8/2=4
   ok; watch for any odd H (pad to even or give the remainder to one rank, document
   the imbalance). Halo cost grows relative to tile at tiny H.
7. **Self-attention K/V all_gather** — each rank computes K/V for its local tokens,
   all_gather to full sequence, SDPA(local Q, full K/V) → output stays local rows.
   MQA so K/V is [B,S,head_dim]; cheap. Cross-attention skips this (garment K/V
   already full on both ranks).

## Staged build (each stage: implement → unit-test TP==full → keep)
- **S0 Design+microbench**: halo send/recv + GroupNorm-stat all_reduce + K/V
  all_gather micro-benchmarks on the real link; confirm comm budget < ~1s/step.
  Decide H-split helper API (own_rows(h0,h1), exchange_halo(x,k), gn_sync(...)).
- **S1 Spatial depthwise conv + halo**: wrap `SepConv2d.dw_conv` (and shortcut/pw
  pass-through). Unit-test one ResnetBlock2D: spatial output == full (atol ~1e-5).
- **S2 Spatial GroupNorm**: stat-all_reduce GroupNorm replacement. Unit-test in
  isolation AND inside a resnet block.
- **S3 Spatial attention**: self (all_gather K/V + rope slice) and cross (local).
  Unit-test on real self+cross attention modules == full.
- **S4 Samplers**: spatial down/up + their convs. Unit-test each == full.
- **S5 Full denoiser wiring**: scatter input rows on rank0, run spatial denoiser,
  gather output to rank0. End-to-end equivalence vs single-Pi (compare_outputs.py,
  expect pixel atol ~3/255 like channel-TP). Add `--spatial` flag to
  inference_lowmem.py paralleling the existing `--tp` path; reuse tp_bootstrap.
- **S6 Latency + optimize**: measure per-step; overlap halo exchange with compute,
  fuse GroupNorm stat reduces, batch halos where layers are adjacent. Compare to
  the channel-TP 86.2s/step baseline.

## Honest cost / payoff
- Effort ≈ the whole of stage 1+1b+2 combined (it's a from-scratch second
  parallelization). GroupNorm sync + downsample seam are the tricky bits.
- Payoff on 2-Pi: realistic ~1.6–1.8x (vs channel-TP 1.36x) — i.e. ~1.2–1.3x MORE.
- Compare against distillation (8→4 steps = free ~2x, stacks with everything and
  needs no parallel code). If latency is the only goal, distillation is the bigger,
  cheaper win; spatial-TP is the bigger PARALLELISM win and the basis for 4-Pi.
```
T_spatial(2) ≈ T_compute/2 + comm  ≈ 117/2 + ~0.5  ≈ 59s/step  (~2x ideal)
realistic with low-res sync overhead + load imbalance: ~65–73s/step (~1.6–1.8x)
```
