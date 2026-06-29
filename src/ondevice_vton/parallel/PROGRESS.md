# Multi-Pi Parallelization — Progress / Resume

**Goal:** cut single-image inference LATENCY via 2-way tensor parallelism (TP) of
the main denoiser, over the direct wired GbE link between two Pi 5s.

## ▶ RESUME HERE (updated 2026-06-26)
STATE: all code COMMITTED + PUSHED to origin/master @ **994d193** (was previously
all uncommitted). Cluster standard is now **4 Pis** (.1 main/rank0 .. .4); see
`parallel/PI_SETUP.md` + `parallel/SPATIAL_4PI.md` + memory `multi-pi-cluster`.

DONE + VERIFIED on 2 Pis:
- SPATIAL (H-band) parallelism (parallel/sp_*.py, `--spatial`): equivalent to
  channel-TP/single (max|diff|=4) and **67.9s/step = 1.26x faster than channel-TP
  (85.4s)** -- splits the mem-bound depthwise/GroupNorm/sampler ~47%. garment
  denoiser auto channel-TP'd to fit RAM. world-agnostic; world=4 math-verified on
  localhost (sp_test_conv.py 4 = 14/14).
- VAE decode spatial split (tile round-robin + 2 all_reduce): equivalent, ~Nx.

RUN (4-Pi default): `bash parallel/run_sp_multi.sh 6 output/spatial`
(2-Pi: pass "192.168.100.2" as the 4th arg, or use run_sp_pair.sh).

NEXT (resume here):
1. Bring up the cluster: .3 needs passwordless SSH; .4 is down (IP/power/cable);
   provision .3/.4 per PI_SETUP.md (clone+setup_env+download_ckpt). .2 just `git pull`.
2. FIRST REAL 4-Pi RUN is unproven -> smoke-test 1 step (correctness + does it FIT
   in RAM: garment channel-TP only partially shards at world=4 since 14 heads % 4),
   then 6 steps. Watch switch-latency on the many small collectives.
3. Bigger latency levers: step-count distillation (linear, biggest, needs training);
   overlap self-attn person-K/V all_gather with compute; 4-Pi spatial x channel hybrid.
Old channel-TP resume notes below.

## ▶ RESUME (channel-TP, historical)
Stages 1 + 1b + 2 (attention + channel-parallel conv + Megatron-MLP FFN) DONE +
VERIFIED on 2 Pis: equivalence PASS, 1.36x/step. Channel-TP is now ~TAPPED OUT:
the remaining ~47% of the step (depthwise convs + GroupNorm + up/down samplers) is
NON-matmul work that channel-TP can't split, and it's the Amdahl floor (max ~2.1x
even at N=inf; measured 4-Pi would be ~1.5-1.65x). To break it you must SPATIAL-
parallelize (splits depthwise/norm/sampler by space, but needs attention all-gather
+ halo + GroupNorm stat sync). That's the only lever left for big gains.
NOTE: comm is ~0 on the 2-Pi DIRECT cable; 4-Pi needs a switch -> comm becomes real.
Bring up the direct link first (eth0 has NO ip on resume — set static IPs).

Bring up the link (each Pi, needs sudo; not persistent across reboot):
    rank0 (this Pi):  sudo ip addr add 192.168.100.1/24 dev eth0
    rank1 (peer):     sudo ip addr add 192.168.100.2/24 dev eth0
    verify:           ping -c1 192.168.100.2   (and ssh in)

Run TP (same cmd both Pis, only --rank differs; launch rank1 on peer first or
together — gloo init_process_group blocks until both join):
    # rank 0 — THIS Pi (also the writer; saves output)
    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 GLOO_SOCKET_IFNAME=eth0 \
      .venv/bin/python inference_lowmem.py --tp --rank 0 --world_size 2 \
      --data_dir single_data --output_dir output/tp --order unpaired \
      --test_batch_size 1 --num_inference_steps 1 --guidance_scale 2.5 \
      --mixed_precision no --checkpoint_path checkpoint
    # rank 1 — PEER: identical but --rank 1
Equivalence: also run WITHOUT --tp to output/single, then compare the two PNGs
(atol on pixels). Latency: bump --num_inference_steps and read per-step time.

## Setup facts (don't re-derive)
- rank 0 = this Pi, eth0 `192.168.100.1`; rank 1 = peer, eth0 `192.168.100.2`
- gloo pinned to eth0 via `GLOO_SOCKET_IFNAME=eth0`
- Link measured: 117 MB/s raw TCP, gloo all_gather 138 MB/s, RTT ~283us
- Stage 0 verdict: **~1.83x feasible, PASS**
- Both Pis: same venv (torch 2.12.1+cpu); checkpoints present
- Single-Pi baseline (fp32 + garment caching): ~119 s/step, 8 steps

## TP scheme (Megatron-style, per main-denoiser attention)
- column-parallel `to_q/to_k/to_v` (each rank keeps its head slice; no comm)
- row-parallel `to_out[0]` (partial output per rank) -> **one all_reduce** sums them
- bias kept on rank 0 only (so all_reduce counts it once)
- per attention: exactly 1 all_reduce of [B, S, out_dim]

## Results (2-Pi, fp32, person0+cloth1, 1024x768)
- Equivalence PASS at every stage: 2-Pi TP image == single-Pi, max|diff|=3/255,
  99.98% within +/-1 LSB. Pure float-reordering noise. compare_outputs.py.
- Latency (pure main-denoiser forward, steady cached step; TP_TIME instrumentation):
    single-Pi                         117.07 s/step   1.00x
    Stage 1 (attention TP)            101.57 s/step   1.15x
    Stage 1+1b (+channel-parallel conv) 94.51 s/step  1.24x
    Stage 1+1b+2 (+Megatron-MLP FFN)  86.20 s/step    1.36x   <-- current
    N=inf ceiling (channel-TP)        55.4 s/step     2.11x
- KEY: params != compute. FFN is 48% of PARAMS but only ~14% of the STEP (big
  weights, plain matmul); attention is 25% params but ~26% time (O(S^2) SDPA).
  Measured compute split: attention ~31s, pointwise-conv ~14s, FFN ~16.6s,
  REPLICATED rest ~55s (depthwise convs + GroupNorm + up/down samplers). Model
  T(N)=55.4 + 61.6/N (comm~0 on direct cable) fits: T(2)=86.2 exactly.
- 4-Pi estimate (channel-TP, corrected): ~1.5-1.65x (was wrongly 2.1-2.4x before
  measuring FFN). The ~47% replicated non-matmul floor caps it; switch comm hurts.

## Non-TP latency levers explored (2026-06-26)
- INT8 quant (PyTorch dynamic+static, qnnpack): 0.68-1.13x => DEAD END. fp32 base
  (oneDNN/ACL) already optimized; qnnpack int8 doesn't beat it despite A76 dotprod.
  `parallel/int8_microbench.py`.
- channels-last (NHWC): conv micro-bench 2-5x BUT full denoiser only ~1.04x steady
  (conv is a fraction; GroupNorm NHWC penalty + format thrashing). Free + lossless
  (max|diff|=4), stacks w/ TP. `inference_lowmem.py --channels_last`. torch.compile
  adds ~0 over NHWC -> skipped.
- Overclock: default 2.4GHz/ondemand; arm_freq=2800 (+reboot, cooling) ~= 1.15x.
  Stacked best estimate: TP 1.36x x channels-last ~1.41x x overclock ~1.62x.

## TODO (delete each as done)
- [ ] (biggest, needs training) Step-count reduction: LCM/Turbo distillation
      (latency is LINEAR in steps).
- [ ] (big-gain, hard) Spatial parallelism to split the replicated depthwise/norm/
      sampler ~47%: halo for conv, all-gather K/V for attention, GroupNorm stat sync.
- [ ] (combo) Make --channels_last + --tp co-exist (re-apply NHWC after sharding,
      since slicing weights drops the memory format).
- [ ] (4-Pi only) fuse/overlap the 112 all_reduces once a switch makes comm matter
- [ ] Stage 2: compute/comm overlap, fuse small all_reduces
- [ ] (later) rope path in TPAttnProcessor (not needed yet; GQA/MQA already done)
- [ ] (later) 4-Pi: 2-way TP x 2-way data-parallel via router

## Done (terse log)
- VAE decode SPATIAL split DONE + VERIFIED (2026-06-26): the TiledVaeDecoder already
  decodes 20 independent feather-blended tiles -> distribute tiles round-robin across
  ranks + 2 all_reduces (out, acc). Embarrassingly parallel, no halo/norm/attn. Wired
  in inference_lowmem.py (sp_rank/sp_world, gated by --spatial). 2-Pi: equivalent
  (max|diff|=4) and Inferring total 267.8s->251.0s (VAE ~halved, ~17s saved; ~Nx in
  general). Doc: parallel/VAE_SPATIAL_SPLIT.md.
- 4-Pi readiness DONE (2026-06-26): core sp_*.py is world-agnostic, math-verified at
  world=4 on localhost (sp_test_conv.py 4 -> 14/14, max|diff| 1.9e-6). New launcher
  parallel/run_sp_multi.sh takes a peer-IP list (world=1+#peers); the old 2-Pi path
  re-verified through it (equivalent). Needs a SWITCH for 3+ Pis (2-Pi cable is p2p)
  + 192.168.100.(rank+1) static IPs + per-peer clone. Doc: parallel/SPATIAL_4PI.md.
  NOT yet run on real 4 Pis (only 2 available); switch comm latency on the many small
  collectives is the open question at N=4.
- SPATIAL (H-band) parallelism DONE + VERIFIED (2026-06-26): independent scheme,
  new parallel/sp_*.py, channel-TP files untouched, `--spatial` flag. Splits the
  feature map along H (weights stay full). sp_common (band scatter/gather/halo),
  sp_groupnorm (per-group sum/sqsum/count all_reduce -- channel-TP local-group
  trick can't apply when the spatial reduction axis is split), sp_conv (kernel>1
  stride1 convs HALO; Downsample2D/Upsample2D GATHER-FULL), sp_attention (self:
  drop garment queries + gather person K/V to full; cross: query-parallel; query
  RoPE uses band offset, self key uses original prefix/suffix incl garment).
  Unit test sp_test_conv.py 14/14 PASS. 2-Pi 1-step AND 6-step equivalence vs
  channel-TP (==single): max|diff|=4/255, 99.99% within 1 LSB. **Latency: ~67.9
  s/step (69.5,68.0,68.3,67.5,67.0,67.1) vs channel-TP 85.4 = 1.26x FASTER** --
  it splits the mem-bound depthwise/GroupNorm/sampler ~47% that channel-TP and
  overclock could not touch. KEY MEMORY FIX: spatial keeps the MAIN denoiser
  weights full -> higher peak than channel-TP -> rank0 (this Pi, ~1.3GB agent
  overhead -> only ~6.5GB free) OOMs at the model-load/embedding peak. Fixed by
  CHANNEL-TP sharding the GARMENT denoiser inside the --spatial branch (runs once,
  all_reduced output is full/replicated = what main self-attn needs) -> ~0.75GB
  off the peak. Also MALLOC_ARENA_MAX=2 + optional uneven band (SP_BAND_FRAC0).
  Harness parallel/run_sp_pair.sh. Bug log: (a) original self-attn RoPEs garment
  tokens too; (b) all_gather_object(pickle) churned RAM -> int all_gather; (c) the
  single-Pi baseline itself now OOMs at full res (only the 2-Pi splits fit).
- Overclock arm_freq=2800 DONE + VERIFIED (2026-06-26): BOTH Pis stable at 2800
  under load, throttled=0x0. SILICON LOTTERY: rank0 (RPi0) boots 2800 at default
  voltage; rank1 (RPi1) FAILED to boot 2800 at default V -> fixed with
  `over_voltage_delta=50000` (+0.05V; Pi5 voltage knob is over_voltage_delta in uV,
  NOT the old over_voltage). RPi1 then: load freq 2800MHz, throttled 0x0, 50.5C.
  Config lines live in each Pi's /boot/firmware/config.txt under [all]. Pi5 does
  NOT auto-revert a bad-OC config.txt -> remote OC reboot can brick SSH (edit SD
  offline to recover).
  **MEASURED (2026-06-26, NEGATIVE): overclock gives almost NOTHING on the real
  denoiser.** 2-Pi channel-TP step: 86.2s/step @default -> ~85.4s/step @2800 =
  ~1.01x (NOT the estimated 1.15x). The denoiser step is MEMORY-BANDWIDTH-bound
  (depthwise conv + GroupNorm over 128x96 maps = the replicated ~47%), and arm_freq
  doesn't scale LPDDR bandwidth. Same root cause as INT8/compile/channels-last all
  being weak. => overclock is not a real latency lever; big wins must come from step
  reduction (distillation) or SPATIAL parallelism (actually splits the mem-bound 47%).
- 2-Pi TP run on cloth3+person2 (2026-06-26): 6 steps, ~85.4s/step denoiser,
  denoise 8m55s + tiled VAE ~2m40s = 11m35s total infer. Output good (RGBA person2
  .png handled via .convert("RGB") added to inference_lowmem.py dataset __getitem__).
- Stage 0 complete: link + gloo benched, PASS (1.83x).
  scripts: `parallel/link_bench.py`, `parallel/tp_stage0_commbench.py`
- Env reproducible: `setup_env.sh` + `requirements.txt` (peer cloned + set up).
- Stage 1 MVP core: TP attention sharding (head-parallel q/k/v + row-parallel
  to_out + all_reduce) verified == full attention, diff ~1e-7 (self & cross).
  `parallel/tp_attention.py`, test `parallel/tp_test_attention.py`.
- Model is MQA (kv_heads=1, query heads 4/8/14, head_dim 64). TP updated to
  shard query heads + replicate/shard kv (shard_attention_, tp_supported,
  TPAttnProcessor now GQA/MQA-aware). Verified on REAL denoiser: all 60
  Attention modules TP == full, max|diff| 2.2e-5. test `parallel/tp_test_denoiser.py`.
- rope path added to TPAttnProcessor: the REAL denoiser feeds image_rotary_emb to
  attention (the per-module test missed it -> first 2-Pi run crashed). rope is
  per-head on head_dim, so TP on local heads is exact. Verified TP==full with rope
  on real self+cross attention: `parallel/tp_test_rope.py` (max|diff| ~4e-6).
- 2-Pi run harness: `parallel/run_tp_pair.sh` (rank1 on peer via ssh kept open
  locally + rank0 here; DON'T background on the remote -> ssh channel deadlocks).
  Peer repo: `/home/willtek/Mobile_VTON-ondevice-optmization` (NOT under bootcamp);
  needs tp_attention.py/tp_bootstrap.py + pipeline + single_data synced (scp).
  `parallel/compare_outputs.py` checks image equivalence.
- Stage 2 Megatron-MLP FFN: `parallel/tp_ffn.py`. Splits inner dim: GEGLU.proj
  column-parallel with INTERLEAVED slice (proj out is [hidden|gate], each rank
  takes the matching slice of BOTH), out Linear row-parallel + all_reduce hook.
  All 36 FFN TP==full: `parallel/tp_test_ffn.py` (max|diff| 1.5e-4). Wired into
  inference after conv sharding. 2-Pi: equivalence PASS, 86.20s/step = 1.36x.
- Stage 1b channel-parallel resnet conv: `parallel/tp_conv.py`. Splits the MIDDLE
  channels (col-parallel pw1 + time_emb_proj, GroupNorm resliced onto whole groups
  = local stats, depthwise dw2 sliced, row-parallel pw2 + all_reduce via forward
  hook). All 16 ResnetBlock2D TP-able (sep convs, default time-embed, 32 groups
  aligned). Verified TP==full on all 16: `parallel/tp_test_conv.py` (max|diff|
  2.4e-5). Wired into inference_lowmem.py after attention sharding.
- Distributed bootstrap landed: `parallel/tp_bootstrap.py` (init_tp/shutdown_tp,
  gloo on eth0 from CLI rank/master args) + `inference_lowmem.py` wired with
  `--tp/--rank/--world_size/--master_addr/--master_port/--tp_iface` (shards
  denoiser after eval, gates file writes to rank 0, tears PG down). Smoke-tested
  localhost 2-rank: `parallel/tp_test_bootstrap.py` (all_reduce PASS). Not yet
  run on real 2 Pis (link was down).
