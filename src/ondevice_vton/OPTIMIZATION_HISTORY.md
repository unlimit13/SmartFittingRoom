# Mobile-VTON On-Device 최적화 히스토리

라즈베리파이 5(8GB, Cortex-A76 CPU)에서 Mobile-VTON 추론을 돌리기 위해 적용한
**OOM 방지 + 속도 개선** 패치들의 전체 기록. baseline부터 4-Pi 병렬화까지 step별 성과를 정리.

> 측정 환경: Raspberry Pi 5 (8GB LPDDR, Cortex-A76 @2.4GHz), torch 2.12.1+cpu (oneDNN/ACL),
> 입력 1024×768, fp32. 병렬 구간은 `192.168.100.0/24` eth0 + GbE 스위치.
> 상세 resume 트래커는 `parallel/PROGRESS.md`.

---

## 0. Baseline (시작점)

`inference_lowmem.py` (= `run_lowmem.sh`), **bf16 + tiled VAE**, 1024×768.
- 끝까지 돌긴 하지만 **bf16에서 ~34분/스텝**, 8스텝 풀런 **약 5시간 11분**.
- 문제: ① 8GB RAM에 안 들어가면 OOM, ② 너무 느림.

이 두 축(메모리 / 속도)을 따로 공략했다.

---

## 1. OOM 방지 — "8GB에 욱여넣기"

| 패치 | 내용 | 효과 |
|---|---|---|
| **Tiled VAE decode** | VAE 디코딩을 20개 독립 타일로 쪼개 feather-blend | 디코더 피크 메모리 대폭 ↓ (VAE가 최대 메모리 소비처) |
| **단계별 모듈 해제** | 파이프라인 단계마다 안 쓰는 모듈을 메모리에서 내림(sequential offload) | 동시 상주 메모리 ↓ |
| **Garment denoiser 캐싱** | garment 브랜치는 step마다 안 변함 → 1회 계산 후 캐시 | 매 step garment 재계산 제거 (속도 + 메모리) |
| **RGBA→RGB 변환** | dataset `__getitem__`에서 cloth/person `.convert("RGB")` | person2.png 같은 RGBA 입력 크래시 방지 |
| **(spatial 전용) garment denoiser channel-TP 샤딩** | spatial은 메인 가중치를 full로 들어 RAM 피크가 높음 → garment를 channel-TP로 쪼개 ~0.75GB 회수 | rank0 모델로드 피크 OOM 해결 |
| **MALLOC_ARENA_MAX=2** | glibc malloc arena 수 제한 | 단편화로 인한 RSS 부풀음 억제 |
| **SP_BAND_FRAC0 (불균등 밴드)** | rank0가 에이전트 오버헤드로 free RAM이 적음 → rank0 밴드를 작게 | rank0 피크 ↓ |

> ⚠️ spatial 병렬화는 메인 디노이저 가중치를 **full로 유지**(channel-TP는 절반으로 쪼갬)해서
> 단일 Pi보다 RAM 피크가 높다. 풀해상도 단일-Pi는 이제 OOM나고 **2-Pi 이상 split만 들어간다.**

---

## 2. 속도 — 단일 Pi 레버

### ✅ 정밀도: bf16 → fp32 (압도적 1순위, ~11x)

가장 큰 단일 개선. Cortex-A76엔 **bf16 ISA가 없어** bf16 matmul이 oneDNN의 bf16→fp32 에뮬레이션을 탄다.

마이크로벤치(4스레드, Linear 4096×1280): **fp32 80 GFLOP/s vs bf16 3.4 vs fp16 0.4**.

| 정밀도 | per-step | 8-step 풀런 |
|---|---|---|
| bf16 (baseline) | ~34분 | ~5시간 11분 |
| **fp32 (`--mixed_precision no`)** | **~3분** | **~30분** |

→ **약 11배**. 품질 동일, OOM도 안 남(lowmem 기법 덕). `run_single_lowmem.py`를 fp32로 변경 완료.

### ❌ 시도했지만 실패한 레버 (전부 메모리 대역폭 바운드)

denoiser step은 **LPDDR 대역폭 바운드**(depthwise conv + GroupNorm)라, "산술을 싸게" 만드는 시도는 모두 무력했다.

| 레버 | 결과 | 이유 |
|---|---|---|
| fp16 | **200배 느림** | 이 CPU에 최적 커널 없음 |
| INT8 동적+정적 양자화 (qnnpack) | **0.68~1.13x** (일부 더 느림) | fp32 baseline(oneDNN/ACL)이 이미 최적화됨; A76 dotprod로도 못 이김. 진짜 INT8은 ONNX/TFLite/ggml export 필요 |
| channels-last (NHWC) | **~1.04x** | conv는 빠르나(2-5x) 전체의 일부 + GroupNorm NHWC 페널티 + 포맷 thrashing. 무손실·무료지만 미미 |
| torch.compile | **~0 추가** | NHWC 대비 이득 없고 컴파일에 수분 소요 |
| 오버클럭 arm_freq=2800 | **~1.01x** | arm_freq는 LPDDR 대역폭을 안 올림. (실리콘 로터리: RPi1은 `over_voltage_delta=50000` 필요) |

**교훈: in-stack 소프트웨어 레버(INT8/compile/NHWC/OC)는 여기서 다 약하다.** 큰 이득은
"step을 덜 돌기(distillation/solver)" 또는 "병렬화로 쪼개기"에서만 나온다.

---

## 3. 멀티-Pi 병렬화 — latency 절감

목표: 단일 이미지 추론 **지연시간**을 디노이저를 여러 Pi로 쪼개 줄인다.
벤치 기준 = **순수 메인 디노이저 forward, steady cached step** (TP_TIME 계측).

### 3-A. Channel-TP (Megatron 스타일) — 채널/헤드/FFN 내부차원 분할

| Stage | 추가 내용 | per-step | vs 단일 |
|---|---|---|---|
| 단일 Pi (기준) | — | 117.07s | 1.00x |
| Stage 1 | attention TP (head 분할 q/k/v + row-parallel to_out + 1 all_reduce; MQA 인지) | 101.57s | **1.15x** |
| Stage 1b | + channel-parallel resnet conv (중간 채널 분할, GroupNorm 그룹 단위, row-parallel pw2) | 94.51s | **1.24x** |
| Stage 2 | + Megatron-MLP FFN (GEGLU.proj interleaved column-parallel + row-parallel out) | 86.20s | **1.36x** |
| (이론 천장) | N=∞ | 55.4s | 2.11x |

- 전 stage **등가성 PASS** (TP 이미지 == 단일, max\|diff\|=3/255, 99.98% within 1 LSB).
- **핵심 교훈: params ≠ compute.** FFN은 파라미터 48%지만 시간은 ~14%(큰 가중치·평범한 matmul);
  attention은 파라미터 25%지만 시간 ~26%(O(S²) SDPA).
- **Channel-TP는 여기서 TAPPED OUT**: step의 ~47%(depthwise conv + GroupNorm + up/down sampler)는
  matmul이 아니라 채널 분할 불가 → Amdahl 바닥. 4-Pi라도 ~1.5-1.65x가 한계.

### 3-B. Spatial (H-band) 병렬화 — 이미지 공간(H) 분할 ⭐

channel-TP가 못 쪼개는 **mem-bound 47%를 공간으로 쪼개는** 독립 스킴 (`parallel/sp_*.py`, `--spatial`).
- conv(kernel>1): **halo** 교환 / Downsample·Upsample: **gather-full**
- GroupNorm: 공간 축이 쪼개지므로 per-group sum/sqsum/count **all_reduce**
- self-attn: 전역이라 person **K/V를 all_gather**해 full 구성 / cross-attn: query-parallel

| 구성 | per-step | vs 단일 | vs channel-TP |
|---|---|---|---|
| 2-Pi spatial | **67.9s** | **1.72x** | 1.26x 빠름 (85.4s 대비) |

- 등가성 PASS (max\|diff\|=4/255, 99.99% within 1 LSB).
- channel-TP + 오버클럭이 못 건드린 **mem-bound depthwise/GroupNorm/sampler를 실제로 분할**해서 더 빠름.

### 3-C. VAE decode spatial split

TiledVaeDecoder의 20개 독립 타일을 랭크에 round-robin 분배 + 2 all_reduce. halo/norm/attn 없는
embarrassingly parallel → 디코드에서 ~Nx. (`parallel/VAE_SPATIAL_SPLIT.md`)

### 3-D. 4-Pi 확장 ⭐ (2026-06-29, 첫 실제 실행)

코드는 world-agnostic(world=4 수학 검증 14/14). 첫 실제 4-Pi 실행 (person0+cloth3, 6-step):

| 구성 | denoiser per-step | vs 단일 | vs 2-Pi spatial |
|---|---|---|---|
| **4-Pi spatial** | **34.53s** | **3.39x** | **1.97x** |

- step별 편차 거의 없음(33.8~35.7s). **OOM 없음. 스위치 comm 페널티 없음** (PROGRESS의 마지막 미해결 질문 해소).
- 6-step 전체: 총 추론 5분53초 (메인 denoise 3분44초 @ ~35.5s/step incl garment+CFG; VAE decode ~2분),
  모델로드 ~58s 포함 wall 6분51초.
- 셋업: checkpoint 3.5G·single_data를 **WAN 없이 LAN rsync로 .3/.4에 직송**.

---

## 4. 전체 성과 한눈에 (순수 디노이저 per-step, 단일 Pi 기준 배속)

```
정밀도 baseline (풀-스텝 기준, 별도 지표): bf16 ~34min → fp32 ~3min/step  ≈ 11x
─────────────────────────────────────────────────────────────────────
디노이저 forward per-step (steady, fp32):
  단일 Pi                         117.07s   1.00x  ████████████████████ 
  channel-TP  attention           101.57s   1.15x  █████████████████▍
  channel-TP  +conv                94.51s   1.24x  ████████████████▏
  channel-TP  +FFN                 86.20s   1.36x  ██████████████▋
  spatial 2-Pi                     67.90s   1.72x  ███████████▋
  spatial 4-Pi                     34.53s   3.39x  █████▉
```

**누적: bf16 baseline → fp32(11x) → 4-Pi spatial(추가 3.4x).**

---

## 5. 남은 레버 (미적용)

| 레버 | 학습 | 품질 | 4-Pi 현재 기대 | 비고 |
|---|---|---|---|---|
| **CFG-skip / guidance interval** | 불필요 | 약간 손실(검증필요) | **높음 ~15-25%** | 매 step cond+uncond 2회 → 일부 step uncond 생략 |
| **step 수 ↓ (4~5) / 스케줄러 튜닝** | 불필요 | trade-off(검증필요) | **높음(선형)** | FlowMatchEuler + scheduler_shift 스윕 |
| K/V all_gather ↔ compute overlap | 불필요 | **무손실** | 낮음 (comm≈0) | 스케일아웃 보험 |
| spatial × channel 하이브리드 (2×2) | 불필요 | **무손실** | 낮음~불확실 | 8·16 Pi에서 진가 |
| LCM/Turbo **distillation** | **필요** | — | (가장 큼, 선형) | 제외 대상 |
| (폐기) INT8/compile/NHWC/오버클럭 | — | — | ~0 | 대역폭 바운드 |

---

*생성: 2026-06-29 · 출처: `parallel/PROGRESS.md`, 측정 로그, 메모리 노트.*
