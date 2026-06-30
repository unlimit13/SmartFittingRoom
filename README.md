# LG Styler Smart Fitting Room

> **On-Device AI 기반 실시간 멀티모달 패션 추천 시스템**
> SW Bootcamp 13기 | Embedded Linux 기반 On-Device AI 프로젝트

---

## 프로젝트 개요

웹캠으로 착용 의상을 감지하고, **한국어 자연어 입력**과 결합하여 무신사 DB에서 어울리는 코디를 실시간으로 추천합니다.
추천 파이프라인(감지·임베딩·검색·리랭킹)의 모든 AI 추론은 클라우드 없이 **Raspberry Pi 5 위에서 완전 온디바이스**로 실행됩니다.
(가상 피팅만 백엔드를 선택: 기본은 fal-ai 클라우드 API, 선택적으로 RPi5 클러스터 온디바이스 추론.)

**"LG Styler 앞에 서면, 오늘의 코디를 추천해드립니다."**

---

## 주요 기능

- **실시간 의류 감지** — MediaPipe Pose로 사람을 감지하여 랜드마크 기반 바운딩 박스를 상의/하의/신발 영역으로 수직 분할
- **앵커 카테고리 선택** — 상의(tops) 또는 하의(bottoms) 중 하나를 앵커로 지정하여 CLIP 이미지 임베딩 기반 유사 검색 수행
- **혼합 점수 기반 코디 세트 추천** — 앵커 카테고리를 CLIP유사도+텍스트유사도+색상호환성 혼합 점수로 리랭킹해 상위 후보(`NUM_CANDIDATES=3`)를 뽑고, 각 앵커 상품의 스냅 코디(snap_id)가 있으면 그 코디로, 없으면 카테고리별 독립 검색으로 나머지 슬롯을 채워 **최대 3개 코디 세트**(상의+하의+신발) 추천
- **색상 팔레트 추출** — OpenCV K-means로 앵커 크롭 영역의 지배색 3개를 추출하여 UI에 표시
- **무신사 상품 QR코드** — 추천된 각 상품에서 바로 무신사 구매 페이지로 이동 가능
- **감지 스트림 분리** — `/detection_feed`로 MediaPipe Pose 감지 오버레이 MJPEG 스트림 별도 제공
- **포즈 기반 자동 트리거** — MediaPipe Pose(model_complexity=0)로 관절 추적, 화면 중앙 존에 3초 유지 시 추천 자동 발동, 홀드 진행바·잔여 시간 시각화
- **가상 피팅(Virtual Try-On)** — 추천 상의/하의를 착용자에게 순차 합성(tops → bottoms), SSE 스트리밍으로 단계별 결과 표시. 백엔드는 `VTON_BACKEND`로 선택: 기본=fal-ai/fashn/tryon v1.6 클라우드 API, `ondevice`=온디바이스 Mobile-VTON
- **분산 가상 피팅(spatial-parallel)** — 온디바이스 Mobile-VTON 디노이저(1024×768)를 단일 Pi OOM 없이 여러 RPi5에 H축(row-band) 공간 분산하여 추론(`src/ondevice_vton/parallel/`, 2대 이상·권장 4대, 전용 vton venv)

---

## 시스템 아키텍처

### Phase 1 — 단일 RPi (Pilot)

```
[웹캠]
  ↓
[MediaPipe Pose] ── person 감지(랜드마크 → 바운딩박스) → 수직 분할
  ↓ anchor_category 크롭 (tops 또는 bottoms)   ↓
[CLIP ViT-B/32 ONNX]                        [OpenCV K-means]
 CLS토큰 + visual_projection                  색상 팔레트 추출 (3색)
 → L2 정규화 512-dim anchor_vec
  ↓
[FAISS IndexFlatL2] × 3 ── 동일 anchor_vec로 tops/bottoms/shoes 카테고리별 독립 검색 → 각 Top-20 후보
  ↓
[Reranker] ── α×CLIP유사도 + β×텍스트유사도 + γ×색상호환성 → 앵커 상위 3개(NUM_CANDIDATES=3) 선정
  ↓
[세트 구성] ── 각 앵커 상품의 스냅 코디(snap_id) 사용, 없으면 카테고리별 독립 검색으로 폴백
  ↓
[Flask 웹 앱] ── MJPEG 스트리밍 + 최대 3개 코디 세트 JSON + QR코드
```

**추천 구조:**
- 현재 DB: 118개 상품 (tops 40 / bottoms 40 / shoes 38), 40개 스냅 (snap_outfits.json)
- anchor_category 크롭의 CLIP 벡터로 앵커 카테고리를 검색·리랭킹해 상위 3개를 뽑고, 각 앵커 상품의 스냅 코디(snap_id)로 나머지 슬롯을 채움 (스냅이 없으면 카테고리별 독립 검색으로 폴백) → 최대 3개 코디 세트 구성

### Phase 2 — 분산 가상 피팅 (spatial-parallel Mobile-VTON 클러스터)

`VTON_BACKEND=ondevice`이면 가상 피팅 디노이저(1024×768)를 단일 Pi에서 처리할 수 없어
여러 RPi5에 **H축(row-band) 방향으로 활성화 텐서를 공간 분산**하여 추론합니다.
가중치는 모든 rank에 복제되고, 각 rank는 모든 `[B,C,H,W]` 텐서의 연속 row-band `[h0:h1)`만 소유합니다.

```
rank0 (앱 서빙 + 디노이저 최상단 band + 분산 런처)
  ↕ 경계 연산만 통신 (halo conv · GroupNorm all-reduce · self-attn K/V all-gather)
rank1..N-1 (이하 연속 band 추론 — peer)
  → 입구에서 한 번 scatter, 출구에서 한 번 gather → 단일 Pi와 동치인 합성 결과
```

- 분할 규칙은 `band_bounds`(결정적·균등 분할, 나머지는 앞 rank가 +1). 이 분할 로직은
  torch 비의존 모듈 `sp_bands.py`(`sp_common`이 re-export)로 분리되어 `tests/test_sp_common.py`(R-11)로
  메인 스위트에서 실제 실행·통과 검증된다.
- 통신을 동반하는 분산 추론 코드(scatter/gather·halo·GroupNorm·attention)는 `src/ondevice_vton/parallel/`에 있으며
  torch 포함 **전용 vton venv**(`requirements_vton.txt`)에서 실행. 클러스터 구성·실행은 [RUN.md §9](./RUN.md) 참고.

---

## 기술 스택

| 역할 | 기술 |
|---|---|
| 의류 감지 | MediaPipe Pose (model_complexity=0) |
| 이미지 임베딩 | CLIP ViT-B/32 (ONNX) |
| 한국어 텍스트 임베딩 | `jhgan/ko-sroberta-multitask` (ONNX) |
| 유사도 검색 | FAISS (CPU, IndexFlatL2) |
| 색상 분석 | OpenCV K-means |
| 웹 서버 | Flask (MJPEG 스트리밍) |
| 포즈 추적 | MediaPipe Pose (model_complexity=0) |
| 가상 피팅 | fal-ai/fashn/tryon v1.6 (기본, 외부 API) 또는 온디바이스 Mobile-VTON (`VTON_BACKEND=ondevice`) |
| 분산 가상 피팅 | spatial-parallel (H-band) — torch + torch.distributed (gloo), RPi5 클러스터 |
| 추론 런타임 | ONNX Runtime (aarch64) |
| 테스트 | pytest |
| 데이터 수집 | requests + BeautifulSoup (무신사) |

---

## 장비

| 항목 | 사양 |
|---|---|
| 보드 | Raspberry Pi 5 Model B × 4대 |
| CPU | ARM Cortex-A76, 4코어 (aarch64) |
| RAM | 8GB |
| OS | Debian GNU/Linux 12 (bookworm) |
| 카메라 | 각 보드에 카메라 모듈 연결 |

---

## 프로젝트 구조

```
.
├── src/
│   ├── app.py                 # Flask 진입점 (/, /detection_feed, /pose_poll, /pose_reset, /recommend, /tryon, /product_image, /brand_logo, /health)
│   ├── camera.py              # 웹캠 캡처 + MJPEG 스트리밍 (백그라운드 스레드)
│   ├── detector.py            # MediaPipe Pose person 감지 + 수직 분할 (tops/bottoms/shoes)
│   ├── embedder.py            # CLIP ViT-B/32 이미지 임베딩 (CLS + visual_projection → 512-dim)
│   ├── text_encoder.py        # ko-sroberta Mean Pooling 한국어 텍스트 임베딩 (768-dim)
│   ├── searcher.py            # FAISS 유사도 검색 (카테고리 필터, over-fetch k×3)
│   ├── reranker.py            # HSV 색상 호환성 점수 + 팔레트 추출 (+ 리랭킹 로직 보유)
│   ├── recommender.py         # 추천 파이프라인 (anchor 리랭킹 상위 3개 → 스냅/폴백 세트 구성 → 최대 3개 코디 세트)
│   ├── pose.py                # MediaPipe 포즈 추적 + 존 감지 + 오버레이 렌더링
│   ├── tryon.py               # 가상 피팅 — fal-ai/fashn/tryon v1.6 외부 API 백엔드 (기본)
│   ├── tryon_ondevice.py      # 가상 피팅 — 온디바이스 Mobile-VTON 백엔드 (VTON_BACKEND=ondevice)
│   ├── ondevice_vton/         # 벤더링된 Mobile-VTON + spatial-parallel(H-band) 분산 추론 (parallel/sp_*.py)
│   └── templates/
│       └── index.html         # 웹 UI (anchor 토글 + 코디 세트 카드 3개)
├── data/
│   ├── musinsa_db/
│   │   ├── tops/              # 상의 이미지 (40장)
│   │   ├── bottoms/           # 하의 이미지 (40장)
│   │   ├── shoes/             # 신발 이미지 (38장)
│   │   ├── metadata.json      # 118개 상품 메타데이터 (product_id, snap_id, category, url, image_path, name, style_text, dominant_color)
│   │   └── snap_outfits.json  # 40개 스냅 코디 (snap_id → {tops, bottoms, shoes} 상품 ID 목록, 세트 구성에 사용)
│   └── faiss_index/
│       ├── index.bin          # CLIP 이미지 벡터 기반 FAISS 인덱스 (118벡터)
│       ├── id_map.json        # 인덱스 순서 → product_id 매핑
│       └── style_vectors.npy  # 상품별 ko-sroberta 스타일 벡터 (118×768)
├── musinsa_out/
│   └── result.json            # 크롤러 원본 출력 (snap_id 기반 코디 세트, 남녀 통합, gender 필드 포함)
├── models/
│   ├── clip_image_encoder.onnx    # ~310MB
│   ├── clip_preprocessor/         # visual_projection.npy (768×512)
│   └── ko_sroberta/               # ONNX + tokenizer (~460MB)
├── crawl_musinsa.py           # 무신사 snap 크롤러 (snap_id 단위 코디 세트 수집)
├── scripts/
│   ├── setup_models.py        # ONNX 모델 다운로드·변환 (최초 1회)
│   ├── convert_musinsa_out.py # musinsa_out → data/musinsa_db/ + snap_outfits.json
│   ├── build_image_index.py   # CLIP FAISS 인덱스 빌드
│   └── build_style_vectors.py # ko-sroberta 스타일 벡터 빌드
├── tests/                     # pytest 자동화 테스트 (11개 파일, R-01~R-13 매핑)
├── deliverables/              # 제출 산출물
├── project_plans/             # 구현 계획 문서
├── project_guidelines/        # 평가 가이드
├── requirements.txt           # 보드(RPi5) — 앱 실행 + ONNX 추론
├── requirements_local.txt     # 로컬(Mac) — 모델 변환·크롤링·인덱스 빌드
├── requirements_vton.txt      # vton 전용 venv — 온디바이스 Mobile-VTON(torch 등)
├── .env.example               # 환경 변수 예시 (VTON_BACKEND, FAL_KEY 등)
├── RUN.md
└── README.md
```

---

## 빠른 시작

> 전체 실행 가이드는 [RUN.md](./RUN.md)를 참고하세요.

### [보드 — RPi5] 앱 실행

```bash
# 1. SSH 접속
ssh willtek@10.56.130.185
cd /home/willtek/work/Project/SmartFittingRoom

# 2. 가상환경 생성 및 활성화 (최초 1회)
python3 -m venv env
source env/bin/activate

# 3. 패키지 설치
pip install -r requirements.txt

# 4. models / data 다운로드 (구글 드라이브, 최초 1회)
pip install gdown
gdown 1eJcoJdGNR4G3x8MMlnQXskF_hjQdIAGs -O models.zip
gdown 1oVkg5RhHaNFbN4d2zy7_Ue4gAEYLuCb4 -O data.zip
unzip models.zip && rm models.zip
unzip data.zip   && rm data.zip

# 5. 앱 실행
python src/app.py
# → 브라우저에서 http://10.56.130.185:5000 접속

# 6. 테스트 실행
pytest tests/ -v --tb=short 2>&1 | tee test-results/pytest_log.txt
```

### [로컬 — Mac] 데이터·모델 직접 빌드 시

```bash
cd SmartFittingRoom

# 가상환경 생성 및 활성화
python3 -m venv env
source env/bin/activate

# 패키지 설치
pip install -r requirements_local.txt

# ONNX 모델 변환 + 크롤링 + 인덱스 빌드 (인터넷 필요)
python scripts/setup_models.py
python crawl_musinsa.py                  # snap 기반 크롤링 (musinsa_out/ 생성)
python scripts/convert_musinsa_out.py    # snap → data/musinsa_db/ + snap_outfits.json
python scripts/build_image_index.py
python scripts/build_style_vectors.py

# 보드로 전송
rsync -av ./data/   willtek@10.56.130.185:/home/willtek/work/Project/SmartFittingRoom/data/
rsync -av ./models/ willtek@10.56.130.185:/home/willtek/work/Project/SmartFittingRoom/models/
```

---

## 요구사항

| ID | 내용 | 검증 기준 |
|---|---|---|
| R-01 | 웹캠 라이브 피드 표시 | MJPEG 스트림 끊김 없이 출력 |
| R-02 | 의류 영역 자동 감지 | 바운딩 박스 반환, 신뢰도 ≥ 0.5 |
| R-03 | CLIP 이미지 임베딩 | 512-dim 벡터, 추론 ≤ 500ms |
| R-04 | 색상 팔레트 추출 | 앵커 크롭에서 지배색 3개 UI 표시 |
| R-05 | FAISS 유사도 검색 | 앵커 카테고리 Top-20 반환, ≤ 100ms |
| R-06 | 한국어 텍스트 임베딩 | ko-sroberta ONNX로 768-dim 벡터 인코딩 후 Reranker 혼합 점수에 반영 |
| R-07 | 혼합 점수 기반 코디 세트 표시 | 앵커 리랭킹 상위 3개 + 스냅/폴백 → 최대 3개 코디 세트(상의+하의+신발) + QR코드 |
| R-08 | 무신사 QR코드 생성 | 스캔 시 상품 페이지 이동 |
| R-09 | 전체 응답 시간 ≤ 2초 | 5회 평균 ≤ 2,000ms |
| R-10 | AI 추론 On-Device 동작 (가상 피팅 제외) | AI 추론 파이프라인은 인터넷 차단 시 정상 동작; 가상 피팅은 `FAL_KEY` 외부 API 사용 |
| R-11 | 분산 가상 피팅 (spatial-parallel Mobile-VTON) | row-band 분할이 결정적·무중첩으로 전체 높이를 덮음(`tests/test_sp_common.py`), N대 RPi5에서 단일 Pi와 동치인 합성 결과 (클러스터 시연) |
| R-12 | 포즈 기반 자동 추천 트리거 | 화면 중앙 존 3초 유지 → `/recommend` 자동 호출, `triggered: true` 확인 |
| R-13 | 가상 피팅(Virtual Try-On) | `/tryon` SSE로 tops·bottoms 합성 이미지 반환 및 UI 표시 |

---

## 팀

SW Bootcamp 13기 | 팀원 4명

---

## 라이선스

본 프로젝트는 교육 목적으로 제작되었습니다.
무신사 크롤링 데이터는 교육용 비상업적 목적으로만 사용합니다.
