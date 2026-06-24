# LG Styler Smart Fitting Room

> **On-Device AI 기반 실시간 멀티모달 패션 추천 시스템**
> SW Bootcamp 13기 | Embedded Linux 기반 On-Device AI 프로젝트

---

## 프로젝트 개요

웹캠으로 착용 의상을 감지하고, **한국어 자연어 입력**과 결합하여 무신사 DB에서 어울리는 코디를 실시간으로 추천합니다.
모든 AI 추론은 클라우드 없이 **Raspberry Pi 5 위에서 완전 온디바이스**로 실행됩니다.

**"LG Styler 앞에 서면, 오늘의 코디를 추천해드립니다."**

---

## 주요 기능

- **실시간 의류 감지** — YOLOv8n으로 사람(COCO class 0)을 감지 후 바운딩 박스를 상의/하의/신발 영역으로 수직 분할
- **앵커 카테고리 선택** — 상의(tops) 또는 하의(bottoms) 중 하나를 앵커로 지정하여 CLIP 이미지 임베딩 기반 유사 검색 수행
- **스냅 기반 코디 세트 추천** — 무신사 스냅(snap) 단위로 묶인 19개 코디 세트 중 앵커 유사도 상위 3개 완전 코디(상의+하의+신발)를 추천
- **색상 팔레트 추출** — OpenCV K-means로 앵커 크롭 영역의 지배색 3개를 추출하여 UI에 표시
- **무신사 상품 QR코드** — 추천된 각 상품에서 바로 무신사 구매 페이지로 이동 가능
- **감지 스트림 분리** — `/detection_feed`로 YOLOv8n 감지 오버레이 MJPEG 스트림 별도 제공
- **분산 Edge AI** (Phase 2 — 계획됨, 미구현) — 4대 RPi5가 역할 분담하는 마이크로서비스 파이프라인

---

## 시스템 아키텍처

### Phase 1 — 단일 RPi (Pilot)

```
[웹캠]
  ↓
[YOLOv8n ONNX] ── person 감지(COCO class 0) → 수직 분할
  ↓ anchor_category 크롭 (tops 또는 bottoms)   ↓
[CLIP ViT-B/32 ONNX]                        [OpenCV K-means]
 CLS토큰 + visual_projection                  색상 팔레트 추출 (3색)
 → L2 정규화 512-dim 임베딩
  ↓
[FAISS IndexFlatL2] ── 카테고리 필터 → 후보 Top-20
  ↓ (snap_id 기반 그룹핑)
snap_id별 최고 점수로 집계 → anchor_score 상위 Top-3 스냅 선정
  ↓
[snap_outfits.json] ── snap_id → {tops, bottoms, shoes} 상품 목록 조회
  ↓
[Flask 웹 앱] ── MJPEG 스트리밍 + 3개 코디 세트 JSON + QR코드
```

**스냅 기반 추천 구조:**
- 무신사 스냅(snap) = 실제 코디 사진에서 태깅된 완전한 코디 세트 (상의+하의+신발)
- 현재 DB: 56개 상품, 19개 스냅 코디
- anchor_category 상품의 CLIP 유사도로 어울리는 코디 세트 전체를 추천

### Phase 2 — 분산 Edge AI (4대 RPi, 계획됨 — 미구현)

```
RPi1 (카메라 + YOLO + Web UI)
  → RPi2 (CLIP 이미지 임베딩 + ko-sroberta 텍스트 임베딩)
    → RPi3 (FAISS 검색 + 텍스트 리랭킹)
      → RPi4 (결과 집계 + 색상 호환성 + 최종 응답)
```

---

## 기술 스택

| 역할 | 기술 |
|---|---|
| 의류 감지 | YOLOv8n (ONNX) |
| 이미지 임베딩 | CLIP ViT-B/32 (ONNX) |
| 한국어 텍스트 임베딩 | `jhgan/ko-sroberta-multitask` (ONNX) |
| 유사도 검색 | FAISS (CPU, IndexFlatL2) |
| 색상 분석 | OpenCV K-means |
| 웹 서버 | Flask (MJPEG 스트리밍) |
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
│   ├── app.py                 # Flask 진입점 (/, /video_feed, /detection_feed, /recommend, /tryon, /product_image, /health)
│   ├── camera.py              # 웹캠 캡처 + MJPEG 스트리밍 (백그라운드 스레드)
│   ├── detector.py            # YOLOv8n person 감지 + 수직 분할 (tops/bottoms/shoes)
│   ├── embedder.py            # CLIP ViT-B/32 이미지 임베딩 (CLS + visual_projection → 512-dim)
│   ├── text_encoder.py        # ko-sroberta Mean Pooling 한국어 텍스트 임베딩 (768-dim)
│   ├── searcher.py            # FAISS 유사도 검색 (카테고리 필터, over-fetch k×3)
│   ├── reranker.py            # HSV 색상 호환성 점수 + 팔레트 추출 (+ 리랭킹 로직 보유)
│   ├── recommender.py         # 스냅 기반 파이프라인 (anchor_category → Top-3 코디 세트)
│   ├── tryon.py               # 가상 피팅 (외부 서비스 연동)
│   └── templates/
│       └── index.html         # 웹 UI (anchor 토글 + 코디 세트 카드 3개)
├── data/
│   ├── musinsa_db/
│   │   ├── tops/              # 상의 이미지 (19장)
│   │   ├── bottoms/           # 하의 이미지 (19장)
│   │   ├── shoes/             # 신발 이미지 (18장)
│   │   ├── metadata.json      # 56개 상품 메타데이터 (product_id, snap_id, category, url, image_path, name, style_text, dominant_color)
│   │   └── snap_outfits.json  # 19개 스냅 코디 (snap_id → {tops, bottoms, shoes} 상품 ID 목록)
│   └── faiss_index/
│       ├── index.bin          # CLIP 이미지 벡터 기반 FAISS 인덱스 (56벡터)
│       ├── id_map.json        # 인덱스 순서 → product_id 매핑
│       └── style_vectors.npy  # 상품별 ko-sroberta 스타일 벡터 (56×768)
├── musinsa_out/
│   └── result.json            # 크롤러 원본 출력 (snap_id 기반 코디 세트)
├── models/
│   ├── yolov8n.onnx               # ~13MB
│   ├── clip_image_encoder.onnx    # ~310MB
│   ├── clip_preprocessor/         # visual_projection.npy (768×512)
│   └── ko_sroberta/               # ONNX + tokenizer (~460MB)
├── crawl_musinsa.py           # 무신사 snap 크롤러 (snap_id 단위 코디 세트 수집)
├── scripts/
│   ├── setup_models.py        # ONNX 모델 다운로드·변환 (최초 1회)
│   ├── convert_musinsa_out.py # musinsa_out → data/musinsa_db/ + snap_outfits.json
│   ├── build_image_index.py   # CLIP FAISS 인덱스 빌드
│   └── build_style_vectors.py # ko-sroberta 스타일 벡터 빌드
├── tests/                     # pytest 자동화 테스트 (9개 파일)
├── deliverables/              # 제출 산출물
├── project_plans/             # 구현 계획 문서
├── project_guidelines/        # 평가 가이드
├── requirements.txt
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
gdown 1cNytfQV3mA8MGTsQgqnRI2anwGFFsdir -O data.zip
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
| R-06 | 한국어 텍스트 임베딩 | ko-sroberta ONNX로 768-dim 벡터 인코딩 (reranker.py에 구현, 현재 미사용) |
| R-07 | 스냅 기반 코디 세트 Top-3 표시 | 3개 완전 코디 세트(상의+하의+신발) + QR코드 포함 |
| R-08 | 무신사 QR코드 생성 | 스캔 시 상품 페이지 이동 |
| R-09 | 전체 응답 시간 ≤ 2초 | 5회 평균 ≤ 2,000ms |
| R-10 | 완전 On-Device 동작 | 인터넷 차단 상태에서 정상 동작 |
| R-11 | 분산 Edge AI (Phase 2) | 4대 보드 정상 통신 + 동일 결과 |

---

## 팀

SW Bootcamp 13기 | 팀원 4명

---

## 라이선스

본 프로젝트는 교육 목적으로 제작되었습니다.
무신사 크롤링 데이터는 교육용 비상업적 목적으로만 사용합니다.
