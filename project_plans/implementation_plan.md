# LG Styler Smart Fitting Room — 구현 계획

> On-Device AI 기반 실시간 패션 추천 시스템 (시각 + 한국어 텍스트 멀티모달)
> 팀원 4명 / 기간 1주일 / 장비: Raspberry Pi 5 × 4대

---

## 목표

웹캠으로 착용 의상을 감지하고, **한국어 자연어 텍스트 조건**과 결합하여
무신사 DB에서 코디를 추천하는 On-Device AI 웹 앱을 구현한다.
"LG Styler 앞에 서면 오늘의 코디를 추천해준다"는 컨셉으로 브랜딩.

---

## 단계 구성

| 단계 | 내용 | 기간 |
|---|---|---|
| Phase 1 | 단일 RPi 기반 Smart Fitting Room (Pilot) | Day 1~4 |
| Phase 2 | 4대 RPi 분산 Edge AI 파이프라인 | Day 5~6 |
| 마무리 | 테스트·문서·시연 영상 | Day 7 |

---

## Phase 1: 단일 RPi Pilot

### 시스템 파이프라인

```
[웹캠]
  ↓
[YOLOv8n ONNX] — person 감지(COCO class 0) → 수직 분할 비율로 크롭
  tops: 0~45%, bottoms: 40~80%, shoes: 75~100%
  ↓ (tops/bottoms/shoes 각 크롭 개별 처리)        ↓
[CLIP ViT-B/32 ONNX]                         [OpenCV K-means]
 CLS 토큰(768-dim) + visual_projection.npy    색상 팔레트 추출 (3색, BGR→RGB)
 → L2 정규화 512-dim 벡터
  ↓ (카테고리별 개별 검색, over-fetch k×3)
[FAISS IndexFlatL2] — 카테고리 필터 → 후보 Top-50
  ↓
[ko-sroberta ONNX + Mean Pooling] ← [한국어 텍스트 입력]
 L2 정규화 768-dim 벡터 ↔ 상품 스타일 벡터(style_vectors.npy) 내적
  ↓
[Reranker] — α×clip_sim + β×text_sim + γ×color_compat
  텍스트 있음: α=0.4, β=0.4, γ=0.2
  텍스트 없음: α=0.6, β=0.0, γ=0.4
  카테고리당 top-1 추출 → 전체 합산 정렬 → 최종 Top-3
  ↓
[Flask 웹 앱] — MJPEG 스트리밍 + 추천 JSON + base64 QR코드
```

**2단계 검색 구조:**
- 1단계 (시각): CLIP 이미지 임베딩 → FAISS로 카테고리당 후보 50개 추출 (이미지↔이미지)
- 2단계 (텍스트): ko-sroberta Mean Pooling으로 한국어 입력 인코딩 → 사전 빌드된 스타일 벡터와 내적 → 리랭킹 (텍스트↔텍스트)
- 카테고리별 top-1 선정 후 전체 정렬하여 top-3 반환

### 사용 보드

- **RPi1** (10.56.130.185) — 메인 보드, 카메라 + 전체 파이프라인 실행

### 기술 스택

| 역할 | 선택 | 비고 |
|---|---|---|
| 의류 감지 | YOLOv8n (ONNX) | COCO person class 0 감지 + 수직 분할 (tops/bottoms/shoes) |
| 이미지 임베딩 | CLIP ViT-B/32 (ONNX) | CLS 토큰(768-dim) × visual_projection.npy → 512-dim, L2 정규화 |
| 한국어 텍스트 임베딩 | `jhgan/ko-sroberta-multitask` (ONNX) | Mean Pooling → 768-dim, L2 정규화, 추론 ~200ms |
| 유사도 검색 | FAISS IndexFlatL2 | CPU 전용, over-fetch(k×3) 후 카테고리 필터 |
| 색상 분석 | OpenCV K-means (BGR) | k=3, 빈도순 정렬 → #RRGGBB hex 반환 |
| 색상 호환성 | HSV 규칙 기반 | 유사색(±30°)=1.0, 보색(150~210°)=0.6, 무채색=0.8, 기타=0.4 |
| 웹 서버 | Flask | MJPEG 스트리밍 + REST API + base64 annotated frame |
| QR코드 생성 | `qrcode` 라이브러리 | PNG base64로 JSON에 포함 |
| 테스트 | pytest | 자동화 테스트 7종 (모델 없을 시 skip 처리) |

### 추론 시간 예측 (RPi5 기준)

| 단계 | 예상 시간 |
|---|---|
| YOLO 감지 | ~50ms |
| CLIP 이미지 인코딩 | ~300ms |
| FAISS 검색 (Top-50) | ~5ms |
| ko-sroberta 텍스트 인코딩 | ~200ms |
| 스타일 벡터 리랭킹 | ~5ms |
| **전체** | **~560ms** → R-06 (2초 이내) 충족 |

### 프로젝트 디렉토리 구조

```
project/
├── src/
│   ├── app.py              # Flask 진입점, 라우팅 (/, /video_feed, /recommend, /health)
│   ├── camera.py           # 웹캠 캡처 백그라운드 스레드 + MJPEG 스트리밍 (FPS 오버레이)
│   ├── detector.py         # YOLOv8n ONNX: person 감지 + 수직 분할 → crops dict
│   ├── embedder.py         # CLIP ViT-B/32 ONNX: CLS 토큰 + visual_projection → 512-dim
│   ├── text_encoder.py     # ko-sroberta ONNX: Mean Pooling → 768-dim L2 정규화
│   ├── searcher.py         # FAISS 검색 (over-fetch × 3, 카테고리 필터)
│   ├── reranker.py         # 리랭킹 + HSV 색상 호환성 + 팔레트 추출 (extract_palette)
│   ├── recommender.py      # 통합 파이프라인: 카테고리별 검색 → 합산 top-3
│   └── templates/
│       └── index.html      # 웹 UI (카메라 피드 + 텍스트 입력 + 추천 패널)
├── data/
│   ├── musinsa_db/
│   │   ├── tops/           # 상의 이미지 (convert_musinsa_out.py 생성)
│   │   ├── bottoms/        # 하의 이미지
│   │   ├── shoes/          # 신발 이미지
│   │   └── metadata.json   # product_id, url, category, image_path, name, style_text, dominant_color
│   └── faiss_index/
│       ├── index.bin       # CLIP 이미지 벡터 기반 FAISS 인덱스
│       └── style_vectors.npy  # 상품별 ko-sroberta 스타일 벡터 (사전 빌드)
├── musinsa_out/
│   └── musinsa_db/         # 새 크롤러 원본 출력 (snap_id 기반 코디 세트)
├── models/
│   ├── yolov8n.onnx
│   ├── clip_image_encoder.onnx
│   ├── clip_preprocessor/      # CLIP 전처리 설정
│   └── ko_sroberta/            # ko-sroberta ONNX 모델 파일
├── scripts/
│   ├── convert_musinsa_out.py  # musinsa_out 스냅 데이터 → data/musinsa_db 변환
│   ├── build_image_index.py    # CLIP 이미지 임베딩 → FAISS 인덱스 빌드
│   └── build_style_vectors.py  # 상품 태그 → ko-sroberta 스타일 벡터 빌드
├── tests/
│   ├── test_detection.py
│   ├── test_embedding.py       # CLIP 이미지 임베딩 검증
│   ├── test_text_encoding.py   # ko-sroberta 한국어 인코딩 검증
│   ├── test_search.py
│   ├── test_reranking.py       # 텍스트 리랭킹 로직 검증
│   ├── test_api.py
│   └── test_stream.py
├── test-results/               # pytest 실행 결과 로그
├── requirements.txt
├── README.md
└── RUN.md
```

### 데이터 준비 (사전 작업, 로컬 맥에서 실행)

새 크롤러가 `musinsa_out/musinsa_db/result.json` (snap_id 기반 코디 세트)을 생성하면,
`convert_musinsa_out.py`로 기존 플랫 상품 목록 형식으로 변환한다.
snap 단위 해시태그가 각 상품의 `style_text`로 활용된다.

**원본 데이터 스키마** (`musinsa_out/musinsa_db/result.json`):
```json
{
  "snap_id": "1518836080194369513",
  "description": "스타일링 설명 (한국어)",
  "hashtags": ["dailylook", "ootd", "데일리룩", ...],
  "items_by_slot": {
    "상의": [{ "goodsNo": "6237058", "name": "린넨 니트", "saved_path": "images/상의/...", ... }],
    "하의": [...],
    "신발": [...]
  }
}
```

**변환 후 metadata.json 스키마** (`data/musinsa_db/metadata.json`):
```json
{
  "product_id": "musinsa_6237058",
  "category": "tops",
  "url": "https://www.musinsa.com/products/6237058",
  "image_path": "tops/musinsa_6237058.jpg",
  "name": "린넨 스카시 카라 반팔 니트 OLIVE",
  "style_text": "린넨 스카시 카라 반팔 니트 OLIVE, dailylook, ootd, 남자데일리룩, ...",
  "dominant_color": "#3D6B9F"
}
```

```bash
# 1. musinsa_out 스냅 데이터 변환 (로컬 맥)
python scripts/convert_musinsa_out.py

# 2. CLIP 이미지 인덱스 빌드
python scripts/build_image_index.py

# 3. ko-sroberta 스타일 벡터 빌드
python scripts/build_style_vectors.py

# 4. RPi로 전송
rsync -av ./data/ willtek@10.56.130.185:/home/willtek/project/data/
rsync -av ./models/ willtek@10.56.130.185:/home/willtek/project/models/
```

예상 용량: 이미지 ~200MB + 모델 ~700MB + 인덱스/벡터 ~10MB

### 웹 UI 레이아웃

```
┌─────────────────────────────────────────────────────────┐
│  LG Styler Smart Fitting Room                           │
├──────────────────────┬──────────────────────────────────┤
│                      │  감지된 아이템                    │
│  [카메라 라이브 피드] │  [ 상의 ] [ 하의 ] [ 신발 ]     │
│  (MJPEG 스트림)      │                                  │
│                      │  색상 팔레트  ■ ■ ■              │
│  [바운딩 박스 오버레이]│──────────────────────────────────│
│                      │  어떤 상황인가요?                 │
│                      │  [텍스트 입력창            ] [→] │
│                      │  예: "맑은 날 여자친구와 데이트"  │
│                      │──────────────────────────────────│
│                      │  오늘의 추천 코디                 │
│                      │  [썸네일] [썸네일] [썸네일]      │
│                      │  [QR → 무신사 링크]              │
└──────────────────────┴──────────────────────────────────┘
```

### Flask 주요 엔드포인트

| 엔드포인트 | 역할 |
|---|---|
| `GET /` | 메인 웹 UI |
| `GET /video_feed` | MJPEG 스트리밍 |
| `POST /recommend` | 이미지 감지 + 텍스트 입력 → 추천 결과 반환 (JSON) |
| `GET /health` | 서버 상태 확인 |

`/recommend` 요청 body:
```json
{
  "text_query": "맑은 날 여자친구와 데이트하려고 해. 추천해줘.",
  "use_camera": true
}
```

### 리랭킹 로직 상세

```
1단계 출력: FAISS Top-50 후보 (CLIP 이미지 유사도 기준)

2단계 입력:
  - 텍스트 쿼리 벡터 (ko-sroberta)
  - 각 후보 상품의 스타일 벡터 (ko-sroberta, 사전 빌드)

2단계 점수:
  최종_점수 = α × CLIP_유사도 + β × 텍스트_유사도 + γ × 색상_호환성
  (기본값: α=0.4, β=0.4, γ=0.2)

  텍스트 입력 없을 시: α=0.6, β=0.0, γ=0.4 (시각+색상만 사용)

출력: 점수 상위 Top-3
```

### 색상 호환성 로직

HSV 색공간 기반 규칙:
- **유사색**: 색상각 차이 ±30° 이내 → 높은 호환성 점수
- **보색**: 색상각 차이 150~210° → 중간 호환성 점수
- **무채색 (흰/회/검)**: 모든 색상과 호환 → 기본 점수 부여

---

## Phase 2: 분산 Edge AI 파이프라인 (계획됨 — 미구현)

> **현재 상태**: Phase 1 단일 RPi 파이프라인 완성. Phase 2 워커 서버는 미구현.

### 역할 분담

| 보드 | IP | 역할 |
|---|---|---|
| RPi1 | 10.56.130.185 | 카메라 캡처 + YOLOv8n 감지 + 웹 UI 서빙 |
| RPi2 | 10.56.130.178 | CLIP 이미지 임베딩 + ko-sroberta 텍스트 임베딩 워커 |
| RPi3 | 10.56.130.182 | FAISS 검색 + 텍스트 리랭킹 워커 |
| RPi4 | 미확인 | 결과 집계 + 색상 호환성 점수 적용 + 최종 응답 |

### 통신 방식

보드 간 통신은 **HTTP REST** 우선 (단순, 안정적):

```
RPi1: 크롭 이미지 + 텍스트 쿼리 → POST /embed (RPi2:5001)
RPi2: CLIP 이미지 벡터 + ko-sroberta 텍스트 벡터 → POST /search (RPi3:5002)
RPi3: Top-50 후보 + 리랭킹 결과 → POST /aggregate (RPi4:5003)
RPi4: 색상 호환성 최종 점수 → Top-3 결과 → RPi1에 반환
```

시간 여유 시 ZeroMQ(PUSH/PULL 패턴)로 교체해 지연 감소.

### Phase 2 디렉토리 추가 (미구현)

```
src/
├── worker_embed.py       # RPi2용 — CLIP 이미지 + ko-sroberta 텍스트 임베딩 워커 (미구현)
├── worker_search.py      # RPi3용 — FAISS 검색 + 텍스트 리랭킹 워커 (미구현)
└── worker_aggregate.py   # RPi4용 — 색상 호환성 + 결과 집계 서버 (미구현)
```

---

## 요구사항 → 테스트 매핑 (예비)

| 요구사항 ID | 내용 | 검증 기준 | 담당 테스트 |
|---|---|---|---|
| R-01 | 웹캠 라이브 피드 표시 | `/video_feed` 200 OK, 끊김 없음 | `test_stream.py` |
| R-02 | 의류 감지 (상/하/신발) | 바운딩 박스 반환, 신뢰도 ≥ 0.5 | `test_detection.py` |
| R-03 | CLIP 이미지 임베딩 추출 | 512-dim 벡터 반환, 추론 ≤ 500ms | `test_embedding.py` |
| R-04 | 색상 팔레트 추출 | 지배색 3개 반환 | `test_search.py` |
| R-05 | 유사 아이템 시각 검색 | Top-50 후보 반환, 검색 ≤ 100ms | `test_search.py` |
| R-06 | 한국어 텍스트 입력 기반 리랭킹 | 텍스트 쿼리 입력 시 결과 순위 변화 확인 | `test_text_encoding.py`, `test_reranking.py` |
| R-07 | 전체 추천 응답 시간 | `/recommend` 응답 ≤ 2초 | `test_api.py` |
| R-08 | QR코드 생성 | 무신사 URL 포함 QR 이미지 반환 | `test_api.py` |
| R-09 | (Phase 2) 분산 파이프라인 동작 | 4대 보드 정상 통신, 동일 결과 반환 | 별도 integration test |

---

## 팀원 역할 분담

| 팀원 | Phase 1 담당 | Phase 2 담당 |
|---|---|---|
| A | `crawl_musinsa.py` + `build_image_index.py` + `build_style_vectors.py` + `metadata.json` 설계 | RPi3 `worker_search.py` |
| B | `detector.py` (YOLOv8n) + `embedder.py` (CLIP) + ONNX 변환 | RPi2 `worker_embed.py` |
| C | `app.py` + `camera.py` + `templates/index.html` (MJPEG + 텍스트 입력 UI) | RPi1 통합 + RPi4 `worker_aggregate.py` |
| D | `text_encoder.py` + `reranker.py` + `recommender.py` + `tests/` 전체 | 통합 테스트 + RUN.md 작성 |

---

## 일정

| Day | 주요 작업 |
|---|---|
| Day 1 | 환경 세팅 (ONNX Runtime, FAISS, sentence-transformers) + 무신사 크롤링 시작 (이미지 + 태그) |
| Day 2 | 크롤링 완료 + CLIP 인덱스 빌드 + ko-sroberta 스타일 벡터 빌드 + YOLOv8n ONNX 추론 검증 |
| Day 3 | Flask 웹 앱 + MJPEG 스트리밍 + CLIP 이미지 검색 연결 (텍스트 없는 버전 완성) |
| Day 4 | ko-sroberta 텍스트 리랭킹 + 텍스트 입력 UI + 색상 호환성 + QR코드 → **Phase 1 완성** |
| Day 5 | Phase 2 워커 서버 구현 (RPi2 임베딩, RPi3 검색, RPi4 집계) + 보드 간 통신 연결 |
| Day 6 | 분산 파이프라인 통합 테스트 + 안정화 |
| Day 7 | pytest 전체 실행 + test-results/ 저장 + RUN.md + 시연 영상 녹화 + 산출물 정리 |

---

## 산출물 체크리스트 (output_guide.md 기준)

- [x] `요구사항_명세서.md` — R-01~R-11 + 검증 기준 + 시연 타임스탬프 (`deliverables/요구사항명세서.md`)
- [ ] `결과보고서.pdf/.pptx` — 설계·구현·결과 정리
- [ ] `결과파일.zip` — src/ + tests/ + test-results/ + README.md + RUN.md + requirements.txt + .git/
- [ ] `시연영상.mp4` — 아래 타임스탬프 기준 녹화
- [ ] `포스터.pdf/.pptx`
- [ ] `평가결과.html` — 평가 Agent 직접 실행 결과

### 시연 영상 타임스탬프 (예정)

| 시간 | 장면 | 요구사항 |
|---|---|---|
| 00:00 | 시스템 부팅 / 웹 앱 접속 | — |
| 00:15 | 카메라 앞에 서기 → 라이브 피드 확인 | R-01 |
| 00:25 | YOLO 바운딩 박스 감지 표시 | R-02 |
| 00:35 | 색상 팔레트 추출 결과 | R-04 |
| 00:45 | 텍스트 없이 시각 검색만으로 Top-3 추천 | R-03, R-05, R-07 |
| 01:00 | 텍스트 입력: "맑은 날 여자친구와 데이트하려고 해" → 추천 결과 변화 | R-06 |
| 01:15 | QR코드 스캔 → 무신사 상품 페이지 연결 | R-08 |
| 01:30 | (Phase 2) 4대 RPi 분산 파이프라인 구조 설명 | R-09 |

---

## 환경 세팅 (RPi1 기준)

```bash
# SSH 접속
ssh LGBoard

# Python 패키지 설치
pip install -r requirements.txt

# requirements.txt 주요 항목
flask==3.0.3
opencv-python==4.10.0.84
onnxruntime==1.19.2              # aarch64 지원 버전
ultralytics==8.2.100             # YOLOv8n
faiss-cpu==1.8.0
numpy==1.26.4
Pillow==10.4.0
qrcode==7.4.2
pytest==8.3.3
requests==2.32.3
beautifulsoup4==4.12.3           # 크롤러용
transformers==4.44.2             # ko-sroberta 토크나이저
sentence-transformers==3.1.1     # ko-sroberta 모델 로드 + ONNX 변환
optimum[onnxruntime]==1.22.0     # HuggingFace → ONNX 변환
open-clip-torch==2.26.1          # CLIP ONNX 변환용

# 앱 실행
python src/app.py                # http://10.56.130.185:5000

# 테스트 실행
pytest tests/ -v --tb=short 2>&1 | tee test-results/pytest_log.txt
```

---

## 주요 리스크 및 대응

| 리스크 | 대응 |
|---|---|
| CLIP 추론이 RPi5에서 느릴 경우 | 추론 주기를 0.5초→1초로 늘리거나, 이미지 해상도 축소 |
| ko-sroberta ONNX 변환 실패 | `sentence-transformers` 직접 로드로 폴백 (느리지만 동작) |
| 무신사 크롤링 차단 | User-Agent 변경 + 요청 간격 조절, 또는 수동 수집으로 전환 |
| Phase 2 보드 간 지연 누적 | HTTP 타임아웃 설정 + Phase 1 결과로 데모 폴백 |
| RPi4 IP 미확인 | 팀 내 확인 후 `config` 파일에 별칭 등록 |
