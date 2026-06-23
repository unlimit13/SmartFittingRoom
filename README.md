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

- **실시간 의류 감지** — YOLOv8n으로 상의/하의/신발 영역 자동 감지 및 바운딩 박스 표시
- **이미지 유사도 검색** — CLIP ViT-B/32 임베딩 + FAISS로 무신사 DB(1,000장)에서 유사 아이템 검색
- **한국어 텍스트 조건 반영** — ko-sroberta로 자연어 입력을 인코딩하여 추천 결과 리랭킹
  - 예: *"맑은 날 여자친구와 데이트하려고 해. 추천해줘."*
- **색상 팔레트 기반 코디 완성** — 착용 색상과 어울리는 아이템 우선 추천
- **무신사 상품 QR코드** — 추천 결과에서 바로 구매 페이지로 이동
- **분산 Edge AI** (Phase 2) — 4대 RPi5가 역할 분담하는 마이크로서비스 파이프라인

---

## 시스템 아키텍처

### Phase 1 — 단일 RPi (Pilot)

```
[웹캠]
  ↓
[YOLOv8n ONNX] ── 의류 영역 크롭
  ↓                        ↓
[CLIP ViT-B/32 ONNX]   [OpenCV K-means]
 이미지 임베딩              색상 팔레트 추출
  ↓
[FAISS] ── Top-50 후보 검색
  ↓
[ko-sroberta ONNX] ← [한국어 텍스트 입력]
 텍스트 리랭킹 → 최종 Top-3
  ↓
[Flask 웹 앱] ── 브라우저에서 시연
```

### Phase 2 — 분산 Edge AI (4대 RPi)

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
│   ├── app.py                 # Flask 진입점
│   ├── camera.py              # 웹캠 캡처 + MJPEG 스트리밍
│   ├── detector.py            # YOLOv8n 의류 감지
│   ├── embedder.py            # CLIP 이미지 임베딩
│   ├── text_encoder.py        # ko-sroberta 한국어 텍스트 임베딩
│   ├── searcher.py            # FAISS 유사도 검색
│   ├── reranker.py            # 텍스트 리랭킹 + 색상 호환성 점수
│   ├── recommender.py         # 통합 추천 파이프라인
│   ├── worker_embed.py        # Phase 2: RPi2 임베딩 워커
│   ├── worker_search.py       # Phase 2: RPi3 검색 워커
│   ├── worker_aggregate.py    # Phase 2: RPi4 집계 워커
│   └── templates/
│       └── index.html         # 웹 UI
├── data/
│   ├── musinsa_db/            # 무신사 의류 이미지 1,000장 + metadata.json
│   └── faiss_index/           # 사전 빌드된 FAISS 인덱스 + 스타일 벡터
├── models/
│   ├── yolov8n.onnx
│   ├── clip_image_encoder.onnx
│   └── ko_sroberta/
├── scripts/
│   ├── crawl_musinsa.py       # 무신사 크롤러
│   ├── build_image_index.py   # CLIP FAISS 인덱스 빌드
│   └── build_style_vectors.py # ko-sroberta 스타일 벡터 빌드
├── tests/                     # pytest 자동화 테스트
├── test-results/              # 테스트 실행 결과 로그
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

```bash
# 1. SSH 접속
ssh willtek@10.56.130.185

# 2. 패키지 설치
pip install -r requirements.txt

# 3. 데이터 준비 (로컬 맥에서 실행 후 RPi로 전송)
python scripts/crawl_musinsa.py --category tops --count 250
python scripts/build_image_index.py
python scripts/build_style_vectors.py
rsync -av ./data/ willtek@10.56.130.185:/home/willtek/project/data/

# 4. 앱 실행
python src/app.py
# → 브라우저에서 http://10.56.130.185:5000 접속

# 5. 테스트 실행
pytest tests/ -v --tb=short 2>&1 | tee test-results/pytest_log.txt
```

---

## 요구사항

| ID | 내용 | 검증 기준 |
|---|---|---|
| R-01 | 웹캠 라이브 피드 표시 | MJPEG 스트림 끊김 없이 출력 |
| R-02 | 의류 영역 자동 감지 | 바운딩 박스 반환, 신뢰도 ≥ 0.5 |
| R-03 | CLIP 이미지 임베딩 | 512-dim 벡터, 추론 ≤ 500ms |
| R-04 | 색상 팔레트 추출 | 지배색 3개 UI 표시 |
| R-05 | FAISS 유사도 검색 | Top-50 반환, ≤ 100ms |
| R-06 | 한국어 텍스트 기반 리랭킹 | 텍스트 입력 시 추천 순위 변화 확인 |
| R-07 | 최종 코디 추천 Top-3 표시 | 썸네일 + 상품명 + 카테고리 포함 |
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
