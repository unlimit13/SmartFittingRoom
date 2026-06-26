# RUN.md — 실행·재현 가이드

DevOps 8개 기준을 모두 충족합니다.

> **환경 구분**
> - **로컬 (Mac)**: 모델 변환·크롤링·인덱스 빌드 → 결과물을 보드로 전송
> - **보드 (RPi5, 10.56.130.185)**: Flask 앱 실행 + ONNX 추론

---

## 1. 라이브러리 목록

환경별로 분리되어 있습니다.

| 파일 | 대상 환경 | 용도 |
|---|---|---|
| `requirements.txt` | Raspberry Pi 5 (보드) | 앱 실행 + ONNX 추론 |
| `requirements_local.txt` | Mac (로컬) | 모델 변환, 크롤링, 인덱스 빌드 |

**`requirements.txt` (보드):**
```
flask==3.0.3
opencv-python==4.10.0.84
onnxruntime==1.27.0
faiss-cpu==1.8.0
numpy==1.26.4
Pillow==10.4.0
qrcode==7.4.2
pytest==8.3.3
requests==2.32.3
transformers==4.44.2   # AutoTokenizer (ko-sroberta 추론 시 필요)
fal_client==1.0.0
mediapipe==0.10.18
ai-edge-litert==2.1.5
```

**`requirements_local.txt` (로컬):**
```
torch>=2.0.0
ultralytics==8.2.100
transformers==4.44.2
optimum[onnxruntime]==1.22.0
open-clip-torch==2.26.1
sentence-transformers==3.1.1
requests==2.32.3
beautifulsoup4==4.12.3
onnxruntime==1.19.2
faiss-cpu==1.8.0
numpy==1.26.4
opencv-python==4.10.0.84
Pillow==10.4.0
```

---

## 2. 라이브러리 버전 고정

모든 버전이 각 requirements 파일에 고정되어 있습니다.

---

## 3. 빌드·설치 방법

**보드 (RPi5) — 앱 실행 환경:**
```bash
ssh willtek@10.56.130.185
cd /home/willtek/work/Project/SmartFittingRoom

# 가상환경 생성 (최초 1회)
python3 -m venv env

# 가상환경 활성화
source env/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

**로컬 (Mac) — 데이터 준비 환경:**
```bash
cd SmartFittingRoom

# 가상환경 생성 (최초 1회)
python3 -m venv env

# 가상환경 활성화
source env/bin/activate

# 패키지 설치
pip install -r requirements_local.txt
```

---

## 4. 실행 방법

> **권장 경로**: 구글 드라이브에서 사전 빌드된 models / data를 받아 바로 앱을 실행합니다.
> 직접 빌드가 필요한 경우에만 Step 1~3을 수행하세요.

---

### [보드] Step 0: 구글 드라이브에서 모델·데이터 다운로드 (권장)

사전 빌드된 `models/`, `data/`를 구글 드라이브에서 받아 바로 사용합니다.

```bash
# 보드에서 실행
ssh willtek@10.56.130.185
cd /home/willtek/work/Project/SmartFittingRoom
source env/bin/activate

pip install gdown

# models 다운로드 (~780MB)
gdown 1eJcoJdGNR4G3x8MMlnQXskF_hjQdIAGs -O models.zip
unzip models.zip && rm models.zip

# data 다운로드 (~200MB)
gdown 10HAKEtpDnk1nsX11xxkwPIVXeYmK_fmm -O data.zip
unzip data.zip && rm data.zip
```

Step 0 완료 후 바로 **[보드] Step 5: 앱 실행**으로 건너뜁니다.

---

### [로컬] Step 1: ONNX 모델 변환 (최초 1회)

```bash
# 인터넷 연결 필요, ~800MB 다운로드
python scripts/setup_models.py
```

생성 파일:
```
models/
├── yolov8n.onnx                        (~13MB)
├── clip_image_encoder.onnx             (~310MB)
├── clip_preprocessor/visual_projection.npy
└── ko_sroberta/model.onnx + tokenizer  (~460MB)
```

### [로컬] Step 2: 무신사 스냅 데이터 변환

새 크롤러가 `musinsa_out/musinsa_db/result.json`과 이미지를 생성한 이후 실행합니다.

```bash
# snap 기반 크롤링 (musinsa_out/result.json + 이미지 생성)
python crawl_musinsa.py
python scripts/convert_musinsa_out.py
```

생성 파일: `data/musinsa_db/{tops,bottoms,shoes}/*.jpg|png` + `metadata.json`

> **데이터 형식**: `musinsa_out/musinsa_db/result.json` (snap_id 기반 코디 세트) →
> `data/musinsa_db/metadata.json` (상품별 플랫 목록)으로 변환됩니다.

### [로컬] Step 3: FAISS 인덱스 + 스타일 벡터 빌드

```bash
python scripts/build_image_index.py    # CLIP 임베딩 → FAISS 인덱스
python scripts/build_style_vectors.py  # 상품 텍스트 → ko-sroberta 벡터
```

생성 파일: `data/faiss_index/index.bin`, `id_map.json`, `style_vectors.npy`

### [로컬] Step 4: 보드로 전송

```bash
rsync -av ./data/   willtek@10.56.130.185:/home/willtek/work/Project/SmartFittingRoom/data/
rsync -av ./models/ willtek@10.56.130.185:/home/willtek/work/Project/SmartFittingRoom/models/
```

전송 예상 용량: 이미지 ~200MB + 모델 ~800MB + 인덱스 ~10MB

### [보드] Step 5: 앱 실행

```bash
ssh willtek@10.56.130.185
cd /home/willtek/work/Project/SmartFittingRoom
source env/bin/activate
python src/app.py
# → 브라우저에서 http://10.56.130.185:5000 접속
```

---

## 5. 환경 변수·설정 파일

모든 경로는 프로젝트 루트 기준 상대 경로로 자동 설정됩니다.

**가상 피팅(Virtual Try-On) 사용 시 필요:**
```bash
export FAL_KEY=your_fal_api_key_here
```
`FAL_KEY` 미설정 시 `/tryon` 엔드포인트는 동작하지 않으나, 추천 파이프라인(YOLO·CLIP·FAISS)은 정상 동작합니다.

카메라 인덱스를 변경하려면 `src/camera.py`의 `Camera(device_index=0)` 인자를 수정하세요.

---

## 6. 외부 서비스·자원

- **인터넷 필요**: `scripts/setup_models.py` (HuggingFace), `crawl_musinsa.py` (무신사)
- **모델 출처**:
  - YOLOv8n: Ultralytics (자동 다운로드)
  - CLIP ViT-B/32: `openai/clip-vit-base-patch32` (HuggingFace)
  - ko-sroberta: `jhgan/ko-sroberta-multitask` (HuggingFace)
- **앱 실행 중 (보드)**:
  - AI 추론 파이프라인(YOLO·CLIP·FAISS·ko-sroberta): 외부 API 호출 없음, 완전 On-Device
  - **가상 피팅 (선택적)**: `fal-ai/fashn/tryon v1.6` 외부 API 사용 — `FAL_KEY` 환경 변수 필요

---

## 7. 빌드·설치 성공 기준

**로컬:**
```bash
pip install -r requirements_local.txt
python -c "import torch, ultralytics, transformers, optimum, faiss, cv2"
# No errors → 설치 성공
```

**보드:**
```bash
pip install -r requirements.txt
python -c "import flask, cv2, onnxruntime, faiss, numpy, qrcode, transformers"
# No errors → 설치 성공
```

---

## 8. 애플리케이션 실행 확인

```bash
# 보드에서
source env/bin/activate
python src/app.py &
curl http://localhost:5000/health
# 응답: {"status": "ok"}
```

---

## 테스트 실행

```bash
# 보드에서 (가상환경 활성화 후)
source env/bin/activate
pytest tests/ -v --tb=short 2>&1 | tee test-results/pytest_log.txt

# 모델·데이터 없이도 mock 기반 테스트는 동작함 (skipif 처리)
```
