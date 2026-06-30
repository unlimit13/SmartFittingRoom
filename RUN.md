# RUN.md — 실행·재현 가이드

DevOps 8개 기준을 모두 충족합니다.

> **환경 구분**
> - **로컬 (Mac)**: 모델 변환·크롤링·인덱스 빌드 → 결과물을 보드로 전송
> - **보드 (RPi5, 10.56.130.185)**: Flask 앱 실행 + ONNX 추론

> ⚠️ **경로·주소는 사용 환경에 맞게 변경하세요**
> 이 문서의 명령들은 작성자 환경 기준의 예시 값입니다. 아래 항목은 **실제 사용 환경에 맞게 직접 설정**해야 합니다.
> - **프로젝트 경로**: `/home/willtek/work/Project/SmartFittingRoom`, `cd SmartFittingRoom` 등 → 본인이 체크아웃한 실제 경로로 교체
> - **보드 IP·사용자**: `willtek@10.56.130.185` → 본인 보드의 사용자·IP로 교체
> - **클러스터 IP**: `192.168.100.x` (§9 온디바이스 VTON) → 실제 클러스터 네트워크에 맞게 교체
> - **구글 드라이브 ID / `FAL_KEY`**: 본인 자원·키로 교체
> 코드 내부 경로는 프로젝트 루트 기준 상대 경로로 자동 설정되므로(§5 참고) 별도 수정이 필요 없습니다.

---

## 1. 라이브러리 목록

환경별로 분리되어 있습니다.

| 파일 | 대상 환경 | 용도 |
|---|---|---|
| `requirements.txt` | Raspberry Pi 5 (보드) | 앱 실행 + ONNX 추론 |
| `requirements_local.txt` | Mac (로컬) | 모델 변환, 크롤링, 인덱스 빌드 |
| `requirements_vton.txt` | RPi5 클러스터 (선택) | 온디바이스 가상 피팅(Mobile-VTON). 별도 venv — 보드 `transformers`와 버전 충돌하므로 분리 |

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
gdown 1oVkg5RhHaNFbN4d2zy7_Ue4gAEYLuCb4 -O data.zip
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
├── clip_image_encoder.onnx             (~310MB)
├── clip_preprocessor/visual_projection.npy
└── ko_sroberta/model.onnx + tokenizer  (~460MB)
```

> **참고**: `detector.py`가 MediaPipe Pose로 교체되어 `yolov8n.onnx`는 더 이상 필요하지 않습니다.

### [로컬] Step 2: 무신사 스냅 데이터 변환

크롤러가 `musinsa_out/result.json`과 이미지를 생성한 이후 실행합니다.

```bash
# snap 기반 크롤링 (musinsa_out/result.json + 이미지 생성)
python crawl_musinsa.py
python scripts/convert_musinsa_out.py
```

생성 파일: `data/musinsa_db/{tops,bottoms,shoes}/*.jpg|png` + `metadata.json`

> **데이터 형식**: `musinsa_out/result.json` — 남녀 스냅이 한 파일에 통합되어 있으며
> 각 스냅 레코드가 `gender`("male"/"female") 필드를 가집니다(snap_id 기반 코디 세트).
> `scripts/convert_musinsa_out.py`가 이를 읽어 `gender`를 "남"/"여"로 변환하고
> `data/musinsa_db/metadata.json`(상품별 플랫 목록)으로 변환합니다.

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

**가상 피팅(Virtual Try-On) — 백엔드 선택:**

`VTON_BACKEND` 환경 변수로 가상 피팅 백엔드를 고릅니다 (기본값 `api`).

```bash
# (기본) 클라우드 API 백엔드 — fal-ai
export VTON_BACKEND=api          # 생략 가능 (default)
export FAL_KEY=your_fal_api_key_here

# (선택) 온디바이스 백엔드 — Mobile-VTON, RPi 클러스터에서 실행
export VTON_BACKEND=ondevice     # 설정법은 아래 "9. 온디바이스 가상 피팅" 참고
```

`api` 백엔드는 `FAL_KEY` 미설정 시 `/tryon`이 동작하지 않습니다. 두 경우 모두 추천
파이프라인(YOLO·CLIP·FAISS)은 정상 동작합니다.

카메라 인덱스를 변경하려면 `src/camera.py`의 `Camera(device_index=0)` 인자를 수정하세요.

---

## 6. 외부 서비스·자원

- **인터넷 필요**: `scripts/setup_models.py` (HuggingFace), `crawl_musinsa.py` (무신사)
- **모델 출처**:
  - MediaPipe Pose: Google (mediapipe 패키지에 번들, 별도 다운로드 불필요)
  - CLIP ViT-B/32: `openai/clip-vit-base-patch32` (HuggingFace)
  - ko-sroberta: `jhgan/ko-sroberta-multitask` (HuggingFace)
- **앱 실행 중 (보드)**:
  - AI 추론 파이프라인(MediaPipe·CLIP·FAISS·ko-sroberta): 외부 API 호출 없음, 완전 On-Device
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

## 9. 온디바이스 가상 피팅 (선택) — Mobile-VTON 클러스터

`VTON_BACKEND=ondevice`이면 가상 피팅을 **클라우드 없이 RPi5 클러스터에서** 수행합니다
(fal-ai 미사용). 코드는 `src/ondevice_vton/`에 벤더링되어 있으며, 메인 디노이저를
**spatial(H-band) 병렬화**로 여러 Pi에 분산합니다.

> ⚠️ **클러스터 필수**: full 해상도(1024×768)는 단일 Pi에서 OOM이라, **2-Pi 이상(권장 4-Pi)**
> 에서만 동작합니다. 성능 측정·튜닝 내역은 `src/ondevice_vton/OPTIMIZATION_HISTORY.md` 참고.

### 9-1. 클러스터 구성 (4대 RPi5)

상세 체크리스트는 `src/ondevice_vton/parallel/PI_SETUP.md`. 요약:

- 4대 모두 `192.168.100.0/24` eth0 (`rank i → 192.168.100.(i+1)`), 3대 이상은 **GbE 스위치** 필요
  ```bash
  sudo ip addr add 192.168.100.<k>/24 dev eth0   # 재부팅마다 재설정 (영구 아님)
  ```
- rank0(.1)에서 `.2/.3/.4`로 **passwordless SSH**: `ssh-copy-id willtek@192.168.100.<k>`
- **4대 모두 동일 경로에 본 레포 체크아웃** (peer 실행 경로가 rank0와 같아야 함)

### 9-2. vton 전용 venv (각 Pi, 최초 1회)

보드 앱 venv(`env/`)와 **별도**입니다 (`transformers` 버전 충돌 회피).
```bash
cd src/ondevice_vton
bash setup_env.sh        # .venv 생성 + requirements_vton.txt 설치 (torch 2.12.1+cpu 등)
```

### 9-3. Mobile-VTON 체크포인트 (~3.5GB)

`download_ckpt.py`가 HuggingFace(`FlashStight/Mobile-VTON`)에서 받아
`src/ondevice_vton/checkpoint/`에 넣습니다 (gitignore). **모든 Pi가 동일 경로**에 가져야 합니다.
```bash
# 방법 A) 각 Pi에서 HuggingFace로 직접
cd src/ondevice_vton
.venv/bin/python download_ckpt.py        # → src/ondevice_vton/checkpoint/

# 방법 B) rank0에 한 번 받고 LAN(GbE)으로 peer에 직송 (WAN 느릴 때 권장)
for ip in 192.168.100.2 192.168.100.3 192.168.100.4; do
  rsync -a src/ondevice_vton/checkpoint/ \
    willtek@$ip:$(pwd)/src/ondevice_vton/checkpoint/
done
```
받은 후 구조: `checkpoint/{denoiser, denoiser_garment, vae, vae_decoder, text_encoder(_2),
tokenizer(_2), image_encoder}`. 경로를 바꾸려면 `VTON_CHECKPOINT_PATH`로 오버라이드
(기본 `src/ondevice_vton/checkpoint`).

### 9-4. 활성화 + 실행 (rank0 = 앱 구동 보드)

```bash
export VTON_BACKEND=ondevice
# 필요 시 오버라이드 (기본값은 4-Pi 표준):
#   VTON_PEERS="192.168.100.2 192.168.100.3 192.168.100.4"
#   VTON_PEER_DIR=<peer의 src/ondevice_vton 절대경로>   # 기본=rank0와 동일 경로
#   VTON_CHECKPOINT_PATH=<checkpoint 경로>   VTON_STEPS=6
#   VTON_PYTHON / VTON_PEER_PYTHON=<vton .venv의 python>  # 기본 .venv/bin/python
python src/app.py
```
이후 `/tryon`(포즈 트리거 또는 버튼) 호출 시, fal-ai 대신 클러스터에서 추론합니다
(상의→하의 순차 합성, 6-step 기준 이미지당 수 분 소요).

### 9-5. 앱 없이 단독 검증 (런처 직접)

```bash
cd src/ondevice_vton
# checkpoint는 기본 경로(src/ondevice_vton/checkpoint) 사용
bash parallel/run_sp_multi.sh 6 _vton_run/output _vton_run/single_data \
  "192.168.100.2 192.168.100.3 192.168.100.4"
# single_data는 person/cloth 페어 + test_pairs.txt + image_descriptions.txt 필요
```

---

## 테스트 실행

```bash
# 보드에서 (가상환경 활성화 후)
source env/bin/activate
pytest tests/ -v --tb=short 2>&1 | tee test-results/pytest_log.txt

# 모델·데이터 없이도 mock 기반 테스트는 동작함 (skipif 처리)
```
