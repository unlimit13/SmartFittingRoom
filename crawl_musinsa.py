"""
무신사 스냅 - 남자 코디 태그상품(상의/하의/신발) 크롤러 + 누끼컷 선별
======================================================================
[확정/검증 완료]
  - 스냅 상세 : GET https://content.musinsa.com/api2/content/snap/v1/snaps/{snap_id}
      설명글 detail.content | 해시태그 tags[].name | 성별 model.gender
      태그상품 goods[].goodsNo | 착장이미지 medias[].path
  - 상품 카테고리 : https://www.musinsa.com/products/{goodsNo} 의 og:description
      "제품분류 :<대분류> > <소분류>"  (정규식 파싱 검증됨)
  - 누끼 1차 : 테두리 균일도 기반 점수(NUKKI_MIN_SCORE↑). 각 상품의 앞
      NUKKI_FIRST_N개 후보 중 통과분을 모두 저장 (상품당 여러 장 가능).
  - 누끼 2차 : YOLO(person)로 사람 검출된 이미지를 제거(RUN_YOLO).
  - 이미지 수집 : 상품 페이지 HTML(__NEXT_DATA__ 포함)에서 해당 goodsNo 의
      image.msscdn.net 이미지를 정규식으로 수집 (별도 JSON API 불필요).

[출력]
  - 누끼 이미지 : musinsa_out/nukki/{상의|하의|신발}/{snap}_{goods}_{i}.jpg
  - 결과        : musinsa_out/result.json  (이것만 저장. per-set json 없음)
  - 슬롯 MIN_SLOTS개 미만 코디는 저장/유지하지 않음.

[실행 전 확인 — 선택]
  - GOODS_API : 보통 비워두면 됨(""). 상품 페이지 HTML에 이미지가 부족한 브랜드만
      goods.musinsa.com/api2/goods/... URL을 넣어 보강.
  - FEED_URL : 피드 목록 URL (추정값). 비거나 에러면 피드 cURL로 교체.

의존성: requests pillow numpy tqdm ultralytics
인증: Bearer null 로 동작. Cloudflare 차단 시 COOKIES 에 브라우저 쿠키 복사.
"""

import re
import io
import json
import time
import html
import requests
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ----------------------------------------------------------------------
# 설정
# ----------------------------------------------------------------------
PROFILE_ID = "1234627203913810064"
PAGE_SIZE = 36
MAX_SNAPS = 20000               # 최근 순으로 최대 수집 개수 (피드는 최신순 반환 가정)
REQUEST_DELAY = 0.3             # 워커 1개 기준 요청 간 딜레이(초). 병렬이라 낮춤.
WORKERS = 8                     # 동시 처리 프로세스 수. 차단되면 줄이세요.
MIN_SLOTS = 2                   # 상의/하의/신발 중 최소 충족 슬롯 수(미만이면 폐기)
SKIP_EXISTING = True            # result.json 에 이미 있는 스냅은 건너뜀(리줌)
DEBUG_IMG = False               # True 면 이미지 스킵 사유/후보 점수를 로그로 출력
NUKKI_MIN_SCORE = 0.40          # 이 점수 이상이면 저장(첫 N개 중 통과분 전부)
NUKKI_FIRST_N = 4               # 각 상품 후보 중 앞에서 N개까지만 누끼 검사 대상
SAVE_NUKKI = True
FLUSH_EVERY = 50                # N건 처리마다 result.json 중간 저장(중단 대비)

# --- YOLO 후처리: 사람이 검출된 이미지 제거 ---
RUN_YOLO = True
YOLO_MODEL = "yolov8n.pt"       # 미설치 시 최초 실행에 자동 다운로드
YOLO_PERSON_CONF = 0.40         # 이 신뢰도 이상 'person' 검출되면 사람컷으로 보고 제거
YOLO_BATCH = 16

OUT_DIR = Path("musinsa_out")
IMG_DIR = OUT_DIR / "nukki"     # 하위에 상의/하의/신발 폴더 자동 생성
RESULT_JSON = OUT_DIR / "result.json"

USE_GENDER_FIELD = True
MENS_HASHTAGS = {"남자코디", "남성코디", "맨즈룩", "남친룩", "남자데일리룩"}

# 대분류 -> 슬롯. 상의/하의/신발만 통과 (그 외 None = 제외)
MAIN_TO_SLOT = {"상의": "상의", "바지": "하의", "스커트": "하의", "신발": "신발"}

BASE_HEADERS = {
    "accept": "application/json",
    "accept-language": "ko,en-US;q=0.9,en;q=0.8",
    "authorization": "Bearer null",
    "origin": "https://www.musinsa.com",
    "referer": "https://www.musinsa.com/",
    "user-agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"),
}
COOKIES = {
    # "cf_clearance": "...",
    # "__cf_bm": "...",
}

session = requests.Session()
session.headers.update(BASE_HEADERS)
if COOKIES:
    session.cookies.update(COOKIES)

FEED_URL = "https://content.musinsa.com/api2/content/snap/v1/snaps"          # ⚠️ 추정
SNAP_DETAIL_URL = "https://content.musinsa.com/api2/content/snap/v1/snaps/{snap_id}"
PRODUCT_URL = "https://www.musinsa.com/products/{goods_no}"
GOODS_API = ""   # (선택) 이미지가 더 필요하면 goods.musinsa.com/api2/goods/... URL 넣기. 비우면 미사용.

MAX_CANDIDATES = 8        # 상품당 누끼 후보로 받아볼 최대 이미지 수(다운로드 절감)
SCORE_RESIZE_W = 400      # 누끼 점수용 다운로드 해상도(?w=) - 가볍게

# image.msscdn.net 의 jpg/png/webp URL (프로토콜 생략형 포함)
IMG_RE = re.compile(r'(?:https?:)?//image\.msscdn\.net/[^\s"\'\\)]+?\.(?:jpg|jpeg|png|webp)', re.I)

_goods_cache: dict[str, dict | None] = {}

# ----------------------------------------------------------------------
# 누끼 선별
# ----------------------------------------------------------------------
def nukki_score(img: Image.Image) -> float:
    """누끼(균일/단색 배경, 복잡한 장면 아님) 가능성 0~1.
    핵심 신호는 '테두리가 균일한가'(=배경이 복잡하지 않음), 밝기는 보조."""
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w, _ = a.shape
    m = max(1, int(min(h, w) * 0.08))
    border = np.concatenate([
        a[:m].reshape(-1, 3), a[-m:].reshape(-1, 3),
        a[:, :m].reshape(-1, 3), a[:, -m:].reshape(-1, 3),
    ], axis=0)
    brightness = border.mean() / 255.0
    std = border.std(axis=0).mean()
    uniformity = max(0.0, 1.0 - std / 40.0)              # 균일할수록 1 (복잡배경=0)
    bright = max(0.0, min(1.0, (brightness - 0.60) / 0.40))  # 0.6↓ 0점, 1.0 만점
    return float(0.65 * uniformity + 0.35 * bright)


IMG_HEADERS = {  # 이미지 요청 전용 헤더 (CDN 협상/핫링크 대응)
    "accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "referer": "https://www.musinsa.com/",
}


def _download_image(url: str) -> bytes | None:
    """이미지 1장 다운로드. ?w= 우선, 실패 시 원본 URL로 폴백. 2회 재시도."""
    sized = url + (("&" if "?" in url else "?") + f"w={SCORE_RESIZE_W}")
    for candidate in (sized, url):                 # 리사이즈 실패하면 원본 시도
        for attempt in range(2):                   # 일시적 오류 재시도
            try:
                rr = session.get(candidate, headers=IMG_HEADERS, timeout=20)
                if rr.status_code != 200:
                    break                          # 4xx/5xx면 폴백 URL로
                if not rr.headers.get("content-type", "").startswith("image"):
                    break                          # 이미지가 아니면 폴백
                return rr.content
            except Exception:
                time.sleep(0.4)
    return None


def collect_nukki(urls: list[str], debug: bool = False) -> list[dict]:
    """앞에서 NUKKI_FIRST_N개 후보를 검사해, 점수 NUKKI_MIN_SCORE 이상인 것을
    전부 반환. [{url, score, content}, ...]. (YOLO 후처리에서 사람컷 제거)"""
    passed = []
    for u in urls[:NUKKI_FIRST_N]:
        content = _download_image(u)
        if content is None:
            if debug:
                tqdm.write(f"  [img] 다운로드 실패: {u}")
            continue
        try:
            s = nukki_score(Image.open(io.BytesIO(content)))
        except Exception as e:
            if debug:
                tqdm.write(f"  [img] 디코딩 실패({type(e).__name__}): {u}")
            continue
        if debug:
            tqdm.write(f"  [img] score {s:.2f}  {u}")
        if s >= NUKKI_MIN_SCORE:
            passed.append({"url": u, "score": round(s, 3), "content": content})
        time.sleep(0.2)
    return passed


# ----------------------------------------------------------------------
# 상품 정보 (카테고리 + 후보 이미지)
# ----------------------------------------------------------------------
_META = {
    "img": re.compile(r'property=["\']og:image["\'][^>]+content=["\']([^"\']+)', re.I),
    "title": re.compile(r'property=["\']og:title["\'][^>]+content=["\']([^"\']+)', re.I),
    "desc": re.compile(r'(?:property=["\']og:description["\']|name=["\']description["\'])[^>]+content=["\']([^"\']+)', re.I),
}


def _collect_msscdn_images(text: str, goods_no: str) -> list[str]:
    """HTML/JSON 문자열에서 해당 goodsNo 의 msscdn 이미지 URL 수집.
    대표컷(goods_img) 우선, 사이즈 변형 중복 제거."""
    out, seen = [], set()
    for u in IMG_RE.findall(text):
        if not (f"/{goods_no}/" in u or f"{goods_no}_" in u):
            continue                                  # 이 상품 이미지만
        u = "https:" + u if u.startswith("//") else u
        norm = re.sub(r'_(?:\d+|big|small|thumb|org)(?=\.\w+$)', '', u.split("?")[0])
        if norm in seen:
            continue
        seen.add(norm)
        out.append(u)
    # 대표컷(goods_img) 을 상세컷(prd_img)보다 앞에 — 누끼일 확률 높음
    out.sort(key=lambda u: 0 if "goods_img" in u else 1)
    return out


def fetch_goods(goods_no: str) -> dict | None:
    if goods_no in _goods_cache:
        return _goods_cache[goods_no]

    # 상품 페이지 HTML 한 번으로 카테고리 + 이미지 후보 모두 확보
    try:
        h = session.get(PRODUCT_URL.format(goods_no=goods_no),
                        headers={"accept": "text/html"}, timeout=15).text
    except Exception:
        _goods_cache[goods_no] = {"slot": None}
        return _goods_cache[goods_no]

    category = None
    dm = _META["desc"].search(h)
    if dm:
        cm = re.search(r"제품분류\s*:\s*([^>]+?)\s*>", html.unescape(dm.group(1)))
        if cm:
            category = cm.group(1).strip()
    slot = MAIN_TO_SLOT.get(category)
    if slot is None:                                  # 상의/하의/신발 아니면 종료
        _goods_cache[goods_no] = {"slot": None}
        return _goods_cache[goods_no]

    tm = _META["title"].search(h)
    name = html.unescape(tm.group(1)).split(" - ")[0] if tm else None

    # 1차: 상품 페이지 HTML(__NEXT_DATA__ 포함)에서 이미지 수집
    candidates = _collect_msscdn_images(h, goods_no)

    # 2차(선택): GOODS_API 가 설정돼 있으면 추가 수집해 병합
    if GOODS_API:
        try:
            gh = session.get(GOODS_API.format(goods_no=goods_no), timeout=15).text
            for u in _collect_msscdn_images(gh, goods_no):
                if u not in candidates:
                    candidates.append(u)
        except Exception:
            pass

    # 폴백: 아무것도 없으면 og:image 한 장
    if not candidates:
        og = _META["img"].search(h)
        if og:
            candidates = [html.unescape(og.group(1))]

    candidates = candidates[:MAX_CANDIDATES]
    info = {"goodsNo": goods_no, "category_main": category, "slot": slot,
            "name": name, "candidates": candidates}
    _goods_cache[goods_no] = info
    return info


# ----------------------------------------------------------------------
# 스냅
# ----------------------------------------------------------------------
def fetch_feed(page: int) -> dict:
    params = {"profileIds": PROFILE_ID,
              "displayStatuses": "DISPLAY,PROFILE_ONLY,PDP_ONLY",
              "formatTypes": "POST,SHORTS", "page": page, "size": PAGE_SIZE}
    r = session.get(FEED_URL, params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def extract_snap_list(body: dict) -> list[dict]:
    data = body.get("data", body)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("list", "snaps", "content", "items", "results"):
            if isinstance(data.get(k), list):
                return data[k]
    return []


def fetch_snap_detail(snap_id: str) -> dict:
    r = session.get(SNAP_DETAIL_URL.format(snap_id=snap_id), timeout=15)
    r.raise_for_status()
    return r.json()["data"]


def extract_hashtags(snap): return [t.get("name", "") for t in snap.get("tags", [])]
def get_description(snap): return (snap.get("detail") or {}).get("content", "").strip()


def is_mens(snap) -> bool:
    if USE_GENDER_FIELD and (snap.get("model") or {}).get("gender") == "MEN":
        return True
    return bool({t.replace(" ", "") for t in extract_hashtags(snap)} & MENS_HASHTAGS)


def process_snap(snap: dict) -> dict | None:
    if not is_mens(snap):
        return None
    desc = get_description(snap)
    snap_id = snap.get("id")

    # 1) 후보 이미지 수집 (앞 N개 중 통과분 전부, 저장은 검증 통과 후로 지연)
    pending = []   # (slot, goodsNo, category_main, name, url, score, content)
    for g in snap.get("goods", []):
        gno = g.get("goodsNo")
        if not gno:
            continue
        info = fetch_goods(gno)
        time.sleep(REQUEST_DELAY)
        if not info or info.get("slot") is None:
            continue
        if not info.get("candidates"):
            if DEBUG_IMG:
                tqdm.write(f"  [goods {gno}] 후보 이미지 없음(필터 제외 가능)")
            continue
        hits = collect_nukki(info["candidates"], debug=DEBUG_IMG)
        if not hits:
            if DEBUG_IMG:
                tqdm.write(f"  [goods {gno}] 통과 누끼 없음")
            continue
        for hit in hits:
            pending.append((info["slot"], gno, info["category_main"], info["name"],
                            hit["url"], hit["score"], hit["content"]))

    # 2) 슬롯 검증: 상의/하의/신발 중 채워진 슬롯이 MIN_SLOTS 미만이면 폐기(저장 안 함)
    filled = {p[0] for p in pending} & {"상의", "하의", "신발"}
    if len(filled) < MIN_SLOTS:
        return None

    # 3) 통과 코디만 이미지 저장 (상품당 여러 장 가능 → 파일명에 인덱스)
    by_slot = {"상의": [], "하의": [], "신발": []}
    idx_per_goods: dict[str, int] = {}
    for slot, gno, cat, name, url, score, content in pending:
        rel_path = None
        if SAVE_NUKKI and content:
            slot_dir = IMG_DIR / slot
            slot_dir.mkdir(parents=True, exist_ok=True)
            i = idx_per_goods.get(gno, 0)
            idx_per_goods[gno] = i + 1
            ext = url.split("?")[0].rsplit(".", 1)[-1][:4]
            dst = slot_dir / f"{snap_id}_{gno}_{i}.{ext}"   # 코디·상품·장 단위 고유명
            dst.write_bytes(content)
            rel_path = str(dst.relative_to(OUT_DIR))
        by_slot.setdefault(slot, []).append({
            "goodsNo": gno, "category_main": cat, "name": name,
            "nukki_image": url, "nukki_score": score, "saved_path": rel_path})

    return {"snap_id": snap_id, "description": desc,
            "hashtags": extract_hashtags(snap), "items_by_slot": by_slot}


# ----------------------------------------------------------------------
# 워커 / 정리
# ----------------------------------------------------------------------
def count_filled(coordi: dict) -> int:
    bs = coordi.get("items_by_slot", {})
    return sum(1 for s in ("상의", "하의", "신발") if bs.get(s))


def _process_one(sid: str):
    """프로세스 워커: 스냅 1건 처리. (coordi dict | None | {'_error':...})"""
    try:
        return process_snap(fetch_snap_detail(sid))
    except Exception as e:
        return {"_error": f"{sid}: {e}"}


def yolo_filter(results: list[dict]) -> list[dict]:
    """저장된 누끼 이미지에 YOLO를 돌려 사람(person)이 검출된 이미지를 삭제하고,
    그 항목을 result 에서 제거. 제거 후 슬롯 MIN_SLOTS 미만이 된 코디는 통째로 폐기."""
    try:
        from ultralytics import YOLO
    except Exception:
        print("[YOLO] ultralytics 미설치 → 건너뜀.  pip install ultralytics")
        return results

    model = YOLO(YOLO_MODEL)

    # (코디idx, slot, item) 평면화하여 경로 목록 구성
    flat = []
    for ci, r in enumerate(results):
        for slot, lst in r.get("items_by_slot", {}).items():
            for it in lst:
                sp = it.get("saved_path")
                if sp and (OUT_DIR / sp).exists():
                    flat.append((ci, slot, it, OUT_DIR / sp))

    to_remove = set()   # id(item) 기준 제거 표시
    for i in tqdm(range(0, len(flat), YOLO_BATCH), desc="YOLO 사람 제거", unit="batch"):
        chunk = flat[i:i + YOLO_BATCH]
        preds = model([str(p) for *_, p in chunk], verbose=False)
        for (ci, slot, it, path), pred in zip(chunk, preds):
            has_person = any(int(b.cls) == 0 and float(b.conf) >= YOLO_PERSON_CONF
                             for b in pred.boxes)
            if has_person:
                try:
                    path.unlink()
                except Exception:
                    pass
                to_remove.add(id(it))

    # result 재구성 + 슬롯 부족 코디 폐기(남은 이미지도 삭제)
    cleaned = []
    dropped_imgs = 0
    for r in results:
        bs = {}
        for slot, lst in r.get("items_by_slot", {}).items():
            kept = [it for it in lst if id(it) not in to_remove]
            if kept:
                bs[slot] = kept
        r2 = {**r, "items_by_slot": bs}
        if count_filled(r2) >= MIN_SLOTS:
            cleaned.append(r2)
        else:
            for lst in bs.values():            # 폐기 코디의 잔여 이미지 삭제
                for it in lst:
                    sp = it.get("saved_path")
                    if sp and (OUT_DIR / sp).exists():
                        (OUT_DIR / sp).unlink(); dropped_imgs += 1
    print(f"[YOLO] 사람컷 {len(to_remove)}장 제거, 슬롯부족으로 폐기된 잔여 {dropped_imgs}장")
    return cleaned


def _flush(results):
    RESULT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


# ----------------------------------------------------------------------
# 메인
# ----------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(exist_ok=True)

    # 리줌: 기존 result.json 로드 → 처리 완료 스냅 건너뜀
    results, done = [], set()
    if SKIP_EXISTING and RESULT_JSON.exists():
        try:
            results = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
            done = {r.get("snap_id") for r in results}
            print(f"리줌: 기존 {len(results)}건 로드")
        except Exception:
            results, done = [], set()

    # 1단계: 스냅 ID 수집 (최신순, 최대 MAX_SNAPS개)
    snap_ids, page = [], 1
    with tqdm(total=MAX_SNAPS, desc="스냅 목록 수집", unit="snap") as pbar:
        while len(snap_ids) < MAX_SNAPS:
            stubs = extract_snap_list(fetch_feed(page))
            if not stubs:
                break
            before = len(snap_ids)
            snap_ids.extend(s.get("id") for s in stubs if s.get("id"))
            if len(snap_ids) >= MAX_SNAPS:
                snap_ids = snap_ids[:MAX_SNAPS]
            pbar.update(len(snap_ids) - before)
            page += 1
            time.sleep(REQUEST_DELAY)

    pending = [s for s in snap_ids if s not in done]

    # 2단계: 멀티프로세싱 코디 처리 (진행바 + 주기적 result.json 저장)
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(_process_one, sid) for sid in pending]
        pbar = tqdm(as_completed(futures), total=len(futures), desc="코디 처리", unit="snap")
        for n, fut in enumerate(pbar, 1):
            r = fut.result()
            if r and "_error" in r:
                tqdm.write(f"[skip] {r['_error']}")
            elif r:
                results.append(r)
                pbar.set_postfix(saved=len(results))
            if n % FLUSH_EVERY == 0:
                _flush(results)
    _flush(results)

    # 3단계: YOLO 후처리(사람컷 제거)
    if RUN_YOLO:
        results = yolo_filter(results)
        _flush(results)

    print(f"\n완료: 코디 {len(results)}건")
    print(f"  - 누끼 이미지 : {IMG_DIR}/상의·하의·신발/")
    print(f"  - 결과        : {RESULT_JSON}")


if __name__ == "__main__":
    main()