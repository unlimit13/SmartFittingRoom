"""
무신사 스냅 - 남자 코디 상의/하의/신발 누끼 이미지 수집기
======================================================================
파이프라인:
  1) 스냅 피드(최신순, 최대 MAX_SNAPS) → 스냅 id 수집
  2) 스냅 상세 → 남자 코디만(model.gender=="MEN", 보조: 해시태그) + goods[].goodsNo
  3) 상품 페이지 → __NEXT_DATA__ 에서 카테고리(상의/하의/신발) + 갤러리 이미지 전부
     (대표컷 thumbnailImageUrl + 상세컷 goodsImages[]) — 개수 제한 없이 코디 단위로 수집
  4) 다운로드 후 [워커] 테두리 균일도(누끼) 사전필터로 통과분만 저장
  5) [메인] 저장 이미지에 YOLO(person) 배치 → 사람 검출 이미지 제거
     → 최종 = 사람없음 ∩ 누끼. (4·5는 교집합이라 순서 무관, 효율 위해 분리)
  6) 슬롯 확정 : 슬롯마다 (누끼∩사람없음) 최상위 1장, 없으면 폴백(최상위 1장).
  * 상의/하의/신발 3슬롯이 모두 있는 코디만 채택(MIN_SLOTS=3) → 코디당 정확히 3장.

확정 소스(실측):
  - 스냅 상세 : content.musinsa.com/api2/content/snap/v1/snaps/{id}  (Bearer null 동작)
  - 상품 갤러리 : www.musinsa.com/products/{goodsNo} 의 <script id="__NEXT_DATA__">
        props.pageProps.meta.data.{category, thumbnailImageUrl, goodsImages[]}

확인 필요:
  - FEED_URL : 스냅 목록 API(추정). 비거나 에러면 피드 cURL로 교체.

의존성: requests pillow numpy tqdm ultralytics
"""

import re
import io
import json
import time
import requests
import numpy as np
from PIL import Image
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# ----------------------------------------------------------------------
# 설정
# ----------------------------------------------------------------------
PROFILE_ID = "1234627203913810064"
PAGE_SIZE = 36
MAX_SNAPS = 60000
REQUEST_DELAY = 0.3            # 워커당 요청 간 딜레이(초)
WORKERS = 8                   # 동시 처리 프로세스 (차단 시 줄이기)
MIN_SLOTS = 3                 # 상의/하의/신발 3슬롯 모두 있어야 코디 채택(각 1장 → 코디당 3장)
NUKKI_MIN_SCORE = 0.50        # 누끼 임계(빡셈): 밝고 균일한 단색 배경만 통과
NUKKI_EDGE_RATIO = 0.02       # 테두리 띠 두께(짧은 변 대비 비율)
OUTER_AS_TOP = True           # 아우터를 상의로 인정
SKIP_EXISTING = True          # result.json 에 있는 스냅 건너뜀(리줌)
FLUSH_EVERY = 50

RUN_YOLO = True
YOLO_MODEL = "yolov8n.pt"     # 최초 실행 시 자동 다운로드
YOLO_PERSON_CONF = 0.40
YOLO_BATCH = 16

OUT_DIR = Path("musinsa_out")
IMG_DIR = OUT_DIR / "images"  # 하위 상의/하의/신발
RESULT_JSON = OUT_DIR / "result.json"

SLOT_KEYWORDS = [
    ("신발", ["신발", "슈즈", "스니커", "운동화", "부츠", "로퍼", "구두", "샌들", "슬리퍼", "더비", "워커"]),
    ("하의", ["바지", "팬츠", "데님", "슬랙스", "스커트", "반바지", "쇼츠", "숏", "조거", "트레이닝", "레깅스", "하의"]),
    ("상의", ["상의", "티셔츠", "셔츠", "니트", "맨투맨", "후드", "스웨트", "스웨터", "탑", "블라우스", "베스트", "카라"]),
]
if OUTER_AS_TOP:
    SLOT_KEYWORDS.append(("상의", ["아우터", "자켓", "재킷", "코트", "점퍼", "패딩", "가디건", "블루종", "야상", "후리스"]))

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
    # Cloudflare 차단(403/503) 시 브라우저 쿠키 복사:
    # "cf_clearance": "...",
}
IMG_HEADERS = {"accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
               "referer": "https://www.musinsa.com/"}
IMG_BASE = "https://image.msscdn.net"

session = requests.Session()
session.headers.update(BASE_HEADERS)
if COOKIES:
    session.cookies.update(COOKIES)

FEED_URL = "https://content.musinsa.com/api2/content/snap/v1/snaps"          # ⚠️ 추정
SNAP_DETAIL_URL = "https://content.musinsa.com/api2/content/snap/v1/snaps/{snap_id}"
PRODUCT_URL = "https://www.musinsa.com/products/{goods_no}"

_NEXT_RE = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)
_goods_cache: dict[str, dict] = {}


# ----------------------------------------------------------------------
# 분류 / 누끼
# ----------------------------------------------------------------------
def to_slot(category: dict | None) -> str | None:
    if not category:
        return None
    name = (category.get("categoryDepth1Name") or "").replace(" ", "")
    for slot, kws in SLOT_KEYWORDS:
        if any(k in name for k in kws):
            return slot
    return None


def nukki_score(img: Image.Image) -> float:
    """테두리(가장자리 8%)가 균일/밝을수록 1에 가까움 (누끼 가능성)."""
    a = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w, _ = a.shape
    m = max(1, int(min(h, w) * NUKKI_EDGE_RATIO))
    b = np.concatenate([a[:m].reshape(-1, 3), a[-m:].reshape(-1, 3),
                        a[:, :m].reshape(-1, 3), a[:, -m:].reshape(-1, 3)], 0)
    bright = b.mean() / 255.0
    std = b.std(axis=0).mean()
    uni = max(0.0, 1.0 - std / 40.0)
    br = max(0.0, min(1.0, (bright - 0.60) / 0.40))
    return float(0.65 * uni + 0.35 * br)


# ----------------------------------------------------------------------
# 상품 (갤러리 + 카테고리)
# ----------------------------------------------------------------------
def _abs_url(u: str) -> str:
    if u.startswith("//"):
        return "https:" + u
    if u.startswith("/"):
        return IMG_BASE + u
    return u


def fetch_goods(goods_no: str) -> dict:
    if goods_no in _goods_cache:
        return _goods_cache[goods_no]
    info = {"goodsNo": goods_no, "slot": None, "name": None,
            "category_main": None, "images": []}
    try:
        h = session.get(PRODUCT_URL.format(goods_no=goods_no),
                        headers={"accept": "text/html"}, timeout=15).text
        m = _NEXT_RE.search(h)
        if m:
            data = json.loads(m.group(1))["props"]["pageProps"]["meta"]["data"]
            cat = data.get("category") or {}
            info["category_main"] = cat.get("categoryDepth1Name")
            info["slot"] = to_slot(cat)
            info["name"] = data.get("goodsNm")
            urls, seen = [], set()
            for u in ([data.get("thumbnailImageUrl")]
                      + [g.get("imageUrl") for g in data.get("goodsImages", [])]):
                if not u:
                    continue
                au = _abs_url(u)
                if au not in seen:
                    seen.add(au)
                    urls.append(au)
            info["images"] = urls
    except Exception:
        pass
    _goods_cache[goods_no] = info
    return info


def download_image(url: str) -> bytes | None:
    for _ in range(2):
        try:
            rr = session.get(url, headers=IMG_HEADERS, timeout=20)
            if rr.status_code == 200 and rr.headers.get("content-type", "").startswith("image"):
                return rr.content
        except Exception:
            time.sleep(0.4)
    return None


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
    return (snap.get("model") or {}).get("gender") == "MEN"


def process_snap(snap: dict) -> dict | None:
    """남자 코디면 상의/하의/신발 상품의 갤러리를 받아 누끼 통과분만 저장."""
    if not is_mens(snap):
        return {"_drop": "not_mens"}
    snap_id = snap.get("id")

    by_slot = {"상의": [], "하의": [], "신발": []}   # slot -> 갤러리 순서 이미지 후보 목록
    seen_goods = set()
    for g in snap.get("goods", []):
        gno = g.get("goodsNo")
        if not gno or gno in seen_goods:
            continue
        seen_goods.add(gno)
        info = fetch_goods(gno)
        time.sleep(REQUEST_DELAY)
        slot = info.get("slot")
        if slot is None or not info.get("images"):
            continue

        for idx, url in enumerate(info["images"]):
            content = download_image(url)
            if not content:
                continue
            try:
                score = nukki_score(Image.open(io.BytesIO(content)))
            except Exception:
                continue
            is_nukki = score >= NUKKI_MIN_SCORE
            # 누끼면 저장. 누끼가 아니어도 슬롯 첫 이미지면 폴백용으로 1장 저장.
            if not (is_nukki or len(by_slot[slot]) == 0):
                continue
            slot_dir = IMG_DIR / slot
            slot_dir.mkdir(parents=True, exist_ok=True)
            ext = url.split("?")[0].rsplit(".", 1)[-1][:4] or "jpg"
            dst = slot_dir / f"{snap_id}_{gno}_{idx}.{ext}"
            dst.write_bytes(content)
            by_slot[slot].append({
                "goodsNo": gno, "name": info["name"], "category_main": info["category_main"],
                "saved_path": str(dst.relative_to(OUT_DIR)), "source_url": url,
                "nukki_score": round(score, 3), "is_nukki": is_nukki})
            time.sleep(0.15)

    if sum(1 for s in by_slot if by_slot[s]) < MIN_SLOTS:
        for lst in by_slot.values():
            for im in lst:
                p = OUT_DIR / im["saved_path"]
                if p.exists():
                    p.unlink()
        return {"_drop": "below_min_slots"}

    return {"snap_id": snap_id, "description": get_description(snap),
            "hashtags": extract_hashtags(snap), "items_by_slot": by_slot}


def _process_one(sid: str):
    try:
        return process_snap(fetch_snap_detail(sid))
    except Exception as e:
        return {"_error": f"{sid}: {e}"}


def _flush(results):
    RESULT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


# ----------------------------------------------------------------------
# YOLO (사람 검출 이미지 제거) - 메인에서 1회 로드, 배치 처리
# ----------------------------------------------------------------------
def yolo_filter(results: list[dict]) -> list[dict]:
    try:
        from ultralytics import YOLO
    except Exception:
        print("[YOLO] ultralytics 미설치 → 건너뜀.  pip install ultralytics")
        return results
    model = YOLO(YOLO_MODEL)

    flat = []
    for r in results:
        for lst in r.get("items_by_slot", {}).values():
            for im in lst:
                im["is_person"] = False        # 기본값
                p = OUT_DIR / im["saved_path"]
                if p.exists():
                    flat.append((im, p))

    n_person = 0
    for i in tqdm(range(0, len(flat), YOLO_BATCH), desc="YOLO 사람 탐지", unit="batch"):
        chunk = flat[i:i + YOLO_BATCH]
        preds = model([str(p) for _, p in chunk], verbose=False)
        for (im, p), pred in zip(chunk, preds):
            if any(int(b.cls) == 0 and float(b.conf) >= YOLO_PERSON_CONF for b in pred.boxes):
                im["is_person"] = True          # 표시만, 파일 삭제 안 함
                n_person += 1
    print(f"[YOLO] 사람 검출 {n_person}장 표시")
    return results


def finalize_slots(results: list[dict]) -> list[dict]:
    """슬롯마다 1장 확정:
       1순위 = 누끼 ∩ 사람없음 중 최상위, 없으면 폴백 = 제거대상 중 최상위(슬롯 첫 장).
       선택된 1장 외 파일은 모두 삭제."""
    kept_total = fallback_total = trimmed = 0
    out = []
    for r in results:
        reduced = {}
        for slot, lst in r.get("items_by_slot", {}).items():
            if not lst:
                continue
            survivors = [im for im in lst if im.get("is_nukki") and not im.get("is_person")]
            if survivors:
                chosen = survivors[0]            # 갤러리 순서상 최상위 누끼
                chosen["is_fallback"] = False
                kept_total += 1
            else:
                chosen = lst[0]                  # 폴백: 슬롯 최상위 1장
                chosen["is_fallback"] = True
                fallback_total += 1
            for im in lst:                       # 선택 외 전부 삭제
                if im is chosen:
                    continue
                p = OUT_DIR / im["saved_path"]
                if p.exists():
                    p.unlink()
                trimmed += 1
            reduced[slot] = [chosen]
        out.append({**r, "items_by_slot": reduced})
    print(f"[정리] 슬롯 확정: 누끼 {kept_total} / 폴백 {fallback_total}, 잔여 {trimmed}장 삭제")
    return out


def drop_coordis_with_person(results: list[dict]) -> list[dict]:
    """최종 3장 중 사람(is_person)이 하나라도 있으면 코디 전체 제거(사진+result)."""
    out, dropped = [], 0
    for r in results:
        slots = r.get("items_by_slot", {})
        imgs = [im for lst in slots.values() for im in lst]
        has_person = any(im.get("is_person") for im in imgs)
        if has_person:
            for im in imgs:                       # 코디의 사진 전부 삭제
                p = OUT_DIR / im["saved_path"]
                if p.exists():
                    p.unlink()
            dropped += 1
        else:
            out.append(r)
    print(f"[정리] 사람 포함 코디 {dropped}건 제거")
    return out


# ----------------------------------------------------------------------
# 메인
# ----------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(exist_ok=True)
    results, done = [], set()
    if SKIP_EXISTING and RESULT_JSON.exists():
        try:
            results = json.loads(RESULT_JSON.read_text(encoding="utf-8"))
            done = {r.get("snap_id") for r in results}
            print(f"리줌: 기존 {len(results)}건 로드")
        except Exception:
            results, done = [], set()

    # 1) 스냅 id 수집 (피드가 동나거나 에러나면 그때까지 모은 것까지만 사용)
    snap_ids, page, fails = [], 1, 0
    with tqdm(total=MAX_SNAPS, desc="스냅 목록 수집", unit="snap") as pbar:
        while len(snap_ids) < MAX_SNAPS:
            try:
                stubs = extract_snap_list(fetch_feed(page))
            except Exception as e:
                fails += 1
                tqdm.write(f"[feed] page {page} 오류({fails}/3): {e}")
                if fails >= 3:                      # 연속 실패 → 모은 것까지만
                    tqdm.write("[feed] 반복 실패로 목록 수집 종료")
                    break
                time.sleep(2 * fails)               # 백오프 후 같은 페이지 재시도
                continue
            fails = 0
            if not stubs:                           # 더 이상 스냅 없음 → 정상 종료
                break
            before = len(snap_ids)
            snap_ids.extend(s.get("id") for s in stubs if s.get("id"))
            if len(snap_ids) >= MAX_SNAPS:
                snap_ids = snap_ids[:MAX_SNAPS]
            pbar.update(len(snap_ids) - before)
            page += 1
            time.sleep(REQUEST_DELAY)
    print(f"수집된 스냅 id: {len(snap_ids)}개 (목표 {MAX_SNAPS})")

    pending = [s for s in snap_ids if s not in done]

    # 2~4) 멀티프로세싱 수집 + 누끼 사전필터
    funnel = Counter()
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futures = [ex.submit(_process_one, sid) for sid in pending]
        pbar = tqdm(as_completed(futures), total=len(futures), desc="코디 수집", unit="snap")
        for n, fut in enumerate(pbar, 1):
            r = fut.result()
            if r and "_error" in r:
                funnel["error"] += 1
                tqdm.write(f"[skip] {r['_error']}")
            elif r and "_drop" in r:
                funnel[r["_drop"]] += 1
            elif r:
                funnel["collected"] += 1
                results.append(r)
                pbar.set_postfix(collected=len(results))
            if n % FLUSH_EVERY == 0:
                _flush(results)
    _flush(results)
    print("수집 퍼널:", dict(funnel))

    # 5) YOLO 사람 탐지(표시) → 6) 슬롯 확정(누끼 우선, 없으면 폴백 1장)
    if RUN_YOLO:
        results = yolo_filter(results)
    results = finalize_slots(results)
    if RUN_YOLO:
        results = drop_coordis_with_person(results)   # 3장 중 사람 있으면 코디째 제거
    _flush(results)

    total_imgs = sum(len(lst) for r in results
                     for lst in r["items_by_slot"].values())
    print(f"\n완료: 코디 {len(results)}건, 이미지 {total_imgs}장(슬롯당 1장)")
    print(f"  - 이미지 : {IMG_DIR}/상의·하의·신발/")
    print(f"  - 결과   : {RESULT_JSON}")


if __name__ == "__main__":
    main()