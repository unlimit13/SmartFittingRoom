"""
Musinsa product crawler.
Usage: python scripts/crawl_musinsa.py --category tops --count 250
Categories: tops, bottoms, shoes, outer

Saves images to data/musinsa_db/<category>/ and appends to data/musinsa_db/metadata.json.
"""
import argparse
import json
import os
import time
import random

import requests
from PIL import Image
from io import BytesIO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(ROOT, "data", "musinsa_db")
META_PATH = os.path.join(DB_DIR, "metadata.json")

CATEGORY_MAP = {
    "tops": "001",    # 상의
    "bottoms": "003", # 하의
    "shoes": "103",   # 신발
    "outer": "002",   # 아우터
}

STYLE_TAGS = {
    "tops": ["캐주얼", "스트리트", "베이직", "오버핏", "슬림핏", "반팔", "긴팔", "셔츠", "티셔츠", "니트"],
    "bottoms": ["슬랙스", "청바지", "반바지", "조거팬츠", "와이드", "스키니", "데님", "카고"],
    "shoes": ["스니커즈", "로퍼", "샌들", "부츠", "러닝화", "슬리퍼", "하이탑", "어그"],
    "outer": ["자켓", "코트", "후드집업", "바람막이", "블레이저", "패딩", "트렌치코트", "가디건"],
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.musinsa.com/",
    "Accept": "application/json, text/plain, */*",
}

GOODS_API = "https://api.musinsa.com/api2/dp/v1/plp/goods"
PAGE_SIZE = 100


def load_metadata():
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_metadata(items):
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def fetch_goods_page(category_code, page, sort_code="emt_high_click"):
    params = {
        "category": category_code,
        "caller": "CATEGORY",
        "gf": "A",
        "page": page,
        "pageSize": PAGE_SIZE,
        "sortCode": sort_code,
    }
    resp = requests.get(GOODS_API, headers=HEADERS, params=params, timeout=10)
    resp.raise_for_status()
    body = resp.json()
    data = body["data"]
    return data["list"], data["pagination"]["hasNext"]


def download_image(img_url, save_path):
    resp = requests.get(img_url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert("RGB")
    img.save(save_path, "JPEG", quality=85)


def crawl(category, count):
    cat_code = CATEGORY_MAP[category]
    save_dir = os.path.join(DB_DIR, category)
    os.makedirs(save_dir, exist_ok=True)

    metadata = load_metadata()
    existing_ids = {item["product_id"] for item in metadata}

    collected = 0
    page = 1

    while collected < count:
        print(f"[{category}] page {page} ...")
        try:
            goods, has_next = fetch_goods_page(cat_code, page)
        except Exception as e:
            print(f"  fetch error: {e}")
            break

        if not goods:
            print("  no more products")
            break

        for item in goods:
            if collected >= count:
                break

            pid = str(item["goodsNo"])
            product_id = f"musinsa_{pid}"
            if product_id in existing_ids:
                continue

            img_url = item.get("thumbnail")
            if not img_url:
                continue

            try:
                filename = f"{product_id}.jpg"
                img_path = os.path.join(save_dir, filename)
                download_image(img_url, img_path)

                name = item.get("goodsName", f"상품_{pid}")
                tags = random.sample(STYLE_TAGS[category], k=min(4, len(STYLE_TAGS[category])))
                style_text = f"{name}, {', '.join(tags)}"

                record = {
                    "product_id": product_id,
                    "category": category,
                    "url": item.get("goodsLinkUrl", f"https://www.musinsa.com/products/{pid}"),
                    "image_path": f"{category}/{filename}",
                    "name": name,
                    "style_text": style_text,
                    "dominant_color": "#000000",  # updated by build_image_index.py
                }
                metadata.append(record)
                existing_ids.add(product_id)
                collected += 1
                print(f"  [{collected}/{count}] {product_id}: {name[:30]}")

                time.sleep(random.uniform(0.1, 0.3))
            except Exception as e:
                print(f"  error on {pid}: {e}")

        if not has_next:
            print("  no more pages")
            break

        page += 1
        time.sleep(random.uniform(0.3, 0.6))

    save_metadata(metadata)
    print(f"[{category}] done. collected {collected} items. metadata saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", required=True, choices=list(CATEGORY_MAP))
    parser.add_argument("--count", type=int, default=250)
    args = parser.parse_args()
    crawl(args.category, args.count)
