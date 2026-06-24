"""
Convert musinsa_out/musinsa_db/result.json to data/musinsa_db/ format.
Snap-based outfit data → flat product catalog compatible with build_image_index.py.
"""
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "musinsa_out", "musinsa_db")
DST_DIR = os.path.join(ROOT, "data", "musinsa_db")
RESULT_JSON = os.path.join(SRC_DIR, "result.json")

SLOT_TO_CATEGORY = {"상의": "tops", "하의": "bottoms", "신발": "shoes"}
MAX_HASHTAGS = 8


def main():
    with open(RESULT_JSON, encoding="utf-8") as f:
        snaps = json.load(f)

    os.makedirs(DST_DIR, exist_ok=True)
    for cat in ("tops", "bottoms", "shoes"):
        os.makedirs(os.path.join(DST_DIR, cat), exist_ok=True)

    seen = set()
    metadata = []

    for snap in snaps:
        hashtags = snap.get("hashtags", [])
        style_suffix = ", ".join(hashtags[:MAX_HASHTAGS])

        for slot, items in snap["items_by_slot"].items():
            category = SLOT_TO_CATEGORY.get(slot)
            if category is None:
                continue

            for item in items:
                goods_no = item["goodsNo"]
                if goods_no in seen:
                    continue
                seen.add(goods_no)

                ext = os.path.splitext(item["saved_path"])[1]  # .jpg or .png
                filename = f"musinsa_{goods_no}{ext}"
                src_path = os.path.join(SRC_DIR, item["saved_path"])
                dst_path = os.path.join(DST_DIR, category, filename)
                shutil.copy2(src_path, dst_path)

                metadata.append({
                    "product_id": f"musinsa_{goods_no}",
                    "category": category,
                    "url": f"https://www.musinsa.com/products/{goods_no}",
                    "image_path": f"{category}/{filename}",
                    "name": item["name"],
                    "style_text": f"{item['name']}, {style_suffix}" if style_suffix else item["name"],
                })

    meta_path = os.path.join(DST_DIR, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"변환 완료: {len(metadata)}개 상품 → {meta_path}")
    by_cat: dict[str, int] = {}
    for m in metadata:
        by_cat[m["category"]] = by_cat.get(m["category"], 0) + 1
    for cat, cnt in sorted(by_cat.items()):
        print(f"  {cat}: {cnt}개")


if __name__ == "__main__":
    main()
