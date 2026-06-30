"""
Convert musinsa_out result.json to data/musinsa_db/ format.
Snap-based outfit data → flat product catalog + snap_outfits.json.
Each snap record carries a "gender" field ("male" or "female"); each product
entry gets a corresponding "gender" field ("남" or "여").
"""
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(ROOT, "musinsa_out")
DST_DIR = os.path.join(ROOT, "data", "musinsa_db")

GENDER_LABEL = {"male": "남", "female": "여"}
GENDER_PREFIX = {"male": "men", "female": "women"}

SLOT_TO_CATEGORY = {"상의": "tops", "하의": "bottoms", "신발": "shoes"}
MAX_HASHTAGS = 8


def main():
    os.makedirs(DST_DIR, exist_ok=True)
    for cat in ("tops", "bottoms", "shoes"):
        os.makedirs(os.path.join(DST_DIR, cat), exist_ok=True)

    seen = set()
    metadata = []
    snap_outfits = {}

    result_json = os.path.join(SRC_DIR, "result.json")
    with open(result_json, encoding="utf-8") as f:
        snaps = json.load(f)

    for snap in snaps:
        gender = GENDER_LABEL.get(snap.get("gender"))
        if gender is None:
            continue
        snap_prefix = GENDER_PREFIX[snap["gender"]]
        raw_snap_id = snap["snap_id"]
        snap_key = f"{snap_prefix}_{raw_snap_id}"
        hashtags = snap.get("hashtags", [])
        style_suffix = ", ".join(hashtags[:MAX_HASHTAGS])
        snap_outfits[snap_key] = {"tops": [], "bottoms": [], "shoes": []}

        for slot, items in snap["items_by_slot"].items():
            category = SLOT_TO_CATEGORY.get(slot)
            if category is None:
                continue

            for item in items:
                goods_no = item["goodsNo"]
                product_id = f"musinsa_{goods_no}"

                snap_outfits[snap_key][category].append(product_id)

                if goods_no in seen:
                    continue
                seen.add(goods_no)

                ext = os.path.splitext(item["saved_path"])[1]
                filename = f"musinsa_{goods_no}{ext}"
                src_path = os.path.join(SRC_DIR, item["saved_path"])
                dst_path = os.path.join(DST_DIR, category, filename)
                shutil.copy2(src_path, dst_path)

                metadata.append({
                    "product_id": product_id,
                    "gender": gender,
                    "category": category,
                    "snap_id": snap_key,
                    "url": f"https://www.musinsa.com/products/{goods_no}",
                    "image_path": f"{category}/{filename}",
                    "name": item["name"],
                    "style_text": f"{item['name']}, {style_suffix}" if style_suffix else item["name"],
                })

    meta_path = os.path.join(DST_DIR, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    snap_outfits_path = os.path.join(DST_DIR, "snap_outfits.json")
    with open(snap_outfits_path, "w", encoding="utf-8") as f:
        json.dump(snap_outfits, f, ensure_ascii=False, indent=2)

    print(f"변환 완료: {len(metadata)}개 상품 → {meta_path}")
    print(f"snap_outfits: {len(snap_outfits)}개 snap → {snap_outfits_path}")
    by_gender: dict[str, int] = {}
    by_cat: dict[str, int] = {}
    for m in metadata:
        by_gender[m["gender"]] = by_gender.get(m["gender"], 0) + 1
        by_cat[m["category"]] = by_cat.get(m["category"], 0) + 1
    for g, cnt in sorted(by_gender.items()):
        print(f"  {g}: {cnt}개")
    for cat, cnt in sorted(by_cat.items()):
        print(f"  {cat}: {cnt}개")


if __name__ == "__main__":
    main()
