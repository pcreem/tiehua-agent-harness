#!/usr/bin/env python3
"""
Merge AttractionList + AttractionFeeList + AttractionServiceTimeList into unified schema v2.

Official data source: 交通部觀光署開放資料 (ProviderID: A15010000H)
Updated: 2026-10-03
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import hashlib

# ============================================================
# 類別代碼對照表 (基於觀光署分類，需定期同步官方最新版)
# ============================================================
CLASS_CODE_MAP = {
    1:  {"zh-TW": "一般觀光景點", "en": "General Sightseeing", "ja": "一般観光スポット", "category": "sightseeing"},
    2:  {"zh-TW": "歷史古蹟", "en": "Historical Sites", "ja": "史跡", "category": "sightseeing"},
    3:  {"zh-TW": "文化藝術", "en": "Culture & Arts", "ja": "文化芸術", "category": "sightseeing"},
    4:  {"zh-TW": "產業參觀", "en": "Industrial Tours", "ja": "産業見学", "category": "activity"},
    5:  {"zh-TW": "寺廟信仰", "en": "Temples & Religion", "ja": "寺院信仰", "category": "sightseeing"},
    6:  {"zh-TW": "步道", "en": "Trails", "ja": "遊歩道", "category": "nature"},
    7:  {"zh-TW": "其他", "en": "Others", "ja": "その他", "category": "sightseeing"},
    8:  {"zh-TW": "博物館", "en": "Museums", "ja": "博物館", "category": "sightseeing"},
    9:  {"zh-TW": "產業參觀", "en": "Industrial Tours", "ja": "産業見学", "category": "activity"},
    11: {"zh-TW": "休閒農業", "en": "Leisure Agriculture", "ja": "レジャー農業", "category": "activity"},
    12: {"zh-TW": "自然風景區", "en": "Natural Scenic Areas", "ja": "自然風景区", "category": "nature"},
    13: {"zh-TW": "親子共遊", "en": "Family Activities", "ja": "親子遊び", "category": "activity"},
    14: {"zh-TW": "無障礙環境", "en": "Accessible Environments", "ja": "バリアフリー環境", "category": "service"},
    15: {"zh-TW": "餐飲", "en": "Food & Dining", "ja": "飲食", "category": "food"},
    16: {"zh-TW": "森林遊樂區", "en": "Forest Recreation Areas", "ja": "森林遊楽区", "category": "nature"},
    18: {"zh-TW": "遊憩區", "en": "Recreation Areas", "ja": "遊憩区", "category": "nature"},
    19: {"zh-TW": "購物", "en": "Shopping", "ja": "ショッピング", "category": "service"},
    23: {"zh-TW": "步道入口", "en": "Trail Entrances", "ja": "遊歩道入口", "category": "nature"},
    24: {"zh-TW": "旅遊服務", "en": "Tourism Services", "ja": "観光サービス", "category": "service"},
    25: {"zh-TW": "住宿", "en": "Accommodation", "ja": "宿泊", "category": "accommodation"},
    254: {"zh-TW": "活動", "en": "Events", "ja": "イベント", "category": "activity"},
}


def map_class_codes(codes: List[int]) -> Dict[str, Any]:
    """Map attraction class codes to multilingual names and unified category."""
    if not codes:
        return {"class_names": {}, "primary_category": "sightseeing", "all_categories": ["sightseeing"]}

    class_names = {"zh-TW": [], "en": [], "ja": []}
    categories = set()

    for code in codes:
        mapping = CLASS_CODE_MAP.get(code, {"zh-TW": f"未知分類({code})", "en": f"Unknown({code})", "ja": f"不明({code})", "category": "sightseeing"})
        class_names["zh-TW"].append(mapping["zh-TW"])
        class_names["en"].append(mapping["en"])
        class_names["ja"].append(mapping["ja"])
        categories.add(mapping["category"])

    # Primary category: prefer accommodation > food > activity > nature > sightseeing > service
    priority = ["accommodation", "food", "activity", "nature", "sightseeing", "service"]
    primary = next((c for c in priority if c in categories), "sightseeing")

    return {
        "class_names": {k: "、".join(v) for k, v in class_names.items()},
        "primary_category": primary,
        "all_categories": sorted(list(categories)),
    }


def parse_fee_info(fee_info_text: Optional[str]) -> List[Dict[str, Any]]:
    """Parse inline FeeInfo text into structured format."""
    if not fee_info_text:
        return []

    # Simple parsing for common patterns: "全票：80元；優待票：50元..."
    fees = []
    parts = fee_info_text.split("；")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Try to extract name and price
        if "：" in part:
            name, rest = part.split("：", 1)
            price = None
            # Extract numeric price
            import re
            match = re.search(r"(\d+(?:,\d{3})*)\s*元", rest)
            if match:
                price = int(match.group(1).replace(",", ""))
            fees.append({
                "name": name.strip(),
                "price": price,
                "description": rest.strip(),
                "url": None
            })
        else:
            fees.append({
                "name": part,
                "price": None,
                "description": part,
                "url": None
            })
    return fees


def parse_service_time_info(st_text: Optional[str]) -> List[Dict[str, Any]]:
    """Parse inline ServiceTimeInfo text into structured format."""
    if not st_text:
        return []

    # Very basic parsing - the inline text is usually simple like "每日開放" or "09:00-18:00"
    # For complex cases, the separate ServiceTimeList.json is better
    return [{
        "name": "營業時間",
        "description": st_text,
        "schedule": []  # Would need more sophisticated parsing
    }]


def build_unified_record(
    attr: Dict[str, Any],
    fee_map: Dict[str, Any],
    st_map: Dict[str, Any],
) -> Dict[str, Any]:
    """Build unified schema v2 record from merged data."""

    # Basic identifiers
    attraction_id = attr.get("AttractionID", "")
    name = attr.get("AttractionName", "")

    # Class mapping
    class_codes = attr.get("AttractionClasses", [])
    class_info = map_class_codes(class_codes)

    # Address
    addr = attr.get("PostalAddress", {})
    city = addr.get("City", "")
    district = addr.get("Town", "")
    street = addr.get("StreetAddress", "")
    zipcode = addr.get("ZipCode", "")
    full_address = f"{city}{district}{street}".strip()

    # Coordinates
    lat = attr.get("PositionLat")
    lon = attr.get("PositionLon")

    # Description
    description = attr.get("Description", "")

    # Images
    images = attr.get("Images", [])
    image_urls = [img.get("URL") for img in images if img.get("URL")]

    # Telephones
    phones = [tel.get("Tel") for tel in attr.get("Telephones", []) if tel.get("Tel")]

    # Fee info: prefer structured FeeList, fallback to inline FeeInfo
    fee_data = fee_map.get(attraction_id, {})
    structured_fees = fee_data.get("Fees", []) if fee_data else []
    inline_fees = parse_fee_info(attr.get("FeeInfo"))
    fees = structured_fees if structured_fees else inline_fees

    # Service time: prefer structured ServiceTimeList, fallback to inline
    st_data = st_map.get(attraction_id, {})
    structured_st = st_data.get("ServiceTimes", []) if st_data else []
    inline_st = parse_service_time_info(attr.get("ServiceTimeInfo"))
    service_times = structured_st if structured_st else inline_st

    # Parking
    parking = attr.get("ParkingInfo")

    # Facilities (code array, would need separate mapping)
    facilities = attr.get("Facilities", [])

    # Traffic
    traffic = attr.get("TrafficInfo")

    # Website
    website = attr.get("WebsiteURL")

    # Status flags
    is_public = attr.get("IsPublicAccess", True)
    is_free = attr.get("IsAccessibleForFree", True)

    # Provenance
    source = "交通部觀光署開放資料"
    source_url = attr.get("WebsiteURL") or "https://admin.taiwan.net.tw/"
    last_updated = attr.get("UpdateTime", "")
    # Normalize to ISO 8601 if possible
    # The UpdateTime format: "2026-10-03T02:30:03+08:00" - already ISO 8601

    # Compute confidence based on data completeness
    confidence = 0.0
    if lat is not None and lon is not None:
        confidence += 0.3
    if description and len(description) > 50:
        confidence += 0.2
    if fees:
        confidence += 0.1
    if service_times:
        confidence += 0.1
    if image_urls:
        confidence += 0.1
    if traffic:
        confidence += 0.1
    if parking:
        confidence += 0.1
    if class_codes:
        confidence += 0.1

    # Generate deterministic ID hash for versioning
    content_hash = hashlib.sha256(
        f"{attraction_id}|{name}|{description}|{last_updated}".encode()
    ).hexdigest()[:16]

    return {
        "id": attraction_id,
        "content_hash": content_hash,
        "names": {
            "zh-TW": name,
            "en": "",  # TODO: merge from En version
            "ja": "",  # TODO: merge from Ja version
        },
        "class_names": class_info["class_names"],
        "category": class_info["primary_category"],
        "all_categories": class_info["all_categories"],
        "location": {
            "address": {
                "zh-TW": full_address,
                "en": "",
                "ja": "",
            },
            "city": city,
            "district": district,
            "zipcode": zipcode,
            "street": street,
            "latitude": lat,
            "longitude": lon,
            "geo_precision": "address" if lat and lon else "none",
        },
        "description": {
            "zh-TW": description,
            "en": "",
            "ja": "",
        },
        "attributes": {
            "fees": fees,
            "service_times": service_times,
            "parking": parking,
            "facilities_codes": facilities,
            "traffic_info": traffic,
            "website_url": website,
            "phone_numbers": phones,
            "image_urls": image_urls,
            "is_public_access": is_public,
            "is_free_access": is_free,
        },
        "provenance": {
            "source": source,
            "source_url": source_url,
            "provider_id": "A15010000H",
            "license": "政府資料開放授權條款第1版",
            "retrieved_at": "2026-10-03T04:30:00+08:00",
            "upstream_updated_at": last_updated,
            "collector_version": "v2.0.0",
        },
        "quality": {
            "confidence": round(confidence, 2),
            "description_completeness": "full" if description and len(description) > 100 else "partial" if description else "missing",
            "has_coordinates": lat is not None and lon is not None,
            "has_fees": bool(fees),
            "has_service_times": bool(service_times),
            "has_images": bool(image_urls),
            "reviewed_by": "none",
            "review_date": None,
        },
        "i18n_status": {
            "zh-TW": "complete",
            "en": "missing",
            "ja": "missing",
        },
    }


def main():
    base = Path(__file__).resolve().parent.parent / "data"

    print("Loading AttractionList.json...")
    with open(base / "Attraction-json" / "AttractionList.json", encoding="utf-8-sig") as f:
        list_data = json.load(f)
    attractions = list_data["Attractions"]
    print(f"  Loaded {len(attractions)} attractions")

    print("Loading AttractionFeeList.json...")
    with open(base / "Attraction-json" / "AttractionFeeList.json", encoding="utf-8-sig") as f:
        fee_data = json.load(f)
    fee_map = {item["AttractionID"]: item for item in fee_data["AttractionFees"]}
    print(f"  Loaded {len(fee_map)} fee records")

    print("Loading AttractionServiceTimeList.json...")
    with open(base / "Attraction-json" / "AttractionServiceTimeList.json", encoding="utf-8-sig") as f:
        st_data = json.load(f)
    st_map = {item["AttractionID"]: item for item in st_data["AttractionServiceTimes"]}
    print(f"  Loaded {len(st_map)} service time records")

    print("\nMerging records...")
    unified = []
    for i, attr in enumerate(attractions):
        unified.append(build_unified_record(attr, fee_map, st_map))
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(attractions)}")

    print(f"\nTotal merged: {len(unified)}")

    # Statistics
    cats = defaultdict(int)
    cities = defaultdict(int)
    for r in unified:
        cats[r["category"]] += 1
        cities[r["location"]["city"]] += 1

    print("\nCategory distribution:")
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    print("\nTop cities:")
    for city, count in sorted(cities.items(), key=lambda x: -x[1])[:15]:
        print(f"  {city}: {count}")

    # Quality stats
    with_coords = sum(1 for r in unified if r["quality"]["has_coordinates"])
    with_desc = sum(1 for r in unified if r["quality"]["description_completeness"] == "full")
    with_fees = sum(1 for r in unified if r["quality"]["has_fees"])
    with_st = sum(1 for r in unified if r["quality"]["has_service_times"])

    print(f"\nQuality metrics:")
    print(f"  Has coordinates: {with_coords}/{len(unified)} ({with_coords/len(unified)*100:.1f}%)")
    print(f"  Full description: {with_desc}/{len(unified)} ({with_desc/len(unified)*100:.1f}%)")
    print(f"  Has fees: {with_fees}/{len(unified)} ({with_fees/len(unified)*100:.1f}%)")
    print(f"  Has service times: {with_st}/{len(unified)} ({with_st/len(unified)*100:.1f}%)")

    # Write output
    output = {
        "metadata": {
            "source": "交通部觀光署開放資料",
            "source_url": "https://admin.taiwan.net.tw/",
            "provider_id": "A15010000H",
            "license": "政府資料開放授權條款第1版",
            "retrieved_at": "2026-10-03T04:30:00+08:00",
            "upstream_update_time": list_data.get("UpdateTime"),
            "total_records": len(unified),
            "schema_version": "v2.0.0",
            "language": "Zh_tw",
            "note": "Merged from AttractionList + FeeList + ServiceTimeList. En/Ja names/descriptions pending.",
        },
        "attractions": unified,
    }

    out_path = base / "processed" / "attractions_v2.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Written to {out_path} ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Also write a backward-compatible version for the current tool
    # The current tool expects: id, name, city, district, category, tags, description, address, latitude, longitude, family_suitability, indoor_outdoor, source, source_url, last_updated
    print("\nGenerating backward-compatible version...")
    compat = {
        "metadata": output["metadata"],
        "attractions": []
    }
    for r in unified:
        # Build tags from all_categories
        tags = r["all_categories"] + [r["category"]]
        # Family suitability: infer from class_names
        family_suit = "family_friendly" if "activity" in r["all_categories"] or "family" in str(r["class_names"]).lower() else "general"
        # Indoor/outdoor: infer from category
        indoor_outdoor = "outdoor" if r["category"] in ["nature", "sightseeing"] else "indoor"

        compat["attractions"].append({
            "id": r["id"],
            "name": r["names"]["zh-TW"],
            "city": r["location"]["city"],
            "district": r["location"]["district"],
            "category": r["category"],
            "tags": tags,
            "description": r["description"]["zh-TW"],
            "address": r["location"]["address"]["zh-TW"],
            "latitude": r["location"]["latitude"],
            "longitude": r["location"]["longitude"],
            "family_suitability": family_suit,
            "indoor_outdoor": indoor_outdoor,
            "source": r["provenance"]["source"],
            "source_url": r["provenance"]["source_url"],
            "last_updated": r["provenance"]["upstream_updated_at"],
        })

    compat_path = base / "processed" / "attractions.json"
    with open(compat_path, "w", encoding="utf-8") as f:
        json.dump(compat, f, ensure_ascii=False, indent=2)

    print(f"✅ Backward-compatible written to {compat_path} ({compat_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()