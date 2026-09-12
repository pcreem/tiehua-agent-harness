"""Download / prepare tourism data from Taitung County Government Open Data.

Usage:
    python scripts/download_tourism_data.py

This script reads CSV files from data/raw/ and produces
data/processed/attractions.json with normalized schema.

Data Sources:
    - Taitung County Government Open Data
    - Taiwan Ministry of Tourism (觀光署)

Filter: Only keep records where district (行政區(鄉鎮區)名稱) is "台東市" / "臺東市"
"""

import csv
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "attractions.json"

# CSV file -> category mapping
SOURCES = {
    "景點.csv": {"category": "sightseeing"},
    "餐飲.csv": {"category": "food"},
    "住宿.csv": {"category": "accommodation"},
    "活動.csv": {"category": "activity"},
    "步道.csv": {"category": "nature"},
    "旅遊服務站點.csv": {"category": "service"},
}

# Column names in the CSV files (Chinese)
COL_ID = "唯一識別碼"
COL_DATA_TYPE = "資料類型"
COL_LANG = "語系"
COL_NAME = "資料名稱"
COL_DESC = "文字描述"
COL_PHONE = "聯絡電話"
COL_CITY = "縣市名稱"
COL_DISTRICT = "行政區(鄉鎮區)名稱"
COL_ADDRESS = "街道名稱"
COL_PROVIDER_CODE = "資料提供單位機關代碼"
COL_PROVIDER = "資料提供單位"
COL_UPDATE = "資料更新日期時間"

# Target districts to keep (台東市 and 臺東市 variants)
TARGET_DISTRICTS = {"台東市", "臺東市"}


def _is_target_district(district: str) -> bool:
    """Check if district matches target districts (handle 台/臺 variants)."""
    if not district:
        return False
    normalized = district.strip().replace("臺", "台")
    return normalized in TARGET_DISTRICTS


def parse_csv_file(filepath: Path, category: str) -> List[Dict[str, Any]]:
    """Parse a single CSV file and return normalized records for Taitung City only."""
    records: List[Dict[str, Any]] = []

    with open(filepath, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            record_id = row.get(COL_ID, "").strip()
            name = row.get(COL_NAME, "").strip()
            description = row.get(COL_DESC, "").strip()
            city = row.get(COL_CITY, "").strip()
            district = row.get(COL_DISTRICT, "").strip()
            address = row.get(COL_ADDRESS, "").strip()
            provider = row.get(COL_PROVIDER, "").strip()
            provider_code = row.get(COL_PROVIDER_CODE, "").strip()
            updated = row.get(COL_UPDATE, "").strip()

            if not name:
                continue

            # Filter: only keep records for 台東市 district
            if not _is_target_district(district):
                continue

            city_normalized = city.replace("臺", "台") if city else None

            records.append({
                "id": record_id or f"{category}_{name}",
                "name": name,
                "city": city_normalized,
                "district": district,
                "category": category,
                "tags": [category],
                "description": description,
                "address": address,
                "latitude": None,
                "longitude": None,
                "family_suitability": None,
                "indoor_outdoor": None,
                "source": provider or "Taitung County Government Open Data",
                "source_url": "https://media.taiwan.net.tw/",
                "last_updated": updated,
            })

    return records


def build_dataset() -> Dict[str, Any]:
    """Build the complete normalized dataset from all CSV files."""
    all_records: List[Dict[str, Any]] = []

    for filename, config in SOURCES.items():
        filepath = RAW_DIR / filename
        if not filepath.exists():
            print(f"Warning: {filepath} not found, skipping.")
            continue

        records = parse_csv_file(filepath, config["category"])
        all_records.extend(records)
        print(f"  {filename}: {len(records)} records (filtered for 台東市)")

    return {
        "metadata": {
            "source": "Taitung County Government Open Data",
            "source_url": "https://data.taitung.gov.tw/",
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
            "total_records": len(all_records),
            "filter_district": "台東市",
        },
        "attractions": all_records,
    }


def main():
    """Download and process tourism data."""
    print("Processing tourism data (filtering for 台東市 district)...")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"\nDataset written to: {OUTPUT_FILE}")
    print(f"Total records: {dataset['metadata']['total_records']}")
    print(f"Categories: {sorted(set(r['category'] for r in dataset['attractions']))}")


if __name__ == "__main__":
    main()