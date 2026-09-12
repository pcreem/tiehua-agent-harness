"""Tourism Attraction Search Tool

Provides deterministic search over a structured local dataset.
Dataset is built from Taitung County Government Open Data.
"""

from pathlib import Path
from typing import Dict, Any, List
import json


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = BASE_DIR / "data" / "processed" / "attractions.json"

SUPPORTED_CATEGORIES = [
    "sightseeing",
    "food",
    "accommodation",
    "activity",
    "nature",
    "service",
]

SUPPORTED_CITIES = [
    "Taitung",
    "Tainan",
    "Taipei",
    "Kaohsiung",
]

CITY_NAME_MAP = {
    "台東": "Taitung",
    "臺東": "Taitung",
    "台北": "Taipei",
    "臺北": "Taipei",
    "台中": "Taichung",
    "臺中": "Taichung",
    "台南": "Tainan",
    "臺南": "Tainan",
    "高雄": "Kaohsiung",
    "基隆": "Keelung",
    "新竹": "Hsinchu",
    "桃園": "Taoyuan",
    "苗栗": "Miaoli",
    "彰化": "Changhua",
    "南投": "Nantou",
    "雲林": "Yunlin",
    "嘉義": "Chiayi",
    "屏東": "Pingtung",
    "宜蘭": "Yilan",
    "花蓮": "Hualien",
    "澎湖": "Penghu",
    "金門": "Kinmen",
    "連江": "Lienchiang",
}


def _load_dataset() -> Dict[str, Any]:
    """Load the processed attractions dataset from JSON file."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_city(city: str) -> str:
    """Normalize city name (handle 台/臺 variants)."""
    normalized = city.strip().replace("臺", "台")
    return normalized


def _build_reverse_city_map() -> Dict[str, str]:
    """Build English -> Chinese city mapping (first occurrence wins)."""
    reverse_map: Dict[str, str] = {}
    for chinese_name, english_name in CITY_NAME_MAP.items():
        if english_name not in reverse_map:
            reverse_map[english_name] = _normalize_city(chinese_name)
    return reverse_map


REVERSE_CITY_MAP = _build_reverse_city_map()


def _matches_city(attraction: Dict[str, Any], city: str) -> bool:
    """Check if attraction matches the given city.

    Handles both English (e.g., "Taitung") and Chinese (e.g., "台東") city names
    by using a bidirectional mapping.
    """
    if not attraction.get("city"):
        return False

    search_city = _normalize_city(city)
    attraction_city = _normalize_city(attraction["city"])

    # Direct match (Chinese-to-Chinese or English-to-English)
    if search_city == attraction_city:
        return True
    if search_city in attraction_city or attraction_city in search_city:
        return True

    # Map English city name to Chinese for comparison
    if search_city in REVERSE_CITY_MAP:
        chinese_name = REVERSE_CITY_MAP[search_city]
        if chinese_name in attraction_city or attraction_city in chinese_name:
            return True

    # Map Chinese city name to English for comparison (reverse lookup)
    if search_city in CITY_NAME_MAP:
        english_name = CITY_NAME_MAP[search_city]
        if english_name == attraction_city:
            return True

    return False


def _matches_category(attraction: Dict[str, Any], category: str) -> bool:
    """Check if attraction matches the given category."""
    attr_category = attraction.get("category", "").lower()
    search_category = category.lower()

    if attr_category == search_category:
        return True

    # Also check tags
    tags = attraction.get("tags", [])
    if search_category in [tag.lower() for tag in tags]:
        return True

    return False


def search_attractions(
    city: str = "",
    category: str = "",
) -> Dict[str, Any]:
    """
    Search attractions from the structured dataset.

    Args:
        city: City name (e.g., "Taitung", "台東")
        category: Category (e.g., "culture", "sightseeing", "food")

    Returns:
        Structured dict with results or error.
    """
    if not city or not city.strip():
        return {
            "success": False,
            "error_type": "invalid_input",
            "message": "City must not be empty"
        }

    try:
        dataset = _load_dataset()
    except FileNotFoundError as e:
        return {
            "success": False,
            "error_type": "data_not_found",
            "message": str(e)
        }
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "error_type": "invalid_dataset",
            "message": f"Failed to parse dataset: {str(e)}"
        }

    attractions: List[Dict[str, Any]] = dataset.get("attractions", [])

    if not attractions:
        return {
            "success": False,
            "error_type": "invalid_dataset",
            "message": "Dataset contains no attractions"
        }

    # Filter by city
    city_filtered = [a for a in attractions if _matches_city(a, city)]

    if not city_filtered:
        return {
            "success": False,
            "error_type": "city_not_found",
            "message": f"No attractions found for city: {city}"
        }

    # Filter by category if provided
    if category and category.strip():
        category_filtered = [a for a in city_filtered if _matches_category(a, category)]

        if not category_filtered:
            return {
                "success": False,
                "error_type": "category_not_found",
                "message": f"No attractions found for category: {category} in {city}"
            }

        results = category_filtered
    else:
        results = city_filtered

    # Build structured response
    search_results = []
    for r in results:
        search_results.append({
            "id": r.get("id"),
            "name": r.get("name"),
            "city": r.get("city"),
            "district": r.get("district"),
            "category": r.get("category"),
            "tags": r.get("tags", []),
            "description": r.get("description"),
            "address": r.get("address"),
            "latitude": r.get("latitude"),
            "longitude": r.get("longitude"),
            "family_suitability": r.get("family_suitability"),
            "indoor_outdoor": r.get("indoor_outdoor"),
            "source": r.get("source"),
            "source_url": r.get("source_url"),
            "last_updated": r.get("last_updated"),
        })

    return {
        "success": True,
        "city": city,
        "category": category if category else None,
        "count": len(search_results),
        "results": search_results,
        "source": "open-data-taitung",
        "metadata": {
            "dataset_total": len(attractions),
            "dataset_source": dataset.get("metadata", {}).get("source", ""),
            "dataset_url": dataset.get("metadata", {}).get("source_url", ""),
            "dataset_last_updated": dataset.get("metadata", {}).get("last_updated", ""),
            "filter_district": dataset.get("metadata", {}).get("filter_district", ""),
        }
    }


if __name__ == "__main__":
    print(json.dumps(search_attractions("Taitung", "sightseeing"), indent=2, ensure_ascii=False))
