"""Tourism Attraction Search Tool using BaseTool interface.

Provides deterministic search over a structured local dataset.
Dataset is built from Taitung County Government Open Data.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import json
from pydantic import BaseModel, Field

from src.tools.base import BaseTool, ToolResult


BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATASET_PATH = BASE_DIR / "data" / "processed" / "attractions.json"

# Categories from merged official data (schema v2)
SUPPORTED_CATEGORIES = [
    "sightseeing",
    "activity",
    "nature",
    "service",
    "accommodation",
    "food",
]

# All 22 cities/counties from official data
SUPPORTED_CITIES = [
    "新北市", "臺北市", "桃園市", "臺中市", "臺南市", "高雄市",
    "基隆市", "新竹市", "新竹縣", "苗栗縣", "彰化縣", "南投縣",
    "雲林縣", "嘉義市", "嘉義縣", "屏東縣", "宜蘭縣", "花蓮縣",
    "臺東縣", "澎湖縣", "金門縣", "連江縣",
]

# English city names for API compatibility
SUPPORTED_CITIES_EN = [
    "NewTaipei", "Taipei", "Taoyuan", "Taichung", "Tainan", "Kaohsiung",
    "Keelung", "Hsinchu", "HsinchuCounty", "Miaoli", "Changhua", "Nantou",
    "Yunlin", "Chiayi", "ChiayiCounty", "Pingtung", "Yilan", "Hualien",
    "Taitung", "Penghu", "Kinmen", "Lienchiang",
]

CITY_NAME_MAP = {
    # Chinese variants -> Standard Chinese
    "台東": "臺東縣", "臺東": "臺東縣",
    "台北": "臺北市", "臺北": "臺北市",
    "台中": "臺中市", "臺中": "臺中市",
    "台南": "臺南市", "臺南": "臺南市",
    "高雄": "高雄市",
    "基隆": "基隆市",
    "新竹": "新竹市",
    "桃園": "桃園市",
    "苗栗": "苗栗縣",
    "彰化": "彰化縣",
    "南投": "南投縣",
    "雲林": "雲林縣",
    "嘉義": "嘉義市",
    "屏東": "屏東縣",
    "宜蘭": "宜蘭縣",
    "花蓮": "花蓮縣",
    "澎湖": "澎湖縣",
    "金門": "金門縣",
    "連江": "連江縣",
    # English -> Standard Chinese
    "NewTaipei": "新北市", "Taipei": "臺北市", "Taoyuan": "桃園市",
    "Taichung": "臺中市", "Tainan": "臺南市", "Kaohsiung": "高雄市",
    "Keelung": "基隆市", "Hsinchu": "新竹市", "HsinchuCounty": "新竹縣",
    "Miaoli": "苗栗縣", "Changhua": "彰化縣", "Nantou": "南投縣",
    "Yunlin": "雲林縣", "Chiayi": "嘉義市", "ChiayiCounty": "嘉義縣",
    "Pingtung": "屏東縣", "Yilan": "宜蘭縣", "Hualien": "花蓮縣",
    "Taitung": "臺東縣", "Penghu": "澎湖縣", "Kinmen": "金門縣", "Lienchiang": "連江縣",
}

CITY_NAME_MAP = {
    # Chinese variants -> Standard Chinese
    "台東": "臺東縣", "臺東": "臺東縣",
    "台北": "臺北市", "臺北": "臺北市",
    "台中": "臺中市", "臺中": "臺中市",
    "台南": "臺南市", "臺南": "臺南市",
    "高雄": "高雄市",
    "基隆": "基隆市",
    "新竹": "新竹市",
    "桃園": "桃園市",
    "苗栗": "苗栗縣",
    "彰化": "彰化縣",
    "南投": "南投縣",
    "雲林": "雲林縣",
    "嘉義": "嘉義市",
    "屏東": "屏東縣",
    "宜蘭": "宜蘭縣",
    "花蓮": "花蓮縣",
    "澎湖": "澎湖縣",
    "金門": "金門縣",
    "連江": "連江縣",
    # English -> Standard Chinese
    "NewTaipei": "新北市", "Taipei": "臺北市", "Taoyuan": "桃園市",
    "Taichung": "臺中市", "Tainan": "臺南市", "Kaohsiung": "高雄市",
    "Keelung": "基隆市", "Hsinchu": "新竹市", "HsinchuCounty": "新竹縣",
    "Miaoli": "苗栗縣", "Changhua": "彰化縣", "Nantou": "南投縣",
    "Yunlin": "雲林縣", "Chiayi": "嘉義市", "ChiayiCounty": "嘉義縣",
    "Pingtung": "屏東縣", "Yilan": "宜蘭縣", "Hualien": "花蓮縣",
    "Taitung": "臺東縣", "Penghu": "澎湖縣", "Kinmen": "金門縣", "Lienchiang": "連江縣",
}


class AttractionSearchInput(BaseModel):
    """Input schema for attraction search tool."""
    city: str = Field(..., description="City name (e.g., Taitung, Taipei, 台東, 台北)")
    category: str = Field("", description="Category: accommodation, activity, sightseeing, food, nature, service")
    language: str = Field(default="zh-TW", description="Response language: zh-TW, en, ja")


class AttractionResult(BaseModel):
    """Single attraction result."""
    id: Optional[str] = None
    name: str
    city: str
    district: Optional[str] = None
    category: str
    tags: List[str] = Field(default_factory=list)
    description: Optional[str] = None
    address: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    family_suitability: Optional[str] = None
    indoor_outdoor: Optional[str] = None
    source: Optional[str] = None
    source_url: Optional[str] = None
    last_updated: Optional[str] = None


class AttractionSearchOutput(BaseModel):
    """Output schema for attraction search tool."""
    city: str
    category: Optional[str] = None
    count: int
    results: List[AttractionResult]
    source: str = "open-data-taitung"
    language: str = "zh-TW"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AttractionsTool(BaseTool):
    """Tourism attraction search tool using local dataset."""

    name = "search_attractions"
    description = "Search attractions in a city by category"
    input_schema = AttractionSearchInput
    output_schema = AttractionSearchOutput

    def __init__(self):
        super().__init__()
        self._dataset_cache = None

    def _load_dataset(self) -> Dict[str, Any]:
        """Load the processed attractions dataset from JSON file."""
        if self._dataset_cache is not None:
            return self._dataset_cache

        if not DATASET_PATH.exists():
            raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

        with open(DATASET_PATH, "r", encoding="utf-8") as f:
            self._dataset_cache = json.load(f)

        return self._dataset_cache

    def _normalize_city(self, city: str) -> str:
        """Normalize city name to standard Chinese (e.g., 台東 -> 臺東縣)."""
        normalized = city.strip()
        # Handle English names
        if normalized in CITY_NAME_MAP:
            return CITY_NAME_MAP[normalized]
        # Handle 台/臺 variants
        normalized = normalized.replace("台", "臺")
        if normalized in CITY_NAME_MAP:
            return CITY_NAME_MAP[normalized]
        # Already standard?
        if normalized in SUPPORTED_CITIES:
            return normalized
        # Try partial match
        for standard in SUPPORTED_CITIES:
            if normalized in standard or standard in normalized:
                return standard
        return normalized

    def _matches_city(self, attraction: Dict[str, Any], city: str) -> bool:
        """Check if attraction matches the given city."""
        if not attraction.get("city"):
            return False

        search_city = self._normalize_city(city)
        attraction_city = attraction["city"]  # Already standard Chinese
        attraction_district = attraction.get("district", "")

        # Direct match
        if search_city == attraction_city:
            return True
        if search_city in attraction_city or attraction_city in search_city:
            return True

        # Match against district
        if attraction_district:
            if search_city == attraction_district:
                return True
            if search_city in attraction_district or attraction_district in search_city:
                return True

        return False

    def _matches_category(self, attraction: Dict[str, Any], category: str) -> bool:
        """Check if attraction matches the given category."""
        attr_category = attraction.get("category", "").lower()
        search_category = category.lower()

        if attr_category == search_category:
            return True

        # Also check tags
        tags = [tag.lower() for tag in attraction.get("tags", [])]
        if search_category in tags:
            return True

        return False

    def execute(self, input: AttractionSearchInput) -> ToolResult:
        """Execute the attraction search tool."""
        city = input.city.strip()
        language = input.language or "zh-TW"
        if not city:
            return ToolResult.fail(
                error_type="invalid_input",
                message="City must not be empty",
                retryable=False,
            )

        try:
            dataset = self._load_dataset()
        except FileNotFoundError as e:
            return ToolResult.fail(
                error_type="data_not_found",
                message=str(e),
                retryable=False,
            )
        except json.JSONDecodeError as e:
            return ToolResult.fail(
                error_type="invalid_dataset",
                message=f"Failed to parse dataset: {str(e)}",
                retryable=False,
            )

        attractions: List[Dict[str, Any]] = dataset.get("attractions", [])

        if not attractions:
            return ToolResult.fail(
                error_type="invalid_dataset",
                message="Dataset contains no attractions",
                retryable=False,
            )

        # Filter by city
        city_filtered = [a for a in attractions if self._matches_city(a, city)]

        if not city_filtered:
            return ToolResult.fail(
                error_type="city_not_found",
                message=f"No attractions found for city: {city}",
                retryable=False,
            )

        # Filter by category if provided
        category = input.category.strip() if input.category else ""
        if category:
            category_filtered = [a for a in city_filtered if self._matches_category(a, category)]

            if not category_filtered:
                return ToolResult.fail(
                    error_type="category_not_found",
                    message=f"No attractions found for category: {category} in {city}",
                    retryable=False,
                )

            results = category_filtered
        else:
            results = city_filtered

        # Build structured response
        search_results = []
        for r in results:
            search_results.append(AttractionResult(
                id=r.get("id"),
                name=r.get("name"),
                city=r.get("city"),
                district=r.get("district"),
                category=r.get("category"),
                tags=r.get("tags", []),
                description=r.get("description"),
                address=r.get("address"),
                latitude=r.get("latitude"),
                longitude=r.get("longitude"),
                family_suitability=r.get("family_suitability"),
                indoor_outdoor=r.get("indoor_outdoor"),
                source=r.get("source"),
                source_url=r.get("source_url"),
                last_updated=r.get("last_updated"),
            ))

        output = AttractionSearchOutput(
            city=city,
            category=input.category if input.category else None,
            count=len(search_results),
            results=search_results,
            source="taiwan-tourism-bureau",
            language=language,
            metadata={
                "dataset_total": len(attractions),
                "dataset_source": dataset.get("metadata", {}).get("source", "交通部觀光署開放資料"),
                "dataset_url": dataset.get("metadata", {}).get("source_url", "https://admin.taiwan.net.tw/"),
                "dataset_last_updated": dataset.get("metadata", {}).get("last_updated", dataset.get("metadata", {}).get("upstream_update_time", "")),
                "provider_id": dataset.get("metadata", {}).get("provider_id", "A15010000H"),
            }
        )

        return ToolResult.ok(output)


# Helper functions for backward compatibility
def _normalize_city(city: str) -> str:
    """Normalize city name to standard Chinese (e.g., 台東 -> 臺東縣)."""
    normalized = city.strip()
    # Handle English names
    if normalized in CITY_NAME_MAP:
        return CITY_NAME_MAP[normalized]
    # Handle 台/臺 variants
    normalized = normalized.replace("台", "臺")
    if normalized in CITY_NAME_MAP:
        return CITY_NAME_MAP[normalized]
    # Already standard?
    if normalized in SUPPORTED_CITIES:
        return normalized
    # Try partial match
    for standard in SUPPORTED_CITIES:
        if normalized in standard or standard in normalized:
            return standard
    return normalized


def _matches_city(attraction: Dict[str, Any], city: str) -> bool:
    """Check if attraction matches the given city."""
    tool = AttractionsTool()
    return tool._matches_city(attraction, city)


def _matches_category(attraction: Dict[str, Any], category: str) -> bool:
    """Check if attraction matches the given category."""
    tool = AttractionsTool()
    return tool._matches_category(attraction, category)


# Backward compatibility function
def search_attractions(city: str = "", category: str = "", language: str = "zh-TW") -> dict:
    """Backward compatible function."""
    tool = AttractionsTool()
    result = tool(AttractionSearchInput(city=city, category=category, language=language))
    if result.success:
        data = result.data.model_dump()
        data["success"] = True
        return data
    else:
        return {
            "success": False,
            "error_type": result.error.type,
            "message": result.error.message,
        }