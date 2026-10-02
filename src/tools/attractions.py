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
    "台東": "Taitung", "臺東": "Taitung", "台北": "Taipei", "臺北": "Taipei",
    "台中": "Taichung", "臺中": "Taichung", "台南": "Tainan", "臺南": "Tainan",
    "高雄": "Kaohsiung", "基隆": "Keelung", "新竹": "Hsinchu", "桃園": "Taoyuan",
    "苗栗": "Miaoli", "彰化": "Changhua", "南投": "Nantou", "雲林": "Yunlin",
    "嘉義": "Chiayi", "屏東": "Pingtung", "宜蘭": "Yilan", "花蓮": "Hualien",
    "澎湖": "Penghu", "金門": "Kinmen", "連江": "Lienchiang",
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
        """Normalize city name (handle 台/臺 variants)."""
        normalized = city.strip().replace("臺", "台")
        return normalized

    def _build_reverse_city_map(self) -> Dict[str, str]:
        """Build English -> Chinese city mapping (first occurrence wins)."""
        reverse_map: Dict[str, str] = {}
        for chinese_name, english_name in CITY_NAME_MAP.items():
            if english_name not in reverse_map:
                reverse_map[english_name] = self._normalize_city(chinese_name)
        return reverse_map

    def _matches_city(self, attraction: Dict[str, Any], city: str) -> bool:
        """Check if attraction matches the given city."""
        if not attraction.get("city"):
            return False

        search_city = self._normalize_city(city)
        attraction_city = self._normalize_city(attraction["city"])
        attraction_district = self._normalize_city(attraction.get("district", ""))

        # Check against city field
        if search_city == attraction_city:
            return True
        if search_city in attraction_city or attraction_city in search_city:
            return True

        # Check against district field (e.g., "台東市" matches district "臺東市")
        if attraction_district and (search_city == attraction_district or search_city in attraction_district or attraction_district in search_city):
            return True

        # Map English city name to Chinese for comparison
        if search_city in self._build_reverse_city_map():
            chinese_name = self._build_reverse_city_map()[search_city]
            if chinese_name in attraction_city or attraction_city in chinese_name:
                return True
            if attraction_district and (chinese_name in attraction_district or attraction_district in chinese_name):
                return True

        # Map Chinese city name to English for comparison (reverse lookup)
        if search_city in CITY_NAME_MAP:
            english_name = CITY_NAME_MAP[search_city]
            if english_name == attraction_city:
                return True

        return False

    def _matches_category(self, attraction: Dict[str, Any], category: str) -> bool:
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
            source="open-data-taitung",
            language=language,
            metadata={
                "dataset_total": len(attractions),
                "dataset_source": dataset.get("metadata", {}).get("source", ""),
                "dataset_url": dataset.get("metadata", {}).get("source_url", ""),
                "dataset_last_updated": dataset.get("metadata", {}).get("last_updated", ""),
                "filter_district": dataset.get("metadata", {}).get("filter_district", ""),
            }
        )

        return ToolResult.ok(output)


# Helper functions for backward compatibility
def _normalize_city(city: str) -> str:
    """Normalize city name (handle 台/臺 variants)."""
    normalized = city.strip().replace("臺", "台")
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