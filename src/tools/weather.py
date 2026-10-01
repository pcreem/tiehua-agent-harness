"""Open-Meteo Weather Tool using BaseTool interface."""

import requests
from typing import Optional
from pydantic import BaseModel, Field

from src.tools.base import BaseTool, ToolResult


# Constants
WEATHER_CODE_MAP = {
    0: "clear", 1: "clear", 2: "cloudy", 3: "cloudy", 45: "cloudy", 48: "cloudy",
    51: "rain", 53: "rain", 55: "rain", 56: "rain", 57: "rain",
    61: "rain", 63: "rain", 65: "rain", 66: "rain", 67: "rain",
    71: "snow", 73: "snow", 75: "snow", 77: "snow",
    80: "rain", 81: "rain", 82: "rain", 85: "snow", 86: "snow",
    95: "thunderstorm", 96: "thunderstorm", 99: "thunderstorm",
}

CITY_NAME_MAP = {
    "台東": "Taitung", "臺東": "Taitung", "台北": "Taipei", "臺北": "Taipei",
    "台北市": "Taipei", "臺北市": "Taipei", "台中": "Taichung", "臺中": "Taichung",
    "台中市": "Taichung", "臺中市": "Taichung", "台南": "Tainan", "臺南": "Tainan",
    "台南市": "Tainan", "臺南市": "Tainan", "高雄": "Kaohsiung", "高雄市": "Kaohsiung",
    "基隆": "Keelung", "基隆市": "Keelung", "新竹": "Hsinchu", "新竹市": "Hsinchu",
    "新竹縣": "Hsinchu County", "桃園": "Taoyuan", "桃園市": "Taoyuan",
    "苗栗": "Miaoli", "苗栗縣": "Miaoli", "彰化": "Changhua", "彰化縣": "Changhua",
    "南投": "Nantou", "南投縣": "Nantou", "雲林": "Yunlin", "雲林縣": "Yunlin",
    "嘉義": "Chiayi", "嘉義市": "Chiayi", "嘉義縣": "Chiayi", "屏東": "Pingtung",
    "屏東縣": "Pingtung", "宜蘭": "Yilan", "宜蘭縣": "Yilan", "花蓮": "Hualien",
    "花蓮縣": "Hualien", "澎湖": "Penghu", "澎湖縣": "Penghu", "金門": "Kinmen",
    "金門縣": "Kinmen", "連江": "Lienchiang", "連江縣": "Lienchiang",
}

GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_TIMEOUT = 10


class WeatherInput(BaseModel):
    """Input schema for weather tool."""
    city: str = Field(..., description="City name (e.g., Taitung, Taipei, 台東, 台北)")


class WeatherOutput(BaseModel):
    """Output schema for weather tool."""
    city: str
    latitude: float
    longitude: float
    temperature: Optional[float] = None
    precipitation_probability: Optional[int] = None
    daily_precipitation_probability_max: Optional[int] = None
    weather_code: Optional[int] = None
    weather: Optional[str] = None
    source: str = "open-meteo"


class WeatherTool(BaseTool[WeatherInput, WeatherOutput]):
    """Weather tool using Open-Meteo API."""

    name = "get_weather"
    description = "Get current weather for a city"
    input_schema = WeatherInput
    output_schema = WeatherOutput

    def __init__(self, timeout: int = 10):
        super().__init__()
        self.timeout = timeout

    def _normalize_city_name(self, city: str) -> str:
        stripped = city.strip()
        return CITY_NAME_MAP.get(stripped, stripped)

    def _geocode_city(self, city: str) -> dict:
        """Geocode a city name to latitude/longitude."""
        search_name = self._normalize_city_name(city)

        params = {
            "name": search_name,
            "count": 1,
            "language": "zh",
            "format": "json"
        }

        try:
            response = requests.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            if not data.get("results"):
                return {"success": False, "error_type": "city_not_found", "message": "City not found"}

            result = data["results"][0]
            return {
                "success": True,
                "city": result.get("name", city),
                "latitude": result["latitude"],
                "longitude": result["longitude"],
            }

        except requests.exceptions.Timeout:
            return {"success": False, "error_type": "timeout", "message": "Geocoding API request timed out"}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error_type": "api_error", "message": f"Geocoding API request failed: {str(e)}"}

    def _fetch_weather(self, latitude: float, longitude: float) -> dict:
        """Fetch weather data from Open-Meteo Forecast API."""
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,precipitation_probability,weather_code",
            "daily": "precipitation_probability_max",
            "timezone": "auto"
        }

        try:
            response = requests.get(
                "https://api.open-meteo.com/v1/forecast",
                params=params,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            current = data.get("current", {})
            daily = data.get("daily", {})

            weather_code = current.get("weather_code")
            weather = WEATHER_CODE_MAP.get(weather_code, "unknown")

            return {
                "success": True,
                "temperature": current.get("temperature_2m"),
                "precipitation_probability": current.get("precipitation_probability"),
                "weather_code": weather_code,
                "weather": weather,
                "daily_precipitation_probability_max": daily.get("precipitation_probability_max", [None])[0]
            }

        except requests.exceptions.Timeout:
            return {"success": False, "error_type": "timeout", "message": "Weather API request timed out"}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error_type": "api_error", "message": f"Weather API request failed: {str(e)}"}

    def execute(self, input: WeatherInput) -> ToolResult[WeatherOutput]:
        """Execute the weather tool."""
        city = input.city.strip()
        if not city:
            return ToolResult.fail(
                error_type="invalid_input",
                message="City must not be empty",
                retryable=False,
            )

        # Geocode city
        geocode_result = self._geocode_city(city)
        if not geocode_result["success"]:
            return ToolResult.fail(
                error_type=geocode_result["error_type"],
                message=geocode_result["message"],
                retryable=geocode_result["error_type"] in ["timeout", "api_error"],
            )

        # Fetch weather
        weather_result = self._fetch_weather(
            geocode_result["latitude"],
            geocode_result["longitude"]
        )
        if not weather_result["success"]:
            return ToolResult.fail(
                error_type=weather_result["error_type"],
                message=weather_result["message"],
                retryable=weather_result["error_type"] in ["timeout", "api_error"],
            )

        # Build output
        output = WeatherOutput(
            city=geocode_result["city"],
            latitude=geocode_result["latitude"],
            longitude=geocode_result["longitude"],
            temperature=weather_result["temperature"],
            precipitation_probability=weather_result["precipitation_probability"],
            daily_precipitation_probability_max=weather_result["daily_precipitation_probability_max"],
            weather_code=weather_result["weather_code"],
            weather=weather_result["weather"],
            source="open-meteo",
        )

        return ToolResult.ok(output)


# Backward compatibility function
def get_weather(city: str) -> dict:
    """Backward compatible function."""
    tool = WeatherTool()
    result = tool(WeatherInput(city=city))
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


# Helper functions for backward compatibility
def _normalize_city_name(city: str) -> str:
    """Normalize city name to English for Open-Meteo API."""
    stripped = city.strip()
    return CITY_NAME_MAP.get(stripped, stripped)


def geocode_city(city: str) -> Dict[str, Any]:
    """Geocode a city name to latitude/longitude using Open-Meteo Geocoding API."""
    tool = WeatherTool()
    if not city or not city.strip():
        return {
            "success": False,
            "error_type": "invalid_input",
            "message": "City must not be empty"
        }
    return tool._geocode_city(city)


def fetch_weather(latitude: float, longitude: float) -> Dict[str, Any]:
    """Fetch weather data from Open-Meteo Forecast API."""
    tool = WeatherTool()
    return tool._fetch_weather(latitude, longitude)


# Constants for backward compatibility
WEATHER_CODE_MAP = {
    0: "clear", 1: "clear", 2: "cloudy", 3: "cloudy", 45: "cloudy", 48: "cloudy",
    51: "rain", 53: "rain", 55: "rain", 56: "rain", 57: "rain",
    61: "rain", 63: "rain", 65: "rain", 66: "rain", 67: "rain",
    71: "snow", 73: "snow", 75: "snow", 77: "snow",
    80: "rain", 81: "rain", 82: "rain", 85: "snow", 86: "snow",
    95: "thunderstorm", 96: "thunderstorm", 99: "thunderstorm",
}

CITY_NAME_MAP = {
    "台東": "Taitung", "臺東": "Taitung", "台北": "Taipei", "臺北": "Taipei",
    "台北市": "Taipei", "臺北市": "Taipei", "台中": "Taichung", "臺中": "Taichung",
    "台中市": "Taichung", "臺中市": "Taichung", "台南": "Tainan", "臺南": "Tainan",
    "台南市": "Tainan", "臺南市": "Tainan", "高雄": "Kaohsiung", "高雄市": "Kaohsiung",
    "基隆": "Keelung", "基隆市": "Keelung", "新竹": "Hsinchu", "新竹市": "Hsinchu",
    "新竹縣": "Hsinchu County", "桃園": "Taoyuan", "桃園市": "Taoyuan",
    "苗栗": "Miaoli", "苗栗縣": "Miaoli", "彰化": "Changhua", "彰化縣": "Changhua",
    "南投": "Nantou", "南投縣": "Nantou", "雲林": "Yunlin", "雲林縣": "Yunlin",
    "嘉義": "Chiayi", "嘉義市": "Chiayi", "嘉義縣": "Chiayi", "屏東": "Pingtung",
    "屏東縣": "Pingtung", "宜蘭": "Yilan", "宜蘭縣": "Yilan", "花蓮": "Hualien",
    "花蓮縣": "Hualien", "澎湖": "Penghu", "澎湖縣": "Penghu", "金門": "Kinmen",
    "金門縣": "Kinmen", "連江": "Lienchiang", "連江縣": "Lienchiang",
}