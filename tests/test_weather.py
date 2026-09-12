"""Unit tests for weather tool"""

import pytest
from unittest.mock import patch, Mock
from src.tools.weather import get_weather, geocode_city, fetch_weather, WEATHER_CODE_MAP


class TestWeatherTool:
    """Tests for get_weather function"""

    def test_invalid_input_empty_string(self):
        """Test empty city string returns invalid_input error"""
        result = get_weather("")
        assert result["success"] is False
        assert result["error_type"] == "invalid_input"
        assert "empty" in result["message"].lower()

    def test_invalid_input_whitespace(self):
        """Test whitespace-only city string returns invalid_input error"""
        result = get_weather("   ")
        assert result["success"] is False
        assert result["error_type"] == "invalid_input"

    @patch("src.tools.weather.requests.get")
    def test_city_not_found(self, mock_get):
        """Test city not found returns city_not_found error"""
        mock_response = Mock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = get_weather("abcdefg")
        assert result["success"] is False
        assert result["error_type"] == "city_not_found"

    @patch("src.tools.weather.requests.get")
    def test_geocoding_timeout(self, mock_get):
        """Test geocoding timeout returns timeout error"""
        import requests
        mock_get.side_effect = requests.exceptions.Timeout()

        result = get_weather("Taitung")
        assert result["success"] is False
        assert result["error_type"] == "timeout"

    @patch("src.tools.weather.requests.get")
    def test_geocoding_api_error(self, mock_get):
        """Test geocoding API error returns api_error"""
        import requests
        mock_get.side_effect = requests.exceptions.RequestException("Connection error")

        result = get_weather("Taitung")
        assert result["success"] is False
        assert result["error_type"] == "api_error"

    @patch("src.tools.weather.requests.get")
    def test_successful_response(self, mock_get):
        """Test successful weather response"""
        # First call: geocoding
        # Second call: forecast
        geocode_response = Mock()
        geocode_response.json.return_value = {
            "results": [{
                "name": "Taitung",
                "latitude": 22.75,
                "longitude": 121.15,
                "country": "Taiwan",
                "timezone": "Asia/Taipei"
            }]
        }
        geocode_response.raise_for_status.return_value = None

        forecast_response = Mock()
        forecast_response.json.return_value = {
            "current": {
                "temperature_2m": 28.4,
                "precipitation_probability": 80,
                "weather_code": 61
            },
            "daily": {
                "precipitation_probability_max": [85]
            }
        }
        forecast_response.raise_for_status.return_value = None

        mock_get.side_effect = [geocode_response, forecast_response]

        result = get_weather("Taitung")
        assert result["success"] is True
        assert result["city"] == "Taitung"
        assert result["latitude"] == 22.75
        assert result["longitude"] == 121.15
        assert result["temperature"] == 28.4
        assert result["precipitation_probability"] == 80
        assert result["weather_code"] == 61
        assert result["weather"] == "rain"
        assert result["source"] == "open-meteo"

    @patch("src.tools.weather.requests.get")
    def test_forecast_timeout(self, mock_get):
        """Test forecast API timeout returns timeout error"""
        import requests

        geocode_response = Mock()
        geocode_response.json.return_value = {
            "results": [{
                "name": "Taitung",
                "latitude": 22.75,
                "longitude": 121.15,
                "country": "Taiwan",
                "timezone": "Asia/Taipei"
            }]
        }
        geocode_response.raise_for_status.return_value = None

        mock_get.side_effect = [geocode_response, requests.exceptions.Timeout()]

        result = get_weather("Taitung")
        assert result["success"] is False
        assert result["error_type"] == "timeout"

    @patch("src.tools.weather.requests.get")
    def test_forecast_api_error(self, mock_get):
        """Test forecast API error returns api_error"""
        import requests

        geocode_response = Mock()
        geocode_response.json.return_value = {
            "results": [{
                "name": "Taitung",
                "latitude": 22.75,
                "longitude": 121.15,
                "country": "Taiwan",
                "timezone": "Asia/Taipei"
            }]
        }
        geocode_response.raise_for_status.return_value = None

        mock_get.side_effect = [geocode_response, requests.exceptions.RequestException("API down")]

        result = get_weather("Taitung")
        assert result["success"] is False
        assert result["error_type"] == "api_error"


class TestGeocodeCity:
    """Tests for geocode_city function"""

    @patch("src.tools.weather.requests.get")
    def test_geocode_success(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = {
            "results": [{
                "name": "Taipei",
                "latitude": 25.03,
                "longitude": 121.57,
                "country": "Taiwan",
                "timezone": "Asia/Taipei"
            }]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = geocode_city("Taipei")
        assert result["success"] is True
        assert result["city"] == "Taipei"
        assert result["latitude"] == 25.03
        assert result["longitude"] == 121.57

    @patch("src.tools.weather.requests.get")
    def test_geocode_city_not_found(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = {"results": []}
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = geocode_city("nonexistentcity")
        assert result["success"] is False
        assert result["error_type"] == "city_not_found"


class TestFetchWeather:
    """Tests for fetch_weather function"""

    @patch("src.tools.weather.requests.get")
    def test_fetch_weather_success(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = {
            "current": {
                "temperature_2m": 30.0,
                "precipitation_probability": 20,
                "weather_code": 0
            },
            "daily": {
                "precipitation_probability_max": [25]
            }
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        result = fetch_weather(25.03, 121.57)
        assert result["success"] is True
        assert result["temperature"] == 30.0
        assert result["precipitation_probability"] == 20
        assert result["weather_code"] == 0
        assert result["weather"] == "clear"

    @patch("src.tools.weather.requests.get")
    def test_fetch_weather_timeout(self, mock_get):
        import requests
        mock_get.side_effect = requests.exceptions.Timeout()

        result = fetch_weather(25.03, 121.57)
        assert result["success"] is False
        assert result["error_type"] == "timeout"


class TestWeatherCodeMap:
    """Tests for WEATHER_CODE_MAP"""

    def test_clear_codes(self):
        assert WEATHER_CODE_MAP[0] == "clear"
        assert WEATHER_CODE_MAP[1] == "clear"

    def test_cloudy_codes(self):
        assert WEATHER_CODE_MAP[2] == "cloudy"
        assert WEATHER_CODE_MAP[3] == "cloudy"
        assert WEATHER_CODE_MAP[45] == "cloudy"
        assert WEATHER_CODE_MAP[48] == "cloudy"

    def test_rain_codes(self):
        assert WEATHER_CODE_MAP[51] == "rain"
        assert WEATHER_CODE_MAP[61] == "rain"
        assert WEATHER_CODE_MAP[80] == "rain"

    def test_snow_codes(self):
        assert WEATHER_CODE_MAP[71] == "snow"
        assert WEATHER_CODE_MAP[85] == "snow"

    def test_thunderstorm_codes(self):
        assert WEATHER_CODE_MAP[95] == "thunderstorm"
        assert WEATHER_CODE_MAP[99] == "thunderstorm"

    def test_unknown_code_fallback(self):
        # Codes not in map should not crash - tested via get_weather
        assert WEATHER_CODE_MAP.get(999, "unknown") == "unknown"