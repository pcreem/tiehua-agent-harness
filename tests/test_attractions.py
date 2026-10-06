"""Unit tests for tourism attraction search tool"""

import json
import os
from pathlib import Path
import pytest
from src.tools.attractions import (
    search_attractions,
    _normalize_city,
    _matches_city,
    _matches_category,
    SUPPORTED_CATEGORIES,
)


class TestNormalizeCity:
    """Tests for _normalize_city function"""

    def test_normalize_tai_char(self):
        """Test 臺 -> 台 normalization (input normalization)"""
        # Input normalization: 台北 -> 臺北市
        assert _normalize_city("台北") == "臺北市"
        assert _normalize_city("臺北") == "臺北市"
        assert _normalize_city("台中") == "臺中市"

    def test_normalize_whitespace(self):
        """Test whitespace stripping and English name mapping"""
        assert _normalize_city("  Taipei  ") == "臺北市"
        assert _normalize_city(" Taitung ") == "臺東縣"

    def test_normalize_passthrough(self):
        """Test standard city names map to standard Chinese"""
        assert _normalize_city("Taipei") == "臺北市"
        assert _normalize_city("Taitung") == "臺東縣"
        assert _normalize_city("Kaohsiung") == "高雄市"
        assert _normalize_city("臺北市") == "臺北市"


class TestMatchesCity:
    """Tests for _matches_city function"""

    def test_exact_match(self):
        # New dataset uses full city names like "臺東縣"
        attraction = {"city": "臺東縣"}
        assert _matches_city(attraction, "台東") is True
        assert _matches_city(attraction, "Taitung") is True  # English matches via mapping

    def test_substring_match(self):
        attraction = {"city": "臺東縣"}
        assert _matches_city(attraction, "台東") is True

    def test_english_city_match(self):
        """Test that English city name matches Chinese in dataset"""
        attraction = {"city": "臺東縣"}
        assert _matches_city(attraction, "Taitung") is True

    def test_english_city_match_taipei(self):
        """Test English city matching for Taipei"""
        attraction = {"city": "臺北市"}
        assert _matches_city(attraction, "Taipei") is True

    def test_no_city_in_attraction(self):
        attraction = {"city": ""}
        assert _matches_city(attraction, "台東") is False

    def test_none_city(self):
        attraction = {"city": None}
        assert _matches_city(attraction, "台東") is False

    def test_district_match(self):
        """Test matching by district name"""
        attraction = {"city": "臺北市", "district": "大同區"}
        assert _matches_city(attraction, "大同區") is True


class TestMatchesCategory:
    """Tests for _matches_category function"""

    def test_exact_match(self):
        attraction = {"category": "accommodation", "tags": ["accommodation"]}
        assert _matches_category(attraction, "accommodation") is True

    def test_case_insensitive(self):
        attraction = {"category": "ACCOMMODATION", "tags": []}
        assert _matches_category(attraction, "accommodation") is True

    def test_no_match(self):
        attraction = {"category": "activity", "tags": ["activity"]}
        assert _matches_category(attraction, "accommodation") is False

    def test_tag_match(self):
        attraction = {"category": "hotel", "tags": ["accommodation"]}
        assert _matches_category(attraction, "accommodation") is True

    def test_no_category(self):
        attraction = {"category": "", "tags": []}
        assert _matches_category(attraction, "accommodation") is False


class TestSearchAttractions:
    """Tests for search_attractions function"""

    def test_invalid_input_empty_city(self):
        """Test empty city returns invalid_input error"""
        result = search_attractions(city="", category="accommodation")
        assert result["success"] is False
        assert result["error_type"] == "invalid_input"

    def test_invalid_input_whitespace_city(self):
        """Test whitespace-only city returns invalid_input error"""
        result = search_attractions(city="   ", category="accommodation")
        assert result["success"] is False
        assert result["error_type"] == "invalid_input"

    def test_invalid_input_empty_category(self):
        """Test empty category returns results for city only"""
        result = search_attractions(city="Taipei", category="")
        assert result["success"] is True
        assert result["city"] == "Taipei"
        assert result["category"] is None
        assert result["count"] > 0

    def test_city_not_found(self):
        """Test unknown city returns city_not_found"""
        result = search_attractions(city="NonExistentCity123", category="accommodation")
        assert result["success"] is False
        assert result["error_type"] == "city_not_found"

    def test_category_not_found(self):
        """Test unknown category returns category_not_found"""
        result = search_attractions(city="Taipei", category="nonexistent_category")
        assert result["success"] is False
        assert result["error_type"] == "category_not_found"

    def test_successful_search_taipei_sightseeing(self):
        """Test successful search for Taipei sightseeing"""
        result = search_attractions(city="Taipei", category="sightseeing")
        assert result["success"] is True
        assert result["city"] == "Taipei"
        assert result["category"] == "sightseeing"
        assert result["count"] > 0
        assert result["source"] == "taiwan-tourism-bureau"

        # Check result structure
        first_result = result["results"][0]
        assert "id" in first_result
        assert "name" in first_result
        assert "category" in first_result
        assert "description" in first_result
        assert "source" in first_result
        assert "source_url" in first_result
        assert "last_updated" in first_result

    def test_successful_search_chinese_city(self):
        """Test successful search with Chinese city name"""
        result = search_attractions(city="台北", category="sightseeing")
        assert result["success"] is True
        assert result["count"] > 0

    def test_category_filtering(self):
        """Test that category filtering works correctly"""
        # Get all Taipei results
        all_result = search_attractions(city="Taipei", category="")
        assert all_result["success"] is True
        total_count = all_result["count"]

        # Get only sightseeing
        sightseeing_result = search_attractions(city="Taipei", category="sightseeing")
        assert sightseeing_result["success"] is True
        assert sightseeing_result["count"] <= total_count

        # Each result should match the category
        for item in sightseeing_result["results"]:
            assert item["category"] == "sightseeing" or "sightseeing" in [t.lower() for t in item.get("tags", [])]

    def test_source_provenance(self):
        """Test that source provenance is present in results"""
        result = search_attractions(city="Taipei", category="sightseeing")
        assert result["success"] is True

        first = result["results"][0]
        assert first["source"] is not None
        assert isinstance(first["source"], str)
        assert len(first["source"]) > 0

    def test_metadata_present(self):
        """Test that dataset metadata is returned"""
        result = search_attractions(city="Taipei", category="sightseeing")
        assert result["success"] is True
        assert "metadata" in result
        assert "dataset_source" in result["metadata"]
        assert "dataset_url" in result["metadata"]
        assert "dataset_last_updated" in result["metadata"]
        assert "provider_id" in result["metadata"]

    def test_sightseeing_category(self):
        """Test sightseeing category returns attractions"""
        result = search_attractions(city="Taipei", category="sightseeing")
        assert result["success"] is True
        assert result["count"] > 0
        # All results should match the category (either directly or via tags)
        for r in result["results"]:
            assert r["category"] == "sightseeing" or "sightseeing" in [t.lower() for t in r.get("tags", [])]

    def test_activity_category(self):
        """Test activity category returns activities"""
        result = search_attractions(city="Taipei", category="activity")
        assert result["success"] is True
        assert result["count"] > 0
        # All results should match the category (either directly or via tags)
        for r in result["results"]:
            assert r["category"] == "activity" or "activity" in [t.lower() for t in r.get("tags", [])]

    def test_all_cities_covered(self):
        """Test that multiple cities return results"""
        for city in ["Taipei", "Kaohsiung", "Taichung", "Tainan", "Hualien"]:
            result = search_attractions(city=city, category="")
            assert result["success"] is True, f"City {city} failed: {result}"
            assert result["count"] > 0, f"City {city} has 0 results"


class TestDatasetSchema:
    """Tests for dataset schema integrity"""

    def test_all_required_fields_present(self):
        """Test that all results have required fields"""
        result = search_attractions(city="Taipei", category="sightseeing")
        assert result["success"] is True

        required_fields = {
            "id", "name", "city", "district", "category",
            "tags", "description", "address",
            "latitude", "longitude",
            "family_suitability", "indoor_outdoor",
            "source", "source_url", "last_updated"
        }

        for item in result["results"]:
            assert required_fields.issubset(item.keys())

    def test_no_hardcoded_data(self):
        """Test that data comes from the dataset file, not hardcoded"""
        result = search_attractions(city="Taipei", category="")
        assert result["success"] is True
        # Verify dataset has real content
        assert result["metadata"]["dataset_total"] > 6000  # 6,176 total
        assert len(result["results"]) > 0

    def test_coordinates_present(self):
        """Test that all results have valid coordinates"""
        result = search_attractions(city="Taipei", category="sightseeing")
        assert result["success"] is True

        for item in result["results"]:
            assert item["latitude"] is not None
            assert item["longitude"] is not None
            assert -90 <= item["latitude"] <= 90
            assert -180 <= item["longitude"] <= 180


class TestCategories:
    """Tests for category support"""

    def test_supported_categories_exist(self):
        """Test that documented categories exist in dataset"""
        result = search_attractions(city="Taipei", category="")
        assert result["success"] is True

        categories_found = set(r["category"] for r in result["results"])
        # Taipei has these categories (service not present in Taipei)
        expected_in_taipei = {"accommodation", "activity", "sightseeing", "food", "nature"}
        assert expected_in_taipei.issubset(categories_found)

    def test_category_names_match_constants(self):
        """Test that SUPPORTED_CATEGORIES includes expected values"""
        assert "accommodation" in SUPPORTED_CATEGORIES
        assert "activity" in SUPPORTED_CATEGORIES
        assert "sightseeing" in SUPPORTED_CATEGORIES
        assert "food" in SUPPORTED_CATEGORIES
        assert "nature" in SUPPORTED_CATEGORIES
        assert "service" in SUPPORTED_CATEGORIES


class TestMultilingualSupport:
    """Tests for multilingual support"""

    def test_language_parameter_accepted(self):
        """Test that language parameter is accepted"""
        for lang in ["zh-TW", "en", "ja"]:
            result = search_attractions(city="Taipei", category="sightseeing", language=lang)
            assert result["success"] is True
            assert result["language"] == lang