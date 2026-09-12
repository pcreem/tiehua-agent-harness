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
        """Test 臺 -> 台 normalization"""
        assert _normalize_city("臺東") == "台東"
        assert _normalize_city("台東") == "台東"
        assert _normalize_city("臺北市") == "台北市"

    def test_normalize_whitespace(self):
        """Test whitespace stripping"""
        assert _normalize_city("  Taitung  ") == "Taitung"
        assert _normalize_city(" Taitung ") == "Taitung"

    def test_normalize_passthrough(self):
        """Test non-Taiwan cities pass through"""
        assert _normalize_city("Tokyo") == "Tokyo"
        assert _normalize_city("Taipei") == "Taipei"


class TestMatchesCity:
    """Tests for _matches_city function"""

    def test_exact_match(self):
        attraction = {"city": "台東縣"}
        assert _matches_city(attraction, "台東") is True
        assert _matches_city(attraction, "Taitung") is True  # English matches via mapping

    def test_substring_match(self):
        attraction = {"city": "台東縣"}
        assert _matches_city(attraction, "台東") is True

    def test_english_city_match(self):
        """Test that English city name matches Chinese in dataset"""
        attraction = {"city": "台東縣"}
        assert _matches_city(attraction, "Taitung") is True

    def test_english_city_match_taipei(self):
        """Test English city matching for Taipei"""
        attraction = {"city": "台北市"}
        assert _matches_city(attraction, "Taipei") is True

    def test_no_city_in_attraction(self):
        attraction = {"city": ""}
        assert _matches_city(attraction, "台東") is False

    def test_none_city(self):
        attraction = {"city": None}
        assert _matches_city(attraction, "台東") is False


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
        result = search_attractions(city="Taitung", category="")
        assert result["success"] is True
        assert result["city"] == "Taitung"
        assert result["category"] is None
        assert result["count"] > 0

    def test_city_not_found(self):
        """Test unknown city returns city_not_found"""
        result = search_attractions(city="NonExistentCity123", category="accommodation")
        assert result["success"] is False
        assert result["error_type"] == "city_not_found"

    def test_category_not_found(self):
        """Test unknown category returns category_not_found"""
        result = search_attractions(city="Taitung", category="nonexistent_category")
        assert result["success"] is False
        assert result["error_type"] == "category_not_found"

    def test_successful_search_taitung_accommodation(self):
        """Test successful search for Taitung accommodation"""
        result = search_attractions(city="Taitung", category="accommodation")
        assert result["success"] is True
        assert result["city"] == "Taitung"
        assert result["category"] == "accommodation"
        assert result["count"] > 0
        assert result["source"] == "open-data-taitung"

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
        result = search_attractions(city="台東", category="accommodation")
        assert result["success"] is True
        assert result["count"] > 0

    def test_category_filtering(self):
        """Test that category filtering works correctly"""
        # Get all Taitung results
        all_result = search_attractions(city="Taitung", category="")
        assert all_result["success"] is True
        total_count = all_result["count"]

        # Get only accommodation
        accommodation_result = search_attractions(city="Taitung", category="accommodation")
        assert accommodation_result["success"] is True
        assert accommodation_result["count"] <= total_count

        # Each result should match the category
        for item in accommodation_result["results"]:
            assert item["category"] == "accommodation" or "accommodation" in [t.lower() for t in item.get("tags", [])]

    def test_source_provenance(self):
        """Test that source provenance is present in results"""
        result = search_attractions(city="Taitung", category="accommodation")
        assert result["success"] is True

        first = result["results"][0]
        assert first["source"] is not None
        assert isinstance(first["source"], str)
        assert len(first["source"]) > 0

    def test_metadata_present(self):
        """Test that dataset metadata is returned"""
        result = search_attractions(city="Taitung", category="accommodation")
        assert result["success"] is True
        assert "metadata" in result
        assert "dataset_source" in result["metadata"]
        assert "dataset_url" in result["metadata"]
        assert "dataset_last_updated" in result["metadata"]

    def test_accommodation_category(self):
        """Test accommodation category returns attractions"""
        result = search_attractions(city="Taitung", category="accommodation")
        assert result["success"] is True
        assert result["count"] > 0
        assert all(r["category"] == "accommodation" for r in result["results"])

    def test_activity_category(self):
        """Test activity category returns activities"""
        result = search_attractions(city="Taitung", category="activity")
        assert result["success"] is True
        assert result["count"] > 0
        assert all(r["category"] == "activity" for r in result["results"])


class TestDatasetSchema:
    """Tests for dataset schema integrity"""

    def test_all_required_fields_present(self):
        """Test that all results have required fields"""
        result = search_attractions(city="Taitung", category="accommodation")
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
        result = search_attractions(city="Taitung", category="")
        assert result["success"] is True
        # Verify dataset has real content
        assert result["metadata"]["dataset_total"] > 0
        assert len(result["results"]) > 0


class TestCategories:
    """Tests for category support"""

    def test_supported_categories_exist(self):
        """Test that documented categories exist in filtered dataset"""
        # From the dataset (filtered for 台東市 district)
        result = search_attractions(city="Taitung", category="")
        assert result["success"] is True

        categories_found = set(r["category"] for r in result["results"])
        # Filtered dataset for 台東市 has these categories
        expected_in_filtered_data = {"accommodation", "activity"}
        assert expected_in_filtered_data.issubset(categories_found)

    def test_category_names_match_constants(self):
        """Test that SUPPORTED_CATEGORIES includes expected values"""
        assert "accommodation" in SUPPORTED_CATEGORIES
        assert "activity" in SUPPORTED_CATEGORIES