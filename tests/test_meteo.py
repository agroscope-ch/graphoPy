"""
Test suite for SOPRA meteorological data handling
"""

import pytest
import os
from pathlib import Path
from sopra import meteo


class TestStationInfo:
    """Tests for station information retrieval"""

    def test_get_station_info_valid(self):
        """Test retrieval of valid station information"""
        info = meteo.get_station_info("AIG")
        assert info is not None
        assert len(info) == 4
        assert info[0] == 7  # meteoswiss_nr
        assert info[1] == "AIG"
        assert info[2] == "Aigle"
        assert info[3] == "Aigle"

    def test_get_station_info_invalid(self):
        """Test retrieval with invalid station code"""
        info = meteo.get_station_info("INVALID")
        assert info is None

    def test_all_stations_available(self):
        """Test that all 14 stations are defined"""
        assert len(meteo.STATIONS) == 14

        # Test a few key stations
        for code in ["AIG", "BER", "CGI", "BAS"]:
            info = meteo.get_station_info(code)
            assert info is not None


class TestRequiredColumns:
    """Tests for required column definitions"""

    def test_required_columns_defined(self):
        """Test that required columns are properly defined"""
        assert hasattr(meteo, "REQUIRED_COLUMNS")
        assert len(meteo.REQUIRED_COLUMNS) == 5

        expected = ["Tagnr", "Stunde", "Tmit", "Strahlung", "Tbod_5cm"]
        assert meteo.REQUIRED_COLUMNS == expected


class TestPathFunctions:
    """Tests for path handling functions"""

    def test_get_default_pascal_reference_path(self):
        """Test default Pascal reference path retrieval"""
        path = meteo.get_default_pascal_reference_path()
        assert isinstance(path, str)
        assert "gfu_all_years.csv" in path

    def test_get_default_archive_path(self):
        """Test default archive path retrieval"""
        path = meteo.get_default_archive_path()
        assert isinstance(path, str)
        # Should contain platform-specific paths or environment variable


class TestMeteoFileValidation:
    """Tests for meteorological file validation"""

    def test_validate_meteo_file_nonexistent(self):
        """Test validation of non-existent file"""
        is_valid, message = meteo.validate_meteo_file("/nonexistent/file.std")
        assert not is_valid
        assert "error" in message.lower()


class TestDiscoverMeteoFile:
    """Tests for meteorological file discovery"""

    def test_discover_meteo_file_structure(self):
        """Test that discover_meteo_file returns correct structure"""
        # This will likely fail to find the file in test environment,
        # but we can test the function signature and error handling
        result = meteo.discover_meteo_file(
            year=2024,
            s_short="AIG",
            s_name="Aigle",
            s_name_internal="Aigle",
            archive_base="/nonexistent",
        )

        # Should return tuple of (dataframe/None, message)
        assert isinstance(result, tuple)
        assert len(result) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
