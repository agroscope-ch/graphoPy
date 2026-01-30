"""
SOPRA Meteorological Data Module
================================

Utilities for meteorological data  loading, validation, and processing
for the SOPRA model system.

This module provides functions to:
- Loading/discovering meteorological data files across different platforms
- Validate data file formats and completeness
- Process station metadata and configurations
- Handle cross-platform path resolution

Station Coverage:
    Supports 14 Swiss meteorological stations with hourly data
    including air temperature, solar radiation, and soil temperature.
"""

import os
import pandas as pd
import platform
from typing import Tuple, Optional, List

# Station definitions [meteoswiss_nr, short_name, long_name, internal_name]
STATIONS = [
    [7, 'AIG', 'Aigle', 'Aigle'],
    [1011, 'BER', 'Bern / Zollikofen', 'Bern'],
    [51, 'CGI', 'Nyon / Changins', 'Changins'],
    [40, 'BUS', 'Buchs / Aarau', 'Aarau'],
    [19, 'CHU', 'Chur', 'Chur'],
    [22, 'MAG', 'Magadino / Cadenazzo', 'Magadino'],
    [54, 'GUT', 'Güttingen', 'Guettingen'],
    [2, 'PAY', 'Payerne', 'Payerne'],
    [58, 'REH', 'Zürich / Affoltern', 'Reckenholz'],
    [21, 'SIO', 'Sion', 'Sion'],
    [29, 'STG', 'St. Gallen', 'St_Gallen'],
    [6, 'VAD', 'Vaduz', 'Vaduz'],
    [56, 'WAE', 'Wädenswil', 'Waedenswil'],
    [48, 'BAS', 'Basel / Binningen', 'Basel'],
]

# Required columns for SOPRA
REQUIRED_COLUMNS = ['Tagnr', 'Stunde', 'Tmit', 'Strahlung', 'Tbod_5cm']

def get_default_pascal_reference_path():
    """
    Get path to Pascal reference data file.
    
    First checks environment variable SOPRA_PASCAL_REFERENCE_PATH.
    If not set, falls back to default location.
    
    Returns
    -------
    str
        Path to Pascal reference data CSV file
    """
    env_path = os.environ.get('SOPRA_PASCAL_REFERENCE_PATH')
    if env_path and os.path.exists(env_path):
        return env_path
    
    # Default location
    default_path = "output_run_Pascal/gfu_all_years.csv"
    if os.path.exists(default_path):
        return default_path
    
    # Alternative location for package structure
    alt_path = os.path.join(os.path.dirname(__file__), "..", "..", "output_run_Pascal", "gfu_all_years.csv")
    return os.path.normpath(alt_path)


def get_default_archive_path():
    """
    Get default meteorological archive path.
    
    This function provides a portable way to locate meteorological data:
    1. First checks environment variable SOPRA_METEO_ARCHIVE_PATH
    2. Falls back to platform-specific paths (if they exist), specific to Agroscope network
    3. Finally uses current directory + 'sopra_in' as last resort
    
    For portable deployment, users should set the environment variable:
    export SOPRA_METEO_ARCHIVE_PATH="/path/to/your/meteo/archive"
    
    Returns
    -------
    str
        Path to meteorological archive directory
    """
    # Check environment variable first
    env_path = os.environ.get('SOPRA_METEO_ARCHIVE_PATH')
    if env_path and os.path.exists(env_path):
        return env_path
    
    # Platform-specific fallbacks (for backward compatibility)
    # These are example paths - users should set SOPRA_METEO_ARCHIVE_PATH environment variable
    system = platform.system().lower()
    if system == 'windows':
        # Example Windows network drive path - customize via environment variable
        fallback_path = r'O:\Data-Work\10_Support_Resources-PO\13_Meteo_Public\Archiv\Stundenwerte'
    else:
        # Example Linux mount path - customize via environment variable  
        fallback_path = '/mnt/agroscope/Data-Work/10_Support_Resources-PO/13_Meteo_Public/Archiv/Stundenwerte'
    
    # Only use fallback if it actually exists (won't work on other machines)
    if os.path.exists(fallback_path):
        return fallback_path
    
    # Final fallback to current directory + sopra_in
    current_dir_fallback = os.path.join(os.getcwd(), 'sopra_in')
    if os.path.exists(current_dir_fallback):
        return current_dir_fallback
    
    # Return current directory as last resort
    return os.getcwd()

# Get default archive path
DEFAULT_ARCHIVE_BASE = get_default_archive_path()


def discover_meteo_file(year: int, s_short: str, s_name: str, s_name_internal: str, 
                       archive_base: str = DEFAULT_ARCHIVE_BASE) -> Tuple[Optional[str], Optional[str]]:
    """
    Discover meteorological file using multiple naming patterns.
    
    Parameters
    ----------
    year : int
        Year to search for
    s_short : str
        Short station code (e.g., 'BER')
    s_name : str
        Full station name (e.g., 'Bern / Zollikofen')
    s_name_internal : str
        Internal station name (e.g., 'Bern')
    archive_base : str
        Base path to meteorological archive
        
    Returns
    -------
    Tuple[Optional[str], Optional[str]]
        (file_path, error_message) - file_path if found, error_message if not
    """
    year_folder = os.path.join(archive_base, str(year))
    
    if not os.path.exists(year_folder):
        return None, f"Year folder not found: {year_folder}"
    
    year_4digit = str(year)
    year_2digit = str(year)[2:]
    s_name_short = s_name_internal.split('_')[0]
    
    # Create search patterns (same as in the original script)
    patterns = [
        # Try full internal name first (most specific)
        f"{s_name_internal}_{year_4digit}.xlsx",
        f"{s_name_internal}_{year_4digit}.xls", 
        f"{s_name_internal}_{year_2digit}.xlsx",
        f"{s_name_internal}_{year_2digit}.xls",
        # Try full long name format (e.g., Bern_Zollikofen_2024.xlsx)
        f"{s_name.replace(' / ', '_').replace(' ', '_')}_{year_4digit}.xlsx",
        f"{s_name.replace(' / ', '_').replace(' ', '_')}_{year_4digit}.xls",
        f"{s_name.replace(' / ', '_').replace(' ', '_')}_{year_2digit}.xlsx",
        f"{s_name.replace(' / ', '_').replace(' ', '_')}_{year_2digit}.xls",
        # Then try short name (fallback for simplified naming)
        f"{s_name_short}_{year_4digit}.xlsx",
        f"{s_name_short}_{year_4digit}.xls", 
        f"{s_name_short}_{year_2digit}.xlsx",
        f"{s_name_short}_{year_2digit}.xls"
    ]
    
    # Search for files
    for pattern in patterns:
        potential_file = os.path.join(year_folder, pattern)
        if os.path.exists(potential_file):
            return potential_file, None
    
    return None, f"File not found (tried {len(patterns)} patterns)"


def validate_meteo_file(file_path: str) -> Tuple[bool, str]:
    """
    Validate that the meteorological file has required columns and data.
    
    Parameters
    ----------
    file_path : str
        Path to the meteorological file
        
    Returns
    -------
    Tuple[bool, str]
        (is_valid, message) - validation result and descriptive message
    """
    try:
        df = pd.read_excel(file_path)
        available_cols = set(df.columns)
        required_cols = set(REQUIRED_COLUMNS)
        
        if not required_cols.issubset(available_cols):
            missing_cols = required_cols - available_cols
            return False, f"Missing columns: {missing_cols}"
        
        if len(df) == 0:
            return False, "File is empty"
        
        non_null_counts = df[REQUIRED_COLUMNS].count()
        if non_null_counts.min() == 0:
            empty_cols = non_null_counts[non_null_counts == 0].index.tolist()
            return False, f"Empty columns: {empty_cols}"
        
        return True, f"Valid ({len(df)} rows)"
        
    except Exception as e:
        return False, f"Read error: {str(e)[:50]}..."


def test_all_meteo_files(test_years: range, archive_base: str = DEFAULT_ARCHIVE_BASE, 
                        verbose: bool = False) -> Tuple[int, int, List[str]]:
    """
    Test meteorological file discovery and validation for all stations and years.
    
    Parameters
    ----------
    test_years : range
        Range of years to test
    archive_base : str
        Base path to meteorological archive
    verbose : bool
        If True, print detailed progress; if False, only show problems
        
    Returns
    -------
    Tuple[int, int, List[str]]
        (successful_tests, total_tests, failed_tests)
    """
    total_tests = 0
    successful_tests = 0
    failed_tests = []
    
    for year in test_years:
        year_success = 0
        year_total = 0
        year_failures = []
        
        for s_nr, s_short, s_name, s_name_internal in STATIONS:
            total_tests += 1
            year_total += 1
            
            # Test file discovery
            found_file, discovery_error = discover_meteo_file(year, s_short, s_name, s_name_internal, archive_base)
            
            if found_file:
                # Test file validation
                is_valid, validation_msg = validate_meteo_file(found_file)
                
                if is_valid:
                    successful_tests += 1
                    year_success += 1
                else:
                    year_failures.append(f"{s_short}: {validation_msg}")
                    failed_tests.append(f"{year}-{s_short}: {validation_msg}")
            else:
                year_failures.append(f"{s_short}: {discovery_error}")
                failed_tests.append(f"{year}-{s_short}: {discovery_error}")
        
        # Display results for this year
        if verbose or year_success != year_total:
            if year_success == year_total:
                print(f"✅ {year}: All {year_total} stations successful")
            else:
                print(f"📅 {year}: {year_success}/{year_total} successful")
                for failure in year_failures:
                    print(f"   ❌ {failure}")
        elif year_success == year_total:
            # Only show successful years if verbose is False
            print(f"✅ {year}: All {year_total} stations successful")
    
    return successful_tests, total_tests, failed_tests


def get_station_info(station_code: str) -> Optional[Tuple[int, str, str, str]]:
    """
    Get station information by station code.
    
    Parameters
    ----------
    station_code : str
        Station code (e.g., 'BER')
        
    Returns
    -------
    Optional[Tuple[int, str, str, str]]
        (meteoswiss_nr, short_name, long_name, internal_name) or None if not found
    """
    for station in STATIONS:
        s_nr, s_short, s_name, s_name_internal = station
        if s_short == station_code:
            return s_nr, s_short, s_name, s_name_internal
    return None