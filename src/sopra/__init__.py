"""
SOPRA
========================================================

A Python implementation of the SOPRA model for Grapholita funebrana 
(plum fruit moth) population dynamics modeling and risk assessment.

This package provides:
- Core SOPRA model functions for insect population dynamics
- Meteorological data processing utilities
- Validation and verification tools

Usage:
    >>> import sopra
    >>> from sopra import core, meteo
    >>> 
    >>> # Initialize model
    >>> constants = core.assign_const_and_var_gfune()
    >>> values = core.init_value_gfune()
    >>> 
    >>> # Run simulation
    >>> result = core.update_gfune(values, day, hour, temp_air, solar_rad, temp_soil)
"""

__version__ = "1.0.0"
__author__ = "Matthieu Wilhelm"
__email__ = "matthieu.wilhelm@agroscope.admin.ch"

# Import main modules
from . import core
from . import meteo

# Convenience imports for common functions
from .core import (
    assign_const_and_var_gfune,
    init_value_gfune,
    update_gfune,
    get_trunk_temp,
    rate
)

from .meteo import (
    STATIONS,
    get_default_pascal_reference_path,
    get_default_archive_path,
    discover_meteo_file,
    validate_meteo_file,
    get_station_info
)

__all__ = [
    'core',
    'meteo', 
    'assign_const_and_var_gfune',
    'init_value_gfune',
    'update_gfune',
    'get_trunk_temp',
    'rate',
    'STATIONS',
    'get_default_pascal_reference_path',
    'get_default_archive_path',
    'discover_meteo_file',
    'validate_meteo_file',
    'get_station_info'
]