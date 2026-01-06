# SOPRA Python Package

## 🎯 **Overview**

This package contains a complete, standalone Python implementation of the SOPRA model for _Grapholita funebrana_ (plum fruit moth) population dynamics. The implementation has been translated from the original Pascal version and thoroughly validated against Pascal reference results.

## 📦 **Installation**

### **From Source (Recommended for Development)**

```bash
# Clone or download this repository
cd SOPRA_Python_Standalone

# Install in development mode with all dependencies
pip install -e .[all]

# Or install with specific optional dependencies
pip install -e .[jupyter]  # For Jupyter notebook support
pip install -e .[dev]      # For development tools
```

### **Basic Installation**

```bash
# Install only core dependencies
pip install -e .
```

### **Requirements**

- Python 3.8+
- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib >= 3.4.0

Optional dependencies:
- jupyter >= 1.0.0 (for notebook examples)
- networkx >= 2.6.0 (for visualization)

## 📦 **Package Structure**

```
SOPRA_Python_Standalone/
├── src/sopra/                    # Main package directory
│   ├── __init__.py              # Package initialization and exports
│   ├── core.py                  # Core SOPRA model functions
│   ├── meteo.py                 # Meteorological data utilities
│   └── cli.py                   # Command-line interface
├── pyproject.toml               # Package configuration and metadata
├── SOPRA_Demo.ipynb             # Main demonstration notebook
├── stations.txt                 # Station configuration
├── sopra_in/                    # Meteorological input data (2024)
│   ├── metaig24.std            # Aigle meteorological data
│   ├── metber24.std            # Bern meteorological data
│   ├── metcgi24.std            # Changins meteorological data
│   └── ...                     # All 13 Swiss stations (2024)
└── output_run_Pascal/          # Pascal reference data
    └── gfu_all_years.csv       # Pascal validation reference
```

## 🚀 **Quick Start**

### **Using the Package**

```python
import sopra
from sopra import core, meteo

# Initialize SOPRA model
constants = core.assign_const_and_var_gfune()
values = core.init_value_gfune()

# Load meteorological data  
import pandas as pd
meteo_df = pd.read_csv('sopra_in/metaig24.std', sep='\t', header=None,
                      names=['day', 'hour', 'temp_air', 'solar_rad', 'temp_soil'])

# Run simulation for one time step
result = core.update_gfune(
    values=values, day=1, hour=0, temp_air=10.0, 
    solar_rad=100.0, temp_soil=8.0, 
    curr_param=None, constants=constants
)

print(f"Simulation result: {result}")
```

### **Demo Notebook**

```bash
# Launch Jupyter notebook
jupyter notebook SOPRA_Demo.ipynb
```

The demo notebook provides:
- Complete walkthrough of the SOPRA model
- Meteorological data processing examples
- Population dynamics visualization
- Validation against Pascal reference results

### **Command Line Tools**

```bash
# Verify package integrity
sopra-verify
```


## 📊 **Data Format**

### **Input Data (.std files)**
- **Format**: Tab-separated values, no headers
- **Columns**: `day`, `hour`, `temp_air`, `solar_rad`, `temp_soil`
- **Units**: day (1-365), hour (0-23), temperature (°C), solar radiation (W/m²)
- **Resolution**: Hourly meteorological data

### **Output Data**
- **Population stages**: pupae, adults, eggs, larvae, diapause individuals
- **Temporal resolution**: Hourly time series with daily summaries
- **Validation metrics**: Comparison with Pascal reference results

## 🔬 **Model Description**

### **Scientific Background**
- **Species**: *Grapholita funebrana* (plum fruit moth)
- **Model type**: Temperature-dependent population dynamics with delayed response
- **Life cycle**: Overwintering → spring adults → first generation → summer adults → second generation → diapause

### **Key Features**
- **Temperature-dependent development**: Linear rate relationships
- **Delayed response models**: ODE system for stage transitions
- **Multi-generational lifecycle**: Two generations per year
- **Trunk temperature calculation**: Estimates bark temperature for pupae development

### **Model Functions**

| Category | Functions | Description |
|----------|-----------|-------------|
| **Core** | `update_gfune()` | Main simulation step |
| **Environment** | `rate()`, `get_trunk_temp()` | Temperature processing |
| **Initialization** | `assign_const_and_var_gfune()`, `init_value_gfune()` | Model setup |
| **Population** | `del_loop_fun()`, `block_delay_stage()` | Population dynamics |

## 🏢 **Included Stations (2024 Dataset)**

| Code | Station Name | Location |
|------|--------------|----------|
| AIG | Aigle | Western Switzerland |
| BAS | Basel / Binningen | Northern Switzerland |
| BER | Bern / Zollikofen | Central Switzerland |
| BUS | Buchs / Aarau | Central Switzerland |
| CGI | Nyon / Changins | Western Switzerland |
| GUT | Güttingen | Eastern Switzerland |
| MAG | Magadino / Cadenazzo | Southern Switzerland |
| PAY | Payerne | Western Switzerland |
| REH | Zürich / Affoltern | Central Switzerland |
| SIO | Sion | Valais |
| STG | St. Gallen | Eastern Switzerland |
| VAD | Vaduz | Liechtenstein |
| WAE | Wädenswil | Central Switzerland |

## ✅ **Validation Results**

The Python implementation has been thoroughly validated against the original Pascal version:

- **Precision**: Maximum differences < 1e-6 (excellent precision)
- **Coverage**: All 13 stations for 2024 validated successfully
- **Life stages**: All population stages match Pascal reference
- **Seasonal dynamics**: Correct timing of emergence, reproduction, and diapause

## 🛠 **Requirements**

### **Python Dependencies**
```bash
pip install pandas numpy matplotlib pathlib
```

### **Python Version**
- Python 3.7 or higher
- Tested with Python 3.8+

### **System Requirements**
- Windows/Linux/macOS
- Minimum 1GB RAM
- 100MB disk space

## ⚙️ **Configuration**

### **Environment Variables**

For portable deployment across different systems, configure these environment variables:

```bash
# Set path to meteorological data archive
export SOPRA_METEO_ARCHIVE_PATH="/path/to/your/meteo/archive"

# Set path to Pascal reference data (for validation)
export SOPRA_PASCAL_REFERENCE_PATH="/path/to/pascal/reference.csv"
```

**Windows:**
```cmd
set SOPRA_METEO_ARCHIVE_PATH=C:\path\to\your\meteo\archive
set SOPRA_PASCAL_REFERENCE_PATH=C:\path\to\pascal\reference.csv
```

### **Default Behavior**

If environment variables are not set, the system will:
1. Try platform-specific default paths (if they exist)
2. Fall back to relative paths in the current directory
3. Look for `sopra_in/` directory for meteorological data
4. Look for `output_run_Pascal/gfu_all_years.csv` for Pascal reference data

This ensures the package works out-of-the-box for development while remaining portable for deployment.

### **Network Environment Detection**

The SOPRA Demo notebook automatically detects the runtime environment:

**🏢 Agroscope Network Environment:**
- Accesses meteorological archive for comprehensive data processing
- Converts Excel files to .std format for validation
- Full functionality including historical data analysis

**🌐 External Environment:**
- Automatically skips archive-dependent operations
- Uses provided sample `.std` files in `sopra_in/` directory  
- Full model validation and simulation capabilities maintained
- External users can add their own `.std` files as needed

**For External Users:**
- ✅ **No additional setup required** - the notebook handles everything automatically
- ✅ **Complete functionality** available with included sample data
- ✅ **Easy data integration** - just add `.std` files to `sopra_in/` directory
- ✅ **Full validation** - Python vs Pascal comparison works with provided data

## 📖 **Usage Examples**

### **1. Single Station Simulation**
```python
# Load data and run simulation
meteo_data = read_meteo_file("sopra_in/metaig24.std")
results = run_sopra_model(meteo_data, "aig")
print(f"Simulated {len(results)} days for Aigle")
```

### **2. Population Analysis**
```python
# Analyze population peaks
population_cols = ['pupae_w', 'adults_w', 'eggs1', 'larvae1', 'diap']
for col in population_cols:
    peak_value = results[col].max()
    peak_day = results.loc[results[col].idxmax(), 'day']
    print(f"{col}: {peak_value:.6f} on day {peak_day}")
```

### **3. Validation Against Pascal**
```python
# Compare with Pascal reference
comparison = validate_python_vs_pascal("aig", 2024)
print("Validation completed!")
```

## � **API Documentation**

### **sopra.core Module**

The `sopra.core` module contains the main SOPRA model functions:

```python
from sopra import core

# Model initialization
constants = core.assign_const_and_var_gfune()  # Initialize constants
values = core.init_value_gfune()               # Initialize state variables

# Core simulation functions  
result = core.update_gfune(values, day, hour, temp_air, solar_rad, temp_soil)
trunk_temps = core.get_trunk_temp(day, temp_air, solar_rad)
dev_rate = core.rate(b1, b2, temp)
```

**Key Functions:**
- `assign_const_and_var_gfune()`: Initialize biological and physical constants
- `init_value_gfune()`: Initialize population values for all life stages
- `update_gfune()`: Execute one simulation time step
- `get_trunk_temp()`: Compute trunk temperature from meteorological data
- `rate()`: Calculate temperature-dependent development rates

### **sopra.meteo Module**

The `sopra.meteo` module provides meteorological data utilities:

```python
from sopra import meteo

# Station information
stations = meteo.STATIONS
station_info = meteo.get_station_info('AIG')

# File discovery and validation
file_path = meteo.discover_meteo_file(2024, 'AIG', 'Aigle', 'Aigle')
is_valid, message = meteo.validate_meteo_file(file_path)
```

**Key Functions:**
- `discover_meteo_file()`: Find meteorological data files
- `validate_meteo_file()`: Validate data file format and completeness  
- `get_station_info()`: Get station metadata by code
- `get_cross_platform_paths()`: Handle platform-specific paths

## 🧪 **Development and Testing**

### **Development Installation**

```bash
# Install with development tools
pip install -e .[dev]

# Run tests (when available)
pytest

# Code formatting
black src/sopra/
flake8 src/sopra/

# Type checking  
mypy src/sopra/
```

### **Building the Package**

```bash
# Build distribution packages
python -m build

# Install from built package
pip install dist/sopra-1.0.0-py3-none-any.whl
```

### **Custom Meteorological Data**
To use your own meteorological data:

1. Format data as tab-separated `.std` files
2. Place in `sopra_in/` directory
3. Use naming convention: `met{station}{year}.std`

### **Multiple Years**
To process multiple years:

1. Add historical `.std` files to `sopra_in/`
2. Use the validation functions for cross-year analysis
3. Compare results across different years

### **Custom Parameters**
To modify model parameters:

1. Edit constants in `assign_const_and_var_gfune()`
2. Adjust initial values in `init_value_gfune()`
3. Re-run simulations with new parameters

## 📚 **References**

- Original Pascal SOPRA implementation
- Swiss meteorological station network (MeteoSwiss)
- *Grapholita funebrana* biological parameters from experimental studies

## 💡 **Support**

### **Common Issues**

**Q: Import errors when loading functions**
A: Ensure all `.py` files are in the same directory as your notebook

**Q: Missing meteorological data**
A: Check that `.std` files exist in `sopra_in/` directory

**Q: Validation fails**
A: Verify Pascal reference data is available in `output_run_Pascal/`

### **File Structure Verification**
Run this code to verify package integrity:

```python
import os

required_files = [
    'grapholita_fun_utils.py',
    'sopra_meteo_utils.py', 
    'stations.txt',
    'sopra_in/',
    'output_run_Pascal/gfu_all_years.csv'
]

for file_path in required_files:
    status = "✅" if os.path.exists(file_path) else "❌"
    print(f"{status} {file_path}")
```

## 🧪 **Development and Testing**

### **Development Installation**

```bash
# Install with development tools
pip install -e .[dev]

# Run tests (when available)
pytest

# Code formatting
black src/sopra/
flake8 src/sopra/

# Type checking  
mypy src/sopra/
```

### **Building the Package**

```bash
# Build distribution packages
python -m build

# Install from built package
pip install dist/sopra-1.0.0-py3-none-any.whl
```

## 📊 **Validation & Verification**

### **Package Verification**

```bash
# Verify package integrity
sopra-verify

# Or from Python
python verify_package.py
```

### **Pascal Reference Validation**

The package includes validation against Pascal reference results in `output_run_Pascal/gfu_all_years.csv`. Maximum differences have been asssessed (error max < $10^{⁻2}$), demonstrating excellent precision.

## 🌍 **Station Coverage**

The package includes 2024 meteorological data for 13 Swiss stations:

| Code | Station | Records | Code | Station | Records |
|------|---------|---------|------|---------|---------|
| AIG | Aigle | 8,702 | PAY | Payerne | 8,657 |
| BAS | Basel | 8,702 | REH | Zürich/Affoltern | 8,702 |
| BER | Bern | 8,702 | SIO | Sion | 8,699 |
| BUS | Buchs/Aarau | 8,702 | STG | St. Gallen | 8,703 |
| CGI | Nyon/Changins | 8,616 | VAD | Vaduz | 8,703 |
| GUT | Güttingen | 8,638 | WAE | Wädenswil | 8,648 |
| MAG | Magadino | 8,703 | **Total** | **112,877** |

## 📝 Credit and licence

The original SOPRA source codes (in Pascal) were written by Benno Graf and Jörg Samietz (Agroscope, Switzerland). This includes but is not limited to the _Grapholita funebrana_ model. 

This Python implementation has been written by Matthieu Wilhelm (Agroscope, Switzerland).


Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg


---

## 🎉 **Ready to Use**

This package provides everything needed to run the SOPRA *Grapholita funebrana* model in Python. The implementation is validated, documented, and ready for operational use in pest management and research applications.

**Start with `SOPRA_Demo.ipynb` for a complete walkthrough!**