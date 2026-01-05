# SOPRA Python Implementation - Standalone Package

## 🎯 **Overview**

This package contains a complete, standalone Python implementation of the SOPRA model for *Grapholita funebrana* (plum fruit moth) population dynamics. The implementation has been translated from the original Pascal version and thoroughly validated against Pascal reference results.

## 📦 **Package Contents**

```
SOPRA_Python_Standalone/
├── SOPRA_Demo.ipynb              # Main demonstration notebook
├── grapholita_fun_utils.py       # Core SOPRA model functions
├── sopra_meteo_utils.py          # Meteorological data utilities
├── stations.txt                  # Station configuration
├── README.md                     # This file
├── sopra_in/                     # Meteorological input data (2024)
│   ├── metaig24.std             # Aigle meteorological data
│   ├── metber24.std             # Bern meteorological data
│   ├── metcgi24.std             # Changins meteorological data
│   ├── ...                      # All 13 Swiss stations (2024)
└── output_run_Pascal/            # Pascal reference data
    └── gfu_all_years.csv        # Pascal validation reference
```

## 🚀 **Quick Start**

1. **Open the demo notebook**: `SOPRA_Demo.ipynb`
2. **Run all cells**: The notebook provides a complete walkthrough
3. **View results**: Population dynamics and validation results

### **Minimal Example**

```python
import grapholita_fun_utils as gf_utils
import pandas as pd

# Load meteorological data
meteo_df = pd.read_csv('sopra_in/metaig24.std', sep='\t', header=None,
                      names=['day', 'hour', 'temp_air', 'solar_rad', 'temp_soil'])

# Initialize SOPRA model
constants = gf_utils.assign_const_and_var_gfune()
values = gf_utils.init_value_gfune()
curr_param = None

# Run simulation for one time step
result = gf_utils.update_gfune(
    values=values, day=1, hour=0, temp_air=10.0, 
    solar_rad=100.0, temp_soil=8.0, 
    curr_param=curr_param, constants=constants
)
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

## 🔧 **Advanced Usage**

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

## 🎉 **Ready to Use**

This package provides everything needed to run the SOPRA *Grapholita funebrana* model in Python. The implementation is validated, documented, and ready for operational use in pest management and research applications.

**Start with `SOPRA_Demo.ipynb` for a complete walkthrough!**

## Credit and licence

The original SOPRA source codes (in Pascal) were written by Benno Graf and Jörg Samietz (Agroscope, Switzerland). This includes but is not limited to the _Grapholita funebrana_ model. 

This Python implementation has been written by Matthieu Wilhelm (Agroscope, Switzerland).


Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
