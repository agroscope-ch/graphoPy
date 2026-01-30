# SOPRA Data Directory

This directory contains test data and outputs for the SOPRA model.

## Structure

```
data/
├── input/          # Input meteorological data
│   └── sopra_in/   # Standard format meteorological files (.std)
└── output/         # Model outputs and results
    ├── output_run_Pascal/  # Reference outputs from Pascal implementation
    └── output_run_Python/  # Python implementation outputs
```

## Input Data

The `input/sopra_in/` directory contains meteorological data files in standard format:
- Temperature readings
- Station-specific measurements
- Multi-year datasets (2004-2024)

Files are named following the pattern: `met{station}{year}.std`

## Output Data

Output directories contain:
- Model simulation results
- Population dynamics data
- Comparison data between implementations
