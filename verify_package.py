#!/usr/bin/env python3
"""
SOPRA Python Standalone - Package Verification Script

This script verifies that the SOPRA Python standalone package is complete 
and ready to use. Run this script to check package integrity before using
the main demonstration notebook.
"""

import os
import sys
from pathlib import Path

def check_package_integrity():
    """
    Verify that all required files are present and accessible.
    """
    print("🔍 SOPRA PYTHON PACKAGE VERIFICATION")
    print("=" * 40)
    
    # Required files and directories
    required_items = [
        ('grapholita_fun_utils.py', 'file', 'Core SOPRA model functions'),
        ('sopra_meteo_utils.py', 'file', 'Meteorological utilities'),
        ('stations.txt', 'file', 'Station configuration'),
        ('README.md', 'file', 'Documentation'),
        ('SOPRA_Demo.ipynb', 'file', 'Demo notebook'),
        ('sopra_in/', 'dir', 'Meteorological input data directory'),
        ('output_run_Pascal/', 'dir', 'Pascal reference data directory'),
        ('output_run_Pascal/gfu_all_years.csv', 'file', 'Pascal validation data')
    ]
    
    missing_items = []
    present_items = []
    
    for item, item_type, description in required_items:
        if os.path.exists(item):
            present_items.append((item, description))
            if item_type == 'dir':
                file_count = len([f for f in os.listdir(item) if os.path.isfile(os.path.join(item, f))])
                print(f"   ✅ {item:<25} - {description} ({file_count} files)")
            else:
                file_size = os.path.getsize(item)
                size_str = f"{file_size:,} bytes" if file_size < 1024*1024 else f"{file_size/(1024*1024):.1f} MB"
                print(f"   ✅ {item:<25} - {description} ({size_str})")
        else:
            missing_items.append((item, description))
            print(f"   ❌ {item:<25} - {description} (MISSING)")
    
    print(f"\n📊 VERIFICATION SUMMARY")
    print("-" * 25)
    print(f"✅ Present: {len(present_items)}/{len(required_items)}")
    print(f"❌ Missing: {len(missing_items)}")
    
    if missing_items:
        print(f"\n⚠️  MISSING ITEMS:")
        for item, description in missing_items:
            print(f"   • {item} - {description}")
        return False
    
    # Check meteorological data files
    print(f"\n📁 METEOROLOGICAL DATA VERIFICATION")
    print("-" * 35)
    
    sopra_in_dir = Path("sopra_in")
    if sopra_in_dir.exists():
        std_files = list(sopra_in_dir.glob("*.std"))
        if std_files:
            print(f"📄 Found {len(std_files)} .std files:")
            
            station_count = 0
            total_records = 0
            
            for std_file in sorted(std_files):
                try:
                    with open(std_file, 'r') as f:
                        lines = f.readlines()
                        record_count = len(lines)
                        total_records += record_count
                        
                    # Extract station info
                    filename = std_file.stem
                    if filename.startswith('met') and filename.endswith('24'):
                        station_code = filename[3:-2].upper()
                        print(f"   📊 {station_code}: {record_count:,} records")
                        station_count += 1
                    else:
                        print(f"   📊 {filename}: {record_count:,} records")
                        
                except Exception as e:
                    print(f"   ❌ {std_file.name}: Error reading file - {e}")
            
            print(f"\n📈 Total: {station_count} stations, {total_records:,} meteorological records")
        else:
            print("❌ No .std files found in sopra_in directory")
            return False
    
    # Test imports
    print(f"\n🧪 IMPORT VERIFICATION")
    print("-" * 20)
    
    try:
        import grapholita_fun_utils as gfu
        func_count = len([f for f in dir(gfu) if not f.startswith('_')])
        print(f"   ✅ grapholita_fun_utils: {func_count} functions available")
    except ImportError as e:
        print(f"   ❌ grapholita_fun_utils: Import failed - {e}")
        return False
    except Exception as e:
        print(f"   ❌ grapholita_fun_utils: Error - {e}")
        return False
    
    try:
        import sopra_meteo_utils as smu
        print(f"   ✅ sopra_meteo_utils: Module loaded successfully")
    except ImportError as e:
        print(f"   ❌ sopra_meteo_utils: Import failed - {e}")
        return False
    except Exception as e:
        print(f"   ❌ sopra_meteo_utils: Error - {e}")
        return False
    
    # Test basic functionality
    print(f"\n🔧 FUNCTIONALITY TEST")
    print("-" * 20)
    
    try:
        # Test basic SOPRA functions
        constants = gfu.assign_const_and_var_gfune()
        values = gfu.init_value_gfune()
        
        print(f"   ✅ Model initialization: {len(constants)} constants, {len(values)} initial values")
        
        # Test with dummy data
        result = gfu.update_gfune(
            values=values,
            day=1, hour=0, temp_air=10.0, solar_rad=100.0, temp_soil=8.0,
            curr_param=None, constants=constants
        )
        
        if 'updated_values' in result and 'current_param' in result:
            print(f"   ✅ Model simulation: Single step executed successfully")
        else:
            print(f"   ❌ Model simulation: Unexpected output format")
            return False
        
    except Exception as e:
        print(f"   ❌ Model simulation: Error - {e}")
        return False
    
    # Final verdict
    print(f"\n🎉 PACKAGE VERIFICATION COMPLETE")
    print("=" * 35)
    print(f"✅ Package is complete and functional!")
    print(f"🚀 Ready to run SOPRA_Demo.ipynb")
    print(f"📚 See README.md for detailed usage instructions")
    
    return True

if __name__ == "__main__":
    print(__doc__)
    success = check_package_integrity()
    
    if success:
        print(f"\n💡 Next steps:")
        print(f"   1. Open SOPRA_Demo.ipynb in Jupyter")
        print(f"   2. Run all cells for complete demonstration")
        print(f"   3. Explore the validation results")
        sys.exit(0)
    else:
        print(f"\n❌ Package verification failed")
        print(f"   Please check missing files and try again")
        sys.exit(1)