"""
Command Line Interface for SOPRA package
========================================

This module provides CLI entry points for common SOPRA operations.
"""

import sys
import os
from pathlib import Path

def verify_package():
    """
    Verify SOPRA package integrity.
    
    CLI entry point for package verification.
    """
    # Add the package root to path to import the verify module
    package_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(package_root))
    
    try:
        from verify_package import check_package_integrity
        return check_package_integrity()
    except ImportError as e:
        print(f"Error importing verification module: {e}")
        return False
    except Exception as e:
        print(f"Error during package verification: {e}")
        return False

if __name__ == "__main__":
    success = verify_package()
    sys.exit(0 if success else 1)