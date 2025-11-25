#!/usr/bin/env python3
"""
Check if all required modules are present
"""

import os
import sys
from pathlib import Path

def check_modules():
    src_dir = Path("src")
    required_modules = [
        "main.py",
        "preprocess.py", 
        "warp.py",
        "lane_fit.py",
        "temporal.py",
        "overlay.py",
        "metrics.py",
        "utils.py"
    ]
    
    print("Checking required modules...")
    
    missing_modules = []
    for module in required_modules:
        module_path = src_dir / module
        if module_path.exists():
            print(f"✓ {module}")
        else:
            print(f"✗ {module} - MISSING")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\nMissing modules: {missing_modules}")
        print("Please make sure all files are in the src/ directory")
        return False
    else:
        print("\nAll modules found!")
        return True

if __name__ == "__main__":
    check_modules()