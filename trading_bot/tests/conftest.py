"""
Test configuration file
"""
import os
import sys
from pathlib import Path

# Get the project root directory
ROOT_DIR = Path(__file__).parent.parent.absolute()

# Add the project root to Python path
sys.path.insert(0, str(ROOT_DIR))

print(f"Test configuration loaded")
print(f"Project root: {ROOT_DIR}")
print(f"Python path: {sys.path}")
