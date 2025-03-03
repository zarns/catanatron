#!/usr/bin/env python
"""
Wrapper script for running catanatron with proper path setup.
This script ensures all necessary Python modules can be found.
"""

import os
import sys

# Add all necessary paths to Python path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, 'catanatron_core'))
sys.path.insert(0, os.path.join(BASE_DIR, 'catanatron_experimental'))
sys.path.insert(0, os.path.join(BASE_DIR, 'catanatron_rust'))

# Print debug information
print(f"Working directory: {os.getcwd()}")
print(f"Script location: {__file__}")
print(f"Base directory: {BASE_DIR}")
print("Python path:")
for path in sys.path:
    print(f"  {path}")

# Try to verify imports
try:
    import catanatron
    print("✓ catanatron module found")
except ImportError:
    print("✗ catanatron module not found")

try:
    import catanatron_rust
    print("✓ catanatron_rust module found")
except ImportError:
    print("✗ catanatron_rust module not found")

try:
    from catanatron_experimental.play import simulate
    print("✓ Successfully imported the simulation module")
except ImportError as e:
    print(f"✗ Failed to import simulation module: {e}")
    sys.exit(1)

# Run the simulation with command-line arguments
if __name__ == "__main__":
    # Remove the script name from sys.argv
    simulate() 