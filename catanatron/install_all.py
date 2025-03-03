#!/usr/bin/env python
"""
Script to install all catanatron packages in development mode.
Run this script to ensure all packages are properly installed.
"""

import os
import sys
import subprocess
import time

def print_step(message):
    """Print a step message with clear formatting"""
    print("\n" + "=" * 80)
    print(f" {message}")
    print("=" * 80)

def run_command(command, cwd=None):
    """Run a command and print its output"""
    print(f"> {command}")
    process = subprocess.Popen(
        command, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        cwd=cwd
    )
    
    # Print output in real-time
    for line in process.stdout:
        print(line.strip())
    
    process.wait()
    return process.returncode

def main():
    # Get the base directory (repository root)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # List of packages to install in order
    packages = [
        "catanatron_core",
        "catanatron_experimental",
        "catanatron_gym",
        "catanatron_server",
    ]
    
    # Install Rust package first with maturin
    print_step("Installing catanatron_rust with maturin")
    rust_dir = os.path.join(base_dir, "catanatron_rust")
    result = run_command("maturin develop", cwd=rust_dir)
    if result != 0:
        print("Failed to install catanatron_rust")
        sys.exit(1)
    
    # Install each package in development mode
    for package in packages:
        package_dir = os.path.join(base_dir, package)
        if os.path.exists(package_dir):
            print_step(f"Installing {package}")
            result = run_command(f"pip install -e .", cwd=package_dir)
            if result != 0:
                print(f"Failed to install {package}")
                sys.exit(1)
        else:
            print(f"Package directory not found: {package_dir}")
    
    # Verify imports
    print_step("Verifying imports")
    
    try:
        import catanatron
        print("✓ catanatron module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import catanatron: {e}")
    
    try:
        import catanatron_rust
        print("✓ catanatron_rust module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import catanatron_rust: {e}")
    
    # Final success message
    print_step("Installation completed")
    print("You can now run 'catanatron-play --players=R,R --rust' to test the installation")

if __name__ == "__main__":
    main() 