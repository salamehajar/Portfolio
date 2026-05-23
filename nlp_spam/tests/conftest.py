"""
Pytest configuration file.
Adds src directory to Python path.
"""

import sys
import os

# Get the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add src to Python path
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

#print(f"✓ Added {src_path} to Python path")