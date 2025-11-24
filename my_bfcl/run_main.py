#!/usr/bin/env python3
"""
Run main.py with proper HuggingFace cache directory setup.
This ensures models can be downloaded with proper permissions.
"""

import os
import sys
import subprocess
from pathlib import Path

# Set cache directories
cache_dir = Path.home() / "Desktop" / "translate" / "hf_cache"
cache_dir.mkdir(parents=True, exist_ok=True)

# Set environment variables
os.environ['HF_HOME'] = str(cache_dir)
os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
os.environ['HF_DATASETS_CACHE'] = str(cache_dir)
os.environ['TORCH_HOME'] = str(cache_dir / "torch")

print(f"HuggingFace cache directory: {cache_dir}")
print(f"Cache directory exists and is writable: {cache_dir.exists() and os.access(cache_dir, os.W_OK)}")
print()

# Run main.py
print("Starting main.py...")
result = subprocess.run([sys.executable, "main.py"], cwd=Path.cwd())
sys.exit(result.returncode)
