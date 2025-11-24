#!/usr/bin/env python3
"""
Run main.py with proper HuggingFace cache directory setup.
Uses /scratch/tknolast for model cache.
"""

import os
import sys
import subprocess
from pathlib import Path

# Use /scratch/tknolast for HuggingFace cache
cache_dir = Path("/scratch/tknolast/hf_cache")
cache_dir.mkdir(parents=True, exist_ok=True)

# Set environment variables BEFORE importing anything
os.environ['HF_HOME'] = str(cache_dir)
os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
os.environ['HF_DATASETS_CACHE'] = str(cache_dir)
os.environ['TORCH_HOME'] = str(cache_dir / "torch")
os.environ['HUGGINGFACE_HUB_CACHE'] = str(cache_dir)

# Disable symlinks and xet downloader which can cause permission issues
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'  # Disable xet downloader
os.environ['HUGGINGFACE_HUB_CACHE'] = str(cache_dir)  # Use standard HTTP downloads

print(f"HuggingFace cache directory: {cache_dir}")
print(f"Cache directory exists: {cache_dir.exists()}")
print(f"Cache directory writable: {os.access(cache_dir, os.W_OK)}")
print()

# Run main.py with environment variables passed through
print("Starting main.py...")
result = subprocess.run([sys.executable, "main.py"], cwd=Path.cwd(), env=os.environ.copy())
sys.exit(result.returncode)
