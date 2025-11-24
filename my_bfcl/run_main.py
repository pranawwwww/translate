#!/usr/bin/env python3
"""
Run main.py with proper HuggingFace cache directory setup.
Uses /scratch/tknolast for model cache.
Sets environment variables BEFORE importing any HuggingFace modules.
"""

# SET ENVIRONMENT VARIABLES FIRST - BEFORE ANY IMPORTS
import os
import sys
from pathlib import Path

# Use /scratch/tknolast for HuggingFace cache
cache_dir = Path("/scratch/tknolast/hf_cache")
cache_dir.mkdir(parents=True, exist_ok=True)

# Set environment variables BEFORE importing transformers, torch, or huggingface_hub
os.environ['HF_HOME'] = str(cache_dir)
os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
os.environ['HF_DATASETS_CACHE'] = str(cache_dir)
os.environ['TORCH_HOME'] = str(cache_dir / "torch")
os.environ['HUGGINGFACE_HUB_CACHE'] = str(cache_dir)

# Disable xet downloader completely - use standard HTTP only
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Additional safety: disable all xet-related features
os.environ['HF_HUB_DOWNLOAD_USING_XET'] = '0'

print(f"HuggingFace cache directory: {cache_dir}")
print(f"Cache directory exists: {cache_dir.exists()}")
print(f"Cache directory writable: {os.access(cache_dir, os.W_OK)}")
print(f"HF_HUB_ENABLE_HF_TRANSFER: {os.environ.get('HF_HUB_ENABLE_HF_TRANSFER')}")
print()

# Now import and run main
print("Starting main.py...")
import main
