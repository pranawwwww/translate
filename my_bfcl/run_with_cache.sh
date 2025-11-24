#!/bin/bash
# Set HuggingFace cache directory with write permissions
export HF_HOME=~/Desktop/translate/hf_cache
export TRANSFORMERS_CACHE=~/Desktop/translate/hf_cache
export HF_DATASETS_CACHE=~/Desktop/translate/hf_cache

# Ensure directory exists and is writable
mkdir -p $HF_HOME
chmod -R 755 $HF_HOME

echo "HuggingFace cache set to: $HF_HOME"
echo "Running main.py..."

# Run the main script
python main.py
