#!/bin/bash
# Set HuggingFace cache directory to /scratch/tknolast

export HF_HOME=/scratch/tknolast/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME
export HF_HUB_DISABLE_SYMLINKS_WARNING=1

# Ensure directory exists and is writable
mkdir -p $HF_HOME
chmod -R 755 $HF_HOME

echo "HuggingFace cache set to: $HF_HOME"
echo "Cache writable: $(test -w $HF_HOME && echo 'YES' || echo 'NO')"
echo "Running main.py..."
echo ""

# Run the main script with environment variables
python main.py
