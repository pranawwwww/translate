# Llama 3.1 Local GPU Setup

## Changes Made

### 1. **config.py**
   - Added `LLAMA_3_1_8B_INSTRUCT = "meta-llama/Llama-3.1-8B-Instruct"` to `LocalModel` enum
   - Added `LLAMA_3_1_70B_INSTRUCT = "meta-llama/Llama-3.1-70B-Instruct"` to `LocalModel` enum
   - Updated Hindi benchmark config loop to test **Llama 3.1 8B and 70B** instead of Qwen 2.5 14B
   - Configs now run: 2 models × 2 translate modes (full/partial) × 3 noise types = **12 total configs**

### 2. **models/llama_3_1_local_interface.py** (NEW FILE)
   - Created `Llama31LocalInterface` class for local Llama inference
   - Implements Llama 3.1 chat template: `<|begin_of_text|><|start_header_id|>role<|end_header_id|>...<|eot_id|>`
   - Handles both single and batch inference
   - Parses JSON tool calls from model output

### 3. **models/model_factory.py**
   - Added import: `from models.llama_3_1_local_interface import Llama31LocalInterface`
   - Added two cases to `_create_local_model_interface()`:
     ```python
     case LocalModel.LLAMA_3_1_8B_INSTRUCT:
         return Llama31LocalInterface(generator, model_id="meta-llama/Llama-3.1-8B-Instruct")
     case LocalModel.LLAMA_3_1_70B_INSTRUCT:
         return Llama31LocalInterface(generator, model_id="meta-llama/Llama-3.1-70B-Instruct")
     ```

### 4. **main.py**
   - Added model postfix cases:
     ```python
     case LocalModel.LLAMA_3_1_8B_INSTRUCT:
         model_postfix = "_llama3_1_8b"
     case LocalModel.LLAMA_3_1_70B_INSTRUCT:
         model_postfix = "_llama3_1_70b"
     ```

### 5. **call_llm.py**
   - Updated `HF_HOME` from `/scratch/tknolast/bfcl_temp/models_cache` to `/home/tknolast/.hf_cache`
   - This uses home directory instead of `/scratch` to avoid disk space issues

## How to Run

### Prerequisites
- Meta Llama weights access (requires HuggingFace token with access)
- Set HF token: `huggingface-cli login`

### Run Command
```bash
cd /scratch/tknolast/translate/my_bfcl
python main.py
```

### What Will Run
- **Llama 3.1 8B Instruct** on GPU:cuda:0
  - Hindi Full Translation (no noise, synonyms, paraphrases)
  - Hindi Partial Translation (no noise, synonyms, paraphrases)
  
- **Llama 3.1 70B Instruct** on GPU:cuda:1 (if available)
  - Hindi Full Translation (no noise, synonyms, paraphrases)
  - Hindi Partial Translation (no noise, synonyms, paraphrases)

### Storage Notes
- Models will cache to `/home/tknolast/.hf_cache` (needs ~20GB free in home)
- Clean up home directory first if needed:
  ```bash
  conda clean --all -y
  rm -rf ~/.cache/*
  ```

## Model Info

| Model | Size | Params | Disk | VRAM |
|-------|------|--------|------|------|
| Llama 3.1 8B | ~16GB | 8B | ~16GB | ~16GB |
| Llama 3.1 70B | ~140GB | 70B | ~140GB | ~40GB+ |

**Note:** The 70B model requires significant GPU memory. With A100 80GB, use `device_map="auto"` for offloading if needed.

## Output Files
Results will be saved to `result/` directory:
- `result/inference_raw/BFCL_v4_multiple_llama3_1_*_hi_*.json`
- `result/inference_json/BFCL_v4_multiple_llama3_1_*_hi_*.json`
- `result/post_processing/BFCL_v4_multiple_llama3_1_*_hi_*.json`
- `result/evaluation/BFCL_v4_multiple_llama3_1_*_hi_*.json`
- `result/score/BFCL_v4_multiple_llama3_1_*_hi_*.json`
