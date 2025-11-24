# GPU Local Models Setup Guide

This guide explains how to run Granite, Llama 3.1.8B, and Qwen models on GPU for Hindi BFCL evaluation.

## Models You're Running

**3 Local Models × 2 Translation Modes × 3 Noise Types = 18 Configurations**

1. **Granite 3.1 8B Instruct** - IBM's model
2. **Qwen 2.5 7B Instruct** - Alibaba's 7B model
3. **Qwen 2.5 14B Instruct** - Alibaba's 14B model

**Note:** Llama 3.1 8B is an API model (not local) - remove from GPU setup if needed.

---

## Prerequisites

### 1. GPU Requirements
- **NVIDIA GPU** with at least 16GB VRAM (24GB recommended)
- CUDA support enabled
- Driver installed

### 2. Python Environment
```bash
conda create -n bfcl-gpu python=3.10
conda activate bfcl-gpu
```

### 3. Install GPU Dependencies
```bash
# PyTorch with CUDA 13.0 support (YOUR SETUP)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# HuggingFace transformers and required packages
pip install transformers accelerate bitsandbytes
pip install anthropic openai python-dotenv

# For Granite model (might need special handling)
pip install ibm-granite

# Verify GPU support
python -c "import torch; print('GPU Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### Your GPU Setup
- **CUDA Version:** 13.0 ✓
- **GPUs:** 2x NVIDIA A100-SXM4-80GB ✓
- **Total Memory:** 160GB ✓
- **Status:** Ready for inference ✓

---

## Model Setup

### Option A: 4-bit Quantization (Recommended - Uses ~8GB per model)
```python
# In models/granite_interface.py or similar
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto",
)
```

### Option B: Half Precision (Uses ~16GB per model)
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)
```

### Option C: Full Precision (Uses ~32GB per model - not recommended)
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
)
```

---

## Running on GPU

### Step 1: Update config.py (Already Done ✓)
```python
for model in [
    LocalModel.GRANITE_3_1_8B_INSTRUCT,
    LocalModel.QWEN_2_5_7B_INSTRUCT,
    LocalModel.QWEN_2_5_14B_INSTRUCT,
]:
    # Hindi fully/partially translated with no noise/synonyms/paraphrasing
    # = 3 models × 2 modes × 3 noises = 18 configs
```

### Step 2: Check Model Interfaces
Verify these files exist and support GPU:
- `models/granite_interface.py` ✓
- `models/qwen2_5_interface.py` ✓

### Step 3: Run BFCL
```bash
# Run all 18 configs
python main.py
```

**Timeline:**
- Granite 8B: ~2-3 min per config × 6 configs = 12-18 min
- Qwen 7B: ~1-2 min per config × 6 configs = 6-12 min
- Qwen 14B: ~3-5 min per config × 6 configs = 18-30 min
- **Total: 45-60 minutes** (sequential) or **15-20 minutes** (if parallelized)

---

## Monitoring GPU Usage

### During Inference
```bash
# In another terminal - watch GPU usage
nvidia-smi -l 1  # Update every second

# Or for continuous monitoring
watch -n 1 nvidia-smi
```

### Expected GPU Memory
- **Granite 8B (4-bit):** ~8GB VRAM
- **Qwen 7B (4-bit):** ~7GB VRAM
- **Qwen 14B (4-bit):** ~14GB VRAM

---

## Troubleshooting

### Problem: CUDA Out of Memory
**Solution:** 
1. Use 4-bit quantization instead of half precision
2. Reduce batch size in main.py
3. Use smaller model (7B instead of 14B)

### Problem: Model not found / download fails
```bash
# Download model first manually
python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen2.5-7B-Instruct')"
```

### Problem: Slow inference on GPU
**Solution:** Ensure GPU is actually being used
```python
# Check in interface code
model = model.to("cuda")
# Verify: model.device should show 'cuda:0'
```

### Problem: "ImportError: cannot import name 'BitsAndBytesConfig'"
```bash
pip install bitsandbytes --upgrade
```

---

## Optimization Tips

### 1. Parallel Processing
The codebase already uses ThreadPoolExecutor. Local models on GPU can benefit from this.

### 2. Batch Processing
If available, use batch inference:
```python
# In interface: override infer_batch() for better parallelization
```

### 3. Memory Efficient Attention
```python
# In transformers 4.36+
model.config.use_flash_attention_2 = True
```

### 4. Pin Memory
```python
torch.cuda.empty_cache()  # Before each batch
```

---

## Results Location

After running, check results:
```bash
# All results in:
result/score/BFCL_v4_multiple_*.json

# Specific pattern:
result/score/BFCL_v4_multiple_granite_hi_*.json
result/score/BFCL_v4_multiple_qwen_hi_*.json
```

---

## Expected Accuracy (Hindi)

| Model | Typical Accuracy |
|-------|-----------------|
| Granite 8B | 30-50% |
| Qwen 7B | 40-60% |
| Qwen 14B | 50-70% |

(Lower than API models due to model size and Hindi complexity)

---

## Next Steps

1. ✅ Config.py updated with local models
2. ⏳ Install GPU dependencies (see Prerequisites)
3. ⏳ Verify GPU support: `nvidia-smi`
4. ⏳ Run: `python main.py`
5. ⏳ Monitor: `nvidia-smi -l 1`
6. ⏳ Check results: `result/score/`

---

## Need Help?

Check these files for model interface details:
- `models/model_factory.py` - Model loading
- `models/base.py` - Interface architecture
- `DEVELOPER_GUIDE.md` - Model integration guide
