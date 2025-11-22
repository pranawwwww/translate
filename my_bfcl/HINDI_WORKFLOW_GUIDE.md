# Hindi Dataset Generation & BFCL Workflow Guide

## Overview

The my_bfcl codebase is designed to evaluate LLM function calling capabilities across multiple languages and data augmentation strategies. Here's how it works and how to generate Hindi datasets.

---

## Codebase Architecture

### 1. **Data Pipeline**

```
Base Dataset (English)
    ↓
[generate_translated.py] → Translate to Hindi
    ↓
Hindi Full/Partial Datasets
    ↓
[generate_synonym_dataset_hi.py] → Add synonyms
[generate_paraphrased_dataset_hi.py] → Add paraphrasing
    ↓
Hindi Variants: _syno, _para, _noisy, combinations
    ↓
[generate_translated.py] (main inference pipeline)
    ↓
Results: raw → json → post-processing → evaluation → scoring
```

### 2. **Key Files**

| File | Purpose |
|------|---------|
| `generate_translated.py` | Translates English → Hindi/Chinese; also runs main inference pipeline |
| `generate_synonym_dataset.py` | Creates synonym variants (Chinese only - need Hindi version) |
| `generate_paraphrased_dataset.py` | Creates paraphrased variants (Chinese only - need Hindi version) |
| `config.py` | Configuration: models, languages, translation options |
| `dataset/` | Input/output datasets |
| `result/` | Inference results (5 stages) |

---

## Step-by-Step Workflow: Generate Hindi Datasets

### Step 1: Generate Base Hindi Translations

**File:** `generate_translated.py` (already handles Hindi)

The file already includes Hindi configs:
```python
translate_configs = [
    TranslateConfig(language=Language.HINDI, option=TranslateOption.FULLY_TRANSLATED),
    TranslateConfig(language=Language.HINDI, option=TranslateOption.PARTIALLY_TRANSLATED),
]
```

**Run:**
```bash
python generate_translated.py
```

**Output:**
- `dataset/BFCL_v4_multiple_hi_full.json` - Fully translated to Hindi
- `dataset/BFCL_v4_multiple_hi_partial.json` - Partial (English terms kept)

**What it does:**
- Loads base dataset: `BFCL_v4_multiple.json`
- Loads ground truth: `dataset/possible_answer/BFCL_v4_multiple.json`
- For each test case, uses GPT-4o-mini to translate questions to Hindi
- Preserves function parameter names in partial translation
- Saves incrementally (resumable)

**API Used:** OpenAI GPT-4o-mini (via OPENAI_API_KEY)

---

### Step 2: Generate Hindi Synonym Variants

**File:** `generate_synonym_dataset_hi.py` (needs to be created)

This creates synonym-replaced versions of Hindi texts.

**What it does:**
- Loads Hindi dataset: `BFCL_v4_multiple_hi_full.json`
- Uses LLM to replace words with synonyms
- Keeps English words unchanged (parameter names)
- Creates: `BFCL_v4_multiple_hi_full_syno.json`

**Create this file with:**
```bash
# Will provide code below
```

---

### Step 3: Generate Hindi Paraphrased Variants

**File:** `generate_paraphrased_dataset_hi.py` (needs to be created)

This creates paraphrased versions of Hindi texts.

**What it does:**
- Loads Hindi dataset: `BFCL_v4_multiple_hi_full.json`
- Uses LLM to rephrase while keeping meaning
- Preserves English parameter names
- Creates: `BFCL_v4_multiple_hi_full_para.json`

**Create this file with:**
```bash
# Will provide code below
```

---

### Step 4: Generate Hindi Noisy Variants (Optional)

**File:** `revise_noise.py` (existing)

Creates datasets with intentional noise/errors. Can be applied to Hindi datasets if needed.

---

## Step-by-Step Workflow: Run BFCL Inference on Hindi

### Step 1: Configure Models for Hindi

**File:** `config.py`

Modify the `configs` list to include Hindi:

```python
# Add these configs
for model in [ApiModel.GPT_4O_MINI, ApiModel.CLAUDE_SONNET]:
    for translate_mode in [
        Translated(Language.HINDI, TranslateOption.FULLY_TRANSLATED),
        Translated(Language.HINDI, TranslateOption.FULLY_TRANSLATED_PROMPT_TRANSLATE),
    ]:
        for add_noise_mode in [AddNoiseMode.NO_NOISE, AddNoiseMode.SYNONYM, AddNoiseMode.PARAPHRASE]:
            configs.append(Config(model, translate_mode, add_noise_mode))
```

This creates configs for:
- Multiple models: GPT-4o-mini, Claude Sonnet, etc.
- Multiple Hindi variants: full, partial, with/without synonyms, paraphrasing
- Different options: with/without prompt translation

**Total configs:** ~6 models × 2 translate_modes × 3 noise_modes = ~36 configs

---

### Step 2: Run Main Inference Pipeline

**File:** `generate_translated.py` (main production pipeline)

The file already handles inference for any config in `config.py`.

**Run:**
```bash
python generate_translated.py
```

**What happens (5-stage pipeline):**

1. **Stage 1: Raw Inference** (`requires_inference_raw = True`)
   - Loads Hindi dataset
   - Creates model interface
   - Calls model.infer(functions, user_query)
   - Saves raw output: `result/inference_raw/BFCL_v4_multiple_gpt4o_mini_hi_f*.json`

2. **Stage 2: JSON Parsing** (`requires_inference_json = True`)
   - Parses raw output using model-specific parser
   - Converts to standardized JSON format
   - Saves: `result/inference_json/BFCL_v4_multiple_gpt4o_mini_hi_f*.json`

3. **Stage 3: Post-Processing** (`requires_post_processing = True`)
   - Optional LLM-based semantic matching
   - Replaces values if deemed equivalent
   - Saves: `result/post_processing/BFCL_v4_multiple_gpt4o_mini_hi_*.json`

4. **Stage 4: Evaluation** (`requires_evaluation = True`)
   - Compares with ground truth
   - Marks correct/incorrect
   - Saves: `result/evaluation/BFCL_v4_multiple_gpt4o_mini_hi_*.json`

5. **Stage 5: Scoring** (`requires_score = True`)
   - Calculates accuracy
   - Generates summary
   - Saves: `result/score/BFCL_v4_multiple_gpt4o_mini_hi_*.json`

**Configuration:**
```python
# In generate_translated.py
requires_inference_raw = True
requires_inference_json = True
requires_post_processing = True
requires_evaluation = True
requires_score = True
evaluation_caching = False
```

---

## Complete Workflow: From Start to Finish

### Timeline Example

```
1. Generate Base Hindi Translation (30 min - 1 hour)
   python generate_translated.py
   
2. Generate Hindi Synonyms (1-2 hours)
   python generate_synonym_dataset_hi.py
   
3. Generate Hindi Paraphrases (1-2 hours)
   python generate_paraphrased_dataset_hi.py
   
4. Run BFCL for 1 config (30 min - 1 hour per model)
   - Modify config.py to single model
   - python generate_translated.py
   
5. Scale to multiple configs (use SLURM for parallelization)
   - Edit config.py with multiple configs
   - sbatch run.slurm
   
6. Analyze results
   - Check result/score/*.json for accuracy metrics
```

---

## File Dependencies & Data Flow

```
dataset/BFCL_v4_multiple.json (base)
    ↓
generate_translated.py → BFCL_v4_multiple_hi_full.json
    ↓ (splits into)
    ├─ generate_synonym_dataset_hi.py → BFCL_v4_multiple_hi_full_syno.json
    └─ generate_paraphrased_dataset_hi.py → BFCL_v4_multiple_hi_full_para.json
    ↓
config.py (specify which datasets + models)
    ↓
generate_translated.py (main inference)
    ├─ Stage 1: inference_raw/
    ├─ Stage 2: inference_json/
    ├─ Stage 3: post_processing/
    ├─ Stage 4: evaluation/
    └─ Stage 5: score/
```

---

## Environment Setup

### Required .env File
```
OPENAI_API_KEY=sk_...          # For GPT-4o, Claude
DEEPSEEK_API_KEY=sk_...        # For DeepSeek (used in translate)
ANTHROPIC_API_KEY=sk_...       # For Claude
```

### Python Dependencies
```bash
pip install anthropic openai python-dotenv transformers torch
```

---

## Cost & Time Estimates

### Dataset Generation
- Base Translation (Hindi): 30-60 min, ~$0.50-1.00
- Synonyms per dataset: 1-2 hours, ~$0.30-0.50
- Paraphrasing per dataset: 1-2 hours, ~$0.30-0.50
- Total: 3-4 hours, ~$1.00-2.00

### Inference on 1000 test cases
- GPT-4o-mini: 5-10 min, $0.15-0.30
- Claude Sonnet: 10-20 min, $0.30-0.60
- DeepSeek: 5-10 min, $0.05-0.15
- Local Models (Granite): 20-30 min, $0 (compute)

### For 36 configs (6 models × 6 variants)
- Estimated: 6-12 hours total
- Cost: $10-20 for API calls
- Recommend: SLURM batch job for parallelization

---

## Key Configuration Options

### Translate Modes
- `FULLY_TRANSLATED`: Everything in Hindi
- `FULLY_TRANSLATED_PROMPT_TRANSLATE`: Hindi + ask model to use English for params
- `PARTIALLY_TRANSLATED`: Hindi but parameter names stay English

### Noise Modes
- `NO_NOISE`: Clean dataset
- `SYNONYM`: Words replaced with synonyms
- `PARAPHRASE`: Sentences rephrased

### Post-Processing Options
- `DONT_POST_PROCESS`: Use model output as-is
- `POST_PROCESS_DIFFERENT`: Strict parameter matching
- `POST_PROCESS_SAME`: Lenient (multilingual) matching

---

## Running with SLURM

**File:** `run.slurm`

```bash
#!/bin/bash
#SBATCH --job-name=bfcl_hindi
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --mem=64G

python generate_translated.py
```

**Submit:**
```bash
sbatch run.slurm
```

**Monitor:**
```bash
squeue -u $USER
tail -f slurm-*.out
```

---

## Troubleshooting

### Issue: FileNotFoundError for dataset
**Solution:** Ensure you've run `generate_translated.py` first to create Hindi datasets

### Issue: API rate limiting
**Solution:** Reduce batch size or add delays between calls

### Issue: OOM (Out of Memory)
**Solution:** Use smaller models (GPT-4o-mini instead of Claude Sonnet)

### Issue: Parse errors
**Solution:** Check `result/inference_raw/*.json` for actual model output format

---

## Output Structure

### Result Directory
```
result/
├── inference_raw/          # Raw model outputs (strings)
├── inference_json/         # Parsed to JSON format
├── post_processing/        # LLM-matched parameters
├── evaluation/             # Compared with ground truth
└── score/                  # Final accuracy scores
```

### File Naming Convention
```
BFCL_v4_multiple[_MODEL][_LANG][_TRANS_MODE][_NOISE].json

Examples:
  BFCL_v4_multiple_gpt4o_mini_hi_f.json      # GPT-4o Hindi full
  BFCL_v4_multiple_claude_sonnet_hi_par.json # Claude Hindi paraphrased
  BFCL_v4_multiple_granite_hi_fp.json        # Granite Hindi prompt+translate
```

---

## Next Steps

1. **Generate Hindi Synonym script:** (See code below)
2. **Generate Hindi Paraphrase script:** (See code below)
3. **Configure models in config.py**
4. **Run inference with generate_translated.py**
5. **Analyze results in result/score/**

