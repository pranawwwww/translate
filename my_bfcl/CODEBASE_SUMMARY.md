# My_BFCL Codebase - Complete Summary

## What Is This Project?

**my_bfcl** is a comprehensive evaluation framework for testing Large Language Models' (LLMs) ability to generate correct function calls in multiple languages.

**Key Goals:**
- Evaluate models across different languages (English, Chinese, Hindi)
- Test robustness with data augmentation (synonyms, paraphrasing)
- Compare model performance
- Identify multilingual gaps

---

## Architecture Overview

### Component Layers

```
┌─────────────────────────────────────────────────────┐
│  User Interface Layer (config.py)                   │
│  - Choose models, languages, augmentations          │
└────────────────────┬────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│  Data Pipeline Layer                                │
│  - generate_translated.py (translation + inference) │
│  - generate_synonym_dataset_hi.py (synonyms)        │
│  - generate_paraphrased_dataset_hi.py (paraphrasing)│
└────────────────────┬────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│  Processing Layers                                  │
│  - call_llm.py (API interactions)                   │
│  - parse_ast.py (output parsing)                    │
│  - post_processing.py (semantic matching)           │
│  - parse_dataset.py (data loading)                  │
└────────────────────┬────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│  Model Interface Layer (models/)                    │
│  - model_factory.py (instantiation)                 │
│  - *_interface.py (API & local models)              │
└────────────────────┬────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────┐
│  Results Layer (result/)                            │
│  - Stage 1: Raw outputs                             │
│  - Stage 2: Parsed JSON                             │
│  - Stage 3: Post-processed                          │
│  - Stage 4: Evaluated                               │
│  - Stage 5: Scored                                  │
└─────────────────────────────────────────────────────┘
```

### Data Flow

```
Base Dataset (English)
    ↓
[Translation]           → Hindi Full/Partial Datasets
    ↓
[Augmentation]          → Synonyms, Paraphrasing variants
    ↓
[Model Inference]       → Raw outputs from each model
    ↓
[Parsing]               → Convert to standardized JSON
    ↓
[Post-Processing]       → LLM-based semantic matching (optional)
    ↓
[Evaluation]            → Compare with ground truth
    ↓
[Scoring]               → Calculate accuracy metrics
```

---

## File Ecosystem

### Core Files

| File | Purpose | Dependencies |
|------|---------|--------------|
| `config.py` | Configuration center | None |
| `generate_translated.py` | Main pipeline: translate + infer | config.py, models/ |
| `generate_synonym_dataset_hi.py` | Create synonym variants | call_llm.py |
| `generate_paraphrased_dataset_hi.py` | Create paraphrased variants | call_llm.py |
| `call_llm.py` | API interactions | .env (API keys) |
| `parse_ast.py` | Parse model outputs | None |
| `post_processing.py` | Semantic matching | call_llm.py |
| `parse_dataset.py` | Data loading utilities | None |

### Directory Structure

```
my_bfcl/
├── Python Scripts
│   ├── main.py (legacy)
│   ├── generate_translated.py ⭐ (main entry)
│   ├── generate_synonym_dataset_hi.py ⭐ (new)
│   ├── generate_paraphrased_dataset_hi.py ⭐ (new)
│   ├── config.py ⭐ (configuration)
│   ├── call_llm.py
│   ├── parse_ast.py
│   ├── post_processing.py
│   └── parse_dataset.py
│
├── models/ (Model Interfaces)
│   ├── base.py
│   ├── model_factory.py
│   ├── gpt_4o_mini_interface.py
│   ├── claude_sonnet_interface.py
│   ├── claude_haiku_interface.py
│   ├── deepseek_chat_interface.py
│   ├── granite_3_1_8b_instruct_interface.py
│   ├── llama_3_1_interface.py
│   └── qwen2_5_interface.py
│
├── dataset/ (Input Data)
│   ├── BFCL_v4_multiple.json (base English)
│   ├── BFCL_v4_multiple_hi_full.json (Hindi)
│   ├── BFCL_v4_multiple_hi_full_syno.json (Hindi + synonyms)
│   ├── BFCL_v4_multiple_hi_full_para.json (Hindi + paraphrasing)
│   └── possible_answer/ (ground truth)
│
├── result/ (Output Results)
│   ├── inference_raw/
│   ├── inference_json/
│   ├── post_processing/
│   ├── evaluation/
│   └── score/ (FINAL RESULTS)
│
├── Documentation ⭐
│   ├── HINDI_QUICK_START.md
│   ├── HINDI_WORKFLOW_GUIDE.md
│   ├── HINDI_EXECUTION_GUIDE.md
│   ├── DEVELOPER_GUIDE.md
│   ├── README_REFACTORING.md
│   └── ... (other docs)
│
└── Configuration
    ├── .env (API keys - not in repo)
    ├── run.slurm (SLURM job script)
    └── activate_environment.sh
```

**⭐** = Key files for Hindi workflow

---

## How It Works: The Pipeline

### Stage 1: Hindi Dataset Generation

**File:** `generate_translated.py`

```python
# Input: BFCL_v4_multiple.json (English)
# Process: Translate each question to Hindi using GPT-4o-mini
# Output: 
#   - BFCL_v4_multiple_hi_full.json (full translation)
#   - BFCL_v4_multiple_hi_partial.json (partial: Hindi + English terms)
```

### Stage 2: Data Augmentation (Optional)

**Files:** 
- `generate_synonym_dataset_hi.py` - Replace words with synonyms
- `generate_paraphrased_dataset_hi.py` - Rephrase while keeping meaning

```python
# Input: BFCL_v4_multiple_hi_full.json
# Process: LLM-based transformation (synonyms or paraphrasing)
# Output: 
#   - BFCL_v4_multiple_hi_full_syno.json
#   - BFCL_v4_multiple_hi_full_para.json
```

### Stage 3: Model Inference Pipeline

**File:** `generate_translated.py` (main inference)

The pipeline runs 5 stages for each test case:

**Stage 1: Raw Inference**
```
Load test case → Create model interface → Call model → Get raw string
Save: result/inference_raw/
```

**Stage 2: JSON Parsing**
```
Take raw string → Parse function calls → Standardized JSON format
Save: result/inference_json/
```

**Stage 3: Post-Processing** (optional)
```
Take parsed JSON → Compare with ground truth params
For each mismatch: Use LLM to decide if semantically equivalent
Replace if match found (cached)
Save: result/post_processing/
```

**Stage 4: Evaluation**
```
Compare parsed result with ground truth
Mark: correct (True) or incorrect (False)
Save: result/evaluation/
```

**Stage 5: Scoring**
```
Count: total cases, correct cases
Calculate: accuracy = correct / total
Save: result/score/FINAL_RESULT.json
```

---

## Key Features

### 1. Multilingual Support
- English (base)
- Chinese (full + partial)
- Hindi (full + partial) ← NEW

### 2. Data Augmentation
- **Synonyms:** Words replaced with synonyms (maintains meaning)
- **Paraphrasing:** Sentences rephrased (maintains meaning)
- **Noisy:** Intentional errors injected

### 3. Multi-Model Support
- **API Models:** GPT-4o-mini, Claude Sonnet/Haiku, DeepSeek, Llama
- **Local Models:** Granite, Qwen (various sizes)

### 4. Translation Modes
- **Fully Translated:** Everything in target language
- **Partially Translated:** Technical terms in English, rest translated
- **Prompt Translate:** Translated + ask model to use English for params
- **Post-Process Options:** Semantic matching (strict vs. lenient)

### 5. Caching & Resumability
- Incremental processing (resume if interrupted)
- Caching of LLM results (avoid redundant calls)
- Efficient file I/O

---

## Complete Hindi Workflow

### Quick Command Reference

```bash
# 1. Generate Hindi translation (45 min, $1)
python generate_translated.py

# 2. Generate synonyms (1 hour, $0.30)
python generate_synonym_dataset_hi.py

# 3. Generate paraphrases (1 hour, $0.30)
python generate_paraphrased_dataset_hi.py

# 4. Configure models (edit config.py)
# Add configs for Hindi models

# 5. Run inference (5-30 min per config)
python generate_translated.py

# 6. View results
cat result/score/BFCL_v4_multiple_gpt4o_mini_hi_f.json
```

### Expected Output

```json
{
  "accuracy": 0.92,
  "total_cases": 1000,
  "correct_cases": 920
}
```

---

## Configuration Guide

### In `config.py`

Add Hindi configurations:

```python
# Single model test
configs.append(Config(
    model=ApiModel.GPT_4O_MINI,
    translate_mode=Translated(Language.HINDI, TranslateOption.FULLY_TRANSLATED),
    add_noise_mode=AddNoiseMode.NO_NOISE,
))

# Multiple models comparison
for model in [ApiModel.GPT_4O_MINI, ApiModel.CLAUDE_SONNET]:
    for noise in [AddNoiseMode.NO_NOISE, AddNoiseMode.SYNONYM]:
        configs.append(Config(model, translate_mode, noise))
```

### Translation Options

| Option | Description |
|--------|-------------|
| `FULLY_TRANSLATED` | All in Hindi |
| `FULLY_TRANSLATED_PROMPT_TRANSLATE` | Hindi + ask for English params |
| `FULLY_TRANSLATED_POST_PROCESS_DIFFERENT` | Hindi + strict matching |
| `FULLY_TRANSLATED_POST_PROCESS_SAME` | Hindi + lenient matching |

---

## Performance Characteristics

### Time Estimates
- Hindi translation: 45 min
- Synonyms per dataset: 1 hour
- Paraphrasing per dataset: 1 hour
- Inference (per config):
  - GPT-4o-mini: 5-10 min
  - Claude: 10-20 min
  - Local model: 20-30 min

### Cost Estimates
- Hindi translation: $0.50-1.00
- Synonyms per dataset: $0.30-0.50
- Paraphrasing per dataset: $0.30-0.50
- Inference (per config):
  - GPT-4o-mini: $0.15-0.30
  - Claude Sonnet: $0.30-0.60
  - Local models: $0 (compute only)

### Memory Usage
- Python runtime: 2-4 GB
- API models: Minimal GPU
- Granite: 16 GB GPU
- Qwen-72B: 40-50 GB GPU

---

## Documentation Files

### For Quick Start
- `HINDI_QUICK_START.md` - Copy-paste commands
- `HINDI_WORKFLOW_GUIDE.md` - Workflow overview

### For Detailed Understanding
- `HINDI_EXECUTION_GUIDE.md` - Step-by-step guide with examples
- `DEVELOPER_GUIDE.md` - Model interface technical details
- `README_REFACTORING.md` - Refactoring overview

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| FileNotFoundError | Run `generate_translated.py` first |
| API key errors | Check .env file has keys |
| Rate limiting | Wait 1 min, try again |
| Parse failures | Check `result/inference_raw/` for format |
| OOM errors | Use smaller models or API-only |
| Low accuracy | Verify translation quality |

---

## Success Checklist

✅ Environment setup (.env file with API keys)
✅ Hindi datasets generated (BFCL_v4_multiple_hi_*.json)
✅ Models configured in config.py
✅ Inference runs successfully
✅ Results saved to result/score/
✅ Accuracy scores generated
✅ Results compared across models

---

## Next Steps

1. **Understand:** Read `HINDI_QUICK_START.md`
2. **Setup:** Ensure .env with API keys
3. **Execute:** `python generate_translated.py` (translate)
4. **Augment:** `python generate_synonym_dataset_hi.py` (optional)
5. **Configure:** Add models to config.py
6. **Run:** `python generate_translated.py` (infer)
7. **Analyze:** Check `result/score/` for results

---

## Key Insights

### Why This Architecture?

1. **Modularity:** Each script does one thing well
2. **Resumability:** Can interrupt and resume without data loss
3. **Caching:** Avoid redundant expensive LLM calls
4. **Flexibility:** Mix-and-match datasets, models, options
5. **Scalability:** From 1 config to 100+ with SLURM

### Design Decisions

1. **Incremental Save:** Save after each item (resilience)
2. **Sequential Processing:** Easier debugging than parallel
3. **File-based Pipeline:** Inspectable intermediate results
4. **LLM-based Matching:** More accurate than string matching
5. **Standardized Format:** All models output same format

---

## Created Files (New for Hindi)

✨ **generate_synonym_dataset_hi.py** - Create Hindi synonym variants
✨ **generate_paraphrased_dataset_hi.py** - Create Hindi paraphrased variants
✨ **HINDI_QUICK_START.md** - Quick reference
✨ **HINDI_WORKFLOW_GUIDE.md** - Workflow overview
✨ **HINDI_EXECUTION_GUIDE.md** - Detailed step-by-step guide

---

## Summary

The my_bfcl codebase provides a production-grade framework for evaluating LLMs' function calling abilities across languages and augmentation strategies. With the new Hindi support, you can now:

1. Translate test cases to Hindi
2. Create augmented variants (synonyms, paraphrasing)
3. Run inference on multiple models
4. Compare accuracy across models and variants
5. Identify multilingual gaps and robustness

The architecture is built for scale, resumability, and debugging. Start with the quick start guide and follow the step-by-step instructions to generate results.

