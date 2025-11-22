# Hindi Dataset Generation & BFCL Execution Guide

## Quick Start Summary

```bash
# Step 1: Generate base Hindi translation (30-60 min)
python generate_translated.py

# Step 2: Generate Hindi synonyms (1-2 hours)
python generate_synonym_dataset_hi.py

# Step 3: Generate Hindi paraphrases (1-2 hours)
python generate_paraphrased_dataset_hi.py

# Step 4: Configure models in config.py (add Hindi configs)

# Step 5: Run BFCL inference (time varies by model)
python generate_translated.py

# Step 6: Check results
ls result/score/BFCL_v4_multiple_*_hi_*.json
```

---

## Complete Understanding of the Codebase

### What is BFCL?

**BFCL** = Berkeley Function Calling Leaderboard - a benchmark to evaluate LLMs' ability to generate correct function calls.

**Input:** 
- Test cases with function definitions and user questions
- Ground truth: correct function calls and parameters

**Output:**
- Accuracy scores showing how well each model performs
- Broken down by language, augmentation (synonyms, paraphrasing), translation mode

### Core Workflow: 5-Stage Pipeline

```
User Question + Function Definitions
        ↓
[Stage 1] Raw Inference
    Model generates output (string)
    Saved: result/inference_raw/
        ↓
[Stage 2] JSON Parsing
    Parse string to structured format
    Saved: result/inference_json/
        ↓
[Stage 3] Post-Processing (optional)
    LLM-based semantic matching of parameters
    Saved: result/post_processing/
        ↓
[Stage 4] Evaluation
    Compare with ground truth
    Mark correct/incorrect
    Saved: result/evaluation/
        ↓
[Stage 5] Scoring
    Calculate accuracy
    Generate summary report
    Saved: result/score/
```

### Key Files & Their Roles

| File | Purpose | When to Run | Inputs | Outputs |
|------|---------|------------|--------|---------|
| `generate_translated.py` | Translate + Main Pipeline | Stage 1 (translate), Stage 4-5 (infer) | `BFCL_v4_multiple.json` | `_hi_full.json`, `_hi_partial.json`, then results |
| `generate_synonym_dataset_hi.py` | Add synonyms | Stage 2 (dataset prep) | `BFCL_v4_multiple_hi_*.json` | `BFCL_v4_multiple_hi_*_syno.json` |
| `generate_paraphrased_dataset_hi.py` | Add paraphrasing | Stage 2 (dataset prep) | `BFCL_v4_multiple_hi_*.json` | `BFCL_v4_multiple_hi_*_para.json` |
| `config.py` | Configuration | Before running | Models, languages, options | Specifies what to run |

---

## Step-by-Step Execution

### Phase 1: Dataset Preparation (Hindi Translation)

#### What it does:
- Takes English test cases from `BFCL_v4_multiple.json`
- Uses GPT-4o-mini to translate questions to Hindi
- Creates two variants:
  - **Full translation**: Everything in Hindi
  - **Partial translation**: Hindi with English parameter names intact

#### How to run:
```bash
python generate_translated.py
```

#### Configuration (in `generate_translated.py`):
```python
translate_configs: list[TranslateConfig] = [
    TranslateConfig(language=Language.HINDI, option=TranslateOption.FULLY_TRANSLATED),
    TranslateConfig(language=Language.HINDI, option=TranslateOption.PARTIALLY_TRANSLATED),
    # Plus Chinese configs
]
```

#### Output:
```
dataset/BFCL_v4_multiple_hi_full.json       # Full Hindi translation
dataset/BFCL_v4_multiple_hi_partial.json    # Hindi with English terms
```

#### Time: 30-60 minutes
#### Cost: ~$0.50-1.00 (GPT-4o-mini)

---

### Phase 2a: Generate Hindi Synonyms

#### What it does:
- Takes Hindi questions
- Replaces words with synonyms (keeps English parameters unchanged)
- Creates variant datasets for evaluation

#### How to run:
```bash
python generate_synonym_dataset_hi.py
```

#### Configuration (in `generate_synonym_dataset_hi.py`):
```python
translated = True  # Keep English words (parameter names)
postfix_to_generate = [
    "_hi_full",      # Generate for full Hindi dataset
    "_hi_partial",   # Generate for partial Hindi dataset
]
```

#### Output:
```
dataset/BFCL_v4_multiple_hi_full_syno.json    # Full Hindi with synonyms
dataset/BFCL_v4_multiple_hi_partial_syno.json # Partial Hindi with synonyms
```

#### Time: 1-2 hours
#### Cost: ~$0.30-0.50 per dataset

---

### Phase 2b: Generate Hindi Paraphrases

#### What it does:
- Takes Hindi questions
- Rephrases while keeping meaning identical
- Creates another variant for evaluation

#### How to run:
```bash
python generate_paraphrased_dataset_hi.py
```

#### Configuration (in `generate_paraphrased_dataset_hi.py`):
```python
translated = True  # Keep English words
postfix_to_generate = [
    "_hi_full",      # Generate for full Hindi dataset
    "_hi_partial",   # Generate for partial Hindi dataset
]
```

#### Output:
```
dataset/BFCL_v4_multiple_hi_full_para.json    # Full Hindi paraphrased
dataset/BFCL_v4_multiple_hi_partial_para.json # Partial Hindi paraphrased
```

#### Time: 1-2 hours
#### Cost: ~$0.30-0.50 per dataset

---

### Phase 3: Configure Models for Evaluation

#### What it does:
- Specify which models to test
- Specify which Hindi datasets to use
- Specify which augmentations (synonyms, paraphrasing, noise)

#### How to configure (in `config.py`):

```python
# At the end of config.py, add Hindi configurations:

# Option 1: Test single model on all Hindi variants
for model in [ApiModel.GPT_4O_MINI]:
    for translate_mode in [
        Translated(Language.HINDI, TranslateOption.FULLY_TRANSLATED),
        Translated(Language.HINDI, TranslateOption.FULLY_TRANSLATED_PROMPT_TRANSLATE),
        Translated(Language.HINDI, TranslateOption.FULLY_TRANSLATED_POST_PROCESS_SAME),
    ]:
        for add_noise_mode in [
            AddNoiseMode.NO_NOISE,
            AddNoiseMode.SYNONYM,
            AddNoiseMode.PARAPHRASE,
        ]:
            configs.append(
                Config(
                    model=model,
                    translate_mode=translate_mode,
                    add_noise_mode=add_noise_mode,
                )
            )

# Option 2: Test multiple models
for model in [ApiModel.GPT_4O_MINI, ApiModel.CLAUDE_SONNET, ApiModel.CLAUDE_HAIKU]:
    for translate_mode in [
        Translated(Language.HINDI, TranslateOption.FULLY_TRANSLATED),
    ]:
        configs.append(
            Config(
                model=model,
                translate_mode=translate_mode,
                add_noise_mode=AddNoiseMode.NO_NOISE,
            )
        )
```

#### Translation Mode Options:
- `FULLY_TRANSLATED`: Questions in Hindi, ground truth in English
- `FULLY_TRANSLATED_PROMPT_TRANSLATE`: Hindi + ask model to use English for parameter names
- `FULLY_TRANSLATED_POST_PROCESS_DIFFERENT`: Hindi + strict semantic matching
- `FULLY_TRANSLATED_POST_PROCESS_SAME`: Hindi + lenient semantic matching

#### Noise Mode Options:
- `NO_NOISE`: Clean dataset
- `SYNONYM`: With synonym replacements
- `PARAPHRASE`: With paraphrasing

---

### Phase 4: Run BFCL Inference

#### What it does:
- For each configuration:
  1. Loads Hindi dataset
  2. Creates model interface
  3. Calls model for each test case
  4. Parses outputs to JSON
  5. Post-processes if enabled
  6. Evaluates against ground truth
  7. Calculates accuracy

#### How to run:
```bash
python generate_translated.py
```

#### Configuration (in `generate_translated.py`):
```python
requires_inference_raw = True       # Stage 1: Get raw model outputs
requires_inference_json = True      # Stage 2: Parse to JSON
requires_post_processing = True     # Stage 3: Semantic matching (optional)
requires_evaluation = True          # Stage 4: Compare with ground truth
requires_score = True               # Stage 5: Calculate accuracy

evaluation_caching = False          # Cache post-processing results
```

#### Pipeline Details:

**Stage 1: Raw Inference**
- Batches API calls (up to 8 concurrent)
- Saves raw output: `result/inference_raw/BFCL_v4_multiple_gpt4o_mini_hi_f.json`

**Stage 2: JSON Parsing**
- Parses function calls from raw output
- API models: AST-based parsing (Python syntax)
- Local models: JSON-based parsing
- Saves: `result/inference_json/BFCL_v4_multiple_gpt4o_mini_hi_f.json`

**Stage 3: Post-Processing (optional)**
- Uses GPT-4o for semantic matching
- Replaces parameters if deemed equivalent
- Saves: `result/post_processing/BFCL_v4_multiple_gpt4o_mini_hi_f.json`

**Stage 4: Evaluation**
- Compares with ground truth
- Marks correct/incorrect
- Saves: `result/evaluation/BFCL_v4_multiple_gpt4o_mini_hi_f.json`

**Stage 5: Scoring**
- Calculates accuracy: correct_cases / total_cases
- Generates summary
- Saves: `result/score/BFCL_v4_multiple_gpt4o_mini_hi_f.json`

#### Time per configuration:
- GPT-4o-mini: 5-10 minutes
- Claude Sonnet: 10-20 minutes
- Claude Haiku: 5-10 minutes
- DeepSeek: 5-10 minutes
- Local models: 20-30 minutes

#### Cost per configuration:
- GPT-4o-mini: $0.15-0.30
- Claude Sonnet: $0.30-0.60
- Claude Haiku: $0.10-0.20
- DeepSeek: $0.05-0.15

---

### Phase 5: Check Results

#### View accuracy scores:
```bash
# List all results
ls result/score/BFCL_v4_multiple_*_hi_*.json

# View specific result
cat result/score/BFCL_v4_multiple_gpt4o_mini_hi_f.json

# Extract accuracy
python -c "import json; d=json.load(open('result/score/BFCL_v4_multiple_gpt4o_mini_hi_f.json')); print(f\"Accuracy: {d[0]['accuracy']:.2%}\")"
```

#### Result format:
```json
{
  "accuracy": 0.92,
  "total_cases": 1000,
  "correct_cases": 920
}
```

#### Compare results:
```bash
# Create comparison
for f in result/score/BFCL_v4_multiple_*_hi_*.json; do
  accuracy=$(python -c "import json; d=json.load(open('$f')); print(d[0]['accuracy'])")
  model=$(basename "$f" | sed 's/.*multiple_//;s/_hi.*//')
  echo "$model: $accuracy"
done
```

---

## Complete Workflow Examples

### Example 1: Test GPT-4o-mini on Full Hindi

**Step 1: Generate Hindi translation**
```bash
python generate_translated.py
# Wait 30-60 min
# Output: BFCL_v4_multiple_hi_full.json
```

**Step 2: Configure in config.py**
```python
configs = [
    Config(
        model=ApiModel.GPT_4O_MINI,
        translate_mode=Translated(Language.HINDI, TranslateOption.FULLY_TRANSLATED),
        add_noise_mode=AddNoiseMode.NO_NOISE,
    )
]
```

**Step 3: Run inference**
```bash
python generate_translated.py
# Wait 5-10 min
# Output: Results in result/score/
```

**Step 4: Check results**
```bash
cat result/score/BFCL_v4_multiple_gpt4o_mini_hi_f.json
```

---

### Example 2: Compare Multiple Models with Synonyms

**Step 1: Generate Hindi base + synonyms**
```bash
python generate_translated.py  # ~45 min
python generate_synonym_dataset_hi.py  # ~1 hour
```

**Step 2: Configure multiple models**
```python
for model in [ApiModel.GPT_4O_MINI, ApiModel.CLAUDE_SONNET, ApiModel.CLAUDE_HAIKU]:
    configs.append(
        Config(
            model=model,
            translate_mode=Translated(Language.HINDI, TranslateOption.FULLY_TRANSLATED),
            add_noise_mode=AddNoiseMode.SYNONYM,
        )
    )
```

**Step 3: Run all (sequential or SLURM)**
```bash
# Sequential (takes ~45 min total)
python generate_translated.py

# Or use SLURM for parallel execution
sbatch run.slurm
```

**Step 4: Compare accuracies**
```bash
ls result/score/BFCL_v4_multiple_*_hi_f_syno.json
```

---

### Example 3: Full Evaluation (All Variants)

**Step 1: Prepare all Hindi datasets**
```bash
python generate_translated.py           # 45 min - full + partial
python generate_synonym_dataset_hi.py   # 90 min - full + partial versions
python generate_paraphrased_dataset_hi.py  # 90 min - full + partial versions
```

**Step 2: Configure all combinations**
```python
for model in [ApiModel.GPT_4O_MINI, ApiModel.CLAUDE_SONNET]:
    for translate_mode in [
        Translated(Language.HINDI, TranslateOption.FULLY_TRANSLATED),
        Translated(Language.HINDI, TranslateOption.PARTIALLY_TRANSLATED),
    ]:
        for add_noise_mode in [AddNoiseMode.NO_NOISE, AddNoiseMode.SYNONYM, AddNoiseMode.PARAPHRASE]:
            configs.append(Config(model, translate_mode, add_noise_mode))
```

**Step 3: Run all configurations**
```bash
# Total: 2 models × 2 translate_modes × 3 noise_modes = 12 configs
# Time: ~2-3 hours total
# Cost: ~$3-5
python generate_translated.py
```

**Step 4: Analyze results**
```bash
# Create summary script
python -c "
import json
import os

results = {}
for f in os.listdir('result/score'):
    if '_hi_' in f:
        with open(f'result/score/{f}') as file:
            data = json.load(file)
            accuracy = data[0]['accuracy']
            results[f.replace('BFCL_v4_multiple_', '').replace('.json', '')] = accuracy

for config, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f'{config}: {acc:.2%}')
"
```

---

## File Naming Convention

### Dataset Files
```
BFCL_v4_multiple[_LANG][_TRANS_MODE][_NOISE].json

Examples:
  BFCL_v4_multiple.json              # Base English
  BFCL_v4_multiple_hi_full.json      # Full Hindi translation
  BFCL_v4_multiple_hi_partial.json   # Partial Hindi translation
  BFCL_v4_multiple_hi_full_syno.json # Full Hindi + synonyms
  BFCL_v4_multiple_hi_full_para.json # Full Hindi + paraphrasing
```

### Result Files
```
BFCL_v4_multiple[_MODEL][_LANG][_TRANS_MODE][_NOISE].json

Examples:
  result/inference_raw/BFCL_v4_multiple_gpt4o_mini_hi_f.json
  result/inference_json/BFCL_v4_multiple_gpt4o_mini_hi_f.json
  result/post_processing/BFCL_v4_multiple_gpt4o_mini_hi_f.json
  result/evaluation/BFCL_v4_multiple_gpt4o_mini_hi_f.json
  result/score/BFCL_v4_multiple_gpt4o_mini_hi_f.json
```

---

## Parallelization with SLURM

For faster execution across multiple configurations, use SLURM:

**Edit run.slurm:**
```bash
#!/bin/bash
#SBATCH --job-name=bfcl_hindi
#SBATCH --time=12:00:00
#SBATCH --gpus=0
#SBATCH --mem=16G

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

### Issue: File not found error
```
FileNotFoundError: dataset/BFCL_v4_multiple_hi_full.json
```
**Solution:** Run `python generate_translated.py` first to create Hindi translation

### Issue: API rate limiting
```
429 Too Many Requests
```
**Solution:** Reduce batch size or wait between runs

### Issue: Parse failures
```
Many "Error: Could not parse output" results
```
**Solution:** Check `result/inference_raw/*.json` for actual model output format

### Issue: Out of memory
```
CUDA out of memory
```
**Solution:** Use smaller batch size or API models instead of local models

---

## Success Criteria

✓ All Hindi datasets created successfully
✓ Inference runs without errors
✓ Accuracy scores calculated
✓ Results saved to result/score/
✓ Accuracy > 50% for most models (depends on model quality)

---

## Next Steps

1. ✓ Understand the workflow (this document)
2. → Generate Hindi translations: `python generate_translated.py`
3. → Generate synonyms: `python generate_synonym_dataset_hi.py`
4. → Generate paraphrases: `python generate_paraphrased_dataset_hi.py`
5. → Configure models in `config.py`
6. → Run inference: `python generate_translated.py`
7. → Analyze results in `result/score/`

