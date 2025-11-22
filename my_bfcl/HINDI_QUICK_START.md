# Hindi BFCL Quick Commands

## Pre-requisites
```bash
# Ensure .env file has:
OPENAI_API_KEY=sk_...
DEEPSEEK_API_KEY=sk_...
ANTHROPIC_API_KEY=sk_...
```

## Complete Workflow (Copy-Paste Commands)

### 1. Generate Base Hindi Translation
```bash
python generate_translated.py
# Generates:
#   - BFCL_v4_multiple_hi_full.json (45 min, $0.50-1.00)
#   - BFCL_v4_multiple_hi_partial.json
```

### 2. Generate Hindi Synonyms (Optional)
```bash
python generate_synonym_dataset_hi.py
# Generates:
#   - BFCL_v4_multiple_hi_full_syno.json (1 hour, $0.30-0.50)
#   - BFCL_v4_multiple_hi_partial_syno.json
```

### 3. Generate Hindi Paraphrases (Optional)
```bash
python generate_paraphrased_dataset_hi.py
# Generates:
#   - BFCL_v4_multiple_hi_full_para.json (1 hour, $0.30-0.50)
#   - BFCL_v4_multiple_hi_partial_para.json
```

### 4. Configure Models in config.py
Add to bottom of config.py:
```python
# Test single model on Hindi full translation
configs.append(Config(
    model=ApiModel.GPT_4O_MINI,
    translate_mode=Translated(Language.HINDI, TranslateOption.FULLY_TRANSLATED),
    add_noise_mode=AddNoiseMode.NO_NOISE,
))
```

### 5. Run BFCL Inference
```bash
python generate_translated.py
# Generates 5 result files for each config:
#   - result/inference_raw/BFCL_v4_multiple_gpt4o_mini_hi_f.json
#   - result/inference_json/BFCL_v4_multiple_gpt4o_mini_hi_f.json
#   - result/post_processing/BFCL_v4_multiple_gpt4o_mini_hi_f.json
#   - result/evaluation/BFCL_v4_multiple_gpt4o_mini_hi_f.json
#   - result/score/BFCL_v4_multiple_gpt4o_mini_hi_f.json (FINAL ACCURACY)
```

### 6. View Results
```bash
# Show accuracy for all Hindi configs
ls result/score/BFCL_v4_multiple_*_hi_*.json | xargs -I {} sh -c 'echo "{}: $(python -c "import json; print(json.load(open(\"{}\" ))[0][\"accuracy\"]:.2%}")'

# Or view one file
cat result/score/BFCL_v4_multiple_gpt4o_mini_hi_f.json
```

---

## Common Configurations

### Test GPT-4o-mini on Full Hindi (Fastest)
```python
configs = [Config(ApiModel.GPT_4O_MINI, Translated(Language.HINDI, TranslateOption.FULLY_TRANSLATED), AddNoiseMode.NO_NOISE)]
```
**Time:** ~45 min (translate) + 5 min (infer) = ~50 min
**Cost:** ~$1.00

### Test All Models on Full Hindi (Compare)
```python
for model in [ApiModel.GPT_4O_MINI, ApiModel.CLAUDE_SONNET, ApiModel.CLAUDE_HAIKU, ApiModel.DEEPSEEK_CHAT]:
    configs.append(Config(model, Translated(Language.HINDI, TranslateOption.FULLY_TRANSLATED), AddNoiseMode.NO_NOISE))
```
**Time:** ~45 min (translate) + 45 min (infer) = ~90 min
**Cost:** ~$1.50

### Test with Synonyms & Paraphrasing
```python
for translate_mode in [Translated(Language.HINDI, TranslateOption.FULLY_TRANSLATED)]:
    for noise in [AddNoiseMode.NO_NOISE, AddNoiseMode.SYNONYM, AddNoiseMode.PARAPHRASE]:
        configs.append(Config(ApiModel.GPT_4O_MINI, translate_mode, noise))
```
**Time:** ~45 min (translate) + 1 hour (synonyms) + 1 hour (paraphrase) + 15 min (infer) = ~3.25 hours
**Cost:** ~$1.30

---

## File Structure After Completion

```
dataset/
├── BFCL_v4_multiple.json                    (base)
├── BFCL_v4_multiple_hi_full.json            (translated)
├── BFCL_v4_multiple_hi_full_syno.json       (+ synonyms)
├── BFCL_v4_multiple_hi_full_para.json       (+ paraphrasing)
├── BFCL_v4_multiple_hi_partial.json         (partial translation)
└── possible_answer/BFCL_v4_multiple.json    (ground truth)

result/
├── inference_raw/
│   └── BFCL_v4_multiple_gpt4o_mini_hi_f.json
├── inference_json/
│   └── BFCL_v4_multiple_gpt4o_mini_hi_f.json
├── post_processing/
│   └── BFCL_v4_multiple_gpt4o_mini_hi_f.json
├── evaluation/
│   └── BFCL_v4_multiple_gpt4o_mini_hi_f.json
└── score/
    └── BFCL_v4_multiple_gpt4o_mini_hi_f.json  ← ACCURACY SCORE
```

---

## Results Format

Final result file: `result/score/BFCL_v4_multiple_gpt4o_mini_hi_f.json`

```json
{
  "accuracy": 0.92,
  "total_cases": 1000,
  "correct_cases": 920
}
```

---

## Parallelization (SLURM)

For multiple configurations, submit as batch job:

```bash
# Edit config.py with all desired configurations

# Submit
sbatch run.slurm

# Monitor
squeue -u $USER
tail -f slurm-*.out

# Check results
ls -lh result/score/BFCL_v4_multiple_*_hi_*.json
```

---

## Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: BFCL_v4_multiple_hi_full.json` | Run `python generate_translated.py` first |
| `OPENAI_API_KEY not found` | Add to .env: `OPENAI_API_KEY=sk_...` |
| `429 Too Many Requests` | Wait 1 minute, then rerun |
| `CUDA out of memory` | Remove GPU-requiring configs from config.py |
| Low accuracy (< 30%) | Check if dataset generated correctly |

---

## Documentation Files

For detailed understanding, see:
- `HINDI_WORKFLOW_GUIDE.md` - Detailed workflow explanation
- `HINDI_EXECUTION_GUIDE.md` - Step-by-step execution guide with examples
- `DEVELOPER_GUIDE.md` - Model interface details
- `config.py` - All configuration options

---

## Key Points to Remember

1. **Translation comes first** - Must run `generate_translated.py` before anything else
2. **Datasets are reusable** - After creating Hindi datasets, reuse them for multiple inference runs
3. **Incremental processing** - All scripts support resuming (skip already processed items)
4. **Results accumulate** - Each config adds to result files, can run multiple times
5. **Ground truth is English** - Evaluation compares against English ground truth (functions are language-neutral)

---

## Expected Accuracy Ranges

Typical accuracy on Hindi datasets (depends on model quality):
- GPT-4o-mini: 85-92%
- Claude Sonnet: 80-88%
- Claude Haiku: 75-85%
- DeepSeek: 70-80%
- Local models: Varies (30-60%)

Lower accuracy is expected on Hindi due to:
- Translation ambiguities
- Model's Hindi capability
- Complex technical terminology

---

## Total Time & Cost Estimate

### Minimal Setup (1 Model, No Augmentation)
- **Time:** ~50 min
- **Cost:** ~$1.00
- **Configs:** 1

### Standard Setup (3 Models, Full/Partial)
- **Time:** ~2 hours
- **Cost:** ~$2.50
- **Configs:** 6

### Full Evaluation (3 Models, 3 Augmentations, 2 Translation Modes)
- **Time:** ~5 hours
- **Cost:** ~$5.00
- **Configs:** 18

---

## Next Steps

1. Ensure .env file is set up
2. Run: `python generate_translated.py` (wait for completion)
3. Review: `dataset/BFCL_v4_multiple_hi_full.json` (verify translation quality)
4. Configure: Add Hindi configs to `config.py`
5. Execute: `python generate_translated.py` (wait for inference)
6. Analyze: `cat result/score/BFCL_v4_multiple_gpt4o_mini_hi_f.json`

