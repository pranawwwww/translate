"""
Generate Hindi Synonym Dataset

This script takes Hindi datasets and replaces words with synonyms while keeping:
- English words unchanged (parameter names, technical terms)
- Numbers unchanged
- Semantic meaning intact

Supports both full and partial Hindi translations.
"""

import json
import re
from call_llm import api_inference
from config import ApiModel
from parse_dataset import load_json_lines


# Set to True for multilingual (Hindi with English terms), False for pure Hindi
translated = True

# Datasets to process (specify which Hindi variants to generate synonyms for)
postfix_to_generate = [
    "_hi_full",
    "_hi_partial",
]


if translated:
    # For Hindi with English terms - keep English words unchanged
    system_prompt = '''
You are a helpful assistant that replaces words with synonyms of similar meaning while maintaining semantic correctness. Your task is to process word by word and replace each word with a synonym if possible.

IMPORTANT RULES:
1. Replace ONLY non-English words with appropriate synonyms
2. KEEP all English words and numbers unchanged
3. Maintain the semantic meaning and grammatical structure
4. Do NOT perform general paraphrasing, only synonym replacement
5. Process word by word, not phrase by phrase
6. If a word has no suitable synonym or is a proper noun, keep it unchanged

Produce ONLY the modified text with synonyms, without further thoughts or explanations. Consider the example below:

USER: क्या आप calculate कर सकते हैं कि एक कार जो initial speed 20 मी/सेकंड की गति से चल रही है और फिर 10 मी/सेकंड² की acceleration से 5 सेकंड तक तेज़ होती है, उसका विस्थापन कितना होगा?

ASSISTANT: क्या आप calculate कर सकते हैं कि एक कार जो initial speed 20 मी/सेकंड की गति से दौड़ रही है और फिर 10 मी/सेकंड² की acceleration से 5 सेकंड तक तेज़ होती है, उसका स्थानांतरण क्या होगा?
'''
else:
    # For pure Hindi - can replace all words
    system_prompt = '''
You are a helpful assistant that replaces words with synonyms of similar meaning while maintaining semantic correctness. Your task is to process word by word and replace each word with a synonym if possible.

IMPORTANT RULES:
1. Replace words with appropriate synonyms
2. Maintain the semantic meaning and grammatical structure
3. Do NOT perform general paraphrasing, only synonym replacement
4. Process word by word, not phrase by phrase
5. If a word has no suitable synonym or is a proper noun, keep it unchanged

Produce ONLY the modified text with synonyms, without further thoughts or explanations. Consider the example below:

USER: क्या आप calculate कर सकते हैं कि एक कार जो initial speed 20 मी/सेकंड की गति से चल रही है और फिर 10 मी/सेकंड² की acceleration से 5 सेकंड तक तेज़ होती है, उसका विस्थापन कितना होगा?

ASSISTANT: क्या आप compute कर सकते हैं कि एक कार जो initial speed 20 मी/सेकंड की गति से दौड़ रही है और फिर 10 मी/सेकंड² की acceleration से 5 सेकंड तक तेज़ होती है, उसका स्थानांतरण क्या होगा?
'''


def generate_synonym_case(question: str) -> str:
    """
    Generate a synonym-replaced version of a Hindi question.
    
    Args:
        question: Original Hindi question text
        
    Returns:
        Question with words replaced by synonyms
    """
    input_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    synonym_question = api_inference(ApiModel.GPT_4O_MINI, input_messages)
    return synonym_question


# Process each specified postfix
for postfix in postfix_to_generate:
    print(f"\n{'='*70}")
    print(f"Generating synonym dataset for postfix: {postfix}")
    print(f"{'='*70}")
    
    original_dataset_path = f'dataset/BFCL_v4_multiple{postfix}.json'
    synonym_dataset_path = f'dataset/BFCL_v4_multiple{postfix}_syno.json'
    
    # Load original dataset
    print(f"Loading original dataset from: {original_dataset_path}")
    try:
        with open(original_dataset_path, 'r', encoding='utf-8') as f:
            original_data = load_json_lines(f)
        print(f"✓ Loaded {len(original_data)} items")
    except FileNotFoundError:
        print(f"✗ Error: File not found: {original_dataset_path}")
        print(f"  Make sure you've run generate_translated.py first to create Hindi datasets")
        continue
    
    # Load existing synonym data if it exists
    synonym_data = []
    existing_indices = []
    try:
        with open(synonym_dataset_path, 'r', encoding='utf-8') as f:
            synonym_data = load_json_lines(f)
            existing_indices = [item['id'] for item in synonym_data]
        print(f"✓ Found existing synonym dataset with {len(synonym_data)} items")
        print(f"  Will skip already processed items and continue from where we left off")
    except FileNotFoundError:
        print(f"ℹ No existing synonym dataset found at {synonym_dataset_path}")
        print(f"  Creating new dataset...")
    
    # Process and save incrementally
    processed_count = 0
    skipped_count = 0
    
    with open(synonym_dataset_path, 'w', encoding='utf-8') as f:
        warning_printed = False
        
        for item in original_data:
            item_id = item['id']
            
            # Skip already processed items
            if item_id in existing_indices:
                if not warning_printed:
                    print(f"\n⊘ Skipping {len(existing_indices)} already processed items")
                    warning_printed = True
                skipped_count += 1
                continue
            
            # Generate synonym version
            original_question = item['question'][0][0]['content']
            print(f"\nProcessing item {item_id} (#{processed_count + 1}/{len(original_data) - len(existing_indices)})")
            print(f"  Original: {original_question[:80]}...")
            
            try:
                synonym_question = generate_synonym_case(original_question)
                print(f"  Synonym:  {synonym_question[:80]}...")
                
                # Create synonym item
                synonym_item = item.copy()
                synonym_item['question'][0][0]['content'] = synonym_question
                synonym_data.append(synonym_item)
                processed_count += 1
                
            except Exception as e:
                print(f"  ✗ Error processing item: {e}")
                continue
            
            # Save incrementally (every item)
            f.seek(0)
            f.truncate()
            for n in synonym_data:
                f.write(json.dumps(n, ensure_ascii=False) + '\n')
            f.flush()
        
        # Final sort by ID
        print(f"\nSorting final dataset by ID...")
        synonym_data = sorted(
            synonym_data,
            key=lambda x: int(re.search(r'\d+', x["id"]).group())
            if re.search(r'\d+', x["id"])
            else float('inf')
        )
        
        f.seek(0)
        f.truncate()
        for n in synonym_data:
            f.write(json.dumps(n, ensure_ascii=False) + '\n')
        f.flush()
    
    # Summary
    print(f"\n{'='*70}")
    print(f"✓ Synonym dataset generation complete!")
    print(f"  Total items in dataset: {len(synonym_data)}")
    print(f"  Newly processed: {processed_count}")
    print(f"  Previously processed (skipped): {skipped_count}")
    print(f"  Saved to: {synonym_dataset_path}")
    print(f"{'='*70}\n")

print("\n✓ All Hindi synonym datasets processed successfully!")
