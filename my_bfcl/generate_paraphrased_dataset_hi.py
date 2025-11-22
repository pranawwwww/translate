"""
Generate Hindi Paraphrased Dataset

This script takes Hindi datasets and rephrases questions while keeping:
- English words unchanged (parameter names, technical terms)
- Numbers unchanged
- Semantic meaning identical

Supports both full and partial Hindi translations.
"""

import json
import re
from call_llm import api_inference
from config import ApiModel
from parse_dataset import load_json_lines


# Set to True for multilingual (Hindi with English terms), False for pure Hindi
translated = True

# Datasets to process (specify which Hindi variants to generate paraphrases for)
postfix_to_generate = [
    "_hi_full",
    "_hi_partial",
]


if translated:
    # For Hindi with English terms - keep English words unchanged
    system_prompt = '''
You are a helpful assistant helping rephrasing user requests in Hindi, while accurately preserving their meaning, including numbers and names if they exist. If the request is multilingual, *DON'T* translate English words to other languages and *KEEP* all English words and numbers intact. Do not answer the requirement, just produce another one that is identical in meaning but is phrased differently. Produce ONLY the rephrased requirement, without further thoughts or explanations. Consider the example below:

USER: क्या आप calculate कर सकते हैं कि एक कार जो initial speed 20 मी/सेकंड की गति से चल रही है और फिर 10 मी/सेकंड² की acceleration से 5 सेकंड तक तेज़ होती है, उसका विस्थापन कितना होगा?

ASSISTANT: क्या आप ज्ञात कर सकते हैं कि एक कार, जिसकी initial speed 20 मी/सेकंड है और जो 10 मी/सेकंड² की acceleration से 5 सेकंड तक तेज़ी करती है, का विस्थापन क्या होगा?
'''
else:
    # For pure Hindi - can rephrase all words
    system_prompt = '''
You are a helpful assistant helping rephrasing user requests in Hindi, while accurately preserving their meaning, including numbers and names if they exist. Do not answer the requirement, just produce another one that is identical in meaning but is phrased differently. Produce ONLY the rephrased requirement, without further thoughts or explanations. Consider the example below:

USER: क्या आप calculate कर सकते हैं कि एक कार जो initial speed 20 मी/सेकंड की गति से चल रही है और फिर 10 मी/सेकंड² की acceleration से 5 सेकंड तक तेज़ होती है, उसका विस्थापन कितना होगा?

ASSISTANT: क्या आप ज्ञात कर सकते हैं कि एक कार, जिसकी initial speed 20 मी/सेकंड है और जो 10 मी/सेकंड² की acceleration से 5 सेकंड तक तेज़ी करती है, का विस्थापन क्या होगा?
'''


def generate_paraphrased_case(question: str) -> str:
    """
    Generate a paraphrased version of a Hindi question.
    
    Args:
        question: Original Hindi question text
        
    Returns:
        Paraphrased question with identical meaning but different phrasing
    """
    input_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    paraphrased_question = api_inference(ApiModel.GPT_4O_MINI, input_messages)
    return paraphrased_question


# Process each specified postfix
for postfix in postfix_to_generate:
    print(f"\n{'='*70}")
    print(f"Generating paraphrased dataset for postfix: {postfix}")
    print(f"{'='*70}")
    
    original_dataset_path = f'dataset/BFCL_v4_multiple{postfix}.json'
    paraphrased_dataset_path = f'dataset/BFCL_v4_multiple{postfix}_para.json'
    
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
    
    # Load existing paraphrased data if it exists
    paraphrased_data = []
    existing_indices = []
    try:
        with open(paraphrased_dataset_path, 'r', encoding='utf-8') as f:
            paraphrased_data = load_json_lines(f)
            existing_indices = [item['id'] for item in paraphrased_data]
        print(f"✓ Found existing paraphrased dataset with {len(paraphrased_data)} items")
        print(f"  Will skip already processed items and continue from where we left off")
    except FileNotFoundError:
        print(f"ℹ No existing paraphrased dataset found at {paraphrased_dataset_path}")
        print(f"  Creating new dataset...")
    
    # Process and save incrementally
    processed_count = 0
    skipped_count = 0
    
    with open(paraphrased_dataset_path, 'w', encoding='utf-8') as f:
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
            
            # Generate paraphrased version
            original_question = item['question'][0][0]['content']
            print(f"\nProcessing item {item_id} (#{processed_count + 1}/{len(original_data) - len(existing_indices)})")
            print(f"  Original:    {original_question[:80]}...")
            
            try:
                paraphrased_question = generate_paraphrased_case(original_question)
                print(f"  Paraphrased: {paraphrased_question[:80]}...")
                
                # Create paraphrased item
                paraphrased_item = item.copy()
                paraphrased_item['question'][0][0]['content'] = paraphrased_question
                paraphrased_data.append(paraphrased_item)
                processed_count += 1
                
            except Exception as e:
                print(f"  ✗ Error processing item: {e}")
                continue
            
            # Save incrementally (every item)
            f.seek(0)
            f.truncate()
            for n in paraphrased_data:
                f.write(json.dumps(n, ensure_ascii=False) + '\n')
            f.flush()
        
        # Final sort by ID
        print(f"\nSorting final dataset by ID...")
        paraphrased_data = sorted(
            paraphrased_data,
            key=lambda x: int(re.search(r'\d+', x["id"]).group())
            if re.search(r'\d+', x["id"])
            else float('inf')
        )
        
        f.seek(0)
        f.truncate()
        for n in paraphrased_data:
            f.write(json.dumps(n, ensure_ascii=False) + '\n')
        f.flush()
    
    # Summary
    print(f"\n{'='*70}")
    print(f"✓ Paraphrased dataset generation complete!")
    print(f"  Total items in dataset: {len(paraphrased_data)}")
    print(f"  Newly processed: {processed_count}")
    print(f"  Previously processed (skipped): {skipped_count}")
    print(f"  Saved to: {paraphrased_dataset_path}")
    print(f"{'='*70}\n")

print("\n✓ All Hindi paraphrased datasets processed successfully!")
