from parse_dataset import load_json_lines
import json
from config import *
from parse_ast import *
import re
from call_llm import make_chat_pipeline
from models.model_factory import create_model_interface
from post_processing import (
    load_or_create_cache,
    save_cache,
    process_post_processing_sample
)


# File operation helper functions

def load_json_lines_from_file(file_path: str) -> tuple[list, set]:
    """
    Load JSON lines from a file and extract existing IDs.

    Args:
        file_path: Path to the JSON lines file

    Returns:
        Tuple of (results_list, existing_ids_set)
    """
    results = []
    existing_ids = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if f.readable():
                for line in f:
                    line_json = json.loads(line)
                    id = line_json["id"]
                    results.append(line_json)
                    existing_ids.add(id)
    except FileNotFoundError:
        print(f"File {file_path} not found. It will be created.")
    return results, existing_ids


def write_json_lines_to_file(file_path: str, results: list) -> None:
    """
    Write JSON lines to a file, overwriting existing content.

    Args:
        file_path: Path to the output file
        results: List of dictionaries to write
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
        f.flush()


def sort_results_by_id(results: list) -> list:
    """
    Sort results by numeric ID extracted from the 'id' field.

    Args:
        results: List of dictionaries with 'id' field

    Returns:
        Sorted list of results
    """
    return sorted(
        results,
        key=lambda x: int(re.search(r'\d+', x["id"]).group())
                      if re.search(r'\d+', x["id"])
                      else float('inf')
    )


def append_and_rewrite_json_lines(file_path: str, results: list) -> None:
    """
    Append results to file and rewrite entire file with sorted results.

    Args:
        file_path: Path to the output file
        results: List of dictionaries to write
    """
    sorted_results = sort_results_by_id(results)
    write_json_lines_to_file(file_path, sorted_results)

# Run inference
# Global variable to track if pipeline is initialized (reuse across configs)
_global_pipeline = None
_global_pipeline_model = None

def get_or_create_local_pipeline(local_model: LocalModel):
    """
    Get or create a pipeline for a local model.
    Reuses the same pipeline across configs with the same model.

    Guarantees: If you switch to a different model, the previous model's memory
    is immediately freed (assumes current model will never be used again in this run).
    """
    import torch
    import gc

    global _global_pipeline, _global_pipeline_model

    # If we have a pipeline for the same model, reuse it
    if _global_pipeline is not None and _global_pipeline_model == local_model:
        print(f"Reusing existing pipeline for {local_model.value}")
        return _global_pipeline

    # Different model detected - aggressive cleanup of old pipeline
    if _global_pipeline is not None:
        print(f"Switching from {_global_pipeline_model.value} to {local_model.value}")
        print(f"Freeing memory from previous model...")

        # Properly close the generator to trigger cleanup of its closure variables
        try:
            _global_pipeline.close()
        except (StopIteration, GeneratorExit):
            pass
        except Exception as e:
            print(f"Warning: Error closing generator: {e}")

        # Delete the generator and model references
        _global_pipeline = None
        _global_pipeline_model = None

        # Force immediate garbage collection
        gc.collect()
        gc.collect()  # Run twice to handle reference cycles

        # Clear CUDA cache - this is the key step
        torch.cuda.empty_cache()

        print(f"Memory freed. Loading new model...")

    # Create new pipeline for the new model
    print(f"Creating pipeline for {local_model.value}")
    _global_pipeline = make_chat_pipeline(local_model)
    _global_pipeline_model = local_model
    return _global_pipeline


# Global caches for post-processing parameter matching (shared across all configs)
# Separate caches for different language handling options
post_processing_cache_different_path = "post_processing_match_cache_different.json"
post_processing_cache_same_path = "post_processing_match_cache_same.json"
post_processing_cache_different = load_or_create_cache(post_processing_cache_different_path)
post_processing_cache_same = load_or_create_cache(post_processing_cache_same_path)
post_processing_cache_stats_different = {'hits': 0, 'misses': 0}
post_processing_cache_stats_same = {'hits': 0, 'misses': 0}

for config in configs:
    print(f"Processing config: {config}")
    # config is composed of (model, translate_mode, add_noise_mode)

    # process model configuration
    # map model to model_postfix
    match config.model:
        case ApiModel() as api_model:
            match api_model:
                case ApiModel.GPT_4O_MINI:
                    model_postfix = "_gpt4o_mini"
                case ApiModel.CLAUDE_SONNET:
                    model_postfix = "_claude_sonnet"
                case ApiModel.CLAUDE_HAIKU:
                    model_postfix = "_claude_haiku"
                case ApiModel.DEEPSEEK_CHAT:
                    model_postfix = "_deepseek"
                case ApiModel.LLAMA_3_1_8B:
                    model_postfix = "_llama3_1_8b"
                case ApiModel.LLAMA_3_1_70B:
                    model_postfix = "_llama3_1_70b"
                case _:
                    raise ValueError(f"Unsupported API model: {api_model}")
        case LocalModel() as local_model:
            match local_model:
                case LocalModel.GRANITE_3_1_8B_INSTRUCT:
                    model_postfix = "_granite"
                case LocalModel.LLAMA_3_1_8B_INSTRUCT:
                    model_postfix = "_llama3_1_8b"
                case LocalModel.LLAMA_3_1_70B_INSTRUCT:
                    model_postfix = "_llama3_1_70b"
                case LocalModel.QWEN_2_5_7B_INSTRUCT:
                    model_postfix = "_qwen2_5_7b"
                case LocalModel.QWEN_2_5_14B_INSTRUCT:
                    model_postfix = "_qwen2_5_14b"
                case LocalModel.QWEN_2_5_32B_INSTRUCT:
                    model_postfix = "_qwen2_5_32b"
                case LocalModel.QWEN_2_5_72B_INSTRUCT:
                    model_postfix = "_qwen2_5_72b"
                case _:
                    raise ValueError(f"Unsupported local model: {local_model}")
        case _:
            raise ValueError(f"Unsupported model struct: {config.model}")
    
    post_process_option = PostProcessOption.DONT_POST_PROCESS
    prompt_translate = False
    # map translate_info to language_postfix, translate_dataset_prefix, translate_mode_prefix
    match config.translate_mode:
        case Translated(language, option):
            match language:
                case Language.CHINESE:
                    language_postfix = "_zh"
                case Language.HINDI:
                    language_postfix = "_hi"
            match option:
                case TranslateOption.FULLY_TRANSLATED:
                    translate_dataset_postfix = "_full"
                    translate_mode_postfix = "_f" # fully translated, default
                    translate_postfix = "_f" # fully translated, do not prompt translate
                case TranslateOption.FULLY_TRANSLATED_PROMPT_TRANSLATE:
                    translate_dataset_postfix = "_full"
                    translate_mode_postfix = "_pt"  # prompt translate
                    translate_postfix = "_fp" # fully translated, prompt translate
                    prompt_translate = True
                case TranslateOption.FULLY_TRANSLATED_POST_PROCESS_DIFFERENT:
                    translate_dataset_postfix = "_full"
                    translate_mode_postfix = "_ppd"  # post-process different
                    translate_postfix = "_f" # fully translated, do not prompt translate
                    post_process_option = PostProcessOption.POST_PROCESS_DIFFERENT
                case TranslateOption.FULLY_TRANSLATED_POST_PROCESS_SAME:
                    translate_dataset_postfix = "_full"
                    translate_mode_postfix = "_pps"  # post-process same
                    translate_postfix = "_f" # fully translated, do not prompt translate
                    post_process_option = PostProcessOption.POST_PROCESS_SAME
                case TranslateOption.FULLY_TRANSLATED_PROMPT_TRANSLATE_POST_PROCESS_SAME:
                    translate_dataset_postfix = "_full"
                    translate_mode_postfix = "_ptps"  # prompt translate + post-process same
                    translate_postfix = "_fp" # fully translated, prompt translate
                    post_process_option = PostProcessOption.POST_PROCESS_SAME
                case TranslateOption.PARTIALLY_TRANSLATED:
                    translate_dataset_postfix = "_partial"
                    translate_mode_postfix = "_par" # partial            
                    translate_postfix = "_f" # fully translated, do not prompt translate    
                case _:
                    raise ValueError(f"Unsupported translate option: {option}")
        case NotTranslated():
            language_postfix = ""
            translate_dataset_postfix = ""
            translate_mode_postfix = ""
            translate_postfix = ""
    match config.add_noise_mode:
        case AddNoiseMode.NO_NOISE:
            noise_postfix = ""
        case AddNoiseMode.SYNONYM:
            noise_postfix = "_syno"
        case AddNoiseMode.PARAPHRASE:
            noise_postfix = "_para"
        case _:
            raise ValueError(f"Unsupported add noise mode: {config.add_noise_mode}")
    
    
    dataset_path = f"dataset/BFCL_v4_multiple{language_postfix}{translate_dataset_postfix}{noise_postfix}.json"
    ground_truth_path = f"dataset/possible_answer/BFCL_v4_multiple.json"
    inference_raw_result_path = f"result/inference_raw/BFCL_v4_multiple{model_postfix}{language_postfix}{translate_postfix}{noise_postfix}.json"
    inference_json_result_path = f"result/inference_json/BFCL_v4_multiple{model_postfix}{language_postfix}{translate_postfix}{noise_postfix}.json"
    post_processing_result_path = f"result/post_processing/BFCL_v4_multiple{model_postfix}{language_postfix}{translate_mode_postfix}{noise_postfix}.json"
    evaluation_result_path = f"result/evaluation/BFCL_v4_multiple{model_postfix}{language_postfix}{translate_mode_postfix}{noise_postfix}.json"
    score_path = f"result/score/BFCL_v4_multiple{model_postfix}{language_postfix}{translate_mode_postfix}{noise_postfix}.json"

    test_cases, _ = load_json_lines_from_file(dataset_path)
    ground_truths, _ = load_json_lines_from_file(ground_truth_path)

    if requires_inference_raw:
        try:
            inference_raw_results, existing_inference_ids = load_json_lines_from_file(inference_raw_result_path)
            # Filter out entries with error results
            inference_raw_results = [
                entry for entry in inference_raw_results
                if not (isinstance(entry.get("result"), str) and "Error: An error occurred" in entry.get("result", ""))
            ]
            existing_inference_ids = {entry["id"] for entry in inference_raw_results}
        except FileNotFoundError:
            print(f"File {inference_raw_result_path} not found. It will be created.")
            inference_raw_results = []
            existing_inference_ids = set()
        
        printed_warning = False
        # Filter cases that haven't been processed yet
        cases_to_process = [case for case in test_cases if case['id'] not in existing_inference_ids]
        if not printed_warning and len(cases_to_process) < len(test_cases):
            print(f"Warning: some test cases already exist in inference result file. Skipping {len(test_cases) - len(cases_to_process)} cases.")
            printed_warning = True

        # Determine model type and create interface once (outside batch loop)
        is_api_model = isinstance(config.model, ApiModel)
        is_local_model = isinstance(config.model, LocalModel)

        # Batch processing configuration
        if is_api_model:
            batch_size = 8  # Process 8 cases at a time for better GPU utilization
        else:
            batch_size = 12  # Smaller batch size for local models to avoid OOM

        if is_api_model:
            model_interface = create_model_interface(config.model)
        elif is_local_model:
            local_model = config.model
            generator = get_or_create_local_pipeline(local_model)
            model_interface = create_model_interface(local_model, generator)
        else:
            raise ValueError(f"Unsupported model type: {type(config.model)}")

        # Process in batches
        for batch_start in range(0, len(cases_to_process), batch_size):
            batch_end = min(batch_start + batch_size, len(cases_to_process))
            batch_cases = cases_to_process[batch_start:batch_end]

            print(f"\nProcessing batch {batch_start // batch_size + 1}: cases {batch_start} to {batch_end}")

            # Prepare batch data
            batch_functions_list = []
            batch_user_queries = []
            for case in batch_cases:
                functions = case['function']
                user_question = case["question"][0][0]['content']
                batch_functions_list.append(functions)
                batch_user_queries.append(user_question)
            
            print(f"Calling model interface for batch of size {len(batch_cases)}...")
            batch_results = model_interface.infer_batch(
                functions_list=batch_functions_list,
                user_queries=batch_user_queries,
                prompt_passing_in_english=prompt_translate
            )
            print(f"Received batch results.")

            # Process results
            for case, result in zip(batch_cases, batch_results):
                print(f"Inferencing case id {case['id']}, question: {case['question'][0][0]['content']}")
                print("Answer: ", result)
                result_to_write = {
                    "id": case["id"],
                    "result": result
                }
                inference_raw_results.append(result_to_write)
            # Write batch results to file
            write_json_lines_to_file(inference_raw_result_path, inference_raw_results)
        # Final sort and write
        if len(inference_raw_results) > 0:
            append_and_rewrite_json_lines(inference_raw_result_path, inference_raw_results)
    if requires_inference_json:
        # reload inference raw results
        try:
            inference_raw_results, _ = load_json_lines_from_file(inference_raw_result_path)
        except FileNotFoundError:
            print(f"File {inference_raw_result_path} not found. Skipping inference json generation.")
            continue
        if evaluation_caching:
            try:
                inference_json_results, existing_inference_json_ids = load_json_lines_from_file(inference_json_result_path)
            except FileNotFoundError:
                print(f"File {inference_json_result_path} not found. Skipping inference json caching.")
                inference_json_results = []
                existing_inference_json_ids = set()
        else:
            inference_json_results = []
            existing_inference_json_ids = set()
        printed_warning = False
        # Filter samples that haven't been processed yet
        samples_to_process = [sample for sample in inference_raw_results if sample['id'] not in existing_inference_json_ids]
        if not printed_warning and len(samples_to_process) < len(inference_raw_results):
            print(f"Warning: some test cases already exist in inference json result file. Skipping {len(inference_raw_results) - len(samples_to_process)} cases.")
            printed_warning = True

        for inference_raw in samples_to_process:
            id = inference_raw['id']
            # convert raw result to json format
            raw_output = model_interface.parse_output(inference_raw['result'])
            
            # Handle both parsed (list) and unparsed (string) outputs
            if isinstance(raw_output, list):
                # Already parsed (API models like DeepSeek)
                decoded_output = raw_output
            else:
                # Still raw string (Local models like Llama) - need to parse
                decoded_output = raw_to_json(config.model, id, raw_output)
            
            inference_json_entry = {
                "id": id,
                "result": decoded_output
            }
            inference_json_results.append(inference_json_entry)

            # Write batch results to file
            write_json_lines_to_file(inference_json_result_path, inference_json_results)

        # Final sort and write
        if len(inference_json_results) > 0:
            append_and_rewrite_json_lines(inference_json_result_path, inference_json_results)
    if requires_post_processing:
        if post_process_option == PostProcessOption.DONT_POST_PROCESS:
            # Simply copy inference_json results to post_processing results without modification
            try:
                inference_json_results, _ = load_json_lines_from_file(inference_json_result_path)
            except FileNotFoundError:
                print(f"File {inference_json_result_path} not found. Skipping post processing.")
                continue

            if evaluation_caching:
                try:
                    post_processing_results, existing_post_processing_ids = load_json_lines_from_file(post_processing_result_path)
                except FileNotFoundError:
                    print(f"File {post_processing_result_path} not found. Skipping post processing caching.")
                    post_processing_results = []
                    existing_post_processing_ids = set()
            else:
                post_processing_results = []
                existing_post_processing_ids = set()

            # Copy unprocessed results
            for inference_json_line in inference_json_results:
                if inference_json_line['id'] not in existing_post_processing_ids:
                    post_processing_results.append(inference_json_line)

            # Final sort and write
            if len(post_processing_results) > 0:
                append_and_rewrite_json_lines(post_processing_result_path, post_processing_results)

            print(f"Post-processing: Copied {len(inference_json_results)} results without modification (DONT_POST_PROCESS)")

        else:
            # POST_PROCESS_DIFFERENT or POST_PROCESS_SAME: use LLM-based parameter matching
            # Select appropriate cache based on post_process_option
            if post_process_option == PostProcessOption.POST_PROCESS_SAME:
                post_processing_cache = post_processing_cache_same
                post_processing_cache_stats = post_processing_cache_stats_same
                cache_path = post_processing_cache_same_path
            elif post_process_option == PostProcessOption.POST_PROCESS_DIFFERENT:  # POST_PROCESS_DIFFERENT
                post_processing_cache = post_processing_cache_different
                post_processing_cache_stats = post_processing_cache_stats_different
                cache_path = post_processing_cache_different_path
            else:
                raise ValueError(f"Unsupported post process option: {post_process_option}")

            # reload inference json results
            try:
                inference_json_results, _ = load_json_lines_from_file(inference_json_result_path)
            except FileNotFoundError:
                print(f"File {inference_json_result_path} not found. Skipping post processing.")
                continue

            if evaluation_caching:
                try:
                    post_processing_results, existing_post_processing_ids = load_json_lines_from_file(post_processing_result_path)
                except FileNotFoundError:
                    print(f"File {post_processing_result_path} not found. Skipping post processing caching.")
                    post_processing_results = []
                    existing_post_processing_ids = set()
            else:
                post_processing_results = []
                existing_post_processing_ids = set()

            printed_warning = False
            # Filter samples that haven't been processed yet
            samples_to_process = [sample for sample in inference_json_results if sample['id'] not in existing_post_processing_ids]
            if not printed_warning and len(samples_to_process) < len(inference_json_results):
                print(f"Warning: some test cases already exist in post processing result file. Skipping {len(inference_json_results) - len(samples_to_process)} cases.")
                printed_warning = True

            for inference_json_line in samples_to_process:
                id = inference_json_line['id']
                # print(f"Post-processing case id {id}...")
                # Find matching ground truth
                ground_truth_line = next((gt for gt in ground_truths if gt['id'] == id), None)
                if ground_truth_line is None:
                    raise ValueError(f"Ground truth not found for id: {id}")
                # Process with LLM-based parameter matching
                post_processing_entry = process_post_processing_sample(
                    inference_json_line,
                    ground_truth_line,
                    ApiModel.GPT_4O_MINI,  # Use a powerful model for post-processing
                    post_process_option,
                    post_processing_cache,
                    cache_path,
                    post_processing_cache_stats
                )
                post_processing_results.append(post_processing_entry)

                # Write batch results to file
                write_json_lines_to_file(post_processing_result_path, post_processing_results)

            # Final sort and write
            if len(post_processing_results) > 0:
                append_and_rewrite_json_lines(post_processing_result_path, post_processing_results)

            print(f"Post-processing ({post_process_option.name}) completed - Hits: {post_processing_cache_stats['hits']}, Misses: {post_processing_cache_stats['misses']}")
    if requires_evaluation:
        # reload post processing results
        try:
            post_processing_results, _ = load_json_lines_from_file(post_processing_result_path)
        except FileNotFoundError:
            print(f"File {post_processing_result_path} not found. Skipping evaluation.")
            continue
        if evaluation_caching:
            try:
                evaluation_results, existing_evaluation_ids = load_json_lines_from_file(evaluation_result_path)
            except FileNotFoundError:
                print(f"File {evaluation_result_path} not found. Skipping evaluation caching.")
                evaluation_results = []
                existing_evaluation_ids = set()
        else:
            evaluation_results = []
            existing_evaluation_ids = set()
        printed_warning = False
        # Filter samples that haven't been processed yet
        samples_to_process = [
            (post_processing_line, ground_truth_line, test_case)
            for post_processing_line, ground_truth_line, test_case in zip(post_processing_results, ground_truths, test_cases)
            if post_processing_line["id"] not in existing_evaluation_ids
        ]
        if not printed_warning and len(samples_to_process) < len(post_processing_results):
            print(f"Warning: some test cases already exist in evaluation result file. Skipping {len(post_processing_results) - len(samples_to_process)} cases.")
            printed_warning = True

        for (post_processing_line, ground_truth_line, test_case) in samples_to_process:
            id = post_processing_line["id"]
            assert id == ground_truth_line["id"], f"Mismatch in IDs: {id} vs {ground_truth_line['id']}"
            assert id == test_case["id"], f"Mismatch in IDs: {id} vs {test_case['id']}"
            post_processing_result = post_processing_line["result"]
            ground_truth = ground_truth_line["ground_truth"]
            func_description = test_case['function']

            evaluation_result = evaluate_json(id, post_processing_result, ground_truth, func_description)
            evaluation_result["id"] = id
            evaluation_results.append(evaluation_result)

            # Write batch results to file
            write_json_lines_to_file(evaluation_result_path, evaluation_results)

        # Final sort and write
        if len(evaluation_results) > 0:
            append_and_rewrite_json_lines(evaluation_result_path, evaluation_results)
    if requires_score:
        # reload evaluation results
        try:
            evaluation_entries, _ = load_json_lines_from_file(evaluation_result_path)
        except FileNotFoundError:
            print(f"File {evaluation_result_path} not found. Skipping scoring.")
            continue
        # Calculate and write score results
        total_cases = 0
        correct_cases = 0
        wrong_cases = []
        score_results = []

        for evaluation_entry in evaluation_entries:
            total_cases += 1
            if evaluation_entry['valid']:
                correct_cases += 1
            else:
                wrong_cases.append(evaluation_entry)

        accuracy = correct_cases / total_cases if total_cases > 0 else 0.0
        # Add summary score
        score_result = {
            "accuracy": accuracy,
            "total_cases": total_cases,
            "correct_cases": correct_cases,
        }
        score_results.append(score_result)

        # Add wrong cases
        score_results.extend(wrong_cases)

        # Write all results to file
        write_json_lines_to_file(score_path, score_results)
        print(f"Score result written to {score_path}: {score_result}")
    print(f"Completed processing for config: {config}")




