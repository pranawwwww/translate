
from config import ApiModel, LocalModel
from dotenv import load_dotenv
import os
import json
import gc
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed


def api_inference(model: ApiModel, input_messages: list[dict]) -> str:
    """
    Run inference with either OpenAI or Anthropic API.

    Args:
        model: ApiModel enum value.
        input_messages: list of dicts like OpenAI's `messages` format.
            Example:
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ]
    """
    load_dotenv(dotenv_path=".env")

    # --- Validate arguments ---
    if not isinstance(model, ApiModel):
        raise TypeError("`model` must be an ApiModel")

    if not isinstance(input_messages, list) or not all(isinstance(m, dict) for m in input_messages):
        raise TypeError("`input_messages` must be a list of dicts")

    valid_roles = {"system", "user", "assistant"}
    for i, m in enumerate(input_messages):
        if "role" not in m or "content" not in m:
            raise ValueError(f"Message at index {i} missing 'role' or 'content'")
        if m["role"] not in valid_roles:
            raise ValueError(f"Invalid role '{m['role']}' at index {i}")

    # --- Dispatch based on model ---
    match model:
        case ApiModel.GPT_4O_MINI:
            # Use OpenAI client
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY not found in .env")

            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=model.value,
                messages=input_messages,
                temperature=0
            )

            return response.choices[0].message.content

        case ApiModel.DEEPSEEK_CHAT:
            # Use OpenAI-compatible client for DeepSeek
            from openai import OpenAI
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise EnvironmentError("DEEPSEEK_API_KEY not found in .env")

            client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            response = client.chat.completions.create(
                model=model.value,
                messages=input_messages,
                temperature=0
            )

            return response.choices[0].message.content

        case ApiModel.CLAUDE_SONNET | ApiModel.CLAUDE_HAIKU:
            # Use Anthropic client
            from anthropic import Anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise EnvironmentError("ANTHROPIC_API_KEY not found in .env")

            client = Anthropic(api_key=api_key)

            # Extract system message (first "system" if any) and format it as Claude expects
            system_message = None
            messages_for_claude = []
            for m in input_messages:
                if m["role"] == "system" and system_message is None:
                    # Claude expects system message as a list of content blocks
                    system_message = [{"type": "text", "text": m["content"]}]
                elif m["role"] in {"user", "assistant"}:
                    # Claude expects message content as a list of content blocks
                    messages_for_claude.append(
                        {
                            "role": m["role"],
                            "content": [{"type": "text", "text": m["content"]}]
                        }
                    )

            if not any(m["role"] == "user" for m in messages_for_claude):
                raise ValueError("Claude API requires at least one 'user' message")

            # Build the API request
            kwargs = {
                "model": model.value,
                "max_tokens": 4096,
                "temperature": 0,
                "messages": messages_for_claude
            }

            # Include system message if it exists
            if system_message is not None:
                kwargs["system"] = system_message

            response = client.messages.create(**kwargs)

            # Anthropic returns structured content (list of message blocks)
            return response.content[0].text if response.content else ""

        case _:
            raise ValueError(f"Unsupported model: {model}")


def api_inference_batch(model: ApiModel, batch_messages: List[list[dict]]) -> List[str]:
    """
    Run inference on multiple inputs concurrently using the API.
    Makes parallel API calls for better throughput.

    Args:
        model: ApiModel enum value.
        batch_messages: List of message lists, where each element is like:
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ]

    Returns:
        List of responses in the same order as input batch_messages
    """
    # Use ThreadPoolExecutor for concurrent API calls
    max_workers = min(8, len(batch_messages))  # Up to 8 concurrent requests
    results = [None] * len(batch_messages)  # Pre-allocate to maintain order

    def call_api_with_index(index_and_messages):
        """Helper to call API and track original index"""
        index, messages = index_and_messages
        try:
            response = api_inference(model, messages)
            return index, response
        except Exception as e:
            print(f"Error calling API for batch item {index}: {e}")
            return index, f"Error: {str(e)}"

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(call_api_with_index, (i, messages)): i
            for i, messages in enumerate(batch_messages)
        }

        # Collect results as they complete
        for future in as_completed(futures):
            try:
                index, response = future.result()
                results[index] = response
            except Exception as e:
                index = futures[future]
                print(f"Error processing batch item {index}: {e}")
                results[index] = f"Error: {str(e)}"

    return results


def format_granite_chat_template(messages: list[dict], functions: list[dict] = None, add_generation_prompt: bool = True) -> str:
    """
    Format messages using the Granite chat template.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        functions: Optional list of function definitions for tool calling
        add_generation_prompt: Whether to add the generation prompt at the end

    Returns:
        Formatted prompt string using Granite's chat template
    """
    formatted_prompt = ""

    # Extract system message if present
    if messages and messages[0]["role"] == "system":
        system_prompt = messages[0]["content"]
        messages_to_process = messages[1:]
    else:
        # Default system prompt for Granite
        system_prompt = (
            "Knowledge Cutoff Date: April 2024.\n"
            "Today's Date: April 29, 2025.\n"
            "You are Granite, developed by IBM."
        )
        if functions:
            system_prompt += (
                " You are a helpful AI assistant with access "
                "to the following tools. When a tool is required to answer the user's query, respond "
                "with <|tool_call|> followed by a JSON list of tools used. If a tool does not exist "
                "in the provided list of tools, notify the user that you do not have the ability to fulfill the request."
            )
        messages_to_process = messages

    # Add the system message
    formatted_prompt += (
        f"<|start_of_role|>system<|end_of_role|>{system_prompt}<|end_of_text|>\n"
    )

    # Add tools section if functions are provided
    if functions:
        formatted_prompt += (
            "<|start_of_role|>tools<|end_of_role|>"
            + json.dumps(functions, indent=4)
            + "<|end_of_text|>\n"
        )

    # Add all messages
    for msg in messages_to_process:
        formatted_prompt += (
            "<|start_of_role|>"
            + msg["role"]
            + "<|end_of_role|>"
            + msg["content"]
            + "<|end_of_text|>\n"
        )

    # Add generation prompt if requested
    if add_generation_prompt:
        formatted_prompt += "<|start_of_role|>assistant<|end_of_role|>"

    return formatted_prompt


def make_chat_pipeline(model: LocalModel):
    """
    Returns a generator function that takes populated template(s) and yields model responses.
    Supports both single templates (string) and batch templates (list of strings).

    Args:
        model: A LocalModel enum value specifying which local model to use

    Returns:
        A generator that accepts a template string or list of template strings and yields responses
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    # Extract model_id from the enum
    model_id = model.value
    print(f"Loading local model: {model_id}")

    # --- Environment setup ---
    os.environ["HF_HOME"] = "/work/nvme/bfdz/zluo8/huggingface"

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Set pad token for batch processing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set padding side to left for decoder-only models (prevents generation issues in batch mode)
    tokenizer.padding_side = "left"

    # --- Patch huggingface_hub to force standard HTTP downloads (skip xet entirely) ---
    try:
        import huggingface_hub.file_download as hf_file_download
        
        # Store original _download_to_tmp_and_move
        original_download_to_tmp_and_move = hf_file_download._download_to_tmp_and_move
        
        def patched_download_to_tmp_and_move(tmp_file, destination_path, headers=None, expected_size=None, resume_size=0, url=None, max_retries=5, user_agent=None, timeout=10):
            """Patched version that skips xet and uses http_get directly"""
            # Call http_get directly instead of trying xet_get
            return hf_file_download.http_get(
                url=url,
                temp_file=tmp_file,
                resume_size=resume_size,
                headers=headers,
                expected_size=expected_size,
                timeout=timeout,
                max_retries=max_retries,
                user_agent=user_agent
            )
        
        hf_file_download._download_to_tmp_and_move = patched_download_to_tmp_and_move
        print("✓ Patched huggingface_hub to skip xet downloader")
    except Exception as e:
        print(f"Warning: Could not patch xet downloader: {e}")

    # --- Load model ---
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda:0",  # Keep model on GPU, avoid unnecessary offloading
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        # offload_folder="/work/nvme/bfdz/zluo8/hf_offload",  # Removed for better performance
    )

    hf_model.eval()

    # --- Define the generator pipeline ---
    def chat_generator():
        inputs = yield  # Initial yield to start the generator
        try:
            while True:
                # Wait for a populated template string or list of strings
                if inputs is None:
                    inputs = yield
                    continue

                # Determine if this is a batch (list) or single input (string)
                is_batch = isinstance(inputs, list)

                # Tokenize with appropriate settings
                if is_batch:
                    print(f"Generating responses for batch of {len(inputs)} inputs...")
                    tokenized = tokenizer(
                        inputs,
                        return_tensors="pt",
                        padding=True,           # Pad shorter sequences to max length
                        truncation=True,
                        max_length=4096,
                    ).to(hf_model.device)
                else:
                    print("Generating response...")
                    tokenized = tokenizer(inputs, return_tensors="pt", max_length=4096).to(hf_model.device)

                # Generate
                with torch.inference_mode():
                    outputs = hf_model.generate(
                        **tokenized,
                        max_new_tokens=4096,
                        temperature=0.001,
                        use_cache=True,  # Enable KV cache for faster generation
                    )

                # Decode
                generated_tokens = outputs[:, tokenized["input_ids"].shape[-1]:]

                if is_batch:
                    # Return list of responses for batch
                    responses = tokenizer.batch_decode(
                        generated_tokens,
                        skip_special_tokens=True
                    )
                    print(f"Generation complete for batch of {len(responses)} responses.")
                    result = responses
                else:
                    # Return single response for single input
                    response = tokenizer.decode(
                        generated_tokens[0],
                        skip_special_tokens=True
                    )
                    print("Generation complete.")
                    result = response

                # Yield response and wait for next template(s)
                inputs = yield result
        finally:
            # Cleanup when generator is closed or an exception occurs
            # IMPORTANT: Do NOT delete closure variables (tokenizer, hf_model) here!
            # Deleting them corrupts the generator if it gets reused after an exception.
            # Instead, just move the model to CPU and clear GPU memory.
            print(f"Cleaning up model {model_id}...")
            try:
                hf_model.to("cpu")
                gc.collect()
                torch.cuda.empty_cache()
                print(f"Model {model_id} cleaned up successfully.")
            except Exception as e:
                print(f"Warning: Error during cleanup of {model_id}: {e}")

    # Initialize and prime the generator
    gen = chat_generator()
    next(gen)
    print("Local model loaded and generator is ready.")
    return gen
