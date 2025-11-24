"""
Interface for Meta Llama 3.1 models running locally on GPU.

Handles:
- Local model inference via generator pipeline
- Llama 3.1 chat template formatting
- Output parsing from JSON format (tool_call)
"""

import json
from typing import List, Dict, Any
from models.base import ModelInterface


class Llama31LocalInterface(ModelInterface):
    """Handler for Meta Llama 3.1 8B/70B Instruct local models on GPU."""

    def __init__(self, generator, model_id: str = "meta-llama/Llama-3.1-8B-Instruct"):
        """
        Initialize the Llama 3.1 local interface.

        Args:
            generator: Pre-initialized generator from pipeline
            model_id: Model identifier string
                     Options: "meta-llama/Llama-3.1-8B-Instruct",
                             "meta-llama/Llama-3.1-70B-Instruct"
        """
        self.generator = generator
        self.model_id = model_id

    def infer(self, functions: List[Dict[str, Any]], user_query: str,
              prompt_passing_in_english: bool = True, model=None) -> str:
        """
        Run inference with Llama 3.1 model.

        Args:
            functions: List of available function definitions in JSON format
            user_query: User query as a string
            prompt_passing_in_english: Whether to request English parameter passing
            model: LocalModel enum (for context)

        Returns:
            Raw model output as a string
        """
        if self.generator is None:
            raise RuntimeError("Generator not initialized.")

        system_prompt = self._generate_system_prompt(
            functions=functions,
            prompt_passing_in_english=prompt_passing_in_english
        )

        # Format the input using Llama 3.1 chat template
        template = self._format_llama_chat_template(
            system_prompt=system_prompt,
            user_query=user_query,
            functions=functions,
            add_generation_prompt=True
        )

        # Send template to generator and get response
        result = self.generator.send(template)
        return result

    def infer_batch(self, functions_list: List[List[Dict[str, Any]]],
                   user_queries: List[str],
                   prompt_passing_in_english: bool = True) -> List[str]:
        """
        Run batch inference with Llama 3.1 model.

        Args:
            functions_list: List of function lists (one per query)
            user_queries: List of user queries as strings
            prompt_passing_in_english: Whether to request English parameter passing

        Returns:
            List of raw model outputs as strings
        """
        if self.generator is None:
            raise RuntimeError("Generator not initialized.")

        if len(functions_list) != len(user_queries):
            raise ValueError("functions_list and user_queries must have same length")

        # Format all templates
        templates = []
        for functions, user_query in zip(functions_list, user_queries):
            system_prompt = self._generate_system_prompt(
                functions=functions,
                prompt_passing_in_english=prompt_passing_in_english
            )
            template = self._format_llama_chat_template(
                system_prompt=system_prompt,
                user_query=user_query,
                functions=functions,
                add_generation_prompt=True
            )
            templates.append(template)

        # Send batch to generator
        results = self.generator.send(templates)
        return results

    def parse_output(self, raw_output: str) -> str:
        """
        Parse Llama 3.1 model output to extract function calls in JSON format.

        Llama may output:
        - Valid JSON: [{"name": "func", "arguments": {...}}]
        - Single quotes: [{'name': 'func', 'arguments': {...}}]
        - Mixed quotes or trailing commas

        Args:
            raw_output: Raw model output string

        Returns:
            Raw output as-is (will be handled by raw_to_json in parse_ast.py)
        """
        # Return raw output as string for parse_ast.py to handle
        # This allows parse_ast.raw_to_json() to apply JSON cleanup logic
        return raw_output.strip()

    def _generate_system_prompt(self, functions: List[Dict[str, Any]],
                                prompt_passing_in_english: bool = True) -> str:
        """
        Generate system prompt for Llama 3.1.

        Args:
            functions: List of available function definitions
            prompt_passing_in_english: Whether to emphasize English parameter passing

        Returns:
            System prompt string
        """
        base_prompt = (
            "You are a helpful AI assistant with access to the following tools. "
            "When a tool is required to answer the user's query, respond with a JSON list of tools to use. "
            "Each tool call should be in the format: [{'name': 'tool_name', 'arguments': {...}}]\n\n"
            "Available tools:\n"
        )

        if functions:
            base_prompt += json.dumps(functions, indent=2)

        if prompt_passing_in_english:
            base_prompt += "\n\nIMPORTANT: All parameter values MUST be in English."

        return base_prompt

    def _format_llama_chat_template(self, system_prompt: str, user_query: str,
                                    functions: List[Dict[str, Any]],
                                    add_generation_prompt: bool = True) -> str:
        """
        Format messages using Llama 3.1 chat template.

        Llama 3.1 uses: <|begin_of_text|><|start_header_id|>role<|end_header_id|>content<|eot_id|>

        Args:
            system_prompt: System message content
            user_query: User message content
            functions: Function definitions (included in system prompt)
            add_generation_prompt: Whether to add assistant generation prompt

        Returns:
            Formatted prompt string
        """
        # Llama 3.1 chat template
        messages = []

        # System message
        messages.append(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{system_prompt}<|eot_id|>")

        # User message
        messages.append(f"<|start_header_id|>user<|end_header_id|>\n{user_query}<|eot_id|>")

        # Assistant generation prompt
        if add_generation_prompt:
            messages.append("<|start_header_id|>assistant<|end_header_id|>\n")

        return "".join(messages)
