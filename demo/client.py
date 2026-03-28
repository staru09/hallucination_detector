"""
Client process that sends requests to vLLM server and verifies hidden states.

This module handles:
- Sending requests to the vLLM server
- Extracting filepath from the response
- Loading and verifying hidden states data
- Logging results to shared console
- Sending subsequent queries
"""

import sys
from pathlib import Path
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)

import requests
import safetensors.torch

# Ensure we can import from demo package
# This is needed when running as a multiprocessing subprocess
demo_dir = Path(__file__).parent
parent_dir = demo_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from datasets import load_dataset


def load_sharegpt_dataset(client_id: int, num_queries: int):
    """
    Load ShareGPT dataset and transform conversations to OpenAI chat format.

    Returns:
        Dataset with 'messages' field containing OpenAI chat format messages
    """
    dataset = load_dataset("Aeala/ShareGPT_Vicuna_unfiltered", split="train")
    dataset = dataset.shuffle(seed=42 + client_id)
    dataset = dataset.select(range(num_queries))

    def transform_conversation(example):
        """Convert ShareGPT conversation format to OpenAI chat format."""
        conversation = example["conversations"]
        messages = []
        for turn in conversation:
            role = "user" if turn.get("from") == "human" else "assistant"
            content = turn.get("value", "")
            messages.append({"role": role, "content": content})
        return {"messages": messages}

    dataset = dataset.map(transform_conversation, remove_columns=["conversations"])
    return dataset


def check_hidden_states_conditions(
    filepath: str, expected_len: int
) -> tuple[bool, str]:
    """
    Load hidden states data and check that it matches expected conditions.

    Args:
        filepath: Path to the safetensors file containing hidden states
        client_id: ID of the client process
        query_num: Query number for this request

    Returns:
        Tuple of (condition_met: bool, message: str)
    """
    try:
        if not Path(filepath).exists():
            return False, f"File does not exist: {filepath}"

        # Load the data
        data = safetensors.torch.load_file(filepath)

        # Check for required tensors
        if "hidden_states" not in data:
            return False, "Missing 'hidden_states' tensor"
        if "token_ids" not in data:
            return False, "Missing 'token_ids' tensor"

        hidden_states = data["hidden_states"]
        token_ids = data["token_ids"]

        assert hidden_states.shape[1] == expected_len, (
            f"Expected hidden_states length: {expected_len}, got: {hidden_states.shape[1]}"
        )
        assert token_ids.shape[0] == expected_len, (
            f"Expected token_ids length: {expected_len}, got: {token_ids.shape[0]}"
        )

        return (
            True,
            f"Validation passed - hidden_states shape: {hidden_states.shape}, token_ids shape: {token_ids.shape}",
        )

    except Exception as e:
        return False, f"Error checking conditions: {str(e)}"


def send_request(
    server_url: str, model: str, messages: list, log_prefix: str
) -> Optional[dict]:
    """
    Send a chat completion request to the vLLM server.

    Args:
        server_url: Base URL of the vLLM server
        model: Model name/path
        messages: List of message dictionaries in OpenAI chat format
        log_prefix: Prefix for logging

    Returns:
        The API response dictionary, or None if request failed
    """
    url = f"{server_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 1,
    }

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"{log_prefix} Request failed: {str(e)}")
        return None


def run_client(
    client_id: int,
    server_url: str,
    model: str,
    num_queries: int,
):
    """
    Main client loop that sends queries and verifies hidden states.

    Args:
        client_id: Unique ID for this client process
        server_url: Base URL of the vLLM server
        model: Model name/path
        storage_path: Path where hidden states are stored
        num_queries: Number of queries to send
    """
    logging.info(f"[Client {client_id}] Starting with {num_queries} queries")

    dataset = load_sharegpt_dataset(client_id, num_queries)

    for query_num in range(num_queries):
        log_prefix = f"[Client {client_id}] [Query {query_num + 1}/{num_queries}]"
        # Get messages from pre-transformed dataset
        messages = dataset[query_num]["messages"]

        # Log first message preview
        first_message = messages[0]["content"] if messages else ""
        logging.info(
            f"{log_prefix} Sending request with {len(messages)} messages, first: '{first_message[:50]}...'"
        )

        # Send request to server
        response = send_request(server_url, model, messages, log_prefix)
        if response is None:
            continue

        # Extract filepath from response
        kv_transfer_params = response.get("kv_transfer_params")
        if kv_transfer_params is None:
            logging.error(
                f"{log_prefix} Could not extract kv_transfer_params from response"
            )
            continue
        filepath = kv_transfer_params.get("hidden_states_path")
        if filepath is None:
            logging.error(f"{log_prefix} Could not extract filepath from response")
            continue

        logging.info(f"{log_prefix} Extracted filepath: {filepath}")

        num_prompt_tokens = response.get("usage", {}).get("prompt_tokens", None)

        if num_prompt_tokens is None:
            logging.error(
                f"{log_prefix} Could not extract num_prompt_tokens from response"
            )
            continue

        # Load and check conditions
        condition_met, message = check_hidden_states_conditions(
            filepath,
            num_prompt_tokens,
        )

        if condition_met:
            logging.info(f"{log_prefix} {message}")
        else:
            logging.error(f"{log_prefix} Condition check failed - {message}")

    logging.info(f"{log_prefix} Completed all queries")


if __name__ == "__main__":
    run_client(
        client_id=0,
        server_url="http://localhost:8000",
        model="./demo/qwen3_8b",
        num_queries=5,
    )
