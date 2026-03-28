#!/usr/bin/env python3
"""
Main script to launch multiple client processes for testing hidden states extraction.

Each client process will:
1. Send requests to the vLLM server
2. Extract filepath from the response
3. Load and verify the hidden states data
4. Log results to a shared console
5. Send the next query
"""

import argparse
import multiprocessing
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from demo.client import run_client


def main():
    parser = argparse.ArgumentParser(
        description="Launch multiple client processes to test hidden states extraction"
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=3,
        help="Number of client processes to launch (default: 3)",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8000",
        help="URL of the vLLM server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="./demo/qwen3_8b",
        help="Model name/path (default: ./demo/qwen3_8b)",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        default=25,
        help="Number of queries each client should send (default: 5)",
    )

    args = parser.parse_args()

    # Launch multiple client processes
    processes = []
    for client_id in range(args.num_clients):
        p = multiprocessing.Process(
            target=run_client,
            args=(
                client_id,
                args.server_url,
                args.model,
                args.num_queries,
            ),
        )
        p.start()
        processes.append(p)

    # Wait for all processes to complete
    for p in processes:
        p.join()

    print(f"\nAll {args.num_clients} clients completed.")


if __name__ == "__main__":
    main()
