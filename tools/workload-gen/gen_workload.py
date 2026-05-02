#!/usr/bin/env python3
"""Generate a pre-tokenized multi-LoRA serving workload trace for the bench.

M1 minimum-viable version: CLI-driven, built-in tiny prompt pool, Poisson
arrivals, Pareto/uniform adapter selection. M4 will replace the prompt pool
with ShareGPT and add YAML scenario configs (see IMPLEMENTATION_GUIDE.md
section 4 M4).

Output JSON schema (matches main.cpp load_trace + spec section 4 M1):

    [
      {
        "id":           "req_00001",
        "arrival_time": 1.234,
        "adapter_id":   "reasoning",
        "prompt":       "Why is the sky blue?",
        "input_tokens": [128000, 3923, ...],
        "max_output":   128
      },
      ...
    ]

Tokenization uses HF AutoTokenizer + apply_chat_template(add_generation_prompt=True)
so the resulting tokens already contain the Llama-3 chat envelope and the bench
can call llama_decode directly without re-templating.

Example:

    python gen_workload.py \
        --tokenizer unsloth/Llama-3.2-3B-Instruct \
        --adapters reasoning summarization \
        --duration 60 --rate 0.5 --pareto-alpha 1.5 \
        --max-output 64 --seed 42 \
        --out workloads/m1-smoke.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

# A small built-in prompt pool. Diverse enough to show adapter style differences
# without needing ShareGPT (M4). Add more as M1 acceptance demands.
DEFAULT_PROMPTS = [
    "Why is the sky blue?",
    "Write a short poem about autumn leaves.",
    "Explain Pythagoras' theorem to a 10-year-old.",
    "List five benefits of regular exercise.",
    "Translate 'good morning' into French, Spanish, and Japanese.",
    "What are the symptoms of a common cold?",
    "Describe the plot of Romeo and Juliet in three sentences.",
    "How does photosynthesis work?",
    "Name three programming languages and one use case for each.",
    "What's the difference between weather and climate?",
    "A train leaves city A at 60 mph heading east. Another train leaves city B at 80 mph heading west. They are 350 miles apart. When do they meet? Show your reasoning step by step.",
    "Summarize the key ideas of supply and demand in economics.",
    "What is the capital of Australia, and what is it known for?",
    "Give me three healthy breakfast ideas.",
    "Write a haiku about a quiet morning.",
    "Explain the difference between RAM and ROM.",
    "List five tips for writing clear code.",
    "What causes earthquakes?",
    "Recommend a book for someone who liked The Lord of the Rings.",
    "How do vaccines work, briefly?",
]


def pareto_weights(n_items: int, alpha: float) -> np.ndarray:
    """Truncated Pareto-style weights: w_i ~ 1 / (i+1)^alpha (1-indexed)."""
    if alpha <= 0:
        # alpha=0 collapses to uniform; useful as a sanity baseline.
        w = np.ones(n_items, dtype=np.float64)
    else:
        w = np.array([1.0 / (i + 1) ** alpha for i in range(n_items)], dtype=np.float64)
    return w / w.sum()


def generate(args: argparse.Namespace) -> list[dict]:
    rng = np.random.default_rng(args.seed)

    if args.prompts_file:
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            prompt_pool = json.load(f)
        if not (isinstance(prompt_pool, list) and all(isinstance(p, str) for p in prompt_pool)):
            sys.exit("--prompts-file must hold a JSON array of strings")
    else:
        prompt_pool = DEFAULT_PROMPTS

    adapters = list(args.adapters)
    if not adapters:
        sys.exit("--adapters must list at least one adapter id")
    weights = pareto_weights(len(adapters), args.pareto_alpha)

    print(f"[gen] tokenizer = {args.tokenizer}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.chat_template is None:
        sys.exit(f"tokenizer {args.tokenizer} has no chat_template; pick an instruct model")

    # Sample arrival times via a homogeneous Poisson process (exponential gaps).
    arrivals: list[float] = []
    t = 0.0
    while t < args.duration:
        gap = rng.exponential(1.0 / max(args.rate, 1e-9))
        t += gap
        if t < args.duration:
            arrivals.append(t)

    requests = []
    for i, t in enumerate(arrivals):
        prompt_text = prompt_pool[rng.integers(0, len(prompt_pool))]
        adapter_id  = adapters[rng.choice(len(adapters), p=weights)]
        # Apply chat template + tokenize. add_generation_prompt=True so the
        # model immediately produces the assistant turn.
        # In recent transformers, tokenize=True without return_dict=False yields a
        # BatchEncoding {input_ids, attention_mask}. We only want the bare ids.
        token_ids = tok.apply_chat_template(
            [{"role": "user", "content": prompt_text}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )
        requests.append({
            "id":           f"req_{i:05d}",
            "arrival_time": float(t),
            "adapter_id":   adapter_id,
            "prompt":       prompt_text,
            "input_tokens": [int(x) for x in token_ids],
            "max_output":   int(args.max_output),
        })
    return requests


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer", required=True,
                   help="HF model id or local path; tokenizer must have an instruct chat_template")
    p.add_argument("--adapters", nargs="+", required=True,
                   help="adapter ids (matching <adapter_dir>/<id>.gguf on the device)")
    p.add_argument("--duration", type=float, default=60.0, help="trace span in seconds (default: 60)")
    p.add_argument("--rate", type=float, default=0.5, help="Poisson arrival rate, req/s (default: 0.5)")
    p.add_argument("--pareto-alpha", type=float, default=1.5,
                   help="adapter Pareto skew; 0 = uniform (default: 1.5)")
    p.add_argument("--max-output", type=int, default=128, help="max decode tokens per request (default: 128)")
    p.add_argument("--prompts-file", type=Path, default=None,
                   help="optional JSON array of prompt strings; otherwise built-in pool is used")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True, help="output trace JSON path")
    args = p.parse_args()

    requests = generate(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(requests, f, indent=2)
    print(f"[gen] wrote {len(requests)} requests to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
