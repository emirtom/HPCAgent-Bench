#!/usr/bin/env python3
# Copyright 2026 ETH Zurich and the HPCAgent-Bench authors.
# SPDX-License-Identifier: GPL-3.0-or-later
"""Agent-shaped prefix-cache probe for an OpenAI-compatible vLLM or SGLang server: one table row per concurrency.

Each stream is one conversation with its OWN long prefix (no two streams share more than a few tokens), resent every
round with the previous reply and a fresh user turn appended, as an agent's turns are. The working set is therefore
streams x context, and the row shows whether the server's KV pool holds it: prefix-cache hit (from the server's
/metrics, over this level's warm rounds only), aggregate output tok/s, TTFT, and pool / working set. A level whose ratio
is below about 1 evicts its own prefixes and thrashes. Stdlib only, so it runs in any image.

    python3 prefix-cache-probe.py --base http://localhost:8000 --model q38 --concurrency 1 10 20 40
"""

import argparse
import concurrent.futures
import json
import random
import re
import statistics
import threading
import time
import urllib.request

#: Words the prefixes are drawn from; random words keep the tokenizer from merging a prefix into a few tokens.
WORDS = [
    "kernel",
    "stencil",
    "vector",
    "matrix",
    "tile",
    "cache",
    "loop",
    "stride",
    "buffer",
    "thread",
    "block",
    "warp",
    "reduce",
    "scatter",
    "gather",
    "fused",
    "halo",
    "layout",
    "pointer",
    "index",
    "offset",
    "bound",
    "alias",
    "barrier",
    "atomic",
    "latency",
    "bandwidth",
    "register",
    "spill",
    "unroll",
    "pipeline",
]
#: Characters per token the text is sized by (English-like words with spaces, close for Qwen and GPT tokenizers).
CHARS_PER_TOKEN = 4.2


def text(rng: random.Random, tokens: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(int(tokens * CHARS_PER_TOKEN / 6.5)))


def post_stream(base: str, body: dict[str, object], timeout: float) -> tuple[float, str, int]:
    """(TTFT s, reply text, completion tokens) of one streamed chat completion."""
    request = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps({**body, "stream": True, "stream_options": {"include_usage": True}}).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.monotonic()
    first = 0.0
    parts: list[str] = []
    completion = 0
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            event = json.loads(line[5:])
            usage = event.get("usage")
            if usage:
                completion = int(usage.get("completion_tokens") or 0)
            for choice in event.get("choices") or []:
                delta = choice.get("delta") or {}
                piece = (delta.get("content") or "") + (delta.get("reasoning_content") or delta.get("reasoning") or "")
                if piece and not first:
                    first = time.monotonic() - start
                parts.append(delta.get("content") or "")
    return first, "".join(parts), completion


def metrics(base: str) -> str:
    with urllib.request.urlopen(f"{base}/metrics", timeout=30) as response:
        return response.read().decode()


def counter(text_: str, name: str) -> float:
    """Sum of every sample of metric ``name`` (all label sets), 0 when absent."""
    pattern = re.compile(rf"^{re.escape(name)}(?:\{{[^}}]*\}})? ([0-9.eE+-]+)$", re.MULTILINE)
    return sum(float(value) for value in pattern.findall(text_))


def hit_counters(text_: str) -> tuple[float, float]:
    """(hit tokens, queried tokens): vLLM's prefix-cache counters, else SGLang's cached / prompt token counters."""
    for hits, queries in (
        ("vllm:prefix_cache_hits_total", "vllm:prefix_cache_queries_total"),
        ("vllm:prefix_cache_hits", "vllm:prefix_cache_queries"),
        ("sglang:cached_tokens_total", "sglang:prompt_tokens_total"),
    ):
        if counter(text_, queries) > 0:
            return counter(text_, hits), counter(text_, queries)
    return 0.0, 0.0


def kv_pool_tokens(text_: str) -> int:
    """The server's KV pool in tokens: vLLM's cache_config_info labels, else SGLang's max_total_num_tokens."""
    info = re.search(r"^vllm:cache_config_info\{([^}]*)\}", text_, re.MULTILINE)
    if info:
        labels = dict(re.findall(r'(\w+)="([^"]*)"', info.group(1)))
        blocks, size = labels.get("num_gpu_blocks", ""), labels.get("block_size", "")
        if blocks.isdigit() and size.isdigit():
            return int(blocks) * int(size)
    return int(counter(text_, "sglang:max_total_num_tokens"))


def conversation(
    args: argparse.Namespace, level: int, stream: int, warm: threading.Barrier
) -> list[tuple[float, int, float]]:
    """One stream's rounds: (TTFT, completion tokens, seconds) per round; every stream waits at ``warm`` after its
    cold first round, so the hit counters snapshot there cover warm rounds only."""
    rng = random.Random(f"{args.seed}-{level}-{stream}")
    messages: list[dict[str, str]] = [
        {"role": "system", "content": f"conversation {level}-{stream}. " + text(rng, args.prefix_tokens)},
    ]
    rounds = []
    for turn in range(args.rounds):
        messages.append({"role": "user", "content": f"turn {turn}. " + text(rng, args.turn_tokens)})
        body = {
            "model": args.model,
            "messages": messages,
            "max_tokens": args.max_tokens,
            "temperature": 0.6,
            "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        start = time.monotonic()
        ttft, reply, completion = post_stream(args.base, body, args.timeout)
        rounds.append((ttft, completion, time.monotonic() - start))
        messages.append({"role": "assistant", "content": reply})
        if turn == 0:
            warm.wait()
    return rounds


def level_row(args: argparse.Namespace, level: int, pool: int) -> dict[str, float]:
    before: list[tuple[float, float]] = []
    warm = threading.Barrier(level, action=lambda: before.append(hit_counters(metrics(args.base))))
    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=level) as pool_:
        results = list(pool_.map(lambda stream: conversation(args, level, stream, warm), range(level)))
    elapsed = time.monotonic() - start
    after = hit_counters(metrics(args.base))
    ttfts = sorted(r[0] for rounds in results for r in rounds[1:]) or [0.0]
    context = args.prefix_tokens + args.rounds * (args.turn_tokens + args.max_tokens)
    queried = after[1] - before[0][1]
    return {
        "concurrency": level,
        "output_tok_s": sum(r[1] for rounds in results for r in rounds) / elapsed,
        "per_stream_tok_s": statistics.mean(
            sum(r[1] for r in rounds) / sum(r[2] for r in rounds) for rounds in results
        ),
        "ttft_median_s": statistics.median(ttfts),
        "ttft_p99_s": ttfts[min(len(ttfts) - 1, int(0.99 * len(ttfts)))],
        "prefix_hit": (after[0] - before[0][0]) / queried if queried else float("nan"),
        "pool_over_working_set": pool / (level * context) if pool else float("nan"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", default="http://localhost:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 10, 20, 40])
    parser.add_argument("--prefix-tokens", type=int, default=38000, help="each stream's own prefix (agent p50 ~38k)")
    parser.add_argument("--turn-tokens", type=int, default=1500, help="new user tokens per round (tool results)")
    parser.add_argument("--max-tokens", type=int, default=400, help="reply tokens per round")
    parser.add_argument("--rounds", type=int, default=4, help="round 0 is cold; the hit rate covers the rest")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--seed", default="probe")
    args = parser.parse_args()
    pool = kv_pool_tokens(metrics(args.base))
    print(f"KV pool: {pool} tokens" if pool else "KV pool: unknown (no cache_config_info / max_total_num_tokens)")
    print("conc  out tok/s  per-stream  TTFT med  TTFT p99  prefix hit  pool/working set")
    for level in args.concurrency:
        row = level_row(args, level, pool)
        print(
            f"{level:4d}  {row['output_tok_s']:9.1f}  {row['per_stream_tok_s']:10.1f}  {row['ttft_median_s']:7.2f}s"
            f"  {row['ttft_p99_s']:7.2f}s  {row['prefix_hit']:10.1%}  {row['pool_over_working_set']:16.2f}",
            flush=True,
        )
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
