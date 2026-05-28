"""
Benchmark script: index `examples/demo_repo`, run the ground-truth Q&A
set, and write metrics into docs/metrics.md.

What we measure
---------------

* Indexing latency        -- wall-clock from `create_repo_from_directory`
                             start to RepoIndex return.
* `ask` latency           -- median and p95 over all questions.
* Vector store size       -- bytes on disk after indexing.
* RAG accuracy            -- fraction of questions whose answer's source
                             citations include at least one expected
                             module from `qa_groundtruth.json`.

How to run
----------

    # Real run (needs ANTHROPIC_API_KEY, sentence-transformers, chromadb):
    python -m scripts.benchmark

    # With a custom output location:
    python -m scripts.benchmark --out docs/metrics.md

We deliberately don't add a `--mock` flag here -- the whole point of
the benchmark is end-to-end real behaviour. For correctness-style tests
of the same components, see `tests/`.
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# Allow `python scripts/benchmark.py` even without an editable install.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.repo_service import RepoService  # noqa: E402
from src.core.repo_store import RepoStore  # noqa: E402
from src.indexing.vector_store import VectorStore  # noqa: E402


DEMO_REPO = REPO_ROOT / "examples" / "demo_repo"
GROUNDTRUTH = REPO_ROOT / "examples" / "qa_groundtruth.json"
DEFAULT_OUT = REPO_ROOT / "docs" / "metrics.md"


def main() -> int:
    args = _parse_args()
    storage_dir = Path(args.storage_dir or (REPO_ROOT / ".benchmark_storage"))

    # Fresh storage so the benchmark is reproducible run-to-run.
    if storage_dir.exists():
        shutil.rmtree(storage_dir)
    storage_dir.mkdir(parents=True)

    service = RepoService(
        repo_store=RepoStore(storage_dir=storage_dir),
        vector_store=VectorStore(storage_dir=storage_dir),
    )

    metrics: dict[str, Any] = {}

    # 1. Indexing
    print(f"Indexing {DEMO_REPO} ...")
    t0 = time.perf_counter()
    repo_index = service.create_repo_from_directory(DEMO_REPO, name="demo")
    metrics["index_seconds"] = round(time.perf_counter() - t0, 3)
    metrics["file_count"] = repo_index.file_count
    metrics["module_count"] = repo_index.module_count
    metrics["chunk_count"] = repo_index.chunk_count
    print(
        f"  indexed in {metrics['index_seconds']}s: "
        f"{metrics['module_count']} modules, {metrics['chunk_count']} chunks"
    )

    # 2. Vector store size on disk
    metrics["vector_store_bytes"] = _dir_size(storage_dir / "chromadb")

    # 3. Q&A latency + accuracy
    groundtruth = json.loads(GROUNDTRUTH.read_text(encoding="utf-8"))
    latencies: list[float] = []
    correct = 0
    per_question: list[dict] = []

    for entry in groundtruth["questions"]:
        question = entry["question"]
        expected = set(entry["expected_modules"])
        t1 = time.perf_counter()
        resp = service.ask(repo_index.repo_id, question)
        latencies.append(time.perf_counter() - t1)

        cited = {s.module_path for s in resp.sources}
        hit = bool(expected & cited)
        if hit:
            correct += 1
        per_question.append({
            "question": question,
            "expected": sorted(expected),
            "cited": sorted(cited),
            "correct": hit,
        })
        marker = "OK" if hit else "MISS"
        print(f"  [{marker}] {question}")

    metrics["ask_count"] = len(latencies)
    metrics["ask_median_ms"] = round(statistics.median(latencies) * 1000, 1)
    metrics["ask_p95_ms"] = round(_percentile(latencies, 95) * 1000, 1)
    metrics["rag_accuracy"] = round(correct / len(latencies), 3)

    # 4. Report
    out_path = Path(args.out or DEFAULT_OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(_render_report(metrics, per_question), encoding="utf-8")
    print(f"\nWrote report to {out_path}")
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", help="Path to write the metrics report")
    p.add_argument("--storage-dir", help="Path for benchmark storage (wiped)")
    return p.parse_args()


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _percentile(values: list[float], pct: float) -> float:
    """Plain percentile, no numpy dependency."""
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (pct / 100)
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _render_report(metrics: dict, per_question: list[dict]) -> str:
    lines: list[str] = [
        "# Benchmark Results",
        "",
        "Measured on `examples/demo_repo` with the ground-truth Q&A set "
        "in `examples/qa_groundtruth.json`.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Files indexed | {metrics['file_count']} |",
        f"| Modules indexed | {metrics['module_count']} |",
        f"| Chunks created | {metrics['chunk_count']} |",
        f"| Index time | {metrics['index_seconds']} s |",
        f"| Vector store on disk | {metrics['vector_store_bytes']} bytes |",
        f"| Q&A questions | {metrics['ask_count']} |",
        f"| `ask` latency (median) | {metrics['ask_median_ms']} ms |",
        f"| `ask` latency (p95) | {metrics['ask_p95_ms']} ms |",
        f"| RAG citation accuracy | {metrics['rag_accuracy']:.0%} |",
        "",
        "## Per-question breakdown",
        "",
        "| # | Question | Expected | Cited | Hit |",
        "|---|---|---|---|---|",
    ]
    for i, q in enumerate(per_question, start=1):
        lines.append(
            f"| {i} | {q['question']} | "
            f"{', '.join(q['expected'])} | "
            f"{', '.join(q['cited']) or '(none)'} | "
            f"{'OK' if q['correct'] else 'MISS'} |"
        )
    lines.append("")
    lines.append(
        "Accuracy is computed as the fraction of answers whose source "
        "citations include at least one of the expected modules. This is a "
        "looser metric than 'exact match' but a tighter one than 'the LLM "
        "said something' -- it specifically checks that retrieval surfaced "
        "the right code."
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
