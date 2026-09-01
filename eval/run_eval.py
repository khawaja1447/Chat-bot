"""
Retrieval evaluation.

Measures the retriever alone — no LLM call, no API key, deterministic. That is
deliberate: generation quality is downstream of retrieval, and if the right
passage never reaches the prompt no amount of prompt engineering recovers it.

  Recall@k  fraction of questions with at least one correct chunk in the top k.
            "Correct" means the chunk came from a page that actually contains
            the answer, per eval/golden_set.yaml.
  MRR@10    mean of 1/rank of the first correct chunk. Rewards putting the right
            passage first, not merely somewhere in the list.

Usage:
  python eval/run_eval.py                 # compare retrieval strategies
  python eval/run_eval.py --sweep-chunks  # also sweep chunk size
  python eval/run_eval.py --write         # update docs/EVALUATION.md
"""

from __future__ import annotations

import argparse
import io
import pathlib
import statistics
import sys
import time

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ragbot.config import RagConfig  # noqa: E402
from ragbot.ingest import chunk_document, extract_document  # noqa: E402
from ragbot.retrieval import retrieve  # noqa: E402
from ragbot.store import DocumentStore  # noqa: E402

SAMPLES = ROOT / "samples"
GOLDEN = ROOT / "eval" / "golden_set.yaml"
REPORT = ROOT / "docs" / "EVALUATION.md"

MEASURE_DEPTH = 10


def load_golden() -> list:
    return yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))


def build_store(config: RagConfig) -> DocumentStore:
    chunks = []
    for pdf in sorted(SAMPLES.glob("*.pdf")):
        with open(pdf, "rb") as fh:
            document = extract_document(io.BytesIO(fh.read()), pdf.name)
        chunks.extend(chunk_document(document, config))
    return DocumentStore.from_chunks(chunks)


def is_relevant(hit, case) -> bool:
    return hit.chunk.doc == case["doc"] and hit.chunk.page in case["pages"]


def evaluate(store: DocumentStore, cases: list, config: RagConfig) -> dict:
    """Run every case and return aggregate metrics plus per-case detail."""
    measure_config = config.variant(final_k=MEASURE_DEPTH)
    recall_at = {1: [], 3: [], 5: []}
    reciprocal_ranks = []
    latencies = []
    per_kind = {}
    failures = []

    for case in cases:
        t0 = time.perf_counter()
        hits = retrieve(store, case["question"], measure_config)
        latencies.append((time.perf_counter() - t0) * 1000)

        flags = [is_relevant(h, case) for h in hits]
        first = next((i + 1 for i, ok in enumerate(flags) if ok), None)

        for k in recall_at:
            recall_at[k].append(1.0 if any(flags[:k]) else 0.0)
        reciprocal_ranks.append(1.0 / first if first else 0.0)

        kind = case.get("kind", "other")
        per_kind.setdefault(kind, []).append(1.0 if any(flags[:5]) else 0.0)

        if not any(flags[:5]):
            failures.append(
                {
                    "id": case["id"],
                    "question": case["question"],
                    "expected": f"{case['doc']} p.{case['pages']}",
                    "got": [h.citation for h in hits[:3]],
                }
            )

    return {
        "label": config.label,
        "recall@1": statistics.mean(recall_at[1]),
        "recall@3": statistics.mean(recall_at[3]),
        "recall@5": statistics.mean(recall_at[5]),
        "mrr@10": statistics.mean(reciprocal_ranks),
        "p50_ms": statistics.median(latencies),
        "per_kind": {k: statistics.mean(v) for k, v in sorted(per_kind.items())},
        "failures": failures,
        "n": len(cases),
    }


def format_table(rows: list, first_col: str = "Configuration") -> str:
    header = (
        f"| {first_col} | Recall@1 | Recall@3 | Recall@5 | MRR@10 | p50 latency |\n"
        "|---|---|---|---|---|---|\n"
    )
    body = ""
    for r in rows:
        body += (
            f"| {r['label']} | {r['recall@1']:.3f} | {r['recall@3']:.3f} | "
            f"{r['recall@5']:.3f} | {r['mrr@10']:.3f} | {r['p50_ms']:.0f} ms |\n"
        )
    return header + body


def format_kind_table(rows: list) -> str:
    kinds = sorted({k for r in rows for k in r["per_kind"]})
    header = "| Configuration | " + " | ".join(kinds) + " |\n"
    header += "|---" * (len(kinds) + 1) + "|\n"
    body = ""
    for r in rows:
        cells = " | ".join(f"{r['per_kind'].get(k, float('nan')):.2f}" for k in kinds)
        body += f"| {r['label']} | {cells} |\n"
    return header + body


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep-chunks", action="store_true")
    parser.add_argument("--sweep-candidates", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    cases = load_golden()
    base = RagConfig()
    print(f"{len(cases)} questions over {len(list(SAMPLES.glob('*.pdf')))} documents\n")

    store = build_store(base)
    stats = store.stats()
    print(f"corpus: {stats['documents']} docs, {stats['pages']} pages, {stats['chunks']} chunks\n")

    arms = [
        base.variant(use_hybrid=False, use_reranker=False),
        base.variant(use_hybrid=True, use_reranker=False),
        base.variant(use_hybrid=False, use_reranker=True),
        base.variant(use_hybrid=True, use_reranker=True),
    ]

    rows = []
    for config in arms:
        result = evaluate(store, cases, config)
        rows.append(result)
        print(
            f"{result['label']:<16} R@1 {result['recall@1']:.3f}  "
            f"R@5 {result['recall@5']:.3f}  MRR {result['mrr@10']:.3f}  "
            f"{result['p50_ms']:.0f} ms"
        )

    print("\n" + format_table(rows))
    print("Recall@5 by question type:\n")
    print(format_kind_table(rows))

    best = max(rows, key=lambda r: (r["mrr@10"], r["recall@5"]))
    print(f"best: {best['label']}")
    if best["failures"]:
        print(f"\nremaining failures ({len(best['failures'])}):")
        for f in best["failures"]:
            print(f"  [{f['id']}] {f['question']}")
            print(f"      expected {f['expected']}, got {f['got']}")

    cand_rows = []
    if args.sweep_candidates:
        print("\nreranker candidate depth sweep (hybrid on):")
        for depth in (5, 10, 20, 40, 60):
            config = base.variant(
                rerank_candidates=depth, dense_k=max(depth, 20), sparse_k=max(depth, 20)
            )
            result = evaluate(store, cases, config)
            result["label"] = f"top {depth}"
            cand_rows.append(result)
            print(
                f"  {depth:>3} candidates  R@1 {result['recall@1']:.3f}  "
                f"MRR {result['mrr@10']:.3f}  {result['p50_ms']:.0f} ms"
            )
        print("\n" + format_table(cand_rows, first_col="Candidates reranked"))

    chunk_rows = []
    if args.sweep_chunks:
        print("\nchunk size sweep (hybrid + rerank):")
        for size in (600, 900, 1200, 1800, 2500):
            config = base.variant(chunk_size=size)
            swept = build_store(config)
            result = evaluate(swept, cases, config)
            result["label"] = f"{size} chars"
            chunk_rows.append(result)
            print(
                f"  {size:>5} chars  R@1 {result['recall@1']:.3f}  "
                f"R@5 {result['recall@5']:.3f}  MRR {result['mrr@10']:.3f}"
            )
        print("\n" + format_table(chunk_rows, first_col="Chunk size"))

    if args.write:
        write_report(rows, chunk_rows, cand_rows, stats, len(cases), best)
        print(f"\nwrote {REPORT.relative_to(ROOT)}")
    return 0


def write_report(rows, chunk_rows, cand_rows, stats, n_cases, best) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    dense = next(r for r in rows if r["label"] == "dense-only")

    def lift(metric: str) -> float:
        return (best[metric] - dense[metric]) / max(dense[metric], 1e-9) * 100

    lift_r1, lift_r5, lift_mrr = lift("recall@1"), lift("recall@5"), lift("mrr@10")

    parts = [
        "# Retrieval Evaluation\n",
        "_Generated by `python eval/run_eval.py --write`. Do not edit by hand._\n",
        f"\n**Corpus:** {stats['documents']} documents, {stats['pages']} pages, "
        f"{stats['chunks']} chunks.  \n"
        f"**Questions:** {n_cases}, each labelled with the document and page that "
        "actually contains the answer (`eval/golden_set.yaml`).\n",
        "\nRetrieval is measured on its own — no LLM call, no API key, deterministic. "
        "If the right passage never reaches the prompt, prompt engineering cannot "
        "recover it.\n",
        "\n## Metrics\n",
        "- **Recall@k** — share of questions with at least one correct chunk in the top k.\n"
        "- **MRR@10** — mean of 1/rank of the first correct chunk; rewards ranking it first.\n",
        "\n## Retrieval strategies\n\n",
        format_table(rows),
        f"\n**{best['label']}** scores highest: Recall@1 **{lift_r1:+.1f}%**, "
        f"Recall@5 **{lift_r5:+.1f}%**, MRR **{lift_mrr:+.1f}%** against the dense-only "
        f"baseline, for about {best['p50_ms'] - dense['p50_ms']:+.0f} ms of median latency.\n"
        "\nRecall@1 is the metric to watch. Recall@5 saturates on a corpus this size — "
        "five chunks out of a few dozen is a large share of it — so a high Recall@5 here "
        "says more about the corpus than about the retriever.\n",
        "\n### Recall@5 by question type\n\n",
        format_kind_table(rows),
        "\n`paraphrase` asks for a fact in words the document never uses; `distractor` "
        "has a near-identical passage elsewhere in the corpus; `numeric` requires the "
        "right figure among many nearby ones.\n",
    ]
    if chunk_rows:
        parts += [
            "\n## Chunk size\n\nMeasured with hybrid retrieval and reranking.\n\n",
            format_table(chunk_rows, first_col="Chunk size"),
            "\nThis sweep is why the default is 900 characters rather than a round "
            "1000 or the 2500 an earlier version of this project used. The difference "
            "between the best and worst setting here is larger than the difference "
            "between two of the retrieval strategies above.\n",
        ]
    if cand_rows:
        parts += [
            "\n## Reranker candidate depth\n\n"
            "How many fused candidates the cross-encoder rescores before the top "
            f"{RagConfig().final_k} go to the model.\n\n",
            format_table(cand_rows, first_col="Candidates reranked"),
            "\nDepth buys nothing past about 10 on this corpus and costs latency "
            "roughly linearly, because the reranker runs once per candidate. The "
            "default is 10.\n",
        ]
    if best["failures"]:
        parts += [
            f"\n## Remaining failures ({len(best['failures'])})\n\n",
            "Cases the best configuration still misses at k=5. Kept visible on "
            "purpose — an eval you only read when it is green is decoration.\n\n",
        ]
        for f in best["failures"]:
            parts.append(
                f"- **{f['id']}** — _{f['question']}_  \n"
                f"  expected `{f['expected']}`, got `{', '.join(f['got'])}`\n"
            )
    REPORT.write_text("".join(parts), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
