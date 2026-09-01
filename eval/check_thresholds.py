"""
Retrieval regression gate for CI.

Runs the default configuration against the golden set and fails the build if any
metric has dropped below its floor. The floors sit a little under the measured
values, so ordinary variation does not break the build but a real regression does.

Raise the floors when a change genuinely improves retrieval — that is what keeps
the gate meaningful instead of decorative.

Usage:
  python eval/check_thresholds.py
  python eval/check_thresholds.py --update   # rewrite floors from current results
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.run_eval import build_store, evaluate, load_golden  # noqa: E402
from ragbot.config import RagConfig  # noqa: E402

FLOORS_PATH = ROOT / "eval" / "thresholds.json"
# How far below the recorded value a run may drift before it counts as a regression.
TOLERANCE = 0.02


def measure() -> dict:
    config = RagConfig()
    cases = load_golden()
    store = build_store(config)
    result = evaluate(store, cases, config)
    return {
        "recall@1": result["recall@1"],
        "recall@5": result["recall@5"],
        "mrr@10": result["mrr@10"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true", help="rewrite the floors")
    args = parser.parse_args()

    current = measure()

    if args.update or not FLOORS_PATH.exists():
        FLOORS_PATH.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
        print("recorded floors:")
        for name, value in current.items():
            print(f"  {name:<10} {value:.3f}")
        return 0

    floors = json.loads(FLOORS_PATH.read_text(encoding="utf-8"))
    failures = []
    for name, floor in floors.items():
        value = current.get(name)
        if value is None:
            continue
        status = "ok"
        if value < floor - TOLERANCE:
            status = "REGRESSED"
            failures.append((name, value, floor))
        print(f"  {name:<10} {value:.3f}  (floor {floor:.3f} ± {TOLERANCE})  {status}")

    if failures:
        print("\nRetrieval regressed:")
        for name, value, floor in failures:
            print(f"  {name}: {value:.3f} < {floor:.3f} - {TOLERANCE}")
        print("\nEither fix the regression, or, if the drop is justified, "
              "re-record the floors with --update and say why in the commit.")
        return 1

    print("\nNo regression.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
