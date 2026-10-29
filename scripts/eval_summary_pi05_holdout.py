"""Per-task success-rate breakdown for the pi0.5 LIBERO holdout/TACO experiment.

Reads the per-suite `results_<suite>.json` files produced by
`scripts/eval_pi05_libero40.sh` (i.e. examples/libero/main.py) and splits the
per-task entries into PRETRAIN (the n-per-suite tasks chosen by seed) vs HOLDOUT
(the rest). Prints task-by-task numbers, per-suite aggregates within each group,
and the pretrain-holdout delta.

The split is regenerated from `--split-seed` by importing
`libero_task_split.sample_pretrain_tasks` -- the SAME module the pi0.5 trainer
(scripts/train.py) imports, so analyzer and trainer can never disagree
on which tasks are pretrain vs holdout.

How this differs from the smolvla analyzer (scripts/eval_summary_holdout.py):
  * Input is the four `results_<suite>.json` files (one per suite), not a single
    `eval_info.json`.
  * The pi0.5 eval already keys `per_task` by the task LANGUAGE STRING (with
    {successes, episodes, success_rate}), so we map language -> dataset
    task_index directly via the dataset metadata. No env<->dataset task_id remap
    and no tasks.parquet dependency are needed.

Usage:
    uv run python scripts/eval_summary_pi05_holdout.py ./outputs/libero_eval/pretrain --split-seed 7

    # Single results file is also accepted:
    uv run python scripts/eval_summary_pi05_holdout.py .../pi05_bc_pretrain_15k/libero_spatial/results_libero_spatial.json

    # Override the holdout set explicitly (e.g. for a single-task finetune run):
    uv run python scripts/eval_summary_pi05_holdout.py .../<run> --holdout-task-ids 5 14

Output is plain text on stdout, sorted by suite -> task_id within each group.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Reuse the canonical split + suite mapping. Importing through scripts/ keeps the
# analyzer in lockstep with the trainer (same module train_pi05_taco.py uses).
sys.path.insert(0, str(Path(__file__).parent))
from openpi.training.task_split import (  # noqa: E402
    DEFAULT_REPO_ID,
    SUITE_TO_BENCHMARK,
    _LIBERO_SUITE_RANGES,
    sample_pretrain_tasks,
    task_index_to_string,
)

# Print order for suites (dataset suite names).
SUITE_ORDER = list(_LIBERO_SUITE_RANGES)  # long, goal, object, spatial
# dataset task_index -> dataset suite name.
TASK_TO_SUITE = {tid: suite for suite, r in _LIBERO_SUITE_RANGES.items() for tid in r}
# benchmark suite name (e.g. "libero_10") -> dataset suite name ("long").
BENCHMARK_TO_SUITE = {v: k for k, v in SUITE_TO_BENCHMARK.items()}


def _norm(s: str) -> str:
    """Normalize a language string for matching (eval keys vs dataset metadata)."""
    return s.strip().lower()


def find_results_files(eval_path: Path) -> list[Path]:
    """Accept either a single results_*.json file or a run directory; return all
    per-suite results files found (searched recursively under a directory)."""
    if eval_path.is_file():
        return [eval_path]
    if not eval_path.exists():
        raise SystemExit(f"eval path does not exist: {eval_path}")
    files = sorted(eval_path.rglob("results_*.json"))
    if not files:
        raise SystemExit(
            f"no results_*.json found under {eval_path} "
            f"(expected files like <suite>/results_<suite>.json from eval_pi05_libero40.sh)"
        )
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_path", type=Path,
                    help="Run directory (containing <suite>/results_<suite>.json) or a single results file.")
    ap.add_argument("--split-seed", type=int, default=7,
                    help="Same seed used by train_pi05_taco.py. Default 7.")
    ap.add_argument("--tasks-per-suite", type=int, default=8,
                    help="Tasks per suite that went to pretrain. Default 8 -> 32 pretrain + 8 holdout.")
    ap.add_argument("--holdout-task-ids", type=int, nargs="+", default=None,
                    help="Override the holdout set with explicit dataset task_index values (0..39). "
                         "Useful for single-task finetune runs.")
    ap.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID,
                    help=f"LeRobot dataset whose task_index<->language mapping is used. Default {DEFAULT_REPO_ID}.")
    args = ap.parse_args()

    # The PRETRAIN bucket is ALWAYS the seed-derived 32-task pretrain set (the tasks the
    # pretrain stage actually trained on) — it does NOT depend on --holdout-task-ids.
    # --holdout-task-ids only redefines the HOLDOUT bucket (e.g. a single task that was
    # finetuned). This matters when you finetune on ONE of the 8 held-out tasks: pretrain
    # stays the 32 (forgetting reference), holdout = that one task, and the other 7 held-out
    # tasks belong to NEITHER bucket (they were neither pretrained nor finetuned), so they
    # are excluded from both — not silently folded into "pretrain".
    seed_pretrain, seed_holdout = sample_pretrain_tasks(args.tasks_per_suite, args.split_seed)
    pretrain_ids = sorted(seed_pretrain)
    if args.holdout_task_ids is not None:
        holdout_ids = sorted({int(t) for t in args.holdout_task_ids})
        bad = [t for t in holdout_ids if t not in TASK_TO_SUITE]
        if bad:
            raise SystemExit(f"--holdout-task-ids out of range (must be 0..39): {bad}")
        split_note = (f"seed={args.split_seed} pretrain (n={len(pretrain_ids)}) + "
                      f"explicit holdout override (n_holdout={len(holdout_ids)})")
    else:
        holdout_ids = sorted(seed_holdout)
        split_note = f"seed={args.split_seed}, tasks_per_suite={args.tasks_per_suite}"
    # If an overridden holdout id happens to fall in the seed pretrain set, count it as
    # holdout only (don't double-count).
    holdout_set = set(holdout_ids)
    pretrain_set = set(pretrain_ids) - holdout_set

    # language string -> dataset task_index (normalized). Same metadata source the
    # split module uses to turn task indices into the strings eval matches on.
    idx2str = task_index_to_string(args.repo_id)
    lang2idx = {_norm(s): tid for tid, s in idx2str.items()}
    if len(lang2idx) != len(idx2str):
        raise SystemExit("duplicate normalized task strings in dataset metadata; cannot map language -> index")

    # Aggregate per dataset task_index: (suite, tid) -> [successes, episodes].
    agg: dict[int, list[int]] = {}
    files = find_results_files(args.eval_path)
    for f in files:
        data = json.loads(f.read_text())
        per_task = data.get("per_task", {})
        if not per_task:
            print(f"[warn] {f} has empty per_task; skipping", file=sys.stderr)
            continue
        for lang, m in per_task.items():
            key = _norm(lang)
            if key not in lang2idx:
                raise SystemExit(
                    f"eval task language not found in dataset '{args.repo_id}': '{lang}' (from {f.name}). "
                    f"Dataset and eval are out of sync."
                )
            tid = lang2idx[key]
            s, e = int(m["successes"]), int(m["episodes"])
            if tid in agg:
                agg[tid][0] += s
                agg[tid][1] += e
            else:
                agg[tid] = [s, e]

    if not agg:
        raise SystemExit("no per-task rows parsed from results files")

    def pct(succ: int, ep: int) -> float:
        return 100.0 * succ / ep if ep else float("nan")

    def print_group(group_name: str, ids_set: set[int]):
        members = sorted(
            [(TASK_TO_SUITE[tid], tid, agg[tid]) for tid in agg if tid in ids_set],
            key=lambda r: (SUITE_ORDER.index(r[0]), r[1]),
        )
        if not members:
            print(f"\n{group_name}: no eval rows match this split")
            return None, None, None
        print(f"\n{group_name} ({len(members)} tasks present)")
        print(f"  {'suite':<8} {'task_id':>7}   {'success':>8} {'n':>5}")
        for suite, tid, (s, e) in members:
            print(f"  {suite:<8} {tid:>7}   {pct(s, e):7.1f}% {e:>5}")
        # Per-suite aggregate within the group.
        by_suite: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for suite, _tid, (s, e) in members:
            by_suite[suite][0] += s
            by_suite[suite][1] += e
        print(f"  --- per-suite within {group_name.lower()} ---")
        for suite in SUITE_ORDER:
            if suite in by_suite:
                s, e = by_suite[suite]
                print(f"  {suite:<8}             {pct(s, e):7.1f}% {e:>5}")
        tot_s = sum(s for _, _, (s, _e) in members)
        tot_e = sum(e for _, _, (_s, e) in members)
        print(f"  OVERALL {group_name.lower():<13} {pct(tot_s, tot_e):7.1f}% {tot_e:>5}")
        return pct(tot_s, tot_e), tot_s, tot_e

    print(f"\n== {args.eval_path} ==")
    print(f"split: {split_note}  |  repo_id={args.repo_id}  |  files={len(files)}")
    n_expected = len(pretrain_set) + len(holdout_set)
    if len(agg) != n_expected:
        print(f"[note] {len(agg)} tasks present in results (split defines {n_expected}); "
              f"reporting only what is present.")

    pre_pc, _ps, _pe = print_group("PRETRAIN", pretrain_set)
    hold_pc, _hs, _he = print_group("HOLDOUT", holdout_set)

    # Overall across everything present (sanity vs the eval's own summary).
    all_s = sum(s for s, _e in agg.values())
    all_e = sum(e for _s, e in agg.values())
    print(f"\nALL PRESENT ({len(agg)} tasks)      {pct(all_s, all_e):7.1f}% {all_e:>5}")

    if pre_pc is not None and hold_pc is not None:
        print(f"\nΔ (pretrain - holdout) = {pre_pc - hold_pc:+.1f} pp   "
              f"[pretrain {pre_pc:.1f}%, holdout {hold_pc:.1f}%]")


if __name__ == "__main__":
    main()
