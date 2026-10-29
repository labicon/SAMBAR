"""Verify the LIBERO-40 pretrain/holdout split and (optionally) that an eval directory's
per-task results bucket into it consistently.

Part A — split correctness (no eval data needed): for sample_pretrain_tasks(tasks_per_suite,
seed), checks determinism, disjoint pretrain/holdout, full 0..39 coverage, sorted+unique,
and per-suite counts (tasks_per_suite pretrain + the rest holdout in each of the 4 suites).

Part B — eval consistency (with --eval-dir): every eval per_task LANGUAGE maps 1:1 onto a
dataset task_index (so the train/eval task_id PERMUTATION is irrelevant — we never use the
permuted env task_id), the dataset task_index blocks align with the benchmark suites, and
all 40 tasks are covered exactly once. This is the check that guards against the LIBERO
train/eval string mismatch silently mis-bucketing pretrain vs holdout.

Exit code 0 if all checks pass, 1 otherwise.

Usage:
    uv run python scripts/verify_split_consistency.py --seed 7
    uv run python scripts/verify_split_consistency.py --seed 7 \
        --eval-dir ./outputs/libero_eval/pretrain
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from openpi.training.task_split import (  # noqa: E402
    DEFAULT_REPO_ID,
    SUITE_TO_BENCHMARK,
    _LIBERO_SUITE_RANGES,
    sample_pretrain_tasks,
    task_index_to_string,
)


def _norm(s: str) -> str:
    return s.strip().lower()


def _block(tid: int) -> str:
    for s, r in _LIBERO_SUITE_RANGES.items():
        if tid in r:
            return s
    return "??"


def check_split(seed: int, nps: int, idx2str: dict[int, str] | None) -> tuple[bool, list[int], list[int]]:
    print(f"=== A) split correctness: sample_pretrain_tasks(tasks_per_suite={nps}, seed={seed}) ===")
    pre, hold = sample_pretrain_tasks(nps, seed)
    pre_s, hold_s = set(pre), set(hold)
    ok = True

    pre2, hold2 = sample_pretrain_tasks(nps, seed)
    det = pre2 == pre and hold2 == hold
    ok &= det
    print(f"  deterministic (re-sample identical):           {det}")

    disj = pre_s.isdisjoint(hold_s)
    ok &= disj
    print(f"  pretrain ∩ holdout == ∅:                        {disj}")

    cover = (pre_s | hold_s) == set(range(40))
    ok &= cover
    print(f"  pretrain ∪ holdout == all 40:                   {cover}  (|pre|={len(pre_s)}, |hold|={len(hold_s)})")

    sortuniq = pre == sorted(set(pre)) and hold == sorted(set(hold))
    ok &= sortuniq
    print(f"  both lists sorted & duplicate-free:             {sortuniq}")

    print(f"  per-suite counts (expect {nps} pretrain / {10 - nps} holdout per suite):")
    for s, r in _LIBERO_SUITE_RANGES.items():
        p = sorted(t for t in pre if t in r)
        h = sorted(t for t in hold if t in r)
        good = len(p) == nps and len(h) == 10 - nps and (set(p) | set(h)) == set(r)
        ok &= good
        print(f"    {s:<7} [{r[0]:>2}..{r[-1]:>2}]  pretrain={p}  holdout={h}  -> {'OK' if good else '*** BAD ***'}")

    if idx2str is not None:
        print("  holdout task languages:")
        for t in hold:
            print(f"    tid={t:>2} ({_block(t)}): '{idx2str.get(t, '??')}'")

    print(f"  --> SPLIT {'PASS' if ok else '*** FAIL ***'}\n")
    return ok, pre, hold


def check_eval(eval_dir: str, idx2str: dict[int, str], pre: list[int], hold: list[int]) -> bool:
    lang2idx = {_norm(s): t for t, s in idx2str.items()}
    ok = len(lang2idx) == len(idx2str)
    if not ok:
        print("  *** dataset has colliding normalized task strings ***")

    files = sorted(glob.glob(f"{eval_dir.rstrip('/')}/*/results_*.json"))
    if not files:
        print(f"=== B) eval consistency: no results_*.json under {eval_dir} — skipped ===\n")
        return ok

    print(f"=== B) eval consistency: {eval_dir} ({len(files)} suite files) ===")
    unmatched, all_tids, suite_file_tids, tid_to_file = [], [], {}, {}
    for f in files:
        d = json.loads(pathlib.Path(f).read_text())
        suite_file = d.get("task_suite", pathlib.Path(f).parent.name)
        tids = []
        for lang in d.get("per_task", {}):
            k = _norm(lang)
            if k not in lang2idx:
                unmatched.append((suite_file, lang))
                continue
            t = lang2idx[k]
            tids.append(t)
            all_tids.append(t)
            tid_to_file[t] = suite_file
        suite_file_tids[suite_file] = sorted(tids)

    dup = sorted(t for t in set(all_tids) if all_tids.count(t) > 1)
    distinct_ok = len(set(all_tids)) == 40 and not dup and not unmatched
    ok &= distinct_ok
    print(f"  1) language->task_index: unmatched={len(unmatched)}  duplicates={dup}  distinct={len(set(all_tids))}/40")
    for s, l in unmatched:
        print(f"       UNMATCHED [{s}] '{l}'")

    print("  2) benchmark suite file <-> dataset task_index block:")
    for dsblock, bench in SUITE_TO_BENCHMARK.items():
        r = _LIBERO_SUITE_RANGES[dsblock]
        tids = suite_file_tids.get(bench, [])
        good = bool(tids) and all(t in r for t in tids) and len(tids) == len(r)
        ok &= good
        print(f"     dataset '{dsblock}' [{r[0]}..{r[-1]}] <-> '{bench}': tids={tids} -> {'OK' if good else '*** MISMATCH ***'}")

    ev = set()
    for f in files:
        ev |= {_norm(l) for l in json.loads(pathlib.Path(f).read_text()).get("per_task", {})}
    ds = set(lang2idx)
    cover = ds == ev
    ok &= cover
    print(f"  3) coverage: dataset={len(ds)} eval={len(ev)}  "
          f"missing={[idx2str[lang2idx[l]] for l in sorted(ds - ev)]}  extra={sorted(ev - ds)}")

    # 4) show how the seed split lands in the eval files (eyeball check)
    print("  4) seed split landing (tid -> eval file):")
    for label, ids in [("PRETRAIN", pre), ("HOLDOUT", hold)]:
        miss = [t for t in ids if t not in tid_to_file]
        print(f"     {label}: {len(ids)} tasks, files={sorted({tid_to_file.get(t, 'MISSING') for t in ids})}"
              + (f"  *** missing from eval: {miss} ***" if miss else ""))

    print(f"  --> EVAL {'PASS' if ok else '*** FAIL ***'}\n")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=7, help="split seed (matches train_pi05_*.py --split-seed).")
    ap.add_argument("--tasks-per-suite", type=int, default=8, help="pretrain tasks per suite. Default 8.")
    ap.add_argument("--eval-dir", default=None, help="optional run dir with <suite>/results_<suite>.json to cross-check.")
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID, help=f"dataset for task_index<->language. Default {DEFAULT_REPO_ID}.")
    args = ap.parse_args()

    try:
        idx2str = task_index_to_string(args.repo_id)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] could not load dataset metadata ({e}); language columns/Part B disabled.")
        idx2str = None

    ok, pre, hold = check_split(args.seed, args.tasks_per_suite, idx2str)
    if args.eval_dir:
        if idx2str is None:
            print("=== B) eval consistency: needs dataset metadata — skipped ===")
        else:
            ok &= check_eval(args.eval_dir, idx2str, pre, hold)

    print(f"=== VERDICT: {'ALL PASS' if ok else '*** FAILURES ABOVE ***'} ===")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
