"""Single source of truth for the LIBERO-40 pretrain/holdout task split.

Both the training driver (scripts/train.py) and the evaluation driver
(scripts/evaluate.py) import from here, so a split can never silently
diverge between train and eval.

CRITICAL — index spaces differ between train and eval:
  * Training selects episodes by the LeRobot dataset's `task_index` (0..39), whose
    suite layout is clean blocks of 10 (see _LIBERO_SUITE_RANGES below).
  * Evaluation (examples/libero/main.py) iterates the LIBERO *benchmark library's*
    per-suite `task_id`, which is a PERMUTATION of the dataset's within-block order.
So a split defined by dataset task_index must be translated to task *language strings*
(`task_strings_for_ids`) and matched against `task.language` at eval time. Never pass
raw task numbers across the train↔eval boundary.

This module has no JAX/heavy deps so it is cheap to import from either venv.
"""

from __future__ import annotations

import logging
import random

# LIBERO-40 suite ranges (physical-intelligence/libero), task_index blocks of 10.
# Confirmed via meta/tasks.jsonl: 0-9 long, 10-19 goal, 20-29 object, 30-39 spatial.
_LIBERO_SUITE_RANGES: dict[str, range] = {
    "long": range(0, 10),     # benchmark suite "libero_10"
    "goal": range(10, 20),    # benchmark suite "libero_goal"
    "object": range(20, 30),  # benchmark suite "libero_object"
    "spatial": range(30, 40),  # benchmark suite "libero_spatial"
}

# Map dataset suite name -> LIBERO benchmark suite name (used by eval).
SUITE_TO_BENCHMARK: dict[str, str] = {
    "long": "libero_10",
    "goal": "libero_goal",
    "object": "libero_object",
    "spatial": "libero_spatial",
}

DEFAULT_REPO_ID = "physical-intelligence/libero"


def sample_pretrain_tasks(n_per_suite: int, seed: int) -> tuple[list[int], list[int]]:
    """Seeded random sample of `n_per_suite` task indices from each of the four LIBERO
    suites. Returns (pretrain_task_ids, holdout_task_ids), both globally sorted.
    Reproducible from `seed` alone — no state shared between phases."""
    rng = random.Random(seed)
    pretrain: list[int] = []
    holdout: list[int] = []
    for suite_name, suite_range in _LIBERO_SUITE_RANGES.items():
        ids = list(suite_range)
        if n_per_suite > len(ids):
            raise ValueError(f"suite {suite_name} has only {len(ids)} tasks, can't sample {n_per_suite}")
        chosen = sorted(rng.sample(ids, n_per_suite))
        pretrain.extend(chosen)
        holdout.extend(sorted(set(ids) - set(chosen)))
    return sorted(pretrain), sorted(holdout)


def task_index_to_string(repo_id: str = DEFAULT_REPO_ID) -> dict[int, str]:
    """{task_index: task_string} from the LeRobot dataset metadata."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id)
    return dict(meta.tasks)


def task_strings_for_ids(repo_id: str, task_ids: list[int]) -> tuple[str, ...]:
    """Map dataset task indices to the task language strings the data loader filters on
    (task_names_keep) and that eval matches against task.language."""
    idx2str = task_index_to_string(repo_id)
    missing = [t for t in task_ids if t not in idx2str]
    if missing:
        raise ValueError(f"task indices {missing} not present in {repo_id} (has {len(idx2str)} tasks)")
    return tuple(idx2str[t] for t in task_ids)


def describe_split(task_ids: list[int], label: str) -> None:
    """Log a per-suite breakdown of which task indices are in this split."""
    by_suite: dict[str, list[int]] = {s: [] for s in _LIBERO_SUITE_RANGES}
    for tid in task_ids:
        for s, r in _LIBERO_SUITE_RANGES.items():
            if tid in r:
                by_suite[s].append(tid)
                break
    logging.info(f"{label} task IDs ({len(task_ids)} total):")
    for s, tids in by_suite.items():
        logging.info(f"  {s:<8}: {tids}")
