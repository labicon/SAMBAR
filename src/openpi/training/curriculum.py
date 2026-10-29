"""LIBERO curriculum/data utilities extracted from the existing mixture driver."""

from __future__ import annotations

import logging
import pathlib
from typing import Any

import jax
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch
import tqdm

import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
from openpi.training.task_split import _LIBERO_SUITE_RANGES
from openpi.training.task_split import sample_pretrain_tasks
from openpi.training.task_split import task_strings_for_ids
import openpi.transforms as _transforms


class RemoveStrings(_transforms.DataTransformFn):
    def __call__(self, x):
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


LIBERO40_REPO_ID = "physical-intelligence/libero"

PRETRAIN_SUITES = ("goal", "object", "spatial")

DEFAULT_FINETUNE_TASKS = (
    "turn on the stove and put the moka pot on it",
    "put the white mug on the left plate and put the yellow and white mug on the right plate",
    "put both the alphabet soup and the cream cheese box in the basket",
)

PRETRAIN_HOLDOUT_TASKS = (
    "put the bowl on the plate",  # LIBERO-goal    (task_index 10)
    "pick up the alphabet soup and place it in the basket",  # LIBERO-object  (task_index 24)
    "pick up the black bowl from table center and place it on the plate",  # LIBERO-spatial (task_index 38)
)


def _make_lerobot_dataset(
    repo_id: str,
    root: str | None,
    action_horizon: int,
    data_config: _config.DataConfig,
    task_names_keep: tuple[str, ...] | None,
) -> _data_loader.Dataset:
    """One LeRobot dataset (optionally from an explicit root), episode-filtered by task
    string, with the per-dataset prompt transform applied. Mirrors
    data_loader.create_torch_dataset, plus `root=` so LIBERO-90 loads from its own
    lerobot root without touching HF_LEROBOT_HOME."""
    meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id, root=root)
    dataset = lerobot_dataset.LeRobotDataset(
        repo_id,
        root=root,
        delta_timestamps={
            key: [t / meta.fps for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
    )

    if task_names_keep is not None:
        # Same Subset-by-frame-indices approach as create_torch_dataset (LeRobot v2.0's
        # episodes= arg is broken for subsetting; see the comment there).
        keep_set = set(task_names_keep)
        kept = {
            ep_idx for ep_idx, ep_info in meta.episodes.items() if any(t in keep_set for t in ep_info.get("tasks", []))
        }
        if not kept:
            raise ValueError(f"task_names_keep ({len(keep_set)} names) matched 0 episodes in {repo_id}")
        offset = 0
        frame_indices: list[int] = []
        for ep_idx in sorted(meta.episodes.keys()):
            length = meta.episodes[ep_idx]["length"]
            if ep_idx in kept:
                frame_indices.extend(range(offset, offset + length))
            offset += length
        logging.info(
            f"[{repo_id}] task_names_keep ({len(keep_set)} names) kept "
            f"{len(kept)}/{len(meta.episodes)} episodes ({len(frame_indices)}/{offset} frames)"
        )
        dataset = torch.utils.data.Subset(dataset, frame_indices)
    else:
        logging.info(f"[{repo_id}] keeping ALL {meta.total_episodes} episodes ({meta.total_frames} frames)")

    if data_config.prompt_from_task:
        # Per-dataset: each repo has its own task_index -> string table.
        dataset = _data_loader.TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(meta.tasks)])

    return dataset


def pretrain_task_strings(holdout: tuple[str, ...] = PRETRAIN_HOLDOUT_TASKS) -> tuple[str, ...]:
    """LIBERO-40 task strings for the goal/object/spatial suites (ids 10..39), MINUS the
    `holdout` tasks (1 per suite → 27 pretrain tasks). Raises if a holdout string is not
    present in those suites (guards against typos vs meta/tasks.jsonl)."""
    ids = sorted(i for s in PRETRAIN_SUITES for i in _LIBERO_SUITE_RANGES[s])
    all_strings = task_strings_for_ids(LIBERO40_REPO_ID, ids)
    holdout_set = set(holdout)
    missing = holdout_set - set(all_strings)
    if missing:
        raise ValueError(f"pretrain holdout task(s) not in goal/object/spatial suites: {sorted(missing)}")
    return tuple(s for s in all_strings if s not in holdout_set)


def build_raw_dataset(args: Any, data_config: _config.DataConfig, action_horizon: int) -> _data_loader.Dataset:
    """The un-normalized dataset for the current mode: LIBERO-27 (goal/object/spatial minus
    1 held-out task each) + LIBERO-90 concat for pretrain/norm_stats, the 3 finetune tasks of
    LIBERO-40 for the finetune modes."""
    if args.mode not in ("pretrain", "norm_stats") and getattr(args, "finetune_repo_id", None):
        return _make_lerobot_dataset(
            args.finetune_repo_id, args.finetune_root, action_horizon, data_config, args.finetune_tasks
        )
    if getattr(args, "setting", "broad") == "narrow":
        ids, _ = sample_pretrain_tasks(args.tasks_per_suite, args.split_seed)
        keep = (
            task_strings_for_ids(LIBERO40_REPO_ID, ids)
            if args.mode in ("pretrain", "norm_stats")
            else args.finetune_tasks
        )
        return _make_lerobot_dataset(LIBERO40_REPO_ID, None, action_horizon, data_config, keep)
    if args.mode in ("pretrain", "norm_stats"):
        keep = pretrain_task_strings(args.pretrain_holdout_tasks)
        logging.info(
            f"[pretrain] {len(keep)} goal/object/spatial tasks (held out {len(args.pretrain_holdout_tasks)}: "
            f"{list(args.pretrain_holdout_tasks)})"
        )
        ds40 = _make_lerobot_dataset(LIBERO40_REPO_ID, None, action_horizon, data_config, keep)
        ds90 = _make_lerobot_dataset(
            args.libero90_repo_id,
            str(pathlib.Path(args.libero90_root) / args.libero90_repo_id) if args.libero90_root else None,
            action_horizon,
            data_config,
            None,
        )
        logging.info(
            f"[pretrain mixture] LIBERO-27={len(ds40)} frames + LIBERO-90={len(ds90)} frames "
            f"= {len(ds40) + len(ds90)} (natural proportions)"
        )
        return torch.utils.data.ConcatDataset([ds40, ds90])

    logging.info(f"[{args.mode}] finetune tasks:")
    for t in args.finetune_tasks:
        logging.info(f"  - {t}")
    return _make_lerobot_dataset(LIBERO40_REPO_ID, None, action_horizon, data_config, args.finetune_tasks)


def create_mixed_data_loader(
    args: Any, config: _config.TrainConfig, data_sharding: jax.sharding.Sharding
) -> _data_loader.DataLoader:
    """Norm-stats-applying train loader over build_raw_dataset (uniform shuffle over the
    concat = natural-proportion mixing). Same loader classes as create_data_loader, so
    checkpointing (save_state -> data_config()) works unchanged."""
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = build_raw_dataset(args, data_config, config.model.action_horizon)
    dataset = _data_loader.transform_dataset(dataset, data_config)
    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=data_sharding,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    return _data_loader.DataLoaderImpl(data_config, loader)


def run_norm_stats(args: Any, config: _config.TrainConfig) -> None:
    """Full pass over the pretrain mixture computing state/actions RunningStats, saved to
    <assets_dirs>/<asset_id> — exactly where the config's _load_norm_stats reads them."""
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = build_raw_dataset(args, data_config, config.model.action_horizon)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
    )
    if args.max_frames is not None and args.max_frames < len(dataset):
        num_batches = args.max_frames // config.batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // config.batch_size
        shuffle = False
    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}
    for batch in tqdm.tqdm(loader, total=num_batches, desc="Computing mixture norm stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: s.get_statistics() for key, s in stats.items()}
    assets_dir = config.data.assets.assets_dir or config.assets_dirs
    output_path = pathlib.Path(assets_dir) / data_config.asset_id
    logging.info(f"Writing mixture norm stats to: {output_path}")
    normalize.save(output_path, norm_stats)
