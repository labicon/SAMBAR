"""Supervised LoRA baseline."""

from __future__ import annotations

import dataclasses
import json
import logging
import pathlib
import platform
import sys
import time
from typing import Any

import tyro

from baselines._impl import supervised as _plain_train
import openpi.training.config as _config
import openpi.training.weight_loaders as _weight_loaders

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from openpi.training.task_split import describe_split  # noqa: E402
from openpi.training.task_split import sample_pretrain_tasks  # noqa: E402
from openpi.training.task_split import task_strings_for_ids  # noqa: E402


@dataclasses.dataclass(frozen=True)
class CliArgs:
    exp_name: str
    pretrained_path: str

    # LoRA pi0.5 LIBERO config. Its freeze_filter exposes only LoRA adapters.
    config_name: str = "libero_lora"

    # Same split convention as the TACO scripts. With no explicit override, train
    # on the seed-derived eight held-out tasks.
    tasks_per_suite: int = 8
    split_seed: int = 7
    finetune_task_ids: tuple[int, ...] | None = None

    # Common TrainConfig overrides. None keeps the base config default.
    num_train_steps: int | None = None
    batch_size: int | None = None
    lr: float | None = None
    seed: int | None = None
    fsdp_devices: int | None = None
    save_interval: int | None = None
    log_interval: int | None = None
    checkpoint_base_dir: str | None = None
    overwrite: bool = False
    resume: bool = False
    wandb_enabled: bool = False


def resolve_finetune_task_ids(args: CliArgs) -> list[int]:
    if args.finetune_task_ids:
        task_ids = sorted({int(x) for x in args.finetune_task_ids})
        bad = [t for t in task_ids if not (0 <= t < 40)]
        if bad:
            raise ValueError(f"finetune_task_ids contains out-of-range task(s): {bad}")
        logging.info(f"[lora_bc] using user-specified finetune task_ids: {task_ids}")
        return task_ids

    _pretrain_ids, holdout_ids = sample_pretrain_tasks(args.tasks_per_suite, args.split_seed)
    logging.info(
        f"[lora_bc] no --finetune-task-ids -> using held-out {len(holdout_ids)} tasks from split_seed={args.split_seed}"
    )
    return holdout_ids


def build_config(args: CliArgs, task_ids: list[int]) -> _config.TrainConfig:
    base = _config.get_config(args.config_name)
    task_strings = task_strings_for_ids(base.data.repo_id, task_ids)
    data = dataclasses.replace(base.data, task_names_keep=task_strings)

    updates: dict[str, Any] = {
        "exp_name": args.exp_name,
        "data": data,
        "weight_loader": _weight_loaders.CheckpointWeightLoader(args.pretrained_path),
        "overwrite": args.overwrite,
        "resume": args.resume,
        "wandb_enabled": args.wandb_enabled,
    }
    if args.lr is not None:
        updates["lr_schedule"] = dataclasses.replace(base.lr_schedule, peak_lr=args.lr, decay_lr=args.lr)
    for field_name in (
        "num_train_steps",
        "batch_size",
        "seed",
        "fsdp_devices",
        "save_interval",
        "log_interval",
        "checkpoint_base_dir",
    ):
        value = getattr(args, field_name)
        if value is not None:
            updates[field_name] = value

    return dataclasses.replace(base, **updates)


def write_plan(config: _config.TrainConfig, args: CliArgs, task_ids: list[int]) -> None:
    """Write a small sidecar before training starts.

    The standard train loop owns checkpoint directory creation, so keep this in a
    sibling file instead of pre-creating the checkpoint directory.
    """
    out = pathlib.Path(config.checkpoint_base_dir) / config.name
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / f"{config.exp_name}.plan.json"
    sched = config.lr_schedule
    plan = {
        "script": pathlib.Path(__file__).name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "exp_name": config.exp_name,
        "config_name": args.config_name,
        "pretrained_path": args.pretrained_path,
        "checkpoint_dir": str(config.checkpoint_dir),
        "task_ids_trained": task_ids,
        "num_tasks_trained": len(task_ids),
        "num_train_steps": config.num_train_steps,
        "batch_size": config.batch_size,
        "fsdp_devices": config.fsdp_devices,
        "seed": config.seed,
        "lr_peak": getattr(sched, "peak_lr", None),
        "lr_decay": getattr(sched, "decay_lr", None),
        "lr_warmup_steps": getattr(sched, "warmup_steps", None),
        "lr_decay_steps": getattr(sched, "decay_steps", None),
        "ema_decay": config.ema_decay,
        "wandb_enabled": config.wandb_enabled,
    }
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2, default=str))
    logging.info(f"Wrote LoRA-BC plan -> {plan_path}")


def main(args: CliArgs) -> None:
    _plain_train.init_logging()
    logging.info(f"Running on: {platform.node()}")
    logging.info("Mode: lora_bc_finetune (plain BC, no TACO/no CL penalty)")

    task_ids = resolve_finetune_task_ids(args)
    describe_split(task_ids, "LORA FINETUNE")

    config = build_config(args, task_ids)
    logging.info(f"[lora_bc] selecting {len(task_ids)} task strings via task_names_keep")
    write_plan(config, args, task_ids)

    _plain_train.main(config)


if __name__ == "__main__":
    main(tyro.cli(CliArgs))
