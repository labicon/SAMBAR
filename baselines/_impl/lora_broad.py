"""Supervised LoRA baseline for broad pretraining."""

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
from openpi.training.curriculum import DEFAULT_FINETUNE_TASKS  # noqa: E402


@dataclasses.dataclass(frozen=True)
class CliArgs:
    exp_name: str
    pretrained_path: str

    # LoRA pi0.5 config carrying the libero90mix (117-task pretrain) norm stats.
    # Its freeze_filter exposes only the LoRA adapters.
    config_name: str = "libero_broad_lora"

    # Finetune task strings (must match meta/tasks.jsonl of physical-intelligence/libero).
    finetune_tasks: tuple[str, ...] = DEFAULT_FINETUNE_TASKS

    # Common TrainConfig overrides. None keeps the base config default.
    num_train_steps: int | None = None
    batch_size: int | None = None
    lr: float | None = None
    seed: int | None = None
    num_workers: int | None = None
    fsdp_devices: int | None = None
    save_interval: int | None = None
    # Retain every checkpoint with step % keep_period == 0 (orbax keeps only the
    # newest otherwise). Set alongside --save-interval, e.g. 500/500.
    keep_period: int | None = None
    log_interval: int | None = None
    checkpoint_base_dir: str | None = None
    overwrite: bool = False
    resume: bool = False
    wandb_enabled: bool = False


def build_config(args: CliArgs) -> _config.TrainConfig:
    base = _config.get_config(args.config_name)
    if not args.finetune_tasks:
        raise SystemExit("--finetune-tasks must name at least one task string")
    data = dataclasses.replace(base.data, task_names_keep=tuple(args.finetune_tasks))

    updates: dict[str, Any] = {
        "exp_name": args.exp_name,
        "data": data,
        "weight_loader": _weight_loaders.CheckpointWeightLoader(args.pretrained_path),
        "overwrite": args.overwrite,
        "resume": args.resume,
        "wandb_enabled": args.wandb_enabled,
    }
    # NOTE: --lr sets peak AND floor, so the cosine decay flattens. Omit it to keep the
    # config's 2.5e-5 -> 2.5e-6 schedule (same as the BC/TACO libero90mix runs).
    if args.lr is not None:
        updates["lr_schedule"] = dataclasses.replace(base.lr_schedule, peak_lr=args.lr, decay_lr=args.lr)
    for field_name in (
        "num_train_steps",
        "batch_size",
        "seed",
        "num_workers",
        "fsdp_devices",
        "save_interval",
        "keep_period",
        "log_interval",
        "checkpoint_base_dir",
    ):
        value = getattr(args, field_name)
        if value is not None:
            updates[field_name] = value

    return dataclasses.replace(base, **updates)


def write_plan(config: _config.TrainConfig, args: CliArgs) -> None:
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
        "finetune_tasks": list(args.finetune_tasks),
        "num_tasks_trained": len(args.finetune_tasks),
        "norm_stats_asset_id": config.data.assets.asset_id,
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
    logging.info("Mode: lora_bc_finetune on libero90mix (plain BC, no TACO/no CL penalty)")

    config = build_config(args)
    logging.info(f"[lora_bc] finetuning on {len(args.finetune_tasks)} task string(s):")
    for t in args.finetune_tasks:
        logging.info(f"  - {t}")
    write_plan(config, args)

    _plain_train.main(config)


if __name__ == "__main__":
    main(tyro.cli(CliArgs))
