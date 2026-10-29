"""L2-SP baseline for broad pretraining."""

from __future__ import annotations

import dataclasses
import functools
import json
import logging
import os
import pathlib
import platform
import sys
from typing import Any, Literal

import etils.epath as epath
import jax
import jax.numpy as jnp
import numpy as np
import tqdm_loggable.auto as tqdm
import tyro
import wandb

import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

sys.path.insert(0, str(pathlib.Path(__file__).parent))
# L2-SP machinery (frozen anchor, drift, train step) — unchanged.
from baselines._impl.l2 import L2Args  # noqa: E402
from baselines._impl.l2 import init_logging  # noqa: E402
from baselines._impl.l2 import init_train_state  # noqa: E402
from baselines._impl.l2 import make_reference  # noqa: E402
from baselines._impl.l2 import plot_training_metrics  # noqa: E402
from baselines._impl.l2 import train_step  # noqa: E402

# Partial-finetuning freeze filter (same helper the libero90mix BC/TACO script uses).
# LIBERO-90-mix data curriculum — unchanged.
from openpi.training.curriculum import DEFAULT_FINETUNE_TASKS  # noqa: E402
from openpi.training.curriculum import PRETRAIN_HOLDOUT_TASKS  # noqa: E402
from openpi.training.curriculum import create_mixed_data_loader  # noqa: E402


@dataclasses.dataclass(frozen=True)
class CliArgs:
    """Top-level CLI: picks the base TrainConfig, overrides a few key fields, and adds
    the mode + dataset roots + L2 hyperparameters."""

    exp_name: str
    mode: Literal["pretrain", "bc_finetune", "l2_finetune"]

    config_name: str = "sambar_broad"

    # Checkpoint to initialize the finetune phases from (the pretrain output's
    # `params` dir). Required for bc_finetune / l2_finetune.
    pretrained_path: str | None = None

    # LIBERO-90 dataset location: a SEPARATE lerobot root (never the LIBERO-40 cache).
    libero90_repo_id: str = "jesbu1/libero_90_lerobot"
    libero90_root: str | None = None

    # Finetune task strings (must match meta/tasks.jsonl of the LIBERO-40 dataset).
    finetune_tasks: tuple[str, ...] = DEFAULT_FINETUNE_TASKS

    # Tasks held out from the pretrain mixture (1 per goal/object/spatial suite → 27).
    pretrain_holdout_tasks: tuple[str, ...] = PRETRAIN_HOLDOUT_TASKS

    # Common TrainConfig overrides (None → keep the config default).
    num_train_steps: int | None = None
    batch_size: int | None = None
    # Peak (and floor) learning rate override, same semantics as the sibling scripts.
    lr: float | None = None
    seed: int | None = None
    num_workers: int | None = None
    fsdp_devices: int | None = None
    save_interval: int | None = None
    # Retain every checkpoint with step % keep_period == 0 (orbax keeps only the
    # newest otherwise). Set alongside --save-interval, e.g. 500/500.
    keep_period: int | None = None
    log_interval: int | None = None
    overwrite: bool = False
    resume: bool = False
    wandb_enabled: bool = False
    checkpoint_base_dir: str | None = None

    # Default for THIS script: penalty over ALL params (whole-model L2-SP), matching the
    l2: L2Args = dataclasses.field(default_factory=lambda: L2Args())


# === config assembly ========================================================


def build_config(args: CliArgs) -> _config.TrainConfig:
    """Clone the base config; record the finetune task subset in the data config (for
    hparams/checkpoint self-documentation — the loader is built by this script), point
    the finetune phases' weight_loader at the pretrain checkpoint, apply overrides."""
    base = _config.get_config(args.config_name)

    updates: dict[str, Any] = {
        "exp_name": args.exp_name,
        "overwrite": args.overwrite,
        "resume": args.resume,
        "wandb_enabled": args.wandb_enabled,
    }
    if args.mode in ("bc_finetune", "l2_finetune"):
        if not args.pretrained_path:
            raise SystemExit(f"--pretrained-path is required in {args.mode} mode")
        updates["weight_loader"] = _weight_loaders.CheckpointWeightLoader(args.pretrained_path)
        updates["data"] = dataclasses.replace(base.data, task_names_keep=tuple(args.finetune_tasks))
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
        v = getattr(args, field_name)
        if v is not None:
            updates[field_name] = v
    return dataclasses.replace(base, **updates)


def _write_hparams(args: CliArgs, config: _config.TrainConfig) -> None:
    """Dump the full hyperparameter set to <checkpoint_dir>/hparams.json."""
    import time

    sched = config.lr_schedule
    hp = {
        "exp_name": config.exp_name,
        "mode": args.mode,
        "script": pathlib.Path(__file__).name,
        "cli_args": dataclasses.asdict(args),
        "resolved": {
            "config_name": args.config_name,
            "num_train_steps": config.num_train_steps,
            "batch_size": config.batch_size,
            "lr_peak": getattr(sched, "peak_lr", None),
            "lr_decay": getattr(sched, "decay_lr", None),
            "lr_warmup_steps": getattr(sched, "warmup_steps", None),
            "lr_decay_steps": getattr(sched, "decay_steps", None),
            "fsdp_devices": config.fsdp_devices,
            "seed": config.seed,
            "ema_decay": config.ema_decay,
            "weight_loader": type(config.weight_loader).__name__,
            "checkpoint_dir": str(config.checkpoint_dir),
            "libero90_repo_id": args.libero90_repo_id,
            "libero90_root": args.libero90_root,
            "finetune_tasks": list(args.finetune_tasks),
            "l2": dataclasses.asdict(args.l2),
        },
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    p = pathlib.Path(config.checkpoint_dir) / "hparams.json"
    p.write_text(json.dumps(hp, indent=2, default=str))
    logging.info(f"Wrote hyperparameters -> {p}")


# === main ===================================================================


def main(args: CliArgs):
    init_logging()
    logging.info(f"Running on: {platform.node()}")
    logging.info(f"Mode: {args.mode} | L2 args: {args.l2}")

    config = build_config(args)
    is_l2 = args.mode == "l2_finetune"

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update(
        "jax_compilation_cache_dir",
        str(epath.Path(os.environ.get("JAX_COMPILATION_CACHE_DIR", "~/.cache/jax")).expanduser()),
    )

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    mngr, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir, keep_period=config.keep_period, overwrite=config.overwrite, resume=config.resume
    )
    _write_hparams(args, config)

    if config.wandb_enabled:
        wandb.init(
            name=config.exp_name,
            config={
                **dataclasses.asdict(config),
                "mode": args.mode,
                "l2": dataclasses.asdict(args.l2),
                "finetune_tasks": list(args.finetune_tasks),
            },
            project=config.project_name,
        )
    else:
        wandb.init(mode="disabled")

    # Mode-aware loader: pretrain → LIBERO-27 + LIBERO-90 mixture; finetune modes → the
    # --finetune-tasks subset of LIBERO-40 (same construction as the sibling scripts).
    data_loader = create_mixed_data_loader(args, config, data_sharding)
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)
    state, state_sharding = init_train_state(config, init_rng, mesh, tx)
    jax.block_until_ready(state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(state.params)}")

    if resuming:
        state = _checkpoints.restore_state(mngr, state, data_loader)

    # Frozen reference z = θ_pre. Inert in pretrain/bc modes (pen_scale=0). FSDP-shard z
    # the SAME way as the parameters (fsdp_sharding is a deterministic function of leaf shape,
    # so θ and z stay co-sharded → no resharding in θ−z).
    z_raw = make_reference(state, args.l2)
    z_sharding = sharding.fsdp_sharding(z_raw, mesh, log=False)
    z = jax.device_put(z_raw, z_sharding)
    if is_l2:
        logging.info(
            f"[l2] penalizing {sum(int(np.prod(jnp.asarray(v).shape)) for v in jax.tree.leaves(z)) / 1e6:.1f}M "
            f"params (penalty_subset=all, trained=all, "
            f"); λ={args.l2.lam} (penalty = λ/2·Σ‖θ−z‖²)"
        )

    step_fn = jax.jit(
        functools.partial(train_step, config, args.l2),
        in_shardings=(
            replicated_sharding,  # rng
            state_sharding,  # state
            data_sharding,  # batch
            z_sharding,  # z
            replicated_sharding,  # pen_scale
        ),
        out_shardings=(
            state_sharding,
            replicated_sharding,
            replicated_sharding,
            replicated_sharding,
            replicated_sharding,
            replicated_sharding,
        ),
        donate_argnums=(1,),
    )

    start_step = int(state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps), initial=start_step, total=config.num_train_steps, dynamic_ncols=True
    )

    metrics_path = pathlib.Path(config.checkpoint_dir) / "train_metrics.jsonl"
    metrics_file = open(metrics_path, "a")  # append mode to resume from checkpoint
    logging.info(f"Logging training metrics to: {metrics_path}")

    infos: list[dict[str, Any]] = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            pen_scale = jnp.float32(1.0 if is_l2 else 0.0)
            rng_step = jax.random.fold_in(train_rng, state.step)
            state, task_loss, total_loss, gnorm, penalty, drift = step_fn(rng_step, state, batch, z, pen_scale)

        info = jax.device_get(
            {
                "task_loss": task_loss,
                "total_loss": total_loss,
                "grad_norm": gnorm,
                "penalty": penalty,
                "drift": drift,
                "pen_scale": pen_scale,
            }
        )
        infos.append(info)
        if step % config.log_interval == 0:
            reduced = {k: float(np.mean([d[k] for d in infos])) for k in infos[0]}
            pbar.write(f"Step {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in reduced.items()))
            metrics_file.write(json.dumps({"step": int(step), **reduced}) + "\n")
            metrics_file.flush()
            wandb.log(reduced, step=step)
            infos = []
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(mngr, state, data_loader, step)

    metrics_file.close()
    logging.info("Waiting for checkpoint manager to finish")
    mngr.wait_until_finished()
    plot_training_metrics(metrics_path, pathlib.Path(config.checkpoint_dir))


if __name__ == "__main__":
    main(tyro.cli(CliArgs))
