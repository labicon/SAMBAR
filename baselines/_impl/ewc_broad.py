"""EWC baseline for broad pretraining."""

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
# EWC / weighted-L2 machinery (Fisher, weighted drift, train step) — unchanged.
from baselines._impl.ewc import WL2Args  # noqa: E402
from baselines._impl.ewc import compute_fisher  # noqa: E402
from baselines._impl.ewc import init_logging  # noqa: E402
from baselines._impl.ewc import init_train_state  # noqa: E402
from baselines._impl.ewc import make_fisher_step  # noqa: E402
from baselines._impl.ewc import make_reference  # noqa: E402
from baselines._impl.ewc import normalize_fisher  # noqa: E402
from baselines._impl.ewc import plot_training_metrics  # noqa: E402
from baselines._impl.ewc import train_step  # noqa: E402

# Partial-finetuning freeze filter (same helper the libero90mix BC/TACO script uses).
# LIBERO-90-mix data curriculum — unchanged.
from openpi.training.curriculum import DEFAULT_FINETUNE_TASKS  # noqa: E402
from openpi.training.curriculum import PRETRAIN_HOLDOUT_TASKS  # noqa: E402
from openpi.training.curriculum import create_mixed_data_loader  # noqa: E402


@dataclasses.dataclass(frozen=True)
class CliArgs:
    """Top-level CLI: picks the base TrainConfig, overrides a few key fields, and adds
    the mode + dataset roots + weighted-L2 hyperparameters."""

    exp_name: str
    mode: Literal["pretrain", "bc_finetune", "weighted_l2"]

    config_name: str = "sambar_broad"

    # Checkpoint to initialize the finetune phases from (the pretrain output's
    # `params` dir). Required for bc_finetune / weighted_l2.
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

    # Default for THIS script: penalty + Fisher over ALL params (whole-model EWC), matching
    wl2: WL2Args = dataclasses.field(default_factory=lambda: WL2Args())


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
    if args.mode in ("bc_finetune", "weighted_l2"):
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
            "wl2": dataclasses.asdict(args.wl2),
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
    logging.info(f"Mode: {args.mode} | WL2 args: {args.wl2}")

    config = build_config(args)
    is_wl2 = args.mode == "weighted_l2"

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update(
        "jax_compilation_cache_dir",
        str(epath.Path(os.environ.get("JAX_COMPILATION_CACHE_DIR", "~/.cache/jax")).expanduser()),
    )

    rng = jax.random.key(config.seed)
    train_rng, init_rng, fisher_rng = jax.random.split(rng, 3)

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
                "wl2": dataclasses.asdict(args.wl2),
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

    # Frozen anchor z = θ_pre. Inert in pretrain/bc modes (pen_scale=0). FSDP-shard z
    # the SAME way as the parameters (fsdp_sharding is a deterministic function of leaf
    # shape, so θ, z and Ω stay co-sharded → no resharding in θ−z).
    z_raw = make_reference(state, args.wl2)
    consensus_sharding = sharding.fsdp_sharding(z_raw, mesh, log=False)
    z = jax.device_put(z_raw, consensus_sharding)

    # Diagonal Fisher Ω_old = E_pretrain[grad²] at θ_pre over the 117-task pretrain MIXTURE.
    # Frozen anchor ⇒ frozen weights ⇒ no refresh. Sharded like z. Non-wl2 modes get an inert
    # ones-tree (pen_scale=0 makes the penalty vanish anyway).
    if is_wl2:
        logging.info(
            f"[wl2] estimating Ω_old at θ_pre over the LIBERO-27 + LIBERO-90 mixture "
            f"({args.wl2.fisher_batches} batches)..."
        )
        fisher_step = make_fisher_step(config, args.wl2, out_shardings=consensus_sharding)
        old_iter = iter(create_mixed_data_loader(dataclasses.replace(args, mode="pretrain"), config, data_sharding))
        omega = compute_fisher(fisher_step, fisher_rng, state, old_iter, args.wl2.fisher_batches, mesh)
        del old_iter  # release the mixture loader's workers; Ω_old is fixed for the run
        if args.wl2.normalize_weights:
            omega = jax.jit(normalize_fisher, out_shardings=consensus_sharding)(omega)
        n_pen = sum(int(np.prod(jnp.asarray(v).shape)) for v in jax.tree.leaves(z)) / 1e6
        logging.info(
            f"[wl2] penalizing {n_pen:.1f}M params (penalty_subset=all, "
            f"trained=all); λ={args.wl2.lam}, "
            f"normalize_weights={args.wl2.normalize_weights} (penalty = λ/2·Σ Ω_old·‖θ−z‖²)"
        )
    else:
        omega = jax.jit(lambda t: jax.tree.map(jnp.ones_like, t), out_shardings=consensus_sharding)(z)

    step_fn = jax.jit(
        functools.partial(train_step, config, args.wl2, state_sharding.params),
        in_shardings=(
            replicated_sharding,  # rng
            state_sharding,  # state
            data_sharding,  # batch
            consensus_sharding,  # z
            consensus_sharding,  # omega
            replicated_sharding,  # pen_scale
        ),
        out_shardings=(
            state_sharding,
            replicated_sharding,
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
            pen_scale = jnp.float32(1.0 if is_wl2 else 0.0)
            rng_step = jax.random.fold_in(train_rng, state.step)
            state, task_loss, total_loss, gnorm, penalty, drift, w_drift = step_fn(
                rng_step, state, batch, z, omega, pen_scale
            )

        info = jax.device_get(
            {
                "task_loss": task_loss,
                "total_loss": total_loss,
                "grad_norm": gnorm,
                "penalty": penalty,
                "drift": drift,
                "w_drift": w_drift,
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
