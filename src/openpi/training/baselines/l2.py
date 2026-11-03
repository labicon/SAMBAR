"""L2-SP baseline (original numerical implementation)."""

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
import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import tyro
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
from openpi.training.sambar import parameters
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

# Shared train/eval split logic (single source of truth — see module docstring).
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from openpi.training.task_split import describe_split  # noqa: E402
from openpi.training.task_split import sample_pretrain_tasks  # noqa: E402
from openpi.training.task_split import task_strings_for_ids  # noqa: E402


@dataclasses.dataclass(frozen=True)
class L2Args:
    """Plain-L2 (L2-SP) hyperparameters. Used only in l2_finetune mode."""

    # Coefficient λ for the penalty (λ/2)·Σ‖θ_G − z‖². Fixed, applied every step.
    lam: float = 0.02
    # Optional separate frozen reference for the penalty. If None, the reference is the
    # weights loaded at init (θ_pre = the --pretrained-path stage-1 checkpoint). Set e.g.
    # to "gs://openpi-assets/checkpoints/pi05_base/params" to anchor on the pi0.5 base.
    anchor_path: str | None = None


@dataclasses.dataclass(frozen=True)
class CliArgs:
    """Top-level CLI: picks a base TrainConfig, overrides a few key fields, and adds the
    mode + task-split + L2 hyperparameters."""

    exp_name: str
    mode: Literal["pretrain", "bc_finetune", "l2_finetune"]

    config_name: str = "sambar_narrow"  # shared full-pi0.5 configuration

    # Checkpoint to initialize the finetune phases from (the stage-1 pretrain output's
    # `params` dir). Required for bc_finetune / l2_finetune.
    pretrained_path: str | None = None

    # Task split. The held-out finetune set is determined by --split-seed alone.
    tasks_per_suite: int = 8
    split_seed: int = 7
    # Override the seed-derived holdout with explicit task indices (0..39) for finetune modes.
    finetune_task_ids: tuple[int, ...] | None = None

    # Common TrainConfig overrides (None → keep the config default).
    num_train_steps: int | None = None
    batch_size: int | None = None
    # Peak (and floor) learning rate override; rewrites the CosineDecaySchedule
    # peak_lr=decay_lr (warmup/decay_steps kept). None → keep config default (5e-5).
    lr: float | None = None
    seed: int | None = None
    fsdp_devices: int | None = None
    save_interval: int | None = None
    log_interval: int | None = None
    overwrite: bool = False
    resume: bool = False
    wandb_enabled: bool = False
    checkpoint_base_dir: str | None = None

    l2: L2Args = dataclasses.field(default_factory=L2Args)


def init_logging():
    """Match scripts/train.py logging format."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def resolve_finetune_task_ids(args: CliArgs) -> list[int]:
    """Honor an explicit --finetune-task-ids list, else fall back to the seed-derived
    holdout (complement of the pretrain split)."""
    if args.finetune_task_ids:
        ft_ids = sorted({int(x) for x in args.finetune_task_ids})
        for t in ft_ids:
            if not (0 <= t < 40):
                raise ValueError(f"finetune_task_ids contains out-of-range task: {t}")
        logging.info(f"[{args.mode}] using user-specified finetune task_ids: {ft_ids}")
        return ft_ids
    _pretrain, ft_ids = sample_pretrain_tasks(args.tasks_per_suite, args.split_seed)
    logging.info(
        f"[{args.mode}] no --finetune-task-ids → using held-out {len(ft_ids)} tasks from split_seed={args.split_seed}"
    )
    return ft_ids


# === train-state init (single model; mirrors scripts/train.py) ==============


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, tx: optax.GradientTransformation
) -> tuple[training_utils.TrainState, Any]:
    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)
        params = nnx.state(model)
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=False)
    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)
    return train_state, state_sharding


# === frozen reference z = θ_pre =====================================


def make_reference(state: training_utils.TrainState, l2_args: L2Args) -> Any:
    """Build the frozen reference z = θ_pre as a pure pytree of device arrays.

    Default: the subset of the weights loaded at init (θ_pre = the stage-1 checkpoint).
    If `l2_args.anchor_path` is set, the leaves are taken from that checkpoint instead,
    matched by path against the init structure."""
    z = parameters(state.params)
    if l2_args.anchor_path is None:
        # parameters returns views that ALIAS state.params buffers; step_fn donates
        # `state`, so an aliasing z triggers a "buffer previously donated" crash. Copy.
        return jax.tree.map(jnp.copy, z)

    logging.info(f"[l2] loading frozen reference from anchor: {l2_args.anchor_path}")
    anchor = _model.restore_params(download.maybe_download(l2_args.anchor_path), restore_type=np.ndarray)
    flat_anchor = traverse_util.flatten_dict(anchor, sep="/")
    flat_z = traverse_util.flatten_dict(z, sep="/")
    missing = [k for k in flat_z if k not in flat_anchor]
    if missing:
        raise ValueError(f"anchor checkpoint missing {len(missing)} parameter keys, e.g. {missing[:3]}")
    flat_new = {k: jnp.asarray(flat_anchor[k], dtype=jnp.asarray(flat_z[k]).dtype) for k in flat_z}
    return traverse_util.unflatten_dict(flat_new, sep="/")


def l2_drift_sq(params: nnx.State, z: Any) -> jnp.ndarray:
    """Σ‖θ_G − z‖² over all parameters (sum of squared residuals)."""
    theta = parameters(params)
    return sum(
        jnp.sum((t_leaf - z_leaf) ** 2)
        for t_leaf, z_leaf in zip(jax.tree.leaves(theta), jax.tree.leaves(z), strict=True)
    )


# === train step (single model; penalty gated by pen_scale) ==================


def train_step(
    config: _config.TrainConfig,
    l2_args: L2Args,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
    z: Any,
    pen_scale: jnp.ndarray,
) -> tuple[training_utils.TrainState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Forward + backward + Adam update. Adds (λ/2)·Σ‖θ_G − z‖² over all parameters to the loss, scaled by `pen_scale` (0 for pretrain/bc, 1 for
    active l2_finetune). No dual variable.
    Returns: (state, task_loss, total_loss, gnorm, penalty, drift)."""
    model = nnx.merge(state.model_def, state.params)
    model.train()
    obs, actions = batch

    def loss_fn(m: _model.BaseModel, rng_: at.KeyArrayLike, obs_, act_):
        task_loss = jnp.mean(m.compute_loss(rng_, obs_, act_, train=True))
        params = nnx.state(m)
        penalty = 0.5 * l2_args.lam * l2_drift_sq(params, z)
        return task_loss + pen_scale * penalty, (task_loss, penalty)

    rng_step = jax.random.fold_in(rng, state.step)
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (total_loss, (task_loss, penalty)), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, rng_step, obs, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params)
    full_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=full_params, opt_state=new_opt_state)
    gnorm = optax.global_norm(grads)
    drift = jnp.sqrt(l2_drift_sq(full_params, z))
    return new_state, task_loss, total_loss, gnorm, penalty, drift


# === lightweight metric plots ===============================================


def plot_training_metrics(metrics_path: pathlib.Path, output_dir: pathlib.Path):
    """Plot loss curves + parameter drift from train_metrics.jsonl. Best-effort (skips on error)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        logging.warning(f"matplotlib unavailable, skipping plots: {e}")
        return
    rows = [json.loads(l) for l in metrics_path.read_text().splitlines() if l.strip()]
    if not rows:
        return
    steps = [r["step"] for r in rows]

    fig, ax = plt.subplots(figsize=(10, 6))
    for k in ("task_loss", "total_loss"):
        if k in rows[0]:
            ax.plot(steps, [r[k] for r in rows], label=k, linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(output_dir / "losses.png", dpi=100, bbox_inches="tight")
    plt.close()

    if "drift" in rows[0]:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(steps, [r["drift"] for r in rows], label="‖θ_parameter − z‖", linewidth=2, color="tab:red")
        ax2 = ax.twinx()
        if "penalty" in rows[0]:
            ax2.plot(steps, [r["penalty"] for r in rows], label="penalty", linewidth=1, color="tab:blue", alpha=0.6)
            ax2.set_ylabel("penalty")
        ax.set_xlabel("Step")
        ax.set_ylabel("parameter drift")
        ax.set_title("parameter drift from θ_pre")
        ax.grid(True, alpha=0.3)
        plt.savefig(output_dir / "qkv_drift.png", dpi=100, bbox_inches="tight")
        plt.close()
    logging.info(f"Saved plots to {output_dir}")


# === config assembly ========================================================


def build_config(args: CliArgs, task_strings: tuple[str, ...]) -> _config.TrainConfig:
    """Clone the base config: set the task subset (task_names_keep) and, for the finetune
    phases, point the weight_loader at the stage-1 checkpoint so init params = θ_pre."""
    base = _config.get_config(args.config_name)
    data = dataclasses.replace(base.data, task_names_keep=task_strings)

    updates: dict[str, Any] = {
        "exp_name": args.exp_name,
        "data": data,
        "overwrite": args.overwrite,
        "resume": args.resume,
        "wandb_enabled": args.wandb_enabled,
    }
    if args.mode in ("bc_finetune", "l2_finetune"):
        if not args.pretrained_path:
            raise SystemExit(f"--pretrained-path is required in {args.mode} mode")
        updates["weight_loader"] = _weight_loaders.CheckpointWeightLoader(args.pretrained_path)
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
        v = getattr(args, field_name)
        if v is not None:
            updates[field_name] = v
    return dataclasses.replace(base, **updates)


# === main ===================================================================


def _write_hparams(args: CliArgs, config: _config.TrainConfig, task_ids) -> None:
    """Dump the full hyperparameter set to <checkpoint_dir>/hparams.json (self-documenting
    checkpoint; written once at training start, independent of wandb)."""
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
            "repo_id": config.data.repo_id,
            "weight_loader": type(config.weight_loader).__name__,
            "checkpoint_dir": str(config.checkpoint_dir),
            "num_tasks_trained": len(task_ids),
            "task_ids_trained": list(task_ids),
        },
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    p = pathlib.Path(config.checkpoint_dir) / "hparams.json"
    p.write_text(json.dumps(hp, indent=2, default=str))
    logging.info(f"Wrote hyperparameters -> {p}")


def main(args: CliArgs):
    init_logging()
    logging.info(f"Running on: {platform.node()}")
    logging.info(f"Mode: {args.mode} | L2 args: {args.l2}")

    repo_id = _config.get_config(args.config_name).data.repo_id

    if args.mode == "pretrain":
        pretrain_ids, holdout_ids = sample_pretrain_tasks(args.tasks_per_suite, args.split_seed)
        describe_split(pretrain_ids, "PRETRAIN")
        describe_split(holdout_ids, "HOLDOUT")
        task_ids = pretrain_ids
    else:
        task_ids = resolve_finetune_task_ids(args)
        describe_split(task_ids, "FINETUNE")

    task_strings = task_strings_for_ids(repo_id, task_ids)
    logging.info(f"[{args.mode}] selecting {len(task_strings)} task strings via task_names_keep")

    config = build_config(args, task_strings)
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
    _write_hparams(args, config, task_ids)

    if config.wandb_enabled:
        wandb.init(
            name=config.exp_name,
            config={
                **dataclasses.asdict(config),
                "mode": args.mode,
                "l2": dataclasses.asdict(args.l2),
                "task_ids": task_ids,
                "split_seed": args.split_seed,
            },
            project=config.project_name,
        )
    else:
        wandb.init(mode="disabled")

    data_loader = _data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True)
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)
    state, state_sharding = init_train_state(config, init_rng, mesh, tx)
    jax.block_until_ready(state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(state.params)}")

    if resuming:
        state = _checkpoints.restore_state(mngr, state, data_loader)

    # (e.g. "all" = whole-model L2-SP). Independent of what is trained (training is always full).

    # Frozen reference z = θ_pre. For non-L2 modes it is inert (pen_scale=0). FSDP-shard z
    # the SAME way as the parameters (fsdp_sharding is a deterministic function of leaf shape, so
    # θ and z stay co-sharded → no resharding in θ−z). With fsdp_devices=1 this is full replication.
    z_raw = make_reference(state, args.l2)
    z_sharding = sharding.fsdp_sharding(z_raw, mesh, log=False)
    z = jax.device_put(z_raw, z_sharding)
    if is_l2:
        logging.info(
            f"[l2] penalizing {sum(int(np.prod(jnp.asarray(v).shape)) for v in jax.tree.leaves(z)) / 1e6:.1f}M "
            f"params (penalty_subset=all); "
            f"λ={args.l2.lam} (penalty = λ/2·Σ‖θ−z‖²)"
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

    # Metrics logging file (saved regardless of wandb_enabled).
    metrics_path = pathlib.Path(config.checkpoint_dir) / "train_metrics.jsonl"
    metrics_file = open(metrics_path, "a")
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
