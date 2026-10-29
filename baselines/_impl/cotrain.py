"""Sequential co-training baseline with old-data replay."""

from __future__ import annotations

import dataclasses
import functools
import json
import logging
import os
import pathlib
import platform
import sys
import time
from typing import Any

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
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from openpi.training.task_split import describe_split  # noqa: E402
from openpi.training.task_split import sample_pretrain_tasks  # noqa: E402
from openpi.training.task_split import task_strings_for_ids  # noqa: E402


@dataclasses.dataclass(frozen=True)
class CliArgs:
    exp_name: str
    pretrained_path: str
    # New task(s) learned in this stage (dataset task_index 0..39).
    finetune_task_ids: tuple[int, ...]

    config_name: str = "sambar_narrow"
    tasks_per_suite: int = 8
    split_seed: int = 7
    # Tasks learned in earlier sequential stages; replayed together with the pretrain tasks.
    old_task_ids: tuple[int, ...] = ()
    # Replay the seed's 32 pretrain tasks (False => replay only --old-task-ids).
    replay_pretrain: bool = True
    # Fraction of every batch drawn from the old task set.
    old_fraction: float = 0.5

    num_train_steps: int | None = None
    batch_size: int | None = None
    lr: float | None = None
    seed: int | None = None
    fsdp_devices: int | None = None
    save_interval: int | None = None
    # Retain every checkpoint with step % keep_period == 0 (orbax keeps only the newest otherwise).
    keep_period: int | None = None
    log_interval: int | None = None
    overwrite: bool = False
    resume: bool = False
    wandb_enabled: bool = False
    checkpoint_base_dir: str | None = None


def init_logging():
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


def resolve_task_sets(args: CliArgs) -> tuple[list[int], list[int], list[int]]:
    """(finetune_ids, old_ids, pretrain_ids). old = pretrain (if replayed) ∪ carried-over tasks."""
    for t in (*args.finetune_task_ids, *args.old_task_ids):
        if not (0 <= t < 40):
            raise SystemExit(f"task id {t} outside LIBERO-40")
    ft_ids = sorted({int(t) for t in args.finetune_task_ids})
    if not ft_ids:
        raise SystemExit("--finetune-task-ids is required")
    pretrain_ids, _ = sample_pretrain_tasks(args.tasks_per_suite, args.split_seed)
    old = {int(t) for t in args.old_task_ids}
    if args.replay_pretrain:
        old |= set(pretrain_ids)
    old_ids = sorted(old)
    overlap = sorted(set(old_ids) & set(ft_ids))
    if overlap:
        raise SystemExit(f"old (replay) task set overlaps the finetune set on {overlap}")
    if args.old_fraction > 0 and not old_ids:
        raise SystemExit("--old-fraction > 0 but the old task set is empty")
    return ft_ids, old_ids, pretrain_ids


def split_batch(batch_size: int, old_fraction: float, n_devices: int) -> tuple[int, int]:
    """(new, old) sub-batch sizes; old = old_fraction·B rounded to a multiple of the device count."""
    if not 0.0 <= old_fraction < 1.0:
        raise SystemExit(f"--old-fraction must be in [0, 1), got {old_fraction}")
    b_old = int(round(old_fraction * batch_size / n_devices)) * n_devices
    b_new = batch_size - b_old
    if b_new <= 0 or b_new % n_devices or batch_size % n_devices:
        raise SystemExit(f"cannot split batch {batch_size} (old_fraction {old_fraction}) over {n_devices} devices")
    return b_new, b_old


# === train-state init (mirrors the shared training state initialization) ===============


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


# === train step ===============================================================================


def train_step(
    config: _config.TrainConfig,
    param_sharding,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
    new_mask: jnp.ndarray,
) -> tuple[training_utils.TrainState, dict[str, jnp.ndarray]]:
    """BC step on the concatenated batch; new_mask[i] = 1 for new-task rows (for per-part loss logs)."""
    model = nnx.merge(state.model_def, state.params)
    model.train()
    obs, actions = batch

    def loss_fn(m: _model.BaseModel, rng_: at.KeyArrayLike, obs_, act_):
        per_sample = jnp.mean(m.compute_loss(rng_, obs_, act_, train=True), axis=-1)  # [B]
        return jnp.mean(per_sample), per_sample

    rng_step = jax.random.fold_in(rng, state.step)
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, per_sample), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, rng_step, obs, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params)
    full_params = nnx.state(model)
    if param_sharding is not None:
        full_params = jax.lax.with_sharding_constraint(full_params, param_sharding)
    new_state = dataclasses.replace(state, step=state.step + 1, params=full_params, opt_state=new_opt_state)

    n_new = jnp.maximum(jnp.sum(new_mask), 1.0)
    n_old = jnp.maximum(jnp.sum(1.0 - new_mask), 1.0)
    info = {
        "loss": loss,
        "new_loss": jnp.sum(per_sample * new_mask) / n_new,
        "old_loss": jnp.sum(per_sample * (1.0 - new_mask)) / n_old,
        "grad_norm": optax.global_norm(grads),
    }
    return new_state, info


# === config / loaders =========================================================================


def build_config(args: CliArgs, task_strings: tuple[str, ...]) -> _config.TrainConfig:
    base = _config.get_config(args.config_name)
    updates: dict[str, Any] = {
        "exp_name": args.exp_name,
        "data": dataclasses.replace(base.data, task_names_keep=task_strings),
        "overwrite": args.overwrite,
        "resume": args.resume,
        "wandb_enabled": args.wandb_enabled,
        "weight_loader": _weight_loaders.CheckpointWeightLoader(args.pretrained_path),
    }
    if args.lr is not None:
        updates["lr_schedule"] = dataclasses.replace(base.lr_schedule, peak_lr=args.lr, decay_lr=args.lr)
    for field_name in (
        "num_train_steps",
        "batch_size",
        "seed",
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


def build_loader(config: _config.TrainConfig, task_strings: tuple[str, ...], batch_size: int, seed: int, data_sharding):
    cfg = dataclasses.replace(
        config,
        batch_size=batch_size,
        seed=seed,
        data=dataclasses.replace(config.data, task_names_keep=task_strings),
    )
    return _data_loader.create_data_loader(cfg, sharding=data_sharding, shuffle=True)


def _write_hparams(args: CliArgs, config: _config.TrainConfig, extra: dict) -> None:
    sched = config.lr_schedule
    hp = {
        "exp_name": config.exp_name,
        "mode": "cotrain",
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
            "jax_device_count": jax.device_count(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "save_interval": config.save_interval,
            "keep_period": config.keep_period,
            "seed": config.seed,
            "repo_id": config.data.repo_id,
            "checkpoint_dir": str(config.checkpoint_dir),
            **extra,
        },
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    p = pathlib.Path(config.checkpoint_dir) / "hparams.json"
    p.write_text(json.dumps(hp, indent=2, default=str))
    logging.info(f"Wrote hyperparameters -> {p}")


# === main =====================================================================================


def main(args: CliArgs):
    init_logging()
    logging.info(f"Running on: {platform.node()} | {args}")

    repo_id = _config.get_config(args.config_name).data.repo_id
    ft_ids, old_ids, pretrain_ids = resolve_task_sets(args)
    describe_split(ft_ids, "NEW (finetune)")
    if old_ids:
        describe_split(old_ids, "OLD (replay)")
    ft_strings = task_strings_for_ids(repo_id, ft_ids)
    old_strings = task_strings_for_ids(repo_id, old_ids) if old_ids else ()

    config = build_config(args, ft_strings)
    n_dev = jax.device_count()
    b_new, b_old = split_batch(config.batch_size, args.old_fraction, n_dev)
    logging.info(
        f"[cotrain] batch {config.batch_size} = {b_new} new ({len(ft_ids)} task(s)) + {b_old} old "
        f"({len(old_ids)} task(s): {len(set(old_ids) & set(pretrain_ids))} pretrain + "
        f"{len(set(old_ids) - set(pretrain_ids))} carried) | effective old fraction {b_old / config.batch_size:.3f}"
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
    _write_hparams(
        args,
        config,
        {
            "task_ids_trained": ft_ids,
            "old_task_ids_replayed": old_ids,
            "num_old_tasks": len(old_ids),
            "batch_new": b_new,
            "batch_old": b_old,
            "old_fraction_effective": b_old / config.batch_size,
            "pretrained_path": args.pretrained_path,
        },
    )
    if config.wandb_enabled:
        wandb.init(
            name=config.exp_name,
            project=config.project_name,
            config={
                **dataclasses.asdict(config),
                "mode": "cotrain",
                "task_ids": ft_ids,
                "old_task_ids": old_ids,
                "old_fraction": args.old_fraction,
            },
        )
    else:
        wandb.init(mode="disabled")

    new_loader = build_loader(config, ft_strings, b_new, config.seed, data_sharding)
    new_iter = iter(new_loader)
    old_iter = iter(build_loader(config, old_strings, b_old, config.seed + 1, data_sharding)) if b_old > 0 else None

    def next_batch():
        new = next(new_iter)
        if old_iter is None:
            return new
        old = next(old_iter)
        return jax.tree.map(lambda a, b: jnp.concatenate([a, b], axis=0), new, old)

    new_mask = jax.device_put(np.concatenate([np.ones(b_new, np.float32), np.zeros(b_old, np.float32)]), data_sharding)
    batch = next_batch()
    logging.info(f"Initialized co-training batch:\n{training_utils.array_tree_to_info(batch)}")

    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)
    state, state_sharding = init_train_state(config, init_rng, mesh, tx)
    jax.block_until_ready(state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(state.params)}")
    if resuming:
        state = _checkpoints.restore_state(mngr, state, new_loader)

    step_fn = jax.jit(
        functools.partial(train_step, config, state_sharding.params),
        in_shardings=(replicated_sharding, state_sharding, data_sharding, data_sharding),
        out_shardings=(state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    start_step = int(state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps), initial=start_step, total=config.num_train_steps, dynamic_ncols=True
    )
    metrics_path = pathlib.Path(config.checkpoint_dir) / "train_metrics.jsonl"
    metrics_file = open(metrics_path, "a")  # noqa: SIM115
    logging.info(f"Logging training metrics to: {metrics_path}")

    infos: list[dict[str, Any]] = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            state, info = step_fn(train_rng, state, batch, new_mask)
        infos.append(info)
        if step % config.log_interval == 0:
            reduced = {k: float(np.mean([jax.device_get(d[k]) for d in infos])) for k in infos[0]}
            if b_old == 0:
                reduced.pop("old_loss")
            pbar.write(f"Step {step}: " + ", ".join(f"{k}={v:.4f}" for k, v in reduced.items()))
            metrics_file.write(json.dumps({"step": int(step), **reduced}) + "\n")
            metrics_file.flush()
            wandb.log(reduced, step=step)
            infos = []
        batch = next_batch()

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            # new_loader carries the norm-stats assets saved alongside the checkpoint (shared config).
            _checkpoints.save_state(mngr, state, new_loader, step)

    metrics_file.close()
    logging.info("Waiting for checkpoint manager to finish")
    mngr.wait_until_finished()


if __name__ == "__main__":
    main(tyro.cli(CliArgs))
