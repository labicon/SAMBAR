"""One SAMBAR training entry point for narrow, broad, and sequential experiments."""

from __future__ import annotations

import dataclasses
import functools
import json
import logging
import pathlib
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import wandb

from openpi.training import checkpoints
from openpi.training import config as configs
from openpi.training import curriculum
from openpi.training import importance
from openpi.training import optimizer
from openpi.training import sambar as method
from openpi.training import sambar_math
from openpi.training import sharding
from openpi.training import task_split
from openpi.training import weight_loaders


@dataclasses.dataclass(frozen=True)
class Args:
    exp_name: str
    mode: Literal["pretrain", "sambar", "sequential", "importance", "norm_stats"]
    setting: Literal["narrow", "broad"] = "narrow"
    pretrained_path: str | None = None
    importance_path: str | None = None
    split_seed: int = 7
    tasks_per_suite: int = 8
    finetune_task_ids: tuple[int, ...] | None = None
    finetune_tasks: tuple[str, ...] | None = None
    # A current-task-only LeRobot dataset can be supplied without storing old demonstrations.
    finetune_repo_id: str | None = None
    finetune_root: str | None = None
    sequence: tuple[int, ...] = (3, 8, 11)
    libero90_repo_id: str = "jesbu1/libero_90_lerobot"
    # Optional LeRobot cache parent; dataset directory is ROOT/REPO_ID.
    libero90_root: str | None = None
    pretrain_holdout_tasks: tuple[str, ...] = curriculum.PRETRAIN_HOLDOUT_TASKS
    num_train_steps: int | None = None
    batch_size: int | None = None
    lr: float | None = None
    seed: int | None = None
    num_workers: int | None = None
    fsdp_devices: int | None = None
    save_interval: int | None = None
    keep_period: int | None = None
    log_interval: int | None = None
    checkpoint_base_dir: str | None = None
    max_frames: int | None = None
    overwrite: bool = False
    resume: bool = False
    wandb_enabled: bool = False
    sambar: method.SambarArgs = dataclasses.field(default_factory=method.SambarArgs)


def checkpoint_dir(path):
    result = pathlib.Path(path).expanduser().resolve()
    return result.parent if result.name == "params" else result


def resolve_tasks(args):
    """Resolve current task identities. Finetuning never reads old-task demonstrations."""
    if args.mode in ("pretrain", "importance", "norm_stats"):
        if args.setting == "narrow":
            ids, _ = task_split.sample_pretrain_tasks(args.tasks_per_suite, args.split_seed)
            names = task_split.task_strings_for_ids(curriculum.LIBERO40_REPO_ID, ids)
            tasks = [curriculum.LIBERO40_REPO_ID + "::" + x for x in names]
        else:
            names = curriculum.pretrain_task_strings(args.pretrain_holdout_tasks)
            root = str(pathlib.Path(args.libero90_root) / args.libero90_repo_id) if args.libero90_root else None
            meta = curriculum.lerobot_dataset.LeRobotDatasetMetadata(args.libero90_repo_id, root=root)
            extra = list(meta.tasks.values())
            if len(extra) != 90:
                raise ValueError(f"Broad curriculum requires 90 LIBERO-90 tasks, found {len(extra)}")
            tasks = [curriculum.LIBERO40_REPO_ID + "::" + x for x in names]
            tasks += [args.libero90_repo_id + "::" + x for x in extra]
    else:
        if args.finetune_tasks:
            names = args.finetune_tasks
        elif args.finetune_task_ids:
            names = task_split.task_strings_for_ids(curriculum.LIBERO40_REPO_ID, list(args.finetune_task_ids))
        elif args.setting == "broad":
            names = curriculum.DEFAULT_FINETUNE_TASKS
        else:
            _, ids = task_split.sample_pretrain_tasks(args.tasks_per_suite, args.split_seed)
            names = task_split.task_strings_for_ids(curriculum.LIBERO40_REPO_ID, ids)
        tasks = [curriculum.LIBERO40_REPO_ID + "::" + x for x in names]
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("Current tasks must be nonempty and unique")
    return dataclasses.replace(args, finetune_tasks=tuple(names)), tasks


def build_config(args):
    base = configs.get_config("sambar_" + args.setting)
    updates = dict(
        exp_name=args.exp_name, overwrite=args.overwrite, resume=args.resume, wandb_enabled=args.wandb_enabled
    )
    if args.pretrained_path:
        path = (
            args.pretrained_path
            if str(args.pretrained_path).endswith("/params")
            else str(checkpoint_dir(args.pretrained_path) / "params")
        )
        updates["weight_loader"] = weight_loaders.CheckpointWeightLoader(path)
    if args.lr is not None:
        updates["lr_schedule"] = dataclasses.replace(base.lr_schedule, peak_lr=args.lr, decay_lr=args.lr)
    for key in (
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
        if getattr(args, key) is not None:
            updates[key] = getattr(args, key)
    return dataclasses.replace(base, **updates)


def _method_state(theta_pre, omega_old, omega_new, z, u):
    return dict(theta_pre=theta_pre, omega_old=omega_old, omega_new=omega_new, z=z, u=u)


def run_stage(args):
    method.init_logging()
    if args.sambar.fisher_batches < 1 or args.sambar.dual_period < 1 or args.sambar.fisher_refresh < 0:
        raise ValueError("fisher_batches and dual_period must be positive; fisher_refresh must be nonnegative")
    if args.mode in ("sambar", "importance") and not args.pretrained_path:
        raise ValueError("--pretrained-path is required")
    args, current_tasks = resolve_tasks(args)
    config = build_config(args)
    loader_args = dataclasses.replace(args, mode="pretrain") if args.mode == "importance" else args
    if args.mode == "norm_stats":
        curriculum.run_norm_stats(loader_args, config)
        return None
    if config.batch_size % jax.device_count() or config.num_train_steps < 1:
        raise ValueError("Batch size must divide across devices and train steps must be positive")
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    train_rng, init_rng, fisher_rng = jax.random.split(jax.random.key(config.seed), 3)
    tx = optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)
    state, state_sharding = method.init_train_state(config, init_rng, mesh, tx)
    jax.block_until_ready(state)
    params = method.parameters(state.params)
    param_sharding = sharding.fsdp_sharding(params, mesh, log=False)
    fisher_step = method.make_fisher_step(config, args.sambar, out_shardings=param_sharding)
    loader = curriculum.create_mixed_data_loader(loader_args, config, data_sharding)
    data_iter = iter(loader)
    if args.mode == "importance":
        omega = method.compute_fisher(fisher_step, fisher_rng, state, data_iter, args.sambar.fisher_batches, mesh)
        output = (
            pathlib.Path(args.importance_path)
            if args.importance_path
            else checkpoint_dir(args.pretrained_path) / "importance"
        )
        importance.save(output, omega, current_tasks, setting=args.setting, fisher_batches=args.sambar.fisher_batches)
        logging.info("Saved initial LIBERO importance to %s", output)
        return checkpoint_dir(args.pretrained_path)

    is_sambar = args.mode == "sambar"
    manager, resuming = checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir, keep_period=config.keep_period, overwrite=config.overwrite, resume=config.resume
    )
    manifest_path = pathlib.Path(config.checkpoint_dir) / "run.json"
    manifest = {
        "args": dataclasses.asdict(args),
        "tasks": current_tasks,
        "config": config.name,
        "steps": config.num_train_steps,
        "base_pi05_importance_available": False,
        "resolved": {k: getattr(config, k) for k in ("batch_size", "seed", "fsdp_devices")},
    }
    if resuming:
        previous = json.loads(manifest_path.read_text())
        # A resume may change runtime logging only, not the scientific run definition.
        runtime_keys = {
            "resume",
            "overwrite",
            "wandb_enabled",
            "log_interval",
            "save_interval",
            "keep_period",
            "checkpoint_base_dir",
            "num_workers",
            "fsdp_devices",
        }
        for key in manifest["args"].keys() - runtime_keys:
            if previous["args"][key] != json.loads(json.dumps(manifest["args"][key])):
                raise ValueError(f"Cannot resume with a changed {key}")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    old_tasks = []
    omega_old = omega_new = None
    if is_sambar:
        history_path = (
            pathlib.Path(args.importance_path)
            if args.importance_path
            else checkpoint_dir(args.pretrained_path) / "importance"
        )
        metadata = importance.read_metadata(history_path)
        if metadata["setting"] != args.setting:
            raise ValueError("Initial importance belongs to a different curriculum")
        old_tasks = metadata["tasks"]
        all_tasks = importance.merge_task_ids(old_tasks, current_tasks)
    else:
        all_tasks = current_tasks
    if resuming:
        if is_sambar:
            abstract = jax.tree.map(lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype, sharding=x.sharding), params)
            state = dataclasses.replace(
                state, method_state=_method_state(abstract, abstract, abstract, abstract, abstract)
            )
        state = checkpoints.restore_state(manager, state, loader)
        saved = state.method_state
        state = dataclasses.replace(state, method_state=None)
        if is_sambar:
            theta_pre, omega_old, omega_new, z, u = [
                saved[k] for k in ("theta_pre", "omega_old", "omega_new", "z", "u")
            ]
        elif saved is not None:
            raise ValueError("Pretraining checkpoint unexpectedly contains a SAMBAR method state")
    if not resuming or not is_sambar:
        theta_pre = jax.device_put(method.make_reference(state, args.sambar), param_sharding)
    new_loader_iter = iter(curriculum.create_mixed_data_loader(loader_args, config, data_sharding))
    weighted_z_fn = jax.jit(
        functools.partial(method.weighted_z, beta=args.sambar.beta, eps=args.sambar.eps),
        in_shardings=(param_sharding,) * 4,
        out_shardings=param_sharding,
    )
    dual_fn = jax.jit(
        functools.partial(method.dual_step, args.sambar),
        in_shardings=(param_sharding,) * 3,
        out_shardings=(param_sharding, replicated, replicated),
    )
    if not resuming or not is_sambar:
        if is_sambar:
            omega_old, _ = importance.load(history_path, template=params)
            omega_new = method.compute_fisher(
                fisher_step, fisher_rng, state, new_loader_iter, args.sambar.fisher_batches, mesh
            )
            z = weighted_z_fn(theta_pre, method.parameters(state.params), omega_old, omega_new)
        else:
            z = jax.jit(lambda t: jax.tree.map(jnp.copy, t), out_shardings=param_sharding)(theta_pre)
        u = jax.jit(lambda t: jax.tree.map(jnp.zeros_like, t), out_shardings=param_sharding)(z)
    step_fn = jax.jit(
        functools.partial(method.train_step, config, args.sambar, state_sharding.params),
        in_shardings=(replicated, state_sharding, data_sharding, param_sharding, param_sharding, replicated),
        out_shardings=(state_sharding, replicated, replicated, replicated),
        donate_argnums=(1,),
    )
    wandb.init(
        mode="online" if args.wandb_enabled else "disabled", name=args.exp_name, project="sambar", config=manifest
    )
    drift = dual_norm = 0.0
    infos = []
    start = int(state.step)
    # Preserve the original RNG folding, gradient update, consensus and dual-update order.
    with (pathlib.Path(config.checkpoint_dir) / "train_metrics.jsonl").open("a") as metrics:
        for step in range(start, config.num_train_steps):
            batch = next(data_iter)
            with sharding.set_mesh(mesh):
                active = jnp.float32(is_sambar and step >= args.sambar.warmup_steps)
                rng_step = jax.random.fold_in(train_rng, state.step)
                state, task_loss, total_loss, grad_norm = step_fn(rng_step, state, batch, u, z, active)
            if is_sambar:
                if args.sambar.fisher_refresh > 0 and (step + 1) % args.sambar.fisher_refresh == 0:
                    omega_new = None
                    if (step + 1) % args.sambar.dual_period == 0:
                        z = None
                    omega_new = method.compute_fisher(
                        fisher_step,
                        jax.random.fold_in(fisher_rng, step + 1),
                        state,
                        new_loader_iter,
                        args.sambar.fisher_batches,
                        mesh,
                    )
                if (step + 1) % args.sambar.dual_period == 0:
                    with sharding.set_mesh(mesh):
                        theta = method.parameters(state.params)
                        z = weighted_z_fn(theta_pre, theta, omega_old, omega_new)
                        u, drift, dual_norm = dual_fn(theta, u, z)
            infos.append(
                jax.device_get(
                    dict(
                        task_loss=task_loss,
                        total_loss=total_loss,
                        grad_norm=grad_norm,
                        rho_scale=active,
                        drift=drift,
                        dual_norm=dual_norm,
                    )
                )
            )
            if step % config.log_interval == 0:
                values = {k: float(np.mean([d[k] for d in infos])) for k in infos[0]}
                metrics.write(json.dumps(dict(step=step, **values)) + "\n")
                metrics.flush()
                logging.info("Step %s: %s", step, values)
                wandb.log(values, step=step)
                infos = []
            if (step % config.save_interval == 0 and step > start) or step == config.num_train_steps - 1:
                extra = _method_state(theta_pre, omega_old, omega_new, z, u) if is_sambar else None
                checkpoints.save_state(manager, state, loader, step, method_state=extra)
                extra = None
    manager.wait_until_finished()
    final_dir = pathlib.Path(config.checkpoint_dir) / str(config.num_train_steps - 1)
    # Estimate the completed task at its final weights using exactly the selected estimator.
    # The previous data is never loaded here; cached history is carried forward by task count.
    if not (final_dir / "importance").exists():
        z = u = theta_pre = omega_new = params = None
        if resuming:
            saved = None
        current = method.compute_fisher(
            fisher_step, fisher_rng, state, new_loader_iter, args.sambar.fisher_batches, mesh
        )
        if is_sambar:
            current = jax.jit(
                functools.partial(sambar_math.update_history, old_count=len(old_tasks), new_count=len(current_tasks)),
                out_shardings=param_sharding,
            )(omega_old, current)
        importance.save(
            final_dir / "importance",
            current,
            all_tasks,
            setting=args.setting,
            fisher_batches=args.sambar.fisher_batches,
        )
    manager.close()
    wandb.finish()
    logging.info("Final checkpoint and importance: %s", final_dir)
    return final_dir


def main(args: Args):
    if args.mode != "sequential":
        return run_stage(args)
    if not args.pretrained_path:
        raise ValueError("Sequential training needs --pretrained-path")
    # The published sequence is narrow-pretrained: pot, drawer, rack (IDs 3, 8, 11).
    # Broad runs can explicitly supply an ordered finetune-tasks sequence.
    if args.setting == "broad":
        sequence = args.finetune_tasks or curriculum.DEFAULT_FINETUNE_TASKS
    else:
        sequence = task_split.task_strings_for_ids(curriculum.LIBERO40_REPO_ID, list(args.sequence))
    previous = args.pretrained_path
    history = args.importance_path
    for index, task in enumerate(sequence, start=1):
        stage = dataclasses.replace(
            args,
            mode="sambar",
            exp_name=f"{args.exp_name}/stage{index}",
            pretrained_path=previous,
            importance_path=history,
            finetune_tasks=(task,),
            finetune_task_ids=None,
        )
        final = run_stage(stage)
        previous, history = str(final / "params"), str(final / "importance")
    return final
