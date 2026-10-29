"""SAMBAR numerical implementation extracted from train_pi05_weighted_taco_ar.py.

Source: pi05_taco commit 8245c09fe15ed542a2aa4740076d8ca0b417bc9b.
The minibatch-mean-gradient-square estimator and stabilized weighted anchor are retained.
Parameter selection is fixed to all parameters.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import sambar_math
from openpi.training.sambar_math import weighted_z as weighted_z
import openpi.training.config as _config
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


@dataclasses.dataclass(frozen=True)
class SambarArgs:
    rho: float = 0.02
    dual_period: int = 1
    beta: float = 1.0
    eps: float = 1e-8
    fisher_batches: int = 50
    fisher_refresh: int = 0
    warmup_steps: int = 0
    anchor_path: str | None = None


def parameters(params):
    return params.filter(nnx.Param).to_pure_dict()


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


def make_reference(state: training_utils.TrainState, wargs: SambarArgs) -> Any:
    """Frozen all-parameter reference, copied independently of donated training buffers."""
    z = parameters(state.params)
    if wargs.anchor_path is None:
        return jax.tree.map(jnp.copy, z)
    logging.info(f"[sambar] loading frozen θ_pre reference from anchor: {wargs.anchor_path}")
    anchor = _model.restore_params(download.maybe_download(wargs.anchor_path), restore_type=np.ndarray)
    flat_anchor = traverse_util.flatten_dict(anchor, sep="/")
    flat_z = traverse_util.flatten_dict(z, sep="/")
    missing = [k for k in flat_z if k not in flat_anchor]
    if missing:
        raise ValueError(f"anchor checkpoint missing {len(missing)} all-parameter keys, e.g. {missing[:3]}")
    # keep anchor leaves on host (numpy); the caller's device_put shards them across the mesh
    flat_new = {k: np.asarray(flat_anchor[k], dtype=jnp.asarray(flat_z[k]).dtype) for k in flat_z}
    return traverse_util.unflatten_dict(flat_new, sep="/")


def make_fisher_step(config: _config.TrainConfig, wargs: SambarArgs, out_shardings=None):
    """Jitted: square of the gradient of the mean BC loss for one batch.
    out_shardings keeps the squared-grad tree FSDP-sharded — without it XLA may emit the
    output replicated, which can exhaust memory on large parameter leaves."""

    def fisher_step(rng, state: training_utils.TrainState, batch):
        model = nnx.merge(state.model_def, state.params)
        model.train()
        obs, actions = batch

        def loss_fn(m, rng_, obs_, act_):
            return jnp.mean(m.compute_loss(rng_, obs_, act_, train=True))

        diff_state = nnx.DiffState(0, config.trainable_filter)
        grads = nnx.grad(loss_fn, argnums=diff_state)(model, rng, obs, actions)
        g_all = parameters(grads)
        return jax.tree.map(lambda g: g * g, g_all)

    return jax.jit(fisher_step, out_shardings=out_shardings)


def compute_fisher(fisher_step, rng, state, loader_iter, n_batches: int, mesh) -> Any:
    """Average squared-grad over n_batches → diagonal Fisher Ω (all-parameter pure-dict, on device)."""
    accum = None
    for i in range(n_batches):
        batch = next(loader_iter)
        with sharding.set_mesh(mesh):
            sq = fisher_step(jax.random.fold_in(rng, i), state, batch)
        accum = sq if accum is None else jax.tree.map(jnp.add, accum, sq)
    jax.block_until_ready(accum)
    return jax.tree.map(lambda a: a / float(n_batches), accum)


def train_step(
    config: _config.TrainConfig,
    wargs: SambarArgs,
    param_sharding,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
    u: Any,
    z: Any,
    rho_scale: jnp.ndarray,
) -> tuple[training_utils.TrainState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    model = nnx.merge(state.model_def, state.params)
    model.train()
    obs, actions = batch

    def loss_fn(m: _model.BaseModel, rng_: at.KeyArrayLike, obs_, act_):
        task_loss = jnp.mean(m.compute_loss(rng_, obs_, act_, train=True))
        params = nnx.state(m)
        penalty = sambar_math.penalty(parameters(params), u, z, wargs.rho)
        return task_loss + rho_scale * penalty, task_loss

    rng_step = jax.random.fold_in(rng, state.step)
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (total_loss, task_loss), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, rng_step, obs, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params)
    full_params = nnx.state(model)
    # Pin the reassembled full param tree to the FSDP sharding. Without this, XLA's
    # propagation materializes it replicated (a 16GB/GPU temp = the whole ~4B-param tree)
    # before resharding to the state out_sharding.
    if param_sharding is not None:
        full_params = jax.lax.with_sharding_constraint(full_params, param_sharding)

    new_state = dataclasses.replace(state, step=state.step + 1, params=full_params, opt_state=new_opt_state)
    gnorm = optax.global_norm(grads)
    return new_state, task_loss, total_loss, gnorm


def dual_step(wargs: SambarArgs, theta_ft: Any, u: Any, z: Any):
    """u ← u + ρ (θ_ft − z), with drift and dual-norm diagnostics."""
    new_u = jax.tree.map(lambda u_leaf, t_leaf, z_leaf: u_leaf + wargs.rho * (t_leaf - z_leaf), u, theta_ft, z)
    drift = jnp.sqrt(
        sum(
            jnp.sum((t_leaf - z_leaf) ** 2)
            for t_leaf, z_leaf in zip(jax.tree.leaves(theta_ft), jax.tree.leaves(z), strict=True)
        )
    )
    return new_u, drift, sambar_math.dual_norm(new_u)
