"""Exercise the real Pi0/π0.5 modules at reduced dimensions, without downloads.

Only vocabulary/model sizes are replaced in this test. Vision, attention,
flow-matching loss, autodiff, optimizer and action sampling use production code.
This is not validation of pretrained weights or full-size GPU execution.
"""

import functools

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import gemma
from openpi.models import pi0_config
from openpi.models import siglip
from openpi.training import config
from openpi.training import optimizer
from openpi.training import sambar
from openpi.training import sharding
from openpi.training import weight_loaders


def test_reduced_pi05_loss_importance_update_and_actions(monkeypatch):
    vision_module = siglip.Module
    monkeypatch.setattr(siglip, "Module", lambda **kwargs: vision_module(**dict(kwargs, variant="mu/28")))
    monkeypatch.setattr(gemma, "PALIGEMMA_VOCAB_SIZE", 128)
    spec = pi0_config.Pi0Config(
        pi05=True,
        dtype="float32",
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_dim=8,
        action_horizon=2,
        max_token_len=4,
    )
    cfg = config.TrainConfig(
        name="cpu_pi05",
        model=spec,
        weight_loader=weight_loaders.NoOpWeightLoader(),
        ema_decay=None,
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=1, decay_steps=10),
    )
    mesh = sharding.make_mesh(1)
    tx = optimizer.create_optimizer(cfg.optimizer, cfg.lr_schedule, weight_decay_mask=None)
    state, state_sharding = sambar.init_train_state(cfg, jax.random.key(0), mesh, tx)
    batch = (spec.fake_obs(), spec.fake_act())
    options = sambar.SambarArgs(fisher_batches=1)
    fisher = sambar.make_fisher_step(cfg, options)
    with sharding.set_mesh(mesh):
        omega = fisher(jax.random.key(1), state, batch)
    assert all(np.isfinite(x).all() and np.all(np.asarray(x) >= 0) for x in jax.tree.leaves(omega))
    assert any(np.any(np.asarray(x) > 0) for x in jax.tree.leaves(omega))

    reference = sambar.make_reference(state, options)
    zeros = jax.tree.map(jnp.zeros_like, reference)
    step = jax.jit(functools.partial(sambar.train_step, cfg, options, state_sharding.params))
    with sharding.set_mesh(mesh):
        state, task_loss, total_loss, grad_norm = step(
            jax.random.key(2),
            state,
            batch,
            zeros,
            reference,
            jnp.float32(1),
        )
        # The inherited warmup schedule starts at zero; the next step updates weights.
        state, task_loss, total_loss, grad_norm = step(
            jax.random.key(3),
            state,
            batch,
            zeros,
            reference,
            jnp.float32(1),
        )
    assert int(state.step) == 2
    assert np.isfinite([task_loss, total_loss, grad_norm]).all()
    assert any(
        not np.array_equal(a, b)
        for a, b in zip(jax.tree.leaves(reference), jax.tree.leaves(sambar.parameters(state.params)), strict=True)
    )
    policy = nnx.merge(state.model_def, state.params)
    policy.eval()
    actions = jax.jit(policy.sample_actions)(jax.random.key(4), batch[0], num_steps=2)
    assert actions.shape == (1, 2, 8)
    assert np.isfinite(actions).all()
