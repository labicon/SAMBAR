"""Small CPU checks for the estimator, anchor, and approved saved-history lifecycle."""

import dataclasses
from types import SimpleNamespace

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from openpi.training import importance
from openpi.training import sambar
from openpi.training import sambar_math
from openpi.training import sharding
from openpi.training.utils import TrainState


class ToyModel(nnx.Module):
    def __init__(self, weights=(0.0, 0.0)):
        self.w = nnx.Param(jnp.array(weights))

    def compute_loss(self, rng, obs, actions, *, train):
        return (obs @ self.w.value - actions) ** 2


def toy_state(weights=(0.0, 0.0)):
    model = ToyModel(weights)
    graph, params = nnx.split(model)
    tx = optax.adam(0.01)
    return TrainState(
        step=jnp.array(0), params=params, model_def=graph, tx=tx, opt_state=tx.init(params), ema_decay=None
    )


def assert_tree_equal(a, b):
    assert jax.tree.structure(a) == jax.tree.structure(b)
    for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b), strict=True):
        np.testing.assert_array_equal(x, y)


def test_mean_gradient_is_squared_after_inner_reduction():
    state = toy_state()
    fn = sambar.make_fisher_step(SimpleNamespace(trainable_filter=nnx.Param), sambar.SambarArgs())
    obs = jnp.ones((2, 2))
    # Individual gradients are opposite and nonzero. Their batch mean must cancel.
    cancel = fn(jax.random.key(0), state, (obs, jnp.array([1.0, -1.0])))
    np.testing.assert_array_equal(cancel["w"], [0.0, 0.0])
    reinforce = fn(jax.random.key(0), state, (obs, jnp.array([1.0, 1.0])))
    np.testing.assert_array_equal(reinforce["w"], [4.0, 4.0])
    omega = sambar.compute_fisher(
        fn,
        jax.random.key(0),
        state,
        iter([(obs, jnp.array([1.0, -1.0])), (obs, jnp.array([1.0, 1.0]))]),
        2,
        sharding.make_mesh(1),
    )
    np.testing.assert_array_equal(omega["w"], [2.0, 2.0])


def test_selected_anchor_epsilon_weights_current_parameters():
    old, current = {"w": jnp.array([2.0, 2.0])}, {"w": jnp.array([8.0, 8.0])}
    zero = {"w": jnp.zeros(2)}
    assert_tree_equal(sambar_math.weighted_z(old, current, zero, zero, 1.0, 1e-8), current)
    z = sambar_math.weighted_z(old, current, {"w": jnp.array([3.0, 0.0])}, {"w": jnp.array([1.0, 0.0])}, 1.0, 1.0)
    np.testing.assert_allclose(z["w"], [4.4, 8.0])


@pytest.mark.parametrize("count", [32, 117])
def test_three_stage_task_count_weighted_history(count):
    history = {"w": jnp.array([2.0])}
    for i, value in enumerate([4.0, 6.0, 8.0]):
        history = sambar_math.update_history(history, {"w": jnp.array([value])}, count + i, 1)
    np.testing.assert_allclose(history["w"], [(2 * count + 18) / (count + 3)])
    with pytest.raises(ValueError):
        sambar_math.update_history(history, history, 0, 1)


def test_importance_roundtrip_and_mismatched_checkpoint(tmp_path):
    path = tmp_path / "importance"
    omega = {"layer": {"w": jnp.array([0.0, 1e-12, 2.0])}}
    importance.save(path, omega, ["dataset::task-a"], setting="narrow", fisher_batches=50)
    restored, meta = importance.load(path)
    assert isinstance(restored["layer"]["w"], np.ndarray)
    assert_tree_equal(restored, omega)
    assert meta["task_count"] == 1
    restored, _ = importance.load(path, template=omega)
    assert isinstance(restored["layer"]["w"], jax.Array)
    assert_tree_equal(restored, omega)
    with pytest.raises(ValueError, match="shapes"):
        importance.load(path, template={"layer": {"w": jnp.zeros(4)}})
    with pytest.raises(ValueError, match="tree"):
        importance.load(path, template={"other": jnp.zeros(3)})
    with pytest.raises(FileExistsError):
        importance.save(path, omega, ["dataset::task-a"], setting="narrow", fisher_batches=50)
    with pytest.raises(ValueError, match="disjoint"):
        importance.merge_task_ids(meta["tasks"], ["dataset::task-a"])


def test_checkpoint_preserves_method_and_optimizer_for_next_update(tmp_path):
    from openpi.training import checkpoints

    state = toy_state((0.2, -0.3))
    cfg = SimpleNamespace(trainable_filter=nnx.Param)
    args = sambar.SambarArgs()
    theta = sambar.parameters(state.params)
    z = jax.tree.map(lambda x: x + 0.3, theta)
    u = jax.tree.map(jnp.ones_like, theta)
    batch = (jnp.array([[1.0, 2.0], [2.0, 1.0]]), jnp.array([0.5, -0.1]))
    state, *_ = sambar.train_step(cfg, args, None, jax.random.key(0), state, batch, u, z, jnp.array(1.0))
    extra = dict(theta_pre=theta, omega_old=theta, omega_new=theta, z=z, u=u)
    loader = SimpleNamespace(data_config=lambda: SimpleNamespace(norm_stats=None, asset_id=None))
    manager, _ = checkpoints.initialize_checkpoint_dir(
        tmp_path / "run", keep_period=None, overwrite=False, resume=False
    )
    checkpoints.save_state(manager, state, loader, 1, method_state=extra)
    manager.wait_until_finished()
    restored = checkpoints.restore_state(manager, dataclasses.replace(state, method_state=extra), loader)
    assert_tree_equal(restored.method_state, extra)
    recovered = dataclasses.replace(restored, method_state=None)
    a = sambar.train_step(cfg, args, None, jax.random.key(1), state, batch, u, z, jnp.array(1.0))
    b = sambar.train_step(
        cfg,
        args,
        None,
        jax.random.key(1),
        recovered,
        batch,
        restored.method_state["u"],
        restored.method_state["z"],
        jnp.array(1.0),
    )
    assert_tree_equal(a, b)
    manager.close()


def test_stage_uses_cached_history_and_resumes_without_old_demos(tmp_path, monkeypatch):
    import itertools

    from openpi.training import runner

    old_dir = tmp_path / "pretrained"
    old_omega = {"w": jnp.ones(2)}
    importance.save(
        old_dir / "importance", old_omega, [f"data::old-{i}" for i in range(32)], setting="narrow", fisher_batches=2
    )
    config = SimpleNamespace(
        name="toy",
        exp_name="stage",
        batch_size=2,
        num_train_steps=2,
        seed=0,
        fsdp_devices=1,
        checkpoint_dir=tmp_path / "stage",
        keep_period=None,
        overwrite=False,
        resume=False,
        log_interval=1,
        save_interval=1,
        wandb_enabled=False,
    )
    monkeypatch.setattr(runner, "build_config", lambda args: config)
    monkeypatch.setattr(runner.optimizer, "create_optimizer", lambda *a, **k: None)
    config.optimizer = config.lr_schedule = None
    config.trainable_filter = nnx.Param

    def initialize(config, rng, mesh, tx):
        state = toy_state((0.2, -0.3))
        return state, sharding.fsdp_sharding(state, mesh, log=False)

    monkeypatch.setattr(sambar, "init_train_state", initialize)
    batch = (jnp.array([[1.0, 2.0], [2.0, 1.0]]), jnp.array([0.5, -0.1]))

    class Loader:
        def data_config(self):
            return SimpleNamespace(norm_stats=None, asset_id=None)

        def __iter__(self):
            return itertools.repeat(batch)

    seen = []

    def make_loader(args, *a):
        seen.append(args.finetune_tasks)
        assert args.mode == "sambar" and args.finetune_tasks == ("new-task",)
        return Loader()

    monkeypatch.setattr(runner.curriculum, "create_mixed_data_loader", make_loader)
    monkeypatch.setattr(runner.wandb, "init", lambda **k: None)
    monkeypatch.setattr(runner.wandb, "log", lambda *a, **k: None)
    monkeypatch.setattr(runner.wandb, "finish", lambda: None)
    args = runner.Args(
        exp_name="stage",
        mode="sambar",
        pretrained_path=str(old_dir),
        finetune_tasks=("new-task",),
        sambar=sambar.SambarArgs(fisher_batches=2),
    )
    result = runner.run_stage(args)
    omega, meta = importance.load(result / "importance")
    assert meta["task_count"] == 33 and len(seen) == 2
    assert np.isfinite(omega["w"]).all()
    config.resume = True
    # A completed run must recover state without recomputing current or old importance.
    monkeypatch.setattr(sambar, "compute_fisher", lambda *a, **k: pytest.fail("unexpected recomputation"))
    assert runner.run_stage(dataclasses.replace(args, resume=True)) == result
