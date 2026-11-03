"""CPU integration checks using real training/checkpoint plumbing and synthetic inputs.

The policy and datasets are deliberately tiny; these tests do not validate full π0.5
weights, downloaded demonstrations, CUDA execution or robot/simulator performance.
Run with XLA_FLAGS=--xla_force_host_platform_device_count=2 to also exercise sharding.
"""

import dataclasses
import json
from types import SimpleNamespace

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import model
from openpi.shared import array_typing as at
from openpi.training import checkpoints
from openpi.training import config
from openpi.training import curriculum
from openpi.training import data_loader
from openpi.training import importance
from openpi.training import optimizer
from openpi.training import runner
from openpi.training import sambar
from openpi.training import task_split
from openpi.training import weight_loaders


class SmallPolicy(model.BaseModel):
    def __init__(self, spec, rngs):
        super().__init__(spec.action_dim, spec.action_horizon, spec.max_token_len)
        self.input = nnx.Linear(spec.action_dim, spec.width, rngs=rngs)
        self.hidden = nnx.Linear(spec.width, spec.width, rngs=rngs)
        self.output = nnx.Linear(spec.width, spec.action_dim, rngs=rngs)

    def compute_loss(self, rng, observation, actions, *, train=False):
        prediction = self.output(jnp.tanh(self.hidden(jnp.tanh(self.input(observation.state)))))
        return jnp.mean((prediction[:, None, :] - actions) ** 2, axis=-1)

    def sample_actions(self, rng, observation, **kwargs):
        prediction = self.output(jnp.tanh(self.hidden(jnp.tanh(self.input(observation.state)))))
        return jnp.repeat(prediction[:, None, :], self.action_horizon, axis=1)


@dataclasses.dataclass(frozen=True)
class SmallConfig(model.BaseModelConfig):
    action_dim: int = 8
    action_horizon: int = 2
    max_token_len: int = 4
    width: int = 8

    @property
    def model_type(self):
        return model.ModelType.PI05

    def create(self, rng):
        return SmallPolicy(self, nnx.Rngs(rng))

    def inputs_spec(self, *, batch_size=1):
        with at.disable_typechecking():
            observation = model.Observation(
                images={"base_0_rgb": jax.ShapeDtypeStruct((batch_size, 8, 8, 3), jnp.float32)},
                image_masks={"base_0_rgb": jax.ShapeDtypeStruct((batch_size,), jnp.bool_)},
                state=jax.ShapeDtypeStruct((batch_size, self.action_dim), jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct((batch_size, self.max_token_len), jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct((batch_size, self.max_token_len), jnp.bool_),
            )
        return observation, jax.ShapeDtypeStruct((batch_size, self.action_horizon, self.action_dim), jnp.float32)


@pytest.fixture
def cpu_workflow(tmp_path, monkeypatch):
    # A 4 MiB matrix makes the original FSDP threshold select a real shard when
    # two virtual CPU devices are requested. One-device tests use a tiny policy.
    fsdp = 2 if jax.device_count() > 1 else 1
    spec = SmallConfig(width=1024 if fsdp == 2 else 8)
    names = {i: f"task-{i}" for i in range(40)}
    names.update(zip((10, 24, 38), curriculum.PRETRAIN_HOLDOUT_TASKS, strict=True))
    monkeypatch.setattr(task_split, "task_index_to_string", lambda repo: names)
    monkeypatch.setattr(
        curriculum.lerobot_dataset,
        "LeRobotDatasetMetadata",
        lambda *a, **kw: SimpleNamespace(tasks={i: f"broad-{i}" for i in range(90)}),
    )
    selected = []

    def dataset(repo, root, horizon, data_config, tasks):
        selected.append((repo, tasks))
        return data_loader.FakeDataset(spec, 8)

    monkeypatch.setattr(curriculum, "_make_lerobot_dataset", dataset)
    bases = {
        "sambar_" + setting: config.TrainConfig(
            name="sambar_" + setting,
            model=spec,
            data=config.FakeDataConfig(),
            weight_loader=weight_loaders.NoOpWeightLoader(),
            ema_decay=None,
            lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=1, decay_steps=10, peak_lr=0.001),
            checkpoint_base_dir=str(tmp_path),
            batch_size=2,
            num_workers=0,
            num_train_steps=3,
            save_interval=1,
            keep_period=1,
            log_interval=1,
            fsdp_devices=fsdp,
            wandb_enabled=False,
        )
        for setting in ("narrow", "broad")
    }
    monkeypatch.setattr(runner.configs, "get_config", bases.__getitem__)
    return spec, selected


def assert_finite_checkpoint(path):
    params = model.restore_params(path / "params", restore_type=np.ndarray)
    assert params
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(params))
    metrics = [json.loads(s) for s in (path.parent / "train_metrics.jsonl").read_text().splitlines()]
    assert metrics
    assert all(np.isfinite(row["total_loss"]) for row in metrics)
    return params


@pytest.mark.parametrize(("setting", "count"), [("narrow", 32), ("broad", 117)])
def test_pretrain_and_three_stage_checkpoint_handoff(cpu_workflow, setting, count):
    _, selected = cpu_workflow
    options = sambar.SambarArgs(fisher_batches=2, fisher_refresh=2, dual_period=2, warmup_steps=1)
    base = runner.Args(exp_name="pretrain", mode="pretrain", setting=setting, sambar=options)
    pretrained = runner.main(base)
    assert_finite_checkpoint(pretrained)
    assert importance.read_metadata(pretrained / "importance")["task_count"] == count
    sequence = dataclasses.replace(base, mode="sequential", exp_name="sequence", pretrained_path=str(pretrained))
    selected.clear()
    final = runner.main(sequence)
    before = assert_finite_checkpoint(final)
    for stage in range(1, 4):
        path = final.parent.parent / f"stage{stage}" / final.name
        assert importance.read_metadata(path / "importance")["task_count"] == count + stage
    assert all(tasks is not None and len(tasks) == 1 for _, tasks in selected)
    runner.main(dataclasses.replace(sequence, resume=True))
    after = model.restore_params(final / "params", restore_type=np.ndarray)
    for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(after), strict=True):
        np.testing.assert_array_equal(a, b)
    exported = pretrained.parent / "exported_importance"
    runner.main(
        dataclasses.replace(base, mode="importance", pretrained_path=str(pretrained), importance_path=str(exported))
    )
    assert importance.read_metadata(exported)["task_count"] == count


def test_resume_after_an_intermediate_checkpoint(cpu_workflow, monkeypatch):
    base = runner.Args(exp_name="pretrain", mode="pretrain", sambar=sambar.SambarArgs(fisher_batches=1))
    pretrained = runner.main(base)
    stage = dataclasses.replace(
        base,
        exp_name="interrupted",
        mode="sambar",
        pretrained_path=str(pretrained),
        finetune_task_ids=(3,),
        num_train_steps=4,
    )
    save = checkpoints.save_state

    def interrupt(manager, state, loader, step, **kwargs):
        save(manager, state, loader, step, **kwargs)
        if step == 1:
            manager.wait_until_finished()
            manager.close()
            raise RuntimeError("simulated interruption after checkpoint")

    with monkeypatch.context() as patch:
        patch.setattr(checkpoints, "save_state", interrupt)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            runner.main(stage)
    final = runner.main(dataclasses.replace(stage, resume=True))
    assert_finite_checkpoint(final)
    assert importance.read_metadata(final / "importance")["task_count"] == 33
    steps = [json.loads(s)["step"] for s in (final.parent / "train_metrics.jsonl").read_text().splitlines()]
    assert steps == [0, 1, 2, 3]


def test_one_step_checkpoint_can_resume(cpu_workflow):
    args = runner.Args(
        exp_name="one_step", mode="pretrain", num_train_steps=1, sambar=sambar.SambarArgs(fisher_batches=1)
    )
    final = runner.main(args)
    before = (final.parent / "train_metrics.jsonl").read_text()
    runner.main(dataclasses.replace(args, resume=True))
    assert (final.parent / "train_metrics.jsonl").read_text() == before
