"""Check curriculum identities and routing without downloading demonstrations."""

from types import SimpleNamespace

from openpi.training import curriculum
from openpi.training import runner
from openpi.training import task_split


def test_narrow_split_and_default_sequence():
    old, holdout = task_split.sample_pretrain_tasks(8, 7)
    assert len(old) == 32 and len(holdout) == 8
    assert set(old).isdisjoint(holdout)
    assert set(old) | set(holdout) == set(range(40))
    assert {3, 8, 11}.issubset(holdout)


def test_broad_is_27_plus_90(monkeypatch):
    names = [f"old{i}" for i in range(27)] + list(curriculum.PRETRAIN_HOLDOUT_TASKS)
    monkeypatch.setattr(curriculum, "task_strings_for_ids", lambda repo, ids: names)
    monkeypatch.setattr(
        curriculum.lerobot_dataset,
        "LeRobotDatasetMetadata",
        lambda *a, **kw: SimpleNamespace(tasks={i: f"libero90-{i}" for i in range(90)}),
    )
    _, tasks = runner.resolve_tasks(runner.Args(exp_name="broad", mode="pretrain", setting="broad"))
    assert len(tasks) == 117
    assert all(not t.endswith(h) for t in tasks for h in curriculum.PRETRAIN_HOLDOUT_TASKS)


def test_explicit_current_task_does_not_access_pretrain_metadata(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Old dataset metadata must not be needed for explicit task names")

    monkeypatch.setattr(task_split, "task_strings_for_ids", forbidden)
    args, tasks = runner.resolve_tasks(
        runner.Args(
            exp_name="stage",
            mode="sambar",
            finetune_tasks=("new-task",),
            finetune_repo_id="local/current",
            finetune_root="/current",
        )
    )
    seen = []
    monkeypatch.setattr(curriculum, "_make_lerobot_dataset", lambda *a: seen.append(a) or "dataset")
    assert curriculum.build_raw_dataset(args, "config", 10) == "dataset"
    assert seen == [("local/current", "/current", 10, "config", ("new-task",))]
    assert len(tasks) == 1


def test_baseline_default_penalties_use_all_parameters():
    import dataclasses

    from baselines._impl.ewc import WL2Args
    from baselines._impl.l2 import L2Args

    for cls in (L2Args, WL2Args):
        assert {"qk_only", "penalty_group"}.isdisjoint(f.name for f in dataclasses.fields(cls))


def test_evaluation_aggregates_by_language():
    from scripts.evaluate import summarize

    values = summarize(
        [
            dict(task="old_task", successes=2, episodes=5),
            dict(task="new task", successes=4, episodes=5),
            dict(task="unselected", successes=5, episodes=5),
        ],
        ["old task"],
        ["new_task"],
    )
    assert values["pretrain"]["success_rate"] == 0.4
    assert values["new"]["success_rate"] == 0.8
    assert values["all"]["success_rate"] == 0.6
