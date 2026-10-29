"""Portable importance artifacts; no demonstration data is needed to read them."""

import json
import pathlib
import uuid

import jax
import numpy as np
import orbax.checkpoint as ocp


def read_metadata(directory):
    path = pathlib.Path(directory)
    with (path / "metadata.json").open() as stream:
        metadata = json.load(stream)
    if metadata.get("version") != 1:
        raise ValueError(f"Unsupported importance artifact: {path}")
    tasks = metadata["tasks"]
    if not tasks or len(set(tasks)) != len(tasks) or metadata["task_count"] != len(tasks):
        raise ValueError("Importance task identities/count are inconsistent")
    return metadata


def save(directory, omega, tasks, *, setting, fisher_batches):
    """Write arrays and provenance atomically, without overwriting an existing artifact."""
    path = pathlib.Path(directory).resolve()
    tasks = list(tasks)
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("Importance requires unique, nonempty task identities")
    if path.exists():
        raise FileExistsError(f"Importance already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}-{uuid.uuid4().hex}.tmp")
    staging.mkdir()
    with ocp.PyTreeCheckpointer() as writer:
        writer.save(staging / "arrays", ocp.args.PyTreeSave(omega))
    metadata = {
        "version": 1,
        "setting": setting,
        "task_count": len(tasks),
        "tasks": tasks,
        "estimator": "mean_over_batches(square(gradient(mean(flow_matching_loss))))",
        "fisher_batches": fisher_batches,
        "history_update": "task_count_weighted_mean_of_saved_stage_estimates",
        "base_pi05_importance_available": False,
    }
    (staging / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    staging.rename(path)


def load(directory, template=None):
    """Restore onto the current device layout, or to CPU NumPy arrays without a template."""
    path = pathlib.Path(directory).resolve()
    metadata = read_metadata(path)
    with ocp.PyTreeCheckpointer() as reader:
        item = reader.metadata(path / "arrays").tree
        if template is None:
            restore_args = jax.tree.map(lambda _: ocp.ArrayRestoreArgs(restore_type=np.ndarray), item)
        else:
            if jax.tree.structure(item) != jax.tree.structure(template):
                raise ValueError("Importance does not match the model parameter tree")
            for expected, actual in zip(jax.tree.leaves(template), jax.tree.leaves(item), strict=True):
                if tuple(expected.shape) != tuple(actual.shape):
                    raise ValueError("Importance does not match the model parameter shapes")
            restore_args = jax.tree.map(
                lambda x: ocp.ArrayRestoreArgs(restore_type=jax.Array, sharding=x.sharding), template
            )
        omega = reader.restore(path / "arrays", ocp.args.PyTreeRestore(item=item, restore_args=restore_args))
    return omega, metadata


def merge_task_ids(old, new):
    """Reject accidental repeated-task counting before updating a saved average."""
    old, new = list(old), list(new)
    if len(set(old)) != len(old) or len(set(new)) != len(new) or set(old) & set(new):
        raise ValueError("Historical and new task identities must be unique and disjoint")
    return old + new
