"""Local LeRobot-format data integration; no public data or tokenizer download."""

import dataclasses
import io

import datasets
import jax
import numpy as np
import sentencepiece

from openpi.models import tokenizer
from openpi.training import config
from openpi.training import curriculum
from openpi.training import runner
from openpi.training import sharding


def test_local_lerobot_task_selection_and_model_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets.config, "HF_DATASETS_CACHE", tmp_path / "datasets-cache")
    # Use the real tokenizer wrapper with a tiny locally generated vocabulary.
    # This checks the interface, not compatibility with pretrained token IDs.
    vocabulary = io.BytesIO()
    sentencepiece.SentencePieceTrainer.train(
        sentence_iterator=iter(["old task", "new task", "Task State Action 0 1 2 3"]),
        model_writer=vocabulary,
        model_type="char",
        vocab_size=32,
        hard_vocab_limit=False,
    )
    tokenizer_file = tmp_path / "tokenizer.model"
    tokenizer_file.write_bytes(vocabulary.getvalue())
    download = tokenizer.download.maybe_download
    monkeypatch.setattr(
        tokenizer.download,
        "maybe_download",
        lambda path, **kw: tokenizer_file if str(path).endswith("paligemma_tokenizer.model") else download(path, **kw),
    )

    root = tmp_path / "dataset"
    features = {
        "image": {"dtype": "image", "shape": (16, 16, 3), "names": ["height", "width", "channel"]},
        "wrist_image": {"dtype": "image", "shape": (16, 16, 3), "names": ["height", "width", "channel"]},
        "state": {"dtype": "float32", "shape": (8,), "names": None},
        "actions": {"dtype": "float32", "shape": (7,), "names": None},
    }
    dataset = curriculum.lerobot_dataset.LeRobotDataset.create(
        "local/cpu",
        fps=10,
        root=root,
        features=features,
        use_videos=False,
    )
    for task in ("old task", "new task"):
        for frame in range(4):
            dataset.add_frame(
                {
                    "image": np.full((16, 16, 3), frame, dtype=np.uint8),
                    "wrist_image": np.full((16, 16, 3), frame, dtype=np.uint8),
                    "state": np.full(8, frame, dtype=np.float32),
                    "actions": np.full(7, frame, dtype=np.float32),
                    "task": task,
                }
            )
        dataset.save_episode()

    cfg = config.get_config("sambar_narrow")
    cfg = dataclasses.replace(
        cfg,
        model=dataclasses.replace(cfg.model, action_horizon=2, action_dim=8, max_token_len=12),
        batch_size=2,
        num_workers=0,
    )
    args = runner.Args(
        exp_name="local",
        mode="sambar",
        finetune_tasks=("new task",),
        finetune_repo_id="local/cpu",
        finetune_root=str(root),
    )
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    raw = curriculum.build_raw_dataset(args, data_config, cfg.model.action_horizon)
    assert len(raw) == 4
    assert all(raw[i]["prompt"] == "new task" for i in range(4))
    np.testing.assert_array_equal(raw[3]["actions"], np.full((2, 7), 3, dtype=np.float32))

    mesh = sharding.make_mesh(1)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    loader = curriculum.create_mixed_data_loader(args, cfg, data_sharding)
    observation, actions = next(iter(loader))
    assert observation.state.shape == (2, 8)
    assert observation.images["base_0_rgb"].shape == (2, 224, 224, 3)
    assert observation.tokenized_prompt.shape == (2, 12)
    assert actions.shape == (2, 2, 8)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves((observation, actions)))
    np.testing.assert_array_equal(actions[..., -1], 0)
