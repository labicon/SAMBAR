# Setup

Run commands from the repository root. Training targets Linux with NVIDIA GPUs and Python 3.11/3.12; all-parameter π0.5 plus optimizer and SAMBAR state requires substantial GPU memory. Use `--fsdp-devices` to shard across the selected devices. No hardware robot connection is needed.

## Training environment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then:

```bash
uv sync --frozen --extra cuda
uv run --extra cuda python scripts/train.py --help
```

Use `--extra cuda` on GPU training invocations. The lockfile preserves the source JAX 0.5.3, Flax 0.10.2, Optax 0.2.4 and Orbax 0.11.13 stack. LeRobot is pinned to source revision `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5`.

Base model parameters load from `gs://openpi-assets/checkpoints/pi05_base/params` through the original downloader. Keep sufficient local disk space for the model, data, optimizer checkpoints and importance arrays. `OPENPI_DATA_HOME` can redirect the model cache; `HF_LEROBOT_HOME` can redirect the dataset cache.

The data loaders use these existing LeRobot datasets:

| Dataset | Role |
| --- | --- |
| `physical-intelligence/libero` | 40 tasks; narrow curriculum and new-task data |
| `jesbu1/libero_90_lerobot` | The additional 90 broad-pretraining tasks |

They download through the pinned LeRobot implementation when not cached. `--libero90-root ROOT` specifies a separate cache parent containing `ROOT/jesbu1/libero_90_lerobot`. The two repositories keep their own task-index tables. Task language, rather than cross-repository numeric IDs, identifies tasks.

The checked-in normalization statistics live under `assets/sambar/{narrow,broad}`. To recompute them with the same preprocessing/sampler:

```bash
uv run --extra cuda python scripts/compute_norm_stats.py --setting narrow
uv run --extra cuda python scripts/compute_norm_stats.py --setting broad
```

## LIBERO setup and evaluation

See [the LIBERO README](../examples/libero/README.md) for simulator installation, dataset/asset setup and evaluation. All simulator-specific commands live beside the example in `examples/libero/`.
