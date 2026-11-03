# Validation and remaining limits

The release has CPU integration coverage. Full-size π0.5 training and LIBERO rollouts have not been run, so this is not a claim that the published experiments have been reproduced.

## CPU checks

The checks run locally with Python 3.12.4 and a clean environment installed from `uv.lock` using `uv sync --frozen`, without the CUDA extra. They do not download model weights or demonstrations.

Recorded results: **18 tests passed** in the complete CPU suite; **all four workflow tests also passed** in a separate process configured with two virtual CPU devices. The source-equivalence comparison passed in the same locked environment. Python syntax checks passed for 82 files, shell syntax checks passed for all four launchers, and Ruff passed for the new tests and changed checkpoint module.

| Check | What it exercises | What is substituted |
| --- | --- | --- |
| Numerical regression | Mean-minibatch-gradient-square estimator, selected epsilon anchor, task-count history and artifact validation | Small analytical NNX model |
| Training workflow | Real model initialization, AdamW, Torch batching, parameter loading, pretraining, three-stage SAMBAR, Fisher refresh, importance export and checkpoints | Small MLP, synthetic observations and task metadata for the 32/117-task curricula |
| Resume | Interrupted-stage method/optimizer restore, completed-sequence resume and one-step checkpoint resume | Same small model and synthetic inputs; data iterator still restarts |
| Two-device workflow | The four workflow tests with real JAX FSDP partitioning, sharded importance, checkpoint writes and restores | Two virtual CPU devices and a larger MLP matrix that reaches the 4 MiB sharding threshold |
| π0.5 model integration | Production vision encoder, language/action attention, flow-matching loss, importance gradients, optimizer updates and action sampling | Random initialization, dummy Gemma dimensions, small SigLIP variant and 128-token vocabulary |
| LeRobot data integration | On-disk Parquet/image loading, task selection, action-chunk padding, normalization, tokenization interface and model batch shapes | Two short locally generated episodes and a locally generated SentencePiece vocabulary |

The workflow tests verify history counts of 33/34/35 and 118/119/120 and assert that SAMBAR requests only current-task datasets after pretraining. The numerical tests independently check the running-average values.

The one-step resume regression exposed an inherited checkpoint bug: a completed checkpoint numbered `0` was classified as absent. The check now distinguishes an empty checkpoint directory from a valid step-zero checkpoint. This changes checkpoint bookkeeping only; the selected estimator, anchor and training update are unchanged.

## Reproduce without a GPU

From the repository root:

```bash
uv sync --frozen
JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 uv run --frozen pytest -q

# Separate process: JAX reads device configuration at startup.
XLA_FLAGS=--xla_force_host_platform_device_count=2 JAX_PLATFORMS=cpu \
  HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  uv run --frozen pytest -q tests/test_workflow_cpu.py
```

Dependency installation requires internet access. The tests themselves use local fixtures, write checkpoints to temporary directories and disable W&B reporting. The two-device command simulates device partitioning on CPU; it does not test CUDA, NCCL or inter-GPU communication.

During refactoring, the SAMBAR functions were also compared directly against functions extracted from `train_pi05_weighted_taco_ar.py` at source commit `8245c09fe15ed542a2aa4740076d8ca0b417bc9b`. Fisher estimation, weighted anchor, optimizer and dual updates produced exactly equal CPU arrays on a small NNX model. The L2/EWC all-parameter steps also matched. That comparison was repeated in the clean locked environment.

## Still unverified

- Loading the full released π0.5 checkpoint and training on the actual public demonstrations.
- GPU memory requirements, CUDA compatibility, multi-GPU throughput and long-run numerical stability.
- Full baseline training, GRPO simulator interaction, policy-server-to-LIBERO rollouts and benchmark success rates.

The reduced model and local dataset checks validate their respective interfaces separately; they are not a full pretrained-model-to-simulator experiment. CLI help checks and source-equivalence checks likewise do not establish successful full training. A future full-model smoke test is still needed before claiming GPU runtime validation or reproduced results.
