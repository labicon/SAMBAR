# SAMBAR

Research code for importance-weighted continual finetuning of π0.5 on LIBERO. The project website remains in [`docs/`](docs/).

SAMBAR uses the calculation from `train_pi05_weighted_taco_ar.py`: the gradient of the **mean minibatch loss is squared**, then averaged across batches. The selected weighted anchor retains epsilon in both numerator and denominator. Training and regularization use all parameters.

- [Installation, datasets and LIBERO](research/SETUP.md)
- [32-task, 117-task and three-stage experiments](research/EXPERIMENTS.md)
- [SFT, L2, EWC, RETAIN, LoRA, co-training and GRPO baselines](baselines/README.md)
- [Method, importance artifacts and source provenance](research/METHOD.md)
- [File guide](research/FILES.md)

```bash
uv sync --frozen --extra cuda
uv run --extra cuda python scripts/train.py --mode pretrain --setting narrow --exp-name pretrain32
```

Pretraining writes a LIBERO importance artifact alongside its final checkpoint. Later SAMBAR stages read that artifact and update its task-count-weighted running average after learning the current task. They do not recompute historical importance from previous-task demonstrations. **The released base π0.5 checkpoint does not include importance for its original pretraining tasks; this code estimates LIBERO importance only.**

This release excludes hardware integrations, plain TACO trainers, subset experiments, and the requested obsolete queue/evaluation scripts. LIBERO is installed from a pinned upstream revision rather than copied into this repository.

CPU tests cover the estimator, anchor, importance serialization and checkpoint state. Full π0.5 GPU training and simulator rollouts must be validated on the target training system; this refactor does not establish reproduced benchmark scores.
