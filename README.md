# SAMBAR

**Selective Anchoring via Method of Multipliers for Balanced Knowledge Acquisition and Retention in Vision-Language-Action Models**

[Project page](https://iconlab.negarmehr.com/SAMBAR/) · [Paper](docs/SAMBAR_arxiv_final.pdf) · [Experiment guide](docs/experiments.md)

Research code for continual finetuning of π0.5 on LIBERO, built on [openpi](https://github.com/Physical-Intelligence/openpi). It includes narrow (32-task) and broad (117-task) pretraining, three-stage sequential finetuning, and the selected comparison baselines.

## Repository structure

The layout follows the openpi-based [RETAIN code repository](https://github.com/yajatyadav/RETAIN_code), with a dedicated folder for baseline commands:

```text
SAMBAR/
├── scripts/                       # Training, normalization and policy serving
│   ├── train.py
│   ├── compute_norm_stats.py
│   └── serve_policy.py
├── src/openpi/                     # Model and reusable research implementation
│   ├── models/
│   ├── policies/
│   ├── serving/
│   ├── shared/
│   └── training/
│       ├── config.py              # Model, optimizer and dataset configurations
│       ├── runner.py              # Shared experiment/stage orchestration
│       ├── sambar.py              # Selected weighted_taco_ar.py calculations
│       ├── sambar_math.py
│       ├── importance.py
│       └── baselines/             # Reusable SFT, L2, EWC, LoRA and replay code
├── baselines/                     # Baseline entry points and sequence launchers
├── examples/libero/                # Simulator setup, rollouts and evaluation
├── assets/sambar/                  # Narrow/broad normalization statistics
├── packages/openpi-client/         # Shared policy client
├── docs/                          # Setup, experiments, method notes and website
└── tests/                         # CPU regression tests
```


## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run from the repository root:

```bash
uv sync --frozen --extra cuda
```

See [setup](docs/setup.md) for the pinned environment, base model and dataset locations. LIBERO uses a separate environment installed from pinned upstream; see the [LIBERO README](examples/libero/README.md).

## Training

The unified trainer covers both pretraining curricula and sequential SAMBAR:

```bash
# Narrow pretraining: 32 tasks, with importance saved after training.
uv run --extra cuda python scripts/train.py \
  --mode pretrain --setting narrow --exp-name pretrain32

# Broad pretraining: 117 tasks.
uv run --extra cuda python scripts/train.py \
  --mode pretrain --setting broad --exp-name pretrain117

# Three successive narrow tasks, carrying the model and saved importance forward.
uv run --extra cuda python scripts/train.py \
  --mode sequential --setting narrow --exp-name sequence \
  --pretrained-path checkpoints/sambar_narrow/pretrain32/29999 \
  --sequence 3 8 11
```

The checked-in normalization statistics can be recomputed through the same data pipeline:

```bash
uv run --extra cuda python scripts/compute_norm_stats.py --setting narrow
```

See [experiments](docs/experiments.md) for independent finetuning, broad sequences, importance export and inherited hyperparameters. SAMBAR trains and regularizes all parameters.

## Baselines

[SFT, L2, EWC, RETAIN, LoRA, co-training and GRPO](baselines/README.md) have their entry points and sequence launchers in `baselines/`. Their shared implementations live in `src/openpi/training/baselines/`.

```bash
uv run --extra cuda python -m baselines.train --method sft --setting narrow \
  --exp-name sft_task3 --pretrained-path checkpoints/sambar_narrow/pretrain32/29999/params \
  --finetune-task-ids 3
```

## LIBERO evaluation

```bash
bash examples/libero/setup.sh
uv run --extra cuda python examples/libero/evaluate.py \
  --checkpoint checkpoints/sambar_narrow/sequence/stage3/29999 \
  --setting narrow --task-ids 3 8 11 --output outputs/sequence-stage3
```

The evaluation driver starts `scripts/serve_policy.py` and runs the simulator in its separate environment. For standalone serving and rollout commands, see [examples/libero/README.md](examples/libero/README.md).

## Method and importance

SAMBAR preserves the calculations selected from `weighted_taco_ar.py`: square the gradient of the **mean minibatch loss**, then average across batches. The weighted anchor keeps epsilon in both numerator and denominator.

Pretraining saves LIBERO importance alongside the final checkpoint. Later SAMBAR stages use that artifact and update its task-count-weighted running average after the current task. **Original π0.5 pretraining importance is unavailable; the saved estimate covers LIBERO pretraining only.** See [method and provenance](docs/method.md) for the exact formulas and approved history handling.

This release includes the LIBERO research workflows. Hardware integrations, plain TACO and parameter-subset experiments are excluded. [CPU validation](docs/validation.md) covers the calculations, three-stage training/checkpoint workflow, reduced-size π0.5 model and local LeRobot data pipeline. Full π0.5 GPU training and simulator rollouts remain unverified.
