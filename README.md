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
└── docs/                          # Setup, experiments, method notes and website
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

## Acknowledgements

This work is supported by the National Science Foundation, under grants ECCS-2438314 CAREER Award, CNS-2529645. This work was also supported in part by an Office of Naval Research (ONR) Young Investigator Program (YIP) award. This research was made possible by GPU resources provided via the NVIDIA Academic Grant.

## Disclaimer

The code in this repository was refactored and reorganized using OpenAI Codex. 

## Citation

If you use SAMBAR in your research, please cite:

```bibtex
@article{shrivastava2026sambar,
  title   = {SAMBAR: Selective Anchoring via Method of Multipliers for Balanced
             Knowledge Acquisition and Retention in Vision-Language-Action Models},
  author  = {Shrivastava, Aayushi and Zhou, Xunlan and Zhao, Hongrui and
             Chen, Ziyu and Mehr, Negar},
  journal = {arXiv preprint arXiv:2609.32108},
  year    = {2026}
}
```
