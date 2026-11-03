"""Compute LIBERO normalization statistics using the shared SAMBAR data pipeline."""

import tyro

from openpi.training.runner import Args
from openpi.training.runner import run_stage


if __name__ == "__main__":
    args = tyro.cli(Args, default=Args(exp_name="norm_stats", mode="norm_stats"))
    if args.mode != "norm_stats":
        raise SystemExit("Use scripts/train.py for training and importance export.")
    run_stage(args)
