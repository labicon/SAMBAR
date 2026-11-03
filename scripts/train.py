"""Unified SAMBAR CLI: pretrain, finetune, three-stage sequence, or export importance."""
import tyro
from openpi.training.runner import Args, main

if __name__ == "__main__":
    main(tyro.cli(Args))
