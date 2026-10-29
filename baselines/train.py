"""Baseline CLI; dispatches to the retained original numerical implementations."""

import argparse
import importlib
import sys

import tyro


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--method", choices=("sft", "l2", "ewc", "lora", "cotrain"), required=True)
    parser.add_argument("--setting", choices=("narrow", "broad"), default="narrow")
    argv = sys.argv[1:] if argv is None else argv
    if not argv or ("--help" in argv and "--method" not in argv):
        parser.print_help()
        print(
            "Use --method METHOD --help for its original training options. RETAIN: python -m baselines.merge_checkpoints_linear"
        )
        return
    selected, rest = parser.parse_known_args(argv)
    if selected.method == "cotrain" and selected.setting == "broad":
        parser.error("The original co-training implementation supports the narrow/sequential setting only")
    name = "l2" if selected.method == "sft" else selected.method
    name += "_broad" if selected.setting == "broad" else ""
    module = importlib.import_module("baselines._impl." + name)
    options = {"exp_name": tyro.MISSING}
    if selected.method in ("lora", "cotrain"):
        options["pretrained_path"] = tyro.MISSING
    else:
        options["mode"] = {"sft": "bc_finetune", "l2": "l2_finetune", "ewc": "weighted_l2"}[selected.method]
    if selected.method == "cotrain":
        options["finetune_task_ids"] = tyro.MISSING
    default = module.CliArgs(**options)
    args = tyro.cli(module.CliArgs, default=default, args=rest)
    module.init_logging() if hasattr(module, "init_logging") else None
    module.main(args)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    main()
