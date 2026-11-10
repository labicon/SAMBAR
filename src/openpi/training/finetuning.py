"""LIBERO finetuning budgets used by the SAMBAR and supervised baseline trainers."""

import dataclasses
import json
from pathlib import Path

BROAD_TASK_STEPS = {
    "turn on the stove and put the moka pot on it": 500,
    "put the white mug on the left plate and put the yellow and white mug on the right plate": 1000,
    "put both the alphabet soup and the cream cheese box in the basket": 500,
}


def _resume_budget(base, args, sambar):
    """Recover the schedule used by an existing run, including older trainer defaults."""
    directory = Path(args.checkpoint_base_dir or base.checkpoint_base_dir) / base.name / args.exp_name
    if not directory.is_dir():
        return None
    if sambar:
        path = directory / "run.json"
    elif (directory / "hparams.json").is_file():
        path = directory / "hparams.json"
    else:
        path = directory.parent / (directory.name + ".plan.json")
    if not path.is_file():
        return None
    record = json.loads(path.read_text())
    saved = record.get("resolved", record)
    previous = record.get("args", record.get("cli_args", {}))
    retired_options = {
        "sambar": {"anchor_path": None, "fisher_refresh": 0, "warmup_steps": 0, "dual_period": 1},
        "l2": {"anchor_path": None},
        "wl2": {"anchor_path": None, "normalize_weights": False},
    }
    for group, defaults in retired_options.items():
        options = previous.get(group, {})
        for key, default in defaults.items():
            if options.get(key, default) != default:
                raise ValueError(f"Cannot resume a run that used {group}.{key}={options[key]}")
    if previous.get("replay_pretrain") is False:
        raise ValueError("Cannot resume co-training with pretraining replay disabled")
    # Older SAMBAR manifests recorded explicit LR overrides in args and inherited the base schedule otherwise.
    old_lr = previous.get("lr")
    return dataclasses.replace(
        base,
        num_train_steps=record["steps"] if sambar else saved["num_train_steps"],
        batch_size=saved["batch_size"],
        lr_schedule=dataclasses.replace(
            base.lr_schedule,
            peak_lr=saved.get("lr_peak", base.lr_schedule.peak_lr if old_lr is None else old_lr),
            decay_lr=saved.get("lr_decay", base.lr_schedule.decay_lr if old_lr is None else old_lr),
        ),
    )


def finetuning_config(base, args, setting, *, sambar=False, sequential=False, grpo_warmup=False):
    """Select the training budget before the caller applies explicit CLI overrides."""
    if args.resume:
        saved = _resume_budget(base, args, sambar)
        if saved is not None:
            return saved
    if grpo_warmup:
        default_steps, batch_size, lr = 100, 16, 5e-5
    elif sequential or setting == "narrow":
        default_steps = 20_000 if sambar and not sequential else 10_000
        batch_size, lr = 16, 1e-5
    else:
        batch_size, lr = 64, 2.5e-5
        default_steps = args.num_train_steps
        if default_steps is None:
            tasks = args.finetune_tasks or ()
            budgets = {BROAD_TASK_STEPS.get(task) for task in tasks}
            if len(budgets) != 1 or None in budgets:
                raise ValueError(
                    "Choose a broad finetuning task or pass --num-train-steps. "
                    "Pot-on-stove and items-into-basket use 500 steps, and mugs-on-plates uses 1000."
                )
            default_steps = budgets.pop()

    return dataclasses.replace(
        base,
        num_train_steps=default_steps,
        batch_size=batch_size,
        lr_schedule=dataclasses.replace(base.lr_schedule, peak_lr=lr, decay_lr=lr),
    )
