"""Evaluate narrow/broad checkpoints across seeds using the retained LIBERO rollout loop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def summarize(results, pretrain_tasks, new_tasks):
    """Aggregate by task language, never by simulator task indices."""

    def normalize(text):
        return " ".join(text.lower().replace("_", " ").split())

    pre, new = {normalize(x) for x in pretrain_tasks}, {normalize(x) for x in new_tasks}
    groups = {"pretrain": [0, 0], "new": [0, 0], "all": [0, 0]}
    for row in results:
        task = normalize(row["task"])
        if task in pre:
            group = "pretrain"
        elif task in new:
            group = "new"
        else:
            continue
        for bucket in (group, "all"):
            groups[bucket][0] += row["successes"]
            groups[bucket][1] += row["episodes"]
    return {
        key: {"successes": x[0], "episodes": x[1], "success_rate": x[0] / x[1] if x[1] else None}
        for key, x in groups.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--setting", choices=("narrow", "broad"), default="narrow")
    parser.add_argument("--config", help="Use libero_lora / libero_broad_lora / libero_grpo_lora for adapters")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--trials", type=int, help="Defaults: narrow=10, broad=5, matching the website settings")
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--task-ids", type=int, nargs="+", default=[3, 8, 11])
    parser.add_argument(
        "--tasks", nargs="+", help="Explicit new task descriptions, e.g. one task for single-task evaluation"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--sim-python", default=str(ROOT / "examples/libero/.venv/bin/python"))
    parser.add_argument("--server-timeout", type=float, default=900)
    args = parser.parse_args()
    if args.trials is not None and args.trials < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Trials must be positive and seeds must be unique")
    # Import training/data dependencies only after argparse (help remains lightweight).
    from openpi.training.curriculum import DEFAULT_FINETUNE_TASKS
    from openpi.training.curriculum import LIBERO40_REPO_ID
    from openpi.training.curriculum import pretrain_task_strings
    from openpi.training.task_split import sample_pretrain_tasks
    from openpi.training.task_split import task_strings_for_ids

    if args.setting == "narrow":
        pre_ids, _ = sample_pretrain_tasks(8, args.split_seed)
        pretrain = list(task_strings_for_ids(LIBERO40_REPO_ID, pre_ids))
        learned = args.tasks or list(task_strings_for_ids(LIBERO40_REPO_ID, args.task_ids))
    else:
        pretrain = list(pretrain_task_strings())
        learned = args.tasks or list(DEFAULT_FINETUNE_TASKS)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", args.port))
        except OSError:
            parser.error("Evaluation port is already in use; choose a separate port for this checkpoint")
    config = args.config or "sambar_" + args.setting
    command = [
        sys.executable,
        str(ROOT / "scripts/serve_policy.py"),
        "--checkpoint",
        args.checkpoint,
        "--config",
        config,
        "--port",
        str(args.port),
    ]
    with (output / "server.log").open("w") as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        try:
            deadline = time.monotonic() + args.server_timeout
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"Policy server exited; see {output / 'server.log'}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/healthz", timeout=2) as response:
                        if response.status == 200:
                            break
                except (OSError, urllib.error.URLError):
                    pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("Policy server did not become ready")
                time.sleep(1)
            per_seed = []
            suites = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
            if args.setting == "broad":
                suites.append("libero_90")
            keep = output / "task_descriptions.json"
            keep.write_text(json.dumps(pretrain + learned))
            for seed in args.seeds:
                rows = []
                seed_pretrain = list(pretrain)
                for suite in suites:
                    folder = output / f"seed{seed}" / suite
                    folder.mkdir(parents=True, exist_ok=True)
                    cmd = [
                        args.sim_python,
                        str(ROOT / "examples/libero/main.py"),
                        "--args.host",
                        "127.0.0.1",
                        "--args.port",
                        str(args.port),
                        "--args.task-suite-name",
                        suite,
                        "--args.seed",
                        str(seed),
                        "--args.num-trials-per-task",
                        str(args.trials or (10 if args.setting == "narrow" else 5)),
                        "--args.video-out-path",
                        str(folder),
                    ]
                    if suite != "libero_90":
                        cmd += ["--args.task-descriptions-file", str(keep)]
                    subprocess.run(cmd, check=True, cwd=ROOT)
                    result = json.loads((folder / f"results_{suite}.json").read_text())
                    for language, row in result["per_task"].items():
                        rows.append({"task": language, "successes": row["successes"], "episodes": row["episodes"]})
                        if suite == "libero_90":
                            seed_pretrain.append(language)
                stats = summarize(rows, seed_pretrain, learned)
                per_seed.append({"seed": seed, "groups": stats, "per_task": rows})
            # Keep per-seed data so confidence intervals can be recomputed transparently.
            (output / "summary.json").write_text(
                json.dumps(
                    {"checkpoint": args.checkpoint, "config": config, "setting": args.setting, "seeds": per_seed},
                    indent=2,
                )
                + "\n"
            )
            print(json.dumps(per_seed, indent=2))
        finally:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()


if __name__ == "__main__":
    main()
