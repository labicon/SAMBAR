# Experiments

All SAMBAR settings use `scripts/train.py`. Flags shown below are kebab-case; `--help` lists the inherited numeric settings. GPU examples include `--extra cuda`.

| Setting | Pretraining | Finetuning |
| --- | --- | --- |
| Narrow | Seed-7 sample of 8 tasks from each of LIBERO long, goal, object, spatial: 32 total | Held-out tasks; the source sequential curriculum is 3 → 8 → 11 |
| Broad | 27 goal/object/spatial tasks plus LIBERO-90: 117 total | The three held-out long-horizon tasks below, independently or in an explicit sequence |
| Sequential | A completed pretraining checkpoint and importance artifact | Three stages; each starts from the previous stage's parameters and saved importance |

Numeric IDs refer to `physical-intelligence/libero` metadata, not LIBERO simulator task IDs. Narrow default `3 → 8 → 11` is pot/stove → bowl/drawer → bottle/rack. The eight seed-7 held-out tasks remain available through explicit `--finetune-task-ids`.

Broad excludes these three tasks from the 30 goal/object/spatial tasks:

- `put the bowl on the plate`
- `pick up the alphabet soup and place it in the basket`
- `pick up the black bowl from table center and place it on the plate`

Its default new-task list, in order, is:

1. `turn on the stove and put the moka pot on it`
2. `put the white mug on the left plate and put the yellow and white mug on the right plate`
3. `put both the alphabet soup and the cream cheese box in the basket`

## Pretraining and initial importance

```bash
uv run --extra cuda python scripts/train.py --mode pretrain --setting narrow --exp-name pretrain32
uv run --extra cuda python scripts/train.py --mode pretrain --setting broad --exp-name pretrain117
```

The final checkpoint and importance folder are `checkpoints/sambar_SETTING/NAME/STEP/`, where STEP is `num_train_steps - 1`. At the inherited 30,000-step default, STEP is 29999. The initial importance covers only the 32/117 LIBERO tasks, not unknown original π0.5 pretraining tasks.

For an existing LIBERO-pretrained checkpoint, estimate/save importance without retraining:

```bash
uv run --extra cuda python scripts/train.py --mode importance --setting narrow --exp-name export32 \
  --pretrained-path /path/to/pretrain/checkpoint
```

Use `--setting broad` for a 117-task checkpoint. `--importance-path` optionally chooses the output artifact directory in this mode. Do not overwrite an artifact with importance from a different model or curriculum.

## Independent finetuning

```bash
uv run --extra cuda python scripts/train.py --mode sambar --setting narrow --exp-name task3 \
  --pretrained-path checkpoints/sambar_narrow/pretrain32/29999 --finetune-task-ids 3

uv run --extra cuda python scripts/train.py --mode sambar --setting broad --exp-name moka \
  --pretrained-path checkpoints/sambar_broad/pretrain117/29999 \
  --finetune-tasks 'turn on the stove and put the moka pot on it'
```

Repeat with the same pretraining checkpoint for independent tasks. Passing multiple task IDs/names trains their combined data in one stage. In `sambar` mode, `--importance-path` selects the incoming history artifact; otherwise it is loaded from the incoming checkpoint's `importance/` folder.

## Three-stage sequential finetuning

```bash
uv run --extra cuda python scripts/train.py --mode sequential --setting narrow --exp-name sequence \
  --pretrained-path checkpoints/sambar_narrow/pretrain32/29999 --sequence 3 8 11

uv run --extra cuda python scripts/train.py --mode sequential --setting broad --exp-name broad_sequence \
  --pretrained-path checkpoints/sambar_broad/pretrain117/29999
```

The narrow sequence reproduces the source task order. The broad command explicitly applies sequential mode to the broad new-task list; it does not imply the original broad results used sequential training. Outputs are `sequence/stage1`, `sequence/stage2`, `sequence/stage3`. History counts become 33/34/35 or 118/119/120. Each completed stage estimates only its current task for the history update.

Use `--finetune-tasks 'task one' 'task two' 'task three'` to specify a broad sequence. Use `--resume` with the same run arguments to resume a saved run; method and optimizer state are recovered, while the inherited data iterator restarts. `--overwrite` explicitly replaces a run directory.

For a task-only local dataset without historical demonstrations:

```bash
uv run --extra cuda python scripts/train.py --mode sambar --setting narrow --exp-name local_current \
  --pretrained-path /path/to/previous/checkpoint \
  --finetune-repo-id local/current --finetune-root /path/to/current_lerobot_dataset \
  --finetune-tasks 'exact language string from the original LIBERO task'
```

This dataset must contain the same features/action conventions and valid LeRobot metadata as the source datasets. For separate local datasets per sequential stage, invoke each stage explicitly and chain its checkpoint.

## Inherited defaults

| Parameter | Narrow | Broad |
| --- | --- | --- |
| Training steps | 30,000 | 30,000 |
| Global batch size | 16 | 16 |
| LR peak | 5e-5 | 2.5e-5 |
| LR floor | 5e-5 | 2.5e-6 |
| LR warmup / decay steps | 1,000 / 30,000 | 1,000 / 30,000 |
| Action horizon | 10 | 10 |
| EMA | Disabled | Disabled |
| rho / beta / epsilon | 0.02 / 1 / 1e-8 | 0.02 / 1 / 1e-8 |
| Fisher batches / refresh | 50 / 0 | 50 / 0 |
| Dual period / penalty warmup | 1 / 0 | 1 / 0 |

These are the source configuration defaults, not a claim that every historical result used them. `--lr` retains the original override semantics: set both peak and floor. Other overrides include `--num-train-steps`, `--batch-size`, `--fsdp-devices`, `--seed`, `--sambar.rho`, `--sambar.fisher-batches` and `--sambar.fisher-refresh`. Every run records arguments and task identities in `run.json`.
