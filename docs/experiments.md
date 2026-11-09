# Experiments

All SAMBAR settings use `scripts/train.py`. Use `--help` to list the training options. GPU examples include `--extra cuda`.

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

For a new pretraining run, the trainer saves the final checkpoint and its LIBERO importance under `checkpoints/sambar_SETTING/NAME/STEP/`, where STEP is `num_train_steps - 1`.

The examples below use the [released LIBERO checkpoints](https://huggingface.co/Berkeley-ICON-Lab/SAMBAR-checkpoints) downloaded under `./checkpoints`. Before finetuning, estimate importance once for the checkpoint you will use.

```bash
uv run --extra cuda python scripts/train.py --mode importance --setting narrow --exp-name export32 \
  --pretrained-path checkpoints/libero-32/15000

uv run --extra cuda python scripts/train.py --mode importance --setting broad --exp-name export117 \
  --pretrained-path checkpoints/libero-117/9999
```

The estimate covers the matching 32 or 117 LIBERO pretraining tasks. By default it is saved in the checkpoint’s `importance/` folder. Use `--importance-path` to choose a different output directory.

## Reported SAMBAR finetuning settings

| Regime | Task | Batch size | Learning rate | Iterations per task |
| --- | --- | ---: | ---: | ---: |
| Broad (117 tasks), independent | pot-on-stove, items-into-basket | 64 | 2.5 × 10⁻⁵ | 500 |
| Broad (117 tasks), independent | mugs-on-plates | 64 | 2.5 × 10⁻⁵ | 1,000 |
| Narrow (32 tasks), independent | Each held-out task | 16 | 1 × 10⁻⁵ | 20,000 |
| Narrow (32 tasks), sequential | Each of the three successive tasks | 16 | 1 × 10⁻⁵ | 10,000 |

SAMBAR used ρ = 0.02 and estimated the current task’s diagonal Fisher from 50 minibatches at the start of each stage. For SFT, L2, EWC, RETAIN, LoRA, co-training and GRPO, see the [baseline finetuning settings](../baselines/README.md#baseline-finetuning-settings).

## Independent finetuning

```bash
uv run --extra cuda python scripts/train.py --mode sambar --setting narrow --exp-name task3 \
  --pretrained-path checkpoints/libero-32/15000 --finetune-task-ids 3 \
  --batch-size 16 --lr 1e-5 --num-train-steps 20000

uv run --extra cuda python scripts/train.py --mode sambar --setting broad --exp-name moka \
  --pretrained-path checkpoints/libero-117/9999 \
  --finetune-tasks 'turn on the stove and put the moka pot on it' \
  --batch-size 64 --lr 2.5e-5 --num-train-steps 500
```

Repeat with the same pretraining checkpoint for independent tasks. Passing multiple task IDs/names trains their combined data in one stage. In `sambar` mode, `--importance-path` selects the incoming history artifact; otherwise it is loaded from the incoming checkpoint's `importance/` folder.

## Three-stage sequential finetuning

```bash
uv run --extra cuda python scripts/train.py --mode sequential --setting narrow --exp-name sequence \
  --pretrained-path checkpoints/libero-32/15000 --sequence 3 8 11 \
  --batch-size 16 --lr 1e-5 --num-train-steps 10000

# Optional sequence starting from the broad checkpoint with the same per-stage budget.
uv run --extra cuda python scripts/train.py --mode sequential --setting broad --exp-name broad_sequence \
  --pretrained-path checkpoints/libero-117/9999 \
  --batch-size 16 --lr 1e-5 --num-train-steps 10000
```

The narrow sequence reproduces the source task order. The broad command explicitly applies sequential mode to the broad new-task list; it does not imply the original broad results used sequential training. Outputs are `sequence/stage1`, `sequence/stage2`, `sequence/stage3`. History counts become 33/34/35 or 118/119/120. Each completed stage estimates only its current task for the history update.

Use `--finetune-tasks 'task one' 'task two' 'task three'` to specify a broad sequence. Use `--resume` with the same run arguments to resume a saved run; method and optimizer state are recovered, while the inherited data iterator restarts. `--overwrite` explicitly replaces a run directory.

For a task-only local dataset without historical demonstrations:

```bash
uv run --extra cuda python scripts/train.py --mode sambar --setting narrow --exp-name local_current \
  --pretrained-path /path/to/previous/checkpoint \
  --finetune-repo-id local/current --finetune-root /path/to/current_lerobot_dataset \
  --finetune-tasks 'exact language string from the original LIBERO task' \
  --batch-size 16 --lr 1e-5 --num-train-steps 10000
```

This dataset must contain the same features/action conventions and valid LeRobot metadata as the source datasets. For separate local datasets per sequential stage, invoke each stage explicitly and chain its checkpoint.

Each finetuning run records its arguments and task identities in `run.json`.
