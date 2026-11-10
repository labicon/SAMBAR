# Baselines

`python -m baselines.train` selects the method and curriculum. The implementations under [`src/openpi/training/baselines/`](../src/openpi/training/baselines/) retain their source losses and update rules. LoRA trains the configured adapters. GPU examples include `--extra cuda`.

| Method | Entry point | Historical demonstrations |
| --- | --- | --- |
| SFT | `baselines.train --method sft` | Not used by its loss |
| L2 / L2-SP | `baselines.train --method l2` | Not used; fixed reference checkpoint |
| EWC / weighted L2 | `baselines.train --method ewc` | Source behavior retained: old-demo Fisher estimate |
| RETAIN | `baselines.RETAIN` | Not used in the merge |
| LoRA | `baselines.train --method lora` | Not used by its loss |
| Co-training | `baselines.train --method cotrain` | Intentionally replays old data |
| Simple Recipe Works (GRPO + LoRA) | `baselines.train_grpo` | Online simulator rollouts after a current-task LoRA warm-up |

## Baseline finetuning settings

The supervised trainers select these defaults from the regime and task. The learning rate is the configured peak. For RETAIN, the budget applies to SFT before the checkpoint merge.

| Pretraining regime | Finetuning task | Batch size | Learning rate | Iterations per task |
| --- | --- | ---: | ---: | ---: |
| Broad (117 tasks) | pot-on-stove, items-into-basket | 64 | 2.5 × 10⁻⁵ | 500 |
| Broad (117 tasks) | mugs-on-plates | 64 | 2.5 × 10⁻⁵ | 1,000 |
| Narrow (32 tasks), independent finetuning | Each held-out task | 16 | 1 × 10⁻⁵ | 10,000 |
| Narrow (32 tasks), three-stage sequential finetuning | Each successive task | 16 | 1 × 10⁻⁵ | 10,000 |

The broad runs follow the [RETAIN evaluation protocol](https://arxiv.org/abs/2512.08333). Available GPU memory limited the narrow and sequential runs to a batch size of 16. For the penalty-based baselines, λ<sub>EWC</sub> = λ<sub>L2</sub> = 0.02, matching SAMBAR’s ρ. EWC estimated its diagonal Fisher from 50 minibatches at the start of each stage.

## Narrow and broad

For SFT, L2, EWC or LoRA, replace METHOD below with `sft`, `l2`, `ewc` or `lora`:

```bash
uv run --extra cuda python -m baselines.train --method METHOD --setting narrow \
  --exp-name task3 --pretrained-path checkpoints/sambar_narrow/pretrain32/29999/params \
  --finetune-task-ids 3

uv run --extra cuda python -m baselines.train --method METHOD --setting broad \
  --exp-name moka --pretrained-path checkpoints/sambar_broad/pretrain117/29999/params \
  --finetune-tasks 'turn on the stove and put the moka pot on it'
```

For all three broad tasks in a combined stage, omit `--finetune-tasks` and supply `--num-train-steps`. The reported budgets are for independent runs from the same pretrained checkpoint. `--method METHOD --help` lists that implementation's full options. Full-parameter output configs are `sambar_narrow`/`sambar_broad`; LoRA outputs use `libero_lora`/`libero_broad_lora`.

L2 adds `lambda/2 * sum((theta-reference)**2)`, with `--l2.lam 0.02` by default. EWC adds `lambda/2 * sum(importance*(theta-reference)**2)`, with `--wl2.lam 0.02`, 50 Fisher minibatches and raw importance weights.

## Sequential finetuning sequences

Chain each stage's final checkpoint into the next `--pretrained-path` for IDs 3, 8, 11. The following Bash example works for SFT, L2, EWC and LoRA:

```bash
method=sft                         # sft, l2, ewc, or lora
config=sambar_narrow
if [[ "$method" == lora ]]; then config=libero_lora; fi
previous=checkpoints/sambar_narrow/pretrain32/29999/params
learned=()
stage=0
for task in 3 8 11; do
  stage=$((stage + 1))
  extra=()
  if [[ "$method" == ewc && ${#learned[@]} -gt 0 ]]; then
    extra=(--old-task-ids "${learned[@]}")
  fi
  uv run --extra cuda python -m baselines.train --method "$method" --setting narrow \
    --exp-name "${method}_sequence/stage${stage}" --pretrained-path "$previous" \
    --finetune-task-ids "$task" "${extra[@]}"
  previous="checkpoints/${config}/${method}_sequence/stage${stage}/9999/params"
  learned+=("$task")
done
```

For EWC, `old-task-ids` joins previously learned tasks to the original 32-task Fisher data. It does not use SAMBAR's cached running average. The broad EWC source computes history from its 117-task pretraining mixture; it does not implement accumulation of earlier broad finetuning tasks. That source limitation is retained instead of silently defining a different EWC protocol.

## RETAIN (linear checkpoint merge)

First run the SFT baseline, then:

```bash
uv run --extra cuda python -m baselines.RETAIN \
  --sft checkpoints/sambar_narrow/task3/9999 \
  --pretrained checkpoints/sambar_narrow/pretrain32/29999 \
  --alpha 0.5 --out checkpoints/retain/task3
```

The retained formula is `alpha * SFT + (1-alpha) * reference`. Arguments identify checkpoint step directories, not their `params` children. The script checks parameter keys/shapes, uses the source interpolation precision and copies inference assets. The same command supports broad checkpoints. For sequential RETAIN, train SFT from the previous merged checkpoint, merge against that same incoming checkpoint, and carry the merged checkpoint to the next task. Specify alpha explicitly for each experiment.

## Co-training

The requested `train_pi05_cotrain_sequential.py` implementation is now [`src/openpi/training/baselines/cotrain.py`](../src/openpi/training/baselines/cotrain.py); its three-stage launcher is preserved here:

```bash
PRETRAIN=checkpoints/sambar_narrow/pretrain32/29999/params \
TASKS='3 8 11' OLD_FRACTION=0.5 EVAL=0 \
bash baselines/train_cotrain_seq_libero.sh
```

The launcher retains its 10,000-step, batch-16, LR-1e-5 defaults; set `STEPS`, `BATCH` and `LR` to match a comparison protocol. It mixes pretraining and already-learned tasks with the current task. A single stage can be launched through `--method cotrain --finetune-task-ids 3`; later stages add `--old-task-ids`. The source implementation supports the narrow/sequential curriculum only. No new broad replay protocol has been invented.

## Simple Recipe Works (GRPO + LoRA)

This RL baseline optimizes the policy through environment rollouts and uses the rollout hyperparameters from [Simple Recipe Works](https://arxiv.org/abs/2603.11653). The narrowly pretrained checkpoint achieved 0–5% success on the held-out tasks, so successful rollouts were too rare to provide a reliable reward signal. Before collecting rollouts at each stage, we trained the LoRA adapters for 100 behavior-cloning iterations on that task’s demonstrations.

```bash
PRETRAIN=checkpoints/sambar_narrow/pretrain32/29999/params \
TASKS='3 8 11' SFT_STEPS=100 SKIP_EVAL=1 \
bash baselines/train_grpo_lora_seq_libero.sh
```

Run LIBERO setup first. This is the source whole-model LoRA SFT/GRPO baseline, separate from SAMBAR. The launcher preserves its numeric defaults, including the short per-task SFT phase. `TRAIN_GPUS`, `ENV_GPUS`, `EVAL_GPU`, `SFT_STEPS`, `SFT_LR` and `SFT_BATCH` select runtime/experiment settings. It uses `libero_grpo_lora`, including its original adapter rank and noise/likelihood implementation. `uv run --extra cuda python -m baselines.train_grpo --help` lists its options.

The retained sequence shell scripts target Linux/Bash. Evaluation defaults in those scripts are inherited and differ from `examples/libero/evaluate.py`; choose matching seeds, trials and training budgets for comparisons. Hardware integrations and plain TACO are excluded.
