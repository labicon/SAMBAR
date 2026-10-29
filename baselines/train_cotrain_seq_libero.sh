#!/usr/bin/env bash
# Sequential co-training (experience replay) baseline for pi0.5 on LIBERO: learn TASKS one after another
# with BC, mixing OLD_FRACTION of every batch from the old tasks (32 pretrain tasks + tasks learned in
# earlier stages). Stage k initializes from stage k-1's final checkpoint. Optional LIBERO-40 eval
# (TRIALS episodes per task, default 20), bucketed into pretrain vs learned tasks:
#   EVAL=0 none | EVAL=1 after every stage | EVAL=final only the last stage's checkpoint.
#
# Usage:
#   GPUS=4,5,6,7 OLD_FRACTION=0.5 ./baselines/train_cotrain_seq_libero.sh
#   GPUS=0,1 TASKS="3 8 11" STEPS=10000 LR=1e-5 EVAL=1 EVAL_GPU=1 ./baselines/train_cotrain_seq_libero.sh
#   GPUS=2,3 OLD_FRACTION=0.5 EVAL=final TRIALS=20 ./baselines/train_cotrain_seq_libero.sh
# Extra args are passed through to every stage (e.g. --no-replay-pretrain).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TASKS="${TASKS:-3 8 11}"
GPUS="${GPUS:-0,1}"
OLD_FRACTION="${OLD_FRACTION:-0.5}"
STEPS="${STEPS:-10000}"
LR="${LR:-1e-5}"
BATCH="${BATCH:-16}"
SPLIT_SEED="${SPLIT_SEED:-7}"
SAVE="${SAVE:-2000}"
CONFIG="${CONFIG:-sambar_narrow}"
PRETRAIN="${PRETRAIN:?Set PRETRAIN to the pretrained checkpoint params directory}"
TAG="${TAG:-pi05_seq_cotrain_old${OLD_FRACTION/./_}}"
EVAL="${EVAL:-0}"
EVAL_GPU="${EVAL_GPU:-${GPUS##*,}}"
EVAL_PORT="${EVAL_PORT:-8150}"
TRIALS="${TRIALS:-20}"
LOG_ROOT="${LOG_ROOT:-./outputs/train_logs}"
EVAL_ROOT="${EVAL_ROOT:-./outputs/libero_eval}"

export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-./data/openpi-cache}"
export TMPDIR="${TMPDIR:-./data/tmp}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-./data/jax-cache}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-./data/hf-cache/lerobot}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
mkdir -p "$LOG_ROOT" "$EVAL_ROOT" "$TMPDIR"

case "$EVAL" in 0|1|final) ;; *) echo "FATAL: EVAL must be 0, 1 or final (got $EVAL)" >&2; exit 1 ;; esac
read -r -a task_list <<< "$TASKS"
last_task="${task_list[-1]}"

init="$PRETRAIN"
done_tasks=()
name=""
for task in $TASKS; do
    if [[ ${#done_tasks[@]} -eq 0 ]]; then name="task${task}"; else name="${name}_to_task${task}"; fi
    exp="${TAG}_${name}_seed${SPLIT_SEED}_lr${LR}"
    old_args=()
    [[ ${#done_tasks[@]} -gt 0 ]] && old_args=(--old-task-ids "${done_tasks[@]}")
    echo "=== stage: task ${task} (carried old: ${done_tasks[*]:-none}) exp=${exp} init=${init}"

    CUDA_VISIBLE_DEVICES="$GPUS" XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}" \
        uv run --extra cuda python -m baselines.train --method cotrain \
        --exp-name "$exp" --config-name "$CONFIG" --pretrained-path "$init" \
        --finetune-task-ids "$task" "${old_args[@]}" --old-fraction "$OLD_FRACTION" --split-seed "$SPLIT_SEED" \
        --num-train-steps "$STEPS" --lr "$LR" --batch-size "$BATCH" \
        --save-interval "$SAVE" --keep-period "$SAVE" \
        "$@" 2>&1 | tee "$LOG_ROOT/${exp}.log"

    ckpt_root="./checkpoints/${CONFIG}/${exp}"
    step="$(find "$ckpt_root" -mindepth 1 -maxdepth 1 -type d -regex '.*/[0-9]+' -printf '%f\n' | sort -n | tail -1)"
    [[ -n "$step" ]] || { echo "FATAL: no checkpoint under $ckpt_root" >&2; exit 1; }
    done_tasks+=("$task")
    init="${ckpt_root}/${step}/params"

    if [[ "$EVAL" == 1 || ( "$EVAL" == final && "$task" == "$last_task" ) ]]; then
        label="${exp}_step${step}_all40"
        GPU="$EVAL_GPU" PORT="$EVAL_PORT" TRIALS="$TRIALS" LABEL="$label" OUT_ROOT="$EVAL_ROOT" EVAL_SET=all \
            ./scripts/eval_pi05_libero40.sh "$CONFIG" "${ckpt_root}/${step}" 2>&1 | tee "$EVAL_ROOT/${label}.log"
        uv run --extra cuda python scripts/eval_summary_pi05_holdout.py "$EVAL_ROOT/$label" \
            --split-seed "$SPLIT_SEED" --holdout-task-ids "${done_tasks[@]}" | tee "$EVAL_ROOT/$label/summary_pretrain_vs_learned.txt"
    fi
done
echo "=== done: final checkpoint ${init%/params}"
