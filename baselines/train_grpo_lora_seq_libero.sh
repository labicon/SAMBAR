#!/usr/bin/env bash
# Sequential continual learning with the continual-vla-rl recipe (GRPO + whole-model LoRA) for pi0.5
# on LIBERO, in the UNSEEN-task setting ("option B").
#
# Starts from the 32-task BC pretrain checkpoint, which has never seen TASKS. Each new task arrives as
# one stage with two phases on the SAME LoRA adapter (only LoRA trains, config libero_grpo_lora):
#   1. SFT: BC on the new task's demos (scripts/train.py --mode bc_finetune, LoRA-only), so the
#      policy reaches partial competence. The paper instead starts from an SFT ckpt that already saw
#      every task; this is the per-task analog for tasks that are genuinely new.
#   2. RL: online GRPO on that task (-m baselines.train_grpo), paper defaults.
# Stage k starts from stage k-1's final RL checkpoint. After each stage the checkpoint is evaluated on
# all LIBERO-40 tasks (scripts/eval_pi05_libero40.sh) and bucketed into the 32 pretrain tasks vs the
# tasks learned so far (scripts/eval_summary_pi05_holdout.py). No old-task data is ever used.
#
# Usage:
#   TRAIN_GPUS=0,1,2,3 ENV_GPUS=4,5,6,7 EVAL_GPU=7 ./baselines/train_grpo_lora_seq_libero.sh
#   TASKS="3 8 11" SFT_STEPS=100 SFT_LR=5e-5 ./baselines/train_grpo_lora_seq_libero.sh --grpo.noise-level 0.5
# Extra args are passed through to every RL phase (e.g. --grpo.* overrides).
# SKIP_SFT=1 skips phase 1 (paper-style RL only). SKIP_EVAL=1 skips the per-stage LIBERO-40 eval.
# EVAL_SFT=1 also evaluates the post-SFT checkpoint (separates BC-induced vs RL-induced forgetting).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TASKS="${TASKS:-3 8 11}"
SEED="${SEED:-1234}"
SPLIT_SEED="${SPLIT_SEED:-7}"
PRETRAIN="${PRETRAIN:?Set PRETRAIN to the pretrained checkpoint params directory}"
CONFIG="${CONFIG:-libero_grpo_lora}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3}"
ENV_GPUS="${ENV_GPUS:-4,5,6,7}"
EVAL_GPU="${EVAL_GPU:-7}"
EVAL_PORT="${EVAL_PORT:-8130}"
TRIALS="${TRIALS:-20}"
SFT_STEPS="${SFT_STEPS:-100}"  # pilot (task 3, lr 5e-5): 50→10%, 100→50%, 150→66%, 200→80%, 250→92%
SFT_LR="${SFT_LR:-5e-5}"
SFT_BATCH="${SFT_BATCH:-16}"
TAG="${TAG:-pi05_sft_grpo_lora_seq}"
LOG_ROOT="${LOG_ROOT:-./outputs/train_logs}"
EVAL_ROOT="${EVAL_ROOT:-./outputs/libero_eval}"

export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-./data/openpi-cache}"
export TMPDIR="${TMPDIR:-./data/tmp}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-./data/jax-cache}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-./data/hf-cache/lerobot}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
mkdir -p "$LOG_ROOT" "$EVAL_ROOT" "$TMPDIR"

last_ckpt() {  # newest numeric step dir under $1
    local step
    step="$(find "$1" -mindepth 1 -maxdepth 1 -type d -regex '.*/[0-9]+' -printf '%f\n' | sort -n | tail -1)"
    [[ -n "$step" ]] || { echo "FATAL: no checkpoint under $1" >&2; exit 1; }
    echo "$1/$step"
}

eval_ckpt() {  # eval_ckpt <ckpt_dir> <label> <learned task ids...>
    local ckpt_dir="$1" label="$2"
    shift 2
    GPU="$EVAL_GPU" PORT="$EVAL_PORT" TRIALS="$TRIALS" LABEL="$label" OUT_ROOT="$EVAL_ROOT" EVAL_SET=all \
        ./scripts/eval_pi05_libero40.sh "$CONFIG" "$ckpt_dir" 2>&1 | tee "$EVAL_ROOT/${label}.log"
    uv run --extra cuda python scripts/eval_summary_pi05_holdout.py "$EVAL_ROOT/$label" \
        --split-seed "$SPLIT_SEED" --holdout-task-ids "$@" | tee "$EVAL_ROOT/$label/summary_pretrain_vs_learned.txt"
}

init="$PRETRAIN"
done_tasks=()
name="${TAG}"
for task in $TASKS; do
    if [[ ${#done_tasks[@]} -eq 0 ]]; then
        name="${TAG}_task${task}"
    else
        name="${name}_to_task${task}"
    fi
    exp="${name}_seed${SEED}"
    ckpt_root="./checkpoints/${CONFIG}/${exp}"
    echo "=== stage: task ${task} (old: ${done_tasks[*]:-none}) exp=${exp} init=${init}"

    old_args=()
    if [[ ${#done_tasks[@]} -gt 0 ]]; then
        old_args=(--old-task-ids "${done_tasks[@]}")
    fi

    # --- phase 1: SFT (BC, LoRA-only) on the new task's demos ---
    if [[ "${SKIP_SFT:-0}" != 1 ]]; then
        sft_exp="${exp}_sft"
        if [[ -d "./checkpoints/${CONFIG}/${sft_exp}/$((SFT_STEPS - 1))/params" ]]; then
            echo "=== [task ${task}] SFT checkpoint already exists -> reusing ${sft_exp}/$((SFT_STEPS - 1))"
        else
        echo "=== [task ${task}] SFT: ${SFT_STEPS} steps, lr ${SFT_LR}, batch ${SFT_BATCH} -> ${sft_exp}"
        CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}" \
            uv run --extra cuda python -m baselines.train --method sft \
            --exp-name "$sft_exp" --config-name "$CONFIG" --finetune-task-ids "$task" \
            --pretrained-path "$init" --split-seed "$SPLIT_SEED" --seed "$SEED" \
            --num-train-steps "$SFT_STEPS" --save-interval "$SFT_STEPS" --lr "$SFT_LR" --batch-size "$SFT_BATCH" \
            2>&1 | tee "$LOG_ROOT/${sft_exp}.log"
        fi
        sft_ckpt="$(last_ckpt "./checkpoints/${CONFIG}/${sft_exp}")"
        if [[ "${EVAL_SFT:-0}" == 1 && "${SKIP_EVAL:-0}" != 1 ]]; then
            eval_ckpt "$sft_ckpt" "${sft_exp}_step$(basename "$sft_ckpt")_all40" "${done_tasks[@]}" "$task"
        fi
        init="${sft_ckpt}/params"
    fi

    # --- phase 2: online GRPO on the new task ---
    CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}" \
        uv run --extra cuda python -m baselines.train_grpo \
        --exp-name "$exp" --config-name "$CONFIG" --task-id "$task" "${old_args[@]}" \
        --pretrained-path "$init" --seed "$SEED" --split-seed "$SPLIT_SEED" --env-gpus "$ENV_GPUS" \
        "$@" 2>&1 | tee "$LOG_ROOT/${exp}.log"

    ckpt_dir="$(last_ckpt "$ckpt_root")"
    done_tasks+=("$task")
    init="${ckpt_dir}/params"

    if [[ "${SKIP_EVAL:-0}" != 1 ]]; then
        eval_ckpt "$ckpt_dir" "${exp}_step$(basename "$ckpt_dir")_all40" "${done_tasks[@]}"
    fi
done
echo "=== done: final checkpoint ${init%/params}"
