#!/usr/bin/env bash
# Compatibility entry point for the retained sequential baseline launchers.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
config="${1:?Provide config name}"
checkpoint="${2:?Provide checkpoint directory}"
label="${LABEL:-$(basename "$(dirname "$checkpoint")")_$(basename "$checkpoint")}"
output="${OUT_ROOT:-./outputs/libero_eval}/$label"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
uv run --extra cuda python scripts/evaluate.py --checkpoint "$checkpoint" --config "$config" \
    --setting narrow --seeds "${SEED:-7}" --trials "${TRIALS:-20}" \
    --task-ids 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
               20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 \
    --port "${PORT:-8000}" --output "$output"
# Keep the prior summary tool's directory layout for the original wrappers.
for suite in libero_spatial libero_object libero_goal libero_10; do
    mkdir -p "$output/$suite"
    cp "$output/seed${SEED:-7}/$suite/results_$suite.json" "$output/$suite/"
done
