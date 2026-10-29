#!/usr/bin/env bash
# Install the source-matching LIBERO revision; do not vendor its code/assets in this repository.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
libero_revision=8f1084e3132a39270c3a13ebe37270a43ece2a01
libero_path="${LIBERO_SOURCE:-$repo_root/third_party/libero}"
if [[ ! -d "$libero_path" ]]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$libero_path"
fi
if [[ -n "$(git -C "$libero_path" status --porcelain)" ]]; then
    echo "LIBERO checkout has local changes; use a clean LIBERO_SOURCE path." >&2
    exit 1
fi
git -C "$libero_path" checkout --detach "$libero_revision"
uv venv --python 3.11 examples/libero/.venv
uv pip install --python examples/libero/.venv/bin/python -r examples/libero/requirements.txt
uv pip install --python examples/libero/.venv/bin/python --no-deps -e "$libero_path"
uv pip install --python examples/libero/.venv/bin/python -e packages/openpi-client
echo "LIBERO installed at $libero_path ($libero_revision). See research/SETUP.md for data and rendering setup."
