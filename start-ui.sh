#!/usr/bin/env bash

set -e

source "${BASH_SOURCE[0]%/*}/lib.include.sh"

# Xet is buggy. Disabled by default unless already defined - https://github.com/Nerogar/OneTrainer/issues/949
if [[ -z "${HF_HUB_DISABLE_XET+x}" ]]; then
    export HF_HUB_DISABLE_XET=1
fi

# Clear Python bytecode cache to ensure code changes take effect
# This fixes issues where stale .pyc files cause NameError or other bugs
# after editing Python modules.
echo "Clearing Python bytecode cache..."
find "${SCRIPT_DIR}" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "${SCRIPT_DIR}" -name "*.pyc" -delete 2>/dev/null || true
find "${SCRIPT_DIR}" -name "*.pyo" -delete 2>/dev/null || true
echo "Cache cleared."

prepare_runtime_environment

run_python_in_active_env "scripts/train_ui.py" "$@"
