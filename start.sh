#!/usr/bin/env bash
set -euo pipefail
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

task_root="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
task_python="${CAREER_PYTHON:-}"

if [[ -z "$task_python" ]] && command -v conda >/dev/null 2>&1; then
    task_conda_base="$(conda info --base)"
    if [[ -x "$task_conda_base/envs/career-agent/bin/python" ]]; then
        task_python="$task_conda_base/envs/career-agent/bin/python"
    fi
fi
if [[ -z "$task_python" ]]; then
    for task_candidate in "$HOME/miniconda3/envs/career-agent/bin/python" \
        "$HOME/anaconda3/envs/career-agent/bin/python" \
        /opt/miniconda3/envs/career-agent/bin/python; do
        if [[ -x "$task_candidate" ]]; then
            task_python="$task_candidate"
            break
        fi
    done
fi
if [[ -z "$task_python" ]] && command -v python3 >/dev/null 2>&1; then
    task_python="$(command -v python3)"
fi
if [[ -z "$task_python" || ! -x "$task_python" ]]; then
    echo "找不到 Python。请激活 career-agent，或设置 CAREER_PYTHON 为该环境的 Python 完整路径。" >&2
    exit 1
fi

exec "$task_python" "$task_root/deploy/local_stack.py" "$@"
