#!/usr/bin/env bash
task_root="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
"$task_root/start.sh" "$@"
task_result=$?
if [[ -t 0 ]]; then
    read -r -p "按回车关闭此窗口…" task_reply
fi
exit "$task_result"
