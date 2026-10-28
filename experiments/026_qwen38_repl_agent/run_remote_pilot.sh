#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SSH=(ssh -p 40020 -i "$HOME/.ssh/id_ed25519" root@213.192.2.77)
SCP=(scp -P 40020 -i "$HOME/.ssh/id_ed25519")
REMOTE=/root/qwen38_repl_4090
LOCAL_OUT="$REPO/kaggle_runs/qwen38_repl_4090"
mkdir -p "$LOCAL_OUT"

python "$REPO/experiments/026_qwen38_repl_agent/prepare_pilot_data.py" \
  --repo "$REPO" --out "$LOCAL_OUT/data"
"${SSH[@]}" "mkdir -p $REMOTE/code $REMOTE/output"
"${SCP[@]}" -r \
  "$REPO/experiments/026_qwen38_repl_agent/public_notebook/output_files/agent" \
  "$REPO/experiments/026_qwen38_repl_agent/pilot_config.json" \
  "$REPO/experiments/026_qwen38_repl_agent/pilot_runner.py" \
  "$REPO/experiments/026_qwen38_repl_agent/setup_remote.sh" \
  "$REPO/experiments/026_qwen38_repl_agent/start_server.sh" \
  "root@213.192.2.77:$REMOTE/code/"
"${SCP[@]}" -r "$LOCAL_OUT/data" "root@213.192.2.77:$REMOTE/"

"${SSH[@]}" "chmod +x $REMOTE/code/*.sh && $REMOTE/code/setup_remote.sh"
"${SSH[@]}" "pkill -f '[l]lama-server' || true; nohup $REMOTE/code/start_server.sh >$REMOTE/run/server.log 2>&1 & echo \$! >$REMOTE/run/server.pid"
for _ in $(seq 1 180); do
  if "${SSH[@]}" "curl -fsS http://127.0.0.1:8080/health >/dev/null"; then break; fi
  sleep 5
done
"${SSH[@]}" "curl -fsS http://127.0.0.1:8080/health"
"${SSH[@]}" "bash -o pipefail -c 'cd $REMOTE/code && python pilot_runner.py --config pilot_config.json --data-dir $REMOTE/data --output-dir $REMOTE/output 2>&1 | tee $REMOTE/run/pilot.log'"
"${SCP[@]}" -r "root@213.192.2.77:$REMOTE/output" "root@213.192.2.77:$REMOTE/run" "$LOCAL_OUT/"
