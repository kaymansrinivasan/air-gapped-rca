#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
board="${1:?Use: bash scripts/finish_benchmarks.sh jetson|rb3}"
case "$board" in
  jetson)
    data_root="${RCA_DATA_ROOT:-$HOME/Documents/airgap-rca}"
    interpreter="$data_root/.venv-llamaindex-jetson/bin/python"
    power_args=(--jetson-power)
    ;;
  rb3)
    data_root="${RCA_DATA_ROOT:-$PWD}"
    interpreter="$PWD/.venv-rca/bin/python"
    power_args=()
    ;;
  *) echo "Choose jetson or rb3." >&2; exit 2 ;;
esac

"$interpreter" -m unittest discover -s tests -q
if [[ "$board" == rb3 ]]; then
  if ! curl -fsS --max-time 5 http://127.0.0.1:8000/health >/dev/null; then
    runtime="$HOME/radxa-dragon-q6a-qwen3.5-0.8b-qcs6490-qnn-npu"
    nohup "$runtime/install-2k/launch.sh" > /tmp/rb3-qwen-server.log 2>&1 < /dev/null &
  fi
  "$interpreter" - <<'PY'
import json
import time
from urllib.request import urlopen
for attempt in range(30):
    try:
        with urlopen('http://127.0.0.1:8000/health', timeout=2) as response:
            health = json.load(response)
        if health.get('ready') is True:
            print('RB3 Qwen server ready.')
            break
    except (OSError, ValueError):
        pass
    time.sleep(2)
else:
    raise SystemExit('RB3 server not ready; inspect /tmp/rb3-qwen-server.log.')
PY
fi

output="$data_root/artifacts/evaluation/final_suite_v1"
mkdir -p "$output"
nohup "$interpreter" -u -m src.benchmark_suite \
  --backend "$board" --data-root "$data_root" --output-dir "$output" \
  --resume --electricity-per-kwh 0.50 --currency MYR "${power_args[@]}" \
  >> "$output/run.log" 2>&1 < /dev/null &
pid=$!
echo "Started benchmark suite, PID $pid."
echo "Illustrative electricity scenario: MYR 0.50/kWh; this is not a verified utility tariff."
echo "Progress: tail -n 8 $output/run.log"
echo "Final report: $output/REPORT.md"
echo "The suite resumes saved questions when this command is run again."
