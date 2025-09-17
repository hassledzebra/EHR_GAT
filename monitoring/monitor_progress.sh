#!/bin/bash
# Simple progress monitor - run this script to check experiment progress

PROGRESS_FILE="/home/zhan1/EPI/experiment_progress.txt"
STATUS_FILE="/home/zhan1/EPI/experiment_status.json"

echo "=== EPI Experiment Progress Monitor ==="
echo "Current time: $(date)"
echo

if [ -f "$STATUS_FILE" ]; then
    echo "=== Current Status ==="
    python3 -c "
import json
with open('$STATUS_FILE', 'r') as f:
    status = json.load(f)
print(f'Phase: {status["phase"]}')
print(f'Progress: {status["completed"]}/{status["total"]} ({status["progress_percent"]:.1f}%)')
print(f'Last updated: {status["timestamp"]}')
if status.get('details'):
    print('Details:', status['details'])
"
    echo
fi

if [ -f "$PROGRESS_FILE" ]; then
    echo "=== Recent Progress Messages ==="
    tail -20 "$PROGRESS_FILE"
    echo
    echo "=== Live monitoring (Ctrl+C to stop) ==="
    tail -f "$PROGRESS_FILE"
else
    echo "Progress file not found. Experiments may not have started yet."
fi
