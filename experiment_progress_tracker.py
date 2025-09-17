#!/usr/bin/env python3
"""
Progress tracking system for EPI model experiments.
Writes status updates to files that can be monitored.
"""

import os
import time
import json
from datetime import datetime
from pathlib import Path

class ProgressTracker:
    def __init__(self, base_dir="/home/zhan1/EPI"):
        self.base_dir = Path(base_dir)
        self.progress_file = self.base_dir / "experiment_progress.txt"
        self.status_file = self.base_dir / "experiment_status.json"

        # Initialize files
        self.log_message("🔬 Progress tracker initialized")
        self.update_status("initialized", 0, 0)

    def log_message(self, message):
        """Log a timestamped message to the progress file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.progress_file, "a") as f:
            f.write(f"[{timestamp}] {message}\n")
        print(f"[{timestamp}] {message}")

    def update_status(self, phase, completed, total, details=None):
        """Update the JSON status file with current progress."""
        status = {
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "completed": completed,
            "total": total,
            "progress_percent": (completed / total * 100) if total > 0 else 0,
            "details": details or {}
        }

        with open(self.status_file, "w") as f:
            json.dump(status, f, indent=2)

    def start_experiments(self, total_experiments):
        """Mark the start of experiments."""
        self.log_message(f"🚀 Starting {total_experiments} experiments")
        self.log_message("📊 Configuration: 500 comorbidities, 800 epochs, self-connect=True")
        self.log_message("📝 Sample sizes: 10%, 30%, 50%, 70%, 90% (3 runs each)")
        self.update_status("running_experiments", 0, total_experiments)

    def experiment_completed(self, experiment_id, success, current_count, total_count, duration_min=None):
        """Mark an individual experiment as completed."""
        status_emoji = "✅" if success else "❌"
        duration_str = f" ({duration_min:.1f} min)" if duration_min else ""
        self.log_message(f"{status_emoji} Experiment {experiment_id} completed{duration_str}")

        self.update_status("running_experiments", current_count, total_count, {
            "last_experiment": experiment_id,
            "last_success": success,
            "last_duration_min": duration_min
        })

    def experiments_finished(self, successful, failed):
        """Mark all experiments as finished."""
        total = successful + failed
        self.log_message(f"🏁 All experiments completed: {successful}/{total} successful")
        self.update_status("experiments_complete", total, total, {
            "successful": successful,
            "failed": failed
        })

    def start_analysis(self):
        """Mark the start of analysis phase."""
        self.log_message("📊 Starting results analysis...")
        self.update_status("analyzing_results", 0, 1)

    def analysis_complete(self, summary_stats):
        """Mark analysis as complete."""
        self.log_message("📈 Analysis complete - generating plots...")
        self.update_status("generating_plots", 0, 1, summary_stats)

    def plots_complete(self, plot_files):
        """Mark plots as complete."""
        self.log_message("🎨 Plots generated:")
        for plot_file in plot_files:
            self.log_message(f"   📊 {plot_file}")
        self.update_status("complete", 1, 1, {"plot_files": plot_files})

    def error_occurred(self, error_message):
        """Log an error."""
        self.log_message(f"💥 Error: {error_message}")
        self.update_status("error", 0, 0, {"error": error_message})

    def get_current_status(self):
        """Get the current status."""
        if self.status_file.exists():
            with open(self.status_file, "r") as f:
                return json.load(f)
        return None

def create_monitoring_script():
    """Create a simple monitoring script."""
    monitor_script = '''#!/bin/bash
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
print(f'Phase: {status[\"phase\"]}')
print(f'Progress: {status[\"completed\"]}/{status[\"total\"]} ({status[\"progress_percent\"]:.1f}%)')
print(f'Last updated: {status[\"timestamp\"]}')
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
'''

    with open("/home/zhan1/EPI/monitor_progress.sh", "w") as f:
        f.write(monitor_script)

    os.chmod("/home/zhan1/EPI/monitor_progress.sh", 0o755)

if __name__ == "__main__":
    # Demo the progress tracker
    tracker = ProgressTracker()
    create_monitoring_script()

    tracker.log_message("📋 Progress tracking system ready")
    tracker.log_message("💡 To monitor progress, run: ./monitor_progress.sh")
    tracker.log_message("📁 Progress files:")
    tracker.log_message("   📄 experiment_progress.txt - Human-readable log")
    tracker.log_message("   📄 experiment_status.json - Machine-readable status")
    tracker.log_message("   📄 monitor_progress.sh - Live monitoring script")