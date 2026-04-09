"""
logger.py
---------
Logs attention states and metrics to a timestamped CSV file.

Output: logs/session_YYYYMMDD_HHMMSS.csv
Columns: timestamp, state, ear, yaw, pitch

Also prints a session summary (% time in each state) on exit.
"""

import csv
import os
import time
from datetime import datetime
from collections import Counter


class AttentionLogger:

    LOG_INTERVAL = 0.5  # seconds between log entries (avoid huge files)

    def __init__(self):
        os.makedirs("logs", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = f"logs/session_{ts}.csv"

        self._rows: list[dict] = []
        self._last_log_time: float = 0.0

    # ── Public API ────────────────────────────────────────────────────

    def log(self, elapsed: float, state, metrics: dict):
        """
        Record one data point (throttled to LOG_INTERVAL seconds).

        Args:
            elapsed : seconds since session start
            state   : AttentionState enum value
            metrics : dict with keys ear, yaw, pitch
        """
        now = time.time()
        if now - self._last_log_time < self.LOG_INTERVAL:
            return
        self._last_log_time = now

        self._rows.append({
            "timestamp_s" : round(elapsed, 2),
            "state"       : state.value,
            "ear"         : metrics.get("ear",   "N/A"),
            "yaw_deg"     : metrics.get("yaw",   "N/A"),
            "pitch_deg"   : metrics.get("pitch", "N/A"),
        })

    def save(self):
        """Write CSV and print session summary."""
        if not self._rows:
            print("[Logger] No data recorded.")
            return

        fieldnames = ["timestamp_s", "state", "ear", "yaw_deg", "pitch_deg"]
        with open(self.filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._rows)

        print(f"\n[Logger] Session log saved → {self.filepath}")
        self._print_summary()

    # ── Private ───────────────────────────────────────────────────────

    def _print_summary(self):
        states = [r["state"] for r in self._rows]
        counts = Counter(states)
        total  = len(states)
        duration = self._rows[-1]["timestamp_s"] if self._rows else 0

        print("\n" + "="*40)
        print("       SESSION SUMMARY")
        print("="*40)
        print(f"  Duration : {int(duration//60):02d}m {int(duration%60):02d}s")
        print(f"  Samples  : {total}")
        print("-"*40)
        for state_val, count in counts.most_common():
            pct = count / total * 100
            bar = "█" * int(pct / 5)
            print(f"  {state_val:<14s} {pct:5.1f}%  {bar}")
        print("="*40 + "\n")
