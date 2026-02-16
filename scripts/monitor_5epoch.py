#!/usr/bin/env python3
"""Tail a training log and append a row to CSV every 5 epochs.
Usage: python scripts/monitor_5epoch.py --log runs/sgn_test1.out --csv runs/sgn_test1_reports.csv --out runs/sgn_test1_monitor.out
Writes human-readable lines to --out and CSV rows to --csv.
"""
import argparse
import time
import re
import sys
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--log", required=True, help="training log file to tail")
parser.add_argument("--csv", required=True, help="CSV output path")
parser.add_argument("--out", required=True, help="human monitor output file")
parser.add_argument("--period", type=int, default=5, help="emit every N epochs (default: 5)")
args = parser.parse_args()

pat = re.compile(r'^Epoch\s+(\d+):\s+train_loss=([0-9.eE+-]+)(?:,\s*val_loss=([0-9.eE+-]+))?')
log_path = Path(args.log)
csv_path = Path(args.csv)
out_path = Path(args.out)
period = args.period

# wait for log file to appear
while not log_path.exists():
    time.sleep(0.5)

with log_path.open('r') as f, out_path.open('a') as mo:
    # seek to end and follow
    f.seek(0, 2)
    try:
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.2)
                continue
            m = pat.search(line)
            if not m:
                continue
            epoch = int(m.group(1))
            if epoch % period != 0:
                continue
            train_loss = float(m.group(2))
            val_loss = float(m.group(3)) if m.group(3) else None
            out = f"[LIVE REPORT] epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss if val_loss is not None else 'N/A'}\n"
            mo.write(out)
            mo.flush()
            sys.stdout.write(out)
            # append to CSV
            row = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'timestamp_utc': pd.Timestamp.utcnow().isoformat()
            }
            pd.DataFrame([row]).to_csv(csv_path, mode='a', header=not csv_path.exists(), index=False)
    except KeyboardInterrupt:
        pass
