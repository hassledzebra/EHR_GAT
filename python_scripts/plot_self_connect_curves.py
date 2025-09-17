import re
import os
import sys
import argparse
from typing import List, Tuple, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOG_PATH = 'self_connect_sweep.log'

def parse_log(path: str) -> List[Tuple[int, float, float, float]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Log file not found: {path}")

    pct = None
    results = []  # list of (pct, auc, sens, spec)

    # Regex patterns
    pct_pat = re.compile(r"Running with sample fraction:\s*([0-9.]+)\s*\(pct=([0-9]+)\)")
    final_test_pat = re.compile(r"^Final test performance \(for CSV\):")
    sens_pat = re.compile(r"^Sensitivity \(Recall\):\s*([0-9.]+)")
    spec_pat = re.compile(r"^Specificity:\s*([0-9.]+)")
    auc_pat = re.compile(r"^AUC-ROC:\s*([0-9.]+)")

    lines = open(path, 'r', errors='ignore').read().splitlines()

    i = 0
    current_pct = None
    while i < len(lines):
        m = pct_pat.search(lines[i])
        if m:
            # record the latest pct seen
            try:
                current_pct = int(m.group(2))
            except Exception:
                current_pct = None
            i += 1
            continue

        if final_test_pat.search(lines[i]):
            # scan forward a small window for sens, spec, auc
            sens = spec = auc = None
            for j in range(i+1, min(i+30, len(lines))):
                ms = sens_pat.search(lines[j])
                if ms:
                    sens = float(ms.group(1))
                    continue
                mp = spec_pat.search(lines[j])
                if mp:
                    spec = float(mp.group(1))
                    continue
                ma = auc_pat.search(lines[j])
                if ma:
                    auc = float(ma.group(1))
                    continue
                # stop early if we've seen all
                if sens is not None and spec is not None and auc is not None:
                    break
            if current_pct is not None and sens is not None and spec is not None and auc is not None:
                results.append((current_pct, auc, sens, spec))
        i += 1

    # Deduplicate by pct keeping the last occurrence (most recent run)
    by_pct = {}
    for p, auc, s, sp in results:
        by_pct[p] = (auc, s, sp)
    out = [(p, *by_pct[p]) for p in sorted(by_pct.keys())]
    return out


def plot_curves(data: List[Tuple[int, float, float, float]], out_path: str = 'self_connect_curves.png'):
    if not data:
        raise ValueError('No data parsed from log; ensure runs have completed and log has content.')
    xs = [p for p, _, _, _ in data]
    aucs = [a for _, a, _, _ in data]
    sens = [s for _, _, s, _ in data]
    specs = [sp for _, _, _, sp in data]

    plt.figure(figsize=(8,5))
    plt.plot(xs, aucs, marker='o', label='Test AUC-ROC')
    plt.plot(xs, sens, marker='o', label='Test Sensitivity')
    plt.plot(xs, specs, marker='o', label='Test Specificity')
    plt.xlabel('Sample size (%)')
    plt.ylabel('Metric value')
    plt.title('Self-Connect (diag) — 800 epochs, CPU')
    plt.ylim(0.45, 0.85)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Plot self-connect curves from log')
    parser.add_argument('--log', type=str, default=LOG_PATH, help='Path to self_connect_sweep.log')
    parser.add_argument('--pcts', type=str, default=None, help='Comma-separated whitelist of percentages to plot, e.g., 10,30,50,70,90')
    parser.add_argument('--out', type=str, default='self_connect_curves.png', help='Output image path')
    args = parser.parse_args()

    data = parse_log(args.log)
    if args.pcts:
        keep = set()
        for tok in args.pcts.split(','):
            tok = tok.strip()
            if not tok:
                continue
            try:
                keep.add(int(tok))
            except Exception:
                pass
        data = [row for row in data if row[0] in keep]

    out = plot_curves(data, args.out)
    # Print a small summary table
    print('pct,aucroc,sensitivity,specificity')
    for p, a, s, sp in data:
        print(f"{p},{a:.4f},{s:.4f},{sp:.4f}")
    print(f"Saved plot: {out}")

if __name__ == '__main__':
    main()
