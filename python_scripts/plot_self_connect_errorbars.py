import csv
import os
from collections import defaultdict
from statistics import mean, pstdev

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CSV = 'sample_size_performance.csv'

KEEP_PCTS = {10, 30, 50, 70, 90}

def load_rows(path=CSV):
    rows = []
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        hdr_index = {name: i for i, name in enumerate(header)}
        for r in reader:
            if not r:
                continue
            try:
                sample_pct = int(r[hdr_index.get('sample_pct', 2)])
            except Exception:
                continue
            device = r[hdr_index.get('device', 10)]
            epochs = int(float(r[hdr_index.get('epochs', 9)]))
            # Handle presence/absence of self_connect/seed columns
            self_idx = hdr_index.get('self_connect')
            if self_idx is not None and self_idx < len(r):
                self_conn = int(r[self_idx])
            else:
                # Assume 0 if not present
                self_conn = 0
            auc = float(r[hdr_index.get('test_aucroc', 18)])
            prauc = float(r[hdr_index.get('test_prauc', 19)])
            acc = float(r[hdr_index.get('test_acc', 20)])
            sens = float(r[hdr_index.get('test_sens', 21)])
            spec = float(r[hdr_index.get('test_spec', 22)])
            rows.append({
                'pct': sample_pct,
                'device': device,
                'epochs': epochs,
                'self_connect': self_conn,
                'auc': auc,
                'sens': sens,
                'spec': spec,
            })
    return rows

def aggregate(rows):
    # Filter for self_connect=1, device=cpu, epochs=800, pct in KEEP_PCTS
    rows = [r for r in rows if r['self_connect']==1 and r['device']=='cpu' and r['epochs']==800 and r['pct'] in KEEP_PCTS]
    by_pct = defaultdict(list)
    for r in rows:
        by_pct[r['pct']].append(r)
    agg = {}
    for pct, vals in by_pct.items():
        aucs = [v['auc'] for v in vals]
        sens = [v['sens'] for v in vals]
        specs = [v['spec'] for v in vals]
        agg[pct] = {
            'auc_mean': mean(aucs), 'auc_std': (pstdev(aucs) if len(aucs)>1 else 0.0),
            'sens_mean': mean(sens), 'sens_std': (pstdev(sens) if len(sens)>1 else 0.0),
            'spec_mean': mean(specs), 'spec_std': (pstdev(specs) if len(specs)>1 else 0.0),
            'n': len(vals),
        }
    return agg

def plot_errorbars(agg, out='self_connect_errorbars.png'):
    pcts = sorted(agg.keys())
    auc_m = [agg[p]['auc_mean'] for p in pcts]
    auc_s = [agg[p]['auc_std'] for p in pcts]
    sen_m = [agg[p]['sens_mean'] for p in pcts]
    sen_s = [agg[p]['sens_std'] for p in pcts]
    spe_m = [agg[p]['spec_mean'] for p in pcts]
    spe_s = [agg[p]['spec_std'] for p in pcts]

    plt.figure(figsize=(8,5))
    plt.errorbar(pcts, auc_m, yerr=auc_s, marker='o', capsize=3, label='AUC-ROC')
    plt.errorbar(pcts, sen_m, yerr=sen_s, marker='o', capsize=3, label='Sensitivity')
    plt.errorbar(pcts, spe_m, yerr=spe_s, marker='o', capsize=3, label='Specificity')
    plt.xlabel('Sample size (%)')
    plt.ylabel('Metric value')
    plt.title('Self-Connect (diag) — 800 epochs, CPU (mean ± std)')
    plt.ylim(0.45, 0.85)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    return out

def main():
    rows = load_rows()
    agg = aggregate(rows)
    print('pct,n,auc_mean,auc_std,sens_mean,sens_std,spec_mean,spec_std')
    for p in sorted(agg.keys()):
        a = agg[p]
        print(f"{p},{a['n']},{a['auc_mean']:.4f},{a['auc_std']:.4f},{a['sens_mean']:.4f},{a['sens_std']:.4f},{a['spec_mean']:.4f},{a['spec_std']:.4f}")
    out = plot_errorbars(agg)
    print(f"Saved plot: {out}")

if __name__ == '__main__':
    main()

