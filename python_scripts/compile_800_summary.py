import csv
import os
from collections import defaultdict
from statistics import mean, pstdev

CSV = 'sample_size_performance.csv'

def load_rows(path=CSV):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        idx = {name:i for i,name in enumerate(header)}
        # Backward compatibility for earlier rows without new columns
        def gi(name, default=None):
            return idx.get(name, default)
        rows=[]
        for r in reader:
            if not r: continue
            try:
                epochs = int(float(r[gi('epochs', 9)]))
            except Exception:
                continue
            if epochs != 800:
                continue
            device = r[gi('device', 10)]
            if device != 'cpu':
                continue
            pct = int(r[gi('sample_pct', 2)])
            # self_connect may be absent in old rows; treat as 0
            self_conn_idx = gi('self_connect')
            self_conn = int(r[self_conn_idx]) if (self_conn_idx is not None and self_conn_idx < len(r)) else 0
            auc = float(r[gi('test_aucroc', 18)])
            sens = float(r[gi('test_sens', 21)])
            spec = float(r[gi('test_spec', 22)])
            rows.append({'pct':pct,'self_connect':self_conn,'auc':auc,'sens':sens,'spec':spec})
    return rows

def agg(rows):
    buckets = defaultdict(list)
    for r in rows:
        buckets[(r['self_connect'], r['pct'])].append(r)
    out = []
    for (sc,pct), vals in sorted(buckets.items()):
        aucs=[v['auc'] for v in vals]
        sens=[v['sens'] for v in vals]
        specs=[v['spec'] for v in vals]
        n=len(vals)
        out.append({
            'self_connect': sc,
            'pct': pct,
            'n': n,
            'auc_mean': mean(aucs), 'auc_std': (pstdev(aucs) if n>1 else 0.0),
            'sens_mean': mean(sens), 'sens_std': (pstdev(sens) if n>1 else 0.0),
            'spec_mean': mean(specs), 'spec_std': (pstdev(specs) if n>1 else 0.0),
        })
    return out

def write_summary(rows, path='summary_800.csv'):
    hdr = ['self_connect','pct','n','auc_mean','auc_std','sens_mean','sens_std','spec_mean','spec_std']
    with open(path,'w') as f:
        f.write(','.join(hdr)+'\n')
        for r in rows:
            f.write(','.join(map(str,[r[h] for h in hdr]))+'\n')
    return path

def main():
    rows = load_rows()
    summary = agg(rows)
    print('self_connect,pct,n,auc_mean,auc_std,sens_mean,sens_std,spec_mean,spec_std')
    for r in summary:
        print(','.join(map(str,[r[k] for k in ['self_connect','pct','n','auc_mean','auc_std','sens_mean','sens_std','spec_mean','spec_std']])))
    out = write_summary(summary)
    print(f'Wrote {out}')

if __name__ == '__main__':
    main()

