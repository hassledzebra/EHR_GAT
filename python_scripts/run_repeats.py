import subprocess
import shlex
import time

SAMPLE_PCTS = [10, 30, 50, 70, 90]
SEEDS = [0, 1, 2]

PY = "/Library/Frameworks/Python.framework/Versions/3.12/bin/python3"

def run_one(pct: int, seed: int):
    cmd = f"{PY} -u test_sample_size_500.py --sample_pct {pct} --epochs 800 --device cpu --self_connect_diag --seed {seed}"
    print(f"\n=== Running pct={pct}, seed={seed} ===\n{cmd}")
    start = time.time()
    try:
        subprocess.run(shlex.split(cmd), check=True)
    except subprocess.CalledProcessError as e:
        print(f"Run failed: pct={pct}, seed={seed}, rc={e.returncode}")
    dur = time.time() - start
    print(f"Finished pct={pct}, seed={seed} in {dur/60:.1f} min")

def main():
    for pct in SAMPLE_PCTS:
        for seed in SEEDS:
            run_one(pct, seed)

if __name__ == '__main__':
    main()

