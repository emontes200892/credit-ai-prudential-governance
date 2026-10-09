"""Portable entry point: python run_experiments.py --out results/my_run."""
import sys
import os
from pathlib import Path
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(variable, '1')
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from credit_ai.run import main

if __name__ == "__main__":
    main()
