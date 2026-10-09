"""Numerically compare two complete runs without timestamp-sensitive figures."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('candidate', type=Path)
    args = parser.parse_args()
    count = 0
    for reference in sorted(args.reference.glob('*.csv')):
        candidate = args.candidate / reference.name
        a, b = pd.read_csv(reference), pd.read_csv(candidate)
        if list(a.columns) != list(b.columns) or a.shape != b.shape:
            raise AssertionError(f'Schema mismatch: {reference.name}')
        for column in a:
            if pd.api.types.is_numeric_dtype(a[column]) and pd.api.types.is_numeric_dtype(b[column]):
                np.testing.assert_allclose(a[column], b[column], rtol=1e-10, atol=1e-12, equal_nan=True,
                                           err_msg=f'{reference.name}:{column}')
            else:
                pd.testing.assert_series_equal(a[column], b[column], check_names=False,
                                               obj=f'{reference.name}:{column}')
        count += 1
    print(f'OK: all {count} CSV output files agree within rtol=1e-10, atol=1e-12.')


if __name__ == '__main__':
    main()
