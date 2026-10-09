"""Verify all bundled files offline: python verify_package.py."""
import hashlib
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    count, errors = 0, []
    for line in (root / 'CHECKSUMS.sha256').read_text().splitlines():
        expected, name = line.split('  ', 1)
        path = root / name
        if not path.is_file():
            errors.append(f'Missing: {name}')
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(f'Changed: {name}')
        count += 1
    if errors:
        raise SystemExit('\n'.join(errors))
    print(f'OK: {count} bundled file checksums verified.')


if __name__ == '__main__':
    main()
