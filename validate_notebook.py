"""Execute notebook code cells in-process when socket-based kernels are unavailable.

Needs requirements-dev.txt. Usage: python validate_notebook.py
This validates sequential cell execution, not a live Jupyter server.
"""
import os
from pathlib import Path
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output


def main():
    path = Path(__file__).resolve().parent / 'notebooks/Explorar_reproducibilidad.ipynb'
    notebook = nbformat.read(path, as_version=4)
    shell = InteractiveShell.instance()
    old_directory = Path.cwd()
    count = 0
    try:
        os.chdir(path.parent)
        for cell in notebook.cells:
            if cell.cell_type != 'code':
                continue
            count += 1
            with capture_output(stdout=True, stderr=True, display=True) as captured:
                result = shell.run_cell(cell.source, store_history=False)
            if result.error_before_exec or result.error_in_exec:
                raise RuntimeError(f'Cell {count} failed') from (result.error_before_exec or result.error_in_exec)
            outputs = []
            if captured.stdout:
                outputs.append(nbformat.v4.new_output('stream', name='stdout', text=captured.stdout))
            if captured.stderr:
                outputs.append(nbformat.v4.new_output('stream', name='stderr', text=captured.stderr))
            for rich in captured.outputs:
                outputs.append(nbformat.v4.new_output('display_data', data=rich.data,
                                                      metadata=rich.metadata or {}))
            cell.outputs, cell.execution_count = outputs, count
        notebook.metadata['validation'] = {
            'method': 'sequential in-process IPython execution', 'code_cells': count,
            'all_cells_passed': True, 'live_jupyter_kernel_tested': False}
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
    finally:
        os.chdir(old_directory)
    print(f'OK: {count} notebook code cells executed in-process, zero errors.')


if __name__ == '__main__':
    main()
