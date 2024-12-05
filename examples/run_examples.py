#!/usr/bin/env python
"""
Run all the Jupyter examples

Adapated from https://github.com/sciris/sciris/blob/main/docs/tutorials/run_tutorials
"""

import os
from glob import glob
import nbformat
import nbconvert.preprocessors as nbp

folder = '.'
timeout = 600


def execute(path):
    """ Executes a single Jupyter notebook and returns success/failure """
    try:
        with open(path) as f:
            print(f'Executing {path}...')
            nb = nbformat.read(f, as_version=4)
            ep = nbp.ExecutePreprocessor(timeout=timeout, kernel_name='python3')
            ep.preprocess(nb, {'metadata': {'path': os.path.dirname(path)}})
        return f'Success: {path}'
    except nbp.CellExecutionError as e:
        return f'Error for {path}: {str(e)}'
    except Exception as e:
        return f'Error processing {path}: {str(e)}'


def main():
    """ Executes the notebooks in parallel and prints the results """
    notebooks = sorted(glob('examples/*.ipynb'))
    results = []
    for notebook in notebooks:
        results.append(execute(notebook))
        
    return ''.join([res + '\n' for res in results])


if __name__ == '__main__':
    results = main()
    print(results)