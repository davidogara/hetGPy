# Contributing to hetGPy

Thanks for your interest in contributing to hetGPy! This document outlines rules for contributing.


## Development installation

Please install both the `requirements.txt` and `requirements_dev.txt` file
```bash
git clone https://github.com/davidogara/hetGPy.git
cd hetGPy
pip install -r requirements.txt
pip install -r requirements_dev.txt
pip install -e .
```



### Docstrings
This project uses [numpy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html)


### Testing

We use python's `pytest` module:
```bash
pytest
```



### Documentation

hetGPy uses sphinx to generate documentation, and ReadTheDocs to host documentation.
To build the documentation locally, ensure that sphinx and its plugins are properly installed (see `docs/requirements_docs.txt`).

Documentation can be built locally via:
```bash
cd docs 
make html
```


## Pull Requests

Pull requests and feedback are greatly appreciated.



## Issues

Please submit any issues via the issue tracker.

