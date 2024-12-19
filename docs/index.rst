.. hetGPy documentation master file, created by
   sphinx-quickstart on Wed Jan 31 10:46:22 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to hetGPy's documentation!
==================================

This landing page contains links to a set of example notebooks and the API reference.


Statement of Need
------------------

`hetGPy` is a Python package for heteroskedastic Gaussian Process modeling. It is a Python port of the `hetGP <https://cran.r-project.org/web/packages/hetGP/index.html>` R package.

`hetGPy` can be used for Gaussian Process regression, surrogate (emulator) modeling, and Bayesian Optimization. 

The primary audience is anyone who uses Gaussian Process modeling, but is especially designed for modeling computer experiments where models exhibit heteroskedasticity (i.e. having a non-constant noise structure).

Python is a popular language for computer experiments and simulation. Thus, we hope `hetGPy` will be a valuable addition to the Python and simulation communities.

For questions, please contact:
David O'Gara  
Division of Computational and Data Sciences, Washington University in St. Louis  
david.ogara@wustl.edu

Installation
------------

`hetGPy` is available via `PyPI <https://pypi.org/project/hetgpy/>`(``pip install hetgpy```)


Or from `Github <https://github.com/davidogara/hetGPy>``.




.. toctree::
   :maxdepth: 2
   :caption: Examples:

   examples/index

.. toctree::
   :maxdepth: 1
   :caption: Reference:

   api/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
