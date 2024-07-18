## hetGPy: Heteroskedastic Gaussian Process Modeling in Python

`hetGPy` is a Python implementation of the `hetGP` R library.

This package is designed to be a "pure" Python implementation of `hetGP`, with the goals of:
*	Matching the behavior of the `R` package
*	Having minimal dependencies (i.e. mostly `numpy` and `scipy`)

The motivation for such a package is due to the rising popularity of implementing simulation models (also known as computer experiments) in Python. 


## Installing and Environments

* `hetGPy` is not yet available as a compiled package on pypi, but you can build the package by doing the following:
* At the moment, we use [poetry](https://python-poetry.org/) so dependencies can be installed with:

1. Create a virtual environment. Assuming you have python3.10 installed, run:
```
python3.10 -m venv .venv
```

2. Check if `eigen` is installed as a submodule (note that you only have to do this once):
```
git submodule update
```
You should see the `eigen` subdirectory populate with many files.

3. Install dependencies with [poetry](https://python-poetry.org/)
```
poetry install --no-root
poetry build
```

4. Activate the virtual environment:
```
poetry shell
```

5. If you want to develop locally (and run the tests), run:
```
python setup_cpp.py
```


To quickly check if the installation worked, try running:
```
pytest tests/test_cov_gen.py
```

Which tests some of the covariance functions.

After this you should be able to run the examples in the `examples` folder. It is suggested to use:
```
poetry run jupyter lab
```

If you wish to use `hetgpy` elsewhere, after running `poetry build` you should be able to install the `whl` file using pip.


## Note on Dependencies
*	`hetGPy` requires `scipy>=1.14.0` which fixed a [memory leakage issue](https://github.com/scipy/scipy/issues/20768) when using `L-BFGS-B` in `scipy.optimize.minizmize`. That version of scipy requires Python 3.10. 

*	Since `hetGPy` is designed for large-scale problems, this was chosen as a necessary feature. Experienced users may be able to roll back some of the dependencies, but this is not the recommended use.

*	`hetGPy` also requires a c++17 compiler for the underlying covariance functions.


## Contact
For questions regarding this package, please contact:  
David O'Gara  
Division of Computational and Data Sciences, Washington University in St. Louis  
david.ogara@wustl.edu

## References

Binois M, Gramacy RB (2021). “hetGP: Heteroskedastic Gaussian Process Modeling and Sequential Design in R.” _Journal of Statistical Software_,
  *98*(13), 1-44. doi:10.18637/jss.v098.i13 <https://doi.org/10.18637/jss.v098.i13>