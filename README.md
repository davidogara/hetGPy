## hetGPy

Python implementation of the `hetGP` R library.

This package is designed to be a "pure" Python implementation of `hetGP`, with the goals of:
*	Matching the behavior of the `R` package
*	Having minimal dependencies (i.e. `numpy` and `scipy`)

The motivation for such a package is due to the rising popularity of implementing simulation models (also known as computer experiments) in Python. 

For project progress, please see `TODO.md`

## Installing and Environments

* `hetGPy` is not yet available as a compiled package, but you can build the package by doing the following:
* At the moment, we use [poetry](https://python-poetry.org/) so dependencies can be installed with:

1. Install dependencies with [poetry](https://python-poetry.org/)
```
poetry install
poetry build
```
2. Activate the virtual environment:
```
poetry shell
```

3. Compile the `c++` files that underlie the kernel functions:

```
python setup_cpp.py
```

Note that you may need to run `poetry install` or `poetry build` a second time.

To quickly check if the installation worked, try running:
```
pytest tests/test_cov_gen.py
```

Which tests some of the covariance functions.

After this you should be able to run the examples in the `examples` folder. It is suggested to use:
```
poetry run jupyter lab
```

## Contact
For questions regarding this package, please contact:  
David O'Gara  
Division of Computational and Data Sciences, Washington University in St. Louis  
david.ogara@wustl.edu

## References

Binois M, Gramacy RB (2021). “hetGP: Heteroskedastic Gaussian Process Modeling and Sequential Design in R.” _Journal of Statistical Software_,
  *98*(13), 1-44. doi:10.18637/jss.v098.i13 <https://doi.org/10.18637/jss.v098.i13>