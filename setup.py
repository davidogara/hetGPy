# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['hetgpy']

package_data = \
{'': ['*'], 'hetgpy': ['src/*']}

install_requires = \
['jax>=0.4.28,<0.5.0',
 'jaxlib>=0.4.28,<0.5.0',
 'joblib>=1.3.2,<2.0.0',
 'matplotlib>=3.9.1,<4.0.0',
 'numba>=0.60.0,<0.61.0',
 'numpy>=1.20.2,<2.0.0',
 'scipy==1.14.0',
 'tqdm>=4.66.4,<5.0.0']

setup_kwargs = {
    'name': 'hetgpy',
    'version': '0.1.0',
    'description': 'Python implementation of hetGP',
    'long_description': '## hetGPy: Heteroskedastic Gaussian Process Modeling in Python\n\n`hetGPy` is a Python implementation of the `hetGP` R library.\n\nThis package is designed to be a "pure" Python implementation of `hetGP`, with the goals of:\n*\tMatching the behavior of the `R` package\n*\tHaving minimal dependencies (i.e. mostly `numpy` and `scipy`)\n\nThe motivation for such a package is due to the rising popularity of implementing simulation models (also known as computer experiments) in Python. \n\n\n## Installing and Environments\n\n* `hetGPy` is not yet available as a compiled package on pypi, but you can build the package by doing the following:\n* At the moment, we use [poetry](https://python-poetry.org/) so dependencies can be installed with:\n\n1. Create a virtual environment. Assuming you have python3.10 installed, run:\n```\npython3.10 -m venv .venv\n```\n\n2. Check if `eigen` is installed as a submodule (note that you only have to do this once):\n```\ngit submodule update\n```\nYou should see the `eigen` subdirectory populate with many files.\n\n3. Install dependencies with [poetry](https://python-poetry.org/)\n```\npoetry install --no-root\npoetry build\n```\n\n4. Activate the virtual environment:\n```\npoetry shell\n```\n\n5. If you want to develop locally (and run the tests), run:\n```\npython setup_cpp.py\n```\n\n\nTo quickly check if the installation worked, try running:\n```\npytest tests/test_cov_gen.py\n```\n\nWhich tests some of the covariance functions.\n\nAfter this you should be able to run the examples in the `examples` folder. It is suggested to use:\n```\npoetry run jupyter lab\n```\n\nIf you wish to use `hetgpy` elsewhere, after running `poetry build` you should be able to install the `whl` file using pip.\n\n\n## Note on Dependencies\n*\t`hetGPy` requires `scipy>=1.14.0` which fixed a [memory leakage issue](https://github.com/scipy/scipy/issues/20768) when using `L-BFGS-B` in `scipy.optimize.minizmize`. That version of scipy requires Python 3.10. \n\n*\tSince `hetGPy` is designed for large-scale problems, this was chosen as a necessary feature. Experienced users may be able to roll back some of the dependencies, but this is not the recommended use.\n\n*\t`hetGPy` also requires a c++17 compiler for the underlying covariance functions.\n\n\n## Contact\nFor questions regarding this package, please contact:  \nDavid O\'Gara  \nDivision of Computational and Data Sciences, Washington University in St. Louis  \ndavid.ogara@wustl.edu\n\n## References\n\nBinois M, Gramacy RB (2021). “hetGP: Heteroskedastic Gaussian Process Modeling and Sequential Design in R.” _Journal of Statistical Software_,\n  *98*(13), 1-44. doi:10.18637/jss.v098.i13 <https://doi.org/10.18637/jss.v098.i13>',
    'author': "David O'Gara",
    'author_email': 'david.ogara@wustl.edu',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.10',
}
from build_hetgpy import *
build(setup_kwargs)

setup(**setup_kwargs)
