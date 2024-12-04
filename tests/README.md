## Testing Directory for hetGPy

Since `hetGPy` is a port of the `hetGP` R package, many of the tests require comparing 
the output between the Python and R packages.

Each `*.py` script has a corresponding `*.R` script (with the same name) in the `R` directory.

The results of the R script are saved under `R/results/`.