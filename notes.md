## Notes on the `hetGPy` package

2023-11-28:
*   Some early progress on general architecture (a model class, covariance functions, etc.)
*   Investigated feasibility of linking the Rcpp functions directly to the package, whether via `pybind11` or `boost`
*   Ultimately I am leaning away from that, since:  
    *   It appears that most of the distance functions can be expressed as matrix algebra operations
    *   It looks like some of the cpp functions were taken out of R because they don't have easy broadcasting operations (unlike in `numpy`)
    *   I should be able to `numba` all of the performance-critical functions
    *   I'm also wary of bogging down the package with a bunch of c++ code (since Rcpp is the standard in R, but Python has several ways to do it, so I'm worried about running into compiler issues)
    *   Finally, it seems like figuring out how to provide the `None` datatype to the C++ functions could be a lot of work, so the (likely incremental) performance gains may not be worth it.
*   Successfully able to pass the following tests:
    *   find_reps
    *   cov_Gaussian
    *   homoskedastic log likelihood