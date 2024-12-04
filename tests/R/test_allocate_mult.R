library(hetGP)
library(MASS)
library(yaml)
X <- matrix(mcycle$times, ncol = 1)
Z <- mcycle$accel
nvar <- 1
## Model fitting
data_m <- find_reps(X, Z, rescale = TRUE)

model <- mleHetGP(X = list(X0 = data_m$X0, Z0 = data_m$Z0, mult = data_m$mult),
                  Z = Z, lower = rep(0.1, nvar), upper = rep(5, nvar),
                  covtype = "Matern5_2")
## Compute best allocation                  
A <- allocate_mult(model, N = 1000)


out <- list(
theta  = model$theta,
g      = model$g,
Delta  = model$Delta,
Lambda = model$Lambda,
A = A)

cat(as.yaml(out),file = 'results/test_allocate_mult.yaml')
