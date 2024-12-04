library(MASS)
library(hetGP)
library(yaml)
X = as.matrix(mcycle$times)
Z = mcycle$accel
xgrid = as.matrix(seq(0,60,length.out = 301))
model = mleHetGP(X = X, Z = Z, 
                 covtype = "Gaussian",
                 lower = 1, upper = 100,maxit=2e2,
                 settings = list(factr=10e7))
EIs = crit_EI(x = xgrid, model = model)
proposal = xgrid[which.max(EIs)]

out = list(
  proposal = proposal
)

cat(as.yaml(out),file = 'results/test_EI.yaml')