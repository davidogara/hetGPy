library(hetGP)
library(MASS)
library(R.matlab)
library(yaml)

mcycle_prep <- find_reps(
  X = as.matrix(mcycle$times),
  Z = mcycle$accel
)



SIR = readMat('../data/SIR.mat')

X <- SIR[['X']][1:200,]
Z <- SIR[['Y']][1:200]

SIR_prep <- find_reps(X,Z)

out = list(
  mcycle = mcycle_prep,
  SIR = SIR_prep
)

cat(as.yaml(out),file='results/test_find_reps.yaml')