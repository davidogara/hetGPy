library(hetGP)
library(R.matlab)
library(yaml)
m = readMat('../data/SIR.mat')
X = m[['X']]
Z = m[['Y']]

model = mleHomGP(
X = X, 
Z = Z, 
covtype = "Matern5_2",
lower   = c(0.05, 0.05), # original paper used Matern5_2
upper   = c(10, 10),
maxit   = 1e4
)

xgrid = seq(0,1,length.out=10)
Xgrid = as.matrix(expand.grid(xgrid,xgrid))
p = predict(model,Xgrid)

out = list(
  theta = model$theta,
  g = model$g,
  mean = p$mean,
  sd2 = p$sd2
)

cat(as.yaml(out),file='results/test_homGP_SIR.yaml')