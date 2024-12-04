library(R.matlab)
library(hetGP)
library(yaml)
m = readMat("../data/SIR.mat")
X = m[["X"]]
Z = m[["Y"]]
xseq  = seq(0,1,length.out = 100)
xgrid = as.matrix(expand.grid(xseq,xseq))

run_SIR <- function(ctype){
model = mleHetGP(
  X = X,
  Z = Z,
  covtype = ctype,
  lower = c(0.05,0.05),
  upper = c(2,2),
  maxit = 50
)
preds = predict(model,xgrid)
# predictive interval
preds$upper = qnorm(0.95, preds$mean, sqrt(preds$sd2 + preds$nugs)) 
preds$theta = model$theta
return(preds[c('mean','upper','theta')])
}

out = list(
  Matern5_2 = run_SIR("Matern5_2")
)

cat(as.yaml(out),file='results/test_hetGP_SIR.yaml')