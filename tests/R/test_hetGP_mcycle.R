library(MASS)
library(hetGP)
library(yaml)
X = as.matrix(mcycle$times)
Z = mcycle$accel
xgrid = as.matrix(seq(0,60,length.out = 301))
model_gauss = mleHetGP(X = X, Z = Z, 
                 covtype = "Gaussian",
                 lower = 1, upper = 100,maxit=2e2,
                 settings = list(factr=10e7))
preds_gauss = predict(model_gauss,xgrid)
# predictive interval
preds_gauss$upper = qnorm(0.95, preds_gauss$mean, sqrt(preds_gauss$sd2 + preds_gauss$nugs)) 
preds_gauss$theta = model_gauss$theta

model_matern = mleHetGP(X = X, Z = Z, 
                       covtype = "Matern5_2",
                       lower = 1, upper = 100,maxit=2e2,
                       settings = list(factr=10e7))
preds_matern = predict(model_matern,xgrid)
# predictive interval
preds_matern$upper = qnorm(0.95, preds_matern$mean, sqrt(preds_matern$sd2 + preds_matern$nugs)) 
preds_matern$theta = model_matern$theta

out = list(
  Gaussian = preds_gauss[c("mean","upper","theta")],
  Matern5_2 = preds_matern[c("mean","upper","theta")]
)
cat(as.yaml(out),file='results/test_hetGP_mcycle.yaml')

