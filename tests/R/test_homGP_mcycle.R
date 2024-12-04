library(hetGP)
library(MASS)
library(yaml)
hom <- mleHomGP(mcycle$times, mcycle$accel, covtype = "Gaussian",
                maxit = 100, settings = list('factr'=10))

Xgrid <- matrix(seq(0, 60, length = 301), ncol = 1)
p <- predict(x = Xgrid, object = hom)

## motopred
out = list(
# fit
theta = hom$theta, 
g = hom$g,
# preds
mean = p$mean,
sd2 = p$sd2)

cat(as.yaml(out),file='results/test_homGP_mcycle.yaml')
