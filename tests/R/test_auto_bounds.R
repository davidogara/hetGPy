library(hetGP)
library(yaml)
X <- matrix(c(1, 34,
       30,  3,
       45, 11,
       13, 10,
       17,  9),byrow=T,nrow=5)
    
auto_bounds <- function(X, min_cor = 0.01, max_cor = 0.5, covtype = "Gaussian", p = 0.05){
Xsc <- find_reps(X, rep(1, nrow(X)), rescale = T) # rescaled distances
# sub in distance function
dists <- plgp::distance(Xsc$X0) # find 2 closest points
repr_low_dist <- quantile(x = dists[lower.tri(dists)], probs = p) # (quantile on squared Euclidean distances)
repr_lar_dist <- quantile(x = dists[lower.tri(dists)], probs = 1-p)

if(covtype == "Gaussian"){
theta_min <- - repr_low_dist/log(min_cor)
theta_max <- - repr_lar_dist/log(max_cor)
return(list(lower = theta_min * (Xsc$inputBounds[2,] - Xsc$inputBounds[1,])^2,
            upper = theta_max * (Xsc$inputBounds[2,] - Xsc$inputBounds[1,])^2))
}else{
tmpfun <- function(theta, repr_dist, covtype, value){
  cov_gen(matrix(sqrt(repr_dist/ncol(X)), ncol = ncol(X)), matrix(0, ncol = ncol(X)), type = covtype, theta = theta) - value
}
theta_min <- uniroot(tmpfun, interval = c(sqrt(.Machine$double.eps), 100), covtype = covtype, value = min_cor, 
                     repr_dist = repr_low_dist, tol = sqrt(.Machine$double.eps))$root
theta_max <- uniroot(tmpfun, interval = c(sqrt(.Machine$double.eps), 100), covtype = covtype, value = max_cor,
                     repr_dist = repr_lar_dist, tol = sqrt(.Machine$double.eps))$root
return(list(lower = theta_min * (Xsc$inputBounds[2,] - Xsc$inputBounds[1,]),
            upper = max(1, theta_max) * (Xsc$inputBounds[2,] - Xsc$inputBounds[1,])))
}
} 

out <- list(
  Gaussian = auto_bounds(X,covtype = "Gaussian"),
  Matern3_2 = auto_bounds(X,covtype = "Matern3_2"),
  Matern5_2 = auto_bounds(X,covtype = "Matern5_2")
)
cat(as.yaml(out),file = 'results/test_auto_bounds.yaml')
