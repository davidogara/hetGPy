# generate_data.R
# @author: David O'Gara
# part of hetGPy/notebooks/R

library(lhs)
library(R.matlab)
library(magrittr)
out.dir = 'data'
if (!dir.exists(out.dir)){
  dir.create(out.dir)
}
source('optimization_functions.R')

# f function
# d size
set.seed(1213)
make_design = function(N,d){
  X <- randomLHS(N, d)
  
  # choose replicates
  reps = sample(1:10, size = nrow(X), replace = T)
  X = X[rep(1:nrow(X),reps),]
  X
}


N = 1000
## 1. Branin
fn = paste0(out.dir,'/','Branin','.mat')
X  = make_design(N, d = 2)
Y  = apply(X,1,braninsc) + rnorm(n = nrow(X),mean = 0, sd = 2)
writeMat(con = fn, X = X, Y = Y)


## 2. Goldstein-Price
fn = paste0(out.dir,'/','Goldstein-Price','.mat')
X  = make_design(N, d = 2)
Y  = apply(X,1, goldprsc) + rnorm(n = nrow(X),mean = 0, sd = 2)
writeMat(con = fn, X = X, Y = Y)


## 3. Rosenbrock
fn = paste0(out.dir,'/','Rosenbrock','.mat')
X  = make_design(N, d = 4)
Y  = apply(X,1,rosensc) + rnorm(n = nrow(X),mean = 0, sd = 2)
writeMat(con = fn, X = X, Y = Y)

## 4. Hartman 4D
fn = paste0(out.dir,'/','Hartmann-4D','.mat')
X  = make_design(N, d = 4)
Y  = apply(X,1,hart4) + rnorm(n = nrow(X),mean = 0, sd = 2)
writeMat(con = fn, X = X, Y = Y)

## 5. Hartman 6D
fn = paste0(out.dir,'/','Hartmann-6D','.mat')
X  = make_design(N, d = 6)
Y  = apply(X,1,hart6sc) + rnorm(n = nrow(X),mean = 0, sd = 2)
writeMat(con = fn, X = X, Y = Y)

## 6. Sphere 6D
fn = paste0(out.dir,'/','Sphere-6D','.mat')
X  = make_design(N, d = 6)
Y  = apply(X,1,spherefmod) + rnorm(n = nrow(X),mean = 0, sd = 2)
writeMat(con = fn, X = X, Y = Y)

