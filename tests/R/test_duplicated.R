library(yaml)
# 1D example
x <- c(9:20, 1:5, 3:7, 0:8)

D1 = duplicated(x)
D1_L = duplicated(x,fromLast = T)
#2D example

X <- matrix(c(c(1,2,3),c(1,2,3),c(4,5,6)),nrow=3,byrow=T)

D2 = duplicated(X)
D2_L = duplicated(X,fromLast=T)

out  = list(
  x = x,
  X = X,
  D1 = D1,
  D1_L = D1_L,
  D2 = D2,
  D2_L = D2_L
)

cat(as.yaml(out),file = 'results/test_duplicated.yaml')