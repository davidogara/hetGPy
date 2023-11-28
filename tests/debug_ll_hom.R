# test logLikeHom

source('hetGP.R')
source('Covariance_functions.R')
lapply(list.files(path = 'src',pattern = '*.cpp', full.names = T),sourceCpp)
X = mcycle$times %>% as.matrix()
Z = mcycle$accel

prdata = find_reps(X, Z)

out = logLikHom(
  X0 = prdata$X0,
  Z0 = prdata$Z0,
  Z = prdata$Z,
  beta0 = NULL,
  covtype = "Gaussian",
  mult = prdata$mult,
  theta = 0.5,
  g = 0.1
)


out.d = dlogLikHom(
  X0 = prdata$X0,
  Z0 = prdata$Z0,
  Z = prdata$Z,
  beta0 = NULL,
  covtype = "Gaussian",
  mult = prdata$mult,
  theta = 0.5,
  g = 0.1
)


prdata.ani = find_reps(
  X = mcycle$times[1:90] %>% matrix(nrow = 30,byrow = T),
  Z = mcycle$accel[1:30]
)


out.ani = logLikHom(
  X0 = prdata.ani$X0,
  Z0 = prdata.ani$Z0,
  Z = prdata.ani$Z,
  beta0 = NULL,
  covtype = "Gaussian",
  mult = prdata.ani$mult,
  theta = c(1,2,3),
  g = 0.1
)


out.ani.d = dlogLikHom(
  X0 = prdata.ani$X0,
  Z0 = prdata.ani$Z0,
  Z = prdata.ani$Z,
  beta0 = NULL,
  covtype = "Gaussian",
  mult = prdata.ani$mult,
  theta = c(1,2,3),
  g = 0.1
)

# test a big matrix inversion

xx = sample(1:1000, size = 12000, replace = T) %>% matrix(nrow = 4000)
C = cov_gen(xx, theta = 1)
tic = proc.time()
eps = sqrt(.Machine$double.eps)
Ki <- chol(C + diag(eps,nrow = nrow(C)))
Ki <- chol2inv(Ki)
toc = proc.time() - tic