# test_matern_import

import cppimport
from scipy.io import loadmat
matern = cppimport.imp('matern')

m = loadmat('d_matern5_2_1args_theta_k.mat')
theta = 5.871215
tmp = matern.d_matern5_2_1args_theta_k(X1 = m['X1'],theta = theta)
foo=1