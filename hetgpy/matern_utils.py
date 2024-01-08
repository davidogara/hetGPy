# matern_utils
import numpy as np

def d_matern5_2_1args_theta_k(X1, theta):
  # X1 has just one column here
  nr = X1.shape[0]
  s = np.zeros((nr, nr))
  
  
  ptrX1 = X1[1,0]
  ptrX2 = X1[0,0]
  ptrs = s[0,1]
  ptrs2 = s[1,0] # symmetric
  
  for i in range(1,nr): # i < nr i++, ptrX1++){
    for j in range(i):  # j++, ptrs++
      
      tmp = np.abs(ptrX1 - ptrX2) / theta
      ptrs -= ((10./3. - 5.) * tmp - 5 * np.sqrt(5.)/3. * tmp * tmp) / (1 + np.sqrt(5.) * tmp + 5./3. * tmp * tmp) * tmp/theta
      
      ptrs2 = ptrs
      ptrs2 += nr
      ptrX2 +=1
    ptrX1+=1
    ptrX2 -= i
    ptrs += (nr - i)
    ptrs2 += 1 - i*nr
  
  return s