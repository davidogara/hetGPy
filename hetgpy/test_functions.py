'''
Suite of test functions (used in documentation and examples)

'''
import numpy as np
from datetime import datetime

def sirEval(x,seed=None):
  '''Calls the sirSimulate function'''
  return sirSimulate(S0 = x[:,0] * 600 + 1200, I0 = 200 * x[:,1], M = 2000, beta = 0.5, imm = 0,seed=seed)['totI']/800

def sirSimulate(S0 = 1990, I0 = 10, M = 2000, beta = 0.75, gamma = 0.5, imm = 0, seed = None):
  r'''
  Stochastic simulation of an SIR model via the Gillespie algorithm

  For details see Hu and Ludkovski (2017): https://arxiv.org/abs/1509.00980
  '''
  if seed is None: seed = int(datetime.today().timestamp())
  rand = np.random.default_rng(seed)
  curS = np.repeat(S0,M); curI = np.repeat(I0,M); curT  = np.repeat(0,M)
  curS[0] = S0
  curI[0] = I0
  curT[0] = 0
  count = 0
  maxI = I0
  
  # continue until no more infecteds
  while ( (curI[count] >0 & imm==0) | (curT[count] < 100 & imm > 0) ):
    
    # gillespie SSA algorithm: 2 reactions possible
    infRate = beta*curS[count]/(M)*(curI[count]+imm)
    recRate = gamma*curI[count]
    infTime = -1/infRate*np.log( rand.uniform())
    recTime = -1/recRate*np.log( rand.uniform())
    
    if (infTime < recTime):   # infection
      curS[count+1] = curS[count] - 1
      curI[count+1] = curI[count] + 1
      maxI = max(maxI, curI[count+1])
    else:    # recovery
      curI[count+1] = curI[count] - 1
      curS[count+1] = curS[count]
    curT[count+1] = curT[count] + min( infTime,recTime)
    count = count+1
  return(dict(maxI = maxI, totT = curT[count], totI = S0-curS[count],S=curS,I=curI,R=M-curS-curI,T=curT))


def f1d(x):
  '''
  A one-dimensional test function

  References
  ----------
  A. Forrester, A. Sobester, A. Keane (2008), Engineering design via surrogate modelling: a practical guide, John Wiley & Sons
  '''
  if len(x.shape)==1: x = x.reshape(-1,1)
  return np.squeeze(((x*6-2)**2)*np.sin((x*6-2)*2))

def f1d2(x):
  '''
  A one-dimensional test function

  References
  ----------
  A. Boukouvalas, and D. Cornford (2009), Learning heteroscedastic Gaussian processes for complex datasets, Technical report.

  M. Yuan, and G. Wahba (2004), Doubly penalized likelihood estimator in heteroscedastic regression, Statistics and Probability Letters 69, 11-20.
  '''
  return 2*(np.exp(-30*(x - 0.25)**2) + np.sin(x**2)) - 2