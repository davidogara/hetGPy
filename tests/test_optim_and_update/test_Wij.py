# test Wij
from hetgpy.IMSE import Wij
import numpy as np
from rpy2.robjects import r
def test_Wij():
    '''
    From Gramacy 2020 Chapter 10 Figure 10.7 https://bookdown.org/rbg/surrogates/chap10.html#chap10imspe
    '''
    rn = np.array((6, 4, 5, 6.5, 5))
    X0 = np.linspace(0.2, stop = 0.8, num=len(rn)).reshape(-1,1)
    xx = np.arange(0,1.005,step=0.005)
    out = []
    for x in xx:
        Wijs = Wij(mu1=np.vstack((X0,x)), theta=np.array([0.25]), type="Gaussian")
        out.append(Wijs)
    r('''
    library(hetGP)
    rn <- c(6, 4, 5, 6.5, 5) 
    X0 <- matrix(seq(0.2, 0.8, length.out=length(rn)))
    xx <- matrix(seq(0, 1, by=0.005))
    
    out = list()
    i = 0
    for (x in xx){
    Wijs <- Wij(mu1=rbind(X0, x), theta=0.25, type="Gaussian")
    out[as.character(i)] <- list(Wijs)
    i = i + 1
    }
    ''')
    for key,w in enumerate(out):
        w_r = np.array(r(f'out["{key}"]'))
        assert np.allclose(w,w_r)

    return
if __name__ == "__main__":
    test_Wij()
    
    