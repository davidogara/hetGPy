from hetgpy import hetGP
import numpy as np
import pandas as pd

def test_find_reps_on_mcycle():

    model = hetGP.hetGP()
    mcycle = pd.read_csv('tests/data/mcycle.csv',index_col=False)
    X = mcycle['times'].values
    Z = mcycle['accel'].values
    test = model.find_reps(X,Z,rescale=False, return_Zlist=True,normalize=False,
                           inputBounds=None)
    # get data from R output
    X0   = pd.read_csv('tests/data/X0_no_normal_no_rescale.csv')['V1'].values
    Z0   = pd.read_csv('tests/data/Z0_no_normal_no_rescale.csv')['.'].values
    Z    = pd.read_csv('tests/data/Z_no_normal_no_rescale.csv')['.'].values
    mult = pd.read_csv('tests/data/mult_no_normal_no_rescale.csv')['.'].values
    assert np.allclose(X0,test['X0'])
    assert np.allclose(Z0,test['Z0'])
    assert np.allclose(Z,test['Z'])
    assert np.allclose(mult,test['mult'])

    
    Zlist = pd.read_csv('tests/data/Zlist_no_normal_no_rescale.csv')['.'].str.split('|').values
    # Zlist has a funky structure, so need to go row by row
    checkList = np.full(fill_value=False, dtype=bool,shape=len(Zlist))
    for zz in range(len(Zlist)):
        checkList[zz] = np.allclose(np.array(Zlist[zz]).astype(float), test['Zlist'][zz])
    assert checkList.mean()==1.0

if __name__ == "__main__":
    test_find_reps_on_mcycle()