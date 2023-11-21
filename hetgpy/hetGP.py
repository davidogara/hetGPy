import numpy as np


class hetGP:
    def __init__(self):
        return
    
    def find_reps(self,X,Z, return_Zlist = True, rescale = False, normalize = False, inputBounds = None):
        
        if type(X) != np.ndarray:
            raise ValueError(f"X must be a numpy array, is currently: {type(X)}")
        if X.shape[0] == 1:
            if return_Zlist:
                return dict(X0=X,Z0=Z,mult = 1, Z = Z, Zlist = dict(Z))
            return(dict(X0 = X, Z0 = Z, mult = 1, Z = Z))

        if rescale:
            if inputBounds is None:
                inputBounds = np.array([X.min(axis=0),
                                        X.max(axis=0)])
            X = (X - inputBounds[0,:]) @ np.diag(1/(inputBounds[1,:] - inputBounds[0,:]))
        outputStats = None
        if normalize:
            outputStats = np.array([Z.mean(), Z.var()])
            Z = (Z - outputStats[0])/np.sqrt(outputStats[1])
        X0 = np.unique(X, axis = 0)
        if X0.shape[0] == X.shape[0]:
            if return_Zlist:
                return dict(X0 = X, Z0 = Z, mult = np.repeat(1, len(Z)), Z = Z, Zlist = Z,
                  inputBounds = inputBounds, outputStats = outputStats)
            return dict(X0 = X, Z0 = Z, mult = np.repeat(1, len(Z)), Z = Z,
                inputBounds = inputBounds, outputStats = outputStats)
        
        # TODO: consider numba-ing this part. Replicating *split* in R is a bit tricky
        # consider something like: Zsplit = np.split(Z, np.unique(corresp, return_index=True)[1][1:])
        _, corresp = np.unique(X,axis=0,return_inverse=True)
        Zlist = {}
        Z0    = np.zeros(X0.shape[0], dtype=X0.dtype)
        mult  = np.zeros(X0.shape[0], dtype=X0.dtype)
        for ii in np.unique(corresp):
            out = Z[(ii==corresp).nonzero()[0]]
            Zlist[ii] = out
            Z0[ii]    = out.mean()
            mult[ii]  = len(out)
  
        if return_Zlist:
            return dict(X0 = X0, Z0 = Z0, mult = mult, Z = Z,
                Zlist = Zlist, inputBounds = inputBounds, outputStats = outputStats)
        return dict(X0 = X0, Z0 = Z0, mult = mult, Z = Z, inputBounds = inputBounds,
              outputStats = outputStats)