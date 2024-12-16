"""
docstring for find_reps
"""

from __future__ import annotations
import numpy as np
def find_reps(X,Z, return_Zlist = True, rescale = False, normalize = False, inputBounds = None, use_torch=False):
        """Prepare data for use with ``mleHomGP`` and ``mleHetGP``
        
        In particular to find replicated observations

        Parameters
        ----------

        X : array_like
            matrix of design locations, one point per row
        Z : array_like
            vector of observations at ``X``
        return_Zlist: bool 
            to return `Zlist`, see below
        
        rescale: bool
            if ``True``, the inputs are rescaled to the unit hypercube
        normalize: bool 
            if ``True``, the outputs are centered and normalized
        inputBounds: array_like 
            optional matrix of known boundaries in original input space, of size 2 times `X.shape[1]`. If not provided, and ``rescale == True``, it is estimated from the data.   
        
        Returns
        -------
        dict
            dictionary of outputs
        type
            explain types
        out
            dictionary of 

                - ``X0`` matrix with unique designs locations, one point per row
                - ``Z0`` vector of averaged observations at ``X0``
                - ``mult`` number of replicates at ``X0``                
                - ``Z`` vector with all observations, sorted according to ``X0``
                - ``Zlist`` optional list, each element corresponds to observations at a design in ``X0``
                - ``inputBounds`` optional matrix, to rescale back to the original input space
                -  ``outputStats`` optional vector, with mean and variance of the original outputs.
        
        References
        ----------

        Binois et. al (2018)


        Examples
        --------
        >>> from hetgpy.data import mcycle
        >>> from hetgpy.hetGP import hetGP
        >>> X = mcycle['times']
        >>> Z = mcycle['accel']

        >>> model = hetGP()
        >>> out = model.find_reps(X, Z)
        >>> print(out)
        """
        if use_torch:
            raise NotImplementedError("torch implementation is not yet available.")
        if type(X) != np.ndarray:
            raise ValueError(f"X must be a numpy array, is currently: {type(X)}")
        if type(Z) != np.ndarray:
            raise ValueError(f"X must be a numpy array, is currently: {type(Z)}")
        if len(Z.shape)>1:
            if Z.shape[1]>1: raise ValueError(f"Z must be 1D")
            Z = Z.reshape(-1)
        if X.shape[0] == 1:
            if return_Zlist:
                return dict(X0=X,Z0=Z,mult = np.array([1]), Z = Z, Zlist = {0:Z})
            return(dict(X0 = X, Z0 = Z, mult = 1, Z = Z))
        if len(X.shape) == 1: # if x is a 1D series
            raise ValueError(f"X appears to be a 1D array. Suggest reshaping with X.reshape(-1,1)")
        if rescale:
            if inputBounds is None:
                inputBounds = np.array([X.min(axis=0),
                                        X.max(axis=0)])
            X = (X - inputBounds[0,:]) @ np.diag(1/(inputBounds[1,:] - inputBounds[0,:]))
        outputStats = None
        if normalize:
            outputStats = np.array([Z.mean(), Z.var()])
            Z = (Z - outputStats[0])/np.sqrt(outputStats[1])
        #X0 = np.unique(X, axis = 0)
        _, indices, corresp = np.unique(X, axis = 0, return_index=True, return_inverse=True)
        # np.unique sorts by default -- explicit sorting of the *indices* preserves the order of observations in X
        indices.sort() 
        X0 = X[indices,:]
        if X0.shape[0] == X.shape[0]:
            if return_Zlist:
                return dict(X0 = X, Z0 = Z, mult = np.repeat(1, len(Z)), Z = Z, Zlist = {k:Z[k].reshape(-1) for k in range(len(Z))},
                  inputBounds = inputBounds, outputStats = outputStats)
            return dict(X0 = X, Z0 = Z, mult = np.repeat(1, len(Z)), Z = Z,
                inputBounds = inputBounds, outputStats = outputStats)
        
        # TODO: consider numba-ing this part. Replicating *split* in R is a bit tricky
        Zlist = {}
        Z0    = np.zeros(X0.shape[0], dtype=X0.dtype)
        mult  = np.zeros(X0.shape[0], dtype=int)
        for idx, val in enumerate(corresp[indices]): 
            out = Z[(val==corresp).nonzero()[0]]
            Zlist[idx] = out.reshape(-1)
            Z0[idx]    = out.mean()
            mult[idx]  = len(out)
        if return_Zlist:
            return dict(X0 = X0, Z0 = Z0, mult = mult, Z = np.concatenate(list(Zlist.values())),
                Zlist = Zlist, inputBounds = inputBounds, outputStats = outputStats)
        return dict(X0 = X0, Z0 = Z0, mult = mult, Z = np.concatenate(list(Zlist.values())), inputBounds = inputBounds,
              outputStats = outputStats)
    