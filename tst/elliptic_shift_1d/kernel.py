"""Typed NumPy/C interface to the actual nonlinear spherical tensor restriction."""
import ctypes
from pathlib import Path
import numpy as np

FIELDS = ('chi','a','Arr','Khat','Theta','Lambda','alpha','beta')
PARITY = np.array([1,1,1,1,1,-1,1,-1])


class Kernel:
    def __init__(self, library):
        self.path = Path(library).resolve()
        self.lib = ctypes.CDLL(str(self.path))
        pointer = np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        self.lib.spherical_parts.argtypes = ([ctypes.c_int]+[pointer]*7
            +[ctypes.c_double,ctypes.c_double,ctypes.c_int,ctypes.c_double]+[pointer]*3)
        self.lib.spherical_parts.restype = ctypes.c_int
        self.lib.spherical_parts_total_chi.argtypes=self.lib.spherical_parts.argtypes
        self.lib.spherical_parts_total_chi.restype=ctypes.c_int

    def parts(self, r, q, d, dd, beta, dbeta, ddbeta, kappa1=4., kappa2=-.99,
              appendix_c=True, slicing=1., total_chi=None):
        args = [np.ascontiguousarray(a,dtype=np.float64) for a in (r,q,d,dd,beta,dbeta,ddbeta)]
        n=len(r)
        if any(a.shape!=(n,8) for a in args[1:4]) or any(a.shape!=(n,) for a in [args[0],*args[4:]]):
            raise ValueError('Kernel dimensions mismatch')
        R=np.zeros((n,8)); S=np.zeros((n,8)); diagnostic=np.zeros((n,12))
        function=self.lib.spherical_parts
        if total_chi is not None:
            args[1]=args[1].copy();args[1][:,0]=total_chi
            function=self.lib.spherical_parts_total_chi
        bad=function(n,*args,kappa1,kappa2,int(appendix_c),slicing,R,S,diagnostic)
        if bad: raise FloatingPointError(f'Invalid geometry/RHS at point {bad-1}, r={r[bad-1]}')
        return R,S,diagnostic
