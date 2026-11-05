"""Orchestration ABI. All continuum equations live in the shared C++ kernels."""
import ctypes as C
from pathlib import Path
import numpy as np
FIELDS=('chi','gr','gt','Ar','At','Khat','Theta','Lambda','alpha','beta','B','D')
PARITY=np.array([1,1,1,1,1,1,1,-1,1,-1,-1,-1])
class Kernel:
    def __init__(self,library,mode=1,radius=20.,slicing_ratio=2.,separated=False):
        self.lib=C.CDLL(str(Path(library).resolve())); self.path=Path(library).resolve()
        a=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');v=C.c_void_p;i=C.c_int;d=C.c_double
        signatures={'spherical_free_radial_symbol':([v,d,a,a],i),'spherical_expand_chart':([i]+[a]*6,i),'spherical_analytic_normal':([v]+[a]*4,i),'spherical_create':([i,d,d,i],v),'spherical_destroy':([v],None),
                    'spherical_auxiliary':([v,i]+[a]*5,i),'spherical_error':([],C.c_char_p),'spherical_configuration':([v,a],i),
                    'spherical_initial':([v,i,a,a,a,a],i),'spherical_current_parts':([v,i]+[a]*9,i),
                    'spherical_scri':([v]+[a]*4+[d,d,a,a],i),'spherical_symbol':([v,d,a,a,a],i),'spherical_gauge_speeds':([v,d,a,a],i)}
        if hasattr(self.lib,'spherical_outer_reference_jets'):signatures['spherical_outer_reference_jets']=([v,i,a,a,a],i)
        for name,(args,result) in signatures.items():
            f=getattr(self.lib,name);f.argtypes=args;f.restype=result
        self.handle=self.lib.spherical_create(mode,radius,slicing_ratio,int(separated))
        if not self.handle:raise ValueError(self.lib.spherical_error().decode())
        self.config=np.zeros(13);self.check(self.lib.spherical_configuration(self.handle,self.config))
    def check(self,code):
        if code:raise FloatingPointError(self.lib.spherical_error().decode())
    def initial(self,r):
        r=np.ascontiguousarray(r);q=np.zeros((len(r),12));src=np.zeros((len(r),6));omega=np.zeros((len(r),3))
        self.check(self.lib.spherical_initial(self.handle,len(r),r,q,src,omega));return q,src,omega
    def parts(self,r,q,d,dd,source,diagnostics=True):
        q,d,dd=map(np.ascontiguousarray,(q,d,dd));R=np.zeros_like(q);S=np.zeros_like(q);trace=np.zeros_like(q);diag=np.zeros_like(q)
        self.check(self.lib.spherical_current_parts(self.handle,len(r),r,q,d,dd,source,R,S,trace,diag))
        return R,S,trace,diag
    def scri(self,q,d,dd,normal,h,penalty):
        rate=np.zeros(12);dependent=np.zeros(2)
        self.check(self.lib.spherical_scri(self.handle,q,d,dd,normal,h,penalty,rate,dependent));return rate,dependent
    def symbol(self,r,q,direction=(1.,0.,0.)):
        P=np.zeros((26,26));n=np.ascontiguousarray(direction);q=np.ascontiguousarray(q)
        self.check(self.lib.spherical_symbol(self.handle,float(r),q,n,P));return P
    def analytic_normal(self,q,d,dd):
        out=np.zeros(12);self.check(self.lib.spherical_analytic_normal(self.handle,q,d,dd,out));return out
    def expand_chart(self,q,d=None,dd=None):
        q=np.ascontiguousarray(q);d=np.zeros_like(q) if d is None else np.ascontiguousarray(d);dd=np.zeros_like(q) if dd is None else np.ascontiguousarray(dd)
        Q=np.zeros((len(q),12));D=np.zeros_like(Q);DD=np.zeros_like(Q)
        self.check(self.lib.spherical_expand_chart(len(q),q,d,dd,Q,D,DD));return Q,D,DD
    def free_radial_symbol(self,r,q):
        P=np.zeros((12,12));self.check(self.lib.spherical_free_radial_symbol(self.handle,r,np.ascontiguousarray(q),P));return P
    def gauge_speeds(self,r,q):
        out=np.zeros(4);self.check(self.lib.spherical_gauge_speeds(self.handle,r,np.ascontiguousarray(q),out));return out
    def outer_reference_jets(self,r):
        r=np.ascontiguousarray(r);d=np.zeros((len(r),12));dd=np.zeros_like(d)
        self.check(self.lib.spherical_outer_reference_jets(self.handle,len(r),r,d,dd));return d,dd
    def close(self):
        if self.handle:self.lib.spherical_destroy(self.handle);self.handle=None
