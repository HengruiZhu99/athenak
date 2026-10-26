"""Compiled stage integration using exactly the supplied radial operators."""
import ctypes as C
import time
import numpy as np


class FastEngine:
    def __init__(self,evolution,fields,library):
        a=evolution.args;op=evolution.op
        if a.backend!='spherical-g1' or a.origin_variables or a.endpoint not in ('raw-lhopital','analytic-g1'):
            raise ValueError('Compiled engine implements unscaled g=1 and raw/analytic endpoints')
        self.lib=C.CDLL(str(library));self.calls=0;self.advance_seconds=0.
        fp=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
        self.lib.spherical_fast_create.argtypes=[C.c_int]*4+[fp,ip,fp,fp,ip,fp,fp]
        self.lib.spherical_fast_create.restype=C.c_void_p
        self.lib.spherical_fast_destroy.argtypes=[C.c_void_p]
        self.lib.spherical_fast_stats.argtypes=[C.c_void_p,fp]
        self.lib.spherical_fast_rhs.argtypes=[C.c_void_p,fp,fp,C.POINTER(C.c_double)]
        self.lib.spherical_fast_advance.argtypes=[C.c_void_p,fp,C.c_int,C.c_double,C.POINTER(C.c_double)]
        params=np.array([a.kappa1,a.kappa2,a.slicing,a.lapse_restoring,a.scri_sat,
            a.regularity_sat,a.origin_ko,a.origin_ko_radius,a.theta=='appendix-c',a.endpoint=='analytic-g1',getattr(a,'gauge_corner_sat',0.)],dtype=float)
        self.handle=self.lib.spherical_fast_create(len(op.r),fields,op.width,len(op.qindex[0]),
            op.r,np.asarray(op.index,dtype=np.int32),op.d1,op.d2,
            np.asarray(op.qindex,dtype=np.int32),op.qweights,params)
        if not self.handle:raise RuntimeError('Compiled engine initialization failed')

    def close(self):
        if self.handle:self.lib.spherical_fast_destroy(self.handle);self.handle=None

    def stats(self):
        values=np.zeros(14);self.lib.spherical_fast_stats(self.handle,values)
        return int(values[0]),float(values[1])

    def diagnostics(self):
        values=np.zeros(14);self.lib.spherical_fast_stats(self.handle,values)
        if values[0]==0:return {}
        names=['max_scri_pole','max_lapse_pole','max_regularity_trace','min_chi','min_alpha',
            'max_H_rms','max_M_rms','max_Theta_rms','max_Z_rms','max_scri_null','max_scri_clock']
        return {key:float(value) for key,value in zip(names,values[2:])}

    def rhs(self,q):
        out=np.empty_like(q);flux=C.c_double()
        if self.lib.spherical_fast_rhs(self.handle,q,out,C.byref(flux)):
            raise FloatingPointError('Compiled RHS rejected gauge/positivity/finiteness')
        self.calls+=1
        return out,flux.value

    def advance(self,q,steps,dt,energy=0.):
        flux=C.c_double(energy)
        start=time.perf_counter()
        done=self.lib.spherical_fast_advance(self.handle,q,steps,dt,C.byref(flux))
        self.advance_seconds+=time.perf_counter()-start
        completed=done if done>=0 else -done-1
        self.calls+=4*completed
        return completed,flux.value,done<0
