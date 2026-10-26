"""Compiled stage integration using exactly the supplied radial operators."""
import ctypes as C
import time
import numpy as np


class FastEngine:
    def __init__(self,evolution,fields,library):
        a=evolution.args;op=evolution.op;self.ode_method=hasattr(op,'shift_ode_solver')
        self.puncture=getattr(a,'puncture',False)
        if (getattr(a,'puncture_chart',False) or getattr(a,'radial_operator','fd')!='fd'
            or getattr(a,'lapse_variable','deviation') not in ('deviation','log-reference')
            or getattr(a,'metric_variable','deviation') not in ('deviation','log')
            or getattr(a,'slicing_family','linear') not in ('linear','rational','blended-log') or getattr(a,'lapse_background_source','reference') not in ('reference','collapsed-inner')
            or getattr(a,'puncture_boundary','outflow')!='outflow' or getattr(a,'puncture_ko',0.)):
            raise ValueError('This operator/gauge/boundary prototype is not implemented in the compiled engine')
        if (getattr(a,'metric_variable','deviation')=='log' or getattr(a,'lapse_variable','deviation')=='log-reference') and getattr(a,'advection','centered')!='centered':
            raise ValueError('Compiled positive variables currently use centered transport')
        if self.puncture and a.origin_ko:raise ValueError('Regular-center KO is inapplicable to a puncture')
        if a.backend!='spherical-g1' or a.origin_variables or a.endpoint not in ('raw-lhopital','analytic-g1'):
            raise ValueError('Compiled engine implements unscaled g=1 and raw/analytic endpoints')
        self.lib=C.CDLL(str(library));self.calls=0;self.advance_seconds=0.
        self.lib.spherical_fast_abi.restype=C.c_int
        if self.lib.spherical_fast_abi()!=24:raise ValueError('Compiled engine ABI does not match this wrapper')
        self.lib.spherical_fast_last_error.argtypes=[C.c_void_p];self.lib.spherical_fast_last_error.restype=C.c_char_p
        self.failure_message=''
        fp=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
        self.lib.spherical_fast_create.argtypes=[C.c_int]*4+[fp,ip,fp,fp,ip,fp,ip,fp,ip,fp,fp]
        self.lib.spherical_fast_create.restype=C.c_void_p
        self.lib.spherical_fast_destroy.argtypes=[C.c_void_p]
        self.lib.spherical_fast_stats.argtypes=[C.c_void_p,fp]
        self.lib.spherical_fast_set_lapse_source.argtypes=[C.c_void_p,fp]
        self.lib.spherical_fast_set_shift_ode.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_double,fp,fp]
        self.lib.spherical_fast_rhs.argtypes=[C.c_void_p,fp,fp,C.POINTER(C.c_double)]
        self.lib.spherical_fast_advance.argtypes=[C.c_void_p,fp,C.c_int,C.c_double,C.POINTER(C.c_double)]
        params=np.array([a.kappa1,a.kappa2,a.slicing,a.lapse_restoring,a.scri_sat,
            a.regularity_sat,a.origin_ko,a.origin_ko_radius,a.theta=='appendix-c',a.endpoint=='analytic-g1',getattr(a,'gauge_corner_sat',0.),getattr(a,'advection','centered')=='upwind',getattr(a,'puncture',False),getattr(a,'lapse_restoring_form','symmetric')=='multiplicative',getattr(a,'initial_data',None) in ('wormhole','trumpet'),getattr(a,'damping_form','constant')=='reference-lapse',getattr(a,'metric_variable','deviation')=='log',getattr(a,'lapse_variable','deviation')=='log-reference',{'linear':0,'rational':1,'blended-log':2}[getattr(a,'slicing_family','linear')],getattr(a,'slicing_scale',.1),getattr(a,'elliptic_quadrature','prefix')=='inward',getattr(a,'lapse_background_source','reference')=='collapsed-inner',getattr(a,'lapse_restoring_profile','constant')=='physical-log'],dtype=float)
        self.handle=self.lib.spherical_fast_create(len(op.r),fields,op.width,len(op.qindex[0]),
            op.r,np.asarray(op.index,dtype=np.int32),op.d1,op.d2,
            np.asarray(op.qindex,dtype=np.int32),op.qweights,
            np.asarray(op.aindex[0],dtype=np.int32),op.ad1[0],np.asarray(op.aindex[1],dtype=np.int32),op.ad1[1],params)
        if not self.handle:raise RuntimeError('Compiled engine initialization failed')
        self.lib.spherical_fast_set_lapse_source(self.handle,np.ascontiguousarray(evolution.lapse_source,dtype=float))
        if hasattr(op,'shift_ode_solver'):
            solver=op.shift_ode_solver
            self.lib.spherical_fast_set_shift_ode(self.handle,solver.bandwidth,solver.skip,solver.refinements,solver.scale,solver.lu,solver.matrix)

    def close(self):
        if self.handle:self.lib.spherical_fast_destroy(self.handle);self.handle=None

    def stats(self):
        values=np.zeros(24);self.lib.spherical_fast_stats(self.handle,values)
        return int(values[0]),float(values[1])

    def diagnostics(self):
        values=np.zeros(24);self.lib.spherical_fast_stats(self.handle,values)
        if values[0]==0:return {}
        names=['max_scri_pole','max_lapse_pole','max_regularity_trace','min_chi','min_alpha',
            'max_H_rms','max_M_rms','max_Theta_rms','max_Z_rms','max_scri_null','max_scri_clock','max_inner_characteristic','max_Bondi_nonaffinity',
            'max_accepted_H_rms','max_accepted_M_rms','max_accepted_Theta_rms','max_accepted_Z_rms',
            'max_accepted_H_bulk','max_accepted_M_bulk','max_accepted_H_endpoint','max_accepted_M_endpoint']
        result={key:float(value) for key,value in zip(names,values[2:])}
        if getattr(self,'ode_method',False):result['max_elliptic_discrete_residual']=float(values[-1])
        if not self.puncture:result.pop('max_inner_characteristic')
        return result

    def rhs(self,q):
        out=np.empty_like(q);flux=C.c_double()
        if self.lib.spherical_fast_rhs(self.handle,q,out,C.byref(flux)):
            raise FloatingPointError(self.lib.spherical_fast_last_error(self.handle).decode())
        self.calls+=1
        return out,flux.value

    def advance(self,q,steps,dt,energy=0.):
        flux=C.c_double(energy)
        start=time.perf_counter()
        done=self.lib.spherical_fast_advance(self.handle,q,steps,dt,C.byref(flux))
        self.advance_seconds+=time.perf_counter()-start
        completed=done if done>=0 else -done-1
        if done<0:self.failure_message=self.lib.spherical_fast_last_error(self.handle).decode()
        self.calls+=4*completed
        return completed,flux.value,done<0
