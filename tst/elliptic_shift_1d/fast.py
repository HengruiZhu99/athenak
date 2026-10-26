"""Compiled stage integration using exactly the supplied radial operators."""
import ctypes as C
import time
import numpy as np


class FastEngine:
    def __init__(self,evolution,fields,library):
        a=evolution.args;op=evolution.op;self.ode_method=hasattr(op,'shift_ode_solver')
        self.puncture=getattr(a,'puncture',False)
        self.shape=(len(op.r),fields)
        self.af=getattr(a,'puncture_chart',False)
        self.af_endpoint=self.af and op.r[0]==0
        self.trumpet_metric=getattr(a,'metric_variable','deviation') in ('log-trumpet','log-inner-trumpet')
        if (getattr(a,'radial_operator','fd')!='fd'
            or getattr(a,'lapse_variable','deviation') not in ('deviation','log-reference')
            or getattr(a,'metric_variable','deviation') not in ('deviation','log','log-trumpet','log-inner-trumpet')
            or getattr(a,'slicing_family','linear') not in ('linear','rational','blended-log') or getattr(a,'lapse_background_source','reference') not in ('reference','collapsed-inner')
            or getattr(a,'puncture_boundary','outflow')!='outflow' or (getattr(a,'puncture_ko',0.) and not (self.af or (self.puncture and op.r[0]>0)))):
            raise ValueError('This operator/gauge/boundary prototype is not implemented in the compiled engine')
        if getattr(a,'lapse_restoring_profile','constant')=='physical-local' and (getattr(a,'slicing_family','linear') not in ('linear','rational') or getattr(a,'lapse_reference','minkowski')!='minkowski'):raise ValueError('Unsupported local restoring source')
        if self.trumpet_metric and (not self.puncture or op.r[0]<=0):raise ValueError('Trumpet metric chart requires positive physical cutoff')
        if self.af and (fields!=8 or self.ode_method or getattr(a,'elliptic_quadrature','prefix')!='prefix' or getattr(a,'lapse_restoring_form','symmetric')!='multiplicative'):raise ValueError('Compiled AF uses vacuum quadrature shift')
        if self.puncture and a.origin_ko:raise ValueError('Regular-center KO is inapplicable to a puncture')
        if a.backend!='spherical-g1' or a.origin_variables or a.endpoint not in ('raw-lhopital','analytic-g1'):
            raise ValueError('Compiled engine implements unscaled g=1 and raw/analytic endpoints')
        self.lib=C.CDLL(str(library));self.calls=0;self.advance_seconds=0.
        self.lib.spherical_fast_abi.restype=C.c_int
        if self.lib.spherical_fast_abi()!=32:raise ValueError('Compiled engine ABI does not match this wrapper')
        self.lib.spherical_fast_last_error.argtypes=[C.c_void_p];self.lib.spherical_fast_last_error.restype=C.c_char_p
        self.failure_message=''
        fp=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
        self.lib.spherical_fast_create.argtypes=[C.c_int]*4+[fp,ip,fp,fp,ip,fp,ip,fp,ip,fp,fp]
        self.lib.spherical_fast_create.restype=C.c_void_p
        self.lib.spherical_fast_destroy.argtypes=[C.c_void_p]
        self.lib.spherical_fast_stats.argtypes=[C.c_void_p,fp]
        self.lib.spherical_fast_set_lapse_source.argtypes=[C.c_void_p,fp]
        self.lib.spherical_fast_set_af_reference.argtypes=[C.c_void_p,fp]
        self.lib.spherical_fast_set_metric_reference.argtypes=[C.c_void_p,fp]
        self.lib.spherical_fast_set_update_compensation.argtypes=[C.c_void_p,fp]
        self.lib.spherical_fast_get_update_compensation.argtypes=[C.c_void_p,fp]
        self.lib.spherical_fast_set_shift_ode.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_double,fp,fp]
        self.lib.spherical_fast_rhs.argtypes=[C.c_void_p,fp,fp,C.POINTER(C.c_double)]
        self.lib.spherical_fast_advance.argtypes=[C.c_void_p,fp,C.c_int,C.c_double,C.POINTER(C.c_double)]
        params=np.array([a.kappa1,a.kappa2,a.slicing,a.lapse_restoring,a.scri_sat,
            a.regularity_sat,a.origin_ko,a.origin_ko_radius,a.theta=='appendix-c',a.endpoint=='analytic-g1',getattr(a,'gauge_corner_sat',0.),getattr(a,'advection','centered')=='upwind',getattr(a,'puncture',False),getattr(a,'lapse_restoring_form','symmetric')=='multiplicative',getattr(a,'initial_data',None) in ('wormhole','trumpet'),getattr(a,'damping_form','constant')=='reference-lapse',(2 if self.trumpet_metric else int(getattr(a,'metric_variable','deviation')=='log')),getattr(a,'lapse_variable','deviation')=='log-reference',{'linear':0,'rational':1,'blended-log':2}[getattr(a,'slicing_family','linear')],getattr(a,'slicing_scale',.1),getattr(a,'elliptic_quadrature','prefix')=='inward',getattr(a,'lapse_background_source','reference')=='collapsed-inner',{'constant':0,'physical-log':1,'physical-local':2}[getattr(a,'lapse_restoring_profile','constant')],getattr(a,'subtract_local',False),self.af,getattr(a,'puncture_ko',0.),getattr(a,'puncture_ko_radius',.8),getattr(a,'compensated_update',False),getattr(a,'puncture_ko_inner_width',0.)],dtype=float)
        self.handle=self.lib.spherical_fast_create(len(op.r),fields,op.width,len(op.qindex[0]),
            op.r,np.asarray(op.index,dtype=np.int32),op.d1,op.d2,
            np.asarray(op.qindex,dtype=np.int32),op.qweights,
            np.asarray(op.aindex[0],dtype=np.int32),op.ad1[0],np.asarray(op.aindex[1],dtype=np.int32),op.ad1[1],params)
        if not self.handle:raise RuntimeError('Compiled engine initialization failed')
        if getattr(a,'grid_stretch',0.):
            self.lib.spherical_fast_set_grid_jacobian.argtypes=[C.c_void_p,C.c_double,fp]
            if self.lib.spherical_fast_set_grid_jacobian(self.handle,evolution.h,np.ascontiguousarray(evolution.grid_jacobian)):
                raise ValueError('Unsupported mapped-grid geometry configuration')
        if getattr(a,'mapped_ko','physical')=='local':
            self.lib.spherical_fast_set_mapped_ko.argtypes=[C.c_void_p,C.c_int]
            if self.lib.spherical_fast_set_mapped_ko(self.handle,1):raise ValueError('Local mapped KO requires the full AF chart')
        if getattr(a,'ko_order',6)!=6:
            self.lib.spherical_fast_set_ko_order.argtypes=[C.c_void_p,C.c_int]
            if self.lib.spherical_fast_set_ko_order(self.handle,a.ko_order):raise ValueError('Higher-order KO requires the full AF chart')
        self.lib.spherical_fast_set_lapse_source(self.handle,np.ascontiguousarray(evolution.lapse_source,dtype=float))
        if self.trumpet_metric:self.lib.spherical_fast_set_metric_reference(self.handle,np.ascontiguousarray(evolution.metric_profile))
        if self.af:
            refs=np.stack([evolution.refs[key] for key in ['chi','w','alpha','c','u','gap']],axis=1)
            self.lib.spherical_fast_set_af_reference(self.handle,np.ascontiguousarray(refs))
        if hasattr(op,'shift_ode_solver'):
            solver=op.shift_ode_solver
            self.lib.spherical_fast_set_shift_ode(self.handle,solver.bandwidth,solver.skip,solver.refinements,solver.scale,solver.lu,solver.matrix)

    def set_update_compensation(self,correction):
        if correction.shape!=self.shape or not np.all(np.isfinite(correction)):raise ValueError('Invalid accepted-step compensation')
        self.lib.spherical_fast_set_update_compensation(self.handle,np.ascontiguousarray(correction,dtype=float))

    def get_update_compensation(self):
        result=np.zeros(self.shape);self.lib.spherical_fast_get_update_compensation(self.handle,result);return result

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
        if not self.puncture or self.af_endpoint:result.pop('max_inner_characteristic')
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
