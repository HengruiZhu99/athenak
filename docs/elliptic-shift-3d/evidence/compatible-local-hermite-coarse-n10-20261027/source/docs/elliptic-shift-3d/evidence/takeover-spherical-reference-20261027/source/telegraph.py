"""Spherical conformal relaxation flux and independent Python rate oracle.

u=alpha-A; delta D_t alpha=chi*[w B'+(w'+2w/r)B],
tau D_t B=kappa*w*u'-B. This is a disclosed coordinate gauge:
flat-reference radial divergence, componentwise shift advection, no projection.
"""
import ctypes as C
from types import SimpleNamespace
import numpy as np
from fast import FastEngine
from gauge import lapse_coefficient
from numerics import puncture_dissipation


def window(r,inner=.01,outer=.8,width=.08,inner_profile="collar"):
    if not 0<=inner<inner+width<outer-width<outer<1:raise ValueError('Invalid telegraph support')
    def ramp(x):
        y=np.clip(x,0,1);a=y**8;b=(1-y)**8
        return a/(a+b),8*y**7*(1-y)**7/(a+b)**2
    if inner_profile not in ('collar','af-regular'):raise ValueError('Invalid inner coupling profile')
    left,dl=ramp((r-inner)/width);right,dr=ramp((outer-r)/width)
    if inner_profile=='af-regular':left=np.ones_like(r);dl=np.zeros_like(r)
    return left*right,(dl*right-left*dr)/width


def options(n,library,nu=1.,cfl=.05):
    return SimpleNamespace(n=n,library=str(library),origin_grid=False,puncture=True,
        puncture_chart=True,origin_variables=False,backend='spherical-g1',width=7,
        quadrature_width=7,kappa1=4.,kappa2=-.99,theta='appendix-c',slicing=nu,
        lapse_restoring=1.,lapse_restoring_form='multiplicative',endpoint='raw-lhopital',
        scri_sat=4.,regularity_sat=8.,origin_ko=0.,origin_ko_radius=.4,advection='upwind',
        eta=0.,gauge_corner_sat=0.,damping_form='constant',slicing_family='linear',
        slicing_scale=.1,lapse_background_source='collapsed-inner',lapse_restoring_profile='physical-log',
        puncture_ko=.5,puncture_ko_radius=.8,initial_data='wormhole',mass=1/3,
        initial_quadrature_order=96,compensated_update=True,subtract_local=True,cfl=cfl)


def python_rhs(ev,q,B,kappa,tau,support,accepted=False,lapse_attenuation=0.):
    result=ev.rhs(q,accepted=accepted).copy();r=ev.r;w,wp=support
    dB,_=ev.op.derivatives(B,-1)
    # Reconstruct beta independently from the same metric freezing quadrature.
    alpha=ev.total_alpha;chi=ev.total_chi;beta=ev.shift_deviation-r
    angular=np.divide(2*w,r,out=np.zeros_like(r),where=r!=0)
    activation=(1+lapse_attenuation)*alpha/(alpha+lapse_attenuation*(1+r*r)/2) if lapse_attenuation>0 else np.ones_like(r)
    correction=activation*chi*(w*dB+(wp+angular)*B)
    positive=r>0;result[positive,6]+=correction[positive]/alpha[positive]
    adv=ev.op.advection_derivative(B[:,None],np.array([-1]),beta)[:,0]
    rate=beta*adv+(kappa*w*ev.geometry_first[:,6]-B)/tau
    measure=ev.grid_jacobian if getattr(ev.args,'mapped_ko','physical')=='local' else None
    rate+=puncture_dissipation(B[:,None],r,ev.h,ev.args.puncture_ko,ev.args.puncture_ko_radius,measure=measure,order=getattr(ev.args,'ko_order',6))[:,0]/ev.grid_jacobian
    F=lapse_coefficient(r,alpha,ev.args.slicing)
    c=np.sqrt(chi*(F+activation*kappa*w*w/tau))
    return result,rate,np.c_[-beta-c,-beta+c]


class TelegraphEngine:
    def __init__(self,ev,library,kappa,tau,inner=.01,outer=.8,width=.08,nray=0,start=1.,inner_profile="collar",ray_interpolation="linear",ray_exit="legacy",lapse_floor=1e-10,lapse_update="log",lapse_attenuation=0.):
        if not np.isfinite(lapse_attenuation) or not 0<=lapse_attenuation<=1:raise ValueError('lapse_attenuation must lie in [0,1]')
        if lapse_update not in ('log','total'):raise ValueError('Unknown lapse update variable')
        if ray_exit not in ('legacy','dense'):raise ValueError('Unknown ray exit treatment')
        if ray_interpolation not in ('linear','pchip'):raise ValueError('Unknown ray interpolation')
        if not np.isfinite(kappa+tau) or kappa<0 or tau<=0:raise ValueError('Need kappa>=0,tau>0')
        if not np.isfinite(lapse_floor) or lapse_floor<0:raise ValueError('lapse_floor must be finite and nonnegative')
        if not ev.args.puncture_chart or ev.r[0]!=0:raise ValueError('First telegraph implementation requires the full AF domain')
        self.base=FastEngine(ev,8,library);lib=self.base.lib;self.n=len(ev.r);self.nray=nray;self.start=start
        fp=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        lib.telegraph_create.argtypes=[C.c_void_p,C.c_double,C.c_double,fp,fp,C.c_int,C.c_double];lib.telegraph_create.restype=C.c_void_p
        lib.telegraph_destroy.argtypes=[C.c_void_p]
        lib.telegraph_rhs.argtypes=[C.c_void_p,fp,fp,fp,fp,fp]
        lib.telegraph_advance.argtypes=[C.c_void_p,fp,fp,fp,C.c_int,C.c_double,C.c_double]
        lib.telegraph_stats.argtypes=[C.c_void_p,fp]
        lib.telegraph_get_compensation.argtypes=[C.c_void_p,fp]
        lib.telegraph_set_compensation.argtypes=[C.c_void_p,fp]
        w,wp=window(ev.r,inner,outer,width,inner_profile)
        self.handle=lib.telegraph_create(self.base.handle,kappa,tau,w,wp,nray,start)
        if not self.handle:raise RuntimeError('Telegraph engine creation failed')
        self.lapse_floor=lapse_floor
        self.has_lapse_floor=hasattr(lib,'telegraph_set_lapse_floor')
        if self.has_lapse_floor:
            lib.telegraph_set_lapse_floor.argtypes=[C.c_void_p,C.c_double]
            lib.telegraph_apply_lapse_floor.argtypes=[C.c_void_p,fp]
            lib.telegraph_lapse_floor_stats.argtypes=[C.c_void_p,fp]
            if lib.telegraph_set_lapse_floor(self.handle,lapse_floor):
                message=lib.spherical_fast_last_error(self.base.handle).decode();self.close();raise ValueError(message)
        elif lapse_floor!=0:
            self.close();raise ValueError('Frozen library lacks lapse_floor support; use lapse_floor=0 for an unfloored control')
        self.lapse_update=lapse_update
        self.has_total_lapse=hasattr(lib,'telegraph_set_lapse_update')
        if self.has_total_lapse:
            lib.telegraph_set_lapse_update.argtypes=[C.c_void_p,C.c_int]
            lib.telegraph_total_lapse_stats.argtypes=[C.c_void_p,fp]
            if lib.telegraph_set_lapse_update(self.handle,int(lapse_update=='total')):raise ValueError('Invalid lapse update mode')
        elif lapse_update!='log':
            self.close();raise ValueError('Frozen library lacks total-lapse update support')
        self.lapse_attenuation=lapse_attenuation
        if hasattr(lib,'telegraph_set_lapse_attenuation'):
            lib.telegraph_set_lapse_attenuation.argtypes=[C.c_void_p,C.c_double]
            if lib.telegraph_set_lapse_attenuation(self.handle,lapse_attenuation):raise ValueError('Invalid lapse attenuation')
        elif lapse_attenuation!=0:
            self.close();raise ValueError('Frozen library lacks lapse attenuation')
        # Old frozen libraries remain valid linear-mode controls.
        if ray_interpolation=='pchip':
            lib.telegraph_set_ray_interpolation.argtypes=[C.c_void_p,C.c_int]
            if lib.telegraph_set_ray_interpolation(self.handle,1):raise RuntimeError('Ray interpolation setup failed')
        if ray_exit=='dense':
            lib.telegraph_set_ray_exit.argtypes=[C.c_void_p,C.c_int]
            if lib.telegraph_set_ray_exit(self.handle,1):raise RuntimeError('Ray exit setup failed')
    def close(self):
        if self.handle:self.base.lib.telegraph_destroy(self.handle);self.handle=None
        self.base.close()
    def rhs(self,q,B):
        out=np.empty_like(q);outB=np.empty_like(B);speeds=np.empty((self.n,2))
        if self.base.lib.telegraph_rhs(self.handle,q,B,out,outB,speeds):raise FloatingPointError(self.base.lib.spherical_fast_last_error(self.base.handle).decode())
        return out,outB,speeds
    def apply_lapse_floor(self,q):
        """Clamp physical alpha/O on positive-radius nodes; retain AF/scri slots."""
        if self.has_lapse_floor:self.base.lib.telegraph_apply_lapse_floor(self.handle,q)
    def advance(self,q,B,rays,steps,dt,t):
        done=self.base.lib.telegraph_advance(self.handle,q,B,rays,steps,dt,t)
        if done<0:return -done-1,self.base.lib.spherical_fast_last_error(self.base.handle).decode()
        return done,None
    def diagnostics(self):
        extra=np.zeros(5);self.base.lib.telegraph_stats(self.handle,extra)
        result=dict(self.base.diagnostics(),**dict(zip(['max_gauge_speed','max_cfl','max_relaxation_dt','max_front_lapse_gradient','max_front_compression'],map(float,extra))))
        if self.has_lapse_floor:
            floor=np.zeros(3);self.base.lib.telegraph_lapse_floor_stats(self.handle,floor)
            result.update(zip(['max_lapse_floor_radius','max_lapse_floor_log_correction','max_lapse_floor_clamps_per_call'],map(float,floor)))
        if self.has_total_lapse:
            total=np.zeros(2);self.base.lib.telegraph_total_lapse_stats(self.handle,total)
            result.update(zip(['max_lapse_floor_conformal_correction','max_nonpositive_lapse_clamps_per_call'],map(float,total)))
        return result
    def compensation(self,value=None):
        if value is None:
            value=np.zeros(self.n*9+self.nray*2);self.base.lib.telegraph_get_compensation(self.handle,value);return value
        value=np.ascontiguousarray(value,dtype=float)
        if value.shape!=(self.n*9+self.nray*2,):raise ValueError('Bad telegraph checkpoint compensation')
        self.base.lib.telegraph_set_compensation(self.handle,value)
