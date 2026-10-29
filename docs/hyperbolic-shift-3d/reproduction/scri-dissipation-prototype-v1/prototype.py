#!/usr/bin/env python3
"""Local angular dissipation admission study; no production equations/evolution.

Public API: geometry(n), build_plan(n, degree, deadline=None),
plan.apply_b(values), plan.apply_bt(values), plan.apply_q(values, epsilon=1),
plan.mass, plan.points, plan.spectral_upper_bound(). Arrays are point-major,
with arbitrary trailing field dimensions; complex fields are supported.
Q=-epsilon*n*M^-1 B.T M B. h=1/n is the reference angular scale.
"""
import argparse
import ctypes
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

# Bound BLAS concurrency before importing numpy. Each worker is one process.
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def check_time(deadline):
    if deadline is not None and time.monotonic()>deadline:raise TimeoutError('bounded local operator study')

def geometry(n):
    """Native face/k/j order, exact sphere-cell areas, midpoint area formula."""
    points=[];labels=[];area=[];mid=[]
    def primitive(u,v):return np.arctan2(u*v,np.sqrt(1+u*u+v*v))
    for face in range(6):
        a=face//2;c=(a+1)%3;d=(a+2)%3
        for k in range(n):
            for j in range(n):
                u=-1+2*(j+.5)/n;v=-1+2*(k+.5)/n
                scale=1/np.sqrt(1+u*u+v*v);x=np.zeros(3)
                x[a]=(-1 if face%2 else 1)*scale;x[c]=u*scale;x[d]=v*scale
                points.append(x);labels.append((face,j,k))
                ul=-1+2*j/n;uh=-1+2*(j+1)/n;vl=-1+2*k/n;vh=-1+2*(k+1)/n
                area.append(primitive(uh,vh)-primitive(ul,vh)-primitive(uh,vl)+primitive(ul,vl))
                mid.append(4/(n*n*(1+u*u+v*v)**1.5))
    return np.array(points),np.array(labels),np.array(area),np.array(mid)

@dataclass
class Plan:
    n:int
    degree:int
    points:np.ndarray
    labels:np.ndarray
    mass:np.ndarray
    starts:np.ndarray
    rows:np.ndarray
    indices:np.ndarray
    coefficients:np.ndarray
    statistics:dict

    def apply_b(self,values):
        u=np.asarray(values)
        if u.shape[0]!=len(self.mass):raise ValueError('point count mismatch')
        shape=(len(self.coefficients),)+(1,)*(u.ndim-1)
        contributions=self.coefficients.reshape(shape)*(u[self.indices]-u[self.rows])
        return np.add.reduceat(contributions,self.starts[:-1],axis=0)

    def apply_bt(self,values):
        v=np.asarray(values)
        if v.shape[0]!=len(self.mass):raise ValueError('residual row count mismatch')
        out=np.zeros_like(v);shape=(len(self.coefficients),)+(1,)*(v.ndim-1)
        np.add.at(out,self.indices,self.coefficients.reshape(shape)*v[self.rows])
        return out

    def apply_q(self,values,epsilon=1):
        u=np.asarray(values);shape=(len(self.mass),)+(1,)*(u.ndim-1)
        return -epsilon*self.n*self.apply_bt(self.mass.reshape(shape)*self.apply_b(u))/self.mass.reshape(shape)

    def spectral_upper_bound(self):
        # |A|^T |A| row-sum bound, A=sqrt(M) B / sqrt(M).
        root=np.sqrt(self.mass);a=np.abs(self.coefficients)*root[self.rows]/root[self.indices]
        rowsum=np.add.reduceat(a,self.starts[:-1]);bound=np.zeros(len(root))
        np.add.at(bound,self.indices,a*rowsum[self.rows])
        return float(np.max(bound))

def build_plan(n,degree,deadline=None):
    if n not in (6,8,12,16,24,32) or degree not in (5,7):raise ValueError('unsupported bounded study case')
    points,labels,mass,_=geometry(n);count=len(mass)
    powers=[(a,b) for a in range(degree+1) for b in range(degree+1-a)]
    target=2*len(powers)+6  # 48 / 78; include all equidistant last-shell nodes.
    zero=np.polynomial.legendre.legvander(np.array([0.]),degree)[0]
    evaluate=np.array([zero[a]*zero[b] for a,b in powers])
    starts=[0];indices=[];coefficients=[];conditions=[];moments=[];l1=[];radii=[];counts=[];cross=[]
    for begin in range(0,count,128):
        check_time(deadline)
        distance=2-2*points[begin:begin+128]@points.T
        for local in range(len(distance)):
            i=begin+local;ds=distance[local]
            cutoff=np.partition(ds,target-1)[target-1]
            neighbors=np.flatnonzero(ds<=cutoff+2e-13)
            # Keep ties, stable native index order; no owner-face queries.
            if len(neighbors)>2*target:raise RuntimeError('unbounded tied neighborhood')
            normal=points[i];axis=np.eye(3)[np.argmin(np.abs(normal))]
            e1=axis-normal*np.dot(axis,normal);e1/=np.linalg.norm(e1);e2=np.cross(normal,e1)
            delta=points[neighbors]-normal
            xi=delta@e1;eta=delta@e2;radius=np.max(np.hypot(xi,eta))
            if np.min(points[neighbors]@normal)<=0:raise RuntimeError('neighborhood outside local hemisphere')
            vx=np.polynomial.legendre.legvander(xi/radius,degree)
            vy=np.polynomial.legendre.legvander(eta/radius,degree)
            v=np.column_stack([vx[:,a]*vy[:,b] for a,b in powers])
            c,_,rank,singular=np.linalg.lstsq(v.T,evaluate,rcond=1e-13)
            if rank!=len(powers):raise RuntimeError('rank deficient polynomial moments')
            b=-c;center=np.flatnonzero(neighbors==i).item()
            b[center]=-np.sum(np.delete(b,center))
            moments.append(float(np.max(np.abs(v.T@b))))
            conditions.append(float(singular[0]/singular[-1]));l1.append(float(np.sum(np.abs(b))))
            radii.append(float(np.sqrt(max(cutoff,0))*n));counts.append(len(neighbors))
            cross.append(int(np.sum(labels[neighbors,0]!=labels[i,0])))
            indices.extend(neighbors);coefficients.extend(b);starts.append(len(indices))
    starts=np.array(starts,dtype=np.int64);indices=np.array(indices,dtype=np.int64)
    rows=np.repeat(np.arange(count),np.diff(starts));coefficients=np.array(coefficients)
    incoming=np.bincount(indices,minlength=count)
    stats=dict(polynomial_moments=len(powers),target_neighbors=target,neighbor_min=min(counts),neighbor_max=max(counts),
               incoming_max=int(np.max(incoming)),coefficient_l1_max=max(l1),condition_max=max(conditions),
               moment_residual_max=max(moments),chord_radius_over_h_max=max(radii),
               cross_face_rows=int(np.count_nonzero(cross)),cross_face_entries=int(np.sum(cross)),
               mass_min=float(mass.min()),mass_max=float(mass.max()),mass_ratio=float(mass.max()/mass.min()),
               mass_sum=float(mass.sum()),stored_entries=len(indices))
    return Plan(n,degree,points,labels,mass,starts,rows,indices,coefficients,stats)

def norm(u,mass):
    u=np.asarray(u);return float(np.sqrt(np.sum(mass.reshape((-1,)+(1,)*(u.ndim-1))*np.abs(u)**2)/np.sum(mass)))

def smooth_fields(points):
    x,y,z=points.T
    fields={'vector_x':x,'vector_y':y,'vector_z':z,
            'harmonic_l3':x*y*z,'harmonic_l4':(35*z**4-30*z*z+3)/8,
            'harmonic_l6':(231*z**6-315*z**4+105*z*z-5)/16,
            'smooth_exp':np.exp(.3*x-.2*y+.5*z)}
    for i in range(3):
        for j in range(i,3):fields[f'tensor_{i}{j}']=points[:,i]*points[:,j]-(1/3 if i==j else 0)
    return fields

def power_estimate(plan,deadline):
    rng=np.random.default_rng(63218);v=rng.normal(size=len(plan.mass));v/=np.linalg.norm(v)
    root=np.sqrt(plan.mass)
    def action(v):return -root*plan.apply_q(v/root)/plan.n
    value=0;residual=0
    for iteration in range(160):
        check_time(deadline);a=action(v);v=a/np.linalg.norm(a)
        a=action(v);value=float(np.dot(v,a));residual=float(np.linalg.norm(a-value*v))
    return dict(rayleigh=value,residual=residual,iterations=iteration+1,
                interpretation='Rayleigh lower bound / finite power estimate, not a certified upper bound')

def checks(plan,deadline):
    rng=np.random.default_rng(3719);n=len(plan.mass)
    energy=[];adjoint=[]
    for _ in range(4):
        u=rng.normal(size=n)+1j*rng.normal(size=n);v=rng.normal(size=n)+1j*rng.normal(size=n)
        bu=plan.apply_b(u);q=plan.apply_q(u)
        lhs=np.vdot(u,plan.mass*q);rhs=-plan.n*np.sum(plan.mass*np.abs(bu)**2)
        energy.append(float(abs(lhs-rhs)/max(abs(rhs),1)))
        one=np.vdot(v,plan.mass*q);two=np.vdot(plan.apply_q(v),plan.mass*u)
        adjoint.append(float(abs(one-two)/max(abs(one),abs(two),1)))
        if lhs.real>0:raise AssertionError('positive random energy')
    ones=np.ones(n)
    answer=dict(constant_B_max=float(np.max(np.abs(plan.apply_b(ones)))),
                constant_Q_max=float(np.max(np.abs(plan.apply_q(ones)))),
                energy_identity_relative_max=max(energy),weighted_adjoint_relative_max=max(adjoint))
    # The two quarter turns generate all proper cube rotations. Matching exact
    # native nodes avoids any interpolation in this equivariance check.
    lookup={tuple(np.round(x,12)):i for i,x in enumerate(plan.points)}
    vector=rng.normal(size=n);q=plan.apply_q(vector);rotation=[]
    x,y,z=plan.points.T
    for rotated in (np.column_stack((x,-z,y)),np.column_stack((-y,x,z))):
        permutation=np.array([lookup[tuple(np.round(p,12))] for p in rotated])
        rotation.append(norm(plan.apply_q(vector[permutation])-q[permutation],plan.mass)/norm(q,plan.mass))
    answer['cube_rotation_commutator_relative_max']=max(rotation)
    upper=plan.spectral_upper_bound();estimate=power_estimate(plan,deadline)
    answer['normalized_spectral_upper_bound']=upper;answer['normalized_power_estimate']=estimate
    answer['rk4_isolated_dt_epsilon_over_h_upper']=2.785293563405282/upper
    if estimate['rayleigh']>upper*(1+1e-12):raise AssertionError('spectral bound violated')
    return answer

def native_check(library,n):
    lib=ctypes.CDLL(str(Path(library).resolve()));fn=lib.angular_native_rows
    fn.argtypes=[ctypes.c_int,np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')]
    rows=np.empty((6*n*n,7))
    if fn(n,rows):raise RuntimeError('native geometry adapter failed')
    x,labels,mass,mid=geometry(n)
    return dict(native_library_sha256=sha(library),coordinate_max=float(np.max(np.abs(x-rows[:,:3]))),
                labels_exact=bool(np.array_equal(labels,rows[:,3:6])),
                midpoint_area_relative_max=float(np.max(np.abs(mid/rows[:,6]-1))),
                exact_area_sum_error=float(abs(mass.sum()-4*np.pi)))

def study(args):
    start=time.monotonic();deadline=start+args.budget
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    plan=build_plan(args.n,args.degree,deadline)
    result=dict(status='completed',scope='standalone angular operator; no native RHS or evolution',
                n=args.n,degree=args.degree,epsilon=1,scale='h=1/n; Q=-n M^-1 B.T M B',
                prototype_sha256=sha(__file__),numpy_version=np.__version__,plan=plan.statistics,
                native_geometry=native_check(args.native_library,args.n),checks=checks(plan,deadline))
    labels=plan.labels;j=labels[:,1];k=labels[:,2];ej=(j<2)|(j>=args.n-2);ek=(k<2)|(k>=args.n-2)
    regions={'all':np.ones(len(j),dtype=bool),'interior':~(ej|ek),'edge':ej^ek,'corner':ej&ek}
    result['truncation']={}
    for name,u in smooth_fields(plan.points).items():
        check_time(deadline);q=plan.apply_q(u)
        result['truncation'][name]={region:dict(max=float(np.max(np.abs(q[mask]))),l2=norm(q[mask],plan.mass[mask]))
                                    for region,mask in regions.items() if np.any(mask)}
    if args.modes:
        modes=np.load(args.modes)['modes'];measured=[]
        if modes.shape[1]!=19*len(plan.mass):raise ValueError('surface mode size mismatch')
        for index,mode in enumerate(modes):
            u=mode.reshape(-1,19);q=plan.apply_q(u);bu=plan.apply_b(u)
            den=float(np.sum(plan.mass[:,None]*np.abs(u)**2))
            energy=float(np.vdot(u,plan.mass[:,None]*q).real)
            numerator=float(plan.n*np.sum(plan.mass[:,None]*np.abs(bu)**2))
            measured.append(dict(mode_index=index,area_weighted_norm_squared=den,energy_rate_half_norm=energy,
                                 damping_rayleigh=-energy/den,factorized_damping_rayleigh=numerator/den,
                                 interpretation='Added Q energy only; not an eigenvalue shift or full coupled stability result'))
        result['surface_modes']=dict(source_sha256=sha(args.modes),measurements=measured)
    result['seconds']=time.monotonic()-start
    write(out/'report.json',result)
    # Small n6 plan permits independent dense spectrum and mode reproducibility;
    # larger plans are reproducible and intentionally not retained.
    if args.n==6:
        np.savez_compressed(out/'plan.npz',points=plan.points,labels=plan.labels,mass=plan.mass,
                            starts=plan.starts,rows=plan.rows,indices=plan.indices,coefficients=plan.coefficients)
    print(json.dumps(dict(n=args.n,degree=args.degree,seconds=result['seconds'],plan=plan.statistics,checks=result['checks'])),flush=True)

def build_native(args):
    build=args.build_dir.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    flagsfile=build/'tst/hyperbolic_shift_3d/CMakeFiles/hyperbolic_shift_3d_scri_angular.dir/flags.make'
    flags={}
    for line in flagsfile.read_text().splitlines():
        if ' = ' in line:
            k,v=line.split(' = ',1);flags[k]=shlex.split(v)
    library=out/('geometry.dylib' if sys.platform=='darwin' else 'geometry.so')
    source=HERE/'native_geometry.cpp'
    command=['/usr/bin/c++',*flags['CXX_DEFINES'],*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],
             '-dynamiclib' if sys.platform=='darwin' else '-shared','-fPIC',str(source),'-o',str(library)]
    command += [str(build/f'kokkos/{part}/src/libkokkos{part}.a') for part in ('containers','algorithms','core','simd')]
    run=subprocess.run(command,capture_output=True,text=True,timeout=120)
    write(out/'build.json',dict(command=command,returncode=run.returncode,stdout=run.stdout,stderr=run.stderr,
                              adapter_sha256=sha(source),flags_sha256=sha(flagsfile),
                              geometry_source_sha256=sha(ROOT/'src/mesh/cubed_sphere/composite.hpp'),
                              library_sha256=sha(library) if library.exists() else None))
    if run.returncode:raise RuntimeError(run.stderr)
    print(library)

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('build-native');p.add_argument('--build-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('study');p.add_argument('--n',type=int,required=True);p.add_argument('--degree',type=int,choices=(5,7),required=True)
    p.add_argument('--native-library',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--modes',type=Path);p.add_argument('--budget',type=float,default=390)
    args=parser.parse_args()
    if args.command=='build-native':build_native(args)
    else:
        if not 0<args.budget<=390:parser.error('budget must be in (0,390] seconds')
        study(args)

if __name__=='__main__':main()
