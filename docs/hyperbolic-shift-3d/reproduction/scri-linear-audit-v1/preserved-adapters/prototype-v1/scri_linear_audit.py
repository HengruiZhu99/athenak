#!/usr/bin/env python3
"""Bounded frozen-volume scri Arnoldi audit using the existing native equations.

The 19 coordinates per surface row are chi, five Frobenius-orthonormal
tracefree metric components, five tracefree A components, P, Theta, Lambda[3],
B[3]. Finite input perturbations are retracted with the existing algebraic
projection; exact null values and dependent D are recomputed. Output is the
Minkowski tangent projection of the native surface RHS. This is a surface-only
semidiscrete diagnostic, not the full coupled or finite-timestep operator.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path(__file__).with_name('scri_linear_probe.cpp')
FIELDS=['chi','g_diag1','g_diag2','g_xy','g_xz','g_yz',
        'A_diag1','A_diag2','A_xy','A_xz','A_yz','P','Theta',
        'Lambda_x','Lambda_y','Lambda_z','B_x','B_y','B_z']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


class Probe:
    def __init__(self,path,n,cross,budget):
        self.start=time.monotonic();self.budget=budget;self.calls=0
        self.lib=ctypes.CDLL(str(path));ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        self.lib.scri_linear_create.argtypes=[ctypes.c_int,ctypes.c_int];self.lib.scri_linear_create.restype=ctypes.c_void_p
        self.lib.scri_linear_size.argtypes=[ctypes.c_void_p];self.lib.scri_linear_size.restype=ctypes.c_int
        self.lib.scri_linear_evaluate.argtypes=[ctypes.c_void_p,ptr,ptr,ptr];self.lib.scri_linear_evaluate.restype=ctypes.c_int
        self.lib.scri_linear_error.restype=ctypes.c_char_p
        self.lib.scri_linear_destroy.argtypes=[ctypes.c_void_p]
        self.lib.scri_linear_rows.argtypes=[ctypes.c_void_p,ptr]
        self.handle=self.lib.scri_linear_create(n,cross)
        if not self.handle:raise RuntimeError(self.lib.scri_linear_error().decode())
        self.size=self.lib.scri_linear_size(self.handle);self.stats=np.zeros(6)
        self.rows=np.empty((self.size//19,6));self.lib.scri_linear_rows(self.handle,self.rows)
    def close(self):self.lib.scri_linear_destroy(self.handle)
    def evaluate(self,x):
        if time.monotonic()-self.start>self.budget:raise TimeoutError('bounded local diagnostic time budget')
        out=np.empty(self.size);stats=np.empty(6)
        if self.lib.scri_linear_evaluate(self.handle,np.ascontiguousarray(x),out,stats):
            raise RuntimeError(self.lib.scri_linear_error().decode())
        self.calls+=1;self.stats=np.maximum(self.stats,stats)
        return out
    def jvp(self,v,epsilon):
        if np.iscomplexobj(v):return self.jvp(v.real,epsilon)+1j*self.jvp(v.imag,epsilon)
        maximum=np.max(np.abs(v))
        if maximum==0:return np.zeros_like(v)
        scale=epsilon/maximum
        return (self.evaluate(scale*v)-self.evaluate(-scale*v))/(2*scale)


def worker(args):
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    probe=Probe(args.library.resolve(),args.n,args.scheme=='v3',args.budget)
    report=dict(status='running',scope='frozen Minkowski volume, tangent-projected surface RHS',
                n=args.n,scheme=args.scheme,dimension=probe.size,epsilon_max_coordinate=args.epsilon,
                basis=FIELDS,library_sha256=sha(args.library),seed=args.seed,
                maximum_arnoldi_vectors=args.vectors,budget_seconds=args.budget)
    started=time.monotonic();rng=np.random.default_rng(args.seed)
    try:
        zero=np.zeros(probe.size);f0=probe.evaluate(zero)
        report['background_rhs_max']=float(np.max(np.abs(f0)))
        report['background_rhs_l2']=float(np.linalg.norm(f0))
        vectors=[rng.normal(size=probe.size) for _ in range(2)]
        vectors=[v/np.linalg.norm(v) for v in vectors]
        v,w=vectors;av=probe.jvp(v,args.epsilon);aw=probe.jvp(w,args.epsilon)
        checks=[]
        for factor in [5.,1.,.2]:
            other=probe.jvp(v,factor*args.epsilon)
            checks.append(dict(epsilon=factor*args.epsilon,relative_change=float(np.linalg.norm(other-av)/np.linalg.norm(av))))
        report['epsilon_checks']=checks
        report['linearity_checks']=dict(
            negation=float(np.linalg.norm(probe.jvp(-v,args.epsilon)+av)/np.linalg.norm(av)),
            scaling=float(np.linalg.norm(probe.jvp(2*v,args.epsilon)-2*av)/(2*np.linalg.norm(av))),
            additivity=float(np.linalg.norm(probe.jvp(v+w,args.epsilon)-av-aw)/(np.linalg.norm(av)+np.linalg.norm(aw))))
        # Two-pass modified Gram-Schmidt; the small Hessenberg eigensolve is
        # NumPy-only. No inferred convergence from Ritz values is used.
        q=np.zeros((probe.size,args.vectors+1));h=np.zeros((args.vectors+1,args.vectors));q[:,0]=v
        checkpoints=[];m=0
        for column in range(args.vectors):
            if time.monotonic()-probe.start>args.budget*.7:break
            z=probe.jvp(q[:,column],args.epsilon)
            for _ in range(2):
                coefficients=q[:,:column+1].T@z;h[:column+1,column]+=coefficients;z-=q[:,:column+1]@coefficients
            h[column+1,column]=np.linalg.norm(z);m=column+1
            if h[m,column]>1e-12:q[:,m]=z/h[m,column]
            if m%10==0 or m==args.vectors:
                eigenvalues=np.linalg.eigvals(h[:m,:m]);top=sorted(eigenvalues,key=lambda z:z.real,reverse=True)[:3]
                checkpoints.append(dict(vectors=m,rightmost_ritz=[[float(z.real),float(z.imag)] for z in top]))
                report['arnoldi_progress']=checkpoints;write(out/'progress.json',report)
            if h[m,column]<=1e-12:break
        if m==0:raise TimeoutError('no Arnoldi iteration completed')
        eigenvalues,eigenvectors=np.linalg.eig(h[:m,:m]);order=np.argsort(eigenvalues.real)[::-1]
        selected=[]
        for index in order:
            value=eigenvalues[index]
            if value.imag < -1e-8:continue  # retain one member of conjugate pairs
            mode=q[:,:m]@eigenvectors[:,index];mode/=np.linalg.norm(mode)
            action=probe.jvp(mode,args.epsilon)
            fine=probe.jvp(mode,args.epsilon*.2)
            residual=float(np.linalg.norm(action-value*mode))
            relative=residual/(np.linalg.norm(action)+abs(value))
            consistency=float(np.linalg.norm(fine-action)/max(np.linalg.norm(action),1e-30))
            coefficients=np.abs(mode.reshape(-1,19))**2
            energy=np.sum(coefficients,axis=1);edge=(probe.rows[:,4]<2)|(probe.rows[:,4]>=args.n-2)|(probe.rows[:,5]<2)|(probe.rows[:,5]>=args.n-2)
            maximum=int(np.argmax(energy));projected_estimate=float(abs(h[m,m-1]*eigenvectors[-1,index]))
            selected.append(dict(eigenvalue=[float(value.real),float(value.imag)],
                absolute_true_jvp_residual=residual,relative_true_jvp_residual=float(relative),
                hessenberg_residual_estimate=projected_estimate,epsilon_relative_change=consistency,
                residual_and_epsilon_pass=bool(relative<1e-7 and consistency<1e-4),
                field_squared_norm={name:float(e) for name,e in zip(FIELDS,np.sum(coefficients,axis=0))},
                edge_band_squared_norm=float(sum(energy[edge])),maximum_row=maximum,
                maximum_row_geometry=probe.rows[maximum].tolist()))
            if len(selected)==4:break
        gram=q[:,:m].T@q[:,:m]-np.eye(m)
        report.update(status='completed',arnoldi_vectors=m,arnoldi_progress=checkpoints,
                      basis_orthogonality_max=float(np.max(np.abs(gram))),modes=selected,
                      interpretation='Only modes passing fresh residual and epsilon checks are converged candidates; no full-system or finite-dt stability inference.')
        np.savez_compressed(out/'local-krylov.npz',hessenberg=h[:m+1,:m],basis=q[:,:m],f0=f0)
    except TimeoutError as exc:report.update(status='bounded',error=str(exc))
    finally:
        report.update(seconds=time.monotonic()-started,native_rhs_evaluations=probe.calls,
                      max_checks=dict(zip(['null','dependent_D_compatibility','determinant_error','A_trace','pole','clock'],probe.stats.tolist())))
        write(out/'report.json',report);probe.close()
    print(json.dumps(dict(status=report['status'],output=str(out),calls=probe.calls,seconds=report['seconds'])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--build-dir',type=Path,default=ROOT/'build-hyperbolic-mpi')
    p.add_argument('--library',type=Path)
    p.add_argument('--worker',action='store_true')
    p.add_argument('--n',type=int,choices=[6,8],default=6)
    p.add_argument('--scheme',choices=['v2','v3'],default='v2')
    p.add_argument('--vectors',type=int,default=40)
    p.add_argument('--epsilon',type=float,default=2e-6)
    p.add_argument('--seed',type=int,default=738123)
    p.add_argument('--budget',type=float,default=370)
    args=p.parse_args()
    if not 1<=args.vectors<=120 or not 0<args.budget<=390 or not 0<args.epsilon<1e-3:p.error('invalid bounded diagnostic parameters')
    args.output=args.output.resolve()
    if args.worker:
        if not args.library:p.error('--worker requires --library')
        worker(args);return
    args.output.mkdir(parents=True,exist_ok=False);build=args.build_dir.resolve()
    flagfile=build/'tst/hyperbolic_shift_3d/CMakeFiles/hyperbolic_shift_3d_scri_angular.dir/flags.make'
    flags={}
    for line in flagfile.read_text().splitlines():
        if ' = ' in line:
            key,value=line.split(' = ',1);flags[key]=shlex.split(value)
    library=args.output/('probe.dylib' if sys.platform=='darwin' else 'probe.so')
    command=['/usr/bin/c++',*flags['CXX_DEFINES'],*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],
             '-dynamiclib' if sys.platform=='darwin' else '-shared','-fPIC',str(SOURCE),'-o',str(library)]
    command += [str(build/f'kokkos/{part}/src/libkokkos{part}.a') for part in ['containers','algorithms','core','simd']]
    start=time.monotonic();result=subprocess.run(command,capture_output=True,text=True,timeout=120)
    write(args.output/'build.json',dict(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,seconds=time.monotonic()-start))
    if result.returncode:raise RuntimeError(result.stderr)
    source_files=[SOURCE,Path(__file__),*sorted((ROOT/'src/z4c/hyperboloidal').glob('*.hpp')),
                  *sorted((ROOT/'src/mesh/cubed_sphere').glob('*.hpp'))]
    write(args.output/'source.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        sources={str(f.relative_to(ROOT)):sha(f) for f in source_files},library_sha256=sha(library),flags_sha256=sha(flagfile)))
    for scheme in ['v2','v3']:
        cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--library',str(library),'--n',str(args.n),
             '--scheme',scheme,'--vectors',str(args.vectors),'--epsilon',str(args.epsilon),'--seed',str(args.seed),
             '--budget',str(args.budget),'--output',str(args.output/scheme)]
        env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
        start=time.monotonic();result=subprocess.run(cmd,capture_output=True,text=True,timeout=410,env=env)
        write(args.output/(scheme+'-execution.json'),dict(command=cmd,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,seconds=time.monotonic()-start))
        print(result.stdout,flush=True)
        if result.returncode:raise RuntimeError(result.stderr)


if __name__=='__main__':main()
