#!/usr/bin/env python3
"""Prepared bounded full-coupled normal-selector action control; no evolution.

This uses a separate adapter/library and preserves the committed v1-v4 oracle
evidence. A run requires exact default-AD parity with the frozen v4 library.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse, ctypes, hashlib, json, shlex, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import scri_directional_replay as prior

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path(__file__).with_name('scri_candidate_probe.cpp')
NAMES=['null_surface','dependent_D_compatibility','retracted_det','retracted_A_trace',
       'pole','clock','angular_cfl','null_ghost','null_ghost_before','bulk_gap','bulk_cfl']
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def exact(a,b):return bool(np.array_equal(a.view(np.uint64),b.view(np.uint64)))

class Probe:
    def __init__(self,path,n,sampled,deadline):
        self.deadline=deadline;self.calls=0;self.n=n
        self.lib=ctypes.CDLL(str(Path(path).resolve()));ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        self.lib.scri_candidate_create.argtypes=[ctypes.c_int,ctypes.c_double,ctypes.c_double,ctypes.c_int]
        self.lib.scri_candidate_create.restype=ctypes.c_void_p
        self.lib.scri_candidate_error.restype=ctypes.c_char_p
        self.lib.scri_candidate_destroy.argtypes=[ctypes.c_void_p]
        for name in ('size','volume_size'):
            fn=getattr(self.lib,'scri_candidate_'+name);fn.argtypes=[ctypes.c_void_p];fn.restype=ctypes.c_int
        self.lib.scri_candidate_evaluate.argtypes=[ctypes.c_void_p,ptr,ptr,ptr]
        self.lib.scri_candidate_evaluate.restype=ctypes.c_int
        self.handle=self.lib.scri_candidate_create(n,2,.0005,int(sampled))
        if not self.handle:raise RuntimeError(self.lib.scri_candidate_error().decode())
        self.size=self.lib.scri_candidate_size(self.handle);self.volume_size=self.lib.scri_candidate_volume_size(self.handle)
        self.stats=np.zeros(11);self.stats[9]=np.inf
    def close(self):
        if self.handle:self.lib.scri_candidate_destroy(self.handle);self.handle=None
    def evaluate(self,x):
        if time.monotonic()>self.deadline:raise TimeoutError('bounded full-coupled action replay')
        x=np.ascontiguousarray(x,dtype=np.float64)
        if x.shape!=(self.size,):raise ValueError('candidate input coordinate shape')
        out=np.empty(self.size);stats=np.empty(11)
        if self.lib.scri_candidate_evaluate(self.handle,x,out,stats):
            raise RuntimeError(self.lib.scri_candidate_error().decode())
        if not np.isfinite(out).all() or not np.isfinite(stats).all():raise ValueError('nonfinite candidate action/diagnostics')
        if max(stats[0],stats[1],stats[7])>1e-11:raise ValueError('null/dependent-D closure residual exceeds diagnostic gate')
        old_gap=self.stats[9];self.stats=np.maximum(self.stats,stats);self.stats[9]=min(old_gap,stats[9]);self.calls+=1
        return out

def build(args):
    """Run only after header readiness and scheduling authorization."""
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    flags_path=args.build_dir/'tst/hyperbolic_shift_3d/CMakeFiles/hyperbolic_shift_3d_scri_angular.dir/flags.make'
    flags={}
    for line in flags_path.read_text().splitlines():
        if ' = ' in line:
            key,value=line.split(' = ',1);flags[key]=shlex.split(value)
    library=args.output/('candidate.dylib' if sys.platform=='darwin' else 'candidate.so')
    command=['/usr/bin/c++',*flags['CXX_DEFINES'],*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],
        '-dynamiclib' if sys.platform=='darwin' else '-shared','-fPIC',str(SOURCE),'-o',str(library)]
    command.extend(str(args.build_dir/f'kokkos/{part}/src/libkokkos{part}.a') for part in ('containers','algorithms','core','simd'))
    sources=[SOURCE,Path(__file__),Path(prior.__file__),*sorted((ROOT/'src/z4c/hyperboloidal').glob('*.hpp')),
             *sorted((ROOT/'src/mesh/cubed_sphere').glob('*.hpp'))]
    before={str(path.relative_to(ROOT)):prior.sha(path) for path in sources}
    for source in sources:
        destination=args.output/'source-tree'/source.relative_to(ROOT);destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
    start=time.monotonic();result=subprocess.run(command,capture_output=True,text=True,timeout=120)
    write(args.output/'build.json',dict(command=command,returncode=result.returncode,stdout=result.stdout,
                                     stderr=result.stderr,seconds=time.monotonic()-start))
    if result.returncode:raise RuntimeError(result.stderr)
    if any(prior.sha(path)!=before[str(path.relative_to(ROOT))] for path in sources):raise RuntimeError('source changed during diagnostic build')
    write(args.output/'source.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        library_sha256=prior.sha(library),flags_sha256=prior.sha(flags_path),sources=before,
        scope='Local optional CPU normal selector; captured source tree may include reviewed uncommitted prototype'))
    print(str(library))

def extra_metrics(v,action,n):
    result=prior.direction_metrics(v,action,n);nv=26*19*n**3
    for name,sl in [('volume_Lambda',slice(13,16)),('volume_D',slice(23,26))]:
        x=v[:nv].reshape(-1,26)[:,sl];y=action[:nv].reshape(-1,26)[:,sl]
        den=prior.dot(x,x);rate=prior.dot(x,y)/den if den else 0;norm=prior.norm(y)
        result[name]=dict(coordinate_norm=prior.norm(x),action_norm=norm,projected_rate=rate,
                         relative_shape_residual=prior.norm(y-rate*x)/norm if norm else 0)
    return result

def run(args):
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    if not args.worker:
        cmd=[sys.executable,str(Path(__file__).resolve()),'run','--worker','--library',str(args.library.resolve()),
            '--frozen-library',str(args.frozen_library.resolve()),'--frozen-manifest',str(args.frozen_manifest.resolve()),
            '--checkpoint',str(args.checkpoint.resolve()),'--output',str(args.output/'worker')]
        started=time.monotonic()
        try:
            result=subprocess.run(cmd,capture_output=True,text=True,timeout=385)
            execution=dict(command=cmd,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,
                           seconds=time.monotonic()-started)
        except subprocess.TimeoutExpired as exc:
            execution=dict(command=cmd,status='bounded',seconds=time.monotonic()-started,error=str(exc))
        write(args.output/'execution.json',execution)
        if execution.get('returncode',1):raise RuntimeError('candidate action replay did not complete; see execution.json')
        print(execution['stdout']);return
    started=time.monotonic();deadline=started+350
    frozen_manifest=json.loads(args.frozen_manifest.read_text())
    if prior.sha(args.frozen_library)!=frozen_manifest['library_sha256']:raise ValueError('frozen v4 library binding mismatch')
    current_manifest=json.loads((args.library.parent/'source.json').read_text())
    if prior.sha(args.library)!=current_manifest['library_sha256']:raise ValueError('candidate library binding mismatch')
    v,metadata=prior.checkpoint(args.checkpoint)
    if metadata['n']!=16 or metadata['dt']!=.0005 or metadata['epsilon']!=2 or metadata['time']!=.30:
        raise ValueError('replay is predeclared for actual .30 n16 V4 epsilon2/dt.0005 capture')
    v/=prior.norm(v);ad=sampled=frozen=None
    try:
        ad=Probe(args.library,16,False,deadline);sampled=Probe(args.library,16,True,deadline)
        frozen=prior.Probe(args.frozen_library,16,2,.0005,350)
        if not ad.size==sampled.size==frozen.size==len(v):raise ValueError('native coordinate dimension mismatch')
        zero=np.zeros_like(v);bg_ad=ad.evaluate(zero);bg_frozen=frozen.evaluate(zero,True);bg_sampled=sampled.evaluate(zero)
        if not exact(bg_ad,bg_frozen):raise ValueError('default AD background differs bitwise from frozen v4')
        rows=[];saved={};parity_evaluations=1;largest_default_error=0.
        for amplitude in (1e-4,3e-5,1e-5,3e-6):
            scale=amplitude/float(np.max(np.abs(v)));native=[];candidate=[]
            for sign in (1,-1):
                x=(sign*scale)*v;a=ad.evaluate(x);reference=frozen.evaluate(x,True);b=sampled.evaluate(x)
                largest_default_error=max(largest_default_error,float(np.max(np.abs(a-reference))))
                if not exact(a,reference):raise ValueError('default AD full RHS differs bitwise from frozen v4')
                parity_evaluations+=1;native.append(a);candidate.append(b)
            a=(native[0]-native[1])/(2*scale);b=(candidate[0]-candidate[1])/(2*scale);saved[amplitude]=(a,b)
            rows.append(dict(amplitude=amplitude,default_AD=extra_metrics(v,a,16),sampled_normal=extra_metrics(v,b,16),
                candidate_minus_default=extra_metrics(v,b-a,16)))
        # Interleaved inputs provide a direct per-call state-reset check.
        x=(1e-5/np.max(np.abs(v)))*v;first=sampled.evaluate(x);sampled.evaluate(-x)
        reset_exact=exact(first,sampled.evaluate(x))
        if not reset_exact:raise ValueError('candidate per-call reset failed')
        for row in rows:
            a,b=saved[row['amplitude']];ra,rb=saved[1e-5]
            row['default_relative_change_from_1e_5']=prior.norm(a-ra)/prior.norm(ra)
            row['candidate_relative_change_from_1e_5']=prior.norm(b-rb)/prior.norm(rb)
        result=dict(status='completed',scope='Full coupled native normal-selector directional action; no evolution, eigenvalue or stability claim',
            generator='T0 RHS(P(U0+Ez)) at Minkowski, not accepted RK map or nonlinear-checkpoint Jacobian',
            norm='Unweighted point-coordinate Euclidean, Frobenius tracefree tensors; not physical energy',
            checkpoint=metadata,library_sha256=prior.sha(args.library),frozen_library_sha256=prior.sha(args.frozen_library),
            source_manifest_sha256=prior.sha(args.library.parent/'source.json'),frozen_manifest_sha256=prior.sha(args.frozen_manifest),
            script_sha256=prior.sha(__file__),shared_decoder_sha256=prior.sha(prior.__file__),
            default_AD_full_array_bitwise_frozen_parity=True,default_parity_evaluations=parity_evaluations,
            default_AD_maximum_absolute_difference=largest_default_error,candidate_reset_bitwise_exact=reset_exact,
            background_rhs_max=dict(default_AD=float(np.max(np.abs(bg_ad))),sampled_normal=float(np.max(np.abs(bg_sampled)))),
            rows=rows,native_diagnostics=dict(default_AD=dict(zip(NAMES,ad.stats.tolist())),sampled_normal=dict(zip(NAMES,sampled.stats.tolist()))),
            evaluations=dict(default_AD=ad.calls,sampled_normal=sampled.calls,frozen_AD=frozen.calls),seconds=time.monotonic()-started)
        write(args.output/'candidate-action.json',result);print(str(args.output/'candidate-action.json'))
    finally:
        for p in (ad,sampled,frozen):
            if p is not None:p.close()

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    b=sub.add_parser('build');b.add_argument('--build-dir',type=Path,default=ROOT/'build-hyperbolic-mpi');b.add_argument('--output',type=Path,required=True)
    r=sub.add_parser('run');r.add_argument('--library',type=Path,required=True);r.add_argument('--frozen-library',type=Path,required=True)
    r.add_argument('--frozen-manifest',type=Path,required=True);r.add_argument('--checkpoint',type=Path,required=True)
    r.add_argument('--output',type=Path,required=True);r.add_argument('--worker',action='store_true')
    args=parser.parse_args();globals()[args.command](args)
if __name__=='__main__':main()
