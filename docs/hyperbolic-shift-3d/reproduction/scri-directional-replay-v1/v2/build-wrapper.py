#!/usr/bin/env python3
"""Bounded native directional replay of worker-bound hyperboloidal checkpoints.

No evolution or eigensolve. `build` compiles a separate thin oracle, `selftest`
checks it against the preserved n6 oracle, `compare` inspects checkpoint shapes,
and `replay` performs finite-difference Jv amplitude checks on one saved direction.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
import argparse, ctypes, hashlib, importlib.util, json, shlex, struct, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SOURCE=Path(__file__).with_name('scri_directional_probe.cpp')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1<<20),b''):h.update(data)
    return h.hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def norm(v):return float(np.sqrt(np.sum(np.abs(v)**2)))
def dot(v,w):return float(np.sum(v*w))
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

class Probe:
    def __init__(self,path,n,epsilon,dt,budget=350):
        self.start=time.monotonic();self.budget=budget;self.calls=0
        self.lib=ctypes.CDLL(str(Path(path).resolve()));ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        self.lib.scri_directional_create.argtypes=[ctypes.c_int,ctypes.c_double,ctypes.c_double]
        self.lib.scri_directional_create.restype=ctypes.c_void_p
        self.lib.scri_directional_error.restype=ctypes.c_char_p
        self.lib.scri_directional_destroy.argtypes=[ctypes.c_void_p]
        for key in ('size','volume_size'):
            fn=getattr(self.lib,'scri_directional_'+key);fn.argtypes=[ctypes.c_void_p];fn.restype=ctypes.c_int
        self.lib.scri_directional_evaluate.argtypes=[ctypes.c_void_p,ptr,ptr,ptr,ctypes.c_int]
        self.lib.scri_directional_evaluate.restype=ctypes.c_int
        self.handle=self.lib.scri_directional_create(n,epsilon,dt)
        if not self.handle:raise RuntimeError(self.lib.scri_directional_error().decode())
        self.size=self.lib.scri_directional_size(self.handle)
        self.volume_size=self.lib.scri_directional_volume_size(self.handle)
        self.stats=np.zeros(7)
        self.n=n
    def close(self):
        if self.handle:self.lib.scri_directional_destroy(self.handle);self.handle=None
    def evaluate(self,x,enabled=True):
        if time.monotonic()-self.start>self.budget:raise TimeoutError('directional replay CPU budget')
        x=np.ascontiguousarray(x,dtype=np.float64)
        if x.shape!=(self.size,):raise ValueError('directional coordinate shape mismatch')
        out=np.empty(self.size);stats=np.empty(7)
        if self.lib.scri_directional_evaluate(self.handle,x,out,stats,int(enabled)):
            raise RuntimeError(self.lib.scri_directional_error().decode())
        self.calls+=1;self.stats=np.maximum(self.stats,stats)
        if not np.isfinite(out).all():raise RuntimeError('nonfinite directional result')
        return out
    def jvp(self,v,amplitude,enabled=True):
        maximum=float(np.max(np.abs(v)))
        if not maximum:return np.zeros_like(v)
        scale=amplitude/maximum
        return (self.evaluate(scale*v,enabled)-self.evaluate(-scale*v,enabled))/(2*scale)
    def weights(self):
        ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        self.lib.scri_directional_volume_weights.argtypes=[ctypes.c_void_p,ptr]
        values=np.empty(self.volume_size//26);self.lib.scri_directional_volume_weights(self.handle,values)
        if not np.isfinite(values).all() or not (values>0).all():raise ValueError('invalid native coordinate weights')
        return values
    def retraction(self,direction):
        ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        self.lib.scri_directional_retract.argtypes=[ctypes.c_void_p,ptr,ptr,ptr,ptr]
        self.lib.scri_directional_retract.restype=ctypes.c_int
        volume=np.empty((self.volume_size//26,31));surface=np.empty(((self.size-self.volume_size)//19,31));stats=np.empty(2)
        if self.lib.scri_directional_retract(self.handle,np.ascontiguousarray(direction),volume,surface,stats):
            raise RuntimeError(self.lib.scri_directional_error().decode())
        return volume,surface,stats

def checkpoint(path,include_raw=False):
    """Decode all four arrays and independently bind exact bytes to worker output."""
    path=path.resolve();workerpath=path.parent.parent/'worker-status.json'
    worker=json.loads(workerpath.read_text());digest=sha(path)
    matches=[(case['name'],key,value) for case in worker['cases'] for key,value in case.get('artifacts',{}).items()
             if Path(key).name==path.name and value.get('sha256')==digest and value.get('bytes')==path.stat().st_size]
    if len(matches)!=1:raise ValueError('checkpoint is not uniquely bound to original worker artifact')
    with path.open('rb') as f:
        magic,n,nf,kind,admitted,cells,ns,steps,t,dt=struct.unpack('<16siiiiQQQdd',f.read(72))
        if magic not in (b'HYP3D_MPI_V3'.ljust(16,b'\0'),b'HYP3D_MPI_V4'.ljust(16,b'\0')):
            raise ValueError('requires V3 or V4 checkpoint')
        if n not in (6,8,12,16) or nf!=31 or kind!=0 or cells!=19*n**3 or ns!=31*6*n*n:
            raise ValueError('invalid Minkowski dimensions/configuration')
        if admitted not in (0,1) or not np.isfinite([t,dt]).all() or t<0 or dt<=0:raise ValueError('invalid clock/admission')
        epsilon=0.;degree=0
        if magic.startswith(b'HYP3D_MPI_V4'):
            epsilon,degree,reserved=struct.unpack('<dii',f.read(16))
            if epsilon!=2 or degree!=7 or reserved:raise ValueError('requires exact V4 epsilon2 degree7 configuration')
        arrays=[np.fromfile(f,dtype='<f8',count=count) for count in (31*cells,31*cells,ns,ns)]
        if any(len(a)!=count or not np.isfinite(a).all() for a,count in zip(arrays,(31*cells,31*cells,ns,ns))) or f.read(1):
            raise ValueError('checkpoint payload length/finiteness mismatch')
    volume=arrays[0].reshape(-1,31);surface=arrays[2].reshape(-1,31)
    def extract(raw,fields):
        out=np.zeros((len(raw),fields));out[:,0]=raw[:,0]-1
        for source,dest in ((1,1),(7,6)):
            xx,xy,xz,yy,yz,zz=raw[:,source:source+6].T
            out[:,dest]=(xx-yy)/np.sqrt(2)
            out[:,dest+1]=(xx+yy-2*zz)/np.sqrt(6)
            out[:,dest+2:dest+5]=np.column_stack((xy,xz,yz))*np.sqrt(2)
        out[:,11:13]=raw[:,13:15];out[:,13:16]=raw[:,15:18];out[:,16:19]=raw[:,19:22]
        if fields==26:
            out[:,19]=raw[:,18];out[:,20:23]=raw[:,22:25];out[:,23:26]=raw[:,25:28]
        return out
    v=extract(volume,26);s=extract(surface,19);direction=np.concatenate((v.ravel(),s.ravel()))
    def raw_deviation(raw):return max(float(np.max(np.abs(raw[:,0]-1))),float(np.max(np.abs(raw[:,1:]))))
    discarded_squared=sum(float(np.sum((a[:,1]+a[:,4]+a[:,6])**2+(a[:,7]+a[:,10]+a[:,12])**2))/3 for a in (volume,surface))
    direction_squared=dot(direction,direction)
    metadata=dict(path=str(path),sha256=digest,worker_sha256=sha(workerpath),worker_status=worker['status'],
        bound_case=matches[0][0],bound_artifact=matches[0][1],native_checkpoint_admitted=bool(admitted),
        n=n,time=t,dt=dt,steps=steps,epsilon=epsilon,degree=degree,
        tangent_norm=norm(direction),maximum_coordinate=float(np.max(np.abs(direction))),
        raw_volume_maximum_deviation=raw_deviation(volume),raw_surface_maximum_deviation=raw_deviation(surface),
        embedding_trace_discard_norm=np.sqrt(discarded_squared),
        embedding_trace_discard_relative=np.sqrt(discarded_squared/(direction_squared+discarded_squared)),
        full_independent_frobenius_norm=np.sqrt(direction_squared+discarded_squared),
        trace_discarded_max=max(float(np.max(np.abs(a[:,1]+a[:,4]+a[:,6]))) for a in (volume,surface)),
        A_trace_discarded_max=max(float(np.max(np.abs(a[:,7]+a[:,10]+a[:,12]))) for a in (volume,surface)))
    return (direction,metadata,(volume,surface)) if include_raw else (direction,metadata)

def sections(n):
    nv=26*19*n**3
    return {'all':slice(None),'volume':slice(0,nv),'outer_shell':slice(26*13*n**3,nv),'surface':slice(nv,None)}
def direction_metrics(v,action,n):
    answer={}
    for key,sl in sections(n).items():
        x=v[sl];j=action[sl];den=dot(x,x);jn=norm(j)
        a=dot(x,j)/den if den else 0
        answer[key]=dict(coordinate_norm=norm(x),action_norm=jn,projected_rate=a,
            relative_shape_residual=norm(j-a*x)/jn if jn else 0)
    return answer

def build(args):
    args.output.mkdir(parents=True,exist_ok=False)
    flags_path=args.build_dir/'tst/hyperbolic_shift_3d/CMakeFiles/hyperbolic_shift_3d_scri_angular.dir/flags.make'
    flags={}
    for line in flags_path.read_text().splitlines():
        if ' = ' in line:
            k,v=line.split(' = ',1);flags[k]=shlex.split(v)
    library=args.output/('probe.dylib' if sys.platform=='darwin' else 'probe.so')
    cmd=['/usr/bin/c++',*flags['CXX_DEFINES'],*flags['CXX_INCLUDES'],*flags['CXX_FLAGS'],
         '-dynamiclib' if sys.platform=='darwin' else '-shared','-fPIC',str(SOURCE),'-o',str(library)]
    cmd.extend(str(args.build_dir/f'kokkos/{p}/src/libkokkos{p}.a') for p in ('containers','algorithms','core','simd'))
    start=time.monotonic();r=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
    write(args.output/'build.json',dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,seconds=time.monotonic()-start))
    if r.returncode:raise RuntimeError(r.stderr)
    sources=[SOURCE,Path(__file__),*sorted((ROOT/'src/z4c/hyperboloidal').glob('*.hpp')),*sorted((ROOT/'src/mesh/cubed_sphere').glob('*.hpp'))]
    write(args.output/'source.json',dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        library_sha256=sha(library),flags_sha256=sha(flags_path),sources={str(p.relative_to(ROOT)):sha(p) for p in sources}))
    print(str(library))

def selftest(args):
    old=module('preserved_linear_audit',Path(__file__).with_name('scri_linear_audit.py'))
    reference=old.Probe(args.old_library.resolve(),6,True,120,coupled=True)
    p=Probe(args.library,6,2,.0001,120);result={}
    try:
        rng=np.random.default_rng(4791);x=rng.normal(size=p.size)*1e-7;zero=np.zeros(p.size)
        a=p.evaluate(x,False);b=reference.evaluate(x)
        result['old_oracle_epsilon0_max_error']=float(np.max(np.abs(a-b)))
        c=p.evaluate(x,True);p.evaluate(-x,False);d=p.evaluate(x,True)
        result['alternating_operator_reset_exact']=bool(np.array_equal(c,d))
        result['epsilon0_reset_exact']=bool(np.array_equal(a,p.evaluate(x,False)))
        result['zero_epsilon0_max']=float(np.max(np.abs(p.evaluate(zero,False))))
        result['zero_epsilon2_max']=float(np.max(np.abs(p.evaluate(zero,True))))
        try:p.evaluate(np.full(p.size,np.nan));raise AssertionError('nonfinite input accepted')
        except RuntimeError:result['nonfinite_rejected']=True
        result['reset_after_error_exact']=bool(np.array_equal(c,p.evaluate(x,True)))
        assert result['old_oracle_epsilon0_max_error']==0 and all(result[k] for k in ('alternating_operator_reset_exact','epsilon0_reset_exact','nonfinite_rejected','reset_after_error_exact'))
        assert result['zero_epsilon2_max']<1e-9
        result.update(status='passed',scope='n6 native adapter regression; no evolution',library_sha256=sha(args.library),
                      old_library_sha256=sha(args.old_library),calls=p.calls)
    finally:p.close();reference.close()
    write(args.output,result);print(json.dumps(result))

def compare(args):
    args.output.mkdir(parents=True,exist_ok=False)
    inputs=[checkpoint(path) for path in args.checkpoint]
    if len({(m['n'],m['epsilon'],m['dt']) for _,m in inputs})!=1:raise ValueError('comparison needs one matched configuration')
    if any(inputs[i][1]['time']>=inputs[i+1][1]['time'] for i in range(len(inputs)-1)):raise ValueError('times must increase')
    if len(inputs)<2:raise ValueError('comparison requires at least two checkpoints')
    n=inputs[0][1]['n'];rows=[]
    for i,j in [(i,j) for i in range(len(inputs)) for j in range(i+1,len(inputs))]:
        u,um=inputs[i];v,vm=inputs[j];pairs={}
        for name,sl in sections(n).items():
            a=u[sl];b=v[sl];gain=dot(a,b)/dot(a,a)
            pairs[name]=dict(signed_correlation=dot(a,b)/(norm(a)*norm(b)),best_fit_gain=gain,
                best_fit_growth=np.log(gain)/(vm['time']-um['time']) if gain>0 else None,
                relative_shape_residual=norm(b-gain*a)/norm(b))
        rows.append(dict(first=um['time'],second=vm['time'],adjacent=j==i+1,metrics=pairs))
    angular=module('collar_saved_analysis',ROOT/'docs/hyperbolic-shift-3d/reproduction/scri-collar-response-v1/analyze.py')
    plan=angular.prototype.build_plan(n,7,time.monotonic()+180);patterns=[]
    for x,m in inputs:
        nv=26*19*n**3
        patterns.append(angular.summarize(plan,x[:nv].reshape(-1,26),x[nv:].reshape(-1,19),19,slice(13,16),slice(23,26),slice(6,11)))
    weighted=[];retractions=[]
    if args.library:
        p=Probe(args.library,n,inputs[0][1]['epsilon'],inputs[0][1]['dt'],180)
        try:
            vmass=p.weights();smass=plan.mass
            weights={'volume':np.repeat(vmass,26),'outer_shell':np.repeat(vmass[13*n**3:],26),'surface':np.repeat(smass,19)}
            for row in rows:
                u=next(x for x,m in inputs if m['time']==row['first']);v=next(x for x,m in inputs if m['time']==row['second']);metrics={}
                for name,w in weights.items():
                    a=u[sections(n)[name]];b=v[sections(n)[name]]
                    aa=float(np.sum(w*a*a));bb=float(np.sum(w*b*b));ab=float(np.sum(w*a*b));gain=ab/aa
                    metrics[name]=dict(signed_correlation=ab/np.sqrt(aa*bb),best_fit_gain=gain,
                        best_fit_growth=np.log(gain)/(row['second']-row['first']) if gain>0 else None,
                        relative_shape_residual=np.sqrt(np.sum(w*(b-gain*a)**2)/bb))
                weighted.append(dict(first=row['first'],second=row['second'],metrics=metrics))
            for path in args.checkpoint:
                x,m,(rawv,raws)=checkpoint(path,True);rv,rs,stats=p.retraction(x)
                fields=list(range(18))+[19,20,21];fullv=fields+[18,22,23,24,25,26,27]
                diffs=[];maxima=[]
                for raw,retracted,selected in ((rawv,rv,fullv),(raws,rs,fields)):
                    d=(retracted-raw)[:,selected];d[:,[2,3,5,8,9,11]]*=np.sqrt(2)
                    diffs.append(float(np.sum(d*d)));maxima.append(float(np.max(np.abs(d))))
                retractions.append(dict(time=m['time'],volume_frobenius_max_difference=maxima[0],surface_frobenius_max_difference=maxima[1],
                    combined_difference_norm=np.sqrt(sum(diffs)),relative_combined_difference=np.sqrt(sum(diffs))/m['full_independent_frobenius_norm'],
                    native_retracted_determinant_residual=float(stats[0]),native_retracted_A_trace=float(stats[1])))
        finally:p.close()
    result=dict(status='completed',scope='Tangent-extracted saved trajectory directions; no native Jv/evolution',
        norm='Unweighted point-coordinate norm; Frobenius tracefree tensors. Angular quotients use separately stated area/volume norms.',
        script_sha256=sha(__file__),inputs=[m for _,m in inputs],shape_comparisons=rows,angular_patterns=patterns,
        weighted_shape_comparisons=weighted,weighted_norm='Separate native midpoint coordinate volume and exact reference solid angle; no combined area-volume energy',
        native_retraction_comparisons=retractions,library_sha256=sha(args.library) if args.library else None)
    write(args.output/'comparison.json',result);print(str(args.output/'comparison.json'))

def replay(args):
    args.output.mkdir(parents=True,exist_ok=False)
    if not args.worker:
        cmd=[sys.executable,str(Path(__file__).resolve()),'replay','--worker','--library',str(args.library),
             '--checkpoint',str(args.checkpoint),'--output',str(args.output/'worker')]
        start=time.monotonic()
        try:
            r=subprocess.run(cmd,capture_output=True,text=True,timeout=385)
            record=dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,seconds=time.monotonic()-start)
        except subprocess.TimeoutExpired as exc:
            record=dict(command=cmd,status='bounded',seconds=time.monotonic()-start,error=str(exc))
        write(args.output/'execution.json',record)
        if record.get('returncode',1):raise RuntimeError('bounded replay did not complete; see execution.json')
        print(record['stdout']);return
    v,metadata=checkpoint(args.checkpoint)
    if metadata['n']!=16 or metadata['dt']!=.0005 or metadata['epsilon']!=2:
        raise ValueError('actual replay requires exact n16 V4 epsilon2 dt=.0005')
    v/=norm(v);p=Probe(args.library,16,2,.0005,350);result={}
    try:
        amplitudes=[1e-4,3e-5,1e-5,3e-6];saved={};rows=[]
        background=p.evaluate(np.zeros(p.size),True)
        for amp in amplitudes:
            action=p.jvp(v,amp,True);counterfactual=p.jvp(v,amp,False)
            saved[amp]=(action,counterfactual)
            rows.append(dict(amplitude=amp,V4=direction_metrics(v,action,16),epsilon0=direction_metrics(v,counterfactual,16),
                             difference=direction_metrics(v,action-counterfactual,16)))
        ref=saved[1e-5]
        for row in rows:
            a,b=saved[row['amplitude']]
            row['V4_relative_change_from_1e_5']=norm(a-ref[0])/norm(ref[0])
            row['epsilon0_relative_change_from_1e_5']=norm(b-ref[1])/norm(ref[1])
        result=dict(status='completed',scope='Full native tangent generator applied to one measured direction, no eigenvalue certification or evolution',
            norm='Unweighted point-coordinate Euclidean; tensors Frobenius orthonormal; not a physical symmetrizer',
            checkpoint=metadata,library_sha256=sha(args.library),script_sha256=sha(__file__),
            background_rhs_max=float(np.max(np.abs(background))),rows=rows,native_rhs_evaluations=p.calls,
            maximum_native_checks=p.stats.tolist(),seconds=time.monotonic()-p.start)
    finally:p.close()
    write(args.output/'replay.json',result);print(str(args.output/'replay.json'))

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    b=sub.add_parser('build');b.add_argument('--build-dir',type=Path,default=ROOT/'build-hyperbolic-mpi');b.add_argument('--output',type=Path,required=True)
    s=sub.add_parser('selftest');s.add_argument('--library',type=Path,required=True);s.add_argument('--old-library',type=Path,required=True);s.add_argument('--output',type=Path,required=True)
    c=sub.add_parser('compare');c.add_argument('--checkpoint',type=Path,action='append',required=True);c.add_argument('--output',type=Path,required=True);c.add_argument('--library',type=Path)
    r=sub.add_parser('replay');r.add_argument('--library',type=Path,required=True);r.add_argument('--checkpoint',type=Path,required=True);r.add_argument('--output',type=Path,required=True);r.add_argument('--worker',action='store_true')
    args=parser.parse_args();globals()[args.command](args)
if __name__=='__main__':main()
