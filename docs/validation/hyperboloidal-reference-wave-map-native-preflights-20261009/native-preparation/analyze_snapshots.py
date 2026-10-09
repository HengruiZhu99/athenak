"""Read-only binary64 actual-native snapshots with a pinned native constraint probe.

No AthenaK time advance, matrix assembly, eigenmode solve or propagation.
Execution is held until source/recipe and exact case authorization are reviewed.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
READER=ROOT/'build-layer-research/time-projection-controls/rst-reader-gate/restart_reader.py'
ABI=READER.with_name('abi.json')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def finite(x):
    if isinstance(x,dict):
        for y in x.values():finite(y)
    elif isinstance(x,list):
        for y in x:finite(y)
    elif isinstance(x,float):assert np.isfinite(x),'nonfinite JSON value'
def load(p):
    d=json.loads(p.read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
    finite(d);return d

def main():
    np.seterr(all='raise')
    auth_path=Path(sys.argv[1]).resolve();auth=load(auth_path)
    assert auth['snapshot_readback_authorized'] is True
    assert auth['analyzer_sha256']==sha(Path(__file__))
    launch_path=Path(auth['launch_receipt']).resolve();assert sha(launch_path)==auth['launch_receipt_sha256']
    launch=load(launch_path);directory=Path(launch['output_directory']).resolve()
    output=Path(auth['output_directory']).resolve();output.mkdir(parents=True,exist_ok=False)
    (output/'analyzer.py').write_bytes(Path(__file__).read_bytes())
    (output/'authorization.json').write_bytes(auth_path.read_bytes())
    probe=Path(auth['probe_executable']).resolve();assert sha(probe)==auth['probe_executable_sha256']
    probe_recipe=Path(auth['probe_recipe']).resolve();assert sha(probe_recipe)==auth['probe_recipe_sha256']
    probe_info=load(probe_recipe)
    seam_path=Path(auth['seam_receipt']).resolve();assert sha(seam_path)==auth['seam_receipt_sha256']
    seam=load(seam_path)
    assert seam['passed_compile_and_fixed_t0_seam'] is True
    assert seam['probe_executable_sha256']==auth['probe_executable_sha256']
    assert Path(seam['probe_executable']).resolve()==probe
    assert seam['recipe_sha256']==auth['probe_recipe_sha256']
    assert seam['seam_cases_sha256']==sha(HERE/'seam-cases.json')
    assert seam['source_before'][str(HERE/'native_seam_and_snapshot.cpp')]==sha(HERE/'native_seam_and_snapshot.cpp')
    for name,digest in probe_info['source_before'].items():assert sha(Path(name))==digest,name
    assert sha(READER)==auth['reader_sha256'] and sha(ABI)==auth['abi_sha256']
    for name,field in [('input_path','input_sha256'),('executable','executable_sha256'),('build_receipt','build_receipt_sha256')]:
        assert sha(Path(launch[name]))==launch[field],name
    build=load(Path(launch['build_receipt']))
    assert build['passed_compile_link'] and build['executable_sha256']==launch['executable_sha256']
    assert build['compiled_implementation']=='27c19d20696ea6dd4704032c51dfd026218f64f2'
    assert launch['mode']==build['mode']
    assert launch['returncode']==0, 'native process did not complete'
    spec=importlib.util.spec_from_file_location('frozen_binary64_rst',READER)
    reader=importlib.util.module_from_spec(spec);spec.loader.exec_module(reader)
    parameters=reader.parameters(Path(launch['input_path']).read_text())
    critical={'mesh/nghost':'3','problem/mass':'0','problem/pulse_angular':'true',
       'problem/pulse_width':'.35','z4c/hyperboloidal_curvature_radius':'.5',
       'z4c/hyperboloidal_layer_r0':'.05','z4c/hyperboloidal_layer_r1':'.95',
       'z4c/hyperboloidal_kappa1':'10','z4c/hyperboloidal_dissipation':'.1',
       'z4c/hyperboloidal_pole_cfl':'.03','z4c/hyperboloidal_ghost_degree':'2',
       'z4c/hyperboloidal_symmetric_ghosts':'true',
       'z4c/hyperboloidal_physical_trace_lapse':'true','z4c/hyperboloidal_preferred_source':'false',
       'time/integrator':'rk3','time/cfl_number':'.1'}
    for key,value in critical.items():assert parameters[key]==value,key
    n=int(parameters['mesh/nx1']);assert n in [16,24,32]
    for section in ['mesh','meshblock']:
        for axis in [1,2,3]:assert int(parameters[f'{section}/nx{axis}'])==n
    for axis in [1,2,3]:
        assert float(parameters[f'mesh/x{axis}min'])==-1.1
        assert float(parameters[f'mesh/x{axis}max'])==1.1
    a=float(parameters['problem/lapse_pulse']);b=float(parameters['problem/shift_pulse'])
    profile={(0.,0.):0,(.02,.01):1,(.2,.1):2}[(a,b)]
    reference=profile==0;target=float(parameters['time/tlim'])
    paths=sorted(directory.rglob('*.rst'));assert len(paths)>=2,'missing restart series'
    history_paths=list(directory.glob('*.z4c.user.hst'));assert len(history_paths)==1
    history=np.atleast_2d(np.loadtxt(history_paths[0]));assert history.shape[1]==15
    assert np.isfinite(history).all() and (history[:,1]>0).all()
    assert (np.diff(history[:,0])>0).all()
    protected={str(p):sha(p) for p in [Path(__file__),auth_path,launch_path,probe,probe_recipe,
        seam_path,READER,ABI,Path(launch['input_path']),Path(launch['executable']),
        Path(launch['build_receipt']),Path(launch['run_log']),history_paths[0]]+paths}
    protected.update(probe_info['source_before'])
    protected.update(seam['source_before'])
    protected.update(seam['all_compiler_dependency_sha256'])
    protected.update(build['source_before'])
    protected.update(build['all_compiler_dependency_sha256'])
    for name,digest in protected.items():assert sha(Path(name))==digest,name
    dump(output/'protected-inputs-before.json',protected)
    result=[];calls=[];started=time.monotonic();last=-1.
    try:
        for number,path in enumerate(paths):
            rst=reader.read_restart(path);assert rst['time']>last;last=rst['time']
            assert rst['mb_indcs']['ng']==3
            for axis in [1,2,3]:
                assert rst['mb_indcs'][f'nx{axis}']==n
                assert rst['mesh_indcs'][f'nx{axis}']==n
                assert rst['mesh_size'][f'x{axis}min']==-1.1
                assert rst['mesh_size'][f'x{axis}max']==1.1
                assert rst['mesh_size'][f'dx{axis}']==2.2/n
            for key in critical:assert rst['parameters'][key]==parameters[key],key
            raw=np.asarray(rst['data']);assert raw.dtype==np.dtype('<f8')
            assert raw.shape==(1,25,n+6,n+6,n+6)
            h=rst['mesh_size']['dx1'];first=-1.1+(0.5-3)*h
            co=first+np.arange(n+6,dtype=float)*h
            z,y,x=np.meshgrid(co,co,co,indexing='ij');mask=x*x+y*y+z*z<1
            active=raw[0][:,mask]
            assert np.isfinite(active).all()
            alpha=raw[0,18][mask];chi=raw[0,0][mask]
            assert (alpha>0).all() and (chi>0).all()
            metric=np.zeros((int(mask.sum()),3,3))
            aa=np.zeros_like(metric)
            for f,(i,j) in enumerate([(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]):
                metric[:,i,j]=metric[:,j,i]=raw[0,1+f][mask]
                aa[:,i,j]=aa[:,j,i]=raw[0,8+f][mask]
            eigen=np.linalg.eigvalsh(metric)
            assert np.isfinite(eigen).all() and (eigen[:,0]>0).all()
            # Symmetric metric eigenvalues are a field-admission diagnostic,
            # not a PDE/operator spectrum. Native constraint kernel follows.
            infile=output/f'{number:04d}-array-input-metadata.json'
            payload=raw.tobytes(order='C')
            dump(infile,{'rst_path':str(path),'rst_sha256':sha(path),
              'payload_sha256':hashlib.sha256(payload).hexdigest(),'payload_bytes':len(payload),
              'schema':'little endian binary64 [1,25,k,j,i], all stored cells',
              'rst_time':rst['time'],'rst_dt':rst['dt'],'cycle':rst['cycle']})
            command=[str(probe),'--snapshot',str(n)]
            completed=subprocess.run(command,input=payload,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            so=output/f'{number:04d}-probe.stdout';se=output/f'{number:04d}-probe.stderr'
            so.write_bytes(completed.stdout);se.write_bytes(completed.stderr)
            calls.append({'command':command,'returncode':completed.returncode,
               'stdout_sha256':sha(so),'stderr_sha256':sha(se),'input_metadata_sha256':sha(infile)})
            assert completed.returncode==0 and completed.stderr==b''
            q=load(so);assert q['active_count']==int(mask.sum())
            algebraic_tolerance=1e-11 if reference else 1e-10
            assert q['det_max']<=algebraic_tolerance
            assert q['trace_max']<=algebraic_tolerance
            matches=np.flatnonzero(np.abs(history[:,0]-rst['time'])<=1e-12)
            assert len(matches)==1,'RST/history time pairing failed'
            row=history[int(matches[0])]
            expected=np.asarray(q['rms_H_Mcon_Zcon_Theta'])
            error=np.max(np.abs(row[2:6]-expected)/np.maximum(1.,np.abs(expected)))
            assert error<=2e-11,'binary64 native kernel/history RMS mismatch'
            if number==0:
                assert rst['time']==0
                assert q['initial_profile_max_error_reference_small_large'][profile]<=2e-13
                assert np.max(expected)<=1e-9
            if reference:
                assert max(q['reference_deviation_max25'])<=1e-10
                assert np.max(expected)<=1e-9
            factor=.5 if launch['mode']=='wave-map-half' else 1.
            q.update(rst_path=str(path),rst_sha256=sha(path),time=rst['time'],cycle=rst['cycle'],
              restart_header_dt=rst['dt'],history_dt=float(row[1]),history_scaled_rms_error=float(error),
              minimum_conformal_metric_eigenvalue=float(eigen.min()),
              minimum_Penrose_spatial_metric_eigenvalue=float((eigen/chi[:,None]).min()),
              live_state_cap_recomputed=factor*min(.025*h/q['retained_max_gauge_speed'],.03*q['Omega_min']))
            result.append(q)
        assert result[-1]['time']==target, 'target time not reached'
        log=Path(launch['run_log']);assert sha(log)==launch['run_log_sha256']
        console=[{'cycle':int(c),'time_printed':float(t),'dt_printed':float(d)} for c,t,d in re.findall(
            r'cycle=(\d+)\s+time=([\d.eE+\-]+)\s+dt=([\d.eE+\-]+)',log.read_text())]
        assert console and all(np.isfinite(x['dt_printed']) and x['dt_printed']>0 for x in console)
        for name,digest in protected.items():assert sha(Path(name))==digest,('mid-case source/input drift',name)
        dump(output/'protected-inputs-after.json',protected)
        dump(output/'snapshots.json',result)
        dump(output/'receipt.json',{'passed_saved_snapshot_finite_and_diagnostic_gates':True,
           'reference':reference,'mode':launch['mode'],'N':n,'target_time':target,'saved_arrays':len(result),
           'reader_sha256':sha(READER),'abi_sha256':sha(ABI),'analyzer_sha256':sha(Path(__file__)),
           'authorization_sha256':sha(auth_path),'launch_receipt_sha256':sha(launch_path),
           'probe_executable_sha256':sha(probe),'probe_recipe_sha256':sha(probe_recipe),
           'seam_receipt_sha256':sha(seam_path),'protected_inputs_before_after_equal':True,
           'protected_inputs_before_sha256':sha(output/'protected-inputs-before.json'),
           'protected_inputs_after_sha256':sha(output/'protected-inputs-after.json'),
           'snapshots_sha256':sha(output/'snapshots.json'),'calls':calls,'console_dt_series':console,
           'seconds':time.monotonic()-started,
           'scope':'Binary64 active fields and native constraint-kernel readback; no propagation. Console dt only6digits; RST/header/history dt fullprecision with distinct timing semantics. Live-state recomputed cap not presumed identical to saved dt from a prior stage. Relative improvement/long continuation not adjudicated here.'})
        print('PASS binary64 snapshot admission/readback',launch['mode'],n,len(result))
    except Exception as error:
        dump(output/'failure.json',{'passed_saved_snapshot_finite_and_diagnostic_gates':False,
           'error':str(error),'completed_snapshots':result,'calls':calls,
           'analyzer_sha256':sha(Path(__file__)),'authorization_sha256':sha(auth_path),
           'seconds':time.monotonic()-started})
        raise
if __name__=='__main__':main()
