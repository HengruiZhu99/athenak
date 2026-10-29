#!/usr/bin/env python3
"""Read-only angular-filter response of saved, explicitly unresolved vectors/states."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import argparse, hashlib, importlib.util, json, struct, sys, time
import numpy as np

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'tst/hyperbolic_shift_3d/scri_linear_audit.py').is_file())
START=time.monotonic()
def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def relative(path):return str(Path(path).relative_to(ROOT))
P=ROOT/'docs/hyperbolic-shift-3d/reproduction/scri-dissipation-prototype-v1/prototype.py'
prototype=load_module('angular_prototype',P)
audit=load_module('linear_audit',ROOT/'tst/hyperbolic_shift_3d/scri_linear_audit.py')

def energy(u,m):return float(np.sum(m[:,None]*np.abs(u)**2))
def angular(plan,u):
    b=plan.apply_b(u);q=plan.apply_q(u,2);den=energy(u,plan.mass)
    return dict(norm_squared_area=den,B_relative_norm=np.sqrt(energy(b,plan.mass)/den),
                Q2_damping_rayleigh=-float(np.vdot(u,plan.mass[:,None]*q).real)/den,
                Q2_relative_norm=np.sqrt(energy(q,plan.mass)/den),
                energy_identity_error=abs(float(np.vdot(u,plan.mass[:,None]*q).real)+2*plan.n*energy(b,plan.mass))/max(den,1e-300))

spaces={}
def low_degree(plan,u):
    """Restriction of homogeneous Cartesian polynomials L and L-1 spans l<=L."""
    answer={};weighted=u*np.sqrt(plan.mass[:,None]);den=np.sum(np.abs(weighted)**2)
    for L in (2,4,6):
        key=(plan.n,L)
        if key not in spaces:
            x,y,z=plan.points.T;columns=[]
            for d in (L-1,L):
                for a in range(d+1):
                    for b in range(d-a+1):columns.append(x**a*y**b*z**(d-a-b))
            design=np.column_stack(columns)*np.sqrt(plan.mass[:,None])
            left,s,_=np.linalg.svd(design,full_matrices=False)
            assert s[-1]>s[0]*1e-12 and len(s)==(L+1)**2
            spaces[key]=left
        coeff=spaces[key].conj().T@weighted
        answer[str(L)]=float(np.sum(np.abs(coeff)**2)/den)
    return answer

def planes(volume,n):
    # 13..18 are the six outer spherical blocks; native i is fastest.
    return volume.reshape(19,n,n,n,-1)[13:19].transpose(3,0,1,2,4).reshape(n,6*n*n,-1)

def summarize(plan,volume,surface,free_count,lambda_slice,d_slice,A_slice):
    n=plan.n;p=planes(volume,n);free=p[:,:,:free_count]
    radii=.75+.25*(np.arange(n)+.5)/n
    # A C-infinity fixed-coordinate-width taper, zero through .75 and one from .95.
    t=np.clip((radii-.75)/.20,0,1)
    w=np.zeros(n);w[t>=1]=1;mask=(t>0)&(t<1)
    a=np.exp(-1/t[mask]);b=np.exp(-1/(1-t[mask]));w[mask]=a/(a+b)
    total=np.sum(np.abs(volume)**2)+np.sum(np.abs(surface)**2)
    outer_fraction=float(np.sum(np.abs(p)**2)/total)
    byplane=[];wholevolume=float(np.sum(np.abs(volume)**2))
    denominator=0.;dissipation=0.;free_denominator=0.
    for i in range(n):
        item=angular(plan,free[i]);item.update(i=i,radius=float(radii[i]),taper=float(w[i]),
            fraction_all_unweighted=float(np.sum(np.abs(p[i])**2)/total),
            fraction_volume_unweighted=float(np.sum(np.abs(p[i])**2)/wholevolume))
        byplane.append(item)
        # Exact radial shell volume times exact angular cell area. This norm uses
        # D-Lambda and keeps alpha/beta, so the proposed induced D source cancels.
        z=p[i].copy();z[:,d_slice]-=z[:,lambda_slice]
        radial=((.75+.25*(i+1)/n)**3-(.75+.25*i/n)**3)/3
        denominator+=radial*energy(z,plan.mass)
        free_denominator+=radial*energy(free[i],plan.mass)
        dissipation+=radial*w[i]*item['Q2_damping_rayleigh']*item['norm_squared_area']
    peak=int(np.argmax([x['norm_squared_area'] for x in byplane]))
    return dict(surface_fraction_all_unweighted=float(np.sum(np.abs(surface)**2)/total),
                outer_shell_fraction_all_unweighted=outer_fraction,
                other_volume_fraction_all_unweighted=float(np.sum(np.abs(volume[:13*n**3])**2)/total),
                surface=angular(plan,surface),surface_A=angular(plan,surface[:,A_slice]),
                surface_low_degree_fraction=low_degree(plan,surface),
                surface_A_low_degree_fraction=low_degree(plan,surface[:,A_slice]),
                outer_planes=byplane,peak_outer_plane=peak,
                peak_outer_low_degree_fraction=low_degree(plan,free[peak]),
                collar_norm='Exact coordinate shell volume; component Frobenius metric/A; independent D-Lambda. Not a PDE symmetrizer.',
                collar_Q2_damping_rayleigh=dissipation/denominator,
                collar_Q2_free_only_damping_rayleigh=dissipation/free_denominator,
                collar_volume_norm_squared=denominator,
                collar_taper='smoothstep exp(-1/t)/(exp(-1/t)+exp(-1/(1-t))), t=(r-.75)/.20 clipped to [0,1]',
                interpretation='Isolated additive filter response, not an eigenvalue shift, full RHS change, or stability proof')

def native_coordinates_check():
    directory=ROOT/'output/hyperbolic-core-shell/scri-linear-audit-v1/coupled-n6-512'
    probe=audit.Probe(directory/'probe.dylib',6,True,30,coupled=True)
    try:
        points,labels,_,_=prototype.geometry(6)
        assert np.array_equal(points,probe.rows[:,:3])
        assert np.array_equal(labels,probe.rows[:,3:])
        predicted=points[None,:,:]*(.75+.25*(np.arange(6)+.5)/6)[:,None,None]
        actual=planes(probe.volume_rows,6)[:,:,:3]
        error=float(np.max(np.abs(predicted-actual)))
        assert error<5e-16
        return dict(library_sha256=sha(directory/'probe.dylib'),surface_coordinates_exact=True,
                    outer_coordinates_max=error,volume_dimension=probe.volume_size,dimension=probe.size)
    finally:probe.close()

def checkpoint(campaign):
    directory=ROOT/'output/hyperbolic-core-shell/aurora'/campaign
    path=directory/'output/minkowski.checkpoint';workerpath=directory/'worker-status.json'
    worker=json.loads(workerpath.read_text());expected=worker['cases'][0]['artifacts'][f'campaigns/{campaign}/output/minkowski.checkpoint']
    digest=sha(path);assert digest==expected['sha256'] and path.stat().st_size==expected['bytes']
    with path.open('rb') as f:
        magic,n,nf,kind,admitted,cells,ns,steps,t,dt=struct.unpack('<16siiiiQQQdd',f.read(72))
        assert n==16 and nf==31 and cells==19*n**3 and ns==31*6*n*n and not admitted
        epsilon,degree=0.,0
        if magic.startswith(b'HYP3D_MPI_V4'):
            epsilon,degree,reserved=struct.unpack('<dii',f.read(16));assert degree==7 and reserved==0
        values=np.fromfile(f,dtype='<f8',count=cells*nf).reshape(cells,nf)
        compensation=np.fromfile(f,dtype='<f8',count=cells*nf)
        surface=np.fromfile(f,dtype='<f8',count=ns).reshape(-1,nf)
        surface_compensation=np.fromfile(f,dtype='<f8',count=ns)
        assert not f.read(1) and all(np.isfinite(a).all() for a in (values,surface,compensation,surface_compensation))
    # Frobenius Cartesian tensors, keeping trace too for nonlinear data. Free21
    # first, then alpha,beta,D. Native values store background deviations except chi.
    fields=list(range(18))+[19,20,21,18,22,23,24,25,26,27]
    def extract(raw):
        a=raw[:,fields].copy();a[:,0]-=1
        a[:,[2,3,5,8,9,11]]*=np.sqrt(2)
        return a
    return extract(values),extract(surface)[:,:21],dict(path=relative(path),sha256=digest,
        worker_sha256=sha(workerpath),worker_artifact_hash_matched=True,n=n,steps=steps,time=t,
        admitted=bool(admitted),epsilon=epsilon,degree=degree,
        scope='Nonlinear terminal accepted state from failed trajectory; checkpoint is unadmitted; no finite-amplitude eigenmode inference')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    result=dict(status='completed',scope='Read-only saved-vector and terminal-state isolated angular response',
                script_sha256=sha(__file__),prototype_sha256=sha(P),numpy_version=np.__version__,
                native_coordinates=native_coordinates_check(),unresolved_modes=[],terminal_states=[])
    plan6=prototype.build_plan(6,7,START+360)
    modepath=ROOT/'output/hyperbolic-core-shell/scri-linear-audit-v1/coupled-n6-512/v3/local-modes.npz'
    reportpath=modepath.parent/'report.json';report=json.loads(reportpath.read_text())
    modes=np.load(modepath)['modes'];assert modes.shape==(4,110808)
    for i,mode in enumerate(modes):
        original=report['modes'][i]
        assert not original['residual_and_epsilon_pass']
        item=summarize(plan6,mode[:106704].reshape(-1,26),mode[106704:].reshape(-1,19),19,slice(13,16),slice(23,26),slice(6,11))
        item.update(mode_index=i,ritz_value=original['eigenvalue'],relative_J_residual=original['relative_true_jvp_residual'],converged=False)
        result['unresolved_modes'].append(item)
    result['mode_source']=dict(path=relative(modepath),sha256=sha(modepath),report_sha256=sha(reportpath))
    plan16=prototype.build_plan(16,7,START+360)
    result['plans']={str(p.n):dict(statistics=p.statistics,prototype_upper_bound=p.spectral_upper_bound()) for p in (plan6,plan16)}
    for campaign in ('stability-angular-minkowski-v1','stability-dissipative-minkowski-v1'):
        volume,surface,metadata=checkpoint(campaign)
        item=summarize(plan16,volume,surface,21,slice(15,18),slice(25,28),slice(7,13))
        item['source']=metadata;result['terminal_states'].append(item)
    result['seconds']=time.monotonic()-START
    (args.output/'response.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(seconds=result['seconds'],modes=[dict(index=x['mode_index'],surface=x['surface_fraction_all_unweighted'],B=x['surface']['B_relative_norm'],gamma=x['surface']['Q2_damping_rayleigh'],collar=x['collar_Q2_damping_rayleigh'],low6=x['surface_low_degree_fraction']['6']) for x in result['unresolved_modes']],terminal=[dict(time=x['source']['time'],surface=x['surface_fraction_all_unweighted'],B=x['surface']['B_relative_norm'],gamma=x['surface']['Q2_damping_rayleigh'],collar=x['collar_Q2_damping_rayleigh'],low6=x['surface_low_degree_fraction']['6']) for x in result['terminal_states']]),indent=2))
if __name__=='__main__':main()
