#!/usr/bin/env python3
"""HELD saved-operator/actual-source constraint readback; no new compile.

Default prints the plan. --execute also requires a separately reviewed,
hash-bound root authorization; this file does not grant that authorization.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,shutil,subprocess,sys,time

O=Path(__file__).resolve().parent;R=O.parents[2]
CACHE=O.parent/'attempt-1791558113840144000'
CACHE_PINS={'receipt.json':'02d1610281ba84f6451ca4363076e302cb1f8bed7822c67df1d5c35c3c0c17a5',
            'calls.json':'2800a80b43232875bcffa8a152fa55b8bd883f0fa7c7f70dac0512e794fc7407',
            'held-source-linearity.json':'58d6042baeef06f5dfcb001e75ad2d2e1d1ccfc9faf2e4ca261558f4de7651d3'}
P=R/'boundary/total-j-finite-rb-control-20261009'
OP=P/'J0-N8-rb.98-segmentedQ64-a12x24-refinement001/operator.npz'
OLD=R/'continuum/finite-rb-constraint-rate-oracle/run_constraint_rates_v2.py'
PINS={'operator':'2ed0da45a995669f7e3e2fedba231eeda0dcb06f43577125d4a194974c4f4742',
      'exe':'2293e9be6f75042f926f22232039c3c3bdd28826eb9e80061905c272b7adce15',
      'API':'d60ea8266abebf5263eb78b81997b60af472dc7b1fe102bd6f2c21baca1a6017',
      'radial_source':'65cc3df6ff7655a28beb61aab445055f7d84123c07101e6e4e5cfd6ab4251438',
      'old_FD':'cbd7b341492f5f920ddb3935be1c78e83bcc0a2c7ab6f33e26d23cb28d484f31'}
RB=.98;N=8;B=RB*RB;L=[0,0,0,0,1,1,2,2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def plan():
    return {'status':'HELD pending exact source review/authorization','J':0,'N':8,'rb':RB,
        'operator':str(OP),'pins':PINS,'scientific_witnesses':14,'centers':21,'h_levels':5,
        'physical8':['H','Mx','My','Mz','Zx','Zy','Zz','Theta_physical'],
        'same_polynomial_interpolant_in_continuum_and_bulk':True,
        'source_batch_schema':{'width':150,'rhs':[0,22],'raw_input':[33,55],'normals':[146,150]},
        'shared_maps':'fresh at every Cartesian point/channel, WJets e0/e1/e2; direct arbitrary-jet linearity control',
        'measurements':['continuum','bulk','SAT','total','initial','bulk-continuum','total-continuum'],
        'no_zero_projection_or_SAT_gate':True,'no_compile_spectrum_propagation':True,
        'detail_file':str(O/'PLAN.md'),'h0_multiplier':.5,'cached_attempt':str(CACHE),
        'cache_pins':CACHE_PINS,'prior_failed_gate_remains_failed':True}

def witnesses():
    for name,c,env in [('poly-alpha',0,0),('poly-P',2,1),('poly-Theta',3,2),('poly-beta',4,3)]:
        yield {'name':name,'channels':[(c,1.)],'envelope':env,'gauge':c in (0,4)}
    yield {'name':'poly-all-mixed','channels':[(c,(-1.)**c/(c+1)) for c in range(8)],'envelope':6,'gauge':False}
    for c,name in [(0,'gauge-alpha'),(4,'gauge-beta')]:
        yield {'name':name,'channels':[(c,1.)],'envelope':4,'gauge':True}
    shell=[1,2,3,5,6,7]
    for c in shell:yield {'name':'shell-channel-'+str(c),'channels':[(c,1.)],'envelope':5,'gauge':False}
    yield {'name':'shell-mixed','channels':[(c,(-1.)**c/(c+1)) for c in shell],'envelope':5,'gauge':False}

def envelope(rho,env):
    if env==0:return (1.,0.,0.)
    if env==1:return (rho,1.,0.)
    if env==2:return (rho*rho,2*rho,2.)
    if env==3:return (rho**3,3*rho*rho,6*rho)
    if env==4:
        v=math.exp(-8*rho);return (v,-8*v,64*v)
    if env==5:
        t=(rho-.49)/.16;v=math.exp(-t*t);return (v,-2*t*v/.16,(4*t*t-2)*v/(.16*.16))
    return (1+rho/3-rho*rho/5+rho**3/7,1/3-2*rho/5+3*rho*rho/7,-2/5+6*rho/7)

def modal(rho,np,eval_jacobi):
    rho=np.asarray(rho);z=2*rho/B-1;out=np.zeros((len(rho),8,3,N))
    for c,ell in enumerate(L):
        for k in range(N):
            factor=math.sqrt(2*(2*k+ell+1.5)/B**(ell+1.5))
            out[:,c,0,k]=factor*eval_jacobi(k,0,ell+.5,z)
            if k:out[:,c,1,k]=factor*(k+ell+1.5)/B*eval_jacobi(k-1,1,ell+1.5,z)
            if k>=2:out[:,c,2,k]=factor*(k+ell+1.5)*(k+ell+2.5)/B**2*eval_jacobi(k-2,2,ell+2.5,z)
    return out

def error(a,b):
    # Scaled full-array comparison, no BLAS dot/matmul.
    aa=math.hypot(*(float(v) for v in a.flat));bb=math.hypot(*(float(v) for v in b.flat))
    delta=a-b;absolute=math.hypot(*(float(v) for v in delta.flat))
    return {'absolute_l2':absolute,'scaled_l2':absolute/max(1.,aa,bb),
        'absolute_peak':max((abs(float(v)) for v in delta.flat),default=0.)}

def saved_cache_paths():
    for name,value in CACHE_PINS.items():
        if sha(CACHE/name)!=value:raise RuntimeError('saved cache pin changed: '+name)
    controls=json.loads((CACHE/'held-source-linearity.json').read_text())
    receipt=json.loads((CACHE/'receipt.json').read_text())
    assert receipt['sources_unchanged'] and controls['max_row_scaled_l2']<=5e-11
    paths=[CACHE/name for name in CACHE_PINS];count=controls['basis_query_rows'];consumed=0
    for i,call in enumerate(json.loads((CACHE/'calls.json').read_text())):
        if consumed==count:break
        assert call['command'][-1]=='--source-batch' and call['returncode']==0
        assert consumed+call['rows']<=count
        for suffix,key in [('input','input_sha256'),('stdout','stdout_sha256'),('stderr','stderr_sha256')]:
            p=CACHE/'calls'/('%06d.%s'%(i,suffix))
            if sha(p)!=call[key]:raise RuntimeError('saved cache call changed: '+str(p))
            paths.append(p)
        consumed+=call['rows']
    assert consumed==count
    return paths

def load_saved_cache(np):
    controls=json.loads((CACHE/'held-source-linearity.json').read_text())
    xparts=[];yparts=[];consumed=0;count=controls['basis_query_rows']
    for i,call in enumerate(json.loads((CACHE/'calls.json').read_text())):
        if consumed==count:break
        x=np.loadtxt(CACHE/'calls'/('%06d.input'%i),ndmin=2)
        y=np.loadtxt(CACHE/'calls'/('%06d.stdout'%i),ndmin=2)
        assert x.shape==(call['rows'],10) and y.shape==(call['rows'],150)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        xparts.append(x);yparts.append(y);consumed+=call['rows']
    n=controls['shared_points'];x=np.concatenate(xparts).reshape(n,8,3,10)
    y=np.concatenate(yparts).reshape(n,8,3,150)
    points=x[:,0,0,4:7]
    assert np.array_equal(x[:,:,:,:2],np.zeros((n,8,3,2)))
    assert np.array_equal(x[:,:,:,3],np.zeros((n,8,3)))
    for c in range(8):
        for d in range(3):
            assert np.array_equal(x[:,c,d,2],np.full(n,c))
            assert np.array_equal(x[:,c,d,4:7],points)
            assert np.array_equal(x[:,c,d,7:10],np.tile(np.eye(3)[d],(n,1)))
    result={tuple(p):y[i] for i,p in enumerate(points)}
    assert len(result)==n
    return result

def execute(args):
    import numpy as np
    from scipy.special import roots_jacobi,eval_jacobi
    np.seterr(all='raise')
    import warnings
    warnings.filterwarnings('error',category=RuntimeWarning)
    exe=P/'radial-bridge-release';api_header=P/'constraint_rate_api.hpp';radial=P/'radial_bridge.cpp'
    for p,key in [(OP,'operator'),(exe,'exe'),(api_header,'API'),(radial,'radial_source'),(OLD,'old_FD')]:
        if sha(p)!=PINS[key]:raise RuntimeError('source/schema pin changed: '+str(p))
    auth=json.loads(args.authorization.read_text())
    if auth.get('projected_constraint_defect_admitted') is not True:raise RuntimeError('held stage lacks admission')
    for key,value in [('driver_sha256',sha(__file__)),('executable_sha256',PINS['exe']),('operator_sha256',PINS['operator'])]:
        if auth.get(key)!=value:raise RuntimeError('authorization pin mismatch '+key)
    cached_paths=saved_cache_paths()
    spec=importlib.util.spec_from_file_location('frozen_rate_driver',OLD);fd=importlib.util.module_from_spec(spec);spec.loader.exec_module(fd)
    attempt=O/('attempt-'+str(time.time_ns()));attempt.mkdir();(attempt/'calls').mkdir()
    paths=[Path(__file__),O/'PLAN.md',OP,exe,api_header,radial,OLD,args.authorization,
        P/'actual_bridge.cpp',P/'all_m_data.hpp',P/'baseline_dual_spatial.hpp',P/'spatial_dual.hpp',
        P/'configuration_rows.hpp',P/'inputs/basis/basis-data.json',P/'assemble_segmented.py',
        P/'build-release-latest.json',*cached_paths]
    before={str(p.resolve()):sha(p) for p in paths};write(attempt/'source-before.json',before)
    shutil.copyfile(__file__,attempt/Path(__file__).name);shutil.copyfile(O/'PLAN.md',attempt/'PLAN.md')
    launch=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();api=fd.API(exe,attempt)
    start=time.monotonic();rows=[];failure=None
    try:
        a=np.load(OP,allow_pickle=False)
        for key in ('nodal_from_modal','Jbulk','Jsat','E','Kweak','SATload'):
            array=a[key]
            if array.shape!=(64,64) or array.dtype.kind!='f' or not np.isfinite(array).all():
                raise RuntimeError('invalid/nonfinite saved operator array: '+key)
        T=a['nodal_from_modal'];Jbulk=a['Jbulk'];Jsat=a['Jsat'];E=a['E']
        matrix_checks={'bulk_Riesz':error(np.einsum('ij,jk->ik',E,Jbulk,optimize=False),a['Kweak']),
            'SAT_Riesz':error(np.einsum('ij,jk->ik',E,Jsat,optimize=False),a['SATload'])}
        assert all(x['scaled_l2']<=2e-9 for x in matrix_checks.values())
        nodes=(roots_jacobi(N,0,.5)[0]+1)*B/2;node_modes=modal(nodes,np,eval_jacobi)
        Tcheck=np.zeros_like(T)
        for c in range(8):Tcheck[c*N:(c+1)*N,c*N:(c+1)*N]=node_modes[:,c,0,:]
        matrix_checks['analytic_T']=error(Tcheck,T);assert matrix_checks['analytic_T']['scaled_l2']<=5e-11
        write(attempt/'matrix-schema-checks.json',matrix_checks)
        centers=[];global_points={}
        for r in [.15,.30,.50,.70,.90,.96,.975]:
            for direction,n in enumerate(fd.DIRS):
                x=tuple(r*v for v in n);h0=.5*min(.002,(RB-r)/4);hs=[h0/2**j for j in range(5)]
                grids=[fd.stencil(x,h) for h in hs]
                for grid in grids:
                    for point in grid.values():
                        if not (0<math.hypot(*point)<RB):raise RuntimeError('outside strict transition/collar stencil')
                        if point not in global_points:global_points[point]=len(global_points)
                centers.append({'r':r,'direction':direction,'x':x,'h':hs,'grids':grids})
        points=list(global_points);rho=np.array([math.fsum(v*v for v in x) for x in points])
        cache=load_saved_cache(np)
        new_points=[x for x in points if x not in cache]
        queries=[[0,0,c,0,*x,*w] for x in new_points for c in range(8) for w in [(1.,0.,0.),(0.,1.,0.),(0.,0.,1.)]]
        fresh=np.asarray(api.run('--source-batch',queries,150)).reshape(len(new_points),8,3,150)
        data=np.empty((len(points),8,3,150))
        for x,values in zip(new_points,fresh):cache[x]=values
        for i,x in enumerate(points):data[i]=cache[x]
        write(attempt/'cache-reuse.json',{'original_attempt':str(CACHE),'original_gate_still_failed':True,
            'cache_pins':CACHE_PINS,'refined_points':len(points),'reused_points':len(points)-len(new_points),
            'new_points':len(new_points),'new_basis_query_rows':len(queries),
            'new_point_order':new_points,'source_schema_unchanged':True})
        del cache,fresh
        if not np.isfinite(data).all():raise RuntimeError('nonfinite shared actual source map')
        # All coefficient and angular data remain point-dependent through data.
        held=[];expected=[]
        for center in centers:
            x=center['x'];rho_x=math.fsum(v*v for v in x);point=global_points[x]
            choices=[(.37,-.23,.11),envelope(rho_x,6),(.81,.29,-.17)]
            for c in range(8):
                for w in choices:
                    held.append([0,0,c,0,*x,*w]);expected.append(np.einsum('df,d->f',data[point,c],np.asarray(w),optimize=False))
        direct=np.asarray(api.run('--source-batch',held,150));expected=np.asarray(expected)
        linearity=error(direct,expected);assert linearity['scaled_l2']<=5e-11
        row_errors=[]
        for u,v in zip(direct,expected):row_errors.append(error(u,v)['scaled_l2'])
        assert max(row_errors)<=5e-11
        input_absolute=float(np.max(np.abs(data[:,:,:,146:148])))
        output_absolute=float(np.max(np.abs(data[:,:,:,148:150])))
        input_scaled=0.;output_scaled=0.
        for row in data.reshape(-1,150):
            input_scale=max(1.,math.hypot(*(float(v) for v in row[33:55])))
            output_scale=max(1.,math.hypot(*(float(v) for v in row[:22])))
            input_scaled=max(input_scaled,max(abs(float(v)) for v in row[146:148])/input_scale)
            output_scaled=max(output_scaled,max(abs(float(v)) for v in row[148:150])/output_scale)
        assert input_scaled<=5e-11 and output_scaled<=5e-11
        write(attempt/'held-source-linearity.json',{'comparison':linearity,'max_row_scaled_l2':max(row_errors),
            'direct_rows':len(held),'shared_points':len(points),'basis_query_rows':len(points)*24,'new_basis_query_rows':len(queries),
            'raw22_input_normals_absolute':input_absolute,'raw22_input_normals_max_local_scaled':input_scaled,
            'raw22_output_normals_absolute':output_absolute,'raw22_output_normals_max_local_scaled':output_scaled})
        maps=np.concatenate((data[:,:,:,:22],data[:,:,:,33:55]),axis=-1);del data
        modes=modal(rho,np,eval_jacobi)
        with (attempt/'cases.jsonl').open('w') as log:
            for witness in witnesses():
                nodal=np.zeros(64)
                for c,weight in witness['channels']:
                    nodal[c*N:(c+1)*N]=[weight*envelope(float(z),witness['envelope'])[0] for z in nodes]
                X=np.linalg.solve(T,nodal)
                if not np.isfinite(X).all():raise RuntimeError('nonfinite interpolated modal coefficients')
                interpolation=error(np.einsum('ij,j->i',T,X,optimize=False),nodal);assert interpolation['scaled_l2']<=2e-9
                bulk=np.einsum('ij,j->i',Jbulk,X,optimize=False);sat=np.einsum('ij,j->i',Jsat,X,optimize=False)
                if not np.isfinite(bulk).all() or not np.isfinite(sat).all():raise RuntimeError('nonfinite projected source coefficients')
                coeff={'seed':X,'bulk':bulk,'sat':sat,'total':bulk+sat};functions={}
                for label,vector in coeff.items():
                    wjet=np.einsum('pcdk,ck->pcd',modes,vector.reshape(8,N),optimize=False)
                    functions[label]=np.einsum('pcdf,pcd->pf',maps,wjet,optimize=False)
                # Display scalar interpolation errors separately from source comparison.
                interpolation_error=[]
                for c,weight in witness['channels']:
                    expected_env=np.array([tuple(weight*v for v in envelope(float(z),witness['envelope'])) for z in rho])
                    reconstructed=np.einsum('pdk,k->pd',modes[:,c],X[c*N:(c+1)*N],optimize=False)
                    interpolation_error.append({'channel':c,'jet_error':error(reconstructed,expected_env)})
                write(attempt/(witness['name']+'-coefficients.json'),{'witness':witness,'X':X.tolist(),
                    'Ybulk':bulk.tolist(),'Ysat':sat.tolist(),'interpolation_nodal':interpolation,
                    'scalar_envelope_interpolation_error_separate':interpolation_error})
                for center in centers:
                    labels=['continuum','bulk','sat','total','initial'];jets=[]
                    for grid,h in zip(center['grids'],center['h']):
                        for label in labels:
                            source='seed' if label in ('continuum','initial') else label
                            span=slice(0,22) if label=='continuum' else slice(22,44)
                            samples={k:functions[source][global_points[p],span] for k,p in grid.items()}
                            jets.append([*center['x'],*sum(fd.fd_jets(samples,h),[])])
                    output=api.run('--constraint-rate-batch',jets,8)
                    seq={label:output[i::len(labels)] for i,label in enumerate(labels)}
                    seq['bulk_defect']=[fd.difference(x,y) for x,y in zip(seq['bulk'],seq['continuum'])]
                    seq['total_defect']=[fd.difference(x,y) for x,y in zip(seq['total'],seq['continuum'])]
                    information={label:fd.sequence_info(v) for label,v in seq.items()}
                    checks={label:info['order_status']!='unresolved' and info['last_increment_scaled_l2']<=2e-7
                            for label,info in information.items()}
                    checks['total_linearity']=fd.errors(seq['total'][-1],
                        [x+y for x,y in zip(seq['bulk'][-1],seq['sat'][-1])])['scaled_l2']<=2e-7
                    checks['extrapolated_total_linearity']=fd.errors(information['total']['richardson_last'],
                        [x+y for x,y in zip(information['bulk']['richardson_last'],
                                           information['sat']['richardson_last'])])['scaled_l2']<=2e-7
                    if witness['gauge']:
                        checks['gauge_initial_zero']=fd.norm(seq['initial'][-1])<=5e-11
                        checks['gauge_continuum_zero']=fd.norm(seq['continuum'][-1])<=2e-7
                        checks['gauge_continuum_extrapolated_zero']=fd.norm(information['continuum']['richardson_last'])<=2e-7
                    row={'witness':witness,'r':center['r'],'direction':center['direction'],'x':center['x'],'h':center['h'],
                        'sequences':seq,'sequence_info':information,'checks':checks,'passed':all(checks.values()),
                        'bulk_SAT_magnitudes_are_measurements_not_zero_gates':True}
                    log.write(json.dumps(row,allow_nan=False)+'\n');log.flush();rows.append(row)
                    if not row['passed']:raise RuntimeError('unresolved projected-rate case; preserve without adjusting h/tolerances')
                print(witness['name']+': saved '+str(len(rows))+' cases',flush=True)
    except Exception as e:failure={'type':type(e).__name__,'message':str(e)}
    write(attempt/'calls.json',api.calls);after={str(p.resolve()):sha(p) for p in paths}
    receipt={'passed_projection_defect_readback_gate':failure is None and before==after,
        'launch_HEAD':launch,'command':[sys.executable,*sys.argv],'source_before':before,'source_after':after,
        'sources_unchanged':before==after,'error':failure,'cases':len(rows),'calls':len(api.calls),
        'seconds':time.monotonic()-start,'scope':plan()}
    write(attempt/'receipt.json',receipt);print(json.dumps(receipt,indent=2),flush=True)
    if not receipt['passed_projection_defect_readback_gate']:raise SystemExit(1)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true')
    parser.add_argument('--authorization',type=Path);args=parser.parse_args()
    if not args.execute:print(json.dumps(plan(),indent=2));return
    if not args.authorization:raise RuntimeError('source-reviewed authorization required')
    execute(args)
if __name__=='__main__':main()
