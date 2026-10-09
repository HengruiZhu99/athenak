"""Pinned saved approximate C0 mode; no propagation or new matrix generation."""
from pathlib import Path
import hashlib,json,struct,subprocess,time
import numpy as np
from scipy.sparse import load_npz
W=Path(__file__).resolve().parent;R=W.parents[2]
B=R/'build-layer-research/boundary/full-tensor-propagator';V2=R/'build-layer-research/boundary/full-tensor-C0-N20-20261009/full22'
MODE=W
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pins={MODE/'candidate-vectors.npz':'4a96923a3ebe57e983a9b89bb65d0826c01037092c44d45fde24fb4617e3edc7',
      MODE/'candidate-metadata.json':'57262b974c00e6a2dbf9583003b3e64897b1306186637a8a2303570383c8345f',
      B/'server-spatialnorm':'495647e847aa77cca2c51615ed1fd0e007d71cdbb8b5ed310b37e332c7812bf0',
      V2/'spatialnorm-projected-J20.npz':'bfc114a9495b49d01b9275f4efd8a1fe49d167273f8d849631d25cddf9c2a103',
      V2/'spatialnorm-cache0.0001-metadata.json':'cd72ca5486a65e90744ed04f00a4da58f7246e2f63e5743147fba19209babb86'}
for p,h in pins.items():assert sha(p)==h,p
meta=json.loads((V2/'spatialnorm-cache0.0001-metadata.json').read_text());N=meta['points']
coords=np.array(meta['xyz_omega_volume_ginv_chi']);xyz=coords[:,:3];r=np.linalg.norm(xyz,axis=1)
G=np.zeros((N,3,3))
for f,(i,j) in enumerate(((0,0),(0,1),(0,2),(1,1),(1,2),(2,2))):G[:,i,j]=G[:,j,i]=coords[:,5]*coords[:,6+f]
L=np.fromfile(V2/'spatialnorm-cache0.0001-lift.bin',dtype='<f8').reshape(N,22,20)
v=np.load(MODE/'candidate-vectors.npz')['candidate0'];J=load_npz(V2/'spatialnorm-projected-J20.npz');jv=J@v
lam=complex(*json.loads((MODE/'candidate-metadata.json').read_text())['candidates'][0]['lambda'])
started=time.monotonic();cmds=[]
def start(command,name):
    log=(W/(name+'.stderr')).open('w');p=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log)
    m=json.loads(p.stdout.readline());cmds.append({'command':command,'stderr':str(W/(name+'.stderr'))});return p,log,m
native,nlog,nm=start([str(B/'server-spatialnorm'),'20','2.2','0.0001'],'native-diagnostic')
cpp,clog,cm=start([str(W/'comparator')],'comparator-diagnostic')
assert nm['points']==N==cm['points']
assert np.array_equal(np.array(nm['xyz_omega_volume_ginv_chi']),coords)
assert np.array_equal(np.array(cm['xyz_omega'])[:,:3],xyz)
assert np.max(abs(np.array(cm['xyz_omega'])[:,3]-coords[:,3]))==0
(W/'comparator-metadata.json').write_text(json.dumps(cm,indent=2)+'\n')
def receive(p,count):
    data=bytearray()
    while len(data)<count*8:
        b=p.stdout.read(count*8-len(data))
        if not b:raise RuntimeError('diagnostic subprocess stopped')
        data.extend(b)
    a=np.frombuffer(data,dtype='<f8').copy();assert np.isfinite(a).all();return a
def native_apply(x,mode,eps):
    native.stdin.write(mode.encode()+struct.pack('d',eps)+np.asarray(x,dtype='<f8').tobytes());native.stdin.flush()
    return receive(native,N*(7 if mode=='d' else 20))
def native_complex(x,mode,eps):return native_apply(x.real,mode,eps)+1j*native_apply(x.imag,mode,eps)
def theta(x):return np.einsum('pij,pj->pi',L,x.reshape(N,20),optimize=False)[:,17]
def constraint(x,eps):
    q=np.empty((N,8),dtype=complex);q[:,:7]=native_complex(x,'d',eps).reshape(N,7);q[:,7]=theta(x);return q
def cpp_apply(x,mode):
    cpp.stdin.write(mode.encode()+np.asarray(x,dtype='<f8').tobytes());cpp.stdin.flush();return receive(cpp,N*(40 if mode=='p' else 24))
def cpp_complex(x,mode):return cpp_apply(x.real,mode)+1j*cpp_apply(x.imag,mode)
primitive=cpp_complex(v,'p').reshape(N,40);uv=primitive[:,:20].copy().reshape(-1);qv=primitive[:,20:].copy().reshape(-1)
inputs={'v':v,'Jv':jv,'upwind_v':uv,'KO_v':qv,'centered_Jv':jv-uv-qv,'Jv_minus_lambda_v':jv-lam*v}
qs={};amplitudes=[]
for eps in (1e-4,3e-5,1e-5,3e-6):
    row={};
    for name,x in inputs.items():row[name]=constraint(x,eps)
    qs[eps]=row
    print('constraints',eps,flush=True)
q=qs[1e-5]['v'];rdot=qs[1e-5]['Jv']
def parts(x):
    out=np.empty((N,4));out[:,0]=abs(x[:,0])**2;out[:,3]=abs(x[:,7])**2
    for k,i in enumerate((1,4),1):out[:,k]=np.einsum('pi,pij,pj->p',x[:,i:i+3].conj(),G,x[:,i:i+3],optimize=False).real
    assert np.min(out)>-1e-13;return np.maximum(out,0)
def rms(x,mask):
    if not np.any(mask):return None
    return np.sqrt(np.mean(parts(x)[mask],axis=0)).tolist()
def relative(x,y,mask):
    if not np.any(mask):return None
    a=np.sum(parts(x)[mask],axis=0);b=np.sum(parts(y)[mask],axis=0)
    return [float(np.sqrt(xx/yy)) if yy>0 else None for xx,yy in zip(a,b)]
def rates(x,base,mask):
    if not np.any(mask):return None
    cols=[]
    for i,sl in enumerate((slice(0,1),slice(1,4),slice(4,7),slice(7,8))):
        if i in (1,2):num=np.sum(np.einsum('pi,pij,pj->p',base[:,sl].conj(),G,x[:,sl],optimize=False)[mask])
        else:num=np.sum((base[:,sl].conj()*x[:,sl])[mask])
        den=np.sum(parts(base)[mask,i]);z=num/den if den else None;cols.append([float(z.real),float(z.imag)] if z is not None else None)
    return cols
def comparison(x,y,mask):
    d=x-y;p=parts(d)
    return {'actual_rms_H_M_Z_Theta':rms(x,mask),'subsidiary_rms_H_M_Z_Theta':rms(y,mask),
            'defect_rms_H_M_Z_Theta':rms(d,mask),'defect_over_actual':relative(d,x,mask),'defect_over_subsidiary':relative(d,y,mask),
            'actual_projected_complex_rate':rates(x,q,mask),'subsidiary_projected_complex_rate':rates(y,q,mask),
            'defect_peak_magnitude':np.sqrt(np.max(p[mask],axis=0)).tolist() if np.any(mask) else None,
            'defect_peak_radius':[float(r[np.flatnonzero(mask)[np.argmax(p[mask,i])]]) for i in range(4)] if np.any(mask) else None}
for eps,row in qs.items():
    amplitudes.append({'epsilon_max_component':eps,'relative_to_eps1e-5':{name:relative(row[name]-qs[1e-5][name],qs[1e-5][name],np.ones(N,dtype=bool)) for name in inputs}})
native_jv=[]
for eps in (1e-4,3e-5):
    actual=native_complex(v,'f',eps)
    aq=constraint(actual,1e-5)
    native_jv.append({'epsilon_max_component':eps,'actual_centered_RHS_minus_cached_Jv_l2':float(np.linalg.norm(actual-jv)),
                      'actual_centered_RHS_minus_lambda_v_l2':float(np.linalg.norm(actual-lam*v)),
                      'C_h_actual_RHS_minus_C_h_cached_Jv_rms':rms(aq-rdot,np.ones(N,dtype=bool)),
                      'C_h_difference_over_C_h_cached_Jv':relative(aq-rdot,rdot,np.ones(N,dtype=bool))})
operators={}
operator_amplitudes={}
for mode,name in [('k','strict_active_only'),('g','chosen_componentwise_same_ray')]:
    operators[name]=cpp_complex(q,mode).reshape(N,24)
    operator_amplitudes[name]=[]
    for eps,row in qs.items():
        a=cpp_complex(row['v'],mode).reshape(N,24)
        for key in ('centered_active_stencil','fully_nested_primitive_active_stencil'):
            mask=np.array(cm[key],dtype=bool)
            operator_amplitudes[name].append({'epsilon_max_component':eps,'mask':key,
                'Kc_difference_over_Kc_eps1e-5':relative(a[:,:8]-operators[name][:,:8],operators[name][:,:8],mask)})
        if mode=='g':operator_amplitudes[name].append({'epsilon_max_component':eps,'mask':'all_active',
            'Kc_difference_over_Kc_eps1e-5':relative(a[:,:8]-operators[name][:,:8],operators[name][:,:8],np.ones(N,dtype=bool))})
    print('operator',name,flush=True)
for proc,log in ((native,nlog),(cpp,clog)):
    proc.stdin.close();proc.wait();log.close();assert proc.returncode==0
    cmds.append({'exit':proc.returncode})
masks={name:np.array(cm[key],dtype=bool) for name,key in [('centered_active','centered_active_stencil'),('centered_Lx_active','centered_plus_Lx_active_stencil'),('fully_nested_native','fully_nested_primitive_active_stencil')]}
masks['all_active']=np.ones(N,dtype=bool)
edges=[0.,.2,.4,.6,.8,.9,.95,1.]
mask_report={}
for name,mask in masks.items():
    mask_report[name]={'cells':int(mask.sum()),'r_min':float(r[mask].min()),'r_max':float(r[mask].max()),
                       'mode_constraint_squared_fraction':(np.sum(parts(q)[mask],axis=0)/np.sum(parts(q),axis=0)).tolist(),
                       'radial_counts':[int(np.sum(mask&(r>=lo)&(r<hi))) for lo,hi in zip(edges[:-1],edges[1:])]}
results={};arrays={'q_Cv':q,'r_CJv':rdot,'C_upwind_v':qs[1e-5]['upwind_v'],'C_KO_v':qs[1e-5]['KO_v'],
                   'C_centered_Jv':qs[1e-5]['centered_Jv'],'C_Jv_minus_lambda_v':qs[1e-5]['Jv_minus_lambda_v']}
for name,op in operators.items():
    kc,uc,qc=op[:,:8],op[:,8:16],op[:,16:24]
    arrays.update({name+'_Kc':kc,name+'_Uc':uc,name+'_Qc':qc})
    case={}
    for mask_name,mask in masks.items():
        if name=='strict_active_only' and mask_name=='all_active':continue
        cmask=mask if name!='strict_active_only' else mask&masks['centered_active']
        amask=mask if name!='strict_active_only' else mask&masks['centered_Lx_active']
        case[mask_name]={'centered_comparator':comparison(rdot,kc,cmask),
                         'matched_Lx_KO_comparator':comparison(rdot,kc+uc+qc,amask),
                         'centered_primitive_defect':comparison(qs[1e-5]['centered_Jv'],kc,cmask),
                         'upwind_commutator':comparison(qs[1e-5]['upwind_v'],uc,amask),
                         'KO_commutator':comparison(qs[1e-5]['KO_v'],qc,cmask)}
    if name=='chosen_componentwise_same_ray':
        case['radial_bins']=[]
        for lo,hi in zip(edges[:-1],edges[1:]):
            mask=(r>=lo)&(r<hi);case['radial_bins'].append({'r_lower':lo,'r_upper':hi,'cells':int(mask.sum()),
                'centered_comparator':comparison(rdot,kc,mask),'matched_Lx_KO_comparator':comparison(rdot,kc+uc+qc,mask)})
    results[name]=case
# The strict callback skips invalid rows; agreement is asserted only where both evaluate.
strict_vs_extension={}
for name,ids,mask in [('Kc',slice(0,8),masks['centered_active']),('Uc',slice(8,16),masks['centered_Lx_active']),('Qc',slice(16,24),masks['all_active'])]:
    strict_vs_extension[name]=float(np.max(abs(operators['strict_active_only'][:,ids][mask]-operators['chosen_componentwise_same_ray'][:,ids][mask])))
    assert strict_vs_extension[name]<1e-12
linearity=rdot-(qs[1e-5]['centered_Jv']+qs[1e-5]['upwind_v']+qs[1e-5]['KO_v'])
mode_relation=rdot-lam*q
out={'status':'COMPLETED_SAVED_APPROXIMATE_MODE_SUBSIDIARY_COMPARATOR_NO_EVOLUTION',
     'constraint_order':['H_physical','M_cov_x','M_cov_y','M_cov_z','Z_cov_x','Z_cov_y','Z_cov_z','Theta_physical'],
     'normalization':'Native H and unrescaled physical M/Z covectors, stored physical Theta. M/Z group RMS uses analytic Penrose inverse chi*ginv. No energy bound.',
     'candidate_lambda':[lam.real,lam.imag],'generator_residual_l2':float(np.linalg.norm(jv-lam*v)),
     'actual_native_Jv_controls':native_jv,'native_constraint_amplitude_controls':amplitudes,
     'subsidiary_operator_amplitude_controls':operator_amplitudes,
     'q_rms':rms(q,masks['all_active']),'r_CJv_rms':rms(rdot,masks['all_active']),
     'C_Jv_minus_lambda_v_rms':rms(qs[1e-5]['Jv_minus_lambda_v'],masks['all_active']),
     'C_Jv_minus_lambda_v_over_CJv':relative(qs[1e-5]['Jv_minus_lambda_v'],rdot,masks['all_active']),
     'linearity_residual_rms':rms(linearity,masks['all_active']),
     'C_mode_relation_rms':rms(mode_relation,masks['all_active']),
     'linearity_mode_relation_vs_separate_residual_rms':rms(mode_relation-qs[1e-5]['Jv_minus_lambda_v'],masks['all_active']),
     'radial_edges':edges,'masks':mask_report,'comparisons':results,'strict_vs_same_ray_max':strict_vs_extension,
     'limitation':'K_h is a separately discretized coefficient-aware linear subsidiary at the analytic Einstein reference. Strict support removes ghost reads but not product-rule/Hessian/projector/diagnostic mismatch. The same-ray constraint extension is chosen componentwise, not the constraint extension induced by native primitive continuation. No unique boundary cause, eigenvalue certification, native finite-RK claim, or continuum/global stability claim.',
     'seconds':time.monotonic()-started,'commands':cmds,'input_sha256':{str(p):sha(p) for p in pins},
     'source_sha256':sha(__file__),'build_provenance_sha256':sha(W/'build-provenance.json')}
(W/'results.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
np.savez_compressed(W/'diagnostic-arrays.npz',**arrays)
(W/'commands.json').write_text(json.dumps(cmds,indent=2)+'\n')
print('DONE',out['seconds'],'q',out['q_rms'],'CJv',out['r_CJv_rms'],flush=True)
