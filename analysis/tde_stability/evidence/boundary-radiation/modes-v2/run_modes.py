from pathlib import Path
import sys,json,time,hashlib
import numpy as np

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
OLD=REPO/'review/stability-modes-20260919/mode-analysis'
sys.path.insert(0,str(OLD))
sys.path.insert(0,str(REPO/'tst/regression'))
from mode_operator import Operator,SHAPE,ACTIVE,profile,discrepancy
from z4c_background_restart import checkpoint

BIN=ROOT.parent/'bin/athena-radiation-v2-hook'
EXPECTED='d13cb05c79b7a994d6e0f06acca022753a638e3fd8193d09384a7dca8674dacf'
assert hashlib.sha256(BIN.read_bytes()).hexdigest()==EXPECTED
text=(OLD/'input.athinput').read_text().replace('cfl_number = 0.3','cfl_number = 0.15').replace('characteristic_bc_source = zero_rate','characteristic_bc_source = physical_constraint_radiation\ncharacteristic_radiation_areal_shift = 1').replace('extrap_order = 4','extrap_order = 2')
(ROOT/'input.athinput').write_text(text)
op=Operator(str(ROOT/'calls'),binary=BIN,input_file=ROOT/'input.athinput',dt=.0375)
late=REPO/'review/stability-isolation-20260919/gauge-slow/shift_Gamma2/rst/rank_00000000/ks_background.00002.rst'
c=checkpoint(late);assert len(c['state'])==1
v=np.asarray(c['state'][0]).reshape(SHAPE)
seed=v/np.linalg.norm(v)
noise=np.random.default_rng(20260920).normal(size=SHAPE)
noise[22:]=0
tr=(noise[1]+noise[4]+noise[6])/3
for a in [1,4,6]:noise[a]-=tr
seed+=.001*noise/np.linalg.norm(noise)
seed/=np.linalg.norm(seed)
seed.tofile(ROOT/'seed.bin')
(ROOT/'manifest.json').write_text(json.dumps({'binary':str(BIN),'sha256':EXPECTED,'input':str(ROOT/'input.athinput'),'steps':8,'dt_M':.0375,'map_interval_M':.3,'epsilon':1e-4,'vectors':24,'seed_checkpoint':str(late),'seed_checkpoint_sha256':hashlib.sha256(late.read_bytes()).hexdigest(),'seed_checkpoint_time_M':c['time'],'seed_noise_relative_norm':.001,'random_seed':20260920,'OMP_NUM_THREADS':2,'OPENBLAS_NUM_THREADS':1,'scope':'Complete discrete CPU map; not a continuum or production stability result.'},indent=2)+'\n')

z=op.advance(steps=20,label='zero20')
small=seed*1e-4/np.max(abs(seed))
a=op.advance(small,1,label='composition1')
b=op.advance(a,1,label='composition1plus1')
d=op.advance(small,2,label='composition2')
validation={'zero20_all_bits_zero':bool(np.all(z.view(np.uint64)==0)),'composition':discrepancy(b,d)}
(ROOT/'map-validation.json').write_text(json.dumps(validation,indent=2)+'\n')
assert validation['zero20_all_bits_zero'] and validation['composition']['bitwise_equal'],validation
print('ZERO AND COMPOSITION PASS',flush=True)

m=24;q=seed.ravel();Q=np.empty((q.size,m+1));Q[:,0]=q;H=np.zeros((m+1,m));history=[]
for k in range(m):
    start=time.monotonic();w=op.response(Q[:,k].reshape(SHAPE),8,1e-4).ravel()
    for repeat in range(2):
        coeff=Q[:,:k+1].T@w;H[:k+1,k]+=coeff;w-=Q[:,:k+1]@coeff
    H[k+1,k]=np.linalg.norm(w);Q[:,k+1]=w/H[k+1,k]
    eig,Y=np.linalg.eig(H[:k+1,:k+1]);row={'k':k+1,'seconds':time.monotonic()-start,'ritz':[]}
    for index in np.argsort(-abs(eig))[:6]:
        mu=eig[index]
        row['ritz'].append({'real':float(mu.real),'imag':float(mu.imag),'gamma':float(np.log(abs(mu))/.3),'omega':float(np.angle(mu)/.3),'relative_residual':float(abs(H[k+1,k]*Y[-1,index])/abs(mu))})
    history.append(row);(ROOT/'progress.json').write_text(json.dumps(history,indent=2)+'\n');print(json.dumps(row),flush=True)
np.savez_compressed(ROOT/'krylov.npz',Q=Q,H=H)
results={'history':history,'modes':[]}
for index in np.argsort(-abs(eig))[:4]:
    mu=eig[index]
    if mu.imag< -1e-10:continue
    v=(Q[:,:m]@Y[:,index]).reshape(SHAPE).astype(complex);v/=np.max(abs(v))
    n=len(results['modes']);prefix=f'mode{n}'
    for part in ['real','imag']:getattr(v,part).tofile(ROOT/f'{prefix}-{part}.bin')
    def apply(steps,eps,label):
        r=op.response(v.real,steps,eps,label=label+'_real')
        if np.max(abs(v.imag))>1e-10:r=r+1j*op.response(v.imag,steps,eps,label=label+'_imag')
        return r
    r=apply(8,1e-4,prefix+'_direct')
    d=r-mu*v
    rec={'index':n,'mu_real':float(mu.real),'mu_imag':float(mu.imag),'gamma':float(np.log(abs(mu))/.3),'omega':float(np.angle(mu)/.3),'direct_relative_l2':float(np.linalg.norm(d)/np.linalg.norm(mu*v)),'direct_active_relative_l2':float(np.linalg.norm(d[ACTIVE])/np.linalg.norm((mu*v)[ACTIVE])),'profile_real':profile(v.real),'state_real':str(ROOT/f'{prefix}-real.bin'),'state_imag':str(ROOT/f'{prefix}-imag.bin')}
    results['modes'].append(rec)
(ROOT/'results.json').write_text(json.dumps(results,indent=2)+'\n')
print('SOLVE COMPLETE',flush=True)
