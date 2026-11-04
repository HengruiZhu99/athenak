"""Separate diagnostic interpolation error from the preserved late-time state."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path(__file__).resolve().parent;ROOT=next(p for p in D.parents if (p/'CMakeLists.txt').exists());R=ROOT/'output/refined-convergence/endpoint-late-frame-control'
q=json.loads((R/'stationary6.json').read_text());old=json.loads((D/'stationary6-terminal/endpoint.json').read_text());raw=ROOT/'output/refined-convergence/aurora/campaigns/stationary6-n16-v2-segment01/output/state.checkpoint';v=json.loads((D/'stationary6-terminal/independent-verification.json').read_text());h=hashlib.sha256(raw.read_bytes()).hexdigest();assert h==v['checkpoint']['sha256'];assert q['steps']==old['steps'] and q['identity']==old['identity'];assert np.allclose(q['production_full_integrals'],old['integrals'][0],rtol=1e-12,atol=0)
a=np.array(old['integrals']);b=np.array(q['integrals']);before=np.sqrt(a[:,1:]/a[:,:1]);after=np.sqrt(b[:,1:]/b[:,:1]);out=D/'late-frame-control';out.mkdir(exist_ok=True)
result=dict(passed=True,input_checkpoint_sha256=h,production_original_worker_admitted=False,independent_endpoint_verification=True,original_norms_full_cut03=before.tolist(),analysis_frame_norms_full_cut03=after.tolist(),ratios=(after/before).tolist(),analysis_override=q['analysis_override'],replaced_ghosts=q['replaced_ghosts'],maximum_ghost_change=q['maximum_ghost_change'],cpu_native_integral_relative_error=q['native_volume_integral_relative_difference'],qualification='Read-only postprocessing change of halo interpolation only. Same admitted-for-analysis old10M state, unchanged active fields, stencil weights and input identity. Does not undo evolution errors or establish slicing regularity. Outside the changed-Omega region most halos are unchanged by construction; this control rules out this specific reconstruction correction as an explanation of the large late exterior norms, not every possible diagnostic discretization error.')
(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');shutil.copy2(R/'stationary6.json',out/'endpoint.json');shutil.copy2(R/'build.json',out/'build.json')
fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for ax,col,label in zip(axes,[3,4],['Hamiltonian','Momentum']):
 for data,title,style in [(old,'Original native interpolation','-'),(q,'Frame interpolation in analysis only','--')]:
  bins=np.array(data['radial_bins']);ax.semilogy((bins[:,0]+bins[:,1])/2,np.where(bins[:,2]>0,bins[:,col],np.nan),style,label=title)
 ax.axvspan(.24,.30,color='C2',alpha=.12,label='Changed Omega frame');ax.axvspan(.36,.65,color='C3',alpha=.08,label='Gauge transition');ax.axvline(.3,color='k',ls=':',lw=.7)
 ax.set(xlim=(.15,1),xlabel='compact r',ylabel=label+' squared integral per radial bin');ax.grid(alpha=.25);ax.legend(fontsize=8)
fig.suptitle('Same preserved stationary6 state at10M: late exterior errors survive this halo-only correction',fontsize=11);fig.savefig(out/'radial-comparison.png',dpi=160);fig.savefig(out/'radial-comparison.pdf');plt.close(fig);print(json.dumps(result,indent=2))
