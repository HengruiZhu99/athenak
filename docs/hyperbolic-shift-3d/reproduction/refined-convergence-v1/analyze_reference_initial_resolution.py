"""Compare actual native initial diagnostics on the matched enlarged SMR family."""
from pathlib import Path
import json,hashlib,subprocess,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'CMakeLists.txt').exists())
DOC=Path(__file__).resolve().parent;OUT=DOC/'reference-initial-resolution';OUT.mkdir(exist_ok=True)
RAW=ROOT/'output/refined-convergence/reference-initial-resolution'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
rows={};inputs={};identity=None
for n in (12,14,16):
 prefix=RAW/f'n{n}' if n!=16 else ROOT/'output/refined-convergence/reference-v2-stationary-n16-mpi2'
 history=Path(str(prefix)+'.history.jsonl');report=Path(str(prefix)+'.report.json');cp=Path(str(prefix)+'.checkpoint')
 r=json.loads(report.read_text());allrows=[json.loads(x) for x in history.read_text().splitlines()];x=allrows[0]
 assert r['status']=='completed' and r['checkpoint_admitted'] and r['n']==n and r['leaves']==232
 assert x['step']==0 and x['time']==0 and x['horizon_found'] and x['horizon_complete']
 assert x['horizon_max_radius']<.1 and all(abs(q['cut']-c)<1e-15 for q,c in zip(x['fixed'],(.1,.3,.6)))
 assert abs(x['penalty_cfl']-.3125)<1e-10 and abs(x['penalty']-.6510416666660156)<1e-15
 if identity is None:identity=r['identity']
 assert identity==r['identity']
 if n!=16:
  assert len(allrows)==1 and r['steps']==0 and r['time']==0
  verified=json.loads(subprocess.check_output([str(ROOT/'output/refined-convergence/verify-checkpoint'),str(cp)],text=True))
  assert verified['verified'] and verified['identity']==identity and verified['n']==n and verified['steps']==0 and verified['time']==0 and verified['bytes']==r['checkpoint_bytes']==cp.stat().st_size
  (OUT/f'n{n}-checkpoint-verification.json').write_text(json.dumps(verified,indent=2)+'\n')
  shutil.copy2(RAW/f'n{n}-command.json',OUT/f'n{n}-command.json')
  shutil.copy2(RAW/f'n{n}-stdout.log',OUT/f'n{n}-stdout.log')
 else:
  # Reuse completed two-step admission; only its first history row is used.
  verified=json.loads((DOC/'stationary-reference-v2-checks/checkpoint-verification.json').read_text())
  expected=json.loads((DOC/'stationary-reference-v2-checks/verification.json').read_text());assert sha(cp)==expected['checkpoint_sha256']
  assert sha(history)==sha(DOC/'stationary-reference-v2-checks/history.jsonl') and sha(report)==sha(DOC/'stationary-reference-v2-checks/report.json')
 for p in (history,report):shutil.copy2(p,OUT/f'n{n}-{p.suffix[1:] if p.suffix==".jsonl" else "report.json"}')
 inputs[n]={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in (history,report,cp)}
 rows[n]=x
orders={}
for ci,cut in enumerate((.1,.3,.6)):
 orders[str(cut)]={}
 for field in ('H','M','Mr'):
  E=np.array([rows[n]['fixed'][ci][field] for n in (12,14,16)])
  orders[str(cut)][field]=dict(values=E.tolist(),pair12_14=float(np.log(E[0]/E[1])/np.log(14/12)),pair14_16=float(np.log(E[1]/E[2])/np.log(16/14)),fit=float(-np.polyfit(np.log([12,14,16]),np.log(E),1)[0]))
result=dict(identity=identity,inputs=inputs,initial_rows=rows,orders=orders,qualification='Native t0 geometry diagnostics at N12/N14/N16 on the same configured SMR levels and shells, one-rank Serial N12/N14 and existing two-rank Serial N16. Same coordinate cuts; all initial horizons are inside .1. No new evolution, GPU comparison, field self-convergence or evolved accuracy claim. Step sizes scale as 1/N solely to retain the same initial scri penalty CFL; zero evolution steps at N12/N14. Original scri operator remains and may limit asymptotic order.')
(OUT/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
fig,axes=plt.subplots(1,2,figsize=(10.5,4),layout='constrained')
for ax,cut in zip(axes,(.3,.6)):
 for f,marker in zip(('H','M','Mr'),('o','s','^')):
  q=orders[str(cut)][f];ax.loglog([12,14,16],q['values'],marker=marker,label=f"{f}: fitted p={q['fit']:.2f}")
 reference=np.array([12,14,16.]);anchor=orders[str(cut)]['H']['values'][-1];ax.loglog(reference,anchor*(16/reference)**4,ls='--',color='gray',label='N⁻⁴ guide (anchored H16)')
 ax.set(title=f'Initial native constraints, r > {cut}',xlabel='N',ylabel='Constraint RMS');ax.set_xticks([12,14,16],labels=['12','14','16']);ax.minorticks_off();ax.grid(alpha=.25);ax.legend(fontsize=8)
fig.savefig(OUT/'initial-orders.png',dpi=170);fig.savefig(OUT/'initial-orders.pdf');plt.close(fig)
print(json.dumps(orders,indent=2))
