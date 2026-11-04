"""Propose static cuts from measured local work; predictions are not speedups."""
from pathlib import Path
import json,hashlib
import numpy as np
D=Path(__file__).resolve().parent;O=D/'measured-partition-candidate';O.mkdir(exist_ok=True)
cost=[];inputs={};counts=None
for segment,steps in ((1,12753),(2,26667-12753)):
    ranks=[]
    for rank in range(48):
        p=D/f'stationary-reference-v2-segment{segment:02d}/output/state.rank{rank}.generations.jsonl'
        inputs[str(p.relative_to(D))]=hashlib.sha256(p.read_bytes()).hexdigest()
        ranks.append(json.loads(p.read_text().splitlines()[-1]))
    c=np.array([r['active'] for r in ranks]);assert counts is None or np.array_equal(c,counts);counts=c
    # These two local intervals are called sequentially, excluding collective
    # synchronization/closure. This proxy is not total critical-path time.
    cost.append(np.array([r['nested_seconds']['rhs']+r['nested_seconds']['halo'] for r in ranks])/steps)
old=np.r_[0,np.cumsum(counts)];cum=np.r_[0,np.cumsum(cost[0])]
cuts=np.rint(np.interp(np.linspace(0,cum[-1],49),cum,old)).astype(int)
assert cuts[0]==0 and cuts[-1]==old[-1] and np.all(np.diff(cuts)>0)
pred=[]
for v in cost:
    work=np.diff(np.interp(cuts,old,np.r_[0,np.cumsum(v)]))
    pred.append(dict(observed_max_to_mean=float(v.max()/v.mean()),proxy_recut_max_to_mean=float(work.max()/work.mean()),observed_min_max=v[[np.argmin(v),np.argmax(v)]].tolist(),proxy_min_max=work[[np.argmin(work),np.argmax(work)]].tolist()))
record=dict(n=16,ranks=48,total=int(old[-1]),input_sha256=inputs,old_cuts=old.tolist(),proposed_cuts=cuts.tolist(),segment_predictions=pred,qualification='Piecewise-constant local RHS+halo cost proxy inside each old rank. Fit segment01; evaluate same proxy with segment02 timings. No new partition was run on Aurora. Halo dependency sets/communication/cache/occupancy change when cuts move, so proxy improvement is not measured speedup. The omitted boundary/collective costs may dominate. Requires static-SMR constructor parity and a matched measured runtime control before adoption. No dynamic Repartition call or scientific change.')
text=json.dumps(record,indent=2)+'\n';(O/'proposal.json').write_text(text);digest=hashlib.sha256(text.encode()).hexdigest()
(O/'candidate.partition').write_text(f'HYP3D_COST_PARTITION_V1 16 {old[-1]} 48 {digest}\n'+' '.join(map(str,cuts))+'\n')
print(json.dumps(dict(total=int(old[-1]),prediction=pred,descriptor_sha256=hashlib.sha256((O/'candidate.partition').read_bytes()).hexdigest()),indent=2))
