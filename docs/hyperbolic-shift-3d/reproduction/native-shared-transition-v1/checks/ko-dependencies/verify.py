"""Read-only regression: corrected distributed KO10 matches frozen serial data."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[6]
HERE=Path(__file__).resolve().parent
raw=ROOT/'output/native-shared-transition'
identity=lambda p:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def values(p):return np.fromfile(p,dtype='<f8',offset=168)
def physical_difference(a,b):
    x,y=values(a),values(b);assert len(x)==len(y)
    nc,ns=159200,1536
    ix=np.r_[np.arange(31*nc),np.arange(62*nc,62*nc+31*ns)]
    return float(np.max(np.abs(x[ix]-y[ix])/(1+np.maximum(np.abs(x[ix]),np.abs(y[ix])))))
records=[]
for kind in ['minkowski','puncture']:
    ref=raw/f'serial-matrix-smoke/attempt02/c-{kind}.checkpoint'
    fixed=raw/f'ko-dependency-audit/fixed-twelve-rank/{kind}.checkpoint'
    gpu=raw/f'aurora/campaigns/n16-native-transition-v1-c-smoke-segment01/output/{kind}.checkpoint'
    assert ref.read_bytes()==fixed.read_bytes(),kind+' corrected partition discrepancy'
    records.append(dict(kind=kind,reference=identity(ref),fixed_12_rank=identity(fixed),
        old_gpu_12_rank=identity(gpu),fixed_checkpoint_byte_identical=True,old_gpu_difference=physical_difference(ref,gpu)))
old=raw/'ko-dependency-audit/old-twelve-rank/minkowski.checkpoint'
ref=raw/'serial-matrix-smoke/attempt02/c-minkowski.checkpoint'
gpu=raw/'aurora/campaigns/n16-native-transition-v1-c-smoke-segment01/output/minkowski.checkpoint'
assert physical_difference(old,ref)>1e-4 and physical_difference(old,gpu)<1e-10
result=dict(scope='Two-step N16 production KO10 MPI partition regression; not long accuracy or convergence.',
    old_serial12_vs_serial1=physical_difference(old,ref),old_serial12_vs_gpu12=physical_difference(old,gpu),
    old_serial12=identity(old),cases=records)
(HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
