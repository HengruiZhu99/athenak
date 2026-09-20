#!/usr/bin/env python3
"""Complete-step central response for the isolated fixed-mesh vacuum hook.
Raw arrays include all 25 residual fields and physical ghost cells. No restart
loader, ghost extrapolation, or projection is inserted between composed maps.
"""
import hashlib, json, os, shutil, subprocess, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
BINARY=ROOT/'athena-mode-analysis'
INPUT=ROOT/'input.athinput'
SHAPE=(25,24,24,24)
DT=.075
FIELDS=['chi','gxx','gxy','gxz','gyy','gyz','gzz','Khat','Axx','Axy','Axz','Ayy','Ayz','Azz','Gamx','Gamy','Gamz','Theta','alpha','betax','betay','betaz','Bx','By','Bz']
ACTIVE=(slice(None),slice(4,20),slice(4,20),slice(4,20))
ENV=dict(os.environ,OMP_NUM_THREADS='2',OMP_PROC_BIND='false',OPENBLAS_NUM_THREADS='1')
class Operator:
    def __init__(self, label='calls', overrides=(), binary=None, dt=DT, input_file=None):
        self.directory=ROOT/label;self.directory.mkdir(exist_ok=True)
        self.overrides=tuple(overrides);self.binary=Path(binary) if binary else BINARY;self.dt=dt;self.input=Path(input_file) if input_file else INPUT
        self.count=0
    def advance(self, state=None, steps=1, label=None, overrides=()):
        self.count+=1
        key=label or f'call_{self.count:05d}'
        folder=self.directory/key
        folder.mkdir(exist_ok=True)
        cmd=[str(self.binary),'-i',str(self.input),f'time/nlim={steps}',f'problem/mode_output_state={folder}/output.bin']
        if state is not None:
            state=np.asarray(state,dtype=np.float64).reshape(SHAPE)
            state.tofile(folder/'input.bin')
            cmd.append(f'problem/mode_input_state={folder}/input.bin')
        cmd.extend(self.overrides);cmd.extend(overrides)
        start=time.monotonic()
        with (folder/'run.log').open('w') as f:
            p=subprocess.run(cmd,cwd=folder,env=ENV,stdout=f,stderr=subprocess.STDOUT,timeout=180)
        if p.returncode:
            raise RuntimeError(f'{folder}: exit {p.returncode}')
        meta=json.loads((folder/'output.bin.json').read_text())
        output=np.fromfile(folder/'output.bin',dtype=np.float64).reshape(SHAPE)
        log=(folder/'run.log').read_text()
        if not np.isfinite(output).all() or 'Z4C_INVALID_STATE' in log:
            raise RuntimeError(f'Invalid output {folder}')
        if meta['cycle']!=steps or abs(meta['time']-self.dt*steps)>1e-11 or abs(meta['dt']-self.dt)>1e-14:
            raise RuntimeError(f'Unexpected step metadata {meta}')
        rec=dict(label=key,steps=steps,wall_seconds=time.monotonic()-start,metadata=meta,max_abs=float(abs(output).max()),command=cmd)
        with (self.directory/'calls.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
        if label is None:
            (folder/'input.bin').unlink(missing_ok=True);(folder/'output.bin').unlink()
        return output
    def response(self,v,steps=40,epsilon=1e-6,label=None):
        v=np.asarray(v,dtype=np.float64).reshape(SHAPE)
        peak=np.max(abs(v))
        if peak==0:return np.zeros_like(v)
        vp=v*(epsilon/peak)
        plus=self.advance(vp,steps,label=label+'_plus' if label else None)
        minus=self.advance(-vp,steps,label=label+'_minus' if label else None)
        return (plus-minus)*(peak/(2*epsilon))
def discrepancy(a,b):
    d=a-b
    return dict(max_abs=float(np.max(abs(d))),relative_l2=float(np.linalg.norm(d)/max(np.linalg.norm(a),np.finfo(float).tiny)),relative_max=float(np.max(abs(d))/max(np.max(abs(a)),np.finfo(float).tiny)),bitwise_equal=bool(np.array_equal(a.view(np.uint64),b.view(np.uint64))),active_max=float(np.max(abs(d[ACTIVE]))))
def profile(v):
    out={}
    for n,name in enumerate(FIELDS):
        a=v[n];idx=np.unravel_index(np.argmax(abs(a)),a.shape)
        ai=a[4:20,4:20,4:20];ii=np.unravel_index(np.argmax(abs(ai)),ai.shape)
        xyz=[-.0+(q-3.5)*.25-2 for q in idx[::-1]]
        ax=[(q+.5)*.25-2 for q in ii[::-1]]
        out[name]=dict(l2=float(np.linalg.norm(a)),active_l2=float(np.linalg.norm(ai)),max=float(abs(a[idx])),max_xyz=xyz,active_max=float(abs(ai[ii])),active_max_xyz=ax)
    return out
