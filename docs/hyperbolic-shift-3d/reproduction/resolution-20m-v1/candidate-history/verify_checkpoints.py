#!/usr/bin/env python3
"""Compare CPU diagnostic replays with the original GPU DT1 endpoint payloads."""
from pathlib import Path
import hashlib
import json
import struct
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'tst/hyperbolic_shift_3d').exists())
REPLAY=ROOT/'output/hyperbolic-core-shell/candidate-history-aurora-v1/output'
original={16:ROOT/'output/hyperbolic-core-shell/n16-timestep-to3m-v1/n16-original-dt-reproduction-v1/output/puncture.checkpoint',
          24:ROOT/'output/hyperbolic-core-shell/resolution-20m-v1/n24-dt1-segment01/output/puncture.checkpoint'}
expected={16:'628c2109bdf7ca6624afff58b65e8a28baca724e9e73de17b517454b26fe6a31',
          24:'f43b0bc9081ea8bb923a34fbad567ef10ef183932d742fc3da1636865d22f9ef'}
records={}
for n,p in original.items():
    q=REPLAY/f'n{n}.checkpoint'
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    assert sha(p)==expected[n]
    with p.open('rb') as f:hp=f.read(152)
    with q.open('rb') as f:hq=f.read(152)
    assert hp==hq and p.stat().st_size==q.stat().st_size
    header=struct.unpack('<16s4i3Q2d',hp[:72])
    assert header[0].rstrip(b'\0')==b'HYP3D_MPI_DT1' and header[4]==1 and header[-2]==1.
    cells,surface=header[5:7];counts=[31*cells,31*cells,surface,surface]
    a=np.memmap(p,dtype='<f8',mode='r',offset=152)
    b=np.memmap(q,dtype='<f8',mode='r',offset=152)
    offset=0;parts={}
    for name,count in zip(['volume_state','volume_compensation','surface_state','surface_compensation'],counts):
        av=a[offset:offset+count];bv=b[offset:offset+count]
        assert np.isfinite(bv).all()
        delta=np.abs(av-bv);scaled=delta/(1+np.abs(av))
        parts[name]=dict(max_absolute=float(delta.max()),max_scaled=float(scaled.max()),bitwise_equal=bool(np.array_equal(av,bv)))
        assert scaled.max()<1e-8,(n,name,parts[name])
        offset+=count
    records[n]=dict(original=str(p.relative_to(ROOT)),original_sha256=expected[n],
        replay=str(q.relative_to(ROOT)),replay_sha256=sha(q),header_identical=True,parts=parts,
        tolerance='max |replay-original|/(1+|original|) < 1e-8; CPU/GPU and rank-count differences allowed')
(HERE/'checkpoint-comparison.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
