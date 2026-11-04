"""Compare nominal Cartesian cells in the admitted same-dt radial pair."""
from pathlib import Path
import hashlib
import json
import struct
import numpy as np

D=Path(__file__).resolve().parent
R=next(p for p in D.parents if (p/'CMakeLists.txt').exists())

def read_case(name):
    S=D/f'{name}-control-segment01'
    terminal=json.loads((S/'terminal-verification.json').read_text())
    assert terminal['admitted'] and terminal['checkpoint']['status']=='completed'
    P=R/f'output/refined-convergence/stationary-reference-v2-aurora/campaigns/stationary20-reference-{name}-dt4-n16-segment01/output/state.checkpoint'
    h=hashlib.sha256()
    with P.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    assert h.hexdigest()==terminal['checkpoint']['sha256']
    with P.open('rb') as f:
        def zero():
            text=b''
            while True:
                c=f.read(1);assert c
                if c==b'\0':return text.decode()
                text+=c;assert len(text)<16384
        def get(fmt):return struct.unpack('<'+fmt,f.read(struct.calcsize('<'+fmt)))
        assert zero()=='HYP3D_AMR_RF2;checksum-fnv1a64;native31;configured-normalized-boost;'
        identity=f.read(get('Q')[0]).decode();assert identity==terminal['checkpoint']['identity']
        fields=[zero() for _ in range(31)]
        n,kind,sub=get('3i');steps,regrids=get('2Q');time,dt,seed=get('3d')
        tracker=get('3d');floors=get('8d');diagnostics=get('8d')
        leaves=get('Q')[0];topology=f.read(20*leaves)
        arrays=[]
        for _ in range(8):
            length=get('Q')[0]
            arrays.append(np.memmap(P,dtype='<f8',mode='r',offset=f.tell(),shape=(length,)))
            f.seek(8*length,1)
        assert get('Q')[0]==0x414d5231434f4d50;get('Q');assert f.tell()==P.stat().st_size
    assert n==16 and kind==2 and sub==4 and steps==6400 and regrids==0
    assert time==.2 and dt==.00003125 and leaves==232
    return dict(hash=h.hexdigest(),fields=fields,topology=topology,core=leaves*n**3,
                arrays=arrays,tracker=tracker,floors=floors,seed=seed)

a,b=map(read_case,('time','radial'))
assert a['topology']==b['topology'] and a['fields']==b['fields'] and a['core']==b['core']
out=dict(checkpoint_sha256={'time':a['hash'],'radial':b['hash']},
         nominal_cartesian_cells=a['core'],comparisons={},
         tracker_equal=a['tracker']==b['tracker'],
         cumulative_floors_equal=a['floors']==b['floors'],
         saved_horizon_seed_equal=a['seed']==b['seed'])
for label,index,width in [('primary_fields',0,31),('RK_compensation',1,31),
                           ('cell_floor_counts',4,1),('cell_sign_statistics',5,16)]:
    x=a['arrays'][index][:a['core']*width].reshape(-1,width)
    y=b['arrays'][index][:b['core']*width].reshape(-1,width)
    change=np.max(abs(y-x)/(1+abs(x)),axis=0)
    out['comparisons'][label]=dict(byte_equal=x.tobytes()==y.tobytes(),
        max_scaled_change=float(change.max()),per_component_max_scaled_change=change.tolist())
out['qualification']='Admitted same-dt .6M endpoints. Identical nominal Cartesian block topology and cell coordinates. Comparison covers nominal Cartesian cells only, excluding appended evolved-overlap rows; no inference about all halo values or future causal influence. Different radial shells and their interface placement remain the only physical-grid changes.'
O=D/'short-radial-controls-comparison';O.mkdir(exist_ok=True)
(O/'nominal-core-comparison.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
