"""Read-only per-cell sign localization in an already admitted RF2 endpoint."""
from pathlib import Path
import json,struct,hashlib,csv
import numpy as np
D=Path(__file__).resolve().parent;R=next(p for p in D.parents if (p/'CMakeLists.txt').exists())
S=D/'stationary-reference-v2-segment02';O=D/'reference-sign-localization';O.mkdir(exist_ok=True)
P=R/'output/refined-convergence/stationary-reference-v2-aurora/campaigns/stationary20-reference-n16-segment02/output/state.checkpoint'
verified=json.loads((S/'local-verification.json').read_text());term=json.loads((S/'terminal-verification.json').read_text());assert term['admitted']
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
digest=sha(P);assert digest==verified['checkpoint_sha256']==term['checkpoint']['sha256']
with P.open('rb') as f:
    def zero():
        out=b''
        while True:
            c=f.read(1);assert c
            if c==b'\0':return out.decode()
            out+=c;assert len(out)<16384
    def get(fmt):return struct.unpack('<'+fmt,f.read(struct.calcsize('<'+fmt)))
    assert zero()=='HYP3D_AMR_RF2;checksum-fnv1a64;native31;configured-normalized-boost;'
    identity=f.read(get('Q')[0]).decode();assert identity==term['checkpoint']['identity']
    fields=[zero() for _ in range(31)];assert fields[0]=='chi' and fields[18]=='alpha-(1+r2)/2'
    n,kind,sub=get('3i');steps,regrids=get('2Q');time,dt,seed=get('3d');tracker=get('3d');floors=get('8d');diagnostics=get('8d')
    count=get('Q')[0];leaves=np.frombuffer(f.read(16*count),dtype='<i4').reshape(-1,4);f.read(4*count)
    arrays=[]
    for _ in range(8):
        length=get('Q')[0];arrays.append(np.memmap(P,dtype='<f8',mode='r',offset=f.tell(),shape=(length,)));f.seek(8*length,1)
    assert get('Q')[0]==0x414d5231434f4d50;get('Q');assert f.tell()==P.stat().st_size
assert n==16 and regrids==0 and steps==26667 and len(arrays[7])==0
events=arrays[5].reshape(-1,16);active=np.flatnonzero(np.any(events[:,[0,1,2,3,8,9,10,11]]!=0,axis=1))
assert np.all(active<count*n**3),'Noncore events need native shell mapping'
block=active//n**3;idx=active%n**3;ijk=np.column_stack((idx%n,idx//n%n,idx//(n*n)))
h=.8/(2.**leaves[block,3]);xyz=-.4+leaves[block,:3]*h[:,None]+(ijk+.5)*h[:,None]/n
radius=np.linalg.norm(xyz,axis=1)
history=[json.loads(x) for x in (D/'stationary-reference-v2-completed/history.jsonl').read_text().splitlines()]
report=json.loads((S/'output/state.report.json').read_text());totals=events[:,[0,1,2,3,8,9,10,11]].sum(axis=0).tolist();assert totals==report['cumulative_bulk_sign_counts']
summary=dict(checkpoint_sha256=digest,steps=steps,t_over_M=3*time,event_cells=len(active),all_events_cartesian=True,nonpositive_chi_radius_range=[float(radius.min()),float(radius.max())],bounding_box=[xyz.min(axis=0).tolist(),xyz.max(axis=0).tolist()],first_negative_M_range=[float(3*events[active,6].min()),float(3*events[active,6].max())],bulk_sign_counts=totals,surface_sign_counts=report['cumulative_surface_sign_counts'],boundary_sign_counts=report['cumulative_boundary_sign_counts'],complete_candidate_enclosing_radius_range=[min(x['horizon_max_radius'] for x in history if x['horizon_complete']),max(x['horizon_max_radius'] for x in history if x['horizon_complete'])],qualification='Cumulative attempted RK-stage/final candidates, not negative admitted fields. Coordinates reconstructed from exact static logical leaves; no per-event horizon surface is available. An enclosing radius does not prove every event lies inside a distorted horizon. These central sign events do not directly coincide with exterior r>.3 error peak at.90-.95, but causal influence is not excluded.')
with (O/'cells.csv').open('w') as f:
    w=csv.writer(f);w.writerow(['active_id','x','y','z','r','negative_stage','negative_final','minimum_chi_candidate','first_negative_t_over_M'])
    for p,x,r in zip(active,xyz,radius):w.writerow([p,*x,r,events[p,0],events[p,1],events[p,4],3*events[p,6]])
(O/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n');assert sha(P)==digest;print(json.dumps(summary,indent=2))
