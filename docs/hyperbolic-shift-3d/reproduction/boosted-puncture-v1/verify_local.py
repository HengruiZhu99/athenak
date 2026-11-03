"""Verify bounded native initialization, MPI tracker and restart controls."""
from pathlib import Path
import hashlib,json,struct,csv
import numpy as np
ROOT=Path(__file__).resolve().parents[4];RAW=ROOT/'output/boosted-puncture/checks';OUT=Path(__file__).parent
names=['n16-two-step','one-step','restart-two-step','mpi2-two-step']
def identity(p):return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
states={};records={}
for name in names:
 p=RAW/(name+'.checkpoint');data=p.read_bytes();h=struct.unpack('<16s4i3Q2d',data[:72]);boost=struct.unpack('<5d',data[168:208]);a=np.frombuffer(data[208:],'<f8')
 assert h[0].rstrip(b'\0')==b'HYP3D_MPI_BX1' and h[1:5]==(16,31,2,1)
 assert len(data)==208+16*(31*h[5]+h[6]) and np.isfinite(a).all()
 assert boost[0]==.1 and boost[-1]==.16 and all(abs(v)<.16 for v in boost[1:4])
 r=json.loads((RAW/(name+'.report.json')).read_text());hist=list(csv.DictReader((RAW/(name+'.history.csv')).open()))
 assert r['status']=='completed' and r['checkpoint_admitted'] and r['puncture_tracker']
 assert r['max_null_surface']<1e-10 and r['max_null_ghost']<1e-10
 assert [float(hist[-1]['puncture_'+k]) for k in 'xyz']==list(boost[1:4])
 assert abs(boost[2])<1e-15 and abs(boost[3])<1e-15 and boost[1]>0
 states[name]=a
 records[name]=dict(checkpoint=identity(p),time=h[8],steps=h[7],puncture_position=boost[1:4],horizon_mass=float(hist[-1]['horizon_mass']),horizon_found=hist[-1]['horizon_found'],report_header_bytes=r['checkpoint_header_bytes'],max_cfl=r['max_cfl'],null_surface=r['max_null_surface'],null_ghost=r['max_null_ghost'],files={str(p.relative_to(ROOT)):identity(p) for p in RAW.glob(name+'.*')})
comparisons={}
for name in ['restart-two-step','mpi2-two-step']:
 a=states[name];b=states['n16-two-step'];err=float(np.max(abs(a-b)/np.maximum(1,np.maximum(abs(a),abs(b)))))
 # Includes volume/surface compensation; physical fields and tracker are checked separately.
 assert err<1e-10,(name,err)
 comparisons[name]=dict(max_scaled_payload_difference=err,tracker_difference=float(np.max(abs(np.array(records[name]['puncture_position'])-records['n16-two-step']['puncture_position']))))
result=dict(passed=True,scope='SERIAL backend; one/two MPI ranks; not GPU or long-time stability',records=records,comparisons=comparisons,qualification='First two-step report predates reporting-only correction from168 to208; actual BX1 binary header is208 and independently checked. No checkpoint format changed.')
(OUT/'local-verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(comparisons,indent=2))
