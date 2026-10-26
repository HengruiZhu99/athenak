#!/usr/bin/env python3
import sys,json,math,hashlib
from pathlib import Path
import numpy as np
np.seterr(all='raise')
p=Path(sys.argv[1]);fail=[];records=[]
def interp(x,y,xx):
 out=np.zeros((len(xx),y.shape[1]));width=8
 for k,q in enumerate(xx):
  start=max(0,min(len(x)-width,int(np.searchsorted(x,q))-width//2));nodes=x[start:start+width];weights=np.ones(width)
  for i in range(width):
   for j in range(width):
    if i!=j:weights[i]*=(q-nodes[j])/(nodes[i]-nodes[j])
  out[k]=np.einsum('i,ij->j',weights,y[start:start+width])
 return out
def norms(x,h):return {'L2':float(np.sqrt(np.sum(h[:,None]*x*x))),'Linf':float(max(abs(x.ravel())))}
for family in ['synthetic','extended','massive']:
 data={};receipts=[]
 for n in [32,48,64,128,160]:
  tag=f'{family}-N{n}';d=np.load(p/(tag+'.npz'));r=json.loads((p/(tag+'.json')).read_text());assert hashlib.sha256((p/(tag+'.npz')).read_bytes()).hexdigest()==r['payload_sha256'];data[n]=d;receipts.append(r)
 tag=family+'-N48-half';half=np.load(p/(tag+'.npz'));r=json.loads((p/(tag+'.json')).read_text());assert hashlib.sha256((p/(tag+'.npz')).read_bytes()).hexdigest()==r['payload_sha256'];receipts.append(r)
 for r in receipts:
  for key,gate in [('energy_identity_error',1e-10),('energy_balance_error',1e-7),('high_k_local_RK_error',1e-5)]:
   if r[key]>gate:fail.append([family,r['nodes'],r['half'],key,r[key],gate])
 for tindex,t in enumerate([.005,.01,.02]):
  for field in ['U','Q']:
   errors={}
   for n in [32,48,64]:
    errors[n]=norms(data[n][field][tindex]-interp(data[160]['x'],data[160][field][tindex],data[n]['x']),data[n]['h'])
   ref=norms(interp(data[128]['x'],data[128][field][tindex],data[64]['x'])-interp(data[160]['x'],data[160][field][tindex],data[64]['x']),data[64]['h'])
   temporal=norms(half[field][tindex]-data[48][field][tindex],data[48]['h']);spatial=[norms(data[48][field][tindex]-interp(data[n]['x'],data[n][field][tindex],data[48]['x']),data[48]['h']) for n in [32,64]]
   order=math.log(errors[48]['L2']/errors[64]['L2'])/math.log(64/48)
   rr=dict(family=family,time=t,field=field,errors=errors,reference_difference=ref,temporal_difference=temporal,spatial_differences_at48=spatial,finest_L2_order=order)
   for norm in ['L2','Linf']:
    if not errors[32][norm]>errors[48][norm]>errors[64][norm]:fail.append([family,t,field,norm,'nonmonotonic'])
    if ref[norm]>.1*errors[64][norm]:fail.append([family,t,field,norm,'reference_unresolved',ref[norm]/errors[64][norm]])
    for i,sp in enumerate(spatial):
     if temporal[norm]>.1*sp[norm]:fail.append([family,t,field,norm,'time_unresolved',i,temporal[norm]/sp[norm]])
   if order<2:fail.append([family,t,field,'L2_order',order])
   records.append(rr)
r=dict(passed=not fail,failures=fail,records=records,scope='Short source-coupled prescribed-coefficient1D surrogate, not physical or oblique reflection stability',energy_runs=18,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(p/'review.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print(json.dumps({'passed':r['passed'],'failures':fail,'min_L2_order':min(x['finest_L2_order'] for x in records)},indent=2))
