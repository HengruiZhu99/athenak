from pathlib import Path
import json,sys
import numpy as np
case=Path('/lus/flare/projects/CompactBinaryMerger/hzhu/n12-full-scri-20261115');sys.path.insert(0,str(case/'analysis-source/tst/z4c_hyperboloidal'))
from restart_roundtrip import parse,payload
r=json.loads((case/'long-trajectory-26M.json').read_text());a=parse(Path(r['reference']));b=parse(Path(r['candidate']));x=payload(a);y=payload(b)
maximum=np.zeros(39);sqdiff=np.zeros(31);sqref=np.zeros(31);scale=np.zeros(39);surface=np.zeros(39);location=None;largest=0
bulk=(12+6)**3
for block in range(x.shape[0]):
 p=x[block];q=y[block];assert np.isfinite(p).all() and np.isfinite(q).all()
 delta=abs(q-p);maximum=np.maximum(maximum,delta.max(axis=0));scale=np.maximum(scale,abs(p).max(axis=0));sqdiff+=np.sum(delta[:,:31]**2,axis=0);sqref+=np.sum(p[:,:31]**2,axis=0)
 surface=np.maximum(surface,delta[bulk:].max(axis=0))
 if delta.max()>largest:
  slot,field=np.unravel_index(delta.argmax(),delta.shape);largest=float(delta[slot,field]);location=dict(block=block,slot=int(slot),field=int(field),reference=float(p[slot,field]),candidate=float(q[slot,field]))
r['field_max_abs_difference']=maximum.tolist();r['field_reference_max_abs']=scale.tolist();r['field_relative_L2_difference']=np.sqrt(sqdiff/np.maximum(sqref,1e-300)).tolist();r['surface_max_abs_difference']=surface.tolist();r['largest_difference_location']=location
(case/'long-trajectory-26M.json').write_text(json.dumps(r,indent=2)+'\n')
print('largest',location);print('physical field maxima',maximum[:31]);print('relativeL2',np.sqrt(sqdiff[:31]/np.maximum(sqref[:31],1e-300)));print('scri maxima',surface[:31])
