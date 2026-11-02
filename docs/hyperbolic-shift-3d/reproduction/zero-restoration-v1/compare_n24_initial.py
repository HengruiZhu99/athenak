"""Compare preserved t=0 native A volume fields with SG1's actual text input."""
import hashlib,json,struct
from pathlib import Path
import numpy as np
R=Path.cwd();D=R/'docs/hyperbolic-shift-3d/reproduction/zero-restoration-v1/checks'
cp=R/'output/native-shared-transition/initial-n24/puncture.checkpoint'
txt=R/'output/hyperbolic-core-shell/short-gauge-v1/retired-before-n32/inputs/puncture-continuous-n24-smooth-v1.txt'
p=json.loads((D.parents[1]/'native-shared-transition-v1/checks/initial-grid/n24-provenance.json').read_text())
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
assert sha(cp)==p['sha256'] and cp.stat().st_size==p['bytes']
with cp.open('rb') as f:h=struct.unpack('<16s4i3Q2d',f.read(72))
assert h[0].rstrip(b'\0')==b'HYP3D_MPI_NI1' and h[1:5]==(24,31,2,1) and h[7]==h[8]==0
with txt.open() as f:count=int(f.readline());text=np.fromfile(f,sep=' ')
assert count==text.size==h[5]*31
native=np.fromfile(cp,dtype='<f8',offset=168,count=count).reshape(-1,31);text=text.reshape(-1,31)
absolute=np.max(np.abs(native-text),axis=0);scale=1+np.maximum(np.max(np.abs(native),axis=0),np.max(np.abs(text),axis=0))
r={'scope':'Preserved native NI1 t=0 A volume fields versus actual SG1 input. Same native default initializer retained by NZ1. No evolution or cross-formulation restart; does not bound long-time amplification.', 'native_checkpoint_sha256':sha(cp),'SG1_text_sha256':sha(txt),'volume_cells':h[5],'max_absolute_by_field':absolute.tolist(),'scaled_max_by_field':(absolute/scale).tolist(),'largest_shift_absolute_difference':float(max(absolute[22:25])),'largest_nonshift_absolute_difference':float(max(np.r_[absolute[:22],absolute[25:]]))}
(D/'n24-initial-field-difference.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
