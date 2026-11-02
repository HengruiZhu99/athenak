"""Read-only admitted endpoint diagnostic stencil control; campaign binaries unchanged."""
import hashlib,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[6]
OUT=Path(__file__).resolve().parent
cuts='0,0.03,0.06,0.1099,0.13,0.16,0.18,0.2,0.25,0.3,0.4,0.45,0.55,0.6,0.675,0.7,0.725,0.75,0.775,0.8,0.85,0.9,0.95,1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
records=[]
for case,suffix,expected in [('c','control_ko10','68da8cf6f560ec1e91eb5268e0b4d57c20df76c1b8a4a29353343223c0934689'),('d','control_ko10_scri6','4472e9e9b7cfa7c48afb3664465d534b20eec86ddde5b3d48b3e8f4fec276f85')]:
 cp=ROOT/f'output/native-shared-transition/aurora/campaigns/n16-native-transition-ko-dependencies-v1-{case}-evolution-segment01/output/puncture.checkpoint'
 binary=ROOT/f'build-native-diagnostic-operators/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_checkpoint_constraints_native_{suffix}'
 assert sha(cp)==expected
 for mode in ['native','first4','first6']:
  output=OUT/f'{case}-{mode}.json'
  command=[str(binary),'16',str(cp),'.0005','.1099',str(output),cuts,mode]
  with (OUT/f'{case}-{mode}.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
  records.append(dict(command=command,binary_sha256=sha(binary),checkpoint_sha256=expected,output_sha256=sha(output)))
 assert sha(cp)==expected
(OUT/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
