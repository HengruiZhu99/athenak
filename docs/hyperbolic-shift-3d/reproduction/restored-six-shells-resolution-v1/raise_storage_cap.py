"""Freeze a separate launcher for the user's explicit 10 GB authorization."""
from pathlib import Path
import hashlib,json,shutil,subprocess
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;RAW=ROOT/'output/restored-six-shells-resolution/aurora';OLD=RAW/'launcher-restored-six-shells-resolution-v1';NEW=RAW/'launcher-restored-six-shells-resolution-v2'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=json.loads((OLD/'provenance.json').read_text())
for name,v in p['files'].items():assert (OLD/name).stat().st_size==v['bytes'] and sha(OLD/name)==v['sha256']
NEW.mkdir(exist_ok=False)
for name in p['files']:shutil.copy2(OLD/name,NEW/name)
for name in ['storage.py','native_transition_campaign.py']:
 f=NEW/name;s=f.read_text();assert s.count('5_000_000_000')==1
 s=s.replace('5_000_000_000','10_000_000_000').replace('5,000,000,000','10,000,000,000');f.write_text(s)
p.update(launcher_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),storage_authorization='User explicitly raised this campaign cap to10GB. Original v1-bound N20 segment01 remains immutable at5GB; v2 used for later continuations and N24.',parent_resolution_launcher_provenance_sha256=sha(OLD/'provenance.json'),storage_adapter=str(Path(__file__).relative_to(ROOT)),storage_adapter_sha256=sha(Path(__file__)),files={f.name:{'bytes':f.stat().st_size,'sha256':sha(f)} for f in NEW.iterdir() if f.is_file()})
(NEW/'provenance.json').write_text(json.dumps(p,indent=2)+'\n');(HERE/'launcher-v2-provenance.json').write_text(json.dumps(p,indent=2)+'\n');print(NEW)
