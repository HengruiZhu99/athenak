"""Prepare one live synchronization interval from a previous Brill manifest."""
import argparse,hashlib,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('previous_manifest',type=Path)
p.add_argument('--exe',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();m=json.loads(a.previous_manifest.read_text());exe=a.exe.resolve();out=a.output.resolve()
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
checkpoint=Path(m['checkpoint']);assert sha(checkpoint)==m['checkpoint_sha256']
sys.path.insert(0,str(checkpoint.parents[2]))
from workflow_common import setparam
prior=next(c for c in m['cases'] if c['name']=='subcycled');source=Path(prior['directory'])
assert sha(source/'input.athinput')==prior['input_sha256']
with (source/'amr_history.jsonl').open('rb') as f:prefix=f.read(m['initial_info']['history_bytes'])
assert hashlib.sha256(prefix).hexdigest()==prior['amr_prefix_sha256']
assert exe.is_file();out.mkdir(parents=True,exist_ok=False)
s=(source/'input.athinput').read_text()
s=setparam(s,'time','nlim',m['initial_info']['cycle']+1)
s=setparam(s,'time','ndiag',1)
s=setparam(s,'mesh_refinement','amr_history_file',out/'amr_history.jsonl')
(out/'input.athinput').write_text(s);(out/'amr_history.jsonl').write_bytes(prefix)
r=dict(checkpoint=str(checkpoint),checkpoint_sha256=m['checkpoint_sha256'],
       executable=str(exe),executable_sha256=sha(exe),input_sha256=sha(out/'input.athinput'),
       prefix_sha256=sha(out/'amr_history.jsonl'),start=m['start'],
       nlim=m['initial_info']['cycle']+1,parent_manifest=str(a.previous_manifest.resolve()),
       purpose='First synchronization interval only; trace corrector failure without changing tolerances')
(out/'manifest.json').write_text(json.dumps(r,indent=2)+'\n')
print(out/'manifest.json')
