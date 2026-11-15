from pathlib import Path
import json,hashlib,io,tarfile,subprocess
ROOT=Path('/lus/flare/projects/CompactBinaryMerger/hzhu/hybrid-overlap-20261111/full-overlap-30M-20261114/final-comparison')
D=ROOT/'audit-9091875';r=json.loads((D/'results.json').read_text());assert len(r)==4
status=json.loads(subprocess.check_output(['qstat','-xf','-F','json','9091875']));j=next(iter(status['Jobs'].values()));assert j['job_state']=='F' and j['Exit_status']==0
(D/'scheduler-final.json').write_text(json.dumps(status,indent=2)+'\n')
archive=ROOT/'final-checkpoint-audits.tar.gz';manifest={}
with tarfile.open(archive,'w:gz') as t:
 for p in sorted(D.rglob('*')):
  if not p.is_file() or p.suffix not in ['.hst','.json','.log','.py']:continue
  b=p.read_bytes();name=str(p.relative_to(D));manifest[name]={'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)}
  info=tarfile.TarInfo(name);info.size=len(b);t.addfile(info,io.BytesIO(b))
 b=json.dumps(manifest,indent=2).encode();info=tarfile.TarInfo('manifest.json');info.size=len(b);t.addfile(info,io.BytesIO(b))
print(json.dumps({'archive':str(archive),'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':len(manifest)}))
