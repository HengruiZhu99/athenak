"""Preserve authenticated clean historical segments; does not approve or submit."""
from pathlib import Path
import sys,json,shutil
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root/'bisection'))
from controller import sha
from criterion import read_history,classify,first_early_crossing
from resume_case import checkpoint_info
out=root/'early-lapse/historical-endpoints';out.mkdir()
records={}
for role in ['sub','super']:
 source=root/f'fresh_{role}_cpbc_linear';d=out/role;d.mkdir()
 logs=source if role=='sub' else source/'segments/after-58134624'
 if (logs/'run-status').read_text().strip()!='0':raise RuntimeError('Historical segment failed')
 if role=='super':
  manifest=json.loads((logs/'manifest.json').read_text());rst=Path(manifest['checkpoint'])
  if sha(rst)!=manifest['checkpoint_sha256']:raise RuntimeError('Archived checkpoint changed')
 else:rst=sorted((source/'rst').glob('*.rst'))[-1]
 info=checkpoint_info(rst)
 for name in ['input.athinput','initial.coefficients','inputs.sha256','provenance.json','amplitude.txt']:shutil.copyfile(source/name,d/name)
 for name in ['run-status','job-id.txt','stdout.log','stderr.log']:shutil.copyfile(logs/name,d/name)
 history=next(source.glob('*.hst'))
 lines=[x for x in history.read_text().splitlines() if x.startswith('#') or not x.strip() or float(x.split()[0])<=info['time']]
 (d/history.name).write_text('\n'.join(lines)+'\n')
 rows=read_history(d/history.name)
 if rows[-1]['time']!=info['time'] or rows[-1]['cycle']!=info['cycle']:raise RuntimeError('History/restart mismatch')
 if classify(rows,200)!=('disperse' if role=='sub' else 'collapse'):raise RuntimeError('Historical endpoint no longer brackets')
 (d/'rst').mkdir();shutil.copyfile(rst,d/'rst'/rst.name)
 provenance=dict(original_directory=str(source),clean_segment_logs=str(logs),original_history_sha256=sha(history),clean_checkpoint=str(rst),checkpoint_info=info,classification=classify(rows,200),first_stored_early_crossing=first_early_crossing(rows),note='Original clean historical prefix; cancelled continuation is excluded. Classification follows revised lapse rule, not horizon evidence.')
 (d/'historical-import.json').write_text(json.dumps(provenance,indent=2)+'\n')
 records[str(d)]=dict(executable_sha256=json.loads((d/'provenance.json').read_text())['exe_sha256'],files={str(p.relative_to(d)):sha(p) for p in d.rglob('*') if p.is_file()})
(out/'evidence.json').write_text(json.dumps(records,indent=2)+'\n')
print(out)
