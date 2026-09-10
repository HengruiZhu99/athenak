from pathlib import Path
import sys,json,shutil
root=Path(__file__).resolve().parent
sys.path.insert(0,str(Path('/pscratch/sd/h/hzhu/lapse-bisection-recovery-20260910')))
from controller import sha,validate_adoption,validate_termination
from criterion import read_history
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--source-sha',required=True);source_sha=ap.parse_args().source_sha
old=Path('/pscratch/sd/h/hzhu/z4c-vc-performance-perlmutter-20260829/history-extrema-20260901/bisection_N256_20260907')
campaign=Path('/pscratch/sd/h/hzhu/lapse-bisection-recovery-20260910')
if (campaign/'state.json').exists():raise RuntimeError('Campaign already has state')
verified=json.loads((root/'recovery-lapse/validation/verification.json').read_text())
exe=root/'recovery-lapse/build/src/athena'
if verified['status']!='PASS' or sha(exe)!=verified['executable_sha256']:raise RuntimeError('Executable qualification mismatch')
for name,digest in json.loads((root/'recovery-lapse-source-files.json').read_text()).items():
 if sha(root/'recovery-lapse/source'/name)!=digest:raise RuntimeError('Compiled source differs: '+name)
base=root/'recovery-lapse/baseline';base.mkdir()
checks={line.split(maxsplit=1)[1].strip():line.split()[0] for line in (old/'immutable.sha256').read_text().splitlines()}
for name,digest in checks.items():
 if sha(old/name)!=digest:raise RuntimeError('Original immutable baseline changed: '+name)
 target=base/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(old/name,target)
for name in ['source-base.txt','source.patch']:shutil.copyfile(old/name,base/('original-'+name))
shutil.copyfile(old/'generator-qualified.json',base/'generator-qualified.json')
shutil.copyfile(exe,base/'athena.history_extrema');(base/'athena.history_extrema').chmod(0o755)
(base/'export_coefficients').chmod(0o755)
(base/'source-base.txt').write_text(source_sha+'\n');(base/'source.patch').write_text('')
shutil.copyfile(root/'recovery-lapse-source-files.json',base/'source-files.json')
files=sorted(p for p in base.rglob('*') if p.is_file() and p.name!='immutable.sha256')
(base/'immutable.sha256').write_text(''.join(sha(p)+'  '+str(p.relative_to(base))+'\n' for p in files))
evidence=json.loads((root/'recovery-lapse/endpoints/evidence.json').read_text())
for path,record in evidence.items():
 d=Path(path)
 from decimal import Decimal
 validate_adoption(d,Decimal((d/'amplitude.txt').read_text()),(base/'template.athinput').read_text(),sha(exe),record)
 validate_termination(d,read_history(next(d.glob('*.hst'))),historical=True)
q=dict(executable_sha256=sha(exe),source_sha=source_sha,approved_configuration={'boundary_rhs':'full_constraint_bjorhus','extrap_order':2,'vc_single_rank_device_sync':True},runtime_stop_validation=verified,legacy_endpoint_evidence=evidence,boundary_evidence={'fresh_sub_job':'58135543','fresh_super_clean_segment_job':'58134624','t90_to200_job':'58134571','limitation':'Empirical suppression of previous corner instability; not spatial convergence qualification.'})
(campaign/'BOUNDARY_QUALIFIED.json').write_text(json.dumps(q,indent=2)+'\n')
print(base)
