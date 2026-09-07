from pathlib import Path
import json,hashlib,difflib,shutil,datetime
root=Path('/scratch/gpfs/FPRETORI/hz0693/pcgh-clean-reduction-20260907-transfer-001')
assert (root/'build.exit').read_text().strip()=='2'
assert (root/'test.exit').read_text().strip()=='1'
manifest=json.loads((root/'source-manifest.json').read_text())
assert all(hashlib.sha256((root/'source'/f).read_bytes()).hexdigest()==h for f,h in manifest.items())
for name in ['build.log','configure.log','build.exit','test.exit','controller.log','test-controller.log']:
 target=root/('initial-failure-'+name)
 assert not target.exists()
 shutil.copyfile(root/name,target)
file=root/'source/src/pc_gh/pc_gh.hpp';old=file.read_text()
needle=' private:\n  template <int ORDER> void TransferResidualGhosts();'
assert old.count(needle)==1
new=old.replace(needle,' public:\n  template <int ORDER> void TransferResidualGhosts();\n\n private:')
(root/'pc_gh.hpp.before-access-fix').write_text(old)
(root/'cuda-access-fix.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/src/pc_gh/pc_gh.hpp',tofile='b/src/pc_gh/pc_gh.hpp')))
file.write_text(new)
manifest['src/pc_gh/pc_gh.hpp']=hashlib.sha256(file.read_bytes()).hexdigest()
(root/'source-accessfix-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(root/'accessfix-ownership.json').write_text(json.dumps(dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),base='40e0bc1fc6e8f5dd0c474ed7d5d127060ea44937',change='public access for enclosing CUDA extended-lambda member only',equations_changed=False,original_archive_preserved=True),indent=2)+'\n')
print('PASS: original manifest verified, failure preserved, minimal access patch applied')
