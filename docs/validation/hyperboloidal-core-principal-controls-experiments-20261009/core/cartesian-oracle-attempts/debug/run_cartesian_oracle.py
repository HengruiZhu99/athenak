"""One declared held-out Cartesian core-oracle Release/ASan experiment."""
from pathlib import Path
import hashlib
import json
import shlex
import shutil
import subprocess
import time

P=Path(__file__).resolve().parent
old=P.parent/'total-j-local-angular-20261009'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=json.loads((P/'cartesian-polynomial-plan.json').read_text())
assert sha(old/'bridge.cpp')==plan['native_bridge_source_sha256']
assert sha(old/'core_oracle.cpp')==plan['frozen_core_oracle_source_sha256']
assert sha(P/'inputs/flat_formula.hpp')==plan['exact_extracted_formula_body_sha256']
reports={}
for mode in ('release','debug'):
    previous=json.loads((old/'core-oracle-attempts'/mode/'receipt.json').read_text())
    exe=P/('cartesian-oracle-'+mode)
    cmd=[str(P/'cartesian_oracle.cpp') if value==str(old/'core_oracle.cpp') else
         str(exe) if value==str(old/('core-oracle-'+mode)) else value for value in previous['command']]
    assert str(P/'cartesian_oracle.cpp') in cmd
    attempt=P/'cartesian-oracle-attempts'/mode
    assert not attempt.exists()
    attempt.mkdir(parents=True)
    for name in ('cartesian_oracle.cpp','run_cartesian_oracle.py','cartesian-polynomial-plan.json'):
        shutil.copy2(P/name,attempt/name)
    started=time.monotonic();run=subprocess.run(cmd,text=True,capture_output=True)
    (attempt/'build.stdout').write_text(run.stdout);(attempt/'build.stderr').write_text(run.stderr)
    receipt={'command':cmd,'compiler_exit_code':run.returncode,'compile_seconds':time.monotonic()-started,
             'compiler_version':previous['compiler_version'],'source_sha256':sha(P/'cartesian_oracle.cpp'),
             'plan_sha256':sha(P/'cartesian-polynomial-plan.json'),'unchanged_bridge_sha256':sha(old/'bridge.cpp'),
             'unchanged_extracted_formula_sha256':sha(P/'inputs/flat_formula.hpp')}
    (attempt/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    assert run.returncode==0,run.stderr
    dep=[];skip=False
    for value in cmd:
        if skip:skip=False;continue
        if value=='-o':skip=True;continue
        if value.endswith('.a'):continue
        dep.append(value)
    dep+=['-M','-MT','cartesian-oracle']
    dependencies=subprocess.check_output(dep,text=True)
    (attempt/'dependencies.make').write_text(dependencies)
    paths=sorted(set(str(Path(x).resolve()) for x in shlex.split(dependencies.replace('\\\n',' ').split(':',1)[1])))
    receipt.update(executable_sha256=sha(exe),compiler_dependency_hashes={path:sha(path) for path in paths},
                   link_archive_hashes=previous['link_archive_hashes'])
    started=time.monotonic();run=subprocess.run([str(exe)],text=True,capture_output=True)
    (attempt/'stdout.json').write_text(run.stdout);(attempt/'stderr').write_text(run.stderr)
    receipt.update(run_exit_code=run.returncode,run_seconds=time.monotonic()-started)
    (attempt/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    assert run.returncode==0,run.stderr
    reports[mode]=json.loads(run.stdout)
    assert reports[mode]['cases']==plan['expected_cases']
    assert reports[mode]['full22_rhs_scaled']<=plan['tolerances']['actual_full22_vs_FlatFormula_scaled']
    assert reports[mode]['physical8_constraints_scaled']<=plan['tolerances']['actual_physical8_constraints_vs_FlatConstraints_scaled']
    assert not run.stderr
    print(mode,json.dumps(reports[mode]),flush=True)
reports['passed_declared_cartesian_core_oracle']=True
reports['no_new_radial_global_boundary_or_evolution']=True
(P/'cartesian-oracle-report.json').write_text(json.dumps(reports,indent=2)+'\n')
