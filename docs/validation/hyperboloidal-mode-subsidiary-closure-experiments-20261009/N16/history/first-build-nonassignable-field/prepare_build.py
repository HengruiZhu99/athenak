"""Build a read-only saved-mode comparator against pinned C0 sources."""
from pathlib import Path
import hashlib, json, shlex, shutil, subprocess, time
W = Path(__file__).resolve().parent
R = W.parents[2]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
P = R/'build-layer-research/continuum/constraint-propagation/immutable-constraint-propagation-20261009'
M = R/'build-layer-research/continuum/discrete-mode-identification'
pins = {
    P/'manifest.json': '1ef4735a5b1136c7836daf4fc8783a24f5b83e42eca8281fa63bff8735c4040a',
    P/'subsidiary.hpp': '3d840f2f731ec7fab34579ec2a0132acc4066311b5a4876ae9f484d41f78be54',
    M/'candidate-vectors.npz': 'fc88b20d4953f5088aed97d04dce41ad0af039fc794a401a21cceb53280e1eee',
    M/'candidate-metadata.json': 'e411042b90498bb4f08eadcdf9214095028c049df8a0c15b46b5dfd4d3f2d6e1',
}
for p, h in pins.items(): assert sha(p) == h, p
for name, row in json.loads((P/'manifest.json').read_text())['files'].items():
    assert sha(P/name) == row['sha256'], name
old = R/'build-layer-research/boundary/full-tensor-propagator/old-jv-source.cpp'
original = R/'build-layer-research/boundary/full-tensor-global-final'
manifest = json.loads((original/'manifest.json').read_text())
# The exact old lift/projector implementation is an external pinned dependency.
record = {'authorization': 'Parent approved saved-mode comparator; no propagation/eigensolve/native-long.',
          'launch_HEAD': subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),
          'runtime_implementation':'27c19d20696ea6dd4704032c51dfd026218f64f2',
          'scientific_input_pins':{str(p):h for p,h in pins.items()},
          'old_lift_source_sha256':sha(old)}
shutil.copy2(old, W/'inputs/old-jv-source.cpp')
shutil.copy2(P/'subsidiary.hpp', W/'inputs/subsidiary.hpp')
shutil.copy2(P/'manifest.json', W/'inputs/subsidiary-original-manifest.json')
cmd = json.loads((R/'build-layer-research/boundary/full-tensor-conformal-q-early-weight/build-spatialnorm.json').read_text())
cmd = [x for x in cmd if not x.startswith('-I'+str(R/'build-layer-research')) and x != '-DSPATIAL_NORM']
for i, x in enumerate(cmd):
    if x.endswith('/tangent_server.cpp'): cmd[i] = str(W/'comparator.cpp')
    if x.endswith('/server-spatialnorm'): cmd[i] = str(W/'comparator')
(W/'build-command.json').write_text(json.dumps(cmd,indent=2)+'\n')
started=time.monotonic()
p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
(W/'build.log').write_text(p.stdout)
record.update(command=cmd,exit=p.returncode,seconds=time.monotonic()-started,
              compiler_version=subprocess.check_output(['/usr/bin/c++','--version'],text=True))
assert p.returncode == 0, p.stdout
dep=[];skip=False
for x in cmd:
    if skip:skip=False;continue
    if x=='-o':skip=True;continue
    if x.endswith('.a'):continue
    dep.append(x)
dep += ['-M','-MT','comparator']
out=subprocess.check_output(dep,text=True)
(W/'dependencies.make').write_text(out)
paths=sorted(set(str(Path(x).resolve()) for x in shlex.split(out.replace('\\\n',' ').split(':',1)[1])))
record.update(executable_sha256=sha(W/'comparator'),dependency_command=dep,
              compiler_dependency_hashes={p:sha(p) for p in paths},
              link_archive_hashes={x:sha(x) for x in cmd if x.endswith('.a')})
for p,h in record['compiler_dependency_hashes'].items():
    if p.startswith(str(R/'src')+'/'):
        rel=str(Path(p).relative_to(R))
        assert hashlib.sha256(subprocess.check_output(['git','show',record['runtime_implementation']+':'+rel],cwd=R)).hexdigest()==h,rel
(W/'build-provenance.json').write_text(json.dumps(record,indent=2)+'\n')
print('BUILD_OK',record['seconds'],record['executable_sha256'])
