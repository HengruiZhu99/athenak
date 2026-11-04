"""Package fresh bounded controls; reuse immutable b52 science/binary, no build."""
from pathlib import Path
import argparse,json,hashlib,shutil,subprocess,tarfile
HERE=Path(__file__).resolve().parent;ROOT=next(p for p in HERE.parents if (p/'CMakeLists.txt').exists())
p=argparse.ArgumentParser();p.add_argument('destination');a=p.parse_args();out=Path(a.destination).resolve();out.mkdir(parents=True,exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def record(p):return dict(bytes=p.stat().st_size,sha256=sha(p),mode=oct(p.stat().st_mode&0o777))
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
config_dir=ROOT/'inputs/hyperbolic_shift_3d/refined_convergence';configs=[config_dir/f'central20_stationary_smr025_radial{n}_reference.cfg' for n in (12,18)]
def parameters(path):return dict(line.split('=',1)[0].strip() and (line.split('=',1)[0].strip(),line.split('=',1)[1].strip()) for line in path.read_text().splitlines() if '=' in line and not line.strip().startswith('#'))
c,d=map(parameters,configs);assert {k for k in c if c[k]!=d[k]}=={'shell_radii'} and c.keys()==d.keys()
for path in configs+list((HERE/'launcher-v7').glob('*')):
 if not path.is_file():continue
 assert subprocess.check_output(['git','show',commit+':'+str(path.relative_to(ROOT))],cwd=ROOT)==path.read_bytes(),'uncommitted input '+str(path)
shutil.copytree(HERE/'launcher-v7',out/'launcher-v7',ignore=shutil.ignore_patterns('__pycache__'))
frozen=ROOT/'output/refined-convergence/stationary-reference-v2-aurora/launcher-v6';manifest=json.loads((frozen/'manifest.json').read_text())
for name in ('verify-checkpoint','verifier-build.json'):
 assert sha(frozen/name)==manifest[name]['sha256'];shutil.copy2(frozen/name,out/'launcher-v7'/name)
assert (out/'launcher-v7/verify-checkpoint').stat().st_mode&0o111
inputs=out/'radial-controls-v1';inputs.mkdir()
for path in configs:shutil.copy2(path,inputs/path.name)
science=json.loads((HERE/'stationary-reference-v2-build/science.json').read_text())
prov=dict(input_launcher_commit=commit,scientific_source_commit=science['source_commit'],scientific_binary_sha256=science['binary_sha256'],source_manifest_sha256=science['source_manifest_sha256'],input_files={p.name:record(inputs/p.name) for p in configs},sole_grid_change='shell_radii; other parsed parameters identical',qualification='Runtime grid input controls under the unchanged frozen reference-v2 binary/source. No source replacement, rebuild, GPU smoke, restart reuse or mutation of launcher-v6. Matched dt .00003125, native fresh t0 to .6M. Identity includes shell layout; existing Aurora map calibration unchanged. New storage traversal preserves cap/roots/reserves and is separately verified.')
(inputs/'provenance.json').write_text(json.dumps(prov,indent=2)+'\n')
launch={p.name:record(p) for p in sorted((out/'launcher-v7').iterdir()) if p.is_file()};(out/'launcher-v7/manifest.json').write_text(json.dumps(launch,indent=2)+'\n')
records={str(p.relative_to(out)):record(p) for p in out.rglob('*') if p.is_file()};(out/'package-records.json').write_text(json.dumps(records,indent=2)+'\n')
with tarfile.open(out/'package.tar.gz','w:gz',format=tarfile.PAX_FORMAT) as tar:
 for name in sorted(records):tar.add(out/name,arcname=name,recursive=False)
 tar.add(out/'package-records.json',arcname='radial-controls-v1/package-records.json',recursive=False)
print(json.dumps(dict(package=str(out),files=len(records),archive_sha256=sha(out/'package.tar.gz'),scientific_source=science['source_commit']),indent=2))
