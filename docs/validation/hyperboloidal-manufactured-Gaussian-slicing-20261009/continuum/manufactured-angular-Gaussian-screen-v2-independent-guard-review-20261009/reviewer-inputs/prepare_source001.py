from pathlib import Path
import json,hashlib,shutil,difflib
base=Path('/Users/hz0693/research/hyperboloidal');old=base/'build-layer-research/manufactured-angular-Gaussian-screen-held-20261009';out=base/'build-layer-research/manufactured-angular-Gaussian-screen-v2-held-20261009';out.mkdir(exist_ok=False)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as s:
  for b in iter(lambda:s.read(1048576),b''):h.update(b)
 return h.hexdigest()
s0=(old/'screen.py').read_text();s=s0
before='def run(recipe,out):\n    import mpmath as mp\n';after="def run(recipe,out):\n    sys.path.insert(0,recipe['mpmath_parent'])\n    import mpmath as mp\n    if str(Path(mp.__file__).resolve())!=recipe['mpmath_init']:\n        raise RuntimeError('unexpected mpmath import origin')\n"
if s.count(before)!=1:raise RuntimeError('import block count')
s=s.replace(before,after)
before="        if sys.flags.optimize!=0 or os.environ.get('PYTHONDONTWRITEBYTECODE')!='1':raise RuntimeError('unoptimized bytecode-off runtime required')";after="        if sys.flags.optimize!=0 or not sys.flags.isolated or not sys.dont_write_bytecode:\n            raise RuntimeError('isolated unoptimized bytecode-off runtime required')"
if s.count(before)!=1:raise RuntimeError('runtime guard count')
s=s.replace(before,after)
before="        pins.update(recipe['pins'])";after="        if sha(Path(sys.executable).resolve())!=recipe['python_runtime_sha256']:\n            raise RuntimeError('actual Python runtime differs')\n        for key,value in recipe['environment'].items():\n            if os.environ.get(key)!=value:raise RuntimeError('fixed environment differs '+key)\n        pins.update(recipe['pins'])"
if s.count(before)!=1:raise RuntimeError('env guard count')
s=s.replace(before,after)
(out/'screen.py').write_text(s)
r=json.loads((old/'recipe.json').read_text());r['status']='HELD v2 explicit isolated runtime/import admission';r['python_runtime_sha256']=sha(r['python_runtime_path']);r['mpmath_init']=next(p for p in r['pins'] if p.endswith('/mpmath/__init__.py'));r['mpmath_parent']=str(Path(r['mpmath_init']).parent.parent)
for p in sorted(old.iterdir()):
 if p.is_file():r['pins'][str(p)]=sha(p)
(out/'recipe.json').write_text(json.dumps(r,indent=2)+'\n')
(out/'PLAN.md').write_text((old/'PLAN.md').read_text()+'\nV2 changes only admission: isolated unoptimized bytecode-off interpreter hash,\nfixed environment, explicit pinned mpmath parent and actual module origin.\nAll scientific statements, profile/grid, arithmetic and tolerances are unchanged.\nThe v1 source remains unexecuted and byte-exact as a protected historical input.\n')
(out/'admission-only.diff').write_text(''.join(difflib.unified_diff(s0.splitlines(True),s.splitlines(True),fromfile=str(old/'screen.py'),tofile=str(out/'screen.py'))))
shutil.copyfile(Path(__file__),out/'prepare_source001.py')
files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file()]
(out/'source-index.json').write_text(json.dumps(dict(source_only=True,execution_admitted=False,files=files,old_index_sha256=sha(old/'source-index.json'),science_unchanged=True),indent=2)+'\n')
print(json.dumps(dict(index_sha256=sha(out/'source-index.json'),recipe_sha256=sha(out/'recipe.json'),screen_sha256=sha(out/'screen.py'),diff_sha256=sha(out/'admission-only.diff'),pins=len(r['pins']))))
