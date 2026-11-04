from pathlib import Path
import time,json,hashlib,storage_old as old,storage_new as new
base=Path('/lus/flare/projects/CompactBinaryMerger/hzhu/refined-convergence-20261104-v1/stationary-reference-v2')
roots=[base/'source',base/'build',base/'source/src']
old.PRIOR=roots[:-1]
a=time.monotonic();n=new.accounted_bytes(roots);new_seconds=time.monotonic()-a
a=time.monotonic();o=old.capacity(roots[-1],planned=0,reserve=0);old_seconds=time.monotonic()-a
assert o['bytes']==n['bytes'],(o,n)
x=dict(roots=list(map(str,roots)),old=o,new=n,old_seconds=old_seconds,new_seconds=new_seconds,ratio=old_seconds/new_seconds,source_hashes={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ('storage_old.py','storage_new.py')},qualification='Read-only immutable source/build trees with an explicitly duplicated nested source root. New then old cache order; complementary warm scan; not a cold-cache or full-production startup benchmark. No running launcher changed.')
Path('benchmark-reverse.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x,indent=2))
