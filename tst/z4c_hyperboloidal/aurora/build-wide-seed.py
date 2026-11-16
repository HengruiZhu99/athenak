#!/usr/bin/env python3
"""Build only a test seed main; retain the original numerical executable/objects."""
import hashlib,json,subprocess,sys,time
from pathlib import Path
root=Path(sys.argv[1]).resolve();baseline=root.parent/'nonradiative-20261115'
out=root/'seed';out.mkdir(exist_ok=False)
old=json.loads((baseline/'seed-v2/receipt.json').read_text())
source=root/'source/tst/z4c_hyperboloidal/nonradiative_seed.cpp'
(out/'source.cpp').write_bytes(source.read_bytes())
compile_cmd=old['commands'][0].copy()
compile_cmd[compile_cmd.index('-c')+1]=str(out/'source.cpp')
compile_cmd[compile_cmd.index('-o')+1]=str(out/'seed.o')
link=old['commands'][1].copy()
link=[str(out/'seed.o') if x==str(baseline/'seed-v2/seed.o') else x for x in link]
link[link.index('-o')+1]=str(out/'nonradiative_seed')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
objects=[x for x in link if x.endswith('.o') and x!=str(out/'seed.o')]
before={p:sha(p) for p in objects};start=time.monotonic()
subprocess.run(compile_cmd,check=True)
subprocess.run(link,cwd=baseline.parent/'hybrid-overlap-20261111/build-perf-pad32-20261114/src',check=True)
after={p:sha(p) for p in objects};assert before==after
(out/'receipt.json').write_text(json.dumps({'commands':[compile_cmd,link],
    'source_sha256':sha(source),'executable_sha256':sha(out/'nonradiative_seed'),
    'unchanged_object_sha256':after,'seconds':time.monotonic()-start},indent=2)+'\n')
