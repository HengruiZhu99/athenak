#!/usr/bin/env python3
import hashlib,json,pathlib,subprocess,sys,time
root=pathlib.Path(sys.argv[1]);out=root/'seed-v2';out.mkdir(exist_ok=False)
old=json.loads((root/'build/receipt.json').read_text())
source=root/'source/tst/z4c_hyperboloidal/nonradiative_seed.cpp';(out/'source.cpp').write_bytes(source.read_bytes())
compile_cmd=old['commands'][2].copy();compile_cmd[compile_cmd.index('-c')+1]=str(out/'source.cpp');compile_cmd[compile_cmd.index('-o')+1]=str(out/'seed.o')
link=[str(out/'seed.o') if x==str(root/'build/seed.o') else x for x in old['commands'][3]];link[link.index('-o')+1]=str(out/'nonradiative_seed')
objects=[pathlib.Path(x) for x in link if x.endswith('/main.cpp.o')]
base=pathlib.Path('/lus/flare/projects/CompactBinaryMerger/hzhu/hybrid-overlap-20261111/build-perf-pad32-20261114/src')
start=time.monotonic();subprocess.run(compile_cmd,check=True);subprocess.run(link,cwd=base,check=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(out/'receipt.json').write_text(json.dumps(dict(commands=[compile_cmd,link],source_sha256=sha(out/'source.cpp'),executable_sha256=sha(out/'nonradiative_seed'),seconds=time.monotonic()-start),indent=2))
