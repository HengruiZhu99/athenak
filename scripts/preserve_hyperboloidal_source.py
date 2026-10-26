#!/usr/bin/env python3
"""Freeze exactly the source and library named by a run manifest; no evolution."""
import argparse,hashlib,json,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);a=p.parse_args();repo=Path(__file__).resolve().parents[1]
m=json.loads(a.manifest.read_text());root=a.manifest.parent/'frozen-source';root.mkdir(exist_ok=False)
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
script=Path(m['command'][0]);srcroot=script.parents[2] if script.is_absolute() else repo
for name,digest in m['binding']['source_sha256'].items():
 src=srcroot/name
 if sha(src)!=digest:raise SystemExit('Source mismatch: '+name)
 dst=root/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
lib=Path(m['arguments']['library']);lib=lib if lib.is_absolute() else repo/lib
if sha(lib)!=m['binding']['executable_sha256']:raise SystemExit('Library mismatch')
(root/'runtime').mkdir();shutil.copy2(lib,root/'runtime'/lib.name)
if m['arguments'].get('fast_library'):
 fast=Path(m['arguments']['fast_library']);fast=fast if fast.is_absolute() else repo/fast
 if sha(fast)!=m['binding']['fast_executable_sha256']:raise SystemExit('Compiled integration library mismatch')
 shutil.copy2(fast,root/'runtime'/fast.name)
(root/'source-binding.json').write_text(json.dumps(m['binding'],indent=2)+'\n');print(root)
