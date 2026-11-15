#!/usr/bin/env python3
"""Build the zero-step audit against the actual native executable objects."""
import argparse,hashlib,json,pathlib,re,shlex,subprocess
p=argparse.ArgumentParser();p.add_argument('--build',type=pathlib.Path,required=True)
p.add_argument('--output',type=pathlib.Path,required=True)
p.add_argument('--test',choices=['binary_audit','binary_remesh'],default='binary_audit')
a=p.parse_args();root=pathlib.Path(__file__).resolve().parents[2];build=a.build.resolve();out=a.output.resolve()
out.mkdir(parents=True,exist_ok=True)
flags=(build/'src/CMakeFiles/athena.dir/flags.make').read_text()
include=shlex.split(re.search(r'^CXX_INCLUDES = (.*)$',flags,re.M).group(1))
opts=shlex.split(re.search(r'^CXX_FLAGS = (.*)$',flags,re.M).group(1))
link=shlex.split((build/'src/CMakeFiles/athena.dir/link.txt').read_text())
obj=out/(a.test+'.o');exe=out/a.test
cmd=[link[0],*opts,*include,'-c',str(root/('tst/z4c_hyperboloidal/'+a.test+'.cpp')),'-o',str(obj)]
subprocess.run(cmd,check=True)
link=[str(obj) if x=='CMakeFiles/athena.dir/main.cpp.o' else x for x in link]
link[link.index('-o')+1]=str(exe)
subprocess.run(link,cwd=build/'src',check=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
report={'executable_sha256':sha(exe),'source_sha256':{str(p.relative_to(root)):sha(p) for p in (root/'src').rglob('*') if p.is_file() and p.suffix in ('.cpp','.hpp')},'test_sha256':sha(root/('tst/z4c_hyperboloidal/'+a.test+'.cpp')),'scope':'zero-step native lifecycle and RHS; no evolution driver'}
(out/'provenance.json').write_text(json.dumps(report,indent=2)+'\n')
print(exe)
