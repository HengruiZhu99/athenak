#!/usr/bin/env python3
import argparse,subprocess,shlex,re,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
root=Path(__file__).resolve().parents[2];b=args.build.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
flags=(b/'src/CMakeFiles/athena.dir/flags.make').read_text();includes=shlex.split(re.search(r'^CXX_INCLUDES = (.*)$',flags,re.M)[1])
obj=out/'test.o';exe=out/'scri_remesh'
subprocess.run(['/usr/bin/c++','-std=c++17','-O2','-ffp-contract=off',*includes,'-c',str(root/'tst/z4c_hyperboloidal/scri_remesh.cpp'),'-o',str(obj)],check=True)
link=shlex.split((b/'src/CMakeFiles/athena.dir/link.txt').read_text());link=[str(obj) if v=='CMakeFiles/athena.dir/main.cpp.o' else v for v in link];link[link.index('-o')+1]=str(exe)
subprocess.run(link,cwd=b/'src',check=True)
with (out/'run.log').open('w') as log:r=subprocess.run(['mpiexec','-n','2',str(exe),str(root/'tst/z4c_hyperboloidal/n12_initialize.athinput')],cwd=out,stdout=log,stderr=subprocess.STDOUT,timeout=600)
text=(out/'run.log').read_text();print(text[-5000:]);r.check_returncode()
assert all(f'PASS phase={i}' in text for i in [0,1,2]);(out/'results.json').write_text(json.dumps({'passed':True,'phases':['initial','outer_refine','outer_coarsen'],'ranks':2})+'\n')
