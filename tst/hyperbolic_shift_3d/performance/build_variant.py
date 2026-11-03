#!/usr/bin/env python3
"""Compile standalone experiments against the unchanged frozen native library.

Replay exact CMake commands, replacing only source root, output, and the named
experimental definition. All commands are saved before execution.
"""
import argparse,json,pathlib,shlex,subprocess
p=argparse.ArgumentParser();p.add_argument('--build',required=True);p.add_argument('--old-source',required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
build=pathlib.Path(a.build).resolve();out=pathlib.Path(a.output).resolve();out.mkdir(exist_ok=False)
commands=json.loads((build/'compile_commands.json').read_text())
for target,source,definition in [('hyperbolic_shift_3d_amr_evolution','amr_evolution.cpp','HYPERBOLIC_RADIAL_POINT_JET=1'),('hyperbolic_shift_3d_radial_field_batch','test_radial_field_batch.cpp',None)]:
 entry=next(c for c in commands if c['file'].endswith('/'+source) and ('/'+target+'.dir/') in c['command'])
 args=shlex.split(entry['command']);oldobj=args[args.index('-o')+1];newobj=str(out/(target+'.o'))
 args=[x.replace(str(pathlib.Path(a.old_source).resolve()),str(pathlib.Path(a.source).resolve())) for x in args]
 args[args.index('-o')+1]=newobj
 if definition:args+=['-D'+definition]
 link=shlex.split((build/'tst/hyperbolic_shift_3d/CMakeFiles'/f'{target}.dir/link.txt').read_text())
 obj=next(x for x in link if x.endswith('.cpp.o'));link[link.index(obj)]=newobj
 link[link.index('-o')+1]=str(out/target)
 # link.txt's working directory is the target's CMake binary subdirectory.
 cwd=build/'tst/hyperbolic_shift_3d'
 (out/(target+'.commands.json')).write_text(json.dumps({'compile':args,'compile_cwd':entry['directory'],'link':link,'link_cwd':str(cwd)},indent=2)+'\n')
 with (out/(target+'.log')).open('w') as log:
  subprocess.run(args,cwd=entry['directory'],stdout=log,stderr=subprocess.STDOUT,check=True)
  subprocess.run(link,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,check=True)
