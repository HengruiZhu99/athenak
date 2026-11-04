#!/usr/bin/env python3
"""Compile standalone experiments against the unchanged frozen native library.

Replay exact CMake commands, replacing only source root, output, and the named
experimental definition. All commands are saved before execution.
This tool replays archived experiments; the current solver has no such switches.
"""
import argparse,json,pathlib,shlex,subprocess
p=argparse.ArgumentParser();p.add_argument('--build',required=True);p.add_argument('--old-source',required=True);p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--experiment',choices=['mask','setup'],default='mask');a=p.parse_args()
source_root=pathlib.Path(a.source).resolve()
marker_file,marker=(('src/z4c/hyperboloidal/scri_radial_device.hpp','PreparePointJet')
                    if a.experiment=='mask' else
                    ('tst/hyperbolic_shift_3d/coupled_mpi.cpp','HYPERBOLIC_DIRECT_HALO_SETUP'))
if marker not in (source_root/marker_file).read_text():
 p.error('This experiment requires its frozen pre-promotion source archive. '
         'Build hyperbolic_shift_3d_amr_evolution directly for the current default solver.')
build=pathlib.Path(a.build).resolve();out=pathlib.Path(a.output).resolve();out.mkdir(exist_ok=False)
commands=json.loads((build/'compile_commands.json').read_text())
specs=[('hyperbolic_shift_3d_amr_evolution','amr_evolution.cpp','HYPERBOLIC_RADIAL_POINT_JET=1','amr_evolution.cpp','hyperbolic_shift_3d_amr_evolution'),('hyperbolic_shift_3d_radial_field_batch','test_radial_field_batch.cpp',None,'test_radial_field_batch.cpp','hyperbolic_shift_3d_radial_field_batch')]
if a.experiment=='setup':
 specs=[('hyperbolic_shift_3d_amr_evolution','amr_evolution.cpp','HYPERBOLIC_DIRECT_HALO_SETUP=1',src,name) for src,name in [('amr_evolution.cpp','hyperbolic_shift_3d_amr_setup'),('test_amr_setup.cpp','hyperbolic_shift_3d_amr_setup_contract')]]
for target,source,definition,new_source,output_name in specs:
 entry=next(c for c in commands if c['file'].endswith('/'+source) and ('/'+target+'.dir/') in c['command'])
 args=shlex.split(entry['command']);oldobj=args[args.index('-o')+1];newobj=str(out/(output_name+'.o'))
 args=[x.replace(str(pathlib.Path(a.old_source).resolve()),str(pathlib.Path(a.source).resolve())) for x in args]
 args=[str(pathlib.Path(a.source).resolve()/'tst/hyperbolic_shift_3d'/new_source) if x==str(pathlib.Path(a.source).resolve()/'tst/hyperbolic_shift_3d'/source) else x for x in args]
 args[args.index('-o')+1]=newobj
 if definition:args+=['-D'+definition]
 link=shlex.split((build/'tst/hyperbolic_shift_3d/CMakeFiles'/f'{target}.dir/link.txt').read_text())
 obj=next(x for x in link if x.endswith('.cpp.o'));link[link.index(obj)]=newobj
 link[link.index('-o')+1]=str(out/output_name)
 # link.txt's working directory is the target's CMake binary subdirectory.
 cwd=build/'tst/hyperbolic_shift_3d'
 (out/(output_name+'.commands.json')).write_text(json.dumps({'compile':args,'compile_cwd':entry['directory'],'link':link,'link_cwd':str(cwd)},indent=2)+'\n')
 with (out/(output_name+'.log')).open('w') as log:
  subprocess.run(args,cwd=entry['directory'],stdout=log,stderr=subprocess.STDOUT,check=True)
  subprocess.run(link,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,check=True)
