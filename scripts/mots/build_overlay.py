#!/usr/bin/env python3
"""Compile every translation unit affected by changed source/header dependencies.

Reuse immutable objects from an existing CUDA/MPI build with the same CMake
configuration. Refuse missing dependency files rather than guessing an ABI-safe
subset. Intended for an isolated, complete source export on Perlmutter.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['baseline-source','baseline-build','source','output']:
        p.add_argument('--'+key,required=True,type=Path)
    args=p.parse_args()
    old=args.baseline_source.resolve();build=args.baseline_build.resolve()
    source=args.source.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    changed={str(f.relative_to(source)) for f in (source/'src').rglob('*') if f.is_file()
             and (not (old/f.relative_to(source)).exists() or sha(f)!=sha(old/f.relative_to(source)))}
    flags={}
    for line in (build/'src/CMakeFiles/athena.dir/flags.make').read_text().splitlines():
        if ' = ' in line:
            key,value=line.split(' = ',1);flags[key]=shlex.split(value)
    link=shlex.split((build/'src/CMakeFiles/athena.dir/link.txt').read_text())
    compiler=shutil.which('CC')
    if not compiler: raise RuntimeError('Load the original compiler modules first')
    launch=[str(old/'kokkos/bin/kokkos_launch_compiler'),str(old/'kokkos/bin/nvcc_wrapper'),compiler,compiler]
    replacements={};commands=[];affected=[]
    for word in link:
        if not word.endswith('.o'): continue
        obj=build/'src'/word
        dep=Path(str(obj)+'.d')
        if not dep.exists(): raise RuntimeError('Missing compiler dependency file '+str(dep))
        tokens=set(dep.read_text().replace('\\\n',' ').split())
        if not any(str(old/f) in tokens for f in changed): continue
        relative=word.removeprefix('CMakeFiles/athena.dir/').removesuffix('.o')
        src=source/'src'/relative
        if not src.exists(): raise RuntimeError('Cannot resolve source for '+word)
        target=out/(relative.replace('/','_')+'.o')
        includes=[v.replace(str(old),str(source)) for v in flags['CXX_INCLUDES']]
        cmd=launch+flags['CXX_DEFINES']+includes+flags['CXX_FLAGS']+['-o',str(target),'-c',str(src)]
        print('Compiling',relative,flush=True)
        subprocess.run(cmd,check=True);commands.append(cmd);affected.append(relative)
        replacements[word]=str(target)
    # New translation units have no baseline object/dependency file. Compile
    # only files explicitly present in the new production CMake source list.
    extra=[]
    source_list=(source/'src/CMakeLists.txt').read_text()
    for name in sorted(changed):
        if not name.endswith('.cpp') or (old/name).exists(): continue
        relative=name.removeprefix('src/')
        if relative not in source_list:
            raise RuntimeError('New source absent from production CMake list: '+name)
        target=out/(relative.replace('/','_')+'.o')
        includes=[v.replace(str(old),str(source)) for v in flags['CXX_INCLUDES']]
        cmd=launch+flags['CXX_DEFINES']+includes+flags['CXX_FLAGS']+['-o',str(target),'-c',str(source/name)]
        print('Compiling new source',relative,flush=True)
        subprocess.run(cmd,check=True);commands.append(cmd);affected.append(relative)
        extra.append(str(target))
    if changed and not affected: raise RuntimeError('Changed source but no affected objects')
    command=[replacements.get(w,w) for w in link]
    command[command.index('-o')+1]=str(out/'athena')
    command[1:1]=extra
    subprocess.run(command,cwd=build/'src',check=True);commands.append(command)
    (out/'build.json').write_text(json.dumps(dict(changed=sorted(changed),affected=affected,commands=commands,
        executable_sha256=sha(out/'athena')),indent=2)+'\n')
    print('BUILT',out/'athena',flush=True)


if __name__=='__main__': main()
