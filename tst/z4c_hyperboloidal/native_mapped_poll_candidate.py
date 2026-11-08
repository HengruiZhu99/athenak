#!/usr/bin/env python3
"""Compile one mapped-exchange candidate TU and relink frozen native objects.

The control and candidate share one qualified complete-cycle timing driver.
No native build tree is written and no executable is launched by this tool.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time
from types import SimpleNamespace

import native_cycle_timing as timing

sha, require = timing.sha, timing.require
BASELINE = 'f3a85db7528986ddec347b31675a8c5d588120a2380133aed73d5967f7aab05c'
CANDIDATE = 'a67ec7abf8235daac113fd6d05caf7bf36d307d645c9453ed37ec25cbc083e29'
PATCH = 'b5eae25021f2192f3cd8adec612876f28246df681b4d1378f7dccf5facabfee5'
TIMER = '9b0c3a9b0bc2feb6f0c8d2cedffc3ce273f5263b592a14add9f15332cf16d63d'


def verify(output):
    p = json.loads((output/'provenance.json').read_text())
    require(p['schema']=='native-mapped-poll-build-v1' and p['status']=='built'
            and p['build_exit_code']==0, 'candidate build incomplete')
    require(sha(p['timing_proof'])==p['timing_proof_sha256'], 'timing proof changed')
    timing.verify(SimpleNamespace(output=Path(p['timing_proof']).parent))
    for group in ('inputs','dependencies','outputs'):
        for name,digest in p[group].items():
            require(sha(name)==digest, 'changed '+group+': '+name)
    require(p['changed_numerical_sources']==['src/bvals/mapped_exchange.cpp']
            and p['same_timing_driver_control_candidate'] is True,
            'candidate scope changed')
    return p


def build(args):
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    tp=args.timing_proof.resolve();t=json.loads(tp.read_text())
    timing.verify(SimpleNamespace(output=tp.parent))
    require(t['native_executable_sha256']==BASELINE, 'wrong native baseline')
    source=args.source.resolve();patch=args.patch.resolve()
    require(sha(source)==CANDIDATE and sha(patch)==PATCH, 'candidate bytes differ')
    timer_headers=[n for n in t['sources'] if n.endswith('/native_cycle_timing.hpp')]
    require(len(timer_headers)==1 and sha(timer_headers[0])==TIMER, 'wrong rank-export timer')
    database=Path(t['compile_database']);entries=json.loads(database.read_text())
    selected=[e for e in entries if e['file'].endswith('/src/bvals/mapped_exchange.cpp')]
    require(len(selected)==1, 'mapped compile entry absent/ambiguous')
    entry=selected[0];cwd=Path(entry['directory']).resolve()
    original_source=(cwd/entry['file']).resolve()
    original=entry.get('arguments') or shlex.split(entry['command'])
    obj=output/'mapped_exchange.cpp.o';dep=output/'mapped_exchange.d'
    command=[];i=0;ns=no=0
    while i<len(original):
        token=original[i]
        if token=='-o': command+=['-o',str(obj)];no+=1;i+=2;continue
        if token in ('-MF','-MT','-MQ'):i+=2;continue
        if token in ('-MD','-MMD','-MP'):i+=1;continue
        if not token.startswith('-') and (cwd/token).resolve()==original_source:
            command.append(str(source));ns+=1
        else:command.append(token)
        i+=1
    require(ns==no==1 and '-c' in command, 'unknown compile command')
    command+=['-iquote',str(original_source.parent),'-MMD','-MF',str(dep),'-MT',str(obj)]
    # Apply the reviewed patch to an isolated copy, proving it derives from the
    # actual frozen source rather than a similarly named local revision.
    check=output/'patch-check';(check/'src/bvals').mkdir(parents=True)
    (check/'src/bvals/mapped_exchange.cpp').write_bytes(original_source.read_bytes())
    subprocess.run(['patch','-p1','--batch','-i',str(patch)],cwd=check,check=True,
                   stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    require((check/'src/bvals/mapped_exchange.cpp').read_bytes()==source.read_bytes(),
            'patch does not produce exact candidate from frozen source')
    link_file=Path(t['link_file']);link_cwd=Path(t['link_cwd'])
    original_link=shlex.split(link_file.read_text())
    suffix='athena.dir/bvals/mapped_exchange.cpp.o'
    indices=[i for i,x in enumerate(original_link) if x.endswith(suffix)]
    require(len(indices)==1 and original_link.count('-o')==1,'unknown native link')
    mapped_index=indices[0]
    baseline_obj=(link_cwd/original_link[mapped_index]).resolve()
    commands={};executables={}
    for name,timed in [('candidate-normal',False),('candidate-timed',True)]:
        link=list(original_link);link[mapped_index]=str(obj)
        if timed:
            indices=[i for i,x in enumerate(link) if x.endswith('athena.dir/driver/driver.cpp.o')]
            require(len(indices)==1,'unknown driver link')
            link[indices[0]]=t['harness_object']
        exe=output/('athena-'+name);link[link.index('-o')+1]=str(exe)
        commands[name]=link;executables[name]=str(exe)
    inputs={str(x):sha(x) for x in [source,patch,original_source,database,link_file,
             baseline_obj,tp,Path(t['harness_object']),Path(__file__).resolve(),
             Path(timing.__file__).resolve()]}
    for token in original_link:
        if token.endswith(('.o','.a','.so','.dylib')):
            path=(link_cwd/token).resolve();inputs[str(path)]=sha(path)
    p=dict(schema='native-mapped-poll-build-v1',status='building',
        baseline_executable_sha256=BASELINE,timing_proof=str(tp),timing_proof_sha256=sha(tp),
        control_executable=t['executable'],control_executable_sha256=t['executable_sha256'],
        timing_driver=t['harness_object'],timing_driver_sha256=t['harness_object_sha256'],
        changed_numerical_sources=['src/bvals/mapped_exchange.cpp'],
        same_timing_driver_control_candidate=True,original_compile_entry=entry,
        original_compile_command=original,compile_command=command,compile_cwd=str(cwd),
        original_link_command=original_link,link_commands=commands,link_cwd=str(link_cwd),
        candidate_source=str(source),candidate_source_sha256=CANDIDATE,
        candidate_patch=str(patch),candidate_patch_sha256=PATCH,
        original_mapped_object=str(baseline_obj),original_mapped_object_sha256=sha(baseline_obj),
        candidate_object=str(obj),executables=executables,inputs=inputs,gpu_executed=False)
    proof=output/'provenance.json';proof.write_text(json.dumps(p,indent=2)+'\n')
    makefile=output/'candidate.mk'
    recipe='.PHONY: all\nall:\n\tcd '+shlex.quote(str(cwd))+' && '+shlex.join(command)+'\n'
    for link in commands.values():recipe+='\tcd '+shlex.quote(str(link_cwd))+' && '+shlex.join(link)+'\n'
    makefile.write_text(recipe);start=time.monotonic()
    with (output/'build.log').open('w') as log:
        result=subprocess.run(['make','-j64','-f',str(makefile)],stdout=log,stderr=subprocess.STDOUT)
    p.update(build_seconds=time.monotonic()-start,build_exit_code=result.returncode,
             status='failed' if result.returncode else 'built')
    proof.write_text(json.dumps(p,indent=2)+'\n')
    for name,digest in inputs.items():require(sha(name)==digest,'build input changed: '+name)
    result.check_returncode()
    deps=shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1])
    p['dependencies']={str((cwd/n).resolve()):sha((cwd/n).resolve()) for n in deps}
    p['outputs']={str(obj):sha(obj),**{x:sha(x) for x in executables.values()}}
    p['runtime_inputs_unchanged']=True
    proof.write_text(json.dumps(p,indent=2)+'\n');verify(output)
    print(json.dumps({'status':p['status'],'outputs':p['outputs']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('build')
    for key in ('timing-proof','source','patch','output'):p.add_argument('--'+key,type=Path,required=True)
    p=sub.add_parser('verify');p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='build':build(args)
    else:verify(args.output.resolve());print('Verified one-TU mapped polling candidate and common timing driver')
