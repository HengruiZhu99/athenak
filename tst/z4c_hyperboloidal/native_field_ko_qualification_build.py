#!/usr/bin/env python3
"""Execute the pinned four-link KO qualification plan without running a solver."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path('/lus/flare/projects/CompactBinaryMerger/hzhu/scri-native-20261108-side')
WORK = ROOT/'validation/perf-two-node-v1/field-ko-v1'
PLAN = 'd7e4861a0ad97d5ba44e04bc2cef33c2864d533020c379f23fcb989baf8bcd42'
PROOFS = {
    'observer70': ROOT/'validation/perf-two-node-v1/native-timed-observer/build/provenance.json',
    'poison8': ROOT/'validation/stage-trace/build/provenance.json',
    'restart': ROOT/'validation/step-observer/build/provenance.json',
    'amr4': ROOT/'validation/amr-admission-candidate-v1/evolution-build/provenance.json',
}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def verify(output):
    p = json.loads((output/'provenance.json').read_text())
    require(p['schema']=='native-field-ko-qualification-build-v1' and
            p['status']=='built' and p['build_exit_code']==0 and
            p['plan_sha256']==PLAN and p['gpu_executed'] is False,
            'incomplete or different qualification build')
    require(set(p['executables'])=={'observer70','poison8','restart','amr4'},
            'qualification executable coverage differs')
    for group in ('inputs','dependencies','outputs'):
        for name, digest in p[group].items():
            require(sha(name)==digest, 'changed '+group+': '+name)
    return p


def build(plan_path, output):
    require(sha(plan_path)==PLAN, 'qualification plan changed')
    plan = json.loads(plan_path.read_text())
    output.mkdir(parents=True, exist_ok=False)
    inputs = {}

    def bind(path, digest=None):
        path = str(Path(path).resolve())
        digest = digest or sha(path)
        require(path not in inputs or inputs[path]==digest, 'conflicting input: '+path)
        inputs[path] = digest

    def proof(path, digest):
        require(sha(path)==digest, 'changed parent proof: '+str(path))
        bind(path, digest)
        p = json.loads(Path(path).read_text())
        require(p['status']=='built' and p['build_exit_code']==0, 'parent build failed')
        for group in ('inputs','outputs','sources','objects','dependencies','linked_inputs',
                      'builder_sources','baseline_files','candidate_files'):
            for name, value in p.get(group, {}).items():
                require(isinstance(value,str) and len(value)==64, 'unknown hash schema')
                bind(name,value)
        for name,value in p.items():
            if isinstance(value,str) and value.startswith('/') and name+'_sha256' in p:
                bind(value,p[name+'_sha256'])
        return p

    bind(plan_path, PLAN);bind(Path(__file__).resolve())
    candidate = plan['candidate']
    proof(WORK/'build/provenance.json', candidate['proof_sha256'])
    source=Path(candidate['source']);bind(source,candidate['source_sha256'])
    bind(candidate['patch'],candidate['patch_sha256'])
    rhs=Path(candidate['f3_object']);bind(rhs,candidate['f3_object_sha256'])
    coherent_path=ROOT/'validation/amr-admission-candidate-v1/coherent/provenance.json'
    coherent=proof(coherent_path,plan['amr_compatibility']['coherent_proof_sha256'])
    entry=plan['amr_compatibility']['compile_entry']
    database=Path(coherent['compile_database'])
    actual=[e for e in json.loads(database.read_text()) if e['file']==entry['file']]
    require(actual==[entry], 'coherent compile entry changed')
    original_source=Path(entry['file']);cwd=Path(entry['directory'])
    bind(original_source,plan['amr_compatibility']['original_rhs_sha256'])
    bind(original_source.with_name('z4c_hyperboloidal.hpp'),
         plan['amr_compatibility']['distinct_class_header_sha256'])
    obj=output/'coherent-z4c_hyperboloidal_rhs.cpp.o';dep=output/'coherent-rhs.d'
    original=shlex.split(entry['command']);command=[];i=0;ns=no=0
    while i<len(original):
        token=original[i]
        if token=='-o':command+=['-o',str(obj)];no+=1;i+=2;continue
        if token in ('-MF','-MT','-MQ'):i+=2;continue
        if token in ('-MD','-MMD','-MP'):i+=1;continue
        if not token.startswith('-') and (cwd/token).resolve()==original_source:
            command.append(str(source));ns+=1
        else:command.append(token)
        i+=1
    require(ns==no==1 and '-c' in command and '-ffp-contract=off' in command,
            'unexpected coherent RHS recipe')
    command+=['-iquote',str(original_source.parent),'-MMD','-MF',str(dep),'-MT',str(obj)]
    links={};executables={}
    for name,spec in plan['relinks'].items():
        parent=proof(PROOFS[name],spec['proof_sha256'])
        link=list(parent[spec['link_command_key']]);original_link=list(link)
        require(parent['link_cwd']==spec['link_cwd'], 'link directory changed')
        exe=output/('athena-ko-'+name)
        for change in spec['replace_only']:
            index=change['argv_index']
            require(link[index]==change['original'], 'link argument changed')
            link[index]=str(exe if original_link[index-1]=='-o' else obj if name=='amr4' else rhs)
        require([i for i,(a,b) in enumerate(zip(link,original_link)) if a!=b]==[60,211],
                'unexpected qualification link replacements')
        for path,digest in spec['frozen_objects'].items():bind(path,digest)
        for token in original_link:
            if token.endswith(('.o','.a','.so','.dylib')):
                path=(Path(spec['link_cwd'])/token).resolve()
                require(str(path) in inputs, 'unbound parent link input: '+str(path))
        links[name]=dict(original=original_link,command=link,cwd=spec['link_cwd'],
                         parent_proof=str(PROOFS[name]),parent_sha256=spec['proof_sha256'])
        executables[name]=str(exe)
    for name,digest in inputs.items():require(sha(name)==digest, 'prebuild input changed: '+name)
    p=dict(schema='native-field-ko-qualification-build-v1',status='building',
           plan=str(plan_path),plan_sha256=PLAN,inputs=inputs,compile_command=command,
           original_compile_entry=entry,compile_cwd=str(cwd),link_commands=links,
           executables=executables,coherent_rhs_object=str(obj),f3_rhs_object=str(rhs),
           changed_numerical_source='src/z4c_hyperboloidal/z4c_hyperboloidal_rhs.cpp',
           same_candidate_source_both_builds=True,coherent_class_header_preserved=True,
           gpu_executed=False)
    record=output/'provenance.json';write(record,p)
    makefile=output/'qualification.mk'
    recipe='.PHONY: all\nall:\n\tcd '+shlex.quote(str(cwd))+' && '+shlex.join(command)+'\n'
    for link in links.values():recipe+='\tcd '+shlex.quote(link['cwd'])+' && '+shlex.join(link['command'])+'\n'
    makefile.write_text(recipe);start=time.monotonic()
    with (output/'build.log').open('w') as log:
        result=subprocess.run(['make','-j64','-f',str(makefile)],stdout=log,stderr=subprocess.STDOUT)
    p.update(build_exit_code=result.returncode,build_seconds=time.monotonic()-start,
             status='failed' if result.returncode else 'built');write(record,p)
    for name,digest in inputs.items():require(sha(name)==digest, 'postbuild input changed: '+name)
    result.check_returncode()
    deps=shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1])
    p['dependencies']={str((cwd/n).resolve()):sha((cwd/n).resolve()) for n in deps}
    p['outputs']={str(obj):sha(obj),**{v:sha(v) for v in executables.values()}}
    p['runtime_inputs_unchanged']=True;write(record,p);verify(output)
    print(json.dumps(dict(status=p['status'],outputs=p['outputs']),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('build');p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('verify');p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='build':build(args.plan.resolve(),args.output.resolve())
    else:verify(args.output.resolve());print('Verified four KO qualification links and unchanged frozen inputs')
