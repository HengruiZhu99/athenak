#!/usr/bin/env python3
"""Build the exact reviewed candidate in a separate complete CMake source tree.

All translation units are compiled coherently. No baseline object is linked,
and original main source/compiler flags are preserved. No runs are launched.
"""
import argparse
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time

from native_amr_admission_candidate import SOURCES, SAFE_DT, check_package, compile_entries, require, sha


def files(root):
    return {str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()
            and '__pycache__' not in p.parts and '.git' not in p.parts}


def cache_overlay(cache, old_root, old_build, new_root, new_build):
    """Replay the actual resolved build configuration; change only paths/export."""
    result={}
    for line in cache.read_text().splitlines():
        match=re.match(r'^([^/#][^:]*):(BOOL|STRING|PATH|FILEPATH)=(.*)$',line)
        if not match:
            continue
        key,kind,value=match.groups()
        if key.endswith(('_SOURCE_DIR','_BINARY_DIR')):
            continue
        value=value.replace(str(old_build),str(new_build)).replace(str(old_root),str(new_root))
        result[key]=(kind,value)
    require(result.get('PROBLEM')==('STRING','built_in_pgens'),'candidate copy supports built_in_pgens only')
    result['CMAKE_EXPORT_COMPILE_COMMANDS']=('BOOL','ON')
    def bracket(value):
        delimiter='='
        while ']'+delimiter+']' in value:
            delimiter+='='
        return '['+delimiter+'['+value+']'+delimiter+']'
    return ''.join('set('+key+' '+bracket(value)+' CACHE '+kind+' "baseline resolved option" FORCE)\n'
                   for key,(kind,value) in sorted(result.items())), result


def normalized(entry, root, build):
    command=entry.get('arguments') or shlex.split(entry['command'])
    # CMake compile databases omit Makefile-only dependency switches.
    out=[];i=0
    while i<len(command):
        if command[i] in ('-MF','-MT','-MQ'):
            i+=2;continue
        if command[i] in ('-MD','-MMD','-MP'):
            i+=1;continue
        out.append(command[i].replace(str(build),'<BUILD>').replace(str(root),'<SOURCE>'))
        i+=1
    return out


def check(record, complete=True):
    require(record['schema']=='native-amr-coherent-candidate-build-v1','not a coherent candidate build')
    for group in ('baseline_files','candidate_files','objects','dependencies'):
        for name,digest in record.get(group,{}).items():
            require(sha(name)==digest,'changed '+group+' '+name)
    for key in ('package_manifest','baseline_native_executable','baseline_cache','builder'):
        require(sha(record[key])==record[key+'_sha256'],'changed '+key)
    check_package(Path(record['package_manifest']).parent,Path(record['baseline_root']))
    if complete:
        require(record['status']=='built' and record['baseline_inputs_unchanged']
                and record['all_native_compile_recipes_match'] and record['original_main_source_unchanged']
                and record['normalized_link_policy_matches'],
                'candidate compilation is incomplete/inconsistent')
        for key in ('compile_database','link_file'):
            require(sha(record[key])==record[key+'_sha256'],'changed candidate '+key)
        if 'finalizer' in record:
            require(sha(record['finalizer'])==record['finalizer_sha256'],'changed finalizer')
        require(sha(record['executable'])==record['executable_sha256'],'candidate executable changed')


def build(args):
    baseline=args.baseline_build.resolve();output=args.output.resolve();package=args.package.resolve()
    entries,recipe_inputs=compile_entries(baseline,args.compile_database)
    main=[e for e in entries if Path(e['file']).as_posix().endswith('/src/main.cpp')]
    require(len(main)==1,'missing original main entry')
    root=(Path(main[0]['directory'])/main[0]['file']).resolve().parents[1]
    manifest=check_package(package,root)
    output.mkdir(parents=True,exist_ok=False)
    source=output/'source';source.mkdir();candidate_build=output/'build'
    # Physical copies, not hardlinks or source symlinks: patching cannot touch baseline.
    for name in ('CMakeLists.txt','config.hpp.in','LICENSE'):
        shutil.copyfile(root/name,source/name)
    for name in ('src','kokkos','tst/z4c_hyperboloidal'):
        shutil.copytree(root/name,source/name,ignore=shutil.ignore_patterns('.git','__pycache__'))
    baseline_source={str(root/name):digest for name,digest in files(source).items()}
    for name in SOURCES:
        target=source/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(package/'overlay'/name,target)
    before={str(Path(name).relative_to(root)):digest for name,digest in baseline_source.items()}
    after=files(source)
    changed={name for name in before.keys()|after.keys() if before.get(name)!=after.get(name)}
    require(changed==set(SOURCES),'candidate source difference is not exactly reviewed patch')
    require(sha(source/'src/main.cpp')==sha(root/'src/main.cpp'),'ordinary main source changed')
    cache=baseline/'CMakeCache.txt';initial_cache=output/'baseline-options.cmake'
    content,options=cache_overlay(cache,root,baseline,source,candidate_build)
    initial_cache.write_text(content)
    baseline_link=baseline/'src/CMakeFiles/athena.dir/link.txt'
    link=shlex.split(baseline_link.read_text())
    baseline_objects={(baseline/'src'/p).resolve() for p in link if p.endswith(('.o','.a','.so','.dylib'))}
    baseline_files={**{name:digest for name,digest in baseline_source.items()
                      if not Path(name).is_relative_to(root/'tst') or Path(name).suffix in ('.cpp','.hpp','.txt')},
                    **recipe_inputs,str(baseline_link):sha(baseline_link),
                    **{str(p):sha(p) for p in baseline_objects}}
    configure=['cmake','-S',str(source),'-B',str(candidate_build),'-C',str(initial_cache),'-G','Unix Makefiles']
    compile_command=['make','-j64','athena']
    record=dict(schema='native-amr-coherent-candidate-build-v1',status='building',candidate=True,
        baseline_root=str(root),baseline_build=str(baseline),root=str(source),build=str(candidate_build),
        baseline_native_executable=str(baseline/'src/athena'),baseline_native_executable_sha256=sha(baseline/'src/athena'),
        baseline_cache=str(cache),baseline_cache_sha256=sha(cache),
        package_manifest=str(package/'manifest.json'),package_manifest_sha256=sha(package/'manifest.json'),
        production_patch_sha256=manifest['production_patch_sha256'],fixed_dt=SAFE_DT,
        changed_sources=sorted(changed),baseline_files=baseline_files,
        candidate_files={str(source/name):digest for name,digest in after.items()},
        builder=str(Path(__file__).resolve()),builder_sha256=sha(__file__),
        configure_command=configure,build_command=compile_command,cache_options=options,
        original_main_source_unchanged=True,original_main_sha256=sha(source/'src/main.cpp'),
        executable=str(candidate_build/'src/athena'),baseline_objects_linked=False)
    path=output/'provenance.json';path.write_text(json.dumps(record,indent=2)+'\n')
    start=time.monotonic()
    for label,command,cwd in [('configure',configure,output),('build',compile_command,candidate_build)]:
        with (output/(label+'.log')).open('w') as log:
            run=subprocess.run(command,cwd=cwd,stdout=log,stderr=subprocess.STDOUT)
        record[label+'_exit_code']=run.returncode
        if run.returncode:
            record['status']='failed';path.write_text(json.dumps(record,indent=2)+'\n')
            check(record,False);run.check_returncode()
    record['build_seconds']=time.monotonic()-start
    finalize(record,path)


def finalize(record,path):
    baseline=Path(record['baseline_build']);root=Path(record['baseline_root'])
    source=Path(record['root']);candidate_build=Path(record['build'])
    entries,_=compile_entries(baseline)
    baseline_objects={Path(p) for p in record['baseline_files'] if p.endswith(('.o','.a','.so','.dylib'))}
    def native(entry):
        command=entry.get('arguments') or shlex.split(entry['command'])
        return '-o' in command and 'athena.dir/' in command[command.index('-o')+1]
    actual=json.loads((candidate_build/'compile_commands.json').read_text())
    baseline_by_name={str((Path(e['directory'])/e['file']).resolve().relative_to(root)):e for e in entries
                      if native(e)}
    actual_by_name={str((Path(e['directory'])/e['file']).resolve().relative_to(source)):e for e in actual
                    if native(e)}
    require(baseline_by_name.keys()==actual_by_name.keys(),'native translation-unit coverage changed')
    mismatches=[name for name,entry in actual_by_name.items()
                if normalized(entry,source,candidate_build)!=normalized(baseline_by_name[name],root,baseline)]
    record['compile_recipe_mismatches']=mismatches
    require(not mismatches,'candidate flags differ from actual baseline: '+str(mismatches[:3]))
    link_file=candidate_build/'src/CMakeFiles/athena.dir/link.txt';candidate_link=shlex.split(link_file.read_text())
    baseline_link=shlex.split((baseline/'src/CMakeFiles/athena.dir/link.txt').read_text())
    def normalized_link(command,source,build):
        return [token.replace(str(build),'<BUILD>').replace(str(source),'<SOURCE>') for token in command]
    require(normalized_link(candidate_link,source,candidate_build)==normalized_link(baseline_link,root,baseline),
            'candidate link policy differs from baseline flags/libraries/order')
    candidate_objects=[(candidate_build/'src'/p).resolve() for p in candidate_link if p.endswith(('.o','.a','.so','.dylib'))]
    require(not set(candidate_objects)&{p for p in baseline_objects if p.is_relative_to(baseline)},
            'candidate links baseline object/library')
    require(all(p.is_relative_to(candidate_build) for p in candidate_objects if p.suffix in ('.o','.a')),
            'candidate native/Kokkos object outside coherent build')
    dependencies={}
    for name in actual_by_name:
        dep=candidate_build/'src/CMakeFiles/athena.dir'/(name.removeprefix('src/')+'.o.d')
        require(dep.is_file(),'missing candidate dependency file '+str(dep))
        for token in shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1]):
            p=(candidate_build/'src'/token).resolve();dependencies[str(p)]=sha(p)
    require(str(source/'src/z4c_hyperboloidal/z4c_hyperboloidal.hpp') in dependencies,
            'candidate class definition not in native dependencies')
    record.update(status='built',
        executable_sha256=sha(record['executable']),all_native_compile_recipes_match=True,
        native_translation_units=len(actual_by_name),baseline_inputs_unchanged=True,
        objects={str(p):sha(p) for p in candidate_objects},dependencies=dependencies,
        compile_commands=actual,link_command=candidate_link,
        normalized_link_policy_matches=True,link_file=str(link_file),link_file_sha256=sha(link_file),
        compile_database=str(candidate_build/'compile_commands.json'),
        compile_database_sha256=sha(candidate_build/'compile_commands.json'))
    path.write_text(json.dumps(record,indent=2)+'\n');check(record)
    print(json.dumps({k:record[k] for k in ('status','native_translation_units','executable_sha256',
          'baseline_native_executable_sha256','production_patch_sha256')},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('build');p.add_argument('--baseline-build',type=Path,required=True)
    p.add_argument('--package',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--compile-database',type=Path)
    p=sub.add_parser('verify');p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('finalize');p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('audit');p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='build':build(args)
    elif args.command=='verify':
        check(json.loads((args.output/'provenance.json').read_text()));print('Verified coherent candidate and frozen baseline')
    elif args.command=='finalize':
        path=args.output/'provenance.json';record=json.loads(path.read_text())
        require(record['status']=='building','finalization requires unadmitted completed build')
        preserved=args.output/'pre-finalization-failure.json'
        require(not preserved.exists(),'audit attempt already preserved');shutil.copyfile(path,preserved)
        original=Path(record['root'])/'tst/z4c_hyperboloidal'/Path(__file__).name
        require(sha(original)==record['builder_sha256'],'original build script is not preserved')
        record['build_builder_original_path']=record['builder'];record['builder']=str(original)
        record['finalizer']=str(Path(__file__).resolve());record['finalizer_sha256']=sha(__file__)
        record['audit_correction']='Select athena.dir recipes, excluding separate unit-test target recipes; no source/object rebuild'
        finalize(record,path)
    else:
        previous=args.output/'provenance.json';path=args.output/'audited-provenance.json'
        require(not path.exists(),'new audit record required')
        record=json.loads(previous.read_text());require(record['status']=='built','coherent build must exist')
        record['prior_provenance']=str(previous.resolve());record['prior_provenance_sha256']=sha(previous)
        record['finalizer']=str(Path(__file__).resolve());record['finalizer_sha256']=sha(__file__)
        record['audit_correction']='Enforce exact normalized link policy and bind compile/link/finalizer files; no rebuild'
        finalize(record,path)
