#!/usr/bin/env python3
"""Build a benchmark-only entry point against a preserved predecessor variant.

No numerical source is copied or patched. Archived compile/link arguments and
source/binary identities are checked; make -j64 writes only a new output path.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time

from legacy_stage_trace_gpu import require, sha

VARIANTS = {
    'v3': ('01322937c357c48869e13461d8f7d8d9f0299458ea71daf09efb5d124d8e4d30',
           'd8d62fe5bba2310372d2a4c9969c9e7b187be75099d38fb8ca9f0b8ac2d3625f',
           'e63683eea6ede63b87fa424496594c59753ed823a34b039cc6c3317dc6dca89f'),
    'v6': ('ec2493f158a0f47d8d21f70bc16d1975dc89f31629980c4c6aa5b98a5f4d42d4',
           '3947b08cd84611752f7f007f80f3e497f82b38a2616e3dff67d015a40297b6ae',
           '02cf4f230d6967c8e9b27bdd91f875e7295698775900ed50510e5aa53c385cfd'),
    'v8': ('38b35d268a5a4e8015fca3cbfc056c8055ba41661899e2172f596ed38413916c',
           '2893c335bb81343924f83c12ea09cea59f5fd99ea17c7651957969f04176c0a9',
           'a39106792018eae2bc0535444ada15badafb8b80fed28385da4b8ac45e88ac3b'),
    'control': ('494ec8e38fb34806eb3981fb19bdb5e3bc75f46615b05a21d8314a0db6583753',
                None, '87a8d90396bb3ab6e72efadab0e9491322517bf63413f8f4460a42f24f51c115'),
}


def check(record):
    for group in ('sources','objects','dependencies'):
        for path,digest in record.get(group,{}).items():
            require(sha(path)==digest,'changed '+group+': '+path)
    require(sha(record['original_executable'])==record['original_executable_sha256'],
            'preserved predecessor executable changed')
    require(sha(record['commands_file'])==record['commands_sha256'],
            'archived command record changed')


def build(args):
    legacy=args.legacy.resolve(); cwd=args.link_cwd.resolve(); output=args.output.resolve()
    commands=args.commands.resolve(); original_binary=args.original_executable.resolve()
    binary_hash,command_hash,operator_hash=VARIANTS[args.variant]
    require(sha(original_binary)==binary_hash,'wrong preserved variant binary')
    archive=json.loads(commands.read_text())
    if args.variant=='control':
        require(len(archive['commands'])==1,'ambiguous pinned original command')
        entry=archive['commands'][0]
        require(archive['binaries'].get(str(original_binary))==binary_hash,
                'original manifest does not bind control binary')
        for relative,digest in archive['sources'].items():
            require(sha(legacy/relative)==digest,'pinned control source changed: '+relative)
        original_source=legacy/'tst/hyperbolic_shift_3d/endpoint_policy/run.cpp'
    else:
        require(sha(commands)==command_hash,'wrong archived variant command record')
        entry=archive
        original_source=legacy/'tst/hyperbolic_shift_3d/endpoint_policy/profile.cpp'
        require(sha(original_source)=='6ef936917ff17bde4c1c565c4be3324c8599bd3ec13ab5b9e67933b38486c888',
                'archived profile entry source differs')
    require(sha(legacy/'tst/hyperbolic_shift_3d/coupled_mpi.cpp')==operator_hash,
            'archived numerical driver differs')
    compile_command=list(entry['compile']); link=list(entry['link'])
    require(compile_command.count(str(original_source))==1 and
            compile_command.count('-o')==1 and '-c' in compile_command,
            'unrecognized archived compile command')
    for flag in ('-O2','-fp-model=precise','-ffp-contract=off','-fsycl'):
        require(flag in compile_command,'archived FP/device policy differs: '+flag)
    require(link.count('-o')==1 and link[link.index('-o')+1]==str(original_binary),
            'archived link does not produce pinned variant')
    source=Path(__file__).with_suffix('.cpp').resolve()
    output.mkdir(parents=True,exist_ok=False)
    obj=output/'legacy_n12_benchmark.o'; executable=output/'legacy_n12_benchmark'
    dependency=output/'legacy_n12_benchmark.d'
    original_object=compile_command[compile_command.index('-o')+1]
    compile_command[compile_command.index('-o')+1]=str(obj)
    compile_command[compile_command.index(str(original_source))]=str(source)
    compile_command+=['-I'+str(legacy),'-MMD','-MF',str(dependency),'-MT',str(obj)]
    if args.variant!='control': compile_command+=['-DLEGACY_BENCH_HAS_PERF=1']
    require(link.count(original_object)==1,'missing unique predecessor main object')
    link[link.index(original_object)]=str(obj);link[link.index('-o')+1]=str(executable)
    sources={str(p):sha(p) for p in sorted(legacy.rglob('*')) if p.is_file()}
    for p in (source,Path(__file__).resolve(),commands,
              source.with_name('legacy_n12_benchmark_contract.md'),
              source.with_name('legacy_stage_trace_gpu.py')): sources[str(p)]=sha(p)
    objects={}
    for token in link:
        if token.endswith(('.o','.a','.so','.dylib')) and token!=str(obj):
            p=(cwd/token).resolve();objects[str(p)]=sha(p)
    record=dict(schema='legacy-n12-matched-benchmark-build-v1',status='building',
        variant=args.variant,original_executable=str(original_binary),
        original_executable_sha256=binary_hash,commands_file=str(commands),
        commands_sha256=sha(commands),numerical_source_root=str(legacy),
        coupled_mpi_sha256=operator_hash,original_compile_command=entry['compile'],
        original_link_command=entry['link'],compile_command=compile_command,
        link_command=link,working_directory=str(cwd),sources=sources,objects=objects,
        executable=str(executable),harness_object=str(obj),gpu_executed=False,
        runtime_scope='unchanged archived numerical source and reused libraries; benchmark entry TU only')
    proof=output/'provenance.json';proof.write_text(json.dumps(record,indent=2)+'\n');check(record)
    makefile=output/'benchmark.mk'
    makefile.write_text('.PHONY: all\nall:\n\tcd '+shlex.quote(str(cwd))+' && '+
        shlex.join(compile_command)+'\n\tcd '+shlex.quote(str(cwd))+' && '+shlex.join(link)+'\n')
    began=time.monotonic()
    with (output/'build.log').open('w') as log:
        result=subprocess.run(['make','-j64','-f',str(makefile)],stdout=log,stderr=subprocess.STDOUT)
    record.update(build_exit_code=result.returncode,build_seconds=time.monotonic()-began,
                  status='failed' if result.returncode else 'built')
    proof.write_text(json.dumps(record,indent=2)+'\n');check(record);result.check_returncode()
    text=dependency.read_text().replace('\\\n',' ')
    dependencies=[(cwd/token).resolve() for token in shlex.split(text.split(':',1)[1])]
    record.update(dependencies={str(p):sha(p) for p in dependencies},
        executable_sha256=sha(executable),harness_object_sha256=sha(obj),
        original_runtime_inputs_unchanged=True)
    proof.write_text(json.dumps(record,indent=2)+'\n');check(record)
    print(json.dumps({k:record[k] for k in ('variant','status','build_exit_code',
        'executable_sha256','original_runtime_inputs_unchanged')},indent=2))


def verify(args):
    r=json.loads((args.output.resolve()/'provenance.json').read_text())
    require(r['schema']=='legacy-n12-matched-benchmark-build-v1' and r['status']=='built'
            and r['build_exit_code']==0,'benchmark build not successful')
    require(r['original_executable_sha256']==VARIANTS[r['variant']][0] and
            r['coupled_mpi_sha256']==VARIANTS[r['variant']][2], 'variant identity changed')
    check(r)
    for key in ('executable','harness_object'):
        require(sha(r[key])==r[key+'_sha256'],'benchmark output changed: '+key)
    print('Verified benchmark and unchanged archived predecessor inputs')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('build');p.add_argument('--variant',choices=VARIANTS,required=True)
    for key in ('legacy','commands','original-executable','link-cwd','output'):
        p.add_argument('--'+key,type=Path,required=True)
    p=sub.add_parser('verify');p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();{'build':build,'verify':verify}[args.command](args)
