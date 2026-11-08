#!/usr/bin/env python3
"""Build only a diagnostic entry against frozen v8; no solver/profiler execution."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time
import struct
import math

from legacy_n12_benchmark import VARIANTS,check
from legacy_stage_trace_gpu import require,sha

HELPER_SHA='abdc68c25086835224d2373832c545f634ee1f7075080274918051d2def0d604'


def build(args):
    legacy=args.legacy.resolve();cwd=args.link_cwd.resolve();out=args.output.resolve()
    commands=args.commands.resolve();binary=args.original_executable.resolve()
    binary_sha,commands_sha,operator_sha=VARIANTS['v8']
    require(sha(binary)==binary_sha and sha(commands)==commands_sha,'wrong frozen v8 binary/recipe')
    require(sha(legacy/'tst/hyperbolic_shift_3d/coupled_mpi.cpp')==operator_sha,'v8 numerical source differs')
    source=Path(__file__).with_suffix('.cpp').resolve();helper=source.with_name('legacy_n12_benchmark.cpp')
    require(sha(helper)==HELPER_SHA,'qualified diagnostic helper changed')
    archive=json.loads(commands.read_text());compile=list(archive['compile']);link=list(archive['link'])
    old_source=legacy/'tst/hyperbolic_shift_3d/endpoint_policy/profile.cpp'
    require(sha(old_source)=='6ef936917ff17bde4c1c565c4be3324c8599bd3ec13ab5b9e67933b38486c888',
            'v8 original entry changed')
    require(compile.count(str(old_source))==1 and compile.count('-o')==1 and '-c' in compile,
            'unrecognized compile command')
    for flag in ('-O2','-fp-model=precise','-ffp-contract=off','-fsycl'):
        require(flag in compile,'archived FP/device flag missing: '+flag)
    require(link.count('-o')==1 and link[link.index('-o')+1]==str(binary),'original link differs')
    out.mkdir(parents=True,exist_ok=False);obj=out/'endpoint-diagnostics.o';exe=out/'legacy_n12_endpoint_diagnostics'
    dep=out/'endpoint-diagnostics.d';old_object=compile[compile.index('-o')+1]
    compile[compile.index('-o')+1]=str(obj);compile[compile.index(str(old_source))]=str(source)
    compile+=['-I'+str(legacy),'-DLEGACY_BENCH_HAS_PERF=1','-MMD','-MF',str(dep),'-MT',str(obj)]
    require(link.count(old_object)==1,'unique original main object missing')
    link[link.index(old_object)]=str(obj);link[link.index('-o')+1]=str(exe)
    sources={str(p):sha(p) for p in sorted(legacy.rglob('*')) if p.is_file()}
    for p in (source,helper,Path(__file__).resolve(),commands,source.with_name('legacy_n12_endpoint_diagnostics_contract.md'),
              source.with_name('legacy_n12_benchmark.py'),source.with_name('legacy_stage_trace_gpu.py')):
        sources[str(p)]=sha(p)
    objects={}
    for token in link:
        if token.endswith(('.o','.a','.so','.dylib')) and token!=str(obj):
            path=(cwd/token).resolve();objects[str(path)]=sha(path)
    report=dict(schema='legacy-n12-endpoint-diagnostics-build-v1',status='building',
        original_executable=str(binary),original_executable_sha256=binary_sha,commands_file=str(commands),
        commands_sha256=commands_sha,numerical_source_root=str(legacy),coupled_mpi_sha256=operator_sha,
        original_compile_command=archive['compile'],original_link_command=archive['link'],
        compile_command=compile,link_command=link,working_directory=str(cwd),sources=sources,objects=objects,
        executable=str(exe),harness_object=str(obj),helper_sha256=HELPER_SHA,evolution_steps=0,
        gpu_executed=False,performance_measurement=False,
        scope='New entry only; unchanged v8 numerical/diagnostic code; included qualified benchmark main renamed and never called')
    proof=out/'provenance.json';proof.write_text(json.dumps(report,indent=2)+'\n');check(report)
    makefile=out/'endpoint.mk';makefile.write_text('all:\n\tcd '+shlex.quote(str(cwd))+' && '+shlex.join(compile)+
        '\n\tcd '+shlex.quote(str(cwd))+' && '+shlex.join(link)+'\n')
    begin=time.monotonic()
    with (out/'build.log').open('w') as log:
        result=subprocess.run(['make','-j64','-f',str(makefile)],stdout=log,stderr=subprocess.STDOUT)
    report.update(build_exit_code=result.returncode,build_seconds=time.monotonic()-begin,
                  status='failed' if result.returncode else 'built')
    proof.write_text(json.dumps(report,indent=2)+'\n');check(report);result.check_returncode()
    text=dep.read_text().replace('\\\n',' ')
    dependencies=[(cwd/token).resolve() for token in shlex.split(text.split(':',1)[1])]
    report.update(dependencies={str(p):sha(p) for p in dependencies},executable_sha256=sha(exe),
        harness_object_sha256=sha(obj),original_runtime_inputs_unchanged=True)
    proof.write_text(json.dumps(report,indent=2)+'\n');check(report)
    print(json.dumps({k:report[k] for k in ('status','build_exit_code','executable_sha256')}))


SEED_SHA='27245d19010dee629d12c6e59872564bd70c19af8c06aa41c46cbe6b58143f18'
DT=.00046944203735617445


def input_manifest(args):
    seed,accepted=args.seed.resolve(),args.accepted.resolve()
    require(sha(seed/'seed.json')==SEED_SHA,'changed qualified2486 seed manifest')
    seed_report=json.loads((seed/'seed.json').read_text());files={}
    for name,digest in seed_report['output_sha256'].items():
        path=seed/name;require(sha(path)==digest,'changed seed input: '+name);files[str(path)]=digest
    files[str(seed/'seed.json')]=SEED_SHA
    report=json.loads((accepted/'results.json').read_text())
    expected=0.
    for _ in range(2556):expected+=DT
    require(report['schema']=='legacy-n12-original-control-benchmark-v1' and report['passed'] is True and
            report['end_cycle']==2556 and report['time']==expected and report['dt']==DT and report['ranks']==24,
            'not the completed original-control2556 endpoint')
    paths=sorted(accepted.glob('accepted-final-*-rank*.bin'))
    require({p.name for p in paths}=={f'accepted-final-{region}-rank{rank}.bin'
            for region in ('bulk','scri') for rank in range(24)},'accepted trace file coverage differs')
    totals=[0,0]
    for path in paths:
        with path.open('rb') as f:header=f.read(40)
        require(len(header)==40,'truncated trace header')
        magic,count,nk,nv,region,records=struct.unpack('<16sQ4I',header)
        require(magic==b'LEGACY_TRACE_V1\0' and (nk,nv,records)==(6,39,1) and region in (0,1),
                'accepted trace schema differs')
        require(path.stat().st_size==40+336*count,'accepted trace extent differs')
        require(('scri' in path.name)==bool(region),'accepted region/name differs');totals[region]+=count
        files[str(path)]=sha(path)
    require(totals==[229392,864],'accepted complete point coverage differs')
    for path in (accepted/'endpoint-restart.bin',accepted/'results.json'):files[str(path)]=sha(path)
    require(not args.output.exists(),'refuse existing input binding')
    value=dict(schema='legacy-n12-endpoint-inputs-v1',passed=True,seed_directory=str(seed),
        accepted_directory=str(accepted),seed_manifest_sha256=SEED_SHA,cycle=2556,time=expected,dt=DT,
        ranks=24,bulk_points=totals[0],scri_points=totals[1],all31_plus8_accepted_input=True,
        files=files,scope='Exact pre/post input hashes; diagnostic process independently verifies actual loaded31+8')
    args.output.write_text(json.dumps(value,indent=2)+'\n')
    print('Bound exact original-control endpoint and2486 horizon seed')


def verify_inputs(path):
    p=json.loads(path.read_text());require(p['schema']=='legacy-n12-endpoint-inputs-v1' and p['passed'] is True,
                                        'missing endpoint input binding')
    for name,digest in p['files'].items():require(sha(name)==digest,'endpoint/seed input changed: '+name)
    print('Verified unchanged original-control endpoint and2486 horizon seed')


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('build')
    for name in ('legacy','link-cwd','commands','original-executable','output'):p.add_argument('--'+name,type=Path,required=True)
    p=sub.add_parser('verify');p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('inputs')
    for name in ('seed','accepted','output'):p.add_argument('--'+name,type=Path,required=True)
    p=sub.add_parser('verify-inputs');p.add_argument('--manifest',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='build':build(args)
    elif args.action=='inputs':input_manifest(args)
    elif args.action=='verify-inputs':verify_inputs(args.manifest)
    else:
        p=json.loads((args.output/'provenance.json').read_text())
        require(p['schema']=='legacy-n12-endpoint-diagnostics-build-v1' and p['status']=='built' and
                p['build_exit_code']==0 and p['helper_sha256']==HELPER_SHA,'unqualified endpoint build')
        check(p);require(sha(p['executable'])==p['executable_sha256'] and sha(p['harness_object'])==p['harness_object_sha256'],
                         'endpoint executable/object changed')
        print('Verified unchanged diagnostic-only v8 endpoint build')

if __name__=='__main__':main()
