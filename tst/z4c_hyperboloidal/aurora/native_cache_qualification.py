#!/usr/bin/env python3
"""Thin operational glue around existing native qualification executables/readers."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import native_ipc_cache_audit as cache
import observer_run_audit as observer
import amr_run_audit as amr
import native_stage_trace_compare as trace

ROOT,LAUNCH=cache.ROOT,cache.LAUNCH
CASE=cache.CASE/'qualification'
require,sha,read_json,write_json=cache.require,cache.sha,cache.read_json,cache.write_json
MANIFEST='08de71e12bbe7d7ea0dd5013b89e82e530067061e8105cdc1a3327af5463c344'
CAPS={'observer70':600,'poison8':180,'amr4':300,'prefix2':120,'resume8':180}


def manifest():
    p=LAUNCH/'qualification-commands.json'
    require(sha(p)==MANIFEST,'qualification command/readiness manifest changed')
    return read_json(p)


def spec(name):
    matches=[x for x in manifest()['cases'] if x['name']==name]
    require(len(matches)==1 and name in CAPS,'unknown qualification case')
    return matches[0]


def command(name):
    s=spec(name)
    return s['command_prefix']+[s['executable']]+[a.replace('NEW/',str(CASE)+'/') for a in s['arguments']]


def guard(suffix):
    cache.unprofiled()
    cache.base.guard(SimpleNamespace(admission=LAUNCH/'admission.json',directory=CASE,suffix=suffix))
    m=manifest();hashes=dict(m['input_and_reference_sha256'])
    for s in m['cases']:
        hashes[s['executable']]=s['executable_sha256'];hashes[s['provenance']]=s['provenance_sha256']
    coherent=ROOT/'validation/amr-admission-candidate-v1'
    hashes[str(coherent/'coherent/provenance.json')]='f4941ccd95ebee95aac62842aa6e06266872b65621a80f43e8ad5ae4532eb340'
    for name,digest in hashes.items():require(sha(name)==digest,'qualification input changed: '+name)
    log=CASE/f'coherent-verify-{suffix}.txt'
    with log.open('w') as output:
        result=subprocess.run(['/usr/bin/python3.11',str(coherent/'package/tools/native_amr_candidate_cmake.py'),
            'verify','--output',str(coherent/'coherent')],stdout=output,stderr=subprocess.STDOUT)
    require(result.returncode==0,'coherent AMR provenance verify failed')
    write_json(CASE/f'qualification-guard-{suffix}.json',dict(passed=True,inputs=hashes,
        command_manifest_sha256=MANIFEST,base_guard_sha256=sha(CASE/f'guard-{suffix}.json')))


def environment(name):
    cache.configure('timed-K0');cache.unprofiled()
    write_json(CASE/name/'environment.json',cache.base.environment(CASE/name))


def completed(name):
    folder=CASE/name;s=spec(name);reference=Path(s['reference_paths'][0])
    require((folder/'exit-code.txt').read_text().strip()=='0','qualification solver failed')
    require(shlex.split((folder/'command.txt').read_text())==command(name),'qualification command differs')
    cache.configure('timed-K0')
    require(read_json(folder/'environment.json')==cache.base.environment(folder),'qualification environment changed')
    cache.rank_cache(folder/'rank-cache.txt','timed-K0')
    require(shlex.split((folder/'rank-cache-command.txt').read_text())==
            ['mpiexec','-n','24','-ppn','12',cache.base.BINDING,cache.base.WRAPPER,
             '/usr/bin/python3.11',str(LAUNCH/'native_ipc_cache_audit.py'),'rank-environment'],
            'qualification rank cache probe command differs')
    report=dict(schema='native-cache-case-operational-v1',passed=True,case=name,
                performance_admission=False,independent_scientific_review_pending=True)
    if name=='observer70':
        require(read_json(folder/'native-step-observer.json')==read_json(reference/'native-step-observer.json'),
                'observer metadata differs')
        require((folder/'native-stages.csv').read_bytes()==(reference/'native-stages.csv').read_bytes(),
                '280 observer stage rows differ')
        report['full_state']=cache.compare_checkpoints(
            cache.checkpoint(reference/'rst/native_n12_5M.00008.rst',2556),
            cache.checkpoint(folder/'rst/native_n12_5M.00008.rst',2556))
        require(all((folder/n).read_bytes()==(reference/n).read_bytes() for n in cache.DIAGNOSTICS),
                'observer70 diagnostics differ')
        report['stage_rows']=280
    elif name=='poison8':
        result=trace.audit(folder,reference);require(result['passed'],'poison/clean trace comparison failed')
        report['trace_audit_sha256']=sha(folder/'trace-audit.json')
    elif name=='amr4':
        amr.run(folder,24,'evolution');references={}
        for cycle in range(5):
            for rank in range(24):
                for suffix in ('.bin','.txt'):
                    filename=f'cycle{cycle}-rank{rank}'+suffix
                    digest=sha(reference/filename);require(sha(folder/filename)==digest,'AMR snapshot differs: '+filename)
                    references[str(reference/filename)]=digest
        report['reference_snapshots_sha256']=references
    else:
        start,end=(0,2) if name=='prefix2' else (2,8)
        observer.run(folder,24,start,end)
        if name=='prefix2':
            report['full_state']=cache.compare_checkpoints(
                cache.checkpoint(reference/'rst/native_n12_5M.00001.rst',2),
                cache.checkpoint(folder/'rst/native_n12_5M.00001.rst',2))
        else:
            a=(CASE/'prefix2/native-stages.csv').read_text().splitlines()
            b=(folder/'native-stages.csv').read_text().splitlines()
            require(a+b[1:]==(reference/'native-stages.csv').read_text().splitlines(),
                    'joined restart32 stage rows differ from continuous8')
            report['restart_payload_and_diagnostics']='pending existing local full-state and restart diagnostic readers'
    write_json(folder/'qualification-case.json',report)


def gate():
    screen=read_json(cache.CASE/'first-pair-screen.json')
    clean=not screen['clearly_slower'] and all((cache.CASE/n/'operational.json').is_file() for n in cache.ORDER)
    values={}
    if clean:
        for n in cache.ORDER:
            require(read_json(cache.CASE/n/'operational.json')==cache.completed(n),'ABBA case proof changed')
            values[n]=[w['seconds_per_complete_step'] for w in read_json(cache.CASE/n/'native-cycle-timing.json')['windows']]
    c=values.get('timed-C0',[])+values.get('timed-C1',[])
    k=values.get('timed-K0',[])+values.get('timed-K1',[])
    run=clean and max(k)<min(c)
    write_json(cache.CASE/'qualification-gate.json',dict(run_qualification=run,all_four_clean=clean,
        control_windows=c,candidate_windows=k,criterion='max(all six K windows) < min(all six C windows)',
        scope='operational spending gate only; no retention or parity admission'))
    print('run' if run else 'skip')


def finish():
    require(read_json(CASE/'qualification-guard-before.json')==read_json(CASE/'qualification-guard-after.json'),
            'qualification frozen guards changed')
    reports={n:read_json(CASE/n/'qualification-case.json') for n in CAPS}
    require(all(p['passed'] for p in reports.values()),'qualification case incomplete')
    write_json(CASE/'qualification-operational.json',dict(passed=True,cases=reports,
        independent_scientific_review_pending=True,performance_admission=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    for mode in ('command','environment','completed'):
        a=sub.add_parser(mode);a.add_argument('name',choices=tuple(CAPS))
    a=sub.add_parser('guard');a.add_argument('suffix',choices=('before','after'))
    sub.add_parser('gate');sub.add_parser('finish')
    a=p.parse_args()
    if a.mode=='command':sys.stdout.buffer.write(b'\0'.join(x.encode() for x in command(a.name))+b'\0')
    elif a.mode=='environment':environment(a.name)
    elif a.mode=='completed':completed(a.name)
    elif a.mode=='guard':guard(a.suffix)
    elif a.mode=='gate':gate()
    else:finish()
