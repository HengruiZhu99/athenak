#!/usr/bin/env python3
"""Link the qualified observer main and whole-cycle timer; never run the solver.

Both existing instrumentation objects and every native numerical object remain
unchanged. This is measurement infrastructure, not a numerical optimization.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import time
from types import SimpleNamespace

import native_cycle_timing as timing
import native_step_observer as observer

BASELINE = 'f3a85db7528986ddec347b31675a8c5d588120a2380133aed73d5967f7aab05c'
TIMING_PROOF = '89265de52cfec416139ddf4188618b10b91b4e130c3fa844ffeed473d0d492ea'
OBSERVER_PROOF = '642507b2529419d290f00eea609d8ebc0ba6ce5dc52ed41d6cfe2fe995eb1a8a'
sha, require = timing.sha, timing.require


def inputs(timing_path, observer_path):
    require(sha(timing_path) == TIMING_PROOF and sha(observer_path) == OBSERVER_PROOF,
            'not the qualified timing/observer build proofs')
    timing.verify(SimpleNamespace(output=timing_path.parent))
    observer.verify(SimpleNamespace(output=observer_path.parent))
    t, o = [json.loads(p.read_text()) for p in (timing_path,observer_path)]
    require(t['native_executable_sha256'] == o['native_executable_sha256'] == BASELINE,
            'instrumentation objects do not share the frozen native baseline')
    require(t['compile_database_sha256'] == o['compile_database_sha256'] and
            t['link_file_sha256'] == o['link_file_sha256'] and
            t['link_file'] == o['link_file'], 'instrumentation build recipes differ')
    root = Path(o['overlay']['original_source']).parents[1]
    require(Path(t['overlay']['original_source']).parents[2] == root,
            'timing and observer source trees differ')
    native_sources = lambda p:{name:digest for name,digest in p['sources'].items()
                              if Path(name).is_relative_to(root/'src')}
    require(native_sources(t) == native_sources(o) and native_sources(t),
            'native definitions differ across instrumentation objects')
    return t, o


def link_command(t, o, executable):
    command = shlex.split(Path(t['link_file']).read_text())
    for suffix,replacement in [('athena.dir/main.cpp.o',o['harness_object']),
                               ('athena.dir/driver/driver.cpp.o',t['harness_object'])]:
        positions = [i for i,x in enumerate(command) if x.endswith(suffix)]
        require(len(positions) == 1, 'missing/repeated original object '+suffix)
        command[positions[0]] = replacement
    require(command.count('-o') == 1, 'unrecognized native link output')
    command[command.index('-o')+1] = str(executable)
    return command


def verify(path):
    p = json.loads(path.read_text())
    require(p['schema'] == 'native-timed-observer-build-v1' and
            p['status'] == 'built' and p['build_exit_code'] == 0,
            'timed observer did not build')
    t,o = inputs(Path(p['timing_proof']),Path(p['observer_proof']))
    require(link_command(t,o,Path(p['executable'])) == p['link_command'],
            'link changed beyond the two instrumentation objects/output')
    for group in ('linked_inputs','builder_sources'):
        for name,digest in p[group].items(): require(sha(name) == digest,'changed '+group+' '+name)
    require(sha(p['executable']) == p['executable_sha256'],'timed observer executable changed')
    return p


def build(args):
    tp,op = args.timing_proof.resolve(),args.observer_proof.resolve()
    t,o = inputs(tp,op)
    output = args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    executable = output/'athena-timed-observer'
    command = link_command(t,o,executable);cwd = Path(t['link_cwd'])
    linked = [(cwd/token).resolve() for token in command
              if token.endswith(('.o','.a','.so','.dylib'))]
    require(len(linked) == len(set(linked)),'repeated linked object')
    sources = [Path(__file__).resolve(),Path(timing.__file__).resolve(),Path(observer.__file__).resolve(),
               Path(__file__).with_name('native_n12_two_node_timing_contract.md').resolve()]
    p = dict(schema='native-timed-observer-build-v1',status='building',
        baseline_executable_sha256=BASELINE,timing_proof=str(tp),timing_proof_sha256=sha(tp),
        observer_proof=str(op),observer_proof_sha256=sha(op),
        timing_executable_sha256=t['executable_sha256'],observer_executable_sha256=o['executable_sha256'],
        main_object=o['harness_object'],main_object_sha256=o['harness_object_sha256'],
        driver_object=t['harness_object'],driver_object_sha256=t['harness_object_sha256'],
        link_command=command,link_cwd=str(cwd),linked_inputs={str(x):sha(x) for x in linked},
        builder_sources={str(x):sha(x) for x in sources},executable=str(executable),
        compilation_performed=False,only_main_driver_output_replaced=True,
        gpu_executed=False,scope='same whole-cycle timer with qualified stage observer inside each cycle')
    proof = output/'provenance.json';proof.write_text(json.dumps(p,indent=2)+'\n')
    makefile = output/'timed-observer.mk'
    makefile.write_text('.PHONY: all\nall:\n\tcd '+shlex.quote(str(cwd))+' && '+shlex.join(command)+'\n')
    start = time.monotonic()
    with (output/'build.log').open('w') as log:
        result = subprocess.run(['make','-j64','-f',str(makefile)],stdout=log,stderr=subprocess.STDOUT)
    p.update(build_seconds=time.monotonic()-start,build_exit_code=result.returncode,
             status='failed' if result.returncode else 'built')
    proof.write_text(json.dumps(p,indent=2)+'\n');inputs(tp,op);result.check_returncode()
    for name,digest in p['linked_inputs'].items():require(sha(name) == digest,'linked input changed')
    p.update(executable_sha256=sha(executable),runtime_inputs_unchanged=True)
    proof.write_text(json.dumps(p,indent=2)+'\n');verify(proof)
    print(json.dumps({k:p[k] for k in ('status','build_exit_code','executable_sha256',
                                      'runtime_inputs_unchanged')},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('build')
    for name in ('timing-proof','observer-proof','output'):p.add_argument('--'+name,type=Path,required=True)
    p = sub.add_parser('verify');p.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.command == 'build':build(args)
    else:verify(args.output/'provenance.json');print('Verified unchanged native timed observer composition')
