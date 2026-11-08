#!/usr/bin/env python3
"""Offline stdlib controls; no profiler, solver, remote write or GPU execution."""
import argparse
import ast
import copy
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0,str(Path(__file__).with_name('aurora')))
import native_n12_profile_audit as audit
from native_two_node_timing_audit import sha,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);checks=[]
    def ok(name,call):call();checks.append(dict(name=name,passed=True,expected='accept'))
    def no(name,call):
        try:call()
        except (ValueError,KeyError,AssertionError,json.JSONDecodeError) as e:
            checks.append(dict(name=name,passed=True,expected='reject',message=str(e)));return
        raise AssertionError('accepted negative '+name)
    source=Path('build-native/aurora-n12-common-07/run-artifacts/runs/native-n12-common-07/rst/native_n12_5M.00007.rst')
    require=lambda yes: (_ for _ in ()).throw(AssertionError()) if not yes else None
    info=audit.checkpoint(source,2486)
    report=audit.compare_checkpoints(info,info)
    require(report['bitwise_equal'] and report['padded31_plus8_scalar_values']==24704784)
    checks.append(dict(name='actual2486 full24704784 finite payload identity',passed=True,expected='accept'))
    no('wrong expected actual cycle',lambda:audit.checkpoint(source,2490))
    for key in ('parameters','identity','topology','metadata'):
        changed=copy.deepcopy(info);changed[key]={} if key=='parameters' else b'bad'
        no('complete checkpoint '+key,lambda changed=changed:audit.compare_checkpoints(info,changed))
    changed=copy.deepcopy(info);b=bytearray(changed['fixed']);b[120]^=1;changed['fixed']=bytes(b)
    ok('documented unused coarse header excluded',lambda:audit.compare_checkpoints(info,changed))
    changed=copy.deepcopy(info);b=bytearray(changed['fixed']);b[0]^=1;changed['fixed']=bytes(b)
    no('initialized header changed',lambda:audit.compare_checkpoints(info,changed))
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp);left=tmp/'left';right=tmp/'right'
        left.write_bytes(struct.pack('<78d',*([0.]*78)));right.write_bytes(left.read_bytes()[:-8]+struct.pack('<d',1.))
        a=dict(info,path=str(left),offset=0);b=dict(info,path=str(right),offset=0)
        no('synthetic late field/record payload corruption',lambda:audit.compare_checkpoints(a,b))
        right.write_bytes(left.read_bytes());no('synthetic short payload',lambda:audit.compare_checkpoints(a,b))
        left.write_bytes(struct.pack('<d',float('nan')));right.write_bytes(left.read_bytes())
        no('synthetic nonfinite payload',lambda:audit.compare_checkpoints(a,b))
        # Future tool schema is not assumed; only inventory/interval checks are exercised.
        trace=tmp/'trace.json';text=tmp/'unitrace.txt';text.write_text('synthetic summary only\n')
        event=dict(ph='X',cat='synthetic',name='syntheticKernel',ts=1.,dur=2.,args={'local_size':[256,1,1]})
        trace.write_text(json.dumps(dict(traceEvents=[event])))
        # Do not leave synthetic payload files in trace inventory.
        left.unlink();right.unlink()
        ok('synthetic Chrome complete interval and shape inventory',lambda:audit.trace_summary(tmp))
        for key,value in [('dur',float('nan')),('ts',float('inf')),('dur',-1)]:
            changed=dict(event);changed[key]=value;trace.write_text(json.dumps(dict(traceEvents=[changed])))
            no('synthetic invalid Chrome '+key+str(value),lambda:audit.trace_summary(tmp))
        trace.write_text(json.dumps(dict(traceEvents=[dict(ph='M',name='metadata')])));
        no('metadata-only trace is not collection',lambda:audit.trace_summary(tmp))
        trace.write_text('{');no('malformed trace retained, not repaired',lambda:audit.trace_summary(tmp))
    files=[Path(__file__),Path(audit.__file__),Path(__file__).with_name('native_n12_profile_contract.md')]
    for name in ('native_n12_profile_rank.py','native_two_node_timing_audit.py','collect_native_n12_profile.py'):
        path=Path(audit.__file__).with_name(name);ast.parse(path.read_text());files.append(path)
    for name in ('run-native-n12-profile.sh','native-n12-profile-two.pbs'):
        path=Path(audit.__file__).with_name(name);subprocess.run(['bash','-n',str(path)],check=True);files.append(path)
    write_json(args.output/'results.json',dict(schema='native-n12-profile-offline-controls-v1',passed=True,checks=checks,
        actual_fixture=dict(path=str(source),sha256=sha(source)),source_sha256={str(p):sha(p) for p in files},
        scope='Actual2486 full-state identity plus synthetic corruption/trace parser controls and syntax. No Linux process-map sampling, MPI child launch, actual unitrace schema, remote verifier or GPU collection exercised.'))
    print(json.dumps(dict(passed=True,checks=len(checks))))

if __name__=='__main__':main()
