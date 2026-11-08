#!/usr/bin/env python3
"""Input-binding controls only. No endpoint evolution/diagnostic process is run."""
import argparse
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
from types import SimpleNamespace

import legacy_n12_endpoint_diagnostics as tool


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False);checks=[]
    def no(name,call):
        try:call()
        except (RuntimeError,ValueError,KeyError) as e:
            checks.append(dict(name=name,passed=True,expected='reject',message=str(e)));return
        raise AssertionError('accepted negative '+name)
    seed=Path('build-native/n12-matched-benchmark-seed-v1').resolve()
    traces=Path('build-native/aurora-stage-trace-two-v2/run-artifacts/runs/stage-validation-v2-nodes2/legacy-eight').resolve()
    source_hashes={str(seed/'seed.json'):tool.sha(seed/'seed.json')}
    with tempfile.TemporaryDirectory(prefix='endpoint-binding-') as tmp:
        root=Path(tmp);accepted=root/'accepted';accepted.mkdir()
        # Actual earlier trace bytes with explicitly synthetic endpoint metadata:
        # this tests hashing/coverage, not an actual2556 state or diagnostic gate.
        for region in ('bulk','scri'):
            for rank in range(24):
                source=traces/f'accepted8-{region}-rank{rank}.bin'
                target=accepted/f'accepted-final-{region}-rank{rank}.bin'
                os.link(source,target);source_hashes[str(source)]=tool.sha(source)
        os.link(seed/'restart.bin',accepted/'endpoint-restart.bin')
        time=0.
        for _ in range(2556):time+=tool.DT
        report=dict(schema='legacy-n12-original-control-benchmark-v1',passed=True,end_cycle=2556,time=time,dt=tool.DT,ranks=24)
        (accepted/'results.json').write_text(json.dumps(report))
        manifest=root/'inputs.json'
        call=lambda out:tool.input_manifest(SimpleNamespace(seed=seed,accepted=accepted,output=out))
        call(manifest);tool.verify_inputs(manifest)
        checks.append(dict(name='actualseed+48actualtrace byte bindings, synthetic2556 metadata',passed=True,expected='accept'))
        no('existing manifest protected',lambda:call(manifest))
        # Corrupt only a private copy; never write through a link to scientific evidence.
        path=accepted/'accepted-final-scri-rank23.bin';raw=path.read_bytes();path.unlink();path.write_bytes(raw)
        with path.open('r+b') as f:f.seek(-8,2);f.write(struct.pack('<d',123.))
        no('late row eighth chi record corruption',lambda:tool.verify_inputs(manifest))
        path.write_bytes(raw);tool.verify_inputs(manifest)
        checks.append(dict(name='exact restoration after private corruption',passed=True,expected='accept'))
        hidden=accepted/'hidden.bin';path.rename(hidden)
        no('missing48th trace',lambda:call(root/'missing.json'));hidden.rename(path)
        original=path.read_bytes()
        bad=bytearray(original);bad[24:28]=struct.pack('<I',5);path.write_bytes(bad)
        no('wrong six-key header',lambda:call(root/'header.json'));path.write_bytes(original)
        with path.open('r+b') as f:f.truncate(len(original)-8)
        no('truncated late record',lambda:call(root/'truncated.json'));path.write_bytes(original)
        changed=dict(report,end_cycle=2555);(accepted/'results.json').write_text(json.dumps(changed))
        no('wrong endpoint clock',lambda:call(root/'clock.json'))
        (accepted/'results.json').write_text(json.dumps(report))
    assert all(tool.sha(path)==digest for path,digest in source_hashes.items())
    result=dict(schema='legacy-n12-endpoint-binding-controls-v1',passed=True,checks=checks,
        original_scientific_inputs_unchanged=True,fixture_sha256=source_hashes,
        source_sha256={str(p.resolve()):tool.sha(p) for p in (Path(__file__),Path(tool.__file__))},
        scope='Actual qualified2486 seed and existing8-step traces exercise input hashing/header coverage; endpoint JSON is explicitly synthetic. No actual2556 endpoint, C++ loading, diagnostic RHS, GPU or performance admission.')
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,checks=len(checks))))

if __name__=='__main__':main()
