#!/usr/bin/env python3
"""Compare native Cartesian evolution and v1/v2 restarts against the pinned base.

The binaries must use the same compiler, precision and MPI configuration. The
comparison removes only the versioned chart metadata and checks all remaining
binary metadata and state byte-for-byte. No numerical tolerance is used.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess


def normalized(path, version):
    data = path.read_bytes()
    offset = data.index(b'<par_end>\n') + len(b'<par_end>\n')
    body = data[offset:]
    nmb, = struct.unpack_from('=i', body)
    # Individually written native header records: two ints, RegionSize (9 Reals),
    # two RegionIndcs (19 ints each), time/dt, and cycle. Tests use double precision.
    header = struct.calcsize('=ii9d38i2di')
    cycle, = struct.unpack_from('=i', body, header-4)
    if version == 2:
        length, = struct.unpack_from('=Q', body, header)
        assert length == 0, 'Cartesian regression unexpectedly contains an atlas'
        start = header+8
        locations = []
        for m in range(nmb):
            loc = body[start+20*m:start+20*(m+1)]
            assert len(loc) == 20 and struct.unpack_from('=i', loc,16)[0] == 0
            locations.append(loc[:16])
        body = body[:header]+b''.join(locations)+body[start+20*nmb:]
    return body, {'cycle': cycle, 'blocks': nmb, 'bytes': len(body),
                  'sha256': hashlib.sha256(body).hexdigest()}


def run(binary, directory, args):
    directory.mkdir(parents=True, exist_ok=False)
    command = ['mpiexec', '-n', '2', str(binary)] + args
    with (directory/'run.log').open('w') as log:
        subprocess.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT,
                       env=dict(os.environ, OMP_NUM_THREADS='1'), check=True)
    paths = sorted((directory/'rst').glob('*.rst'))
    if not paths:
        raise RuntimeError('Run produced no restart: '+str(directory))
    return paths[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='+', choices=['hydro','z4c_amr'],
                        default=['hydro','z4c_amr'])
    args = parser.parse_args()
    baseline, candidate, output = (p.resolve() for p in
                                  (args.baseline,args.candidate,args.output))
    output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parents[2]
    evidence = {}
    cases = {
        'hydro': ('sod.athinput',['meshblock/nx1=32']),
        'z4c_amr': ('lwave_z4c_bc.athinput',
                    ['mesh/nx1=16','mesh/nx2=16','mesh/nx3=16',
                     'mesh_refinement/max_nmb_per_rank=64']),
    }
    for name in args.cases:
        input_name,options = cases[name]
        input_path = output/(name+'.athinput')
        text = (source/'tst/inputs'/input_name).read_text()
        if '<output1>' not in text:
            text += '\n<output1>\nfile_type = rst\ndt = 100\n'
        input_path.write_text(text)
        common = ['-i',str(input_path), 'time/nlim=2',
                  'output1/file_type=rst','output1/dt=100']+options
        b2 = run(baseline,output/(name+'_base2'),common)
        n2 = run(candidate,output/(name+'_native2'),common)
        b3 = run(baseline,output/(name+'_base3'),['-r',str(b2),'time/nlim=3'])
        # Intentionally override the format key: the saved header must control
        # decoding. Runtime numerical/output overrides remain allowed.
        old = run(candidate,output/(name+'_legacy_restart'),
                  ['-r',str(b2),'time/nlim=3','mesh/logical_location_version=2'])
        new = run(candidate,output/(name+'_native_restart'),
                  ['-r',str(n2),'time/nlim=3','mesh/logical_location_version=1'])
        for label,left,right,lv,rv in [('same_run',b2,n2,1,2),
                                      ('legacy_restart',b3,old,1,2),
                                      ('native_restart',b3,new,1,2)]:
            a,ma = normalized(left,lv)
            b,mb = normalized(right,rv)
            if a != b:
                first = next((i for i,(x,y) in enumerate(zip(a,b)) if x != y),None)
                raise AssertionError(f'{name}/{label}: state differs at byte {first}')
            evidence[name+'/'+label] = {'baseline':str(left),'candidate':str(right),
                                        'bitwise_equal':True, **ma}
        (output/'results.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence,indent=2))


if __name__ == '__main__':
    main()
