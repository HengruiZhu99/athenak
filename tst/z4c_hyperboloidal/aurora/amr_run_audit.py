#!/usr/bin/env python3
"""Operational AMR coverage checks; independent full-field science follows."""
import collections
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys

DT = .0002
HEADER = struct.calcsize('<ii9d38i2di')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def time_at(cycle):
    value = 0.
    for _ in range(cycle):
        value += DT
    return value


def checkpoint(path, ranks, cycle, *, capacity=16):
    with path.open('rb') as stream:
        text = bytearray()
        while True:
            line = stream.readline(262144)
            text.extend(line)
            assert line and len(text) < 8*1024*1024
            if line == b'<par_end>\n':
                break
        fixed = stream.read(HEADER)
        blocks, root = struct.unpack_from('<ii', fixed)
        assert root == 4 and blocks == (106 if cycle < 2 else 113)
        assert struct.unpack_from('<ddi', fixed, HEADER-20) == (time_at(cycle), DT, cycle)
        size, = struct.unpack('<Q', stream.read(8))
        assert size < 1024*1024
        atlas = stream.read(size)
        topology = stream.read(blocks*20)
        assert all(c == 1. for c in struct.unpack(f'<{blocks}f', stream.read(blocks*4)))
        meta = struct.unpack('<8Q', stream.read(64))
        assert meta[:2] == (0x4859504552525354, 1) and meta[2] < 1024*1024
        identity = stream.read(meta[2])
        stride, = struct.unpack('<Q', stream.read(8))
        assert meta[3] == (18**3+12**2)*39 and stride == meta[3]*8
        assert meta[4:] == (ranks, 0, blocks//ranks, 0)
        assert path.stat().st_size == stream.tell()+blocks*stride
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    parameters, block = {}, None
    for line in text.decode().splitlines():
        line = line.split('#', 1)[0].strip()
        if line.startswith('<'):
            block = line[1:line.index('>')]
        elif '=' in line:
            key, value = line.split('=', 1)
            parameters[block, key.strip()] = value.strip()
    for key, value in {('time','dt'):DT, ('mesh_refinement','max_nmb_per_rank'):capacity,
                       ('output1','dcycle'):1, ('z4c_hyperboloidal','ko'):.1,
                       ('z4c_hyperboloidal','chi_floor'):1e-10}.items():
        assert float(parameters[key]) == value
    assert parameters['time','integrator'] == 'rk4_classical'
    assert parameters['mesh_refinement','refinement'] == 'adaptive'
    assert parameters['output1','single_file_per_rank'] == 'false'
    return dict(path=str(path.resolve()), sha256=sha(path), cycle=cycle, time=time_at(cycle),
                dt=DT, writer_ranks=ranks, blocks=blocks, payload_sha256=digest.hexdigest(),
                topology_sha256=hashlib.sha256(topology).hexdigest(),
                atlas_sha256=hashlib.sha256(atlas).hexdigest(),
                identity_sha256=hashlib.sha256(identity).hexdigest())


def records(log, prefix):
    return [json.loads(line[len(prefix):]) for line in log.splitlines() if line.startswith(prefix)]


def snapshots(case, ranks, stems):
    names = set()
    for stem in stems:
        for rank in range(ranks):
            binary = case/f'{stem}-rank{rank}.bin'
            meta = case/f'{stem}-rank{rank}.txt'
            assert binary.is_file() and binary.stat().st_size % (9*4+39*8) == 0
            assert meta.is_file()
            names.update((binary.name, meta.name))
    assert {p.name for p in case.glob('*-rank*.*')} == names
    return len(names)


def run(case, ranks, name):
    assert (case/'exit-code.txt').read_text().strip() == '0'
    log = (case/'run.log').read_text()
    report = dict(schema='native-amr-operational-v1', operational_passed=True,
                  scientific_comparison='pending', ranks=ranks, case=name, fixed_dt=DT,
                  job_id=os.environ.get('PBS_JOBID'), command_sha256=sha(case/'command.txt'))
    if name == 'lifecycle':
        assert 'PASS native field AMR lifecycle' in log
        rows = records(log, 'AMR_REPORT ')
        diagnostics = records(log, 'DIAGNOSTICS_REPORT ')
        assert [r['phase'] for r in rows] == [0,1,2]
        assert [r['blocks'] for r in rows] == [106,127,106]
        assert [r['bulk_points'] for r in rows] == [229392,277236,229392]
        assert [r['scri_points'] for r in rows] == [864,1296,864]
        assert all(math.isfinite(r['compatibility']) and 0 <= r['compatibility'] <= 1e-7 for r in rows)
        assert [r['phase'] for r in diagnostics] == [0,1,2]
        assert all(r['fresh_exact'] and r['repeat_cached'] and r['readonly'] for r in diagnostics)
        report.update(snapshot_files=snapshots(case,ranks,['phase0','phase1','phase2']),
                      phase_reports=rows, diagnostics_reports=diagnostics)
    elif name == 'evolution':
        assert 'PASS four native Driver cycles with two AMR events' in log
        cycles = records(log, 'AMR_CYCLE ')
        stages = records(log, 'AMR_STAGE ')
        assert [r['cycle'] for r in cycles] == list(range(5))
        assert [r['blocks'] for r in cycles] == [106,127,127,106,106]
        assert all(r['time']==time_at(r['cycle']) and r['dt']==DT for r in cycles)
        assert len(stages)==16
        for index,row in enumerate(stages):
            cycle,stage=index//4+1,index%4+1
            assert (row['cycle'],row['stage'])==(cycle,stage)
            assert row['rhs_time']==time_at(cycle-1)+DT*[0,.5,.5,1][stage-1]
            assert row['state_time']==time_at(cycle-1)+DT*[.5,.5,1,1][stage-1]
            assert all(math.isfinite(row[k]) and row[k]>=0 for k in ('compatibility','clock','cfl'))
            assert row['compatibility']<=1e-7
        report.update(snapshot_files=snapshots(case,ranks,[f'cycle{i}' for i in range(5)]),
                      cycle_reports=cycles,stage_rows=16)
    else:
        expected = {'prefix':[0,1,2,2], 'continuous':[0,1,2,3,4,4], 'resumed':[3,4,4]}[name]
        start = 2 if name=='resumed' else 0
        assert 'Terminating on cycle limit' in log
        logged = re.findall(r'^elapsed=\S+ cycle=(\d+) time=(\S+) dt=(\S+)$',log,re.M)
        assert [int(c) for c,_,_ in logged] == list(range(start,expected[-1]+1))
        assert all(float(t)==time_at(int(c)) and float(dt)==DT for c,t,dt in logged)
        files=sorted((case/'rst').glob('*.rst'))
        offset=4 if name=='resumed' else 0
        assert [p.name for p in files]==[f'amr_restart.{i:05d}.rst' for i in range(offset,offset+len(expected))]
        checked=[checkpoint(path,ranks,cycle) for path,cycle in zip(files,expected)]
        for key in ('cycle','time','dt','blocks','payload_sha256','topology_sha256','atlas_sha256','identity_sha256'):
            assert checked[-1][key]==checked[-2][key], ('duplicate final output differs',key)
        report['checkpoints']=checked
    (case/'operational.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


def main():
    mode,directory,*args=sys.argv[1:]
    case=Path(directory)
    if mode=='run':
        run(case,int(args[0]),args[1])
    elif mode=='binding':
        hosts=collections.defaultdict(dict)
        for line in (case/'binding.txt').read_text().splitlines():
            match=re.fullmatch(r'(\S+) local_rank=(\d+) tile=(\d+\.\d+) implicit_scaling=0',line)
            assert match
            host,rank,tile=match.groups();rank=int(rank)
            assert rank not in hosts[host] and tile==f'{rank//2}.{rank%2}'
            hosts[host][rank]=tile
        assert len(hosts)==int(args[0]) and all(set(v)==set(range(12)) for v in hosts.values())
    elif mode=='check-prefix':
        report=json.loads((case/'operational.json').read_text())
        assert report['operational_passed'] and report['case']=='prefix' and report['ranks']==24
        assert (case.parent/'pipeline-complete.txt').is_file()
        status=subprocess.check_output(['qstat','-xf',report['job_id']],text=True)
        assert 'job_state = F' in status and 'Exit_status = 0' in status
        assert checkpoint(case/'rst/amr_restart.00003.rst',24,2)==report['checkpoints'][-1]
        assert sha(case/'command.txt')==report['command_sha256']
        print(status)
        print(json.dumps(dict(prefix_operational_passed=True,job_id=report['job_id'],
              checkpoint_sha256=report['checkpoints'][-1]['sha256'])))
    else:
        raise ValueError(mode)


if __name__=='__main__':
    main()
