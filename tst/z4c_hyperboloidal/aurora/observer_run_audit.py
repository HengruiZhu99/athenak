#!/usr/bin/env python3
"""Operational coverage and restart admission; independent science audit follows."""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys

DT = .00046944203735617445
HEADER = struct.calcsize('<ii9d38i2di')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def time_at(n):
    t = 0.
    for _ in range(n):
        t += DT
    return t


def checkpoint(path, ranks, cycle):
    with path.open('rb') as f:
        size = 0
        while True:
            line = f.readline(262144)
            size += len(line)
            assert line and size < 8*1024*1024
            if line == b'<par_end>\n':
                break
        fixed = f.read(HEADER)
        assert struct.unpack_from('<ii', fixed) == (106, 4)
        assert struct.unpack_from('<ddi', fixed, HEADER-20) == (time_at(cycle), DT, cycle)
        n, = struct.unpack('<Q', f.read(8))
        assert n < 1024*1024
        f.read(n + 106*20)
        assert all(x == 1. for x in struct.unpack('<106f', f.read(106*4)))
        meta = struct.unpack('<8Q', f.read(64))
        assert meta[:2] == (0x4859504552525354, 1)
        assert meta[2] < 1024*1024
        f.read(meta[2])
        stride, = struct.unpack('<Q', f.read(8))
        assert meta[3] == (18**3+12**2)*39 and stride == meta[3]*8
        assert meta[4:] == (ranks, 0, 106//ranks, 0)
        assert path.stat().st_size == f.tell()+106*stride
    return dict(path=str(path.resolve()), sha256=sha(path), cycle=cycle, writer_ranks=ranks)


def run(case, ranks, start, end):
    assert (case/'exit-code.txt').read_text().strip() == '0'
    meta = json.loads((case/'native-step-observer.json').read_text())
    assert meta['schema'] == 'native-step-observer-v1' and meta['passed']
    assert meta['backend'] == 'SYCL' and meta['ranks'] == ranks
    assert (meta['initial_cycle'], meta['final_cycle']) == (start, end)
    assert (meta['initial_time'], meta['final_time'], meta['dt']) == (time_at(start), time_at(end), DT)
    assert meta['stage_rows'] == (end-start)*4
    with (case/'native-stages.csv').open() as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ['step','stage','rhs_time','state_time','dt','compatibility','clock','CFL']
        rows = list(reader)
    assert len(rows) == (end-start)*4
    for q, row in enumerate(rows):
        step, stage = start+q//4+1, q%4+1
        assert (int(row['step']), int(row['stage'])) == (step, stage)
        assert float(row['dt']) == DT
        assert float(row['rhs_time']) == time_at(step-1)+DT*[0,.5,.5,1][stage-1]
        assert float(row['state_time']) == time_at(step-1)+DT*[.5,.5,1,1][stage-1]
        assert all(math.isfinite(float(row[k])) and float(row[k]) >= 0 for k in ['compatibility','clock','CFL'])
        assert float(row['compatibility']) <= 1e-7
    log = (case/'run.log').read_text()
    assert 'Terminating on cycle limit' in log
    logged = re.findall(r'^elapsed=\S+ cycle=(\d+) time=(\S+) dt=(\S+)$', log, re.M)
    assert [int(c) for c,_,_ in logged] == list(range(start,end+1))
    assert all(float(t)==time_at(int(c)) and float(dt)==DT for c,t,dt in logged)
    files = sorted((case/'rst').glob('*.rst'))
    expected = [0,1] if start==0 else [2]
    assert [p.name for p in files] == [f'native_n12_5M.{n:05d}.rst' for n in expected]
    records = [checkpoint(p,ranks,0 if start==0 and i==0 else end) for i,p in enumerate(files)]
    report = dict(schema='native-step-observer-operational-v1', operational_passed=True,
                  scientific_comparison='pending', ranks=ranks, start_cycle=start, end_cycle=end,
                  checkpoints=records, stage_rows=len(rows), job_id=__import__('os').environ.get('PBS_JOBID'),
                  command_sha256=sha(case/'command.txt'), stages_sha256=sha(case/'native-stages.csv'))
    (case/'operational.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


def main():
    mode, directory, *args = sys.argv[1:]
    case = Path(directory)
    if mode == 'binding':
        hosts = collections.defaultdict(dict)
        for line in (case/'binding.txt').read_text().splitlines():
            match = re.fullmatch(r'(\S+) local_rank=(\d+) tile=(\d+\.\d+) implicit_scaling=0',line)
            assert match
            host, rank, tile = match.groups();rank=int(rank)
            assert rank not in hosts[host] and tile==f'{rank//2}.{rank%2}'
            hosts[host][rank]=tile
        assert len(hosts)==int(args[0]) and all(set(x)==set(range(12)) for x in hosts.values())
    elif mode == 'run':
        run(case, *map(int,args))
    elif mode == 'check-prefix':
        report=json.loads((case/'operational.json').read_text())
        assert report['operational_passed'] and (report['ranks'],report['start_cycle'],report['end_cycle'])==(24,0,2)
        current=checkpoint(case/'rst/native_n12_5M.00001.rst',24,2)
        assert current==report['checkpoints'][-1]
        assert sha(case/'native-stages.csv')==report['stages_sha256']
    else:
        raise ValueError(mode)


if __name__ == '__main__':
    main()
