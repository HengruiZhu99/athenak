#!/usr/bin/env python3
"""Guard/collect the two bounded unchanged surface-SMR tests; never submit jobs."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import tarfile

ROOT = Path('/lus/flare/projects/CompactBinaryMerger/hzhu/scri-native-20261108-side')
WORK = ROOT/'validation/surface-smr-gpu-v1'
EXE = '3b50e96c0d642e30a7519385108921cca9e32a65cc64f4822e6a69386fb04336'
PROOF = 'a91999245554bea95d34828ed5807acab82337769c30fe283a283b436f001e63'
INPUT = '0a1ac2023d0abb6ad17931090a61c97a3505b97e7fd4f472f6d5a9342ebc2719'
FIXTURE = '2f4641583d28e2d875c4234b1e2c3386710c3f20fe84e664b8e66ad43345fad3'
BASELINE = 'f3a85db7528986ddec347b31675a8c5d588120a2380133aed73d5967f7aab05c'
SOURCE_MANIFEST = '4706aae696919253201da4cac9dd14bff52743ee8826dc93660a382f255ec9a5'
OBJECT_MANIFEST = '462da52c2e152e9c94818a534314aec72f2a827f30decdb2d23a3559e6954547'
WRAPPER = '/opt/aurora/26.26.0/support/tools/mpi_wrapper_utils/gpu_tile_compact.sh'
WRAPPER_SHA = '4469d67ffee9450a7e09b96e291f422360c3ac04c151cd7dde22e8e129979dbd'
BIND = '--cpu-bind=list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100'
ENV = ('OMP_NUM_THREADS','ZE_FLAT_DEVICE_HIERARCHY','ZE_ENABLE_PCI_ID_DEVICE_ORDER',
       'ONEAPI_DEVICE_SELECTOR','MPIR_CVAR_ENABLE_GPU','ZE_AFFINITY_MASK',
       'NEO_CACHE_DIR','SYCL_CACHE_DIR')


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())
def write(path, value): Path(path).write_text(json.dumps(value, indent=2)+'\n')
def case(nodes): return ROOT/f'runs/surface-smr-gpu-nodes{nodes}'


def terminal(text, job, nodes):
    found = re.search(r'^Job Id:\s*(\S+)', text, re.M)
    require(found and found[1].split('.')[0] == str(job).split('.')[0], 'wrong terminal job')
    for key, value in [('job_state','F'),('Exit_status','0'),('Resource_List.nodect',str(nodes)),
                       ('Job_Name','native-smr-two' if nodes == 2 else 'native-smr-eight')]:
        match = re.search(r'^\s*'+re.escape(key)+r'\s*=\s*(\S+)\s*$',text,re.M)
        require(match and match[1] == value, 'wrong terminal '+key)


def binding(path, nodes):
    owners = set()
    for line in Path(path).read_text().splitlines():
        match = re.fullmatch(r'(\S+) local_rank=(\d+) tile=(\d+\.\d+) implicit_scaling=0',line)
        require(match, 'invalid tile binding')
        host, rank, tile = match.groups(); rank = int(rank)
        require((host,rank) not in owners and tile == f'{rank//2}.{rank%2}', 'duplicate/wrong tile')
        owners.add((host,rank))
    hosts = {h for h,r in owners}
    require(len(hosts) == nodes and all({r for h,r in owners if h == host} == set(range(12))
                                      for host in hosts), 'incomplete rank binding')


def guard():
    require(sha(WORK/'build/provenance.json') == PROOF and sha(WORK/'build/surface_smr') == EXE,
            'unqualified helper build')
    require(sha(WORK/'package/surface_smr.cpp') == FIXTURE and
            sha(WORK/'package/surface_smr.athinput') == INPUT, 'changed fixture/input')
    verify = subprocess.check_output(['/usr/bin/python3.11',str(WORK/'package/surface_smr_gpu.py'),
        'verify','--output',str(WORK/'build')], text=True)
    require(verify == 'Verified unchanged surface SMR fixture, coherent candidate and frozen baseline\n',
            'helper verification output differs')
    build = ROOT/'build-sycl'; sources = read(build/'source-sha256.json')
    objects = read(build/'runtime-object-sha256.json')
    require(sha(build/'source-sha256.json') == SOURCE_MANIFEST and
            sha(build/'runtime-object-sha256.json') == OBJECT_MANIFEST, 'frozen manifests changed')
    for name,digest in sources.items(): require(sha(ROOT/'source'/name) == digest, 'baseline source changed')
    for name,digest in objects['object_sha256'].items(): require(sha(build/name) == digest, 'baseline object changed')
    require(sha(build/'src/athena') == BASELINE and sha(WRAPPER) == WRAPPER_SHA,
            'baseline/wrapper changed')
    return dict(schema='native-surface-smr-runtime-guard-v1',passed=True,
        helper_sha256=EXE,helper_provenance_sha256=PROOF,fixture_sha256=FIXTURE,input_sha256=INPUT,
        baseline_sha256=BASELINE,source_manifest_sha256=sha(build/'source-sha256.json'),
        object_manifest_sha256=sha(build/'runtime-object-sha256.json'),source_files=len(sources),
        objects=len(objects['object_sha256']),wrapper_sha256=WRAPPER_SHA,verify_output=verify)


def numerical_log(text):
    volume = re.findall(r'^nodal_volume comparisons=(\d+) internal_radial_starts=(\d+) analytic_jacobian=(\w+)$',text,re.M)
    values = re.findall(r'^outer_smr blocks=(\d+) surface_patches=(\d+) comparisons=(\d+) failures=(\d+) max_error_over_bound=(\S+) max_absolute=(\S+) collective_rebuild_rejection=(\w+)$',text,re.M)
    require(volume == [('22464','576','passed')] and len(values) == 1, 'missing/duplicate volume/operator record')
    blocks,faces,count,failed,ratio,absolute,rejected = values[0]
    require((blocks,faces,count,failed,rejected) == ('113','9','4228848','0','passed'),
            'operator coverage/failure/rejection gate failed')
    ratio,absolute = float(ratio),float(absolute)
    require(math.isfinite(ratio) and 0 <= ratio <= 1 and math.isfinite(absolute) and absolute >= 0,
            'nonfinite/out-of-bound operator error')
    require(text.splitlines().count('PASS native refined scri operators') == 1, 'missing/duplicate final PASS')
    return dict(blocks=113,surface_patches=9,comparisons=4228848,failed_values=0,
        nodal_volume_comparisons=22464,internal_radial_starts=576,analytic_jacobian='passed',
        collective_rebuild_rejection='passed',max_error_over_bound=ratio,max_absolute_error=absolute)


def audit(nodes):
    directory = case(nodes)
    require((directory/'exit-code.txt').read_text().strip() == '0', 'test did not exit normally')
    expected = ['mpiexec','-n',str(12*nodes),'-ppn','12',BIND,WRAPPER,
                str(WORK/'build/surface_smr'),str(WORK/'package/surface_smr.athinput')]
    require(shlex.split((directory/'command.txt').read_text()) == expected, 'actual command differs')
    require((directory/'cwd.txt').read_text().strip() == str(directory), 'wrong working directory')
    wanted = dict(zip(ENV,('1','COMPOSITE','1','level_zero:gpu','1',None,
                          str(directory/'cache/neo'),str(directory/'cache/sycl'))))
    require(read(directory/'environment.json') == wanted, 'launch environment differs')
    binding(directory/'binding.txt',nodes)
    before,after = [read(directory/f'guard-{when}.json') for when in ('before','after')]
    require(before == after and before['passed'] and before['helper_sha256'] == EXE and
            before['helper_provenance_sha256'] == PROOF and before['input_sha256'] == INPUT and
            before['source_files'] == 1972 and before['objects'] == 228, 'runtime guards differ')
    require(not list(directory.rglob('*.rst')), 'unexpected checkpoint output')
    return dict(schema='native-surface-smr-run-v1',passed=True,nodes=nodes,ranks=12*nodes,
        job_id=(directory/'job-id.txt').read_text().strip(),helper_sha256=EXE,input_sha256=INPUT,
        command=expected,metrics=numerical_log((directory/'run.log').read_text()),
        scope='unchanged manufactured refined surface operators; no evolution/performance claim')


def prior():
    directory = case(2); result = read(directory/'results.json')
    require(result == audit(2), 'two-node result changed')
    status = subprocess.check_output(['qstat','-xf',result['job_id']],text=True)
    terminal(status,result['job_id'],2)
    return dict(prior_job_id=result['job_id'],prior_result_sha256=sha(directory/'results.json'),
                terminal_status=status,helper_sha256=EXE,input_sha256=INPUT)


def collect(nodes, job):
    directory = case(nodes); result = audit(nodes)
    require(str(job) == result['job_id'].split('.')[0], 'collection job differs from actual run')
    status = subprocess.check_output(['qstat','-xf',str(job)],text=True); terminal(status,job,nodes)
    require((directory/'pipeline-complete.txt').is_file(), 'pipeline incomplete')
    if nodes == 8:
        previous = read(directory/'prior-two.json')
        terminal(previous['terminal_status'],previous['prior_job_id'],2)
        require(previous['prior_job_id'].split('.')[0] != str(job) and
                previous['helper_sha256'] == EXE and previous['input_sha256'] == INPUT,
                'eight-node run did not bind prior successful two-node case')
    require(read(directory/'results.json') == result, 'recorded science result differs')
    out = WORK/f'artifacts-{job}'; out.mkdir(exist_ok=False)
    (out/'terminal-qstat.txt').write_text(status)
    files = {str(p.relative_to(ROOT)):p for p in directory.rglob('*') if p.is_file()
             and 'cache' not in p.relative_to(directory).parts}
    for name in ('surface_smr_run_audit.py','run-surface-smr-validation.sh',
                 'surface-smr-two.pbs','surface-smr-eight.pbs','launchers.sha256',
                 'build/provenance.json','build-result.json','stage-verification.json'):
        p = WORK/name; require(p.is_file(),'missing proof '+name); files[str(p.relative_to(ROOT))] = p
    for p in (WORK/'package').iterdir():
        if p.is_file(): files[str(p.relative_to(ROOT))] = p
    files['terminal-qstat.txt'] = out/'terminal-qstat.txt'
    manifest = {name:dict(size=p.stat().st_size,sha256=sha(p)) for name,p in sorted(files.items())}
    write(out/'files.json',manifest)
    archive = out/'run-artifacts.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for name,p in sorted(files.items()): tar.add(p,arcname=name,recursive=False)
        tar.add(out/'files.json',arcname='files.json',recursive=False)
    package = dict(schema='native-surface-smr-collection-v1',job_id=str(job),nodes=nodes,
        ranks=12*nodes,job_state='F',exit_status=0,helper_sha256=EXE,input_sha256=INPUT,
        files=len(files),archive=str(archive),archive_bytes=archive.stat().st_size,
        total_uncompressed_bytes=sum(v['size'] for v in manifest.values()),sha256=sha(archive),
        results=result,independent_review_pending=True)
    write(out/'package.json',package); print(json.dumps(package,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('guard','environment','binding','prior','audit','collect'))
    parser.add_argument('--nodes',type=int,choices=(2,8)); parser.add_argument('--job',type=int)
    args = parser.parse_args()
    if args.command == 'guard': print(json.dumps(guard(),indent=2))
    elif args.command == 'environment': print(json.dumps({k:os.environ.get(k) for k in ENV},indent=2))
    elif args.command == 'prior': print(json.dumps(prior(),indent=2))
    else:
        require(args.nodes in (2,8),'nodes required')
        if args.command == 'binding': binding(case(args.nodes)/'binding.txt',args.nodes)
        elif args.command == 'audit': write(case(args.nodes)/'results.json',audit(args.nodes))
        else: require(args.job is not None,'job required'); collect(args.nodes,args.job)
