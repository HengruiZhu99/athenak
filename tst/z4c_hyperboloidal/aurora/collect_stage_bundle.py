#!/usr/bin/env python3
"""Collect one completed private stage-validation allocation without changing data."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path('/lus/flare/projects/CompactBinaryMerger/hzhu/scri-native-20261108-side')
MAP_EXE = 'a14333e4829718a7903efcabaf7ab523713cdf40ea0f0f5d7a772ea95004a8a7'
ORIGINAL_EXE = '494ec8e38fb34806eb3981fb19bdb5e3bc75f46615b05a21d8314a0db6583753'
ORIGINAL_INPUT = '3d0ce81d74e7b470272fc938802d505e9b53ec4ab1e3a011c6f3fad96bb74ac6'
ORIGINAL_CONFIG = '9c786cc8c3312ddf93389a4b3f1f6e32921811a61f4d131ee53990fb80a1d363'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main(args):
    qstat = subprocess.run(['qstat', '-xf', str(args.job)], text=True, capture_output=True, check=True).stdout
    assert 'job_state = F' in qstat and 'Exit_status = 0' in qstat, qstat
    case = ROOT / ('runs/stage-validation-v2-nodes'+str(args.nodes))
    one = ROOT / ('runs/original-gpu-one-nodes'+str(args.nodes))
    pipeline = json.loads((case/'pipeline.json').read_text())
    assert pipeline['observer_bundle_completed'] and pipeline['native_clean_poison_bytewise_passed']
    assert pipeline['ranks'] == 12*args.nodes and pipeline['runtime_sources_unchanged']
    folder = ROOT / ('validation/stage-artifacts-'+str(args.job))
    folder.mkdir(exist_ok=False)
    (folder/'job-final.txt').write_text(qstat)
    geometry = case/'geometry-map'
    assert (geometry/'exit-code.txt').read_text().strip() == '0'
    assert 'PASS N12 mapping leaves=106 bulk=229392 scri=864 coordinate_mismatches=0' in (geometry/'run.log').read_text()
    map_build = ROOT/'validation/geometry-map-v2/build/provenance.json'
    p = json.loads(map_build.read_text())
    assert p['status'] == 'built' and p['executable_sha256'] == MAP_EXE and sha(Path(p['executable'])) == MAP_EXE
    execution = dict(schema='native-n12-map-execution-v1', exit_status=0, ranks=1,
                     executable_sha256=MAP_EXE, job_id=str(args.job),
                     build_provenance_sha256=sha(map_build), post_teardown_pass_marker=True,
                     artifacts={name:sha(geometry/name) for name in ('mapping.bin', 'topology.bin',
                         'atlas.txt', 'initial-fields.bin', 'initial-boundary.bin', 'run.log')})
    assert not (geometry/'execution.json').exists()
    (geometry/'execution.json').write_text(json.dumps(execution, indent=2)+'\n')
    old = ROOT/'validation/original-eight'
    assert sha(old/'evolve-static') == ORIGINAL_EXE and sha(old/'run-eight.in') == ORIGINAL_INPUT
    source = (old/'run-eight.in').read_bytes()
    actual = (one/'output/run.in').read_bytes()
    assert source.count(b'nlim=8\n') == 1 and actual == source.replace(b'nlim=8\n', b'nlim=1\n')
    assert sha(one/'output/configuration.cfg') == ORIGINAL_CONFIG
    assert (one/'exit-code.txt').read_text().strip() == '0'
    provenance = dict(schema='original-one-step-staging-v1', binary_sha256=ORIGINAL_EXE,
                      binary_origin=str(old/'evolve-static'), configuration_sha256=ORIGINAL_CONFIG,
                      input_sha256=hashlib.sha256(actual).hexdigest(), source_input_sha256=ORIGINAL_INPUT,
                      only_input_change='nlim=8 -> nlim=1', fresh_start=True, ranks=12*args.nodes,
                      custom_library_sha256=sha(ROOT/'reference/libcurrent_spherical.so'),
                      immutable_checks_before_and_after=True, job_id=str(args.job),
                      predecessor_provenance_sha256=sha(old/'copy-provenance.json'))
    assert not (one/'staging-provenance.json').exists()
    (one/'staging-provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    files = []
    for directory in (case, one):
        files.extend(p for p in directory.rglob('*') if p.is_file() and
                     'cache' not in p.relative_to(directory).parts)
    for name in ('stage-trace', 'stage-trace-v2', 'legacy-stage-trace', 'geometry-map-v2'):
        directory = ROOT/'validation'/name
        files.extend(p for p in directory.rglob('*') if p.is_file() and
                     p.suffix != '.o' and p.name not in ('native_stage_trace_gpu',
                         'legacy_stage_trace_gpu', 'n12_checkpoint_map'))
    label = 'two' if args.nodes == 2 else 'eight'
    for relative in (f'stage-validation-v2-{label}.pbs.log',
                     f'stage-validation-v2-{label}-submission.json',
                     'build-sycl/runtime-object-sha256.json', 'build-sycl/source-sha256.json',
                     'source/source-revision.json', 'validation/original-eight/copy-provenance.json',
                     'validation/original-eight/immutable.sha256'):
        files.append(ROOT/relative)
    files.append(folder/'job-final.txt')
    files = sorted(set(files))
    manifest = {str(p.relative_to(ROOT)):dict(sha256=sha(p), size=p.stat().st_size) for p in files}
    (folder/'files.json').write_text(json.dumps(manifest, indent=2)+'\n')
    archive = folder/'run-artifacts.tar.gz'
    with tarfile.open(archive, 'w:gz', compresslevel=1) as tar:
        for p in files:
            tar.add(p, arcname=str(p.relative_to(ROOT)), recursive=False)
        tar.add(folder/'files.json', arcname='files.json', recursive=False)
    result = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  job_id=str(args.job), nodes=args.nodes, ranks=12*args.nodes,
                  job_state='F', exit_status=0, archive=str(archive), sha256=sha(archive),
                  archive_bytes=archive.stat().st_size, files=len(files),
                  total_uncompressed_bytes=sum(v['size'] for v in manifest.values()),
                  scientific_comparison_pending=True)
    (folder/'package.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=int, required=True)
    parser.add_argument('--nodes', type=int, choices=(2,8), required=True)
    main(parser.parse_args())
