#!/usr/bin/env python3
"""Frozen curved exporter -> native per-block samples; never runs a solve.

Run the dedicated pgen in static binary_prepare mode first. Output is new,
atomic per block, with source/ABI/archive/point identities in manifest.json.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np

ARCHIVE_SHA = '5f12381a4a06d5712f42ed58737dd8020a5a263c85de8a350e9b8bbc7c720f3b'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial-data-repo',type=Path,required=True)
    parser.add_argument('--archive',type=Path,
                        help='new solved archive; requires its explicit SHA256')
    parser.add_argument('--archive-sha256',
                        help='required expected digest when --archive is supplied')
    parser.add_argument('--points',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--chunk',type=int,default=1024)
    args=parser.parse_args()
    if args.chunk<=0: raise ValueError('positive chunk size required')
    root=args.initial_data_repo.resolve()
    if (args.archive is None)!=(args.archive_sha256 is None):
        parser.error('--archive and --archive-sha256 must be supplied together')
    archive=(args.archive.resolve() if args.archive is not None else
        root/'validation/hyperboloidal/evidence/akv/build-hyperboloidal/akv/calibration-01/evaluation-01/solve/solution.npz')
    archive_sha=args.archive_sha256 or ARCHIVE_SHA
    if sha(archive)!=archive_sha: raise ValueError('selected archive hash mismatch')
    snapshot=archive.parent/'source_snapshot'
    report=json.loads((archive.parent/'report.json').read_text())
    if report.get('status')!='converged':
        raise ValueError('initial-data solve did not converge')
    for rel,want in report['implementation_sha256'].items():
        if sha(snapshot/rel)!=want: raise ValueError('frozen source mismatch: '+rel)
    native=None
    for receipt in (root/'build-hyperboloidal/hispid-bridge').glob('*.json'):
        if json.loads(receipt.read_text())['sources']==report['bridge_source_sha256']:
            candidate=receipt.with_name('libhispid-hyperboloidal-'+receipt.stem+'.so')
            if candidate.is_file() and sha(candidate)==report['native_bridge_sha256']:
                native=candidate;break
    if native is None: raise ValueError('frozen native seed ABI not available')
    points=sorted(args.points.glob('block-*.points'))
    if not points: raise ValueError('no native sampling requests')
    context=dict(format='HYPERBOL_BINARY_V1',archive_sha256=archive_sha,
        solve_context_sha256=sha(archive.parent/'report.json'),
        source_sha256=report['implementation_sha256'],native_bridge_sha256=sha(native),
        converter_sha256=sha(__file__),point_sha256={p.name:sha(p) for p in points},
        gauge='precollapsed-core',B=0,J=0,Theta=0,D='Lambda',
        internal_mass='1/3',internal_component_targets='1/6',reported_component_targets='1/2',
        rescale_chart=False,renormalize_measured_masses=False)
    identity=hashlib.sha256(json.dumps(context,sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=dict(identity=identity,context=context,status='INCOMPLETE',blocks={})
    path=args.output/'manifest.json'
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    sys.path.insert(0,str(snapshot))
    import hyperboloidal_hispid as hispid
    hispid.build_bridge=lambda **kwargs:(native,report['bridge_source_sha256'])
    if 'cpu_backend' in report:
        import hyperboloidal_cpu as cpu
        cpu.bind(report['cpu_backend'])
    from hyperboloidal_binary_sampler import BinaryCTTSampler
    from hyperboloidal_evolution_export import CurvedEvolutionExporter
    print('Constructing frozen sampler; identity',identity,flush=True)
    sampler=BinaryCTTSampler.from_archive(archive)
    exporter=CurvedEvolutionExporter(sampler,reference=sampler.reference,gauge='precollapsed-core')
    for name,module in sys.modules.items():
        if name.startswith('hyperboloidal_') and getattr(module,'__file__',None):
            if Path(module.__file__).resolve().parent!=snapshot:
                raise ValueError('geometry module escaped frozen snapshot: '+name)
    manifest['centers']=[(np.asarray(h.center)*sampler.free_data.k).tolist() for h in sampler.free_data.holes]
    manifest['flat_omega_radius']=sampler.reference.r0
    for p in points:
        with p.open() as f: header=f.readline().split()
        if len(header)!=4 or header[0]!='HYPERBOL_POINTS_V1': raise ValueError('point header')
        gid,nb,ns=map(int,header[1:])
        data=np.loadtxt(p,skiprows=1,ndmin=2)
        if data.shape!=(nb+ns,34) or not np.isfinite(data).all(): raise ValueError('point inventory')
        fields=np.empty((len(data),31))
        for start in range(0,len(data),args.chunk):
            stop=min(len(data),start+args.chunk)
            fields[start:stop]=exporter.evolution_fields(data[start:stop,:3],native_fields=data[start:stop,3:])
        target=args.output/f'block-{gid}.hbi';temporary=target.with_suffix('.partial')
        with temporary.open('w') as f:
            f.write(f'HYPERBOL_BINARY_V1 {identity} {gid} {nb} {ns}\n')
            np.savetxt(f,np.column_stack((data[:,:3],fields)),fmt='%.17g')
        temporary.rename(target)
        manifest['blocks'][target.name]=dict(sha256=sha(target),bulk=nb,scri=ns)
        path.write_text(json.dumps(manifest,indent=2)+'\n')
        print(target.name,'exported',len(data),'points',flush=True)
    if any(sha(p)!=context['point_sha256'][p.name] for p in points) or sha(archive)!=archive_sha:
        raise ValueError('input changed during export')
    manifest['status']='COMPLETE'
    path.write_text(json.dumps(manifest,indent=2)+'\n')
    print('Use problem/binary_identity='+identity,flush=True)
if __name__=='__main__': main()
