#!/usr/bin/env python3
"""Prepare immutable sources/launcher; build or freeze explicitly. Never submits.

Run build only inside the user's Aurora home task root after reviewing its
storage and current jobs. The wrapper loads the established oneAPI environment.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile
import control
import native_transition_campaign as campaign
import storage
import worker

SOURCE_DIR='source-boosted-refined-core-v1'
BUILD_DIR='build-sycl-mpi-boosted-refined-core-v1'
LAUNCHER=Path(campaign.SCIENCE).parent
FILES=['control.py','worker.py','storage.py','run_debug.pbs','native_transition_campaign.py',
       'native_transition_prepare.py','build_native_transition.sh']


def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo),*args])


def identity(path):
    return dict(bytes=path.stat().st_size,sha256=control.digest(path))


def stage(repo,destination,revision):
    repo=repo.resolve();destination=destination.resolve()
    if destination.exists():raise ValueError('Staging destination must be new')
    revision=git(repo,'rev-parse',revision).decode().strip()
    kokkos=git(repo,'ls-tree',campaign.SOURCE_COMMIT,'kokkos').decode().split()[2]
    destination.mkdir(parents=True)
    for repository,commit,prefix,paths in [
        (repo,campaign.SOURCE_COMMIT,SOURCE_DIR+'/', ['CMakeLists.txt','config.hpp.in','LICENSE','src',
            'tst/hyperbolic_shift_3d','tst/elliptic_shift_1d/spherical_api.cpp']),
        (repo/'kokkos',kokkos,SOURCE_DIR+'/kokkos/',[])]:
        archive=git(repository,'archive','--format=tar','--prefix='+prefix,commit,*paths)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            # Exact repository archives only; reject unsafe paths/links before extraction.
            for entry in tar.getmembers():
                path=Path(entry.name)
                if path.is_absolute() or '..' in path.parts or entry.issym() or entry.islnk():
                    raise ValueError('Unsafe archive member: '+entry.name)
            tar.extractall(destination)
    manifest={str(p.relative_to(destination)):control.digest(p)
              for p in sorted((destination/SOURCE_DIR).rglob('*')) if p.is_file()}
    control.atomic(destination/campaign.MANIFEST,manifest)
    launcher=destination/LAUNCHER;launcher.mkdir(parents=True)
    for name in FILES:
        data=git(repo,'show',revision+':tst/hyperbolic_shift_3d/aurora/boosted_refined_core/'+name)
        (launcher/name).write_bytes(data)
    provenance=dict(source_commit=campaign.SOURCE_COMMIT,kokkos_commit=kokkos,launcher_commit=revision,
        source_manifest=campaign.MANIFEST,source_manifest_sha256=control.digest(destination/campaign.MANIFEST),
        files={name:identity(launcher/name) for name in FILES})
    control.atomic(launcher/'package-provenance.json',provenance)
    control.verify_source(destination,campaign.MANIFEST)
    return dict(destination=str(destination),source_files=len(manifest),source_bytes=sum((destination/p).stat().st_size for p in manifest),**provenance)


def verify_package(root):
    p=json.loads((root/LAUNCHER/'package-provenance.json').read_text())
    if p['source_commit']!=campaign.SOURCE_COMMIT or set(p['files'])!=set(FILES):
        raise ValueError('Incorrect packaged source or launcher inventory')
    if control.verify_source(root,campaign.MANIFEST)!=p['source_manifest_sha256']:
        raise ValueError('Packaged scientific source changed')
    for name,expected in p['files'].items():
        if identity(root/LAUNCHER/name)!=expected:raise ValueError('Packaged launcher changed: '+name)
    return p


def build(root):
    root=root.resolve()
    if not root.is_relative_to(Path.home().resolve()):raise ValueError('Build must stay in the Aurora home task directory')
    verify_package(root)
    if (root/BUILD_DIR).exists() or (root/'build-logs/boosted-refined-core-v1').exists():
        raise ValueError('Preserve prior build/logs; inspect them before an explicit retry')
    policy=dict(storage=dict(limit_bytes=10_000_000_000,planned_new_bytes=400_000_000,
                            atomic_checkpoint_reserve_bytes=100_000_000),commands=[dict(kind='check')])
    before=storage.require_capacity(root,policy,prospective=True)
    logs=root/'build-logs/boosted-refined-core-v1';logs.mkdir(parents=True)
    for name,args in [('compiler',['icpx','--version']),('mpi-wrapper',['mpicxx','-show'])]:
        result=subprocess.run(args,capture_output=True,text=True,check=True)
        (logs/(name+'.txt')).write_text(result.stdout+result.stderr)
    configure=['cmake','-S',SOURCE_DIR,'-B',BUILD_DIR,'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CXX_COMPILER=icpx',
        '-DCMAKE_CXX_FLAGS=-fp-model=precise -fno-sycl-id-queries-fit-in-int','-DCMAKE_CXX_FLAGS_RELEASE=-O2',
        '-DAthena_ENABLE_MPI=ON','-DAthena_SINGLE_PRECISION=OFF','-DAthena_ENABLE_HYPERBOLIC_SHIFT_3D_TESTS=ON',
        '-DKokkos_ENABLE_SYCL=ON','-DKokkos_ENABLE_SERIAL=ON','-DKokkos_ARCH_INTEL_PVC=ON']
    commands=[configure,['make','-C',BUILD_DIR,'-j64',*sorted({Path(campaign.binary(v)).name for v in campaign.VARIANTS}),
       *sorted({'hyperbolic_shift_3d_checkpoint_constraints_native_'+v['suffix'] for v in campaign.VARIANTS.values()})]]
    result=dict(storage_before=before,commands=[])
    for label,args in zip(['configure','build'],commands):
        r=worker.run_bounded(args,root,logs/(label+'.log'),3600,policy)
        result['commands'].append(dict(command=args,**r));control.atomic(logs/'result.json',result)
        storage.require_capacity(root,policy)
        if r['returncode']!=0 or r['timed_out'] or r['storage_error']:
            raise RuntimeError('Build stopped; preserved logs and partial build need diagnosis')
    verify_package(root)
    result['storage_after']=storage.footprint(root);control.atomic(logs/'result.json',result)
    return freeze(root)


def freeze(root):
    root=root.resolve();provenance=verify_package(root)
    if (root/campaign.SCIENCE).exists() or (root/LAUNCHER/'provenance.json').exists():
        raise ValueError('Science is already frozen; do not replace its identities')
    result=json.loads((root/'build-logs/boosted-refined-core-v1/result.json').read_text())
    if len(result['commands'])!=2 or any(c['returncode'] or c['timed_out'] or c['storage_error'] for c in result['commands']):
        raise ValueError('No successful bounded build record')
    cache={}
    for line in (root/BUILD_DIR/'CMakeCache.txt').read_text().splitlines():
        if '=' in line and ':' in line.split('=',1)[0]:
            key,value=line.split('=',1);cache[key.split(':')[0]]=value
    expected=dict(Athena_ENABLE_MPI='ON',Athena_SINGLE_PRECISION='OFF',Athena_ENABLE_HYPERBOLIC_SHIFT_3D_TESTS='ON',
        Kokkos_ENABLE_SYCL='ON',Kokkos_ENABLE_SERIAL='ON',Kokkos_ARCH_INTEL_PVC='ON',CMAKE_BUILD_TYPE='Release',
        CMAKE_CXX_FLAGS='-fp-model=precise -fno-sycl-id-queries-fit-in-int',CMAKE_CXX_FLAGS_RELEASE='-O2')
    if any(cache.get(k)!=v for k,v in expected.items()) or Path(cache.get('CMAKE_CXX_COMPILER','')).name!='icpx':
        raise ValueError('Build is not the intended SYCL/MPI/PVC configuration')
    binaries={};diagnostics={};flags={}
    for case,v in campaign.VARIANTS.items():
        for kind,records in [('mpi',binaries),('checkpoint_constraints',diagnostics)]:
            target='hyperbolic_shift_3d_'+kind+'_native_'+v['suffix'];path=BUILD_DIR+'/tst/hyperbolic_shift_3d/'+target
            f=root/BUILD_DIR/'tst/hyperbolic_shift_3d/CMakeFiles'/(target+'.dir')/'flags.make'
            text=f.read_text();defines={m[0]:m[1] for m in re.findall(r'-D(HYPERBOLIC_[A-Z0-9_]+)=([^\s]+)',text)}
            expected_defs=dict(HYPERBOLIC_REFINED_CORE='1',HYPERBOLIC_BOOST_FREE_INITIAL_GAUGE='1',HYPERBOLIC_BOOSTED_PUNCTURE='1',HYPERBOLIC_SIX_SHELLS='1',HYPERBOLIC_BROAD_SMOOTH_A='1',HYPERBOLIC_NATIVE_WORMHOLE='1',HYPERBOLIC_DIRECT_CHI='1',HYPERBOLIC_DIRECT_LAPSE='1')
            if any(defines.get(k)!=value for k,value in expected_defs.items()):raise ValueError('Missing native/direct-variable definitions')
            for key,enabled in [('HYPERBOLIC_ZERO_SHIFT_RESTORATION',False),('HYPERBOLIC_WIDE_RESTORING_COLLAR',False),('HYPERBOLIC_SHARED_RESTORATION',v['shared']),('HYPERBOLIC_CONSISTENT_KO',v['ko10']),('HYPERBOLIC_SCRI_FIRST6',v['scri6'])]:
                if (defines.get(key,'0')=='1')!=enabled:raise ValueError('Compiled variant does not match '+case)
            records[path]=control.digest(root/path);flags[str(f.relative_to(root))]=identity(f)
    frozen=dict(schema=1,source_commit=campaign.SOURCE_COMMIT,source_manifest=campaign.MANIFEST,
        source_manifest_sha256=provenance['source_manifest_sha256'],binaries=binaries,diagnostic_binaries=diagnostics,
        build_cache_sha256=control.digest(root/BUILD_DIR/'CMakeCache.txt'),flags=flags)
    control.atomic(root/campaign.SCIENCE,frozen)
    provenance['files'][Path(campaign.SCIENCE).name]=identity(root/campaign.SCIENCE)
    control.atomic(root/LAUNCHER/'provenance.json',provenance)
    campaign.science(root)
    return frozen


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['stage','build','freeze'])
    p.add_argument('--root',type=Path);p.add_argument('--repo',type=Path);p.add_argument('--destination',type=Path);p.add_argument('--launcher-commit')
    a=p.parse_args()
    if a.action=='stage':r=stage(a.repo,a.destination,a.launcher_commit)
    elif a.action=='build':r=build(a.root)
    else:r=freeze(a.root)
    print(json.dumps(r,indent=2))


if __name__=='__main__':main()
