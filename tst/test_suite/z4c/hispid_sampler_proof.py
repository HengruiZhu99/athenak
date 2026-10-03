"""Checkpoint-bound sampler proof checks shared by initial horizon drivers."""
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

FIELDS={'gamma','Kij','psi','conformal_metric','Atilde','mean_curvature',
        'correction','attenuation','dgamma'}
TOLERANCE=1e-12


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_migration(path,source):
    path=Path(path).resolve(strict=True);proof=json.loads(path.read_text())
    if (proof.get('schema')!='hispid_portable_sampler_migration_v1'
        or proof.get('purpose')!='sampling_only_no_acceptance_transfer'
        or not all(proof.get(k) is True for k in ('isolated_processes','loaded_images_verified','positive_metric','pure_reference_consumer','passed'))
        or proof.get('criteria',{}).get('field_scaled_linf')!=TOLERANCE):
        raise ValueError('passed separate-process sampler proof with fixed criteria required')
    checkpoint=proof['checkpoint']
    for key in ('file_sha256','source_library_sha256','acceptance','parameterization'):
        if checkpoint.get(key)!=source.get(key):raise ValueError('migration/checkpoint binding differs: '+key)
    if proof['producer_library_sha256']!=source['source_library_sha256']:
        raise ValueError('migration producer mismatch')
    consumer=proof['consumer_library_sha256']
    if not re.fullmatch('[0-9a-f]{64}',consumer) or consumer==proof['producer_library_sha256']:
        raise ValueError('distinct bound consumer required')
    token=source['parameterization']
    custom=re.fullmatch(r'modal_P_C2prolate_map_v3_r([0-9eE+.-]+)_k([0-9eE+.-]+)',token)
    if token=='modal_P_C2prolate_mapped_v2':maps=dict(radial_stretch=.2,angular_stretch=2.)
    elif custom:maps=dict(radial_stretch=float(custom[1]),angular_stretch=float(custom[2]))
    else:raise ValueError('unsupported migration continuous basis/maps')
    if proof['parameterization']!=token or proof['maps']!=maps:
        raise ValueError('migration continuous basis/maps differ')
    workers=proof['workers']
    if len(workers)!=2 or [w['side'] for w in workers]!=['producer','consumer']:
        raise ValueError('separate producer/consumer witnesses required')
    for worker,sha in zip(workers,(proof['producer_library_sha256'],consumer)):
        if (worker.get('passed') is not True or worker.get('returncode')!=0
            or worker['library_sha256']!=sha or worker['checkpoint']!=checkpoint
            or worker['parameterization']!=token or worker['maps']!=maps
            or digest(worker['artifact'])!=worker['artifact_sha256']
            or digest(worker['library_path'])!=sha):
            raise ValueError('migration worker binding failed')
        if any(digest(p)!=s for p,s in worker['dependency_images'].items()):
            raise ValueError('migration dependency changed')
        if any(digest(p)!=s for p,s in worker['runtime_images'].items()):
            raise ValueError('migration runtime changed')
    if (workers[1]['compiled_execution']!='reference' or workers[1]['runtime_images']
        or any('kokkos' in Path(p).name.lower() for p in workers[1]['dependency_images'])):
        raise ValueError('pure non-Kokkos CPU consumer required')
    with np.load(workers[0]['artifact']) as old,np.load(workers[1]['artifact']) as new:
        if set(old.files)!=FIELDS|{'xyz'} or set(old.files)!=set(new.files):
            raise ValueError('migration field inventory differs')
        if not np.array_equal(old['xyz'],new['xyz']) or len(old['xyz'])!=proof['point_count'] or len(old['xyz'])<100:
            raise ValueError('migration point witness differs')
        for field in FIELDS:
            a,b=old[field],new[field]
            if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError('invalid migration field')
            difference=float(np.max(abs(a-b)/(1+abs(a))))
            if difference>TOLERANCE or difference!=proof['differences'][field]:
                raise ValueError('migration field difference failed: '+field)
        if any(np.linalg.eigvalsh(data['gamma'].reshape(-1,3,3)).min()<=0 for data in (old,new)):
            raise ValueError('migration metric is not positive definite')
    return dict(path=str(path),file_sha256=digest(path),consumer_library_sha256=consumer,
                consumer_library_path=workers[1]['library_path'],maps=maps,passed=True,
                consumer_dependency_images=workers[1]['dependency_images'],
                purpose=proof['purpose'])


def import_evidence(stdout,source,migration=None):
    match=re.search(r'HiSpID import source=([0-9a-f]{64}) acceptance=(\w+) consumer=([0-9a-f]{64}) ADM/Z4c relative error=([\deE+.-]+)',stdout)
    if match is None:return dict(passed=False)
    error=float(match[4]);expected=migration['consumer_library_sha256'] if migration else source['source_library_sha256']
    result=dict(source_library_sha256=match[1],acceptance=match[2],consumer_library_sha256=match[3],
                adm_z4c_relative_error=error,explicit_migration=migration is not None)
    result['passed']=bool(match[1]==source['source_library_sha256'] and match[2]==source['acceptance']
        and match[3]==expected and math.isfinite(error) and 0<=error<1e-11)
    image=re.search(r'HiSpID consumer_image=("(?:[^"\\]|\\.)*")',stdout)
    if image:
        loaded=Path(json.loads(image[1])).resolve(strict=True)
        result['loaded_consumer_image']=str(loaded);result['loaded_consumer_sha256']=digest(loaded)
        result['passed'] &= result['loaded_consumer_sha256']==expected
        if migration:result['passed'] &= loaded==Path(migration['consumer_library_path']).resolve(strict=True)
    elif migration:result['passed']=False
    if migration:
        dependencies={}
        for symbol,image in re.findall(r'HiSpID consumer_dependency symbol=("(?:[^"\\]|\\.)*") image=("(?:[^"\\]|\\.)*")',stdout):
            path=str(Path(json.loads(image)).resolve(strict=True));dependencies[path]=digest(path)
        result['consumer_dependency_images']=dependencies
        result['passed'] &= dependencies==migration['consumer_dependency_images']
    return result
