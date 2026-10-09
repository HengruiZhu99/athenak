"""Freeze compact N8 matrix evidence; large scientific data stay metadata-only."""
from pathlib import Path
import datetime, hashlib, json, shutil, subprocess
import numpy as np

P=Path(__file__).resolve().parent
ROOT=P.parents[2]
D=P/'immutable-total-J-finite-rb-N8-matrix-control-20261009'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def read(path):return json.loads(path.read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))

def main():
    assert not D.exists();D.mkdir()
    files=[];large=[];aliases=[];finite_json=0
    def capture(origin,target):
        nonlocal finite_json
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(origin,target)
        assert sha(origin)==sha(target)
        if target.suffix=='.json':read(target);finite_json+=1
        files.append({'path':str(target.relative_to(D)),'origin':str(origin.resolve()),'bytes':target.stat().st_size,'sha256':sha(target)})
    excluded={'constraint-rate-attempts','__pycache__',D.name}
    for origin in sorted(P.rglob('*')):
        relative=origin.relative_to(P)
        if any(part in excluded or part.endswith('.dSYM') for part in relative.parts):continue
        if origin.is_symlink():
            aliases.append({'path':str(relative),'resolved':str(origin.resolve()),'is_directory':origin.is_dir()})
            continue
        if not origin.is_file():continue
        binary=origin.suffix in ('.npz','.bin','.a','.o') or origin.name.startswith('radial-bridge-')
        raw=origin.name in ('queries.txt','output.txt') or origin.stat().st_size>2*1024*1024
        if binary or raw:
            entry={'path':str(relative),'origin':str(origin.resolve()),'bytes':origin.stat().st_size,'sha256':sha(origin),'current_bytes_rehashed':True}
            if origin.suffix=='.npz':
                with np.load(origin,allow_pickle=False) as z:
                    entry['arrays']={k:{'shape':list(z[k].shape),'dtype':str(z[k].dtype)} for k in z.files}
                    assert all(np.isfinite(z[k]).all() for k in z.files if np.issubdtype(z[k].dtype,np.number))
                entry['numeric_arrays_finite']=True
            large.append(entry)
        else:capture(origin,D/'implementation'/relative)
    held=ROOT/'build-layer-research/boundary/total-j-finite-rb-control-held-20261009'
    for origin in sorted(held.rglob('*')):
        if origin.is_file():capture(origin,D/'held-plan'/origin.relative_to(held))
    independent=ROOT/'build-layer-research/continuum/immutable-finite-rb-independent-matrix-readback-20261009'
    assert sha(independent/'index.json')=='f11ef3a00e7943ae9afb8c0610b1d7b3629ea14dce53d6a9437a8cd6408a2667'
    capsule=read(independent/'index.json')
    for item in capsule['files']:
        origin=independent/item['path'];assert sha(origin)==item['sha256']
        capture(origin,D/'independent-matrix-readback'/item['path'])
    capture(independent/'index.json',D/'independent-matrix-readback/index.json')
    forcing_review=ROOT/'build-layer-research/continuum/finite-rb-forcing-family-independent-review-20261009'
    assert sha(forcing_review/'receipt.json')=='875d1ec679f38c2b3e3cc9aecf3b95c3b3655378e3010e6553064ebf7dc94746'
    assert read(forcing_review/'receipt.json')['total_fixed_fields']==179
    for origin in sorted(forcing_review.rglob('*')):
        if not origin.is_file():continue
        if origin.suffix=='.npz':
            large.append({'path':'independent-forcing-review/'+str(origin.relative_to(forcing_review)),
                          'origin':str(origin),'bytes':origin.stat().st_size,'sha256':sha(origin),'current_bytes_rehashed':True})
        else:capture(origin,D/'independent-forcing-review'/origin.relative_to(forcing_review))
    analytic=ROOT/'build-layer-research/continuum/finite-rb-projection-defect/exact-projected/attempt-1791559237496324000'
    assert sha(analytic/'receipt.json')=='98f18f39011601946df3b78d242a42e7ba2be3c6dfe37a174591e1d83e1ab78c'
    for name in ('receipt.json','summary.json'):capture(analytic/name,D/'separate-analytic-point'/name)
    input_pins=[]
    for label,rel,expected in (
        ('basis','continuum/total-j-harmonic-basis/immutable-Cartesian-total-J-basis-20261009/index.json','414f241e986e0c46b7d94820fb060b704166d973284854c53d2e8c64ee6c489e'),
        ('continuum_rates','continuum/finite-rb-constraint-rate-oracle/immutable-finite-rb-C0-constraint-rates-20261009/index.json','3d4c613a814a8a3325a7f980c2e20dcabf3ea08ddbcb10d42026ddca732d4e2f'),
        ('local_angular','boundary/total-j-local-angular-20261009/immutable-C0-spatialnorm-total-J-local-angular-20261009/index.json','b4131aa02f093b7744d3de1b600e829b7213a59779672978bd84c71f6912c513'),
        ('core','boundary/total-j-flat-core-envelope-20261009/immutable-total-J-flat-core-envelope-20261009/index.json','b0fde1e0eb95d6660a9fa3d190eda69207e6153c88da35366b038369ac3aa3d4'),
        ('principal','continuum/harmonic-principal-constraint-sectors/immutable-harmonic-normal-principal-sectors-20261009/index.json','05d4d7308477efe26d26ec7256fd9ba824040849363ed7f723031e5760adc0fc')):
        if not expected:continue
        origin=ROOT/'build-layer-research'/rel;assert sha(origin)==expected
        capture(origin,D/'upstream-indices'/f'{label}.json')
        input_pins.append({'kind':label,'path':str(origin),'sha256':expected})
    family=[]
    for J,n in ((0,33),(1,65),(2,81)):
        r=read(P/f'J{J}-N8-forcing-family-replay001/report.json')
        assert r['passed'] and r['family_count']==n
        family.append({'J':J,'fields':n,'coefficient_error_max_scaled':r['max_coefficient_error_scaled'],'energy_error_max_scaled':r['max_energy_error_scaled']})
    summary={'passed_source_mass_normal_trace_weak_strong_and_integration_controls':True,
             'passed_complete_fixed_forcing_family':True,'forcing_family':family,
             'source_release_executable_sha256':'2293e9be6f75042f926f22232039c3c3bdd28826eb9e80061905c272b7adce15',
             'source_debug_executable_sha256':'75193a3b023eb0004f285fabb1a532efd24881db4cb660b3fa96e084a6e95507',
             'failed_global_Q64_Q128_and_segmented_Q32_preserved':True,
             'projected_constraint_defect_readback':'separate analytic point PASS; general nongauge continuum comparator unresolved; two ordinary-FD stops preserved; no full projected-constraint admission by this capsule',
             'generator_spectrum_or_propagation_performed':False,'stability_CPBC_or_exact_scri_accepted':False,
             'large_data_policy':'all scientific NPZ, raw queries/outputs and executable bytes retained locally and represented only by metadata here',
             'pre_API_executable_preservation_limitation':'eaf81162/eaee accepted source-gate binaries were overwritten before retention; exact source/commands/deps/gates and correction retained; current API2293/7519 bytes retained',
             'compiler_binary_identity':'compiler version and exact commands recorded at build; compiler executable byte hash was not captured at build',
             'upstream_input_indices':input_pins}
    write(D/'summary.json',summary)
    read(D/'summary.json');finite_json+=1
    files.append({'path':'summary.json','origin':'generated summary; source freeze_N8_checkpoint.py copied in implementation',
                  'bytes':(D/'summary.json').stat().st_size,'sha256':sha(D/'summary.json')})
    record={'kind':'Immutable actual finite-ball total-J N8 source/matrix consistency control',
            'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'freeze_HEAD':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip(),
            'production_source_commit':'27c19d20696ea6dd4704032c51dfd026218f64f2',
            'small_file_count':len(files),'small_file_bytes':sum(x['bytes'] for x in files),
            'finite_JSON_readbacks':finite_json,'external_large_file_count':len(large),
            'files':files,'external_large_files':large,'symlink_aliases':aliases,
            'scientific_scope':'actual finite-rb operator consistency only; no generator eigs/propagation, CPBC, exact scri or stability certificate'}
    for item in files:assert sha(D/item['path'])==item['sha256']
    write(D/'index.json',record)
    print(json.dumps({'path':str(D),'index_sha256':sha(D/'index.json'),'small_files':len(files),'small_bytes':record['small_file_bytes'],'large_metadata':len(large),'finite_JSON':finite_json},indent=2))

if __name__=='__main__':main()
