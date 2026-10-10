"""Stdlib compact saved-data association; no operand/rounding recomputation."""
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
OWNER = BASE/'boundary/reference-wave-map-v9-mass-measure-diagnostic-held-20261009'
ROOT = BASE/'wave-map-v9-mass-measure-diagnostic-root-release-20261009'
EXPECTED = {
    OWNER/'attempts/diagnostic001/receipt.json':'80c93fe0ab717c8cbaaedd7ab14bfd0dc77b905f001de8b783418382a6e57d98',
    OWNER/'attempts/diagnostic001/result.json':'862459d97b8ae5704f376d6cf5758826da863187ea5fd30cad797da48ff10926',
    OWNER/'outer-invocation001/receipt.json':'92d16b76f4ada57eea4f6a8dc8571d7d5b0fce8b78d6f48b0ef954a43e2badec',
    OWNER/'source-index.json':'c8d1b760ec7dea2711719078e434139af28496f5050c3ea7a336d6e39ff2103f',
    OWNER/'recipe.json':'66b29ff7f20668bce2bcdee035dc97900ab95f565e8bd2dee2fd8c174ddd7e78',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1 << 20),b''):
            h.update(block)
    return h.hexdigest()


def pin(path):
    path = Path(path).resolve()
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))


def load(path,manifest=False):
    path = Path(path)
    if path.suffix=='.jsonl' or path.stat().st_size > (16 << 20 if manifest else 1 << 20):
        raise RuntimeError('saved compact JSON or named hash manifest only')
    return json.loads(path.read_text(),parse_float=Decimal,
        parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))


def write(name,value):
    (HERE/name).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def check_pins(pins):
    transcript = hashlib.sha256()
    for path,digest in sorted(pins.items()):
        assert sha(path)==digest,path
        transcript.update((path+'\0'+digest+'\n').encode())
    return transcript.hexdigest()


def main():
    start = time.monotonic()
    for path,digest in EXPECTED.items():
        assert sha(path)==digest,path
    before = load(ROOT/'outer-invocation001/pins-before.json',manifest=True)
    after = load(ROOT/'outer-invocation001/pins-after.json',manifest=True)
    assert before==after and len(before)==4653
    pin_digest = check_pins(before)
    originals = list(EXPECTED)+[
        ROOT/'outer-invocation001/receipt.json',ROOT/'outer-invocation001/command.json',
        ROOT/'outer-invocation001/pins-before.json',ROOT/'outer-invocation001/pins-after.json',
        ROOT/'outer-invocation001/stdout.log',ROOT/'outer-invocation001/stderr.log',
        ROOT/'release.json',ROOT/'preparation-receipt001.json',ROOT/'launcher-source-index.json',
        ROOT/'prepare_release001.py',ROOT/'launch.py',ROOT/'scipy-runtime-pins001.json',
        OWNER/'attempts/diagnostic001/progress.json',OWNER/'attempts/diagnostic001/operand-weighting-rounding.json',
        OWNER/'attempts/diagnostic001/pins-before.json',OWNER/'outer-invocation001/pins-before.json',
        OWNER/'outer-invocation001/stdout.log',OWNER/'outer-invocation001/stderr.log',
        HERE/'PLAN.md',HERE/'audit_saved.py']
    compact_pins = [pin(path) for path in originals]
    write('original-input-metadata.json',compact_pins)
    child = load(OWNER/'attempts/diagnostic001/receipt.json')
    wrapper = load(OWNER/'outer-invocation001/receipt.json')
    root = load(ROOT/'outer-invocation001/receipt.json')
    result = load(OWNER/'attempts/diagnostic001/result.json')
    recipe = load(OWNER/'recipe.json')
    for receipt in [child,wrapper,root]:
        assert receipt['completed'] is receipt['inputs_unchanged'] is True
        assert type(receipt['returncode']) is int and receipt['returncode']==0
        assert receipt['original_v9_radial_readback_passed'] is False
    assert root['protected_pins']==4653 and root['scipy_metadata_files']==1332
    assert root['start_new_session'] and root['root_cap_seconds']==180
    assert root['all_prepared_inputs_verified'] and not root.get('root_cap_reached',False)
    assert root['child_receipt_sha256']==EXPECTED[OWNER/'attempts/diagnostic001/receipt.json']
    assert wrapper['child_receipt_sha256']==root['child_receipt_sha256']
    assert root['wrapper_receipt_sha256']==EXPECTED[OWNER/'outer-invocation001/receipt.json']
    assert root['result_sha256']==EXPECTED[OWNER/'attempts/diagnostic001/result.json']
    assert root['environment']==wrapper['environment']==recipe['environment']
    assert root['command'][0:3]==wrapper['command'][0:3]==[recipe['python'],'-B','-s']
    assert root['actual_child_returncode']==wrapper['actual_child_returncode']==0
    for parent,receipt in [(ROOT/'outer-invocation001',root),(OWNER/'outer-invocation001',wrapper)]:
        assert (parent/'stderr.log').stat().st_size==0
        key='stderr.log_sha256' if parent.parent==ROOT else 'stderr_sha256'
        assert sha(parent/'stderr.log')==receipt[key]
        key='stdout.log_sha256' if parent.parent==ROOT else 'stdout_sha256'
        assert sha(parent/'stdout.log')==receipt[key]
    assert result['diagnostic_completed'] is True
    assert result['original_v9_radial_readback_passed'] is False
    assert result['expected_components']==131072 and result['input_map_radius_window']==[609,640]
    for flag in ['E_accumulated','operator_matrices_loaded','source_tables_loaded','SVD_executed','query_or_generator_executed']:
        assert result[flag] is False
    assert result['selected_npz_arrays_accessed']==recipe['selected_npz_arrays']
    rows = result['radii']
    assert [row['radius_index'] for row in rows]==list(range(609,641))
    aggregate = Counter()
    affected = []
    for row in rows:
        assert row['counts']['multiply_components']==4096
        aggregate.update(row['counts'])
        if row['counts'].get('potential_tiny_products',0):
            affected.append(row)
    assert dict(aggregate)==result['counts']==root['counts']
    expected = dict(multiply_components=131072,potential_tiny_products=28,
        zero_or_provably_normal_products=131044,strict_scalar_underflow_observed=28,
        whole_array_underflow_observed=1,rounded_zero=20,rounded_nonzero_subnormal=8,
        exact_below_min_normal=28,exact_below_min_subnormal=28,
        inexact_tiny_before_rounding=28,inexact_tiny_after_rounding=28)
    assert result['counts']==expected
    assert [row['radius_index'] for row in affected]==[627]
    for field in ['first_strict_array_exception','last_strict_array_exception']:
        assert result[field]==dict(exception='underflow encountered in multiply',operation='measure*mass_sum',radius_index=627)
    for field in ['first_inexact_tiny','last_inexact_tiny']:
        assert result[field]['radius_index']==627
    assert len(result['examples'])==28 and all(row['radius_index']==627 for row in result['examples'])
    assert root['seconds']==Decimal('18.992963')
    assert root['child_wall_seconds']==Decimal('11.461643041')
    assert wrapper['seconds']==Decimal('11.382416959') and child['seconds']==Decimal('6.546369042')
    failed = load(recipe['failed_receipt'])
    assert failed['completed'] is False and failed['inputs_unchanged'] is True and failed['returncode']==1
    summary = dict(passed_saved_audit=True,diagnostic_completed=True,root_pins4653_unchanged=True,
        source_result_receipt_hashes_verified=True,counts=expected,affected_radius_rows=affected,
        first_and_last_observed_radius=627,window=[609,640],exact_onset_outside_window_claimed=False,
        maximum_local_rounding_error_exact_saved=result['maximum_local_rounding_error_exact'],
        timings_seconds={key:str(value) for key,value in [('root_wall',root['seconds']),
            ('root_child_wall',root['child_wall_seconds']),('owner_wrapper_wall',wrapper['seconds']),
            ('diagnostic_body',child['seconds'])]},
        original_v9_radial_readback_passed=False,other_outer_product_occurrences_claimed=False,
        scope='Saved E-product classification/provenance only; no matrix qualification or target recomputation',
        arrays_maps_JSONL_decoded=False,scientific_imports_queries_or_rounding_recomputed=False)
    write('summary.json',summary)
    assert check_pins(before)==pin_digest
    assert [pin(path) for path in originals]==compact_pins
    write('receipt.json',dict(passed=True,inputs_unchanged=True,protected_root_pins=4653,
        root_pin_transcript_sha256=pin_digest,compact_original_files=len(originals),
        result_sha256=EXPECTED[OWNER/'attempts/diagnostic001/result.json'],
        source_review_only=False,saved_data_review_only=True,no_scientific_reexecution=True,
        original_v9_radial_FAIL_preserved=True,seconds=time.monotonic()-start))
    print(json.dumps(summary,sort_keys=True))


if __name__ == '__main__':
    main()
