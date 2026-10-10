"""Saved compact JSON/hash audit only; no JSONL decode, interval or target arithmetic."""
from pathlib import Path
import hashlib
import json
import time
import traceback

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
OWNER = BASE/'continuum/manufactured-Gaussian-a2-cache-v6-instrumented-held-20261009'
ROOT = BASE/'Gaussian-a2-cache-v6-root-release-20261009'
CHILD = OWNER/'attempts/certificate001'
INV = ROOT/'certificate-invocation001'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    path = Path(path)
    if path.suffix != '.json' or path.stat().st_size > 1048576:
        raise RuntimeError('only compact JSON may be decoded')
    return json.loads(path.read_text(), parse_constant=lambda s:
                      (_ for _ in ()).throw(ValueError(s)))


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def metadata(path):
    path = Path(path).resolve()
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}


def verify(pins):
    for path, digest in pins.items():
        if sha(path) != digest:
            raise RuntimeError('changed saved source/output '+path)


def main():
    started = time.monotonic()
    if (HERE/'receipt.json').exists() or (HERE/'summary.json').exists():
        raise RuntimeError('fresh one-shot saved review only')
    recipe = load(HERE/'recipe.json')
    pins = dict(recipe['pins'])
    pins[str(Path(__file__).resolve())] = sha(__file__)
    pins[str(HERE/'recipe.json')] = sha(HERE/'recipe.json')
    pins[str(HERE/'PLAN.md')] = sha(HERE/'PLAN.md')
    record = {'passed':False,'saved_data_review_only':True,'scientific_execution':False,
              'certificate_JSONL_decoded':False,'original_actual_FAIL_preserved':True}
    try:
        verify(pins)
        write(HERE/'pins-before.json', pins)
        child = load(CHILD/'receipt.json')
        root = load(INV/'receipt.json')
        progress = load(CHILD/'progress.json')
        contract = load(OWNER/'recipe.json')
        release = load(ROOT/'certificate-release.json')
        command = load(CHILD/'command.json')
        invocation = load(INV/'invocation.json')
        if not (sha(CHILD/'receipt.json') == '1b30e041f720ad1e63eb3067ebaab366b1157e27b508e951216008ac609c27b1'
                and child['completed'] is False and child['passed'] is False
                and child['returncode'] == 1 and child['inputs_unchanged'] is True
                and root['completed'] is False and root['accepted_stage'] is False
                and root['producer_passed'] is False and root['returncode'] == 1
                and root['inputs_unchanged'] is True and root['global_slicing_accepted'] is False
                and root['replay_authorized'] is False and root['report_sha256'] is None
                and root['child_receipt_sha256'] == sha(CHILD/'receipt.json')):
            raise RuntimeError('exact failed root/child classification differs')
        if not (child['recipe_sha256'] == sha(OWNER/'recipe.json')
                and child['source_index_sha256'] == sha(OWNER/'source-index.json')
                and sha(OWNER/'source-index.json') == 'cafc4df240ba9988919f2f4d97522bda2f71a19f1899a625d2be98afa2144397'
                and sha(OWNER/'recipe.json') == '9a7ee237b6fd4d764bac38e62ff4307248fbc4beb57b7a0dbd307e690d0e0742'):
            raise RuntimeError('exact source/recipe binding differs')
        before = load(INV/'pins-before.json')
        after = load(INV/'pins-after.json')
        if before != after:
            raise RuntimeError('root pre/post source transcript differs')
        verify(before)
        for item in root['output_inventory'] + child['outputs'] + child['dynamic_input_pins']:
            if metadata(item['path']) != {key:item[key] for key in ('path','bytes','sha256')}:
                raise RuntimeError('recorded actual saved output/dynamic pin differs')
        partial = metadata(CHILD/'certificate.jsonl')
        if not (partial['bytes'] == 2161493 and partial['sha256'] == '27ae847e229656277a91467682c15969ab3f74d5a2fb837245dafb31cc1568cc'):
            raise RuntimeError('partial certificate metadata differs')
        if any((CHILD/name).exists() for name in ('report.json','certificate-report.json')):
            raise RuntimeError('unexpected accepted report now present')
        if not (progress['status'] == 'domain_time_cap'
                and progress['coverage_fraction_claimed'] is False
                and progress['global_slicing_accepted'] is False
                and progress['completion_requires_full_producer_footer_and_independent_replay'] is True
                and progress['nodes'] == 8677 and progress['leaves'] == 4331 and progress['pending'] == 19):
            raise RuntimeError('fixed final observed-work counters differ')
        roots = progress['root_counters']
        actual = [(item['root'],item['nodes'],item['leaves'],item['pending_nodes']) for item in roots]
        if actual != [(0,557,279,0),(1,8120,4052,17),(2,0,0,1),(3,0,0,1)]:
            raise RuntimeError('fixed reported root work differs')
        if not (sum(item['nodes'] for item in roots) == progress['nodes']
                and sum(item['leaves'] for item in roots) == progress['leaves']
                and sum(item['pending_nodes'] for item in roots) == progress['pending']
                and len(progress['remaining_pending_boxes_in_DFS_order']) == 19):
            raise RuntimeError('compact integer counter association differs')
        cache = progress['cache']
        if {name:cache[name] for name in ('entries','evictions','hits','misses')} != {
                'entries':4096,'evictions':12332,'hits':17164,'misses':16428}:
            raise RuntimeError('observed cache counters differ')
        if not (contract['domain_wall_seconds'] == release['caps']['producer'] == invocation['producer_seconds'] == 600
                and contract['stage_timeouts']['certificate'] == release['caps']['inner_wrapper'] == command['timeout_seconds'] == 660
                and release['caps']['root_process_group'] == invocation['root_process_group_seconds'] == root['process_group_limit_seconds'] == 720
                and root['process_group_timeout'] is False):
            raise RuntimeError('producer/wrapper/root cap association differs')
        error = (CHILD/'stderr.log').read_text()
        if 'UNRESOLVED: declared domain time limit' not in error:
            raise RuntimeError('saved producer domain-cap exception missing')
        unit = load(contract['cached_v3_units_receipt']['path'])
        units = load(contract['cached_v3_units_report']['path'])
        if not (unit['completed'] is True and unit['passed'] is True and unit['inputs_unchanged'] is True
                and unit['returncode'] == 0 and units['passed'] is True
                and units['combined_unit_count'] == 179 and units['cache_unit_count'] == 35
                and units['case_count'] == 144 and units['domain_boxes_evaluated'] == 0):
            raise RuntimeError('unchanged179-unit prerequisite differs')
        summary = {
            'scope':'Observed compact work/provenance of a failed finite-box certificate attempt only',
            'passed_saved_audit':True,'actual_certificate_completed':False,'actual_certificate_passed':False,
            'producer_domain_cap_reached':True,'outer_process_group_cap_reached':False,
            'caps_seconds':{'producer':600,'inner_wrapper':660,'root_process_group':720},
            'saved_wall_seconds':{'root':str(root['seconds']),'child_wrapper':str(child['elapsed_seconds']),
                                  'last_progress':str(progress['elapsed_seconds'])},
            'observed_nodes':progress['nodes'],'observed_leaves':progress['leaves'],'observed_pending':progress['pending'],
            'reported_root_counts':actual,'most_recent_node_role':progress['active_node_role'],
            'reported_cache':{name:cache[name] for name in ('entries','evictions','hits','misses','semantics')},
            'method_evaluations':progress['method_evaluations'],
            'positive_leaf_methods_as_reported':progress['positive_leaf_methods'],
            'source_root_pins_verified':len(before),'source_and_output_hashes_verified':True,
            'partial_certificate_metadata_only':partial,'partial_certificate_decoded':False,
            'unchanged179_units_associated':True,'accepted_report_present':False,
            'coverage_fraction_claimed':False,'regional_or_global_positivity_inferred':False,
            'runtime_cause_theorem_claimed':False,'replay_or_resume_executed':False,
            'limits':[
                'These are producer instrumentation counters, not independent interval or certificate replay.',
                'Root0 reports no pending stack nodes, but no accepted footer/report or replay exists; no regional certificate is inferred.',
                'Root1 reports17 pending nodes and roots2/3 are unvisited; no covered-volume fraction is inferred from counts.',
                'Cache counters describe observed operations only; no timing-cause or cache-performance theorem follows.',
                'No interval bound, Gaussian target, partial JSONL record or positivity arithmetic was evaluated.'
            ]}
        write(HERE/'summary.json',summary)
        write(HERE/'original-input-metadata.json',{str(path):metadata(path) for path in pins})
        record.update(passed=True,source_root_pins_verified=len(before),
                      child_receipt_sha256=sha(CHILD/'receipt.json'),progress_sha256=sha(CHILD/'progress.json'))
    except BaseException as exc:
        record.update(error=type(exc).__name__+': '+str(exc))
        (HERE/'failure.txt').write_text(traceback.format_exc())
    finally:
        try:
            verify(pins)
            if 'before' in locals():
                verify(before)
            record['inputs_unchanged']=True
        except BaseException as exc:
            record.update(passed=False,inputs_unchanged=False,post_pin_failure=str(exc))
        write(HERE/'pins-after.json',pins)
        record['seconds']=time.monotonic()-started
        write(HERE/'receipt.json',record)
    print(json.dumps(record))
    if not(record['passed'] and record['inputs_unchanged']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
