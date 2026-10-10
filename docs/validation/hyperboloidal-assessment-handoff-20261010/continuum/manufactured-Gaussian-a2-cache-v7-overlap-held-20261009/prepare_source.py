"""One-shot source/metadata copying, hashes and AST proofs only; no math imports."""
from pathlib import Path
import ast
import difflib
import hashlib
import json
import shutil

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2] / 'build-layer-research'
OLD = BASE / 'continuum/manufactured-Gaussian-a2-cache-v6-instrumented-held-20261009'
RUN = OLD / 'attempts/certificate001'
PENCIL = BASE / 'continuum/manufactured-Gaussian-a2-Taylor-overlap-pencil-20261009'
ADDENDUM = BASE / 'continuum/Gaussian-a2-cache-v6-inventory-addendum-20261009'


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def pin(path):
    path = Path(path).resolve()
    result = {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}
    if path.suffix in ('.jsonl','.npz','.npy') or result['bytes'] > 1048576:
        result['role'] = 'large_payload_metadata_only'
    return result


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise RuntimeError('nonunique exact source anchor: '+before)
    return source.replace(before,after,1)


def strip_added(source):
    output, inside = [], False
    for line in source.splitlines(keepends=True):
        if line.strip() == '# BEGIN_V7_ADMISSION_OR_UNIT_ONLY':
            if inside: raise RuntimeError('nested marker')
            inside = True
        elif line.strip() == '# END_V7_ADMISSION_OR_UNIT_ONLY':
            if not inside: raise RuntimeError('unmatched marker')
            inside = False
        elif not inside:
            output.append(line)
    if inside: raise RuntimeError('unclosed marker')
    return ''.join(output)


def added(lines, indent=4):
    prefix = ' '*indent
    return prefix+'# BEGIN_V7_ADMISSION_OR_UNIT_ONLY\n'+''.join(prefix+x+'\n' for x in lines)+prefix+'# END_V7_ADMISSION_OR_UNIT_ONLY\n'


def main():
    if (HERE/'source-index.json').exists():
        raise RuntimeError('one-shot frozen source already exists')
    for path, expected in [
        (OLD/'source-index.json','cafc4df240ba9988919f2f4d97522bda2f71a19f1899a625d2be98afa2144397'),
        (OLD/'recipe.json','9a7ee237b6fd4d764bac38e62ff4307248fbc4beb57b7a0dbd307e690d0e0742'),
        (RUN/'receipt.json','1b30e041f720ad1e63eb3067ebaab366b1157e27b508e951216008ac609c27b1'),
        (PENCIL/'index.json','1712e88d5aabe9d3876183f1e1cce3e0fc370be454e431201e72f39833d9434c'),
        (ADDENDUM/'index.json','62be282de52536b2e5f3f5c82bf43f9de309efcbe3e38ab4b146325717aac4f2')]:
        if sha(path) != expected:
            raise RuntimeError('authoritative source/history pin differs: '+str(path))
    oldindex, oldrecipe, failed = load(OLD/'source-index.json'),load(OLD/'recipe.json'),load(RUN/'receipt.json')
    if not (failed['completed'] is False and failed['passed'] is False and failed['inputs_unchanged'] is True
            and type(failed['returncode']) is int and failed['returncode'] == 1
            and failed['source_index_sha256'] == sha(OLD/'source-index.json')
            and failed['recipe_sha256'] == sha(OLD/'recipe.json')):
        raise RuntimeError('v6 failed receipt classification differs')
    if (RUN/'report.json').exists():
        raise RuntimeError('v6 cap failure unexpectedly has a producer report')
    protected = {}
    for entry in oldrecipe['protected_inputs']+oldindex['files']:
        path = Path(entry['path'])
        if sha(path) != entry['sha256'] or path.stat().st_size != entry['bytes']:
            raise RuntimeError('old input differs: '+str(path))
        protected[str(path.resolve())] = pin(path)
    for path in (OLD/'source-index.json',ADDENDUM/'index.json'):
        protected[str(path)] = pin(path)
    history = HERE/'history/v6'
    history.mkdir(parents=True)
    # The v6 top-level source copies are historical; the entire transitive old
    # indexed inventory remains protected at its original immutable locations.
    for path in sorted(OLD.iterdir()):
        if path.is_file():
            if path.suffix in ('.jsonl','.npz','.npy') or path.stat().st_size > 1048576:
                raise RuntimeError('top-level history unexpectedly contains a payload')
            shutil.copyfile(path,history/path.name)
    for path in sorted(RUN.iterdir()):
        if not path.is_file(): continue
        entry = pin(path)
        protected[str(path)] = entry
        if 'role' not in entry:
            shutil.copyfile(path,history/('failed-'+path.name))
    # All partial payloads are only streamed for hashes; none is decoded/copied.
    for prefix in [PENCIL,ADDENDUM,
                   BASE/'Gaussian-a2-cache-v6-root-release-20261009',
                   BASE/'boundary/manufactured-Gaussian-a2-cache-v6-independent-source-review-20261009']:
        for path in sorted(prefix.rglob('*')):
            if path.is_file(): protected[str(path.resolve())] = pin(path)
    saved_review = BASE/'boundary/manufactured-Gaussian-a2-cache-v6-independent-saved-cap-review-20261009'
    # Bind completed compact cap review only if its immutable index exists now.
    if (saved_review/'index.json').is_file():
        for path in sorted(saved_review.rglob('*')):
            if path.is_file(): protected[str(path.resolve())] = pin(path)
    top = ['interval.py','interval_uncached.py','producer_bounds.py','replay_bounds.py',
           'certificate_stage.py','replay_stage.py','unit_stage.py','cache_units.py',
           'instrumentation.py','admission.py','run_once.py']
    for name in top:
        shutil.copyfile(OLD/name,HERE/name)
    changes = {
        'producer_bounds.py': [('if rhi <= sigma:', 'if rhi <= 2 * sigma:'),('elif rlo >= sigma:', 'elif rlo >= 2 * sigma:')],
        'replay_bounds.py': [('if rhi <= sigma:', 'if rhi <= 2 * sigma:'),('elif rlo >= sigma:', 'elif rlo >= 2 * sigma:')],
        'certificate_stage.py': [('[(Q(1, 50), sigma), (sigma, Q(4))]', '[(Q(1, 50), 2 * sigma), (2 * sigma, Q(4))]')],
        'replay_stage.py': [
            ('(Q(1, 50), sigma, Q(0), Q(4) + 8 * sigma)', '(Q(1, 50), 2 * sigma, Q(0), Q(4) + 8 * sigma)'),
            ('(sigma, Q(4), Q(0), Q(4) + 8 * sigma)', '(2 * sigma, Q(4), Q(0), Q(4) + 8 * sigma)')]
    }
    proofs = []
    for name, replacements in changes.items():
        original = (OLD/name).read_text()
        new = original
        for before,after in replacements: new=replace_once(new,before,after)
        (HERE/name).write_text(new)
        reverse = new
        for before,after in reversed(replacements): reverse=replace_once(reverse,after,before)
        if reverse != original: raise RuntimeError('science reverse bytes differ: '+name)
        proofs.append({'file':name,'reverse_byte_equal':True,
                       'reverse_AST_equal':ast.dump(ast.parse(reverse),include_attributes=False)==ast.dump(ast.parse(original),include_attributes=False),
                       'exact_replacements':replacements})
    admission = (OLD/'admission.py').read_text()
    before_guard='    if stage != "certificate":\n        raise RuntimeError("cache v6 instrumentation admits only a fresh diagnostic producer")\n'
    after_guard='    if stage not in ("units", "certificate", "replay"):\n        raise RuntimeError("cache v7 admits only exact separately released stages")\n'
    admission=replace_once(admission,before_guard,after_guard)
    prior_extra=added([
        'v6_pin = recipe["cached_v6_timeout_receipt"]',
        'if authorization.get("prior_v6_timeout_receipt") != v6_pin:',
        '    raise RuntimeError("exact failed cached-v6 receipt pin required")',
        'verify_pin(v6_pin)',
        'v6 = load(v6_pin["path"])',
        'if not (v6.get("completed") is False and v6.get("passed") is False',
        '        and type(v6.get("returncode")) is int and v6.get("returncode") == 1',
        '        and v6.get("inputs_unchanged") is True and v6.get("stage") == "certificate"',
        '        and v6.get("source_index_sha256") == recipe["cached_v6_source_index_sha256"]',
        '        and v6.get("recipe_sha256") == recipe["cached_v6_recipe_sha256"]):',
        '    raise RuntimeError("v6 timeout classification differs")',
        'for entry in (recipe["cached_v6_progress"], recipe["cached_v6_stderr"],',
        '              recipe["cached_v6_command"], recipe["cached_v6_partial_certificate_metadata"]):',
        '    if entry not in v6.get("outputs", []):',
        '        raise RuntimeError("v6 failed receipt does not bind history output")',
        'if "UNRESOLVED: declared domain time limit" not in Path(recipe["cached_v6_stderr"]["path"]).read_text():',
        '    raise RuntimeError("v6 failure is not the declared cap")',
        'if (Path(v6_pin["path"]).parent / "report.json").exists():',
        '    raise RuntimeError("failed v6 unexpectedly has a producer report")'])
    admission=replace_once(admission,'    dependencies = {"prior_timeout": prior}\n',prior_extra+'    dependencies = {"prior_timeout": prior}\n')
    unit_extra=added([
        'if stage in ("certificate", "replay"):',
        '    fresh_unit_pin = authorization["new_units_receipt"]',
        '    fresh_units = verify_receipt(fresh_unit_pin, "units", index_sha)',
        '    if fresh_units.get("recipe_sha256") != recipe_sha:',
        '        raise RuntimeError("new overlap units have a wrong recipe")',
        '    report_pin = authorization["new_units_report"]',
        '    overlap_pin = authorization["new_overlap_report"]',
        '    cache_pin = authorization["new_cache_report"]',
        '    for entry in (report_pin, overlap_pin, cache_pin):',
        '        verify_pin(entry)',
        '        if entry not in fresh_units.get("outputs", []):',
        '            raise RuntimeError("new unit receipt does not bind required report")',
        '    report, overlap, cache = load(report_pin["path"]), load(overlap_pin["path"]), load(cache_pin["path"])',
        '    if not (report.get("passed") is True and report.get("stage") == "units"',
        '            and report.get("source_index_sha256") == index_sha',
        '            and report.get("case_count") == recipe["expected_unit_count"]',
        '            and report.get("cache_unit_count") == recipe["expected_cache_unit_count"]',
        '            and report.get("overlap_unit_count") == recipe["expected_overlap_unit_count"]',
        '            and report.get("combined_unit_count") == recipe["expected_combined_unit_count"]',
        '            and report.get("cache_units_passed") is True and report.get("overlap_units_passed") is True',
        '            and overlap.get("passed") is True',
        '            and overlap.get("case_count") == recipe["expected_overlap_unit_count"]',
        '            and len(overlap.get("cases", [])) == recipe["expected_overlap_unit_count"]',
        '            and all(row.get("passed") is True for row in overlap["cases"])',
        '            and cache.get("passed") is True and cache.get("case_count") == 35',
        '            and len(cache.get("cases", [])) == 35',
        '            and all(row.get("passed") is True for row in cache["cases"])):',
        '        raise RuntimeError("fresh 237-unit overlap prerequisite failed")',
        '    dependencies["new_units"] = fresh_units'])
    admission=replace_once(admission,'    if stage == "replay":\n',unit_extra+'    if stage == "replay":\n')
    (HERE/'admission.py').write_text(admission)
    if strip_added(admission).replace(after_guard,before_guard,1)!=(OLD/'admission.py').read_text():
        raise RuntimeError('admission reverse bytes differ')
    unit=(OLD/'unit_stage.py').read_text()
    extra=added([
        'from overlap_units import run_overlap_units',
        'overlap_report = run_overlap_units(recipe)',
        'if overlap_report["case_count"] != recipe["expected_overlap_unit_count"]:',
        '    raise RuntimeError("overlap-unit recipe count mismatch")',
        '(out / "overlap-report.json").write_text(json.dumps(overlap_report, indent=2, sort_keys=True, allow_nan=False) + "\\n")'])
    unit=replace_once(unit,'    report = {"passed": True, "stage": "units", "source_index_sha256": index_sha,\n',extra+'    report = {"passed": True, "stage": "units", "source_index_sha256": index_sha,\n')
    extra=added([
        'report.update({"overlap_unit_count": overlap_report["case_count"], "overlap_units_passed": True,',
        '               "overlap_report": "overlap-report.json",',
        '               "combined_unit_count": len(rows)+cache_report["case_count"]+overlap_report["case_count"]})',
        'if report["combined_unit_count"] != recipe["expected_combined_unit_count"]:',
        '    raise RuntimeError("combined v7-unit registry count mismatch")'])
    unit=replace_once(unit,'    (out / "report.json").write_text(',extra+'    (out / "report.json").write_text(')
    (HERE/'unit_stage.py').write_text(unit)
    if strip_added(unit)!=(OLD/'unit_stage.py').read_text(): raise RuntimeError('old 179-unit body changed')
    run=(OLD/'run_once.py').read_text()
    oldkeys='("units_receipt", "certificate_receipt", "certificate_payload")'
    newkeys='("units_receipt", "new_units_receipt", "new_units_report", "new_overlap_report", "new_cache_report", "certificate_receipt", "certificate_payload")'
    run=replace_once(run,oldkeys,newkeys)
    (HERE/'run_once.py').write_text(run)
    if run.replace(newkeys,oldkeys,1)!=(OLD/'run_once.py').read_text(): raise RuntimeError('wrapper reverse bytes differ')
    for name in ['interval.py','interval_uncached.py','cache_units.py','instrumentation.py']:
        before,after=(OLD/name).read_text(),(HERE/name).read_text()
        if before!=after: raise RuntimeError('unchanged mathematical/cache source differs')
        proofs.append({'file':name,'byte_equal':True,'AST_equal':ast.dump(ast.parse(before),include_attributes=False)==ast.dump(ast.parse(after),include_attributes=False)})
    output_by_name={Path(entry['path']).name:entry for entry in failed['outputs']}
    recipe=dict(oldrecipe)
    recipe.update(status='HELD fixed 2-sigma series boundary; fresh units then certificate then independent replay require separate exact releases',
        source_revision='v7 fixed 2-sigma method/root boundary only; K32, enclosures, positivity, DFS and resource caps unchanged',
        admitted_stages_after_future_exact_release=['units','certificate','replay'],
        series_interface_sigma_multiple=2,expected_overlap_unit_count=58,expected_combined_unit_count=237,
        instrumentation_only=False,replay_admitted_by_this_candidate=True,
        replay_requires_actual_current_producer_pass=True,new_overlap_units_required=True,
        cached_v6_timeout_receipt=pin(RUN/'receipt.json'),cached_v6_source_index_sha256=sha(OLD/'source-index.json'),
        cached_v6_recipe_sha256=sha(OLD/'recipe.json'),cached_v6_progress=output_by_name['progress.json'],
        cached_v6_stderr=output_by_name['stderr.log'],cached_v6_command=output_by_name['command.json'],
        cached_v6_partial_certificate_metadata=dict(output_by_name['certificate.jsonl'],role='large_payload_metadata_only'),
        reviewed_Taylor_overlap_index=pin(PENCIL/'index.json'),reviewed_Taylor_overlap_note=pin(PENCIL/'ASSESSMENT.md'))
    # Receipt membership uses exact original metadata, so keep the role tag in a
    # separate inventory, not inside the old receipt's payload-pin equality.
    recipe['cached_v6_partial_certificate_metadata']=output_by_name['certificate.jsonl']
    recipe['protected_inputs']=sorted(protected.values(),key=lambda entry:entry['path'])
    exemption={'status','source_revision','admitted_stages_after_future_exact_release','expected_combined_unit_count',
               'instrumentation_only','replay_admitted_by_this_candidate','protected_inputs'}
    if any(recipe[key]!=value for key,value in oldrecipe.items() if key not in exemption):
        raise RuntimeError('original recipe setting changed outside the declared interface/admission scope')
    save(HERE/'recipe.json',recipe)
    schemas={}
    for stage in ['units','certificate','replay']:
        auth={'allow_execution':False,'stage':stage,'source_index_sha256':'exact frozen v7 index',
              'recipe_sha256':'exact frozen v7 recipe','output':'fresh immediate attempts child',
              'prior_timeout_receipt':recipe['cached_v4_timeout_receipt'],
              'prior_v5_timeout_receipt':recipe['cached_v5_timeout_receipt'],
              'prior_v6_timeout_receipt':recipe['cached_v6_timeout_receipt']}
        if stage!='units':
            auth['units_receipt']=recipe['cached_v3_units_receipt']
            for key in ['new_units_receipt','new_units_report','new_overlap_report','new_cache_report']:
                auth[key]={'path':'actual successful fresh v7 units output','bytes':'actual size','sha256':'actual sha256'}
        if stage=='replay':
            for key in ['certificate_receipt','certificate_payload']:
                auth[key]={'path':'actual successful fresh v7 certificate output','bytes':'actual size','sha256':'actual sha256'}
        schemas[stage]=auth
    save(HERE/'authorization-schema.json',schemas)
    diffs={}
    for name in changes.keys()|{'admission.py','unit_stage.py','run_once.py','recipe.json'}:
        target=HERE/(name.replace('.','-')+'-v6-v7.diff')
        target.write_text(''.join(difflib.unified_diff((OLD/name).read_text().splitlines(True),(HERE/name).read_text().splitlines(True),fromfile='v6/'+name,tofile='v7/'+name)))
        diffs[name]=pin(target)
    save(HERE/'source-equalities.json',{'proofs':proofs,'admission_reverse_byte_AST_equal':True,
        'unit_stage_addition_erased_byte_AST_equal':True,'wrapper_new_dynamic_keys_reverse_byte_AST_equal':True,
        'original_144_units_and_35_cache_units_unchanged':True,'new_overlap_units':58,'combined_units':237,
        'formula_and_interval_arithmetic_bodies_unchanged':True,'domain_union_unchanged':True,
        'caps_unchanged':True,'science_dispatch_boundary_changed':True,'diffs':diffs,'no_candidate_imports_or_numerical_execution':True})
    save(HERE/'unit-registry.json',{'expected_old_arithmetic':144,'expected_old_cache':35,'expected_new_overlap':58,
        'expected_combined':237,'new_categories':{'remainder_pencil_identities':3,'remainder_dyadic_bounds':3,
        'two_profile_dimensional_identities':6,'two_profile_exact_domain_union':6,
        'two_formula_modules_two_profiles_seven_dispatch_controls':28,'four_formula_actual_point_overlap':12},
        'selected_point_controls':'R=2sigma,T=0 or sigma, two profiles; not a domain certificate',
        'dispatch_controls':'isolated restored zero-recorders; selected actual formula overlap separately checked',
        'execution_performed':False})
    progress=load(RUN/'progress.json')
    save(HERE/'v6-cap-history.json',{'receipt':pin(RUN/'receipt.json'),'progress':pin(RUN/'progress.json'),
        'partial_tree_metadata':pin(RUN/'certificate.jsonl'),'partial_tree_decoded_or_copied':False,
        'completed':False,'passed':False,'nodes':progress['nodes'],'leaves':progress['leaves'],'pending':progress['pending'],
        'root_counters':progress['root_counters'],'most_recent_node':progress['most_recent_node'],
        'regional_completion_or_coverage_fraction_claimed':False,'v7_does_not_resume_v6':True})
    for path in HERE.rglob('*.py'): ast.parse(path.read_text(),filename=str(path))
    for entry in recipe['protected_inputs']:
        if sha(entry['path'])!=entry['sha256'] or Path(entry['path']).stat().st_size!=entry['bytes']:
            raise RuntimeError('final protected input differs')
    save(HERE/'source-preparation-receipt.json',{'source_only':True,'passed':True,'frozen_owner_sources_unchanged':True,
        'candidate_imports':False,'interval_arithmetic':False,'units_executed':False,'domain_or_replay_execution':False,
        'partial_payload_decode_or_copy':False,'metadata_and_AST_only':True,'protected_input_count':len(protected),
        'preparation_inspection_history':[{'classification':'mechanical read-only path inspection failure',
            'paths':['gaussian.py','certificate.py'],'meaning':'nonexistent old filenames; no candidate import or invocation'}]})
    files=sorted(path for path in HERE.rglob('*') if path.is_file() and path != HERE/'source-index.json')
    save(HERE/'source-index.json',{'source_only':True,'execution_authorized':False,'file_count':len(files),'files':[pin(path) for path in files]})
    print(json.dumps({'source_index':sha(HERE/'source-index.json'),'recipe':sha(HERE/'recipe.json'),
        'files':len(files),'protected_inputs':len(protected),'units':237,'executed':False},sort_keys=True))


if __name__=='__main__':
    main()
