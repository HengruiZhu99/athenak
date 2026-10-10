"""UNEXECUTED metadata-only root review/preparation; never imports candidate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

HERE=Path(__file__).resolve().parent
OWNER=HERE.parent/'continuum/exact-rational-backend-source001-held-20261009'
INDEX='c703481f9c8da663baf5c894aee2ff285d0e427572dbe990a4467874b9d5e4b5'
RECIPE='23e9e2d1ff335a50b4526d230aae974fb9c4b85de00d2b52da21238629c2fd38'


def require(value,message):
    if not value:raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()


def pin(path):
    path=Path(path).resolve()
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))


def load(path):
    def reject(value):raise ValueError('nonfinite JSON: '+value)
    return json.loads(Path(path).read_text(),parse_constant=reject)


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')


def check(entry):
    require(pin(entry['path'])=={key:entry[key] for key in ('path','bytes','sha256')},'pin drift: '+entry['path'])


def main():
    output=HERE/'preparation001';output.mkdir(exist_ok=False)
    receipt=dict(passed=False,source_only_metadata=True,candidate_import_or_registry_or_compile=False)
    before=[]
    try:
        write(output/'argv.json',dict(argv=sys.argv,python=pin(sys.executable),
             flags=dict(isolated=sys.flags.isolated,optimize=sys.flags.optimize,dont_write_bytecode=sys.dont_write_bytecode)))
        ap=argparse.ArgumentParser()
        ap.add_argument('--scripts-index-sha256',required=True)
        ap.add_argument('--root-source-math-reviewed',action='store_true',required=True)
        ap.add_argument('--review',required=True,type=Path)
        ap.add_argument('--review-index-sha256',required=True)
        ap.add_argument('--review-receipt-sha256',required=True)
        args=ap.parse_args()
        require(sys.flags.isolated==1 and sys.dont_write_bytecode and sys.flags.optimize==0
                and os.environ.get('PYTHONOPTIMIZE')=='0','metadata requires -I -B optimize0')
        require(sha(HERE/'source-index.json')==args.scripts_index_sha256,'root source index hash')
        require(sha(OWNER/'source-index.json')==INDEX and sha(OWNER/'recipe.json')==RECIPE,'exact held candidate identity')
        recipe=load(OWNER/'recipe.json')
        require(str(Path(sys.executable).resolve())==recipe['python']['path'],'root metadata interpreter identity')
        proof=load(OWNER/'source-equalities.json')
        require(proof.get('all_three_byte_identical') is True and proof.get('base_fraction_module_AST_identical') is True,
                'reviewed WIP source equality proof')
        for name,digest in proof['original_WIP_hashes'].items():
            require(sha(OWNER/name)==digest,'copied WIP source drift')
        review=args.review.resolve()
        require(sha(review/'index.json')==args.review_index_sha256 and
                sha(review/'receipt.json')==args.review_receipt_sha256,'exact independent review identities')
        independent=load(review/'receipt.json')
        require((independent.get('passed') is True or independent.get('passed_source_review') is True)
                and independent.get('reviewed_source_index_sha256')==INDEX,'review must pass/bind this exact82-unit driver')
        entries=load(review/'index.json')['files']
        if isinstance(entries,dict):entries=[dict(v,path=v.get('path',k)) for k,v in entries.items()]
        review_pins=[]
        for entry in entries:
            path=Path(entry['path']);path=(path if path.is_absolute() else review/path).resolve()
            require(path.is_relative_to(review),'reviewed source path escapes capsule')
            item=dict(entry,path=str(path));check(item);review_pins.append(item)
        values=(load(OWNER/'source-index.json')['files']+load(OWNER/'external-pins.json')+review_pins+
                load(HERE/'source-index.json')['files'])
        values += [pin(OWNER/'source-index.json'),pin(HERE/'source-index.json'),pin(review/'index.json'),pin(review/'receipt.json')]
        before=sorted({item['path']:{key:item[key] for key in ('path','bytes','sha256')} for item in values}.values(),key=lambda item:item['path'])
        for item in before:check(item)
        write(output/'pins-before.json',before)
        root_review=dict(passed=True,root_source_math_reviewed=True,reviewed_source_index_sha256=INDEX,
            recipe_sha256=RECIPE,independent_index=pin(review/'index.json'),independent_receipt=pin(review/'receipt.json'),
            fixed_cases=82,closed_dependencies_required_before_probe=True,root_group_cap_seconds=120,
            source_preparation_only=True,candidate_execution=False)
        write(HERE/'source-review001.json',root_review)
        auth=dict(execution_released=True,scope=recipe['scope'],source_index_sha256=INDEX,recipe_sha256=RECIPE,
            attempt=str(OWNER/'attempts/units001'),source_review_passed=True,
            review_receipt=pin(review/'receipt.json'),review_index=pin(review/'index.json'),
            root_source_review=pin(HERE/'source-review001.json'),root_process_group_cap_seconds=120,
            no_native_RWM_or_gauge_adoption=True)
        write(HERE/'units-authorization.json',auth)
        release=dict(owner=str(OWNER),source_index_sha256=INDEX,recipe_sha256=RECIPE,
            child_attempt=str(OWNER/'attempts/units001'),outer_attempt=str(HERE/'units-invocation001'),
            protected_pins=before+[pin(HERE/'source-review001.json'),pin(HERE/'units-authorization.json')],
            authorization=pin(HERE/'units-authorization.json'),root_source_review=pin(HERE/'source-review001.json'),
            fixed_cases=82,root_process_group_cap_seconds=120,per_command_cap_seconds=60)
        write(HERE/'units-release.json',release)
        for item in before:check(item)
        receipt.update(passed=True,inputs_unchanged=True,protected_count=len(before),
            release=pin(HERE/'units-release.json'),authorization=pin(HERE/'units-authorization.json'))
    except BaseException as exc:
        receipt.update(error=repr(exc),traceback=traceback.format_exc())
    finally:
        failures=[]
        for item in before:
            try:check(item)
            except BaseException as exc:failures.append(dict(path=item['path'],error=repr(exc)))
        receipt['post_pin_failures']=failures
        if failures:receipt.update(passed=False,inputs_unchanged=False)
        write(output/'receipt.json',receipt)
    print(json.dumps(receipt,sort_keys=True))
    return 0 if receipt['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
