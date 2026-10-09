"""Prepare fixed generic collector only; no collection/scientific execution."""
from pathlib import Path
import hashlib
import json
import shutil

P = Path(__file__).resolve().parent
ROOT = P.parents[1]
STAGE = P / 'inner-joint-principal-compiled-collector-held-20261009'
STAGE.mkdir(exist_ok=False)
TEMPLATE = ROOT / 'build-layer-research/boundary/derivative-pilots-inner-pencil-collector-held-20261009'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
if sha(TEMPLATE / 'collect_once.py') != '1bd04e3803377848c18ab4d06825b612bcbf32bf1b9fb3f8df2efcff2dcee741':
    raise ValueError('fixed generic collector differs')
for name in ('collect_once.py', 'prepare_metadata.py'):
    shutil.copyfile(TEMPLATE / name, STAGE / name)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


roots = [
 ('pencil-lineage', P/'inner-joint-principal-candidate-pencil-20261009', 'Original transcription and authoritative v2 pencil/erratum'),
 ('original-count-failure', P/'inner-joint-principal-gate-held-20261009', 'Unexecuted source proposal with preserved count defect'),
 ('count-corrected-v2', P/'inner-joint-principal-gate-v2-held-20261009', 'Source-only v2 with preserved optimization admission defect'),
 ('guard-corrected-v3-and-run', P/'inner-joint-principal-gate-v3-held-20261009', 'Exact v3 sources plus completed18/118/118 gate'),
 ('independent-pencil-review', P/'inner-joint-principal-candidate-independent-review-20261009', 'Independent corrected pencil source/math review'),
 ('independent-v2-admission-failure', P/'inner-joint-principal-gate-v2-independent-source-review-20261009', 'Formula PASS, optimization admission FAIL preserved'),
 ('independent-v3-guard-review', P/'inner-joint-principal-gate-v3-independent-guard-review-20261009', 'Independent narrow guard correction PASS'),
 ('root-release-and-review', ROOT/'build-layer-research/inner-joint-principal-v3-root-release-20261009', 'Exact root source review, authorization, completed outer launch'),
 ('independent-saved-readback', P/'inner-joint-principal-v3-independent-saved-readback-20261009', 'Independent saved118 matrix/row-wave arithmetic and provenance PASS'),
]
for tag, path, role in roots:
    if not path.is_dir():
        raise FileNotFoundError(path)
selected = []
for name in ('prepare_inner_gate_v2.py', 'prepare_inner_gate_v2_recipe.py',
             'prepare_inner_gate_v3_guards.py', 'prepare_inner_saved_review.py'):
    selected.append(dict(source=str(P/name), archive_relative='preparation-context/'+name,
                         role='Exact source-only preparation/capture script'))
child = P/'inner-joint-principal-gate-v3-held-20261009/attempts/gate001'
outer = ROOT/'build-layer-research/inner-joint-principal-v3-root-release-20261009/invocation001'
readback = P/'inner-joint-principal-v3-independent-saved-readback-20261009/attempt001'
gates = [
 dict(source=str(child/'receipt.json'), sha256='bf778caf62d6596261d5c15a7307b791bb39be321f0bc0bc834adfdeeef4d83b',
      expected=dict(completed=True, passed=True, returncode=0, inputs_unchanged=True, release_debug_stdout_byte_equal=True)),
 dict(source=str(outer/'receipt.json'), expected=dict(completed=True, accepted=True, returncode=0, source_drift=[])),
 dict(source=str(readback/'receipt.json'), sha256='263c17cbd8e27830b6815d6d6a18071127057f13c16f53b96175c6c163c7907b',
      expected=dict(passed=True, saved_only=True, cases=118, exact_records=18, commands=10, release_debug_byte_equal=True)),
 dict(source=str(child/'exact-scalar.stdout'), expected=dict(passed=True, exact_scalar_cases=18)),
]
metadata = {'inputs': [], 'scope': 'No external runtime-source copies exist in these nine completed roots; actual runtime/header files remain external metadata-only.'}
write(STAGE/'runtime-copy-metadata.json', metadata)
config = dict(
 scope='Completed coupled-inner constant-reference constrained20 principal gate and source/optimization/count failure lineage; no nonlinear/puncture/BH/evolution adoption',
 destination=str(ROOT/'docs/validation/hyperboloidal-inner-joint-principal-compiled-20261009'),
 production_reference_commit='27c19d20696ea6dd4704032c51dfd026218f64f2',
 roots=[dict(tag=tag, source=str(path), role=role) for tag,path,role in roots],
 selected_files=selected, metadata_only_external_files=[], completion_gates=gates,
 identical_inventory_pairs=[[str(child/'probe-release.stdout'),str(child/'probe-debug.stdout')]],
 metadata_only_runtime_copy_paths=[],
 dependency_manifests=[dict(source=str(child/('dependencies-'+mode+'.json')),keys=[]) for mode in ('release','debug')]+[
     dict(source=str(child/'recipe.json'),keys=['protected_inputs'])],
 README='''# Compiled coupled-inner principal checkpoint

This capsule preserves the original wrong A_t display, authoritative Lambda
correction, first source count defect, count-corrected optimization admission
FAIL, guard-only v3, independent reviews, root release and completed local gate.
The gate passed18 exact Fraction cases and118 actual complete20 tensor matrices
in both Release and ASanUBDebug. The two matrix stdout files are byte-identical.
All10 child commands returned0/empty stderr; root outer elapsed4.019781375s.

Independent saved-only arithmetic reproduces maximum matrix error5.346834e-13
and direct ell/H/V/X second-row wave identities <=6.659118e-13. Original
analyzer left-field/inverse checks and source/compile dependency records remain
separate exact evidence. The resulting finite-positive-lapse/chi characteristic
basis stays complete at the declared speed collisions. This is not uniform
puncture hyperbolicity, nonlinear nonflat-reference source validation, trumpet
formation or native/BH/evolution acceptance. The later required black-hole
test remains a wormhole-to-trumpet inner transition with Minkowski reference.

Every copied original is byte-exact UTF-8 <=1MiB. Executables/DWARF, arrays,
NPY/NPZ/JSONL, libraries/objects and larger files remain original metadata-only.
All external compiler/header/runtime dependencies are rehashed and represented
by metadata, never copied into Git. The generic collector is byte-identical
1bd04e3803377848c18ab4d06825b612bcbf32bf1b9fb3f8df2efcff2dcee741.
It contains assertion gates: root must launch isolated unoptimized Python
(-I -B, PYTHONOPTIMIZE=0, PYTHONDONTWRITEBYTECODE=1) with exact authorization.
The collector is one-shot and refuses an existing destination. No scientific
module or source is executed by collection; historical failures stay failed.
''')
write(STAGE/'scope-config.json', config)
(STAGE/'PLAN.md').write_text('''# Held one-shot compiled-inner compact collection

Reuse the established generic collector1bd04e byte-exactly. Only this fixed
scope-config/metadata changes. Nine complete, inactive roots retain all original
pencil/count/optimization failure lineages, reviews, actual18+118+118 records
and independent saved-only readback. No finite64 or live v4/v5/derivative tree
is included. The sole new destination is hyperboloidal-inner-joint-principal-
compiled-20261009 and must not exist before root-authorized collection.

Source-only metadata preparation may hash files and parse saved JSON; it may
not invoke collect_once.py, compiler/probe/scientific modules or any evolution.
Before preparation, all policy-eligible originals must be UTF-8 and free of
array/archive/binary magic. Raw matrix stdout is ordinary finite JSON <=1MiB.
Dependencies are represented by exact source/hash/size metadata only.

Collector assertions require root's isolated unoptimized -I -B invocation plus
PYTHONOPTIMIZE=0/PYTHONDONTWRITEBYTECODE=1, with exact source-index/recipe hashes
and destination. No recipe flag itself authorizes execution. All successes,
failures, empty logs, original bytes and before/after inventories remain exact.
''')
write(STAGE/'release-schema.json', dict(schema_only=True, one_shot_compact_collection_authorized=False,
      recipe_sha256='ROOT_REVIEWED_RECIPE_SHA', source_index_sha256='ROOT_REVIEWED_INDEX_SHA',
      destination=config['destination'], root_required_launch=['python','-I','-B','collect_once.py'],
      environment={'PYTHONOPTIMIZE':'0','PYTHONDONTWRITEBYTECODE':'1'}))
shutil.copyfile(__file__, STAGE/'prepare_scope_source.py')
print(json.dumps(dict(stage=str(STAGE), collector_sha256=sha(STAGE/'collect_once.py'),
                     scope_sha256=sha(STAGE/'scope-config.json'), no_collection_execution=True),indent=2))
