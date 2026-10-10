"""Source/metadata preparation only: no numerical imports or candidate execution."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import textwrap

HERE = Path(__file__).resolve().parent
B = HERE.parents[1]
OWNER = B/'boundary/reference-wave-map-J0-bulk008-retained-independent-readback-v9-held-20261009'
FAILED = OWNER/'attempts/independent-radial_pair001'
ROOT = B/'wave-map-J0-bulk008-independent-v9-root-release-20261009'
ASSOCIATION = HERE.with_name('reference-wave-map-v9-radial-failure-independent-saved-review-20261009')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1 << 20),b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    assert Path(path).stat().st_size <= 1 << 20
    return json.loads(Path(path).read_text())


def pin(path):
    path = Path(path).absolute()
    return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def main():
    assert not (HERE/'source-index.json').exists()
    assert sha(OWNER/'source-index.json') == '0f276fc8cd813dee704134aac4d91e3f033457982db7fcb00ef445f4620e5241'
    assert sha(FAILED/'receipt.json') == 'bfab682d017693316e885f22bb00b13707aa6f211bf851198a54e05bf03dc4d1'
    assert sha(ASSOCIATION/'index.json') == '8c29a47842e325baa63ad6aafc8ad522ca33ca6eb327c0b867c5ff5b184baca3'
    original = (OWNER/'verify_retained.py').read_text()
    lines = original.splitlines(True)
    tree = ast.parse(original)
    names = ['mm','bilinear','action','jacobi_jet','modal_jet']
    functions = {}
    for name in names:
        matching = [node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name == name]
        assert len(matching) == 1
        node = matching[0]
        functions[name] = textwrap.dedent(''.join(lines[node.lineno-1:node.end_lineno]))
    matrix = original[original.index('    M = np.zeros((20, 20))'):original.index('    P = (np.eye(20)+M)/2')]
    point = original[original.index('        fieldbasis=np.zeros((8,3,nd))'):original.index('        dy[:,qidx]=derivative_arithmetic')]
    point = '\n'.join(line[4:] for line in point.rstrip('\n').split('\n'))+'\n'
    header = '"""Exact copied operand functions; modules supplied only after admission."""\nimport math\n\n'
    definition = 'def build_mass_operand(np,dgemm,weighting,radii,weights,refs,maps,angles,ell,ir):\n'
    setup = ('    nc,N,nd,rb = 8,8,64,.98\n    na = len(angles)\n'
        '    qidx = [0,1,2,8,12,16,18,7,11,15]\n    vidx = [3,4,5,9,13,17,19,6,10,14]\n'
        '    r = radii[ir]\n    ref = refs[ir]\n    c,cr = ref[9:11]\n')
    bodies = '\n'.join(textwrap.indent(functions[name],'    ') for name in names)+'\n'
    tail = ('    hy=action(H,y)\n    measure=weights[ir]/c\n'
        '    mass_sum=bilinear(y,hy,angles)+bilinear(u,u,angles)\n    return measure,mass_sum\n')
    operand = header+definition+setup+bodies+matrix+'\n'+point+tail
    (HERE/'operand_graph.py').write_text(operand)
    generated_tree = ast.parse(operand)
    checks = {}
    for name in names:
        new = next(node for node in ast.walk(generated_tree) if isinstance(node,ast.FunctionDef) and node.name == name)
        old = next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name == name)
        assert ast.dump(new,include_attributes=False) == ast.dump(old,include_attributes=False)
        checks[name] = dict(AST_equal=True,source_text_dedent_equal=True,
            source_sha256=hashlib.sha256(functions[name].encode()).hexdigest())
    assert matrix in operand and point in operand
    assert original.count('measure=weights[ir]/c') == 1
    assert original.count('bilinear(y,hy,angles)+bilinear(u,u,angles)') == 1
    write(HERE/'operand-source-proof.json',dict(original_verifier=pin(OWNER/'verify_retained.py'),
        copied_functions=checks,H_matrix_block_exact=True,modal_point_block_exact=True,
        H_action_rhs_identical=True,measure_rhs_identical=True,mass_sum_rhs_identical=True,
        only_E_accumulation_and_outer_product_removed=True,
        no_derivative_momentum_coefficient_or_source_table_actions=True))
    for name in ['tiny_normalization.py','fast_weighting.py']:
        shutil.copyfile(OWNER/name,HERE/name)
        assert sha(OWNER/name) == sha(HERE/name)
    old_recipe = load(OWNER/'recipe.json')
    case = old_recipe['cases']['radial_pair']
    data = Path(case['directory'])
    recipe = dict(source_only=True,execution_authorized=False,
        scope='Bounded saved outer E measure-times-mass-sum product only; no E accumulation, operator/SVD/query/generator',
        first_radius=609,last_radius=640,total_radii=769,N=8,rb=.98,expected_components=131072,
        example_limit=64,python='/Library/Developer/CommandLineTools/usr/bin/python3',
        resolved_python='/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3.9',
        numpy_root=str(B/'boundary/python-deps/numpy'),
        environment=dict(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',
            PYTHONDONTWRITEBYTECODE='1',PYTHONOPTIMIZE='0',PYTHONPATH=str(B/'boundary/python-deps')+':/Users/hz0693/Documents/Codex/2026-10-06/referenced-chatgpt-conversation-this-is-an/work/venv/lib/python3.9/site-packages'),
        attempt=str(HERE/'attempts/diagnostic001'),outer_attempt=str(HERE/'outer-invocation001'),
        operator_npz=str(data/'operator.npz'),input_map=str(data/'input/output.bin'),
        selected_npz_arrays=['source_coefficient_radii','radial_weights','angular_weights','source_reference_rows'],
        basis_data=old_recipe['context']['basis_data'],failed_receipt=str(FAILED/'receipt.json'),
        failed_progress=str(FAILED/'progress.json'),failed_trace=str(FAILED/'failure.txt'),
        future_command=['/Library/Developer/CommandLineTools/usr/bin/python3','-B','-s',str(HERE/'run_once.py'),
            '--authorization','ROOT_AUTHORIZATION.json','--authorization-sha256','ROOT_HASH'])
    write(HERE/'recipe.json',recipe)
    payload_registry = dict(preparation_decoded_any_operand=False,
        operator_npz=dict(**pin(data/'operator.npz'),access='lazy selected arrays only after release',
            arrays=[dict(name='source_coefficient_radii',shape=[769]),dict(name='radial_weights',shape=[768]),
                dict(name='angular_weights',shape=[288]),dict(name='source_reference_rows',shape=[769,17])]),
        input_map=dict(**pin(data/'input/output.bin'),access='read-only memmap indices609..640 only',
            dtype='<f8',shape=[769,288,8,3,50]),
        J0_L_layout=dict(**pin(recipe['basis_data']),access='channel_layouts[0] only'),
        forbidden_access=['E','Kweak','Kstrong','Gvolume','Fboundary','SATload','Jbulk','Jsat',
            'source_coefficients','source/output.txt','source/queries.txt','coefficient queries'],
        original_metadata_only_payloads=True)
    write(HERE/'operand-registry.json',payload_registry)
    pins = dict(load(FAILED/'pins-before.json'))
    pins.update(load(ROOT/'radial_pair-invocation001/before.json')['protected'])
    for folder in [FAILED,ROOT/'radial_pair-invocation001',ASSOCIATION]:
        for path in folder.rglob('*'):
            if path.is_file():
                pins[str(path)] = sha(path)
    pins[str(OWNER/'source-index.json')] = sha(OWNER/'source-index.json')
    for path,digest in pins.items():
        assert sha(path) == digest
    write(HERE/'input-pins.json',pins)
    write(HERE/'authorization-schema.json',dict(bounded_saved_mass_diagnostic_authorized=False,
        source_index_sha256='EXACT_HELD_INDEX',recipe_sha256='EXACT_RECIPE',
        diagnostic_source_sha256='EXACT_DIAGNOSTIC',wrapper_source_sha256='EXACT_WRAPPER',
        only_selected_operands_no_E_accumulation_no_SVD_no_queries=True,
        independent_review_receipt=dict(path='EXACT_REVIEW_RECEIPT',sha256='EXACT_SHA256')))
    for path in HERE.glob('*.py'):
        ast.parse(path.read_text())
    write(HERE/'preparation.json',dict(source_only=True,inputs_protected=len(pins),
        candidate_imported=False,numerical_imports=False,arrays_decoded=False,
        compiler_queries_or_targets_executed=False,AST_source_equality_only=True,
        original_v9_failed_receipt=pin(FAILED/'receipt.json'),independent_association=pin(ASSOCIATION/'index.json')))
    files = [pin(path) for path in sorted(HERE.iterdir()) if path.is_file() and path.name != 'source-index.json']
    write(HERE/'source-index.json',dict(source_only=True,execution_admitted=False,files=files,
        input_pins_file='input-pins.json',external_unique_pins=len(pins),original_v9_failure_preserved=True,
        expected_radius_rows=32,expected_product_components=131072,no_science_executed=True))
    print(json.dumps(dict(source_index=pin(HERE/'source-index.json'),recipe=pin(HERE/'recipe.json'),
        diagnostic=pin(HERE/'diagnose_mass.py'),wrapper=pin(HERE/'run_once.py'),operand_graph=pin(HERE/'operand_graph.py'))))


if __name__ == '__main__':
    main()
