"""One-shot stdlib source/metadata freeze; no registry, compiler or arithmetic run."""
from pathlib import Path
import ast
import difflib
import hashlib
import json
import shutil
import struct
import sys

P=Path(__file__).resolve().parent
BASE=P.parents[2]/'build-layer-research'
WIP=BASE/'continuum/exact-rational-backend-WIP001'
OLD=BASE/'continuum/exact-signed-product-sum-source002-held-20261009'
REVIEW=BASE/'continuum/exact-rational-backend-independent-source-review-20261009'
WHOLE=BASE/'boundary/reference-wave-map-complete-rational-rows-independent-pencil-20261009'


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


def load(path):return json.loads(Path(path).read_text())


def save(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def main():
    require(sys.flags.isolated and sys.dont_write_bytecode and sys.flags.optimize==0,'metadata preparation -I -B optimize0')
    require(not(P/'source-index.json').exists(),'one-shot freeze already exists')
    expected={'exact_dyadic_ratio.hpp':'83b806893e9ed2db89a4f32dc77eba80a39d16a40774e65feb99c85102a81a2c',
              'probe.cpp':'8f75119efa5c8346efad075363bf65a9021053ab87a90c49f8d836988208e4e7',
              'fraction_units.py':'b8dc7d76efc08802060317e851c68caba3689dba2960a4737257c5474e2d4e94'}
    external={}
    def protect(path):
        item=pin(path);external[item['path']]=item;return item
    for name,value in expected.items():
        require(sha(WIP/name)==value,'reviewed WIP changed')
        shutil.copyfile(WIP/name,P/name);protect(WIP/name)
        require(sha(P/name)==value,'reviewed source copy changed')
    require(sha(REVIEW/'index.json')=='b13762f8bf0105c70852b3cca53ad9f91eb666d2a42f24851a6cd016115a660c',
            'backend review index drift')
    require(sha(REVIEW/'receipt.json')=='752f644df4203d6d1e5c0df6b846c29ada61ccc164b10854b9c06d4278012d69',
            'backend review receipt drift')
    for item in load(OLD/'external-pins.json'):
        require(sha(item['path'])==item['sha256'] and Path(item['path']).stat().st_size==item['bytes'],'old runtime/header baseline drift')
        protect(item['path'])
    oldreceipt=load(OLD/'attempts/units001/receipt.json')
    require(oldreceipt['completed'] is True and oldreceipt['passed'] is True and oldreceipt['inputs_unchanged'] is True
            and oldreceipt['fixed_cases']==70,'old actual standalone gate not PASS')
    for mode in ('release','debug'):
        dependency_pin=oldreceipt[mode+'_dependencies']
        require(sha(dependency_pin['path'])==dependency_pin['sha256'],'old actual compiler dependency record drift')
        protect(dependency_pin['path'])
        for item in load(dependency_pin['path']):
            require(sha(item['path'])==item['sha256'],'old actual compiler dependency drift')
            protect(item['path'])
    for path in [OLD/'source-index.json',OLD/'external-pins.json',OLD/'recipe.json',OLD/'run_gate.py',
                 OLD/'attempts/units001/receipt.json',OLD/'attempts/units001/probe-release.d',
                 OLD/'attempts/units001/probe-debug.d']:
        protect(path)
    for prefix in (REVIEW,WHOLE):
        for path in sorted(prefix.rglob('*')):
            if path.is_file():protect(path)
    # Inspect the completed OLD debug executable's load commands as metadata;
    # do not run it. Bind its actual CLT ASan dylib where a file is available.
    oldexe=Path(oldreceipt['debug_executable']['path']);protect(oldexe)
    blob=oldexe.read_bytes()
    require(blob[:4]==b'\xcf\xfa\xed\xfe','old debug binary is not little-endian MachO64')
    count=struct.unpack_from('<I',blob,16)[0]
    offset,rpaths,libraries=32,[],[]
    for _ in range(count):
        cmd,size=struct.unpack_from('<II',blob,offset)
        require(size>=8 and offset+size<=len(blob),'old load-command bounds')
        if cmd in (0xc,0x80000018,0x8000001f,0x80000023,0x8000001c):
            nameoff=struct.unpack_from('<I',blob,offset+8)[0]
            require(12<=nameoff<size,'old load-command name bounds')
            name=blob[offset+nameoff:offset+size].split(b'\0',1)[0].decode('utf-8')
            (rpaths if cmd==0x8000001c else libraries).append(name)
        offset+=size
    actual_libraries=[]
    for name in libraries:
        candidates=[Path(name)] if name.startswith('/') else [Path(root)/name[7:] for root in rpaths] if name.startswith('@rpath/') else []
        present=[path for path in candidates if path.is_file()]
        actual_libraries.append(dict(load_name=name,available_file_pins=[protect(path) for path in present],
            unavailable_file_semantics='system shared-cache libraries are disclosed metadata, not claimed file-pinned' if not present else None))
    save(P/'old-compiler-runtime-context.json',dict(source=pin(oldexe),rpaths=rpaths,libraries=actual_libraries,
         old_executable_executed=False,old_compiler_dependency_records=[oldreceipt[x+'_dependencies'] for x in ('release','debug')]))
    oldrecipe=load(OLD/'recipe.json')
    recipe=dict(scope='standalone-exact-rational-CPU-units82',source_only=True,execution_admitted=False,
        fixed_cases=82,base_cases=74,carry_range_cases=8,full_exact_operand_comparisons=76,
        attempt_relative='attempts/units001',compiler=oldrecipe['compiler'],python=oldrecipe['python'],
        environment=oldrecipe['environment'],sanitized_environment=oldrecipe['sanitized_environment'],
        release_flags=oldrecipe['release_flags'],debug_flags=oldrecipe['debug_flags'],
        command_timeout_seconds=60,root_process_group_cap_seconds=120,
        local_sources=['exact_dyadic_ratio.hpp','probe.cpp','fraction_units.py','carry_range_units.py',
                       'gate_common.py','run_gate.py','oracle_driver.py','PLAN.md','authorization-schema.json',
                       'source-equalities.json','old-compiler-runtime-context.json','prepare_source.py'],
        external_inputs_file='external-pins.json',backend_source_review=protect(REVIEW/'receipt.json'),
        completed_signed_product_receipt=protect(OLD/'attempts/units001/receipt.json'),
        whole_row_pencil_index=protect(WHOLE/'index.json'),
        closed_dependency_contract='both -MD lists checked before any probe/oracle import; unknown header is a preserved failure',
        canonical_object_contract='only factory/operation-created Dyadics, no raw mutated fields or arbitrary exponent helper arguments',
        resource_contract='4096 limbs/131072 bits, normalized exponents within +/-131072; conservative failures permitted',
        range_contract='abs(exact quotient)<=maxfinite; strictly greater explicit exact_overflow even when IEEE would round maxfinite',
        zero_contract='exact zero +0; nonzero negative tiny result -0',no_gauge_or_native_adoption=True,
        no_scientific_execution_in_preparation=True)
    save(P/'recipe.json',recipe)
    save(P/'authorization-schema.json',dict(execution_released=False,scope=recipe['scope'],
        source_index_sha256='exact frozen source index',recipe_sha256='exact frozen recipe',
        attempt=str(P/'attempts/units001'),source_review_passed=False,
        review_receipt=dict(path='fresh independent full driver/source review receipt',bytes='actual size',sha256='actual digest'),
        review_index=dict(path='same immutable independent index',bytes='actual size',sha256='actual digest')))
    save(P/'source-equalities.json',dict(original_WIP_hashes=expected,
        all_three_byte_identical=True,base_fraction_module_AST_identical=True,
        base74_module_unchanged=True,new_carry_range_module_expected_cases=8,total_expected_cases=82,
        registry_generated=False,compile_or_execution=False))
    save(P/'external-pins.json',sorted(external.values(),key=lambda item:item['path']))
    (P/'runner-provenance.diff').write_text(''.join(difflib.unified_diff((OLD/'run_gate.py').read_text().splitlines(True),
        (P/'run_gate.py').read_text().splitlines(True),fromfile='audited70/run_gate.py',tofile='held82/run_gate.py')))
    for name in ['fraction_units.py','carry_range_units.py','gate_common.py','oracle_driver.py','run_gate.py','prepare_source.py']:
        ast.parse((P/name).read_text(),filename=str(P/name))
    require(ast.dump(ast.parse((P/'fraction_units.py').read_text()),include_attributes=False)==
            ast.dump(ast.parse((WIP/'fraction_units.py').read_text()),include_attributes=False),'base AST changed')
    for name,value in expected.items():require(sha(P/name)==value and sha(WIP/name)==value,'final WIP/copy drift')
    for item in external.values():require(sha(item['path'])==item['sha256'],'final baseline drift')
    save(P/'source-preparation-receipt.json',dict(passed=True,source_only=True,metadata_and_AST_only=True,
        candidate_imports=False,registry_generation=False,Fraction_evaluation=False,compiler_execution=False,
        external_unique_pins=len(external),WIP_sources_unchanged=True,
        mechanical_inspection_history='old root launch.py ENOENT; authoritative audited script is launch_units.py; no driver executed'))
    files=sorted(path for path in P.iterdir() if path.is_file())
    save(P/'source-index.json',dict(source_only=True,execution_admitted=False,files=[pin(path) for path in files],
        external_pins_file='external-pins.json',external_unique_pins=len(external),fixed_cases=82,
        preserved_base_cases=74,new_carry_range_cases=8,no_compile_import_registry_or_unit_execution=True))
    print(json.dumps(dict(index=sha(P/'source-index.json'),recipe=sha(P/'recipe.json'),files=len(files),external=len(external))))


if __name__=='__main__':main()
