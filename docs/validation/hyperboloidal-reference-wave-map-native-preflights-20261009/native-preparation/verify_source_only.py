"""Read back held recipes/source/input invariants. No scientific kernel or compiler."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def finite_json(x):
    if isinstance(x,dict):
        for y in x.values(): finite_json(y)
    elif isinstance(x,list):
        for y in x: finite_json(y)
    elif isinstance(x,float):
        import math
        assert math.isfinite(x)

def main():
    public=ROOT/'src/z4c/hyperboloidal/cartesian_patch.hpp'
    base_exe=ROOT/'build-layer-spatial-norm-native/src/athena'
    all_production=subprocess.check_output(['git','ls-files','src','CMakeLists.txt'],cwd=ROOT,text=True).splitlines()
    prod_hashes={}
    for name in all_production:
        original=subprocess.check_output(['git','show','27c19d20696ea6dd4704032c51dfd026218f64f2:'+name],cwd=ROOT)
        assert (ROOT/name).read_bytes()==original,name
        prod_hashes[name]=sha(ROOT/name)
    assert len(prod_hashes)==365,len(prod_hashes)
    receipts={}
    for mode in ['wave-map','c0','wave-map-half']:
        p=HERE/'recipes'/mode/'recipe.json'; r=json.loads(p.read_text());finite_json(r)
        assert r['compilation_executed'] is False and r['native_execution_authorized'] is False
        assert r['prepare_script_sha256']==sha(HERE/'prepare_build.py')
        for name,digest in r['private_sources_sha256'].items(): assert sha(ROOT/name)==digest,name
        for name,digest in r['base_inputs_sha256'].items(): assert sha(ROOT/name)==digest,name
        for name,digest in r['base_source_sha256'].items(): assert sha(ROOT/name)==digest,name
        for name,item in r['base_link_inputs'].items(): assert sha(Path(name))==item['sha256'],name
        commands=r['compile_commands'];assert len(commands)==6
        assert len(set(row['original_source'] for row in commands))==6
        for row in commands:
            assert '-include' not in row['command']
            assert not any('spatial-norm-family' in x for x in row['command'])
            assert '-O3' in row['command'] and '-DNDEBUG' in row['command']
            assert '-std=c++17' in row['command']
            assert not Path(row['output']).exists()
        cart=HERE/'recipes'/mode/'include/z4c/hyperboloidal/cartesian_patch.hpp'
        source=cart.read_text(); psrc=public.read_text()
        if mode=='c0': assert source==psrc
        else:
            assert source.count('|| !ResearchNativeWaveMapGauge(p,u,xyz,gauge_rhs))')==1
            assert '|| !AssembleGaugeInterior' not in source
            assert source.count('AddMeshUpwindAdvectionWithVelocity<3>')==1
            # Everything from C0 residual subtraction through all diagnostics
            # remains public source, including the existing ghost/projection logic.
            tail='      // Subtract only the analytic Minkowski RHS\'s floating-point residual.'
            assert source[source.index(tail):]==psrc[psrc.index(tail):]
        if mode=='wave-map-half':
            private=HERE/'recipes'/mode/'z4c_newdt.cpp'
            change='    // Private timestep-only control: halve BOTH admitted native caps.\n    dtnew *= Real(0.5);\n'
            assert private.read_text().count(change)==1
            assert private.read_text().replace(change,'')==(ROOT/'src/z4c/z4c_newdt.cpp').read_text()
        assert not Path(r['executable']).exists()
        receipts[mode]={'recipe_sha256':sha(p),'cartesian_overlay_sha256':sha(cart),
                        'compile_sources':[row['original_source'] for row in commands]}
    helper=(HERE/'include/native_wave_map.hpp').read_text()
    assert helper.count('rwm::Assemble(parts,p.omega,rhs)')==1
    assert '#define InteriorLayerGauge' not in helper
    assert '#define AssembleGaugeInterior' not in helper
    rows=json.loads((HERE/'inputs/index.json').read_text())['inputs'];assert len(rows)==17
    for row in rows:
        p=HERE/row['path']; assert sha(p)==row['sha256']
        assert row['native_run_authorized'] is False
    # Analytic source-grid admission, not a run of the planner/kernel.
    grids={}
    for n in [16,24,32]:
        h=2.2/n; first=-1.1+(0.5-3)*h; allocated=n+6
        low=first+3*h;high=first+(allocated-1-3)*h
        assert low<=-1 and high>=1
        grids[str(n)]={'h':h,'first':first,'allocated':allocated,
                       'physical_first_center':low,'physical_last_center':high}
    out={'passed_source_only_readback':True,'scientific_kernel_compiler_and_native_calls':0,
         'production_365_unchanged_from_implementation':prod_hashes,
         'recipes':receipts,'fixed_inputs_verified':len(rows),'grids':grids,
         'base_executable_sha256':sha(base_exe),
         'source_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         'native_execution_held':True}
    (HERE/'source-only-readback.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print('PASS 365 production files, 3 held six-TU recipes, 17 inputs; no compilation/evolution')

if __name__=='__main__': main()
