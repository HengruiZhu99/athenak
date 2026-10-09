#!/usr/bin/env python3
"""Source/metadata preparation only; no numerical imports/compile/queries."""
from pathlib import Path
import ast
import difflib
import hashlib
import json
import shutil


ROOT=Path('/Users/hz0693/research/hyperboloidal')
HERE=Path(__file__).resolve().parent
OLD=ROOT/'build-layer-research/boundary/inner-joint-nonlinear-helper-source003-held-20261009'
ASSOC=ROOT/'build-layer-research/boundary/inner-source003-failure-saved-readback-20261009'


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def pin(p):return {'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def diff(name,a,b):
    (HERE/name).write_text(''.join(difflib.unified_diff(a.splitlines(True),b.splitlines(True),fromfile='frozen-source003',tofile='fresh-outer-source001')))


def main():
    assert not (HERE/'inputs').exists() and not (HERE/'source-index.json').exists()
    protected=read(OLD/'input-pins.json')
    protected += [pin(OLD/n) for n in ['inner_gauge.hpp','probe.cpp','oracle.py','run_once.py','recipe.json','source-index.json']]
    protected += [pin(OLD/'attempts/Release001'/n) for n in ['receipt.json','oracle-report.json']]
    protected += [pin(ASSOC/n) for n in ['index.json','review-receipt.json','PENCIL-ONLY-OUTER-CORRECTION.md','attempt001/summary.json','attempt001/receipt.json','outer_identity/attempt001/summary.json','outer_identity/attempt001/receipt.json']]
    for p in protected:assert sha(Path(p['path']))==p['sha256']
    shutil.copytree(OLD/'inputs',HERE/'inputs')
    (HERE/'history').mkdir(exist_ok=False)
    for name in ['inner_gauge.hpp','probe.cpp','oracle.py','run_once.py','recipe.json','source-index.json']:
        shutil.copyfile(OLD/name,HERE/'history'/name)
    shutil.copyfile(OLD/'inputs/reference_wave_map.hpp',HERE/'history/reference_wave_map.hpp')
    legacy=(OLD/'inputs/reference_wave_map.hpp').read_text()
    oldsig='hyp::GaugeRHSParts<T> Gauge('
    assert legacy.count(oldsig)==1
    renamed=legacy.replace(oldsig,'hyp::GaugeRHSParts<T> LegacyGauge(',1)
    assert renamed.replace('hyp::GaugeRHSParts<T> LegacyGauge(',oldsig,1)==legacy
    (HERE/'inputs/reference_wave_map_legacy.hpp').write_text(renamed)
    diff('legacy-name-only.diff',legacy,renamed)
    shutil.copyfile(HERE/'source-template/reference_wave_map.hpp',HERE/'inputs/reference_wave_map.hpp')
    original=(OLD/'inner_gauge.hpp').read_text()
    start=original.index('// A scalar adapter is explicit:')
    end=original.index('template<class T> T Coefficient(')
    arithmetic=original[start:end]
    header='''// Exact extracted source003 arithmetic bodies; CPU fixed-field duals only.
#ifndef RESEARCH_INNER_ARITHMETIC_TRAITS_HPP_
#define RESEARCH_INNER_ARITHMETIC_TRAITS_HPP_
#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
namespace inner {
'''+arithmetic+'\n} // namespace inner\n#endif\n'
    (HERE/'inputs/arithmetic_traits.hpp').write_text(header)
    rebound=original[:start]+original[end:]
    (HERE/'inner_gauge.hpp').write_text(rebound)
    diff('arithmetic-extraction.diff',original,rebound)
    assert arithmetic in header
    assert original[end:]==rebound[rebound.index('template<class T> T Coefficient('):]
    probe=(OLD/'probe.cpp').read_text()
    rebased=probe.replace('const auto qb=rwm::Gauge(p,u,rwm::ReferenceConnection(p,x));','const auto qb=rwm::LegacyGauge(p,u,rwm::ReferenceConnection(p,x));')
    assert rebased!=probe
    rebased=rebased.replace(':rwm::Gauge(p,u,rwm::ReferenceConnection(p,x));',':rwm::LegacyGauge(p,u,rwm::ReferenceConnection(p,x));')
    anchor='std::cout<<",\\\"input\\\":";Pack(u);'
    assert rebased.count(anchor)==1
    replacement='std::cout<<",\\\"legacy_near\\\":"<<(rwm::UsesLegacyNear(p,u)?"true":"false");'+anchor
    rebased=rebased.replace(anchor,replacement)
    (HERE/'probe.cpp').write_text(rebased)
    diff('probe-baseline-contract.diff',probe,rebased)
    # The scientific registry and high-contrast state constructors are exact.
    for a,b in [('hyp::Z4cJet<double> State(', 'template<class T>void EmitGauge('),
                ('void MainGrid(', 'void Duals('), ('void Duals(', 'void Coefficients('),
                ('void Coefficients(', 'void CoreWitness('),('void CoreWitness(', 'void Invalid('),
                ('void Invalid(', 'void Nonrepresentable(')]:
        assert probe[probe.index(a):probe.index(b)]==rebased[rebased.index(a):rebased.index(b)]
    oracle=(OLD/'oracle.py').read_text()
    block='''                if row['W']==1:
                    require(row['outer_value_bitwise'],str(label)+' W1 native bit mismatch')
                    require(all(float(x[component]).hex()==float(y[component]).hex() for x,y in zip(row['parts'],row['baseline_parts']) for component in range(2)),str(label)+' W1 full dual bit mismatch')
                    require(row['arithmetic']['coefficient_calls']==0,str(label)+' W1 evaluated inner k')
'''
    replacement='''                if row['W']==1:
                    # NEW arithmetic identity, explicitly not the frozen outer
                    # bitwise contract for far states. Original baseline parts
                    # and old outer_value_bitwise remain recorded in every row.
                    uref=field(row['reference']);live=field(row['input'])
                    near=all(v.v>=r.v/2 and v.v<=2*r.v for v,r in
                             ((live['alpha'],uref['alpha']),(live['chi'],uref['chi'])))
                    require(row['legacy_near']==near,str(label)+' route mismatch')
                    if near:
                        require(row['outer_value_bitwise'],str(label)+' legacy-near W1 native bit mismatch')
                        require(all(float(x[component]).hex()==float(y[component]).hex() for x,y in zip(row['parts'],row['baseline_parts']) for component in range(2)),str(label)+' legacy-near W1 full dual bit mismatch')
                    # Every far state still faces all unchanged MP parts/RHS
                    # and dual thresholds above; old bit equality is metadata.
                    require(row['arithmetic']['coefficient_calls']==0,str(label)+' W1 evaluated inner k')
'''
    assert oracle.count(block)==1
    changed=oracle.replace(block,replacement).replace("auth.get('local_nonlinear_helper_execution_admitted')","auth.get('outer_arithmetic_local_execution_admitted')")
    (HERE/'oracle.py').write_text(changed)
    diff('oracle-explicit-outer-contract.diff',oracle,changed)
    runner=(OLD/'run_once.py').read_text()
    runner_new=runner.replace("auth.get('local_nonlinear_helper_execution_admitted')","auth.get('outer_arithmetic_local_execution_admitted')")
    (HERE/'run_once.py').write_text(runner_new)
    diff('runner-admission-only.diff',runner,runner_new)
    recipe=read(OLD/'recipe.json')
    def rebind(v):
        if isinstance(v,str):return v.replace(str(OLD),str(HERE))
        if isinstance(v,list):return [rebind(x) for x in v]
        if isinstance(v,dict):return {k:rebind(x) for k,x in v.items()}
        return v
    recipe=rebind(recipe)
    recipe.update(scope='NEW outer RWM arithmetic identity + unchanged source003 inner equations; finite-Omega point gate only',
                  correction_scope='legacy-name-only extraction, exact003 arithmetic extraction, complete far RWM groups, explicit revised outer comparison contract',
                  source003_failed_receipt=pin(OLD/'attempts/Release001/receipt.json'),
                  source003_failed_report=pin(OLD/'attempts/Release001/oracle-report.json'),
                  saved_failure_mapping_index=pin(ASSOC/'index.json'),
                  old_outer_contract_preserved_as_history=True,
                  new_outer_contract={'near':'all8 split parts exact legacy values+duals at W1',
                                      'far':'same real equations/new arithmetic, old baseline recorded; unchanged MP thresholds'},
                  arithmetic_extraction='exact original003 Number/Audit/Product/Near/field-difference bodies in inputs/arithmetic_traits.hpp',
                  metric_contrast_limit='joint alpha/chi branch only; arbitrary metric/gradient extreme conditioning not certified',
                  additive_witnesses_not_executed_or_appended=True,execution_admitted=False,
                  no_global_stability_regular_scri_or_native_adoption_claim=True)
    # thresholds and ALL fixed query modes/counts are byte-for-byte semantic
    # values from003. The old bitwise threshold is retained as history above;
    # the explicit new contract cannot be confused with the old actual failure.
    assert recipe['thresholds']==read(OLD/'recipe.json')['thresholds']
    assert recipe['expected_record_counts']==read(OLD/'recipe.json')['expected_record_counts']
    write(HERE/'recipe.json',recipe)
    write(HERE/'authorization-schema.json',{'outer_arithmetic_local_execution_admitted':False,
        'recipe_sha256':'ROOT_REVIEW_REQUIRED','source_index_sha256':'ROOT_REVIEW_REQUIRED',
        'allowed_builds':[],'scope':'local point gates only, no matrix/native/evolution'})
    for p in [HERE/'probe.cpp',HERE/'oracle.py',HERE/'run_once.py']:
        if p.suffix=='.py':ast.parse(p.read_text(),filename=str(p))
    write(HERE/'identity-report-source-only.json',{
        'source_only':True,'legacy_restores_original_by_single_name_reversal':True,
        'arithmetic_block_exact003':True,'inner_Coefficient_and_Gauge_suffix_exact003':True,
        'original15740_fixed_cases_and_counts_unchanged':True,'original_MP_targets_and_numeric_thresholds_unchanged':True,
        'scientific_import_compile_or_query_executed':False,
        'original_frozen_RWM':pin(OLD/'inputs/reference_wave_map.hpp'),
        'preserved_RWM':pin(HERE/'history/reference_wave_map.hpp'),
        'new_legacy_name_only':pin(HERE/'inputs/reference_wave_map_legacy.hpp'),
        'arithmetic_traits':pin(HERE/'inputs/arithmetic_traits.hpp'),
        'original_source003_helper':pin(OLD/'inner_gauge.hpp'),
        'source003_actual_failed':True,'legacy_far_accuracy_accepted':False})
    # Keep original source/runtime dependencies protected in addition to every
    # fresh source. No scientific payload is copied into this source suite.
    inputs={p['path']:p for p in protected}
    for p in HERE.rglob('*'):
        if p.is_file() and p.name not in ('input-pins.json','source-index.json'):
            inputs[str(p.resolve())]=pin(p.resolve())
    write(HERE/'input-pins.json',[inputs[k] for k in sorted(inputs)])
    locals=[p for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='source-index.json']
    write(HERE/'source-index.json',{'source_only':True,'execution_admitted':False,
        'scope':recipe['scope'],'files':[pin(p) for p in locals],'protected_input_count':len(inputs),
        'prior_source003_failed':True,'original_supplement_v1_ineligible':True})
    for p in protected:assert sha(Path(p['path']))==p['sha256']
    print(json.dumps({'source_index':pin(HERE/'source-index.json'),'protected':len(inputs),
                      'no_science_executed':True,'legacy_and003_exact_bodies_retained':True}))


if __name__=='__main__':main()
