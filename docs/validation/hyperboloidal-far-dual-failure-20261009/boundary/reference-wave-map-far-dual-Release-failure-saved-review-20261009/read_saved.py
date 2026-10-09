"""Stdlib-only association of saved far-dual FAIL; no target/source recomputation."""
from pathlib import Path
from collections import Counter
from decimal import Decimal
import hashlib
import json
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def load(path):return json.loads(Path(path).read_text(),parse_int=lambda x:-0.0 if x=='-0' else int(x))
def write(path,x):Path(path).write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n')
def verify(pins):
    for path,digest in pins.items():
        if sha(path)!=digest:raise RuntimeError('protected input changed: '+path)

def main():
    r=load(HERE/'recipe.json');owner=Path(r['owner']);out=HERE/'attempt001';out.mkdir(exist_ok=False)
    start=time.monotonic();receipt={'completed':False,'returncode':1,'saved_only':True,'targets_recomputed':False,'candidate_imported':False,'kernel_queries':False}
    pins=r['pins'].copy();pins[str(HERE/'recipe.json')]=sha(HERE/'recipe.json');pins[str(Path(__file__).resolve())]=sha(__file__)
    try:
        if not sys.dont_write_bytecode or not sys.flags.no_user_site:raise RuntimeError('require -B -s')
        config=load(owner/'recipe.json')
        for row in load(config['input_pins'])+load(owner/'source-index.json')['files']:
            if row['path'] in pins and pins[row['path']]!=row['sha256']:raise RuntimeError('conflicting pin')
            pins[row['path']]=row['sha256']
        verify(pins);write(out/'pins-before.json',pins)
        child=load(owner/'attempts/Release001/receipt.json');report=load(owner/'attempts/Release001/oracle-report.json')
        assert child['returncode']==1 and child['completed'] is False and child['passed'] is False and child['source_inputs_unchanged'] is True
        assert report['passed'] is False and report['inputs_unchanged'] is True
        assert report['counts']=={'FD_representatives':16,'closed_positive':18,'direct_dual':2352,'helper_calls':{'closed':36,'direct':4864},'legacy_negative_reuse':3,'records':2373,'total_helper_calls':4900}
        failures=report['failures'];assert len(failures)==63
        assert Counter(f['check'] for f in failures)==Counter({'native-parts-dual':47,'native-rhs-dual':16})
        assert all(f['label']['seed']==13 and f['label']['dual']==1 and f['label']['case'] is None for f in failures)
        expected={(f['label']['base'],f['label']['seed']) for f in failures};assert len(expected)==26
        selected={};all_rows=0;line_counts=Counter()
        with (owner/'attempts/Release001/direct.jsonl').open() as stream:
            for line_number,line in enumerate(stream,1):
                row=json.loads(line,parse_int=lambda x:-0.0 if x=='-0' else int(x));all_rows+=1
                key=(row['base_index'],row['seed_index'])
                if key not in expected:continue
                assert key not in selected and row['seed_id']=='metric-STF' and row['zero_primal_gradients'] is False
                for flag in ('all_input_finite','geometry_valid','positive_lapse_chi','SPD','reference_and_coefficients_zero_tangent'):
                    assert row[flag] is True
                assert row['uses_legacy_near'] is False and row['new']['valid'] and row['new']['assembled']
                selected[key]=(line_number,row)
        assert all_rows==2352 and set(selected)==expected
        components=['regular.alpha','regular.beta0','regular.beta1','regular.beta2','pole.alpha','pole.beta0','pole.beta1','pole.beta2','rhs.alpha','rhs.beta0','rhs.beta1','rhs.beta2']
        byfamily=Counter();byW=Counter();byradius=Counter();bycomponent=Counter();bya=Counter();bydirection=Counter();zero_target=Counter()
        mapped=[];contexts=[]
        for failure in failures:
            label=failure['label'];number,row=selected[(label['base'],label['seed'])];ci=label['component']
            got=(row['new']['parts']+row['new']['rhs'])[ci][1]
            # Reparse saved strings to verify association only; no target calculation.
            assert float(failure['got'])==float(got)
            context={k:row[k] for k in ('base_index','a','nominal_radius','stored_radius','direction','family','seed_index','seed_id','zero_primal_gradients','W_context')}
            target_zero=Decimal(failure['target'])==0
            byfamily[row['family']]+=1;byW[str(row['W_context'])]+=1;byradius[str(row['nominal_radius'])]+=1
            bycomponent[components[ci]]+=1;bya[str(row['a'])]+=1;bydirection[str(row['direction'])]+=1;zero_target[str(target_zero)]+=1
            mapped.append({'source_line':number,**context,'component':components[ci],**failure})
        for (base,seed),(number,row) in sorted(selected.items()):
            # The complete consumed local input/reference and scaled connection are retained;
            # unconsumed g_d/g_dd/A remain in original metadata-only JSONL.
            used=('alpha','chi','P','Theta','alpha_d','chi_d','beta','beta_d','Lambda','g')
            contexts.append({'source_line':number,**{k:row[k] for k in ('base_index','a','nominal_radius','direction','family','seed_id','W_context')},
                'xyz':row['xyz'],'Omega':row['Omega'],'Omega_d':row['Omega_d'],'input':{k:row['input'][k] for k in used},
                'reference':{k:row['reference'][k] for k in used},'connection':row['connection'],
                'new':row['new'],'legacy':row['legacy'],'uses_legacy_near':row['uses_legacy_near']})
        first=failures[0];worst=max(failures,key=lambda x:Decimal(x['error']))
        summary={'saved_readback_passed':True,'original_actual_Release_passed':False,'records_streamed':all_rows,'failure_count':63,'unique_failed_rows':26,
            'failure_checks':dict(Counter(f['check'] for f in failures)),'seed_indices':{'13':63},'seed':'metric-STF',
            'by_family':dict(byfamily),'by_W_context':dict(byW),'by_radius':dict(byradius),'by_component':dict(bycomponent),
            'by_a':dict(bya),'by_direction':dict(bydirection),'target_exact_zero_strings':dict(zero_target),
            'saved_maxima':report['maxima'],'saved_fixed_counts':report['counts'],'first_failure':first,'first_max_error_failure':worst,
            'input_contexts_retained':26,'underlying_payloads_metadata_only':True,
            'source_report_sha256':sha(owner/'attempts/Release001/oracle-report.json'),'source_child_sha256':sha(owner/'attempts/Release001/receipt.json'),
            'limitations':['This associates saved native and target strings; no oracle, inverse, gradient product, coefficient, or source target was recomputed.',
                'The original probe does not export Geometry inverse/value/dual entries or individual flux/connection summands. Their separate numerical error contributions cannot be recovered from these rows alone.',
                'All63 failures are consumed metric-value tangents at a fixed reference; all scalar/gradient/other seeds pass the saved gate, but that does not establish general metric contrast accuracy.',
                'No Debug/native run is admitted by a successful saved-data association. Original far Release FAIL remains immutable.']}
        write(out/'summary.json',summary);write(out/'failure-associations.json',mapped);write(out/'selected-consumed-contexts.json',contexts)
        receipt.update(completed=True,returncode=0,saved_readback_passed=True,protected_paths=len(pins))
    except BaseException as exc:
        receipt['failure']=type(exc).__name__+': '+str(exc);(out/'failure.txt').write_text(traceback.format_exc())
    finally:
        try:verify(pins);receipt['inputs_unchanged']=True
        except BaseException as exc:receipt.update(returncode=1,inputs_unchanged=False,post_pin_failure=str(exc))
        receipt['seconds']=time.monotonic()-start;write(out/'receipt.json',receipt)
    print(json.dumps(receipt))
    if not(receipt['completed'] and receipt['inputs_unchanged'] and receipt['returncode']==0):raise SystemExit(1)

if __name__=='__main__':main()
