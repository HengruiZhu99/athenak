"""Stdlib-only source pin and saved-JSONL arithmetic readback, no model evaluation."""
from pathlib import Path
from decimal import Decimal, localcontext
from collections import Counter
import hashlib
import itertools
import json
import shutil
import sys
import time
import traceback

ROOT = Path('/Users/hz0693/research/hyperboloidal')
HERE = Path(__file__).resolve().parent
OWNER = ROOT / 'build-layer-research/continuum/manufactured-angular-Gaussian-native-inverse-held-20261009'
AUTH = ROOT / 'build-layer-research/manufactured-angular-Gaussian-native-inverse-root-release-20261009/authorization.json'
EXPECTED = {
    OWNER/'source-index.json': '27be52a7a76657a6bf7544f21d2a7357b19f4e101950385314a7171c9ddf64a2',
    OWNER/'attempt001/result.json': 'a4e4636f990537c02b555355b3a6a30ed63bb532a0e8082e915bd3b3c38546f1',
    OWNER/'attempt001/receipt.json': 'c2ea131e2c607e4ee450f64569cf8da16a09727852b242f0a1a204f9c25da0ba',
    OWNER/'outer-invocation001/receipt.json': 'd349a96cdd924359aa0e0d3a43536b19c6c4e6dee86398cb1a23abe3d36caad4',
    AUTH: '6988102dc1e773cb7d44874057aca9deb17579b730f2a7720473ea9388f0ce32',
}


def load(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def pin(path):
    path = Path(path).resolve()
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return {'path': str(path), 'sha256': h.hexdigest(), 'bytes': path.stat().st_size,
            'large_payload': path.suffix.lower() in ('.npz', '.npy', '.jsonl') or path.stat().st_size > 1048576}


def write(name, value):
    path = HERE/name
    if path.exists():
        raise RuntimeError('Refuse overwrite '+str(path))
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def dec(value):
    out = Decimal(value)
    require(out.is_finite(), 'Nonfinite saved decimal '+str(value))
    return out


def finite_adm(value):
    if isinstance(value, dict):
        for item in value.values(): finite_adm(item)
    elif isinstance(value, list):
        for item in value: finite_adm(item)
    else:
        dec(value)


def scaled(x, y):
    return abs(x-y)/max(Decimal(1), abs(x), abs(y))


def main():
    started = time.monotonic()
    protected = {}
    record = {'completed': False, 'returncode': 1, 'saved_data_only': True,
              'model_import_solve_CAS_kernel_compiler_evolution': False}
    try:
        for path, digest in EXPECTED.items():
            require(pin(path)['sha256'] == digest, 'Typed prerequisite drift '+str(path))
            protected[str(path)] = digest
        index, recipe = load(OWNER/'source-index.json'), load(OWNER/'recipe.json')
        outer, result = load(OWNER/'outer-invocation001/receipt.json'), load(OWNER/'attempt001/result.json')
        child = load(OWNER/'attempt001/receipt.json')
        require(child['completed'] and child['returncode']==0 and child['inputs_unchanged'] and child['checks_passed'], 'Owner child did not pass consistency')
        require(outer['completed'] and outer['returncode']==0 and outer['inputs_unchanged'] and outer['accepted_consistency_execution'], 'Owner outer consistency did not pass')
        require(outer['sampled_positivity_accepted'] is False and result['all_sampled_D_positive'] is False, 'Negative domain was suppressed')
        for path, digest in recipe['pins'].items(): protected[path] = digest
        for row in index['files']+outer['output_pins']:
            prior = protected.setdefault(row['path'], row['sha256'])
            require(prior == row['sha256'], 'Conflicting saved/source pin '+row['path'])
        protected[str(Path(__file__).resolve())] = pin(__file__)['sha256']
        prior=HERE.parent/'manufactured-angular-Gaussian-native-inverse-independent-review-20261009'
        for name in ('review_saved.py','receipt.json','failure.txt','inputs-before.json'):
            protected[str(prior/name)]=pin(prior/name)['sha256']
        before = []
        for path, digest in protected.items():
            row = pin(path)
            require(row['sha256']==digest, 'Protected source/data drift '+path)
            before.append(row)
        write('inputs-before.json', before)
        captured = HERE/'captured-sources'
        captured.mkdir(exist_ok=False)
        for row in index['files']:
            require(not pin(row['path'])['large_payload'], 'Unexpected large source')
            shutil.copyfile(row['path'], captured/Path(row['path']).name)
        shutil.copyfile(AUTH, captured/'root-authorization.json')
        levels = [str(x['digits'])+'-'+str(x['height_order']) for x in recipe['levels']]
        keys = set(itertools.product(recipe['sigma'], recipe['epsilon'],
            [x['label'] for x in recipe['native_radii']], recipe['native_times'], recipe['p_values']))
        radius_values = {x['label']:x['value'] for x in recipe['native_radii']}
        require(len(keys)==recipe['records_per_level']==13608, 'Recipe event count mismatch')
        seen = {level:set() for level in levels}
        metrics = {level:{} for level in levels}
        counts, branches, negatives = Counter(), Counter(), []
        profiles = {}
        maxima = {name:Decimal(0) for name in ('inverse_residual','inverse_width','original_map_residual','outer_direct_identity')}
        serialization_bounds = dict(maxima)
        root_abs, root_width, comparison_tol = map(dec, (recipe['root_absolute_tolerance'],recipe['root_width_tolerance'],recipe['comparison_tolerance']))
        with localcontext() as ctx:
            ctx.prec = 170
            with (OWNER/'attempt001/samples.jsonl').open() as stream:
                for line_number, line in enumerate(stream, 1):
                    row = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
                    level = row['level']
                    key = (row['sigma'],row['epsilon'],row['native_radius']['label'],row['native_time'],row['p'])
                    require(level in levels and key in keys and key not in seen[level], 'Registry mismatch/duplicate at '+str(line_number))
                    require(row['native_radius']['value']==radius_values[key[2]], 'Radius label/value mismatch')
                    seen[level].add(key);counts[level]+=1
                    require(level==str(row['digits'])+'-'+str(row['height_order']) and row['s_angular']=='1', 'Level/angular metadata mismatch')
                    root = row['inverse']
                    for label in ('initial','final'):
                        bracket=root[label]
                        require(dec(bracket['lower'])<=dec(bracket['upper']) and dec(bracket['g_lower'])<=0<=dec(bracket['g_upper']), 'Saved inverse sign/order failure')
                    require(dec(root['residual'])<=root_abs and dec(root['width'])<=root_width, 'Saved inverse residual/width gate failure')
                    require(0<=root['newton_steps']<=recipe['newton_iterations'] and 0<=root['bisection_steps']<=recipe['bisection_iterations'], 'Iteration cap exceeded')
                    require(root['formal_interval_enclosure'] is False, 'Unexpected rigorous interval claim')
                    require(dec(row['original_map_residual'])<=root_abs and dec(row['outer_direct_identity'])<=comparison_tol, 'Saved original map/direct identity failure')
                    for label, operand in (('inverse_residual',root['residual']),('inverse_width',root['width']),('original_map_residual',row['original_map_residual']),('outer_direct_identity',row['outer_direct_identity'])):
                        value=dec(operand)
                        maxima[label]=max(maxima[label],value)
                        if value:
                            serialization_bounds[label]=max(serialization_bounds[label],Decimal(10)**(value.adjusted()+1-row['digits']))
                    J,D,ratio,T=map(dec,(row['J'],row['D'],row['D_over_reference'],row['physical_T']))
                    require(J>0 and T>=0, 'Saved inverse J/future failed')
                    require((D>0)==(ratio>0)==row['passed_positive'], 'D sign/value metadata mismatch')
                    require((row['ADM_values'] is not None)==(D>0), 'ADM positive-domain restriction mismatch')
                    if row['ADM_values'] is not None:
                        finite_adm(row['ADM_values'])
                        for field in ('alpha_bar','alpha_bar_over_reference','chi','det_bar_gamma'):
                            require(dec(row['ADM_values'][field])>0, 'Saved positive ADM scalar failed')
                        require(row['no_ADM_reason'] is None, 'Unexpected positive-domain reason')
                    else:
                        require(row['no_ADM_reason']=='D_nonpositive', 'Negative-domain reason missing')
                    outer_branch=row['outer_factored']
                    require(outer_branch==(row['inverse']['kind']=='outer_c_ret'), 'Root/outer branch mismatch')
                    branches[(level,row['radial_branch'],'outer' if outer_branch else 'nonouter')]+=1
                    # Recover the retained root answer from its saved numerical bracket.
                    # This is saved-decimal arithmetic, not a new inverse solution.
                    unknown=(dec(root['final']['lower'])+dec(root['final']['upper']))/2 if outer_branch else T
                    metrics[level][key]=(ratio,J,T,dec(row['retarded_u']),unknown,
                        None if row['ADM_values'] is None else dec(row['ADM_values']['alpha_bar_over_reference']))
                    profile=(level,row['sigma'],row['epsilon'])
                    if profile not in profiles:
                        profiles[profile]={'count':0,'positive':True,'minimum':ratio,'worst':key,'minJ':J}
                    summary=profiles[profile];summary['count']+=1;summary['positive'] &= D>0;summary['minJ']=min(summary['minJ'],J)
                    if ratio<summary['minimum']:summary.update(minimum=ratio,worst=key)
                    if D<=0:
                        negatives.append({'level':level,'sigma':row['sigma'],'epsilon':row['epsilon'],
                            'r':key[2],'t_native':key[3],'p':key[4], 'D_ratio':str(ratio),'J':str(J)})
                    if line_number%13608==0:print(json.dumps({'saved_records_checked':line_number}),flush=True)
            require(line_number==recipe['total_records']==54432 and all(seen[x]==keys for x in levels), 'Incomplete saved event registry')
            comparison_rows=[]
            names=('D_ratio','J','T','u','inverse_unknown_from_saved_bracket','alpha_ratio')
            for kind,left,right in recipe['comparison_pairs']:
                maximum=Decimal(0);worst=None
                for key in sorted(keys):
                    for field,x,y in zip(names,metrics[left][key],metrics[right][key]):
                        require((x is None)==(y is None), 'Cross-level D sign disagreement')
                        if x is None:continue
                        error=scaled(x,y)
                        if error>maximum:maximum=error;worst={'key':key,'field':field}
                require(maximum<=comparison_tol, 'Saved precision/height convergence failed')
                owner_comparison=next(x for x in result['comparison_pairs'] if (x['kind'],x['left'],x['right'])==(kind,left,right))
                # Record decimal reanalysis difference without claiming bit equality with mpmath.
                comparison_rows.append({'kind':kind,'left':left,'right':right,'maximum_scaled':str(maximum),
                    'worst':worst,'difference_from_owner_saved_maximum':str(abs(maximum-dec(owner_comparison['maximum_scaled'])))})
            height_constants={x['level']:dec(x['outer_height_constant']) for x in result['levels']}
            height_max=max(scaled(height_constants[left],height_constants[right]) for _,left,right in recipe['comparison_pairs'])
            require(height_max<=comparison_tol, 'Saved height constant convergence failed')
            profile_rows=[]
            for profile,s in sorted(profiles.items()):
                owner=next(x for x in result['profiles'] if (x['level'],x['sigma'],x['epsilon'])==profile)
                require(s['count']==owner['samples']==1701 and s['positive']==owner['all_sampled_D_positive'] and s['minimum']==dec(owner['minimum_D_over_reference']), 'Saved profile minimum/count mismatch')
                w=owner['worst'];owner_worst=(w['sigma'],w['epsilon'],w['native_radius']['label'],w['native_time'],w['p'])
                require(s['worst']==owner_worst, 'Saved profile worst event mismatch')
                profile_rows.append({'level':profile[0],'sigma':profile[1],'epsilon':profile[2],
                    'samples':s['count'],'all_sampled_D_positive':s['positive'],'minimum_D_over_reference':str(s['minimum']),
                    'worst':s['worst'],'minimum_J':str(s['minJ'])})
            maximum_serialization_comparisons=[]
            for name,value in maxima.items():
                reported=dec(result['maximum_'+name])
                bound=serialization_bounds[name]+(Decimal(10)**(reported.adjusted()+1-110) if reported else Decimal(0))
                difference=abs(value-reported)
                require(difference<=bound,'Saved maxima exceed source decimal serialization bound '+name)
                maximum_serialization_comparisons.append({'field':name,'saved_row_maximum':str(value),'owner_reported_maximum':str(reported),'difference':str(difference),'one_decimal_ulp_sum_bound':str(bound)})
            require(all(x['sigma']=='1/2' and x['epsilon']=='3/4' for x in negatives), 'Unexpected negative profile')
            ref_neg=[{k:v for k,v in x.items() if k not in ('level','D_ratio','J')} for x in negatives if x['level']=='110-256']
            require(all(len([x for x in negatives if x['level']==level])==len(ref_neg) for level in levels), 'Negative event count differs by level')
            domains={name:sorted({x[name] for x in ref_neg}) for name in ('r','t_native','p')}
            negative_counts=Counter(x['level'] for x in negatives)
            report={'passed_source_and_saved_data_review':True,'saved_rows':line_number,
                'records_per_level':dict(counts),'source_index':EXPECTED[OWNER/'source-index.json'],
                'all_saved_inverse_J_positive':True,'all_saved_future_T_nonnegative':True,
                'profile_rows':profile_rows,'comparison_rows':comparison_rows,
                'maximum_height_constant_comparison':str(height_max),
                'saved_maxima':{k:str(v) for k,v in maxima.items()},
                'maximum_serialization_comparisons':maximum_serialization_comparisons,
                'negative_record_count':len(negatives),'negative_counts_per_level':dict(negative_counts),
                'distinct_negative_events_at_each_level':len(ref_neg),'negative_domain_labels':domains,
                'negative_events_reference_110_256':ref_neg,
                'branches':[{'level':k[0],'radial_branch':k[1],'region':k[2],'count':v} for k,v in sorted(branches.items())],
                'all_sampled_D_positive':False,'consistency_pass_is_not_D_positivity':True,
                'source_math_review_no_blocker':True,'formal_interval_or_global_timelike_certificate':False,
                'new_model_evaluation_or_solve':False,
                'decimal_precision':170,'inverse_unknown_scope':'Saved final bracket midpoint, no new root solution; no bit-equality claim to mpmath',
                'limits':['Finite declared grid and four precision/height levels only; no minimum over continuous r/t/p.',
                    'Future/native target-time inverse distinct from physical T; J>0 alone is not timelikeness.',
                    'No embedding jets, full22 RHS, BH, PDE stability, native evolution or reference adoption accepted.']}
            write('report.json',report)
        after=[pin(row['path']) for row in before]
        require(after==before,'Readback protected input drift')
        write('inputs-after.json',after)
        record.update(completed=True,returncode=0,passed_source_and_saved_data_review=True,
            inputs_unchanged=True,source_runtime_output_pins=len(before),saved_rows=line_number,
            negative_records=len(negatives),negative_events_per_level=len(ref_neg))
    except BaseException as exc:
        record.update(failure=repr(exc))
        (HERE/'failure.txt').write_text(traceback.format_exc())
    finally:
        record['seconds']=time.monotonic()-started
        write('receipt.json',record)
    print(json.dumps(record),flush=True)
    if not record['completed']:raise SystemExit(1)


if __name__=='__main__':main()
