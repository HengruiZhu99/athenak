#!/usr/bin/env python3
"""Offline parser controls only; no solver, build, job or timing measurement."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np

import n12_matched_benchmark_compare as comparison
from n12_checkpoint import decode, load_mapping, parse_native, require, sha256


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    result=dict(schema='native-n12-matched-benchmark-offline-controls-v1',passed=False,
                gpu_execution=False,performance_claim=False,checks=[],input_sha256={})
    def check(name,call,accept=True):
        try:
            value=call()
        except (RuntimeError,ValueError,KeyError,IndexError) as error:
            require(not accept,name+' unexpectedly rejected: '+str(error))
            result['checks'].append(dict(name=name,passed=True,rejected=str(error)));return
        require(accept and value is not False,name+' unexpectedly accepted/failed')
        result['checks'].append(dict(name=name,passed=True))
    try:
        _,mapping=load_mapping(args.mapping)
        actual=decode(parse_native(args.checkpoint),mapping)
        seed=comparison.seed_state(args.seed)
        check('actual native2486 and converted SP1 all31+8 exact',lambda:all(
            np.array_equal(actual[r].view('<u8'),seed[r].view('<u8')) for r in ('bulk','surface')))
        check('actual admitted seeded-history parser',lambda:all(
            comparison.histories(actual[r][:,31:],seed[r][:,31:])['failed_values']==0
            for r in ('bulk','surface')))
        records=np.array([[1e-5,1e-5,2,1,0,1e-8,2e-8,.1]],dtype='<f8')
        corrupted=records.copy();corrupted[0,3]+=.5
        check('fractional event rejected',lambda:comparison.histories(corrupted,records),False)
        corrupted=records.copy();corrupted[0,6]=-1
        check('negative correction rejected',lambda:comparison.histories(corrupted,records),False)
        corrupted=records.copy();corrupted[0,4]=1
        check('nonfinite-event count compared exactly',lambda:comparison.histories(corrupted,records)['failed_values']==1)
        corrupted=records.copy();corrupted[0,7]+=1e-15
        check('first-floor time compared exactly',lambda:comparison.histories(corrupted,records)['failed_values']==1)
        corrupted=records.copy();corrupted[0,0]+=5e-15
        check('predeclared chi-unit history gate accepts',lambda:comparison.histories(corrupted,records)['failed_values']==0)
        corrupted=records.copy();corrupted[0,0]+=1e-10
        check('predeclared chi-unit history gate rejects',lambda:comparison.histories(corrupted,records)['failed_values']==1)
        horizon=dict(found=True,complete_surface=True,lmax=12,center=[0,0,0],area=4*np.pi,
                     irreducible_mass_M=1.5,a0=list(np.arange(13,dtype=float)),
                     ac=list(np.arange(169,dtype=float)+1000),
                     **{'as':list(np.arange(169,dtype=float)+2000)})
        # Independent literal loop follows the serialized l/m order, using the
        # rectangular13-column storage rather than the comparator's tuples.
        native=[]
        for l in range(13):
            native.append(float(l))
            for m in range(1,l+1):native.extend((1000.+13*l+m,2000.+13*l+m))
        check('all169 distinct harmonic slots mapped',lambda:comparison.harmonic_comparison(horizon,native)['bitwise_equal'])
        last=copy.deepcopy(horizon);last['as'][168]+=1
        check('last harmonic difference retained without invented gate',lambda:
              comparison.harmonic_comparison(last,native)['coefficients'][-1]['absolute_difference']==1)
        report=dict(full_and_horizon_exterior_sums=[1.,1.,1.,1.,1.,.5,.5,.5,.5,.5])
        target=dict(history=[1.,1.,1.,1.,1.,.5,1.,1.,1.,1.],areal_radius_M=3.,
                    irreducible_mass_M=1.5,shape_coefficients=native)
        check('synthetic diagnostic framing',lambda:comparison.diagnostic_comparison(report,horizon,target)['passed'])
        for name,key,value in [('nonzero center','center',[0,0,.01]),
                               ('wrong endpoint time','time',0.),
                               ('incomplete harmonics','a0',[0.]),
                               ('nonfinite harmonic','ac',[float('nan')]*169)]:
            changed=copy.deepcopy(horizon);changed[key]=value
            check(name,lambda changed=changed:comparison.diagnostic_comparison(report,changed,target),False)
        changed=copy.deepcopy(horizon);changed['irreducible_mass_M']+=1e-5
        check('existing horizon mass gate rejects',lambda:not comparison.diagnostic_comparison(report,changed,target)['passed'])
        failed=args.output/'unqualified-native.json'
        failed.write_text(json.dumps(dict(schema='native-n12-two-node-timing-validation-v1',passed=False,timings_accepted=False)))
        check('unqualified native timing proof rejected',lambda:comparison.native_qualification(failed,args.checkpoint,{}),False)
        old=[dict(step=s,t=comparison.expected_time(s),dt=comparison.DT,stage_clock=.25,CFL=.5)
             for s in range(comparison.START+1,comparison.END+1)]
        observed=[dict(step=s,stage=k,clock=.25,CFL=.5)
                  for s in range(comparison.START+1,comparison.END+1) for k in range(1,5)]
        check('synthetic70-step max-of-four clock/CFL',lambda:all(
            row[key]['passed'] for row in comparison.step_comparison(old,observed) for key in ('clock','CFL')))
        check('missing final stage rejected',lambda:comparison.step_comparison(old,observed[:-1]),False)
        changed=copy.deepcopy(old);changed[-1]['dt']*=2
        check('immutable final dt mismatch rejected',lambda:comparison.step_comparison(changed,observed),False)
        changed=copy.deepcopy(observed);changed[-1]['CFL']+=1e-6
        check('late CFL disagreement retained',lambda:not comparison.step_comparison(old,changed)[-1]['CFL']['passed'])
        changed=copy.deepcopy(observed);changed[-1]['clock']=float('nan')
        check('late nonfinite stage rejected',lambda:comparison.step_comparison(old,changed),False)
        check('same-checkpoint full/fixed norm bridge exact',lambda:comparison.norm_crosscheck([1.,1.,1.,1.,1.],[1.,1.,1.,1.,1.])['passed'])
        check('same-checkpoint norm bridge rejects',lambda:not comparison.norm_crosscheck([1.,1.001,1.,1.,1.],[1.,1.,1.,1.,1.])['passed'])
        check('same-checkpoint norm bridge nonfinite rejected',lambda:comparison.norm_crosscheck([1.,float('nan'),1.,1.,1.],[1.,1.,1.,1.,1.]),False)
        original=dict(full_and_fixed_exterior_sums=[1.]*5+[.5]*5,fixed_outer_sums=[0.]*5)
        post=dict(full_and_horizon_exterior_sums=[1.]*5+[.25]*5,fixed_outer_sums=[.5]*5)
        check('pinned extended-false control compares populated exterior',lambda:
              comparison.control_norms(post,original)['fixed_norms_against_original']['passed'])
        changed=copy.deepcopy(post);changed['fixed_outer_sums'][-1]+=.001
        check('pinned control rejects changed postprocessor fixed norm',lambda:
              comparison.control_norms(changed,original),False)
        changed=copy.deepcopy(original);changed['fixed_outer_sums'][-1]=1.
        check('pinned control rejects nonzero inactive optional accumulator',lambda:
              comparison.control_norms(post,changed),False)
        for path in (args.checkpoint,args.seed/'seed.json',args.mapping/'manifest.json'):
            result['input_sha256'][str(path.resolve())]=sha256(path)
        result['passed']=True
    except Exception as error:
        result['failure']=str(error);raise
    finally:
        result['source_sha256']={name:sha256(Path(__file__).with_name(name)) for name in
            ('n12_matched_benchmark_controls.py','n12_matched_benchmark_compare.py','n12_checkpoint.py')}
        (args.output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=result['passed'],checks=len(result['checks']),gpu_execution=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('seed','checkpoint','mapping','output'):parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
