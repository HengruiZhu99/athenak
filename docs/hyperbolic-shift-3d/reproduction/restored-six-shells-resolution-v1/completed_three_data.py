"""Load completed six-shell controls through their own immutable validators."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[3]
sys.path.insert(0,str(HERE.parent/'native-shared-transition-v1'))
sys.path.insert(0,str(HERE.parent/'restored-six-shells-v1'))
import compare_histories as histories
from frozen_campaign import load
from audit_completed import load_completed

def load_cases():
    raw=ROOT/'output/restored-six-shells-resolution/aurora'
    c16,audit=load_completed()
    cases={16:c16}
    for n,version,count in [(20,1,1),(24,2,2)]:
        helper=load(raw/f'launcher-restored-six-shells-resolution-v{version}/native_transition_campaign.py')
        plans=[f'campaigns/n{n}-restored-six-shells-resolution-v1-r{n}-evolution-segment{s:02d}/plan.json' for s in range(1,count+1)]
        cases[n]=histories.load_native(raw,f'R{n}',plans,helper)
        assert cases[n]['reports'][-1]['status']=='completed'
        for s,r in enumerate(cases[n]['reports']):
            for check in helper.required_checks(f'R{n}','evolution','puncture',s>0):
                if check['path']!=['status']:
                    assert helper.control.evaluate_check(r,check)['passed'],check
        assert audit['binary_sha256']==helper.science(raw)['binaries'][helper.binary(f'R{n}')]
    for c in cases.values():
        assert c['reports'][-1]['t_over_M']==10
        assert abs(c['reports'][-1]['scri_penalty_scale_min']-.6510416666660155)<1e-13
    return cases,audit

def histories_and_sources():
    cases,audit=load_cases()
    return ({n:c['history'] for n,c in cases.items()},
            {n:dict(segments=c['records'],restart_diagnostic_repeats=c['restart_diagnostic_repeats'],
                    qualification='Original N16 policy failure retained' if n==16 else 'Verified matched clipped-penalty policy') for n,c in cases.items()})
