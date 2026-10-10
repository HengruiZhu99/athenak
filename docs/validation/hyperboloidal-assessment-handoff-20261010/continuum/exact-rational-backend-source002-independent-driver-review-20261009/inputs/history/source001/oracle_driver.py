"""Future admitted82-case registry/independent Fraction checker; no import-time math."""
import argparse
from pathlib import Path
import sys

P=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['prepare','check'],required=True)
    parser.add_argument('--recipe',required=True)
    parser.add_argument('--authorization',required=True)
    parser.add_argument('--input')
    parser.add_argument('--report')
    args=parser.parse_args()
    if not(sys.flags.isolated==1 and sys.dont_write_bytecode and sys.flags.optimize==0):
        raise RuntimeError('require -I -B optimization0')
    sys.path.insert(0,str(P))
    from gate_common import admit,check,load,save,require,sha,pin
    recipe,auth,protected=admit(args.recipe,args.authorization)
    attempt=P/'attempts/units001'
    lifecycle=load(attempt/'receipt.json')
    require(lifecycle.get('both_dependency_closures_passed') is True,
            'both compiled dependency closures must pass before any oracle import')
    for mode in ('release','debug'):
        check(lifecycle[mode+'_dependencies']);check(lifecycle[mode+'_executable'])
    # Only this exact post-admission point imports registry/Fraction arithmetic.
    import fraction_units as base
    import carry_range_units as carry
    cases=base.registry()+carry.registry()
    require(len(cases)==82 and len({case['id'] for case in cases})==82,'fixed82 unique registry drift')
    protocol=base.protocol(cases)
    if args.mode=='prepare':
        require(args.input is None and args.report is None,'prepare accepts no foreign data')
        require(not(attempt/'registry.json').exists() and not(attempt/'cases.txt').exists(),'registry destination reused')
        save(attempt/'registry.json',cases)
        (attempt/'cases.txt').write_text(protocol)
        save(attempt/'registry-receipt.json',dict(passed=True,cases=82,base_cases=74,carry_range_cases=8,
            source_index_sha256=sha(P/'source-index.json'),registry=pin(attempt/'registry.json'),protocol=pin(attempt/'cases.txt')))
    else:
        path,report=Path(args.input).resolve(),Path(args.report).resolve()
        require(path in [attempt/'probe-release.stdout',attempt/'probe-debug.stdout'],'only fixed actual saved probe output')
        require(report in [attempt/'oracle-release.json',attempt/'oracle-debug.json'] and not report.exists(),
                'only fresh fixed oracle report')
        generated=load(attempt/'registry-receipt.json')
        require(generated.get('passed') is True and generated.get('source_index_sha256')==sha(P/'source-index.json'),
                'registry receipt mismatch')
        check(generated['registry']);check(generated['protocol'])
        require(load(attempt/'registry.json')==cases and (attempt/'cases.txt').read_text()==protocol,'registry/protocol drift')
        observed=pin(path)
        lines=path.read_text().splitlines()
        require(len(lines)==82,'native output count')
        original=base.check_output('\n'.join(lines[:74])+'\n')
        extra=carry.check_output('\n'.join(lines[74:])+'\n',base)
        require(original['passed'] is True and original['checks']==74 and extra['passed'] is True and extra['checks']==8,
                'all82 exact Fraction controls must pass')
        save(report,dict(passed=True,fixed_cases=82,base_cases=74,carry_range_cases=8,
             exact_full_operand_comparisons=original['exact_full_operand_comparisons']+extra['exact_full_operand_comparisons'],
             base_report=original,carry_range_report=extra,source_index_sha256=sha(P/'source-index.json'),
             probe_stdout=observed))
        check(observed)
    for entry in protected: check(entry)


if __name__=='__main__':
    main()
