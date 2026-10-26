#!/usr/bin/env python3
"""Check diagnostic partitions and quantify independent resolution responses.

This checks bookkeeping and reports convergence. It deliberately makes no
stability or boundary-causality acceptance from finite final norms alone.
"""
import argparse
import csv
import json
import math
from pathlib import Path

METRICS=['H_l2','H_linf','M_l2','Z_l2','weyl_l2']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results',type=Path)
    args=parser.parse_args()
    cases={}
    for directory in sorted(args.results.glob('B*_S*')):
        receipt=json.loads((directory/'receipt.json').read_text())
        assert receipt['exit_code']==0
        profiles=list(csv.DictReader((directory/'cubed_sphere_constraint_regions.csv').open()))
        for region in ['bulk','shell']:
            rows={r['subset']:r for r in profiles if r['region']==region}
            total=rows['all']; radial=[r for key,r in rows.items() if key.startswith('radial_')]
            assert sum(int(r['cells']) for r in radial)==int(total['cells'])
            assert math.isclose(sum(float(r['volume']) for r in radial),float(total['volume']),rel_tol=1e-11)
            for metric in METRICS:
                assert math.isclose(float(total[metric]),receipt['norms'][region][metric],rel_tol=1e-10)
            for metric in ['H_l2','M_l2','Z_l2','weyl_l2']:
                square=sum(float(r['volume'])*float(r[metric])**2 for r in radial)
                assert math.isclose(square,float(total['volume'])*float(total[metric])**2,rel_tol=1e-10)
        cases[directory.name]={'norms':receipt['norms'],'profiles':profiles}
    output={'bookkeeping_pass':True,'cases':cases,'responses':{}}
    for axis,names,resolutions in [('bulk',['B32_S16','B48_S16','B64_S16'],[32,48,64]),
                                   ('shell',['B32_S16','B32_S24','B32_S32'],[16,24,32])]:
        if not all(n in cases for n in names):
            output['responses'][axis]={'incomplete':True}
            continue
        response={}
        for region in ['bulk','shell']:
            response[region]={}
            for metric in METRICS:
                values=[cases[n]['norms'][region][metric] for n in names]
                orders=[math.log(values[i]/values[i+1])/math.log(resolutions[i+1]/resolutions[i])
                        for i in range(2)]
                response[region][metric]={'values':values,'observed_orders':orders,
                                         'finest_over_baseline':values[-1]/values[0]}
        output['responses'][axis]=response
    (args.results/'analysis.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({'bookkeeping_pass':True,'responses':output['responses']},indent=2))


if __name__=='__main__':
    main()
