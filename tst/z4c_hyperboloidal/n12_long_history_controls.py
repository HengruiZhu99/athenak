#!/usr/bin/env python3
"""Synthetic parser/gate controls; no long native evolution is performed."""
import argparse
import copy
import csv
import json
import math
from pathlib import Path

from n12_checkpoint import DT, expected_time, require, sha256
from n12_long_history import LABELS, RMS, STEPS, REFERENCE_SHA, assess, read_outputs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    require(sha256(args.reference)==REFERENCE_SHA,'changed synthetic source reference')
    with args.reference.open() as stream:
        reference={int(row['step']):{k:float(v) for k,v in row.items()} for row in csv.DictReader(stream)}
    fixture=args.output/'synthetic';fixture.mkdir()
    history=['# Athena++ history data','# '+ ' '.join(f'[{i+1}]={name}' for i,name in enumerate(LABELS))]
    horizon=['# synthetic summary; no actual finder execution'];shape=[]
    for step in STEPS:
        r=reference[step];time=expected_time(step)
        values=[0.]*12;values[0]=time;values[1]=DT;values[2]=r['full_volume'];values[7]=r['exterior_volume']
        for key,index in RMS:values[index]=r[key]
        history.append(' '.join(format(x,'.17e') for x in values))
        radius=r['horizon_radius_M']/3;area=4*math.pi*radius*radius
        row=[r['horizon_mass_M']/3,0,0,0,0,area,r['horizon_expansion_rms']**2,0,r['horizon_r'],r['horizon_r']]
        horizon.append(str(step)+' '+format(time,'.6g')+' '+' '.join(format(x,'.15e') for x in row))
        shape.extend([f'# iter = {step}, Time = {time:.6g}',
            ' '.join(format(x,'.6e') for x in [r['horizon_r']*math.sqrt(4*math.pi)]+[0.]*168)])
    paths=[fixture/'synthetic.z4c_hyperboloidal.hst',fixture/'synthetic.horizon_summary_0.txt',fixture/'synthetic.horizon_shape_0.txt']
    for path,lines in zip(paths,[history,horizon,shape]):path.write_text('\n'.join(lines)+'\n')
    h,s,_,_=read_outputs([fixture]);baseline=assess(h,s,reference)
    require(baseline['passed'] and len(baseline['times'])==11,'complete synthetic common-state baseline failed')
    controls={}
    missing=copy.deepcopy(h);missing[3551]=[]
    require(not assess(missing,s,reference)['passed'],'missing final state accepted');controls['missing_final']='rejected'
    changed=copy.deepcopy(h);changed[3551][0]['values'][11]+=1e-4
    require(not assess(changed,s,reference)['passed'],'late exterior Z error accepted');controls['late_exterior_Z']='rejected'
    changed=copy.deepcopy(s);changed[3551][0]['areal_radius_M']+=2e-6
    require(not assess(h,changed,reference)['passed'],'areal radius error accepted');controls['areal_radius']='rejected'
    changed=copy.deepcopy(s);changed[3551][0]['values'][10]+=.1
    require(assess(h,changed,reference)['passed'],'unadopted compact-radius gate introduced');controls['compact_radius_is_diagnostic']='preserved'
    changed=copy.deepcopy(h);changed[3551].append(copy.deepcopy(changed[3551][0]));changed[3551][-1]['values'][11]+=1e-4
    bad=assess(changed,s,reference)
    require(not bad['passed'] and len(bad['times']['3551']['history_rows'])==2 and
            bad['times']['3551']['duplicate_failures'],'bad duplicate hidden');controls['bad_duplicate']='retained and rejected'
    changed=copy.deepcopy(h);changed[3551].append(copy.deepcopy(changed[3551][0]))
    require(assess(changed,s,reference)['passed'],'identical duplicate rejected');controls['identical_duplicate']='retained and passed'
    for name,path,transform,message in [
        ('wrong_exact_history_time',paths[0],lambda text:text.replace(format(expected_time(356),'.17e'),format(expected_time(356)+1e-15,'.17e')),'exact common stored time'),
        ('wrong_summary_printed_time',paths[1],lambda text:text.replace('356 '+format(expected_time(356),'.6g'),'356 0.1'),'printed time'),
        ('incomplete_shape',paths[2],lambda text:text.replace(shape[1], ' '.join(shape[1].split()[:-1]),1),'coefficients')]:
        original=path.read_text();path.write_text(transform(original))
        try:
            try:read_outputs([fixture])
            except ValueError as error:
                require(message in str(error),'wrong parser rejection: '+str(error));controls[name]=str(error)
            else:raise AssertionError(name+' accepted')
        finally:path.write_text(original)
    report=dict(passed=True,scope='synthetic11-time parser/gate tests, not native long-run evidence',
        baseline_passed=True,controls=controls,reference_sha256=REFERENCE_SHA,
        source_sha256={str(Path(__file__).with_name(name).resolve()):sha256(Path(__file__).with_name(name))
            for name in ('n12_long_history_controls.py','n12_long_history.py','n12_checkpoint.py')})
    (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,controls=len(controls),scope=report['scope'])))


if __name__=='__main__':main()
