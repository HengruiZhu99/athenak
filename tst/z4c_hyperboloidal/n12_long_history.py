#!/usr/bin/env python3
"""Compare every common N12 history/horizon time through3551 without selecting duplicates."""
import argparse
import csv
import json
import math
from pathlib import Path
import re

from n12_checkpoint import DT, expected_time, load_mapping, parse_native, require, sha256, validate_header

STEPS = (0,356,711,1066,1421,1776,2131,2486,2841,3196,3551)
REFERENCE_SHA = '1a9cad154bf199e0e28f25384ad5185c17e85066d2afd337cc366b45f4a84435'
LABELS = ['time','dt','Volume','H-rms','M-rms','Theta-rms','Z-rms',
          'eVolume','eH-rms','eM-rms','eTheta-rms','eZ-rms']
RMS = [('full_H',3),('full_M',4),('full_Theta',5),('full_Z',6),
       ('exterior_H',8),('exterior_M',9),('exterior_Theta',10),('exterior_Z',11)]


def table(path):
    result = []
    for line_number, line in enumerate(path.read_text().splitlines(),1):
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        row = [float(x) for x in line.split()]
        require(all(math.isfinite(x) for x in row), 'nonfinite output '+str(path))
        result.append(dict(path=str(path.resolve()), line=line_number, values=row))
    return result


def read_outputs(runs):
    histories, horizons, files = {s:[] for s in STEPS}, {s:[] for s in STEPS}, []
    times = {expected_time(step):step for step in STEPS}
    extra = []
    for directory in runs:
        hst = sorted(directory.glob('*.z4c_hyperboloidal.hst'))
        summaries = sorted(directory.glob('*.horizon_summary_0.txt'))
        require(hst and summaries, 'run lacks ordinary history/horizon files: '+str(directory))
        local_history_steps = set()
        for path in hst:
            headers = [line for line in path.read_text().splitlines() if line.startswith('#')]
            require(len(headers) == 2 and re.findall(r'\[\d+\]=(\S+)', headers[1]) == LABELS,
                    'unexpected native history columns')
            files.append(path)
            for record in table(path):
                row = record['values']; require(len(row)==12 and row[1]==DT, 'history width/dt differs')
                require(0 < row[7] <= row[2] and all(row[i]>=0 for _,i in RMS), 'invalid history volume/RMS')
                step=times.get(row[0],int(round(row[0]/DT)))
                require(0<=step<=3551 and row[0]==expected_time(step),
                        'native history is not at an exact common stored time or nominal intermediate step')
                if step in histories:histories[step].append(record)
                else:extra.append(dict(kind='history',step=step,record=record))
                local_history_steps.add(step)
        for path in summaries:
            files.append(path)
            shape = path.with_name(path.name.replace('horizon_summary_0','horizon_shape_0'))
            require(shape.is_file(), 'missing native complete harmonic shape file')
            files.append(shape); shapes = []
            lines = shape.read_text().splitlines(); index=0
            while index<len(lines):
                if not lines[index].strip(): index+=1; continue
                match = re.fullmatch(r'# iter = (\d+), Time = (\S+)',lines[index].strip())
                require(match is not None and index+1<len(lines), 'invalid native shape framing')
                values=[float(x) for x in lines[index+1].split()]
                require(len(values)==169 and all(math.isfinite(x) for x in values),
                        'native lmax12 shape missing/nonfinite coefficients')
                shapes.append((int(match[1]),float(match[2]),values)); index+=2
            rows=table(path); require(len(rows)==len(shapes), 'horizon summary/shape count differs')
            for record, shape_record in zip(rows,shapes):
                row=record['values']; require(len(row)==12 and row[0]==int(row[0]), 'invalid horizon summary')
                step=int(row[0]); require(step in local_history_steps,
                    'horizon summary has no same-run exact-time history')
                printed=float(format(expected_time(step),'.6g'))
                require(row[1]==printed and shape_record[:2]==(step,printed),
                        'horizon cycle/printed time does not match exact history time')
                require(row[2]>0 and row[7]>0 and row[8]>=0 and row[10]>0 and row[11]>0,
                        'invalid/failed native horizon properties')
                record['shape_coefficients']=shape_record[2]
                record['areal_radius_M']=3*math.sqrt(row[7]/(4*math.pi))
                record['irreducible_mass_M']=record['areal_radius_M']/2
                record['reported_mass_M']=3*row[2]
                record['expansion_rms']=math.sqrt(row[8])
                if step in horizons:horizons[step].append(record)
                else:extra.append(dict(kind='horizon',step=step,record=record))
    return histories,horizons,files,extra


def assess(histories,horizons,reference):
    result=dict(passed=True, common_steps=list(STEPS), times={}, failures=[],
        history_gate='5e-10+1e-6*abs(reference)', horizon_gate='max(1e-10,1e-7*abs(reference)) in M units',
        duplicate_policy='retain every row; test each against reference and all same-time pairs under adopted gates',
        initial_roundoff_scope='time0 retained separately; no convergence inference',
        horizon_status_scope='found/complete inferred from successful native history guard plus finite summary and169 coefficients',
        mask_scope='full harmonic exterior from pinned native guard/source; no per-point mask bits serialized')
    for step in STEPS:
        rows, edges = histories[step], horizons[step]
        if not rows or not edges:
            result['failures'].append(dict(step=step,error='missing common history or horizon'))
            continue
        ref=reference[step]
        require(ref['horizon_found']==1, 'reference horizon not found')
        entry=dict(time=expected_time(step), history_rows=[], horizon_rows=[], duplicate_failures=[])
        for record in rows:
            compared=[]
            for label,index in RMS:
                actual=record['values'][index]; expected=ref[label]; gate=5e-10+1e-6*abs(expected)
                compared.append(dict(field=label,actual=actual,expected=expected,
                    absolute_difference=abs(actual-expected),gate=gate,passed=abs(actual-expected)<=gate))
            entry['history_rows'].append(dict(source=record, comparisons=compared))
            if not all(v['passed'] for v in compared):
                result['failures'].append(dict(step=step,error='history RMS gate',path=record['path'],line=record['line']))
        for record in edges:
            compared=[]
            for name,key in [('areal_radius_M','horizon_radius_M'),('irreducible_mass_M','horizon_mass_M')]:
                actual,expected=record[name],ref[key]; gate=max(1e-10,1e-7*abs(expected))
                compared.append(dict(field=name,actual=actual,expected=expected,
                    absolute_difference=abs(actual-expected),gate=gate,passed=abs(actual-expected)<=gate))
            entry['horizon_rows'].append(dict(source=record, comparisons=compared))
            if not all(v['passed'] for v in compared):
                result['failures'].append(dict(step=step,error='horizon areal/mass gate',path=record['path'],line=record['line']))
        # Every pair is checked; no last-row/favorable-row selection. Compact
        # radius and expansion remain diagnostics, not a newly invented gate.
        for i,a in enumerate(rows):
            for b in rows[i+1:]:
                for label,index in RMS:
                    gate=5e-10+1e-6*min(abs(a['values'][index]),abs(b['values'][index]))
                    if abs(a['values'][index]-b['values'][index])>gate:
                        entry['duplicate_failures'].append('RMS '+label)
        for i,a in enumerate(edges):
            for b in edges[i+1:]:
                for key in ('areal_radius_M','irreducible_mass_M'):
                    gate=max(1e-10,1e-7*min(abs(a[key]),abs(b[key])))
                    if abs(a[key]-b[key])>gate:entry['duplicate_failures'].append(key)
        if entry['duplicate_failures']:
            result['failures'].append(dict(step=step,error='inconsistent duplicates',fields=entry['duplicate_failures']))
        result['times'][str(step)]=entry
    result['passed']=not result['failures']
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,nargs='+',required=True)
    parser.add_argument('--mapping',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--runtime-provenance',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    require(sha256(args.reference)==REFERENCE_SHA,'historical FastFlow CSV differs from pinned reference')
    with args.reference.open() as stream:
        reference={int(row['step']):{k:float(v) for k,v in row.items()} for row in csv.DictReader(stream)}
    require(all(s in reference and reference[s]['t_M']==3*expected_time(s) for s in STEPS),
            'reference common cycles/times differ')
    manifest,_=load_mapping(args.mapping)
    provenance=json.loads(args.runtime_provenance.read_text())
    root=Path(__file__).resolve().parents[2]
    diagnostic=root/'src/z4c_hyperboloidal/diagnostics.cpp'
    source_groups=[provenance.get(name,{}) for name in ('sources','source_sha256')]
    hashes=[digest for group in source_groups for path,digest in group.items()
            if path.endswith('/z4c_hyperboloidal/diagnostics.cpp')]
    require(hashes==[sha256(diagnostic)],'runtime provenance does not bind reviewed horizon/mask guard')
    text=diagnostic.read_text()
    require('RequireAll(horizon_.found && horizon_.complete_surface && surface_.Valid()' in text and
            'mask(q)=surface_.Outside(x)?1:0;' in text,'native complete-horizon/full-mask guard changed')
    checkpoint_steps=set();checkpoints=[]
    required={'fastflow/lmax':12,'fastflow/ntheta':24,'fastflow/flow_iterations_0':300,
        'fastflow/flow_alpha_beta_const_0':.1,'fastflow/initial_radius_0':.02,
        'fastflow/mass_tol_0':1e-10,'fastflow/expansion_rms_tol_0':1e-3,
        'fastflow/center_x_0':0,'fastflow/center_y_0':0,'fastflow/center_z_0':0,
        'fastflow/use_puncture_0':-1,'fastflow/npunct':0}
    for run in args.runs:
        for path in sorted((run/'rst').glob('*.rst')):
            info=parse_native(path)
            if info['cycle'] not in STEPS:continue
            validate_header(info,manifest,info['cycle'])
            for key,value in required.items():
                require(key in info['parameters'] and float(info['parameters'][key])==value,
                        'native horizon setting differs '+key)
            checkpoint_steps.add(info['cycle']);checkpoints.append(path)
    require(checkpoint_steps==set(STEPS),'missing exact native checkpoint metadata at common times')
    histories,horizons,files,extra=read_outputs(args.runs)
    result=assess(histories,horizons,reference)
    result.update(reference_sha256=REFERENCE_SHA, mapping_sha256=manifest['mapping_sha256'],
        exact_checkpoint_times_and_fixed_horizon_settings=True,additional_output_rows=extra,
        runtime_provenance_sha256=sha256(args.runtime_provenance),
        source_sha256={str(Path(__file__).resolve()):sha256(__file__),
                       str(Path(__file__).with_name('n12_checkpoint.py').resolve()):sha256(Path(__file__).with_name('n12_checkpoint.py'))},
        artifact_sha256={str(path.resolve()):sha256(path) for path in files+checkpoints})
    (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'],common_states=len(result['times']),failures=result['failures']),indent=2))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
