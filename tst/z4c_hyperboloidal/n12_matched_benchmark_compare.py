#!/usr/bin/env python3
"""Scientific comparison of completed 24-rank cycle2486->2556 benchmarks.

Reuse the qualified native decoder and trace reader. This does not replace
launcher/build/job or native observer-overhead qualification. Raw timings are
reported only after scientific checks pass and remain provisional here.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics

import numpy as np

from n12_checkpoint import (DT, NF, NR, NBULK, NSURFACE, compare_fields, decode,
    expected_time, load_mapping, parse_native, require, sha256, validate_header)
from n12_long_history import read_outputs
from legacy_stage_trace_compare import load_trace
from legacy_n12_benchmark import VARIANTS
from native_step_observer_validate import stages
from n12_checkpoint_gpu_control import old_checkpoint, resolved_comparison

START, END = 2486, 2556
SEED_SHA = '27245d19010dee629d12c6e59872564bd70c19af8c06aa41c46cbe6b58143f18'


def histories(actual, reference):
    require(actual.shape==reference.shape and actual.shape[1]==NR and
            np.isfinite(actual).all() and np.isfinite(reference).all(), 'invalid chi history arrays')
    for data in (actual,reference):
        require((data[:,2:7]>=0).all() and
                np.equal(data[:,2:5],np.floor(data[:,2:5])).all() and
                (data[:,2:5]<=2**53).all(), 'invalid chi counts/corrections')
    result=dict(comparisons=int(actual.size),failed_values=0,fields=[])
    for f,name in enumerate(('minimum','accepted_minimum','floor_count','nonpositive_count',
                            'nonfinite_count','max_correction','sum_correction','first_floor')):
        delta=np.abs(actual[:,f]-reference[:,f])
        gate=0 if f in (2,3,4,7) else 1e-14+5e-10*np.abs(reference[:,f])
        failed=int((delta>gate).sum());result['failed_values']+=failed
        result['fields'].append(dict(field=f,name=name,failed_values=failed,
            bitwise_equal=bool(np.array_equal(actual[:,f].view('<u8'),reference[:,f].view('<u8'))),
            max_absolute=float(delta.max()),
            gate='exact counts/first-floor time' if f in (2,3,4,7) else '1e-14+5e-10*abs(reference)'))
    return result


def state(actual, reference, mapping):
    result=dict(passed=True,regions={})
    for region in ('bulk','surface'):
        fields=compare_fields(actual[region][:,:NF],reference[region][:,:NF],
            mapping[region+'_keys'],mapping[region+'_xyz'],mapping[region+'_weight'],'short')
        records=histories(actual[region][:,NF:],reference[region][:,NF:])
        result['regions'][region]=dict(fields=fields,records=records)
        result['passed'] &= fields['failed_values']==records['failed_values']==0
    return result


def native_diagnostics(directory, hashes):
    histories,horizons,files,extra=read_outputs([directory])
    require(not any(histories.values()) and not any(horizons.values()),
            'unexpected earlier native output during benchmark')
    for path in files: hashes[str(path.resolve())]=sha256(path)
    h=[x['record'] for x in extra if x['kind']=='history' and x['step']==END]
    a=[x['record'] for x in extra if x['kind']=='horizon' and x['step']==END]
    require(len(extra)==2 and len(h)==len(a)==1,'missing/extra final native diagnostics')
    row=h[0]['values']
    return dict(history=[row[i] for i in (2,3,4,5,6,7,8,9,10,11)],
                areal_radius_M=a[0]['areal_radius_M'],irreducible_mass_M=a[0]['irreducible_mass_M'],
                shape_coefficients=a[0]['shape_coefficients'])


def harmonic_comparison(horizon, native):
    # Exact order used by FastFlow::Write: each l's a0, then ac/as for m=1..l.
    # lmindex(l,m,lmax)=l*(lmax+1)+m. Native ordinary output prints %e (seven
    # significant digits), while horizon.json retains17. Keep raw and equally
    # printed comparisons; neither is an invented coefficient admission gate.
    rows=[]
    for l in range(13):
        components=[('a0',0,l)]+[(kind,m,l*13+m) for m in range(1,l+1) for kind in ('ac','as')]
        for kind,m,index in components:
            actual=float(horizon[kind][index]);reference=float(native[len(rows)])
            printed=float(format(actual,'.6e'))
            rows.append(dict(ordinal=len(rows),l=l,m=m,component=kind,legacy_index=index,
                actual=actual,reference=reference,absolute_difference=abs(actual-reference),
                bitwise_equal=bool(np.float64(actual).view('<u8')==np.float64(reference).view('<u8')),
                equal_at_native_print_precision=bool(np.float64(printed).view('<u8')==np.float64(reference).view('<u8'))))
    require(len(native)==len(rows)==169,'harmonic coefficient coverage differs')
    return dict(coefficients=rows,compared=169,bitwise_equal=all(r['bitwise_equal'] for r in rows),
        equal_at_native_print_precision=all(r['equal_at_native_print_precision'] for r in rows),
        maximum_absolute_difference=max(r['absolute_difference'] for r in rows),
        numeric_admission_gate=None,
        scope='Every coefficient is compared and reported. Completeness/finite values are required; individual coefficient differences are diagnostic only under the existing protocol. Native text has seven significant digits.')


def diagnostic_comparison(report, horizon, native):
    sums=report['full_and_horizon_exterior_sums']
    require(len(sums)==10 and all(math.isfinite(x) and x>=0 for x in sums) and
            0<sums[5]<=sums[0], 'invalid legacy diagnostic sums')
    history=[]
    for offset in (0,5):
        history += [sums[offset]]+[math.sqrt(sums[offset+i]/sums[offset]) for i in range(1,5)]
    require(horizon['found'] and horizon['complete_surface'] and horizon['lmax']==12,
            'incomplete legacy horizon')
    require(len(horizon['a0'])==13 and len(horizon['ac'])==len(horizon['as'])==169 and
            len(horizon['center'])==3 and horizon['center']==[0,0,0] and
            all(math.isfinite(x) for group in ('a0','ac','as','center') for x in horizon[group]),
            'incomplete/nonfinite/nonzero-center legacy harmonic representation')
    for key,expected in (('time',expected_time(END)),('time_M',3*expected_time(END)),
                         ('step',END),('cycle',END)):
        if key in horizon:require(horizon[key]==expected,'legacy horizon endpoint differs: '+key)
    coefficients=harmonic_comparison(horizon,native['shape_coefficients'])
    comparisons=[]
    for name,x,y in zip(('volume','H','M','Theta','Z','exterior_volume','eH','eM','eTheta','eZ'),
                        history,native['history']):
        tolerance=5e-10+1e-6*abs(y)
        comparisons.append(dict(field=name,actual=x,reference=y,absolute_difference=abs(x-y),
                                tolerance=tolerance,passed=abs(x-y)<=tolerance))
    require(math.isfinite(horizon['area']) and horizon['area']>0,'invalid legacy horizon area')
    actual_radius=3*math.sqrt(horizon['area']/(4*math.pi))
    for name,x in [('areal_radius_M',actual_radius),('irreducible_mass_M',horizon['irreducible_mass_M'])]:
        y=native[name];tolerance=max(1e-10,1e-7*abs(y))
        comparisons.append(dict(field=name,actual=x,reference=y,absolute_difference=abs(x-y),
                                tolerance=tolerance,passed=abs(x-y)<=tolerance))
    return dict(passed=all(x['passed'] for x in comparisons),comparisons=comparisons,
        legacy_full_harmonic_surface_present=True,native_complete_coefficients=169,
        harmonic_comparison=coefficients,center=[0,0,0],
        endpoint_time_scope='Legacy horizon JSON omits time; wrapper invokes Find at report.end_cycle/report.time, checked against2556. Native summary/shape printed time and exact history time are checked by read_outputs.',
        phase='Legacy accepted-final state precedes untimed diagnostic RHS; native final history is read-only. Both finders use native2486 persisted seed and final-only search.')


def native_qualification(path, final_path, hashes):
    report=json.loads(path.read_text());hashes[str(path.resolve())]=sha256(path)
    require(report['schema']=='native-n12-two-node-timing-validation-v1' and
            report['passed'] is True and report['timings_accepted'] is True,
            'native timing/observer qualification failed or missing')
    for name,digest in report['source_sha256'].items():
        source=Path(__file__).with_name(name)
        require(sha256(source)==digest,'native qualification reader/source changed: '+name)
        hashes[str(source.resolve())]=digest
    entries=report['checkpoints']['normal']
    require(len(entries)==1 and entries[0]['cycle']==END and
            entries[0]['sha256']==sha256(final_path),'native final not bound to qualified normal case')
    initial=report['initial_capture'];entries=initial['checkpoints']
    require(initial['chi_records_bitwise_exact'] is True and
            len(initial['fields'])==31 and all(x['failed_values']==0 for x in initial['fields']) and
            len(entries)==1 and entries[0]['cycle']==START,'native initial capture incomplete')
    initial_path=Path(entries[0]['path']);digest=sha256(initial_path)
    require(digest==entries[0]['sha256'] and
            report['input_sha256'].get(str(initial_path.resolve()))==digest,
            'native initial checkpoint changed or unbound')
    hashes[str(initial_path.resolve())]=digest
    return report,initial_path


def seed_state(directory):
    path=directory/'restart.bin';offset=path.stat().st_size-2*(NBULK+NSURFACE)*NF*8
    data=np.memmap(path,dtype='<f8',mode='r',offset=offset);result={}
    for region,count,start,file in (('bulk',NBULK,0,'chi-records.bin'),
            ('surface',NSURFACE,2*NBULK*NF,'surface-chi-records.bin')):
        fields=data[start:start+count*NF].reshape(count,NF)
        records=np.fromfile(directory/file,dtype='<f8').reshape(count,NR)
        result[region]=np.column_stack((fields,records))
    return result


def scalar(actual, reference):
    require(math.isfinite(actual) and math.isfinite(reference) and actual>=0 and reference>=0,
            'invalid clock/CFL scalar')
    tolerance=1e-12+5e-10*abs(reference)
    return dict(actual=actual,reference=reference,absolute_difference=abs(actual-reference),
                tolerance=tolerance,passed=abs(actual-reference)<=tolerance)


def norm_crosscheck(actual, reference):
    require(len(actual)==len(reference)==5 and
            all(math.isfinite(x) and x>=0 for x in actual+reference) and
            actual[0]>0 and reference[0]>0,'invalid full/fixed diagnostic sums')
    a=[actual[0]]+[math.sqrt(x/actual[0]) for x in actual[1:]]
    b=[reference[0]]+[math.sqrt(x/reference[0]) for x in reference[1:]]
    rows=[]
    for label,x,y in zip(('volume','H','M','Theta','Z'),a,b):
        tolerance=5e-11*(1+abs(y))
        rows.append(dict(field=label,actual=x,reference=y,absolute_difference=abs(x-y),
                         tolerance=tolerance,passed=abs(x-y)<=tolerance))
    return dict(passed=all(r['passed'] for r in rows),comparisons=rows,
                gate='existing same-checkpoint5e-11*(1+abs(reference))')


def control_norms(post, control):
    original=control['full_and_fixed_exterior_sums']
    require(len(original)==10 and len(post['full_and_horizon_exterior_sums'])==10,
            'missing original/postprocessor norm coverage')
    # The pinned original producer requests exterior_radius=.94, extended=false.
    # Its exterior is populated independently; fixed[0] is an inactive optional
    # accumulator and is serialized as five zeros, not a zero-volume observable.
    require(control['fixed_outer_sums']==[0.]*5,
            'pinned original inactive optional fixed accumulator differs')
    full=norm_crosscheck(post['full_and_horizon_exterior_sums'][:5],original[:5])
    fixed=norm_crosscheck(post['fixed_outer_sums'],original[5:])
    require(full['passed'] and fixed['passed'],
            'separate diagnostic full/fixed norms differ from original')
    return dict(full_norms_against_original=full,fixed_norms_against_original=fixed,
        original_optional_fixed_accumulator=dict(available=False,extended=False,
            serialized_values=control['fixed_outer_sums'],
            compared_observable='full_and_fixed_exterior_sums[5:10], exterior_radius=.94'))


def control_diagnostics(folder, source, control, values, manifest, mapping, seed, hashes):
    input_path=source.parent/'endpoint-inputs.json';binding=json.loads(input_path.read_text())
    hashes[str(input_path.resolve())]=sha256(input_path)
    require(binding['schema']=='legacy-n12-endpoint-inputs-v1' and binding['passed'] is True and
            binding['seed_manifest_sha256']==SEED_SHA and binding['cycle']==END and
            binding['time']==expected_time(END) and binding['dt']==DT and binding['ranks']==24 and
            (binding['bulk_points'],binding['scri_points'])==(NBULK,NSURFACE) and
            binding['all31_plus8_accepted_input'] is True,'endpoint input manifest differs')
    expected_files={binding['seed_directory']+'/'+name:digest for name,digest in seed['output_sha256'].items()}
    expected_files[binding['seed_directory']+'/seed.json']=SEED_SHA
    for name in ['results.json','endpoint-restart.bin']+[
            f'accepted-final-{region}-rank{rank}.bin' for region in ('bulk','scri') for rank in range(24)]:
        expected_files[binding['accepted_directory']+'/'+name]=sha256(source/name)
    require(binding['files']==expected_files,'endpoint manifest does not bind every source field/history/seed')
    for suffix in ('before','after'):
        log=source.parent/f'endpoint-inputs-verify-{suffix}.txt'
        require(log.read_text()=='Verified unchanged original-control endpoint and2486 horizon seed\n',
                'endpoint input verification missing: '+suffix)
        hashes[str(log.resolve())]=sha256(log)
    path=folder/'results.json';post=json.loads(path.read_text());hashes[str(path.resolve())]=sha256(path)
    require(post['schema']=='legacy-n12-endpoint-diagnostics-v1' and post['passed'] is True and
            post['ranks']==24 and post['backend']=='SYCL' and
            post['cycle']==END and post['time']==expected_time(END) and post['dt']==DT and
            (post['bulk_points'],post['scri_points'])==(NBULK,NSURFACE) and
            post['evolution_steps']==0 and post['timing_measurement'] is False,
            'separate control endpoint diagnostics workload differs')
    for key in ('post_load_fields_bitwise_control_endpoint','post_load_records_bitwise_control_endpoint',
                'diagnostics_preserved_chi_records_bitwise'):
        require(post[key] is True,'endpoint diagnostic state/history claim failed: '+key)
    require(all(math.isfinite(post[k]) and post[k]>=0 for k in ('compatibility_max','clock_max','CFL_max'))
            and post['compatibility_max']<=1e-7,'invalid endpoint diagnostic compatibility/clock/CFL')
    require((folder/'exit-code.txt').read_text().strip()=='0','endpoint diagnostic process failed')
    hashes[str((folder/'exit-code.txt').resolve())]=sha256(folder/'exit-code.txt')
    checkpoint=old_checkpoint(source/'endpoint-restart.bin',END,values['bulk'][:,:NF],values['surface'][:,:NF])
    require(resolved_comparison(manifest['physics_identity'],checkpoint['physics_identity'])['all_common_exact'],
            'control endpoint physics differs')
    hashes[str((source/'endpoint-restart.bin').resolve())]=checkpoint['sha256']
    expected={f'accepted-final-{region}-rank{rank}.bin' for region in ('bulk','scri') for rank in range(24)}
    require({p.name for p in folder.glob('*.bin')}==expected,'postprocessor snapshot coverage differs')
    for region,label in [('bulk','bulk'),('surface','scri')]:
        _,actual,proof=load_trace(folder,'legacy','accepted-final',label,24,mapping)
        require(np.array_equal(actual.view('<u8'),values[region].view('<u8')),
                'actual postprocessor accepted31+8 differs: '+region)
        hashes.update(proof['source_sha256'])
    norms=control_norms(post,control)
    return post,dict(passed=True,checkpoint=checkpoint,post_load_fields_and_records_bitwise=True,
        **norms,evolution_steps=0,
        scope='Separate untimed v8 diagnostic process on exact original-control accepted endpoint. Original Step and measured windows remain unchanged.')


def step_comparison(old, observed):
    require(len(old)==END-START and len(observed)==4*(END-START),'immutable/native step coverage differs')
    result=[]
    for i,row in enumerate(old):
        step=START+i+1
        require(int(row['step'])==step and float(row['t'])==expected_time(step) and float(row['dt'])==DT,
                'immutable per-step clock differs')
        group=observed[4*i:4*i+4]
        require([(r['step'],r['stage']) for r in group]==[(step,s) for s in range(1,5)],
                'native stage order differs')
        item=dict(step=step)
        for label,column in [('clock','stage_clock'),('CFL','CFL')]:
            require(all(math.isfinite(r[label]) and r[label]>=0 for r in group),
                    'invalid native stage scalar')
            item[label]=scalar(max(r[label] for r in group),float(row[column]))
        result.append(item)
    return result


def immutable_control(folder, proof_path, native, initial, seed_values, variants, manifest, mapping,
                      qualification, observer_folder, seed, hashes):
    proof=json.loads(proof_path.read_text());hashes[str(proof_path.resolve())]=sha256(proof_path)
    expected=dict(schema='legacy-n12-immutable-control-run-v1',
        executable_sha256=VARIANTS['control'][0],seed_sha256=seed['output_sha256']['restart.bin'],
        input_sha256=seed['output_sha256']['run.in'],
        configuration_sha256=seed['output_sha256']['configuration.cfg'],ranks=24)
    require(all(proof.get(k)==v for k,v in expected.items()),'immutable control run identity differs')
    for file,key in [('command.txt','command_sha256'),('exit-code.txt','exit_code_sha256')]:
        path=folder.parent/file;require(sha256(path)==proof[key],'immutable actual execution record changed')
        hashes[str(path.resolve())]=sha256(path)
    require((folder.parent/'exit-code.txt').read_text().strip()=='0','immutable control invocation failed')
    require(sha256(folder/'run.in')==expected['input_sha256'] and
            sha256(folder/'configuration.cfg')==expected['configuration_sha256'],
            'immutable control input/config changed')
    plan=json.loads((folder/'plan.json').read_text());finished=json.loads((folder/'result.json').read_text())
    require(plan['backend']=='SYCL' and plan['ranks']==24 and
            (plan['bulk_points'],plan['surface_points'])==(NBULK,NSURFACE) and
            plan['nlim']==END and plan['tlim']==2 and plan['dt_M']==3*DT and not plan['host_evolution'],
            'immutable control workload differs')
    require(finished['status']=='step_limit' and not finished['failure'] and
            finished['steps']==END and finished['time_M']==3*expected_time(END) and
            finished['start_M']==3*expected_time(START),'immutable control did not complete70 steps')
    result=dict(passed=True,initial={},final={},step_metrics=[],
        record_scope='SP1 and original campaign do not restore/store eight per-cell ChiRecords; no seeded record equivalence claimed for this control.',
        phase='Immutable initial/final snapshots follow diagnostic RHS and CompleteState. It performs no FastFlow search. Wrapper accepted-final snapshot precedes its extra final diagnostic RHS; short field gates retain this phase difference.')
    for name,step,targets in [('initial',START,{'native_postinit':initial,'legacy_postload_seed':seed_values}),
                             ('final',END,dict(native=native,**variants))]:
        snapshot=folder/name;clock=json.loads((snapshot/'time.json').read_text())
        require(clock['steps']==step and clock['t']==expected_time(step) and clock['t_M']==3*expected_time(step),
                'immutable snapshot clock differs')
        reference={region:np.fromfile(snapshot/file,dtype='<f8').reshape(count,NF)
            for region,file,count in [('bulk','fields.bin',NBULK),('surface','boundary.bin',NSURFACE)]}
        identity=old_checkpoint(snapshot/'restart.bin',step,reference['bulk'],reference['surface'])
        require(resolved_comparison(manifest['physics_identity'],identity['physics_identity'])['all_common_exact'],
                'immutable resolved physics differs')
        entry=dict(checkpoint=identity,comparisons={})
        for target,values in targets.items():
            compared={region:compare_fields(values[region][:,:NF],reference[region],mapping[region+'_keys'],
                mapping[region+'_xyz'],mapping[region+'_weight'],'short') for region in ('bulk','surface')}
            entry['comparisons'][target]=compared
            result['passed'] &= all(x['failed_values']==0 for x in compared.values())
        result[name]=entry
        for path in snapshot.iterdir():
            if path.is_file():hashes[str(path.resolve())]=sha256(path)
    with (folder/'steps.csv').open() as stream:old=list(csv.DictReader(stream))
    observed,_=stages(observer_folder,START,END,24)
    for filename in ('native-stages.csv','native-step-observer.json'):
        path=observer_folder/filename;digest=sha256(path)
        require(qualification['input_sha256'].get(str(path.resolve()))==digest,'native observer metrics unbound')
        hashes[str(path.resolve())]=digest
    result['step_metrics']=step_comparison(old,observed)
    result['passed'] &= all(row[label]['passed'] for row in result['step_metrics'] for label in ('clock','CFL'))
    result['exact_step_clock_coverage']=END-START;result['native_stage_rows']=len(observed)
    for filename in ('run.in','configuration.cfg','plan.json','result.json','steps.csv'):
        path=folder/filename;hashes[str(path.resolve())]=sha256(path)
    return result


def run(args):
    output=args.output.resolve();require(not output.exists(),'new comparison directory required');output.mkdir(parents=True)
    report=dict(schema='native-n12-matched-benchmark-science-v1',passed=False,
        performance_admitted=False,input_sha256={},variants={},provisional_timings={},
        scope='Scientific endpoint/diagnostic comparison only. Build/runtime/job and native timing/observer qualification remain separately required; no final parity claim from one three-window suite.')
    hashes=report['input_sha256']
    try:
        manifest,mapping=load_mapping(args.mapping)
        seed_path=args.seed/'seed.json';seed=json.loads(seed_path.read_text());hashes[str(seed_path.resolve())]=sha256(seed_path)
        require(sha256(seed_path)==SEED_SHA and
                seed['schema']=='native-n12-matched-benchmark-seed-v1' and seed['passed'] and
                seed['start_cycle']==START and seed['end_cycle']==END and
                seed['mapping_sha256']==manifest['mapping_sha256'], 'unqualified seed')
        for name,digest in seed['output_sha256'].items():
            path=args.seed/name;require(sha256(path)==digest,'benchmark seed changed');hashes[str(path.resolve())]=digest
        native_info=parse_native(args.native_checkpoint);validate_header(native_info,manifest,END)
        require(native_info['metadata'][4]==24,'native comparison requires24ranks')
        hashes[str(args.native_checkpoint.resolve())]=sha256(args.native_checkpoint)
        native=decode(native_info,mapping);diagnostics=native_diagnostics(args.native_output,hashes)
        qualification,initial_path=native_qualification(args.native_qualification,args.native_checkpoint,hashes)
        for file,digest in list(hashes.items()):
            if str(Path(file).parent)==str(args.native_output.resolve()):
                require(qualification['input_sha256'].get(file)==digest,'native diagnostic not bound to qualification')
        initial_info=parse_native(initial_path);validate_header(initial_info,manifest,START)
        initial=decode(initial_info,mapping)
        seed_values=seed_state(args.seed)
        report['initial_native_against_legacy_postload']=state(initial,seed_values,mapping)
        report['initial_native_against_legacy_postload']['scope']='Actual native zero-step post-Init owned snapshot versus converted seed. Each legacy wrapper independently requires its actual post-Load/pre-warmup31+8 bitwise to this same seed. Source equality and native reprojection roundoff remain distinct.'
        require(report['initial_native_against_legacy_postload']['passed'],'native/legacy actual initial state differs')
        report['native_qualification_sha256']=sha256(args.native_qualification)
        folders={}
        for item in args.legacy:
            name,path=item.split('=',1);require(name in VARIANTS and name not in folders,'duplicate/unknown variant')
            folders[name]=Path(path).resolve()
        require(set(folders)==set(VARIANTS),'complete control/v3/v6/v8 screening required')
        stored={};pending_timings={};legacy_reports={}
        for name,folder in folders.items():
            result_path=folder/'results.json';r=json.loads(result_path.read_text());hashes[str(result_path)]=sha256(result_path)
            legacy_reports[name]=r
            expected_schema=('legacy-n12-original-control-benchmark-v1' if name=='control'
                             else 'legacy-n12-matched-benchmark-v1')
            require(r['schema']==expected_schema and r['passed'] is True and
                    r['backend']=='SYCL' and r['ranks']==24 and
                    (r['start_cycle'],r['end_cycle'],r['warmup'],r['window_steps'],r['windows'])==(START,END,10,20,3)
                    and r['time']==expected_time(END) and r['dt']==DT,
                    'legacy execution shape/time differs')
            require((r['bulk_points'],r['scri_points'])==(NBULK,NSURFACE) and
                    r['initial_fields_bitwise_seed'] and r['initial_records_bitwise_seed'] and
                    not r['profiling_library_loaded'] and not r['detailed_timers_enabled'] and
                    r['complete_RK4_steps_timed']==60,'legacy initial-state or timing scope failed')
            for key in ('compatibility_max','clock_max','CFL_max'):
                require(math.isfinite(r[key]) and r[key]>=0,'nonfinite/negative '+key)
            require(r['compatibility_max']<=1e-7,'compatibility bound failed')
            require((folder/'exit-code.txt').read_text().strip()=='0','legacy invocation failed')
            hashes[str(folder/'exit-code.txt')]=sha256(folder/'exit-code.txt')
            expected={f'accepted-final-{region}-rank{rank}.bin' for region in ('bulk','scri') for rank in range(24)}
            if name=='control':expected.add('endpoint-restart.bin')
            require({p.name for p in folder.glob('*.bin')}==expected,'incomplete/extra accepted state files')
            stored[name]={}
            for region,label in [('bulk','bulk'),('surface','scri')]:
                _,values,proof=load_trace(folder,'legacy','accepted-final',label,24,mapping)
                stored[name][region]=values;hashes.update(proof['source_sha256'])
            diagnostic_report=r;diagnostic_folder=folder
            if name=='control':
                diagnostic_report,bridge=control_diagnostics(args.control_diagnostics,folder,r,
                    stored[name],manifest,mapping,seed,hashes)
                report['control_diagnostic_bridge']=bridge
                diagnostic_folder=args.control_diagnostics
            horizon_path=diagnostic_folder/'horizon.json';horizon=json.loads(horizon_path.read_text())
            hashes[str(horizon_path)]=sha256(horizon_path)
            state_report=state(stored[name],native,mapping)
            diag_report=diagnostic_comparison(diagnostic_report,horizon,diagnostics)
            elapsed=r['maximum_rank_window_seconds'];rank=np.asarray(r['rank_window_seconds'])
            require(len(elapsed)==3 and rank.shape==(72,) and np.isfinite(rank).all() and
                    (rank>0).all() and all(math.isfinite(x) and x>0 for x in elapsed),
                    'invalid window durations')
            require(np.array_equal(rank.reshape(3,24).max(axis=1),np.asarray(elapsed)),
                    'maximum-rank window duration differs from rank attribution')
            times=[x/20 for x in elapsed]
            pending_timings[name]=dict(window_step_seconds=times,median=statistics.median(times),
                minimum=min(times),maximum=max(times),critical_rank_each_window=rank.reshape(3,24).argmax(axis=1).tolist(),
                complete_cell_updates_per_step=NBULK,scope='one three-window screening; independent order-reversed repeats still required')
            report['variants'][name]=dict(against_native=state_report,diagnostics=diag_report)
        for name in stored:
            report['variants'][name]['against_legacy_control']=state(stored[name],stored['control'],mapping)
            report['variants'][name]['cumulative_metrics_against_control']={key:scalar(
                legacy_reports[name][key],legacy_reports['control'][key]) for key in ('clock_max','CFL_max')}
        report['immutable_control']=immutable_control(args.immutable_control,args.immutable_provenance,
            native,initial,seed_values,stored,manifest,mapping,qualification,args.native_output.parent/'timed-observer',seed,hashes)
        report['passed']=all(v['against_native']['passed'] and v['diagnostics']['passed'] and
                             v['against_legacy_control']['passed'] and
                             all(x['passed'] for x in v['cumulative_metrics_against_control'].values())
                             for v in report['variants'].values()) and report['immutable_control']['passed']
        if report['passed']:
            report['provisional_timings']=pending_timings
            report['fastest_scientifically_qualified_screening_variant']=min(
                folders,key=lambda name:pending_timings[name]['median'])
            report['reference_selection_scope']='All qualified matched-window predecessor builds, including original-source control, are eligible. Immutable campaign per-step timings are context only.'
        report['gates']=dict(fields='existing short1e-12+5e-10|ref|; chi1e-14+5e-10|ref|',
            records='exact counts/first-floor; minima/max/sum correction1e-14+5e-10|ref|, predeclared for nonzero seed',
            history='5e-10+1e-6|ref|',horizon='max(1e-10,1e-7|ref|) in M units')
    except Exception as error:
        report['validation_error']=f'{type(error).__name__}: {error}';raise
    finally:
        report['source_sha256']={str(Path(__file__).with_name(name).resolve()):sha256(Path(__file__).with_name(name))
            for name in ('n12_matched_benchmark_compare.py','n12_checkpoint.py','n12_long_history.py',
                         'legacy_stage_trace_compare.py','legacy_n12_benchmark.py',
                         'native_step_observer_validate.py','n12_checkpoint_gpu_control.py')}
        (output/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    require(report['passed'],'matched scientific comparison failed')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('native-checkpoint','native-output','native-qualification','control-diagnostics','immutable-control',
                 'immutable-provenance','mapping','seed','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--legacy',action='append',required=True,metavar='VARIANT=DIRECTORY')
    result=run(parser.parse_args())
    print(json.dumps({k:result[k] for k in ('passed','performance_admitted','provisional_timings')},indent=2))
