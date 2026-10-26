#!/usr/bin/env python3
"""One bounded native finite-radius stage; no scheduler submission from this tool."""
import argparse,copy,csv,hashlib,json,math,shutil,struct,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
support=Path(__file__).resolve().parent/'source/tst/cubed_sphere'
if support.is_dir():sys.path.insert(0,str(support))
from run_conformal_spherical import run_case,sha
from run_spherical_puncture import parameters


def checkpoint(case,receipt,tag):
    p=case/'probe_checkpoint.bin';meta=receipt['checkpoint_payload']
    assert p.stat().st_size==meta['bytes'] and sha(p)==meta['sha256']
    with p.open('rb') as f:
        assert f.read(16)==b'ATHENA_CSP_CK02!'
        assert f.read(64).decode()==tag
        signature=struct.unpack('<17d',f.read(136))
        fingerprint,time,cycle=struct.unpack('<Qdq',f.read(24))
        assert signature[1]==.98 and signature[-1]==8
        assert time==float(receipt['last_rows'][-1]['time']) and cycle==int(receipt['last_rows'][-1]['cycle'])
        for _ in range(2):
            shape=struct.unpack('<5Q',f.read(40));assert shape[1]==25 and all(v>0 for v in shape)
            f.seek(math.prod(shape)*8,1)
        assert f.tell()+8==p.stat().st_size
    return dict(sha256=meta['sha256'],bytes=meta['bytes'],time=time,cycle=cycle,
                signature=signature,reference_fingerprint=fingerprint,
                note='SHA/header/shape verification; native reader validates FNV checksum and reference on restart')


def monitor(case,receipt,prior=None):
    errors=[];baseline=copy.deepcopy((prior or {}).get('constraint_baseline',{}))
    rows=list(csv.DictReader((case/'cubed_sphere_conformal_regions.csv').open()))
    maxrows=list(csv.DictReader((case/'cubed_sphere_conformal_maxima.csv').open()))
    series={}
    for rows,fields in [(rows,['H_l2','H_linf','M_conformal_l2','Z_conformal_l2']),
                        (maxrows,['M_conformal_linf','Z_conformal_linf'])]:
        for row in rows:
            for field in fields:
                value=float(row[field]);key='/'.join([row['region'],row['subset'],field]);time=float(row['time'])
                if not math.isfinite(value) or value<0:errors.append('invalid '+key);continue
                series.setdefault(key,[]).append([time,value])
                if time<=.02+1e-14:baseline[key]=max(baseline.get(key,0),value)
                if value>1:errors.append('absolute constraint threshold '+key)
                if time>.02+1e-14 and value>100*max(baseline.get(key,0),1e-10):errors.append('growth threshold '+key)
    for row in receipt['last_rows']:
        if not (float(row['min_alpha'])>0 and float(row['min_chi'])>0 and float(row['min_Omega'])>0):errors.append('inadmissible endpoint')
    assert receipt['reference_unchanged'] and receipt.get('initial_profile_unchanged',True)
    gauge=list(csv.DictReader((case/'cubed_sphere_gauge_monitor.csv').open()))
    assert gauge and all(math.isfinite(float(row[k])) for row in gauge for k in ['gauge_deviation_RMS','relative_lapse_RMS','relative_lapse_max'])
    windows=copy.deepcopy((prior or {}).get('window_peaks',{}))
    last=float(receipt['last_rows'][-1]['time']);complete=int(math.floor((last+1e-12)/.02))
    for key,values in series.items():
        bins=windows.setdefault(key,{})
        for time,value in values:
            index=str(int(math.floor((time+1e-12)/.02)));bins[index]=max(bins.get(index,0),value)
        for i in range(3,complete):
            v=[bins.get(str(j)) for j in range(i-3,i+1)]
            if all(x is not None for x in v) and v[0]>1e-10 and all(v[j+1]>=2*v[j] for j in range(3)):
                errors.append('three consecutive equal-window doublings '+key)
    horizon_history=copy.deepcopy((prior or {}).get('horizon_history',{}))
    hp=case/'cubed_sphere_horizon_candidates.csv'
    if hp.exists():
        for row in csv.DictReader(hp.open()):horizon_history[row['time']]=row
        reliable=0
        for time,row in sorted(horizon_history.items(),key=lambda x:float(x[0])):
            if row['found']=='1':
                reliable+=1
                if reliable>=3 and abs(float(row['irreducible_mass'])/.5-1)>.05:errors.append('horizon mass drift above five percent')
            elif reliable>=3:errors.append('horizon candidate lost after reliable detection')
    return dict(passed=not errors,errors=sorted(set(errors)),constraint_baseline=baseline,
                window_peaks=windows,horizon_history=horizon_history,
                regional_series=series,scope='All native regions including full transition and outer collar. Stability monitoring only; accuracy and physical acceptance are separate.')


def configurations():
    common={'cubed_sphere/outer_radius':.98,'cubed_sphere/conformal_interior_radial':False,
            'cubed_sphere/conformal_max_cycles':10000000,'cubed_sphere/conformal_probe_steps':0,
            'cubed_sphere/conformal_fixed_dt':0.,'cubed_sphere/conformal_diss':.1}
    p=parameters();p.update(common);p['cubed_sphere/radial_stretch']=0.;p['cubed_sphere/conformal_quadrupole']=False
    q=dict(common);q.update({'cubed_sphere/radial_stretch':2.8592697712828787,
        'cubed_sphere/conformal_quadrupole':True,'cubed_sphere/conformal_mass':0.,
        'cubed_sphere/conformal_native_layer_gauge':False,'cubed_sphere/conformal_curvature_radius':.5,
        'cubed_sphere/conformal_layer_r0':.05,'cubed_sphere/conformal_layer_r1':.95,'problem/pulse_width':.35})
    return {'puncture':dict(bulk=24,nr=123,na=18,amplitude=0.,kappa=5.,limit=25.,extra=p),
            'pulse':dict(bulk=24,nr=72,na=18,amplitude=.2,kappa=10.,limit=10.,extra=q)}


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('binary',type=Path);a.add_argument('source',type=Path);a.add_argument('output',type=Path);a.add_argument('--previous',type=Path);a.add_argument('--wall',type=float,default=300);args=a.parse_args();assert 0<args.wall<=300
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);source=args.source.resolve()
    binary=out/'athena_frozen';shutil.copy2(args.binary,binary);tag=sha(binary)
    previous=json.loads((args.previous/'summary.json').read_text()) if args.previous else None
    if previous:assert previous['binary_sha256']==tag and previous['all_monitors_passed']
    configs=configurations();record=dict(binary_sha256=tag,runner_sha256=sha(Path(__file__)),configurations=configs,previous=str(args.previous) if args.previous else None,
        source_sha256={str(p.relative_to(source)):sha(p) for p in sorted(set([*source.glob('src/**/*.cpp'),*source.glob('src/**/*.hpp'),source/'inputs/z4c/cubed_sphere_conformal_probe.athinput']))},cases={},monitors={},checkpoints={})
    if previous:assert record['source_sha256']==previous['source_sha256']
    def run(item):
        index,(name,g)=item;extra=dict(g['extra']);time=0
        if previous:
            old=previous['cases'][name];time=float(old['last_rows'][-1]['time'])
            oldcase=Path(old['checkpoint_payload']['path']).parent;checkpoint(oldcase,old,tag)
            assert previous['configurations'][name]==g
            extra['cubed_sphere/conformal_restart']=str((oldcase/'probe_checkpoint.bin').resolve())
        if time>=g['limit']:return name,None,None
        targets=[.02,.1,.2,.5,1.,2.]+[float(i) for i in range(3,int(g['limit'])+1)]
        target=next(t for t in targets if t>time+1e-13)
        extra.update({'cubed_sphere/conformal_checkpoint':str(out/name/'probe_checkpoint.bin'),
            'cubed_sphere/conformal_checkpoint_tag':tag,'cubed_sphere/conformal_wall_seconds':args.wall})
        r=run_case(binary,source,out,name,str(index),g['bulk'],g['nr'],g['na'],g['amplitude'],g['kappa'],.1,target,5,420,.03,extra)
        return name,r,extra
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(run,enumerate(configs.items())))
    for name,r,extra in results:
        if r is None:
            # Completed branch remains tied to its previous authoritative artifact.
            record['cases'][name]=previous['cases'][name];record['monitors'][name]=previous['monitors'][name];record['checkpoints'][name]=previous['checkpoints'][name];continue
        record['cases'][name]=r
        if r['outcome'] not in ['complete','checkpointed'] or not r['reference_unchanged'] or not r['all_rows_finite']:
            record['monitors'][name]=dict(passed=False,errors=['native run failed or timed out']);continue
        record['checkpoints'][name]=checkpoint(out/name,r,tag)
        record['monitors'][name]=monitor(out/name,r,(previous or {}).get('monitors',{}).get(name))
        if previous:assert record['checkpoints'][name]['reference_fingerprint']==previous['checkpoints'][name]['reference_fingerprint']
    record['all_monitors_passed']=all(v['passed'] for v in record['monitors'].values())
    record['physical_goal_complete']=False
    (out/'summary.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(all_monitors_passed=record['all_monitors_passed'],times={n:r['last_rows'][-1]['time'] if r['last_rows'] else None for n,r in record['cases'].items()})))
    raise SystemExit(0 if record['all_monitors_passed'] else 1)

if __name__=='__main__':main()
