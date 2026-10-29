#!/usr/bin/env python3
"""Bounded initial-snapshot checks for explicit 1D-comparable diagnostics.

No time steps are taken. The reference prefix is a pre-change target-zero
puncture run with the same n, input and finder settings. The nonzero fixture
preserves its physical K by offsetting P when adding Theta, and sets a known
radial Lambda on the initial identity conformal metric. Raw fixture/checkpoint
files stay in the requested local output directory.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import time

import numpy as np

NEW = ("Theta_exterior_L2", "Zr_L2", "Zr_exterior_L2", "horizon_irreducible_mass")


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def scalar_close(actual, expected, label):
    if actual is None or not np.isfinite(actual) or not np.isclose(actual,expected,rtol=2e-12,atol=2e-14):
        raise AssertionError((label,actual,expected))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--initial',type=Path,required=True)
    p.add_argument('--coordinates',type=Path,required=True)
    p.add_argument('--reference-prefix',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--n',type=int,default=16)
    p.add_argument('--mpiexec',default='mpiexec')
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    exe=out/'runner';shutil.copy2(args.binary.resolve(),exe)
    initial=args.initial.resolve();coordinates=args.coordinates.resolve()
    hashes={str(f):digest(f) for f in [initial,coordinates,Path(str(args.reference_prefix)+'.checkpoint')]}
    coords=np.genfromtxt(coordinates,delimiter=',',names=True)
    values=np.loadtxt(initial,skiprows=1)
    n=args.n
    assert values.shape==(19*n**3,31) and len(coords)==len(values)
    assert np.all(values[:,1:7]==0),'analytic Z fixture requires initial identity conformal metric'
    r=coords['r'];xyz=np.column_stack([coords[k] for k in ['x','y','z']]);volume=coords['volume']
    theta=.02*(1-r*r)**2
    zr=.03*r*(1-r*r)**2
    fixture=values.copy();fixture[:,14]+=theta;fixture[:,13]-=2*theta
    fixture[:,15:18]=.06*xyz*(1-r*r)[:,None]**2
    fixture_path=out/'nonzero-initial.txt'
    with fixture_path.open('w') as f:
        f.write(str(fixture.size)+'\n');np.savetxt(f,fixture,fmt='%.17g')
    records=[];reports={};histories={}
    for name,input_path,ranks,tolerance in [
        ('unchanged-puncture',initial,1,'.001'),
        ('nonzero-one-rank',fixture_path,1,'.001'),
        ('nonzero-two-rank',fixture_path,2,'.001'),
        ('unresolved-horizon',fixture_path,1,'1e-15')]:
        prefix=out/name
        command=([args.mpiexec,'-n',str(ranks)] if ranks>1 else [])+[str(exe),str(n),str(prefix),'puncture','0','.0005',str(input_path),'-','400','.05',tolerance,'6','12']
        start=time.monotonic();run=subprocess.run(command,capture_output=True,text=True,timeout=410)
        prefix.with_suffix('.log').write_text(run.stdout+run.stderr)
        record=dict(name=name,command=command,seconds=time.monotonic()-start,returncode=run.returncode)
        records.append(record)
        if run.returncode:raise AssertionError(record)
        report=json.loads(prefix.with_suffix('.report.json').read_text())
        with prefix.with_suffix('.history.csv').open() as f:rows=list(csv.DictReader(f))
        assert report['status']=='completed' and report['t']==0 and report['steps']==0 and report['diagnostics_current']
        assert len(rows)==1 and all(k in rows[0] and k in report for k in NEW)
        history={key:float(value) for key,value in rows[0].items()}
        for key in NEW:
            if report[key] is None:assert np.isnan(history[key])
            else:scalar_close(history[key],report[key],name+' CSV/report '+key)
        reports[name]=report;histories[name]=history
    before=Path(str(args.reference_prefix)+'.checkpoint');after=out/'unchanged-puncture.checkpoint'
    assert digest(before)==digest(after),'diagnostic additions changed checkpoint bytes'
    header=struct.unpack('<16s4i3Q2d',after.read_bytes()[:72])
    assert header[4]==1 and header[7]==0 and header[8]==0
    expected={}
    for name in ['nonzero-one-rank','nonzero-two-rank','unresolved-horizon']:
        report=reports[name];history=histories[name]
        scalar_close(history['Theta_L2'],np.sqrt(np.sum(volume*theta**2)/sum(volume)),name+' total Theta')
        scalar_close(report['Zr_L2'],np.sqrt(np.sum(volume*zr**2)/sum(volume)),name+' total Zr')
        if name=='unresolved-horizon':
            assert not report['horizon_found'] and report['horizon_irreducible_mass'] is None
            for field in ['Theta_exterior_L2','Zr_exterior_L2','H_exterior_L2','Mr_exterior_L2']:
                assert report[field] is None and np.isnan(history[field]),field+' unavailable exterior must be null/NaN'
            continue
        assert report['horizon_found']
        mask=r>report['exterior_coordinate_cutoff'];w=volume[mask]
        expected[name]=dict(Theta_exterior_L2=float(np.sqrt(np.sum(w*theta[mask]**2)/sum(w))),
                            Zr_exterior_L2=float(np.sqrt(np.sum(w*zr[mask]**2)/sum(w))))
        for field,value in expected[name].items():scalar_close(report[field],value,name+' analytic '+field)
        scalar_close(report['horizon_irreducible_mass'],report['horizon_areal_radius']/2,name+' area mass')
        assert report['horizon_mass']>=report['horizon_irreducible_mass']-2e-14
    for field in NEW:
        scalar_close(reports['nonzero-one-rank'][field],reports['nonzero-two-rank'][field],'partition '+field)
    assert all(digest(Path(f))==value for f,value in hashes.items()),'source snapshot changed'
    assessment=dict(status='passed',scope='Initial snapshots only; zero evolution steps in all cases',
        appended_columns=NEW,source_inputs=hashes,inputs_unchanged=True,binary_sha256=digest(exe),
        checkpoint_byte_equivalence=dict(reference=str(before.resolve()),sha256=digest(after),
                                         bytes=after.stat().st_size,all_four_arrays_bitwise_equal=True),
        fixture=dict(theta='.02*(1-r^2)^2',radial_z='.03*r*(1-r^2)^2',
                     lambda_field='.06*x*(1-r^2)^2 on gtilde=identity',physical_K_preserved_by_P_offset=True),
        analytic_exterior_values=expected,executions=records,
        observations={name:{k:r.get(k) for k in [*NEW,'horizon_mass','horizon_areal_radius','horizon_found','horizon_expansion_rms']} for name,r in reports.items()})
    (out/'assessment.json').write_text(json.dumps(assessment,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='passed',output=str(out),checkpoint_sha256=digest(after))))


if __name__=='__main__':main()
