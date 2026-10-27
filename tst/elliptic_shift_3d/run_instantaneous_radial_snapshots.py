#!/usr/bin/env python3
"""Three bounded, instantaneous global3D elliptic comparisons; no evolution."""
import argparse
import gzip
import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path

import numpy as np
from radial_snapshot import RadialSnapshot


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def physical_fields(data):
    count=len(data['points']);value=np.zeros((count,22));d=np.zeros((count,22,3));dd=np.zeros((count,22,3,3))
    def scalar(f,name):value[:,f]=data[name];d[:,f]=data[name+'_d'];dd[:,f]=data[name+'_dd']
    scalar(0,'chi');scalar(13,'trace');value[:,13]+=3;scalar(14,'theta');scalar(18,'alpha')
    ti=[0,0,0,1,1,2];tj=[0,1,2,1,2,2]
    for t,(i,j) in enumerate(zip(ti,tj)):
        value[:,1+t]=data['metric'][:,i,j]-(1 if i==j else 0);d[:,1+t]=data['metric_d'][:,:,i,j];dd[:,1+t]=data['metric_dd'][:,:,:,i,j]
        value[:,7+t]=data['a'][:,i,j];d[:,7+t]=data['a_d'][:,:,i,j];dd[:,7+t]=data['a_dd'][:,:,:,i,j]
    for i in range(3):
        for f,name in [(15+i,'lambda'),(19+i,'B')]:
            value[:,f]=data[name][:,i];d[:,f]=data[name+'_d'][:,:,i];dd[:,f]=data[name+'_dd'][:,:,:,i]
    return np.concatenate([value[:,:,None],d,dd.reshape(count,22,9)],axis=2).reshape(count,22*13)


def beta_fields(data):
    count=len(data['points'])
    return np.column_stack([data['beta'],data['beta_d'].reshape(count,9),data['beta_dd'].reshape(count,27)])


def write_fixture(snapshot,points,path,radii):
    before=time.monotonic();data7=snapshot.cartesian(points,7);data9=snapshot.cartesian(points,9)
    row=np.column_stack([points,physical_fields(data7),beta_fields(data7),physical_fields(data9),beta_fields(data9)])
    boundary=snapshot.sample(1.,7)[0];angular_r=[.05,.075,.1,.15,.2,.3,.45,.62,.66,.7,.8,.9,.95]
    with path.open('w') as out:
        out.write(f'RADIAL_SNAPSHOT_CARTESIAN_V1 6 {len(radii)} {len(points)} {snapshot.time:.17g} {snapshot.step} {sha(snapshot.path)}\n')
        out.write(' '.join(format(x,'.17g') for x in radii)+'\n')
        out.write(' '.join(format(x,'.17g') for x in boundary)+'\n')
        out.write(str(len(angular_r))+'\n')
        for r in angular_r:out.write(f'{r:.17g} {snapshot.sample(r,7)[0][6]:.17g}\n')
        np.savetxt(out,row,fmt='%.17g')
    return dict(seconds=time.monotonic()-before,rows=len(points),columns=row.shape[1],bytes=path.stat().st_size,
        sha256=sha(path),positive_physical_chi_alpha=True,
        reference='Interpolated numerical snapshot jets; no artificial L(beta) source; width7 and9 exported independently')


def main():
    p=argparse.ArgumentParser();p.add_argument('--build',type=Path,default=Path('build-shift-3d'))
    p.add_argument('--snapshots',type=Path,default=Path('docs/elliptic-shift-3d/evidence/radial-collapse-snapshots-20261027'))
    p.add_argument('--frozen-source',type=Path,help='Private extracted source archive; compile its headers/driver first')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=Path.cwd();build=a.build.resolve();out=a.output.resolve()
    source_root=a.frozen_source.resolve() if a.frozen_source else root
    out.mkdir(parents=True,exist_ok=False);radii=[.62,.70,1.];driver=source_root/'tst/elliptic_shift_3d/instantaneous_radial_snapshot.cpp'
    sources=sorted((source_root/'src').glob('**/*'))
    sources=[f for f in sources if f.is_file()]+[source_root/'tst/elliptic_shift_3d'/name for name in ['coupled.cpp','instantaneous_radial_snapshot.cpp','radial_snapshot.py']]+[Path(__file__).resolve()]
    hashes={str(f):sha(f) for f in sources}
    manifest=dict(scope='Instantaneous global3D elliptic diagnostic only; no3D evolution/scientific milestone claim',n=6,radii=radii,
        active_cell_budget=5616,solver_timeout_per_case=180,source_root=str(source_root),source_hashes_before=hashes,cases=[],build={})
    def save(): (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    flags=(build/'CMakeFiles/elliptic_shift_3d_coupled_pilot.dir/flags.make').read_text();flag_args=[]
    for name in ['CXX_DEFINES','CXX_INCLUDES','CXX_FLAGS']:
        value=next(line.split('=',1)[1] for line in flags.splitlines() if line.startswith(name+' ='));flag_args+=shlex.split(value)
    if a.frozen_source:
        # No live repository source include directory may precede the archive.
        flag_args=[arg.replace('-I'+str(root/'src'),'-I'+str(source_root/'src')) for arg in flag_args]
    link=shlex.split((build/'CMakeFiles/elliptic_shift_3d_coupled_pilot.dir/link.txt').read_text());remaining=[];i=1
    while i<len(link):
        if link[i]=='-o':i+=2;continue
        if not link[i].endswith('.o'):remaining.append(link[i])
        i+=1
    binary=build/('instantaneous_radial_snapshot_frozen' if a.frozen_source else 'instantaneous_radial_snapshot');command=[link[0],*flag_args,str(driver),'-o',str(binary),*remaining]
    manifest['build']['command']=command;save();start=time.monotonic()
    compiled=subprocess.run(command,cwd=build,text=True,capture_output=True,timeout=120)
    (out/'build.stdout').write_text(compiled.stdout);(out/'build.stderr').write_text(compiled.stderr)
    manifest['build'].update(exit_code=compiled.returncode,seconds=time.monotonic()-start);save()
    if compiled.returncode:print(json.dumps(manifest['build']),flush=True);raise SystemExit(compiled.returncode)
    manifest['binary_sha256']=sha(binary);points_file=out/'points.csv'
    command=[str(binary),'points','6',','.join(map(str,radii)),'1',str(points_file)]
    generated=subprocess.run(command,text=True,capture_output=True,timeout=60)
    (out/'points.stdout').write_text(generated.stdout);(out/'points.stderr').write_text(generated.stderr)
    manifest['point_command']=command;manifest['point_exit_code']=generated.returncode;save()
    if generated.returncode:raise SystemExit(generated.returncode)
    table=np.genfromtxt(points_file,delimiter=',',names=True);points=np.column_stack([table[x] for x in ['x','y','z']])
    if len(points)>5616:raise ValueError('cell budget exceeded')
    for filename in ['profile-0021.npz','profile-0101.npz','profile-0201.npz']:
        snapshot=RadialSnapshot(a.snapshots/filename);name=f't{snapshot.time:g}';case=dict(name=name,time=snapshot.time,snapshot_sha256=sha(snapshot.path),status='preparing')
        manifest['cases'].append(case);save();print(name,'preparing',len(points),'active points',flush=True)
        fixture=out/(name+'.dat');prefix=out/name
        try:
            case['fixture']=write_fixture(snapshot,points,fixture,radii);command=[str(binary),'solve',str(fixture),str(prefix)]
            case['command']=command;case['status']='running';save();start=time.monotonic()
            run=subprocess.run(command,text=True,capture_output=True,timeout=180)
            case.update(exit_code=run.returncode,seconds=time.monotonic()-start,status='finished')
            (out/(name+'.stdout')).write_text(run.stdout);(out/(name+'.stderr')).write_text(run.stderr)
            if run.stdout:
                try:case['result']=json.loads(run.stdout)
                except ValueError:case['unparsed_stdout']=True
        except subprocess.TimeoutExpired as error:
            case.update(exit_code=124,status='timeout',seconds=time.monotonic()-start)
            for suffix,value in [('stdout',error.stdout),('stderr',error.stderr)]:
                (out/(name+'.'+suffix)).write_bytes(value.encode() if isinstance(value,str) else (value or b''))
        except Exception as error:
            case.update(exit_code=2,status='preparation_error',error=f'{type(error).__name__}: {error}')
        case['sources_unchanged_since_build']=all(sha(f)==value for f,value in hashes.items())
        case['binary_unchanged']=sha(binary)==manifest['binary_sha256'];save()
        print(name,case['exit_code'],case.get('seconds'),case.get('error',''),flush=True)
    manifest['sources_unchanged_during_run']=all(sha(f)==value for f,value in hashes.items())
    manifest['binary_unchanged_during_run']=sha(binary)==manifest['binary_sha256'];save()
    compressed=[]
    for path in sorted(out.iterdir()):
        if path.suffix not in ['.csv','.dat']:continue
        data=path.read_bytes();target=path.with_name(path.name+'.gz');packed=gzip.compress(data,compresslevel=6,mtime=0)
        if gzip.decompress(packed)!=data:raise ValueError('Compression roundtrip failed')
        target.write_bytes(packed);compressed.append(dict(original=path.name,compressed=target.name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()));path.unlink()
    (out/'compressed-files.json').write_text(json.dumps(compressed,indent=2)+'\n')
    print(json.dumps(dict(cases=len(manifest['cases']),exit_codes=[c['exit_code'] for c in manifest['cases']],scope=manifest['scope'])),flush=True)


if __name__=='__main__':main()
