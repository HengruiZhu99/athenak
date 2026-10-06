"""Use the actual C++ reader against Python v1/v2 writers and bound sampler."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--native-root',type=Path,required=True)
    ap.add_argument('--library',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--polar-extent',type=int,default=8);args=ap.parse_args();native=args.native_root.resolve();library=args.library.resolve()
    root=Path(__file__).resolve().parents[3];out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(native/'python'))
    from hispid import Backend,Hole
    from checkpoint_export import write_checkpoint
    b=Backend(str(library));exe=out/'reader'
    command=['c++','-std=c++17','-O2','-I'+str(native/'include'),'-I'+str(root/'src/pgen/z4c'),
             str(Path(__file__).with_name('hispid_checkpoint_probe.cpp')),str(library),
             '-Wl,-rpath,'+str(library.parent),'-o',str(exe)]
    subprocess.run(command,check=True,capture_output=True,text=True)
    c=b.config();c.n[:]=[8,args.polar_extent,4];c.far_radius=0;c.inner_max[:]=[0,0];c.inner_flatten=0
    c.hole[0]=Hole(1,center=(0,0,0),spin=(.2,-.3,.7),velocity=(.25,.1,-.2));c.hole[1].mass=0
    xyz=np.array([[1.2,.7,-.4],[2,-1,.8]]);records=[]
    for family in (['qi','trumpet_r0_m'] if args.polar_extent==8 else ['trumpet_r0_m']):
        c.seed_family=family;p=out/(family+'.checkpoint')
        meta=write_checkpoint(p,c,np.zeros(4*np.prod(c.n)),b.library_sha256(),'analytic_seed')
        process=subprocess.run([str(exe),str(p),b.library_sha256()],capture_output=True,text=True,check=True,cwd=native)
        result=json.loads(process.stdout)
        with b.create_sampler(c) as s:
            v=s.sample_with_derivatives(xyz)
        expected=np.concatenate([np.concatenate([v['gamma'][i],v['Kij'][i]]) for i in range(2)]+[v['dgamma'].ravel()])
        actual=np.array(result['values']);error=float(np.max(np.abs(actual-expected)/(1+np.abs(expected))))
        assert error<1e-12 and result['seed_family']==(family=='trumpet_r0_m')
        records.append(dict(family=family,scaled_linf=error,metadata=meta))
    original=p.read_text();rejected=[]
    invalid_cases=[('unknown_family',original.replace('trumpet_r0_m','invalid')),
                       ('mislabelled_v1',original.replace('HISPID_CHECKPOINT 2','HISPID_CHECKPOINT 1')),
                       ('missing_family',original.replace('seed_family trumpet_r0_m\n',''))]
    if args.polar_extent!=8:
        invalid_cases=[('oversized_polar',original.replace(f'n 8 {args.polar_extent} 4', 'n 8 513 4'))]
    for label,text in invalid_cases:
        invalid=out/(label+'.checkpoint');invalid.write_text(text)
        process=subprocess.run([str(exe),str(invalid),b.library_sha256()],capture_output=True,text=True,cwd=native)
        assert process.returncode!=0
        rejected.append(dict(case=label,error=process.stderr.strip()))
    bound=[library,native/'include/HiSpID.h',root/'src/pgen/z4c/hispid_checkpoint.hpp',Path(__file__),Path(__file__).with_name('hispid_checkpoint_probe.cpp'),native/'python/checkpoint_export.py',native/'python/hispid.py',exe]
    result=dict(passed=True,scope='standalone_actual_reader_and_sampler_only',athenak_evolution_or_horizon_run=False,
                records=records,rejected=rejected,compile_command=command,execution_cwd=str(native),
                sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in bound})
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
