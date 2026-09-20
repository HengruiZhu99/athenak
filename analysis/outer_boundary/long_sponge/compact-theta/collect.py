"""Collect completed local compact/Gaussian controls; never launches evolution."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'gpu'))
sys.path.insert(0,str(HERE.parent))
from check_minkowski_checkpoint import validate
from radial_profiles import extract

def history(path):
    names=re.findall(r'\[\d+\]=(\S+)',path.read_text().splitlines()[1])
    data=np.atleast_2d(np.loadtxt(path));assert np.isfinite(data).all()
    return dict(zip(names,data.T))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('study',type=Path);p.add_argument('--output',type=Path,default=HERE);p.add_argument('--cases',nargs='+',default=['compact','gaussian','compact_amp1e7','compact_halfdt']);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);result={}
    for case in a.cases:
        run=a.study/case
        assert (run/'execution.json').exists(),f'{case} has no finished execution record'
        validation=validate(run,8);assert validation['passed']
        dst=a.output/case;dst.mkdir(exist_ok=True)
        (dst/'checkpoint-validation.json').write_text(json.dumps(validation,indent=2)+'\n')
        profile=extract(run,8,validation['cycle'],bin_width=128.)
        (dst/'final-profile.json').write_text(json.dumps(profile,indent=2)+'\n')
        for f in ['input.athinput','execution.json','exit_code.txt']:
            shutil.copyfile(run/f,dst/f)
        histories={}
        for path in sorted(run.glob('*.hst')):
            h=history(path);np.savez_compressed(dst/(path.name+'.npz'),**h)
            histories[path.name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'final':{k:float(v[-1])for k,v in h.items()}}
        user=next(v for k,v in histories.items()if k.endswith('.user.hst')and not k.endswith('.z4c.user.hst'))['final']
        z=next(v for k,v in histories.items()if k.endswith('.z4c.user.hst'))['final']
        result[case]={'time':validation['time_code'],'cycle':validation['cycle'],'stopping_reason':validation['stopping_reason'],'target_reached':validation['target_reached'],'passed':True,'theta_max':user['Theta-max'],'theta_exterior_rms':float(np.sqrt(z['Theta-norm']/z['Volume'])),'theta_core_l2':float(np.sqrt(z['Theta-int2'])),'alpha_res':user['alpha-res'],'beta_res':user['beta-res'],'histories':histories}
    (a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:{f:v for f,v in d.items()if f!='histories'}for k,d in result.items()},indent=2))

if __name__=='__main__':main()
