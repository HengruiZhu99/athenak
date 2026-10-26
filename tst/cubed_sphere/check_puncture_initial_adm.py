#!/usr/bin/env python3
"""Recheck the initial native physical ADM dump, without rerunning evolution."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def function(path,name,namespace):
    node=next(n for n in ast.parse(path.read_text()).body
              if isinstance(n,ast.FunctionDef) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('case',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();root=Path(__file__).resolve().parents[2]
    reader_path=root/'vis/python/bin_convert.py'
    model_path=root/'tst/test_suite/z4c/test_hyperboloidal_wormhole_cpu.py'
    reader=function(reader_path,'read_binary',{'np':np,'os':os})
    model=function(model_path,'initial_geometry',{'np':np})
    def read(kind,number):
        path=a.case/'bin'/f'hyp.{kind}.{number:05}.bin'
        return {k:np.asarray(v) for k,v in reader(str(path))['mb_data'].items()}
    u=read('z4c',0);adm=read('adm',0);final=read('z4c',1)
    mask=u['z4c_active'].astype(bool);m=model(mask.shape,.5)
    radial_metric=np.zeros_like(m['r']);radial_k=np.zeros_like(m['r'])
    for i,j,name in [(0,0,'xx'),(0,1,'xy'),(0,2,'xz'),(1,1,'yy'),(1,2,'yz'),(2,2,'zz')]:
        nn=m['direction'][i]*m['direction'][j];mult=1 if i==j else 2
        radial_metric+=mult*nn*adm['adm_g'+name]
        radial_k+=mult*nn*adm['adm_K'+name]
    trace_metric=sum(adm['adm_g'+n] for n in ['xx','yy','zz'])
    trace_k=sum(adm['adm_K'+n] for n in ['xx','yy','zz'])
    tangent_metric=(trace_metric-radial_metric)/2
    tangent_k=(trace_k-radial_k)/(2*tangent_metric)
    areal=m['r']*np.sqrt(tangent_metric)
    areal_d=(1-m['m']**2)*m['ell']/m['omega']**2
    invariant=areal*(1+areal**2*tangent_k**2-areal_d**2/radial_metric)/2
    probe=mask & (m['r']>.15) & (m['r']<.7)
    np.testing.assert_allclose(invariant[probe],.5,rtol=2e-5,atol=2e-6)
    core=mask & (m['r']<.35)
    changes={k:float(np.max(abs(final['z4c_'+k][core]-u['z4c_'+k][core])))
             for k in ['alpha','Khat']}
    assert min(changes.values())>1e-8
    result={'active_cells':int(mask.sum()),'mass_probe_cells':int(probe.sum()),
        'mass_expected':.5,'mass_max_absolute_error':float(np.max(abs(invariant[probe]-.5))),
        'mass_tolerance':{'rtol':2e-5,'atol':2e-6},'initial_physical_mass_check_pass':True,
        'two_step_core_field_changes':changes,
        'source_sha256':{str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest()
                         for f in [reader_path,model_path,Path(__file__).resolve()]}}
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
