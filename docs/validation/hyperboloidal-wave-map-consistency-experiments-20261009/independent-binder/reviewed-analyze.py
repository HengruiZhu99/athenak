"""Saved48-row actual-RHS comparison only; no source API call."""
from pathlib import Path
import hashlib,json,math,sys
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def analyze(stdout,expected_path,out):
    actual=[json.loads(line)for line in Path(stdout).read_text().splitlines()]
    expected=json.loads(Path(expected_path).read_text());assert len(actual)==len(expected)==48
    def finite(x):
        if isinstance(x,(int,float)):assert math.isfinite(x)
        elif isinstance(x,list):
            for v in x:finite(v)
        elif isinstance(x,dict):
            for v in x.values():finite(v)
    maxima={k:0. for k in ['rhs22_scaled','rhs22_absolute','connection_scaled','source_scaled','physical_constraints_absolute','input_normals_absolute','rate_normals_scaled','rate_normals_absolute','omega_difference_absolute']}
    worst={};records=[]
    for a,e in zip(actual,expected):
        finite(a);finite(e);assert a['point']==e['point']
        assert len(a['actual_rhs22'])==len(e['rhs22'])==22
        re=[abs(x-y)/max(1.,abs(y))for x,y in zip(a['actual_rhs22'],e['rhs22'])]
        ra=[abs(x-y)for x,y in zip(a['actual_rhs22'],e['rhs22'])]
        ce=[abs(x-y)/max(1.,abs(y))for x,y in zip(a['scaled_reference_connection36'],e['scaled_reference_connection36'])]
        se=[abs(x-y)/max(1.,abs(y))for x,y in zip(a['scaled_source4'],e['scaled_source4'])]
        assert len(ce)==36 and len(se)==4
        absmax=lambda xs:max(abs(x)for x in xs)
        vals={'rhs22_scaled':max(re),'rhs22_absolute':max(ra),'connection_scaled':max(ce),'source_scaled':max(se),
          'physical_constraints_absolute':absmax(a['physical_constraints8']),'input_normals_absolute':absmax(a['input_normals2']),
          'rate_normals_absolute':absmax(a['rate_normals2']),
          'rate_normals_scaled':absmax(a['rate_normals2'])/max(1.,absmax(a['actual_rhs22'])),
          'omega_difference_absolute':absmax(a['submitted_minus_native_omega13'])}
        for key,v in vals.items():
            if v>maxima[key]:maxima[key]=v;worst[key]={'index':e['index'],'point_name':e['point_name'],'epsilon':e['epsilon'],'value':v}
        records.append({'index':e['index'],'point_name':e['point_name'],'point':e['point'],'epsilon':e['epsilon'],
          'rhs22_scaled_errors':re,'rhs22_absolute_errors':ra,'connection_scaled_errors':ce,'source_scaled_errors':se,'metrics':vals})
    tol={'rhs22_scaled':5e-9,'connection_scaled':5e-9,'source_scaled':5e-9,'physical_constraints_absolute':5e-9,'input_normals_absolute':5e-11,'rate_normals_scaled':5e-11,'omega_difference_absolute':2e-10}
    checks={k:maxima[k]<=v for k,v in tol.items()}
    receipt={'passed':all(checks.values()),'scope':'finiteOmega nonlinear exact-flat actual48point source consistency; no evolution',
      'source_sha256':sha(__file__),'actual_stdout_sha256':sha(stdout),'expected_sha256':sha(expected_path),
      'cases':48,'maxima':maxima,'worst':worst,'thresholds':tol,'checks':checks,
      'raw22_component_scaled_max':[max(r['rhs22_scaled_errors'][c]for r in records)for c in range(22)]}
    out=Path(out);out.mkdir(exist_ok=False);(out/'comparisons.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n');(out/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    return receipt
if __name__=='__main__':
    r=analyze(*sys.argv[1:]);print(json.dumps(r,indent=2));sys.exit(0 if r['passed']else 1)
