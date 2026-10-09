"""One-shot stdlib wrapper for the exact root-released scalar-values gate."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

P=Path(__file__).resolve().parent
R=P.parents[1]/'native-angular-pulse-flat-IVP-values-root-release-20261009'
INV=P/'values-invocation001'
OUT=P/'values-attempt001'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
dump=lambda p,v:Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def main():
    INV.mkdir(exist_ok=False)
    begin=time.monotonic()
    receipt={'kind':'One-shot released independent scalar-values outer invocation','accepted_native':False,'outer_argv':sys.argv,'outer_source_sha256':sha(__file__)}
    before={}
    try:
        auth=R/'authorization.json'
        review=R/'review.json'
        if sha(auth)!='10e13705ce6123fce25d5181ddd185b1360cb3133395bdb121961435798907bf':
            raise RuntimeError('root authorization hash mismatch')
        if sha(review)!='81578371d86263a4cc8490654b7ad78e0454f1570e2a66ceb832711442573297':
            raise RuntimeError('root review hash mismatch')
        recipe=json.loads((P/'values-recipe.json').read_text())
        authdata=json.loads(auth.read_text())
        before={**authdata['source_pins'],**recipe['dependency_pins'],**recipe['mpmath_python_pins'],recipe['python_runtime_path']:recipe['python_runtime_sha256'],str(auth):sha(auth),str(review):sha(review),str(Path(__file__).resolve()):sha(__file__)}
        for f,expected in before.items():
            if sha(f)!=expected:
                raise RuntimeError('protected preflight hash mismatch: '+f)
        if OUT.exists() or Path(authdata['fresh_output_path'])!=OUT:
            raise RuntimeError('fresh output path guard')
        context=INV/'source-before';context.mkdir()
        for i,f in enumerate([P/'flat_ivp_values.py',P/'VALUES-PLAN.md',P/'values-recipe.json',P/'PLAN.md',P/'KIRCHHOFF-PENCIL.md',auth,review,Path(__file__).resolve()]):
            (context/('%02d-'%i+f.name)).write_bytes(f.read_bytes())
        python=recipe['held_command'][0]
        argv=[python,'-B',str(P/'flat_ivp_values.py'),'--recipe',str(P/'values-recipe.json'),'--authorization',str(auth),'--output',str(OUT)]
        env=os.environ.copy()
        env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
        env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
        receipt.update(command=argv,environment={k:env.get(k) for k in ['PYTHONPATH','PYTHONHOME','PYTHONDONTWRITEBYTECODE','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS']},source_before=before,head_at_launch=subprocess.check_output(['git','rev-parse','HEAD'],cwd=P.parents[2],text=True).strip())
        dump(INV/'launch.json',receipt)
        with (INV/'stdout.log').open('wb') as stdout,(INV/'stderr.log').open('wb') as stderr:
            child=subprocess.Popen(argv,cwd=P.parents[2],env=env,stdout=stdout,stderr=stderr)
            receipt['pid']=child.pid
            dump(INV/'running.json',receipt)
            print('launched',child.pid,str(OUT),flush=True)
            receipt['returncode']=child.wait()
        if receipt['returncode']!=0:
            receipt['passed_outer_process']=False
        else:
            child_receipt=json.loads((OUT/'receipt.json').read_text())
            receipt['passed_outer_process']=bool(child_receipt.get('passed_scalar_values_only') and child_receipt.get('sources_unchanged'))
        receipt['scientific_receipt_sha256']=sha(OUT/'receipt.json') if (OUT/'receipt.json').is_file() else None
    except BaseException as exc:
        receipt.update(passed_outer_process=False,exception_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    finally:
        def safe(f):
            try:return sha(f)
            except BaseException as exc:return 'READ_ERROR:'+type(exc).__name__+':'+str(exc)
        receipt['source_after']={f:safe(f) for f in before}
        receipt['sources_unchanged']=receipt['source_after']==before
        if not receipt['sources_unchanged']:receipt['passed_outer_process']=False
        receipt['seconds']=time.monotonic()-begin
        receipt['stdout_sha256']=safe(INV/'stdout.log')
        receipt['stderr_sha256']=safe(INV/'stderr.log')
        dump(INV/'receipt.json',receipt)
        print('completed',receipt.get('returncode'),receipt.get('passed_outer_process'),flush=True)
    if not receipt.get('passed_outer_process'):raise SystemExit(1)


if __name__=='__main__':main()
