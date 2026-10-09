"""Reproducible scratch-only actual lower-order lapse source gate."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time


HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    prior=json.loads((HERE.parent/'covariant-z4-candidate/receipt.json').read_text())
    flags=prior['commands'][0]['command'][:-3]
    debug=flags.copy();debug[debug.index('-O3')]='-O1';debug.remove('-DNDEBUG')
    debug+=['-g','-fsanitize=address,undefined','-fno-omit-frame-pointer']
    tracked=[ROOT/f for f in subprocess.check_output(['git','ls-files','src','CMakeLists.txt'],cwd=ROOT,text=True).splitlines()]
    own=[HERE/f for f in ('inner_lapse_advection.hpp','helpers.hpp','full20.cpp','nonlinear.cpp','principal.cpp','kernel_symbol_copy.cpp','run_audit.py','check.py')]
    external=[HERE.parent/'discrete-bianchi/immutable-discrete-bulk-20261009/dual_helpers.hpp',
              ROOT/'tst/hyperboloidal/kernel_symbol.cpp',ROOT/'tst/hyperboloidal/check_kernel_symbol.py']
    inputs=tracked+own+external
    before={str(f.relative_to(ROOT)):sha(f) for f in inputs}
    commands=[(flags+[str(HERE/(name+'.cpp')),'-o',str(HERE/name)],None) for name in ('full20','nonlinear','principal')]
    commands.append((debug+[str(HERE/'nonlinear.cpp'),'-o',str(HERE/'nonlinear_debug')],None))
    commands += [([str(HERE/'nonlinear')],'nonlinear.json'),([str(HERE/'nonlinear_debug')],'nonlinear-debug.json'),
                 ([str(HERE/'full20'),'pole'],'poles.json'),([str(HERE/'full20')],'full20.json'),
                 ([sys.executable,str(ROOT/'tst/hyperboloidal/check_kernel_symbol.py'),str(HERE/'principal')],'principal.log'),
                 ([sys.executable,str(HERE/'check.py')],'check.log')]
    results=[];started=time.monotonic()
    for command,output in commands:
        start=time.monotonic()
        if output:
            with (HERE/output).open('w') as stream:
                result=subprocess.run(command,cwd=ROOT,text=True,stdout=stream,stderr=subprocess.PIPE)
        else:result=subprocess.run(command,cwd=ROOT,text=True,capture_output=True)
        row={'command':command,'returncode':result.returncode,'seconds':time.monotonic()-start,'stderr':result.stderr}
        if output:row.update(stdout_file=output,stdout_sha256=sha(HERE/output))
        else:row['stdout']=result.stdout
        results.append(row)
        (HERE/'commands-in-progress.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
        if result.returncode:
            (HERE/'failed-receipt.json').write_text(json.dumps({'source_before':before,'commands':results},indent=2)+'\n')
            result.check_returncode()
    after={str(f.relative_to(ROOT)):sha(f) for f in inputs}
    assert before==after
    receipt={'passed_lower_order_lapse_local_gates':True,'native_global_or_scri_stability_accepted':False,
        'launch_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'runtime_implementation':'27c19d20696ea6dd4704032c51dfd026218f64f2',
        'source_before':before,'source_after':after,'sources_unchanged':True,'commands':results,
        'binary_sha256':{name:sha(HERE/name) for name in ('full20','nonlinear','nonlinear_debug','principal')},
        'compiler':subprocess.check_output(['/usr/bin/c++','--version'],text=True),
        'python':sys.version,'numpy':__import__('numpy').__version__,'seconds':time.monotonic()-started}
    (HERE/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print('PASS actual lower-order lapse gate',sha(HERE/'receipt.json'))


if __name__=='__main__':main()
