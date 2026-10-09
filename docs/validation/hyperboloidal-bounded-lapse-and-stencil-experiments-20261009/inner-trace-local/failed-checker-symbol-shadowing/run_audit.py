"""Fresh scratch actual-kernel trace-only/combined gate, no native build."""
from pathlib import Path
import hashlib,json,subprocess,sys,time

P=Path(__file__).resolve().parent;ROOT=P.parents[2]
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    prior=json.loads((P.parent/'covariant-z4-candidate/receipt.json').read_text())
    flags=prior['commands'][0]['command'][:-3]
    debug=flags.copy();debug[debug.index('-O3')]='-O1';debug.remove('-DNDEBUG')
    debug+=['-g','-fsanitize=address,undefined','-fno-omit-frame-pointer']
    files=[ROOT/name for name in subprocess.check_output(['git','ls-files','src','CMakeLists.txt'],cwd=ROOT,text=True).splitlines()]
    files += [P/name for name in ('inner_conformal_trace.hpp','inner_lapse_advection.hpp','helpers.hpp','full20.cpp','nonlinear.cpp','principal.cpp','kernel_symbol_copy.cpp','check.py','run_audit.py')]
    files += [P.parent/'discrete-bianchi/immutable-discrete-bulk-20261009/dual_helpers.hpp',ROOT/'tst/hyperboloidal/kernel_symbol.cpp',ROOT/'tst/hyperboloidal/check_kernel_symbol.py',P.parent/'inner-lapse-advection-control/full20.json']
    before={str(path.relative_to(ROOT)):sha(path) for path in files}
    commands=[(flags+[str(P/(name+'.cpp')),'-o',str(P/name)],None) for name in ('full20','nonlinear')]
    commands += [(flags+['-DCOMBINED_ADVECTION='+str(mode),str(P/'principal.cpp'),'-o',str(P/name)],None) for mode,name in ((0,'principal-trace'),(1,'principal-combined'))]
    commands += [(debug+[str(P/'nonlinear.cpp'),'-o',str(P/'nonlinear_debug')],None),
                 ([str(P/'nonlinear')],'nonlinear.json'),([str(P/'nonlinear_debug')],'nonlinear-debug.json'),
                 ([str(P/'full20'),'pole'],'poles.json'),([str(P/'full20')],'full20.json')]
    commands += [([sys.executable,str(ROOT/'tst/hyperboloidal/check_kernel_symbol.py'),str(P/name)],name+'.log') for name in ('principal-trace','principal-combined')]
    commands += [([sys.executable,str(P/'check.py')],'check.log')]
    results=[];started=time.monotonic()
    for command,output in commands:
        start=time.monotonic()
        if output:
            with (P/output).open('w') as stream:process=subprocess.run(command,cwd=ROOT,text=True,stdout=stream,stderr=subprocess.PIPE)
        else:process=subprocess.run(command,cwd=ROOT,text=True,capture_output=True)
        row={'command':command,'returncode':process.returncode,'stderr':process.stderr,'seconds':time.monotonic()-start}
        if output:row.update(stdout_file=output,stdout_sha256=sha(P/output))
        else:row['stdout']=process.stdout
        results.append(row);(P/'commands-in-progress.json').write_text(json.dumps(results,indent=2)+'\n')
        if process.returncode:
            (P/'failed-receipt.json').write_text(json.dumps({'source_before':before,'commands':results},indent=2)+'\n')
            process.check_returncode()
    after={str(path.relative_to(ROOT)):sha(path) for path in files};assert before==after
    receipt={'status':'PASS','scope':'Scratch finiteOmega mathematical/actualtensor gates only; no native/global/stability/BH admission','commands':results,
      'source_before':before,'source_after':after,'sources_unchanged':True,'launch_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      'runtime_implementation':'27c19d20696ea6dd4704032c51dfd026218f64f2','seconds':time.monotonic()-started,
      'helper_sha256':sha(P/'inner_conformal_trace.hpp'),'check_report_sha256':sha(P/'check-report.json'),
      'binary_sha256':{name:sha(P/name) for name in ('full20','nonlinear','nonlinear_debug','principal-trace','principal-combined')}}
    (P/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print('PASS trace/combined local gate',sha(P/'receipt.json'))


if __name__=='__main__':main()
