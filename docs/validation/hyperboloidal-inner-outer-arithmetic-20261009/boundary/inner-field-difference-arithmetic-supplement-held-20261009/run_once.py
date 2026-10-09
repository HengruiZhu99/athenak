#!/usr/bin/env python3
"""HELD one-shot supplement build/probe/Fraction gate, after actual-main PASS."""
import argparse,os,shlex,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from gate_context import admitted,guard,pin,read,write_new

def main():
    p=argparse.ArgumentParser();p.add_argument('--authorization',required=True);p.add_argument('--build',choices=['release','debug'],required=True);args=p.parse_args()
    if not(sys.flags.isolated and sys.dont_write_bytecode and not sys.flags.optimize):raise RuntimeError('Require -I -B unoptimized')
    here=Path(__file__).resolve().parent;recipe,index,protected=admitted(here,args.authorization,args.build)
    for k,v in recipe['environment'].items():
        if os.environ.get(k)!=v:raise RuntimeError('Fixed environment '+k)
    attempt=here/'attempts'/recipe['attempt_names'][args.build];attempt.mkdir(parents=True,exist_ok=False)
    commands=[];receipt={'completed':False,'passed':False,'returncode':1,'build':args.build,'scope':'separate fixed scalar field-difference supplement only','commands':commands,'source_index_sha256':pin(here/'source-index.json')['sha256'],'recipe_sha256':pin(here/'recipe.json')['sha256'],'actual_main003_release_receipt':pin(recipe['actual_main003_release_receipt'])}
    def command(name,cmd):
        start=time.monotonic();out,err=attempt/(name+'.stdout'),attempt/(name+'.stderr')
        with out.open('wb') as stdout,err.open('wb') as stderr:done=subprocess.run(cmd,cwd=recipe['repository'],env=os.environ.copy(),stdout=stdout,stderr=stderr)
        commands.append({'name':name,'command':cmd,'seconds':time.monotonic()-start,'returncode':done.returncode,'stdout':pin(out),'stderr':pin(err)})
        if done.returncode:raise RuntimeError('Command failed '+name)
        return out
    try:
        command('compiler-version',[recipe['compiler'],'--version']);command('launch-HEAD',['git','rev-parse','HEAD']);command('python-version',[recipe['python'],'-I','-B','--version'])
        exe=attempt/('field-difference-'+args.build);dep=attempt/'dependencies.d'
        command('compile',[recipe['compiler']]+recipe['compile_flags'][args.build]+['-MD','-MF',str(dep),str(here/'probe.cpp'),'-o',str(exe)])
        known={row['path']:row for row in protected+index['files']};dependencies=[]
        for token in shlex.split(dep.read_text().replace('\\\n',' '))[1:]:
            path=Path(token);path=path if path.is_absolute() else Path(recipe['repository'])/path
            row=pin(path)
            if known.get(row['path'])!=row:raise RuntimeError('New/unpinned compiler dependency '+row['path'])
            dependencies.append(row)
        write_new(attempt/'dependencies.json',dependencies);receipt['executable_before']=pin(exe)
        output=command('probe',[str(exe),'--all']);(attempt/'probe.jsonl').write_bytes(output.read_bytes())
        command('oracle',[recipe['python'],'-I','-B',str(here/'oracle.py'),str(Path(args.authorization).resolve()),args.build,str(attempt)])
        report=read(attempt/'supplement-report.json')
        if report.get('passed') is not True:raise RuntimeError('Supplement oracle FAIL')
        guard(protected);guard(index['files']);guard(dependencies)
        receipt['executable_after']=pin(exe)
        if receipt['executable_after']!=receipt['executable_before']:raise RuntimeError('Executable changed')
        receipt.update(completed=True,passed=True,returncode=0,inputs_unchanged=True,report=pin(attempt/'supplement-report.json'))
    except BaseException as error:
        receipt['failure']={'type':type(error).__name__,'message':str(error)}
        try:guard(protected);guard(index['files']);receipt['inputs_unchanged']=True
        except BaseException as drift:receipt['input_guard_failure']=str(drift)
    finally:
        receipt['output_inventory']=[pin(x) for x in sorted(attempt.rglob('*')) if x.is_file()];write_new(attempt/'receipt.json',receipt)
    print(str(attempt));return receipt['returncode']

if __name__=='__main__':raise SystemExit(main())
