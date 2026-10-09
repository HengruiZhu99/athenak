"""Fresh fixed-recipe Release/ASan-UBSan builds and48point queries only."""
from pathlib import Path
import hashlib,json,shlex,shutil,subprocess,sys,time
from analyze import analyze
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert (HERE/'execution-release.json').is_file(),'independent review/release required'
    recipe=json.loads((HERE/'execution-recipe.json').read_text())
    for path,h in recipe['inputs'].items():assert sha(ROOT/path)==h,path
    A=HERE/'attempts'/str(time.time_ns());A.mkdir(parents=True,exist_ok=False)
    for name in ['probe.cpp','reference_wave_map.hpp','PLAN.md','prepare_input.py','analyze.py','run_gate.py','execution-recipe.json','execution-release.json']:
        shutil.copyfile(HERE/name,A/name)
    base=ROOT/'build-layer-research/continuum/reference-wave-map-gauge-20261009/immutable-local-reference-wave-map-20261009/attempts/1791563579962029000/receipt.json'
    commands=json.loads(base.read_text())['commands'];receipt={'passed':False,'launch_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      'production_implementation':'27c19d20696ea6dd4704032c51dfd026218f64f2','recipe_sha256':sha(HERE/'execution-recipe.json'),
      'execution_release_sha256':sha(HERE/'execution-release.json'),'source_before':recipe['inputs'],'commands':[],'modes':{},'scope':'local source48point only; no evolution or operators'}
    start=time.monotonic()
    def command(name,cmd,data=None):
        st=time.monotonic();q=subprocess.run(cmd,cwd=ROOT,input=data,capture_output=True)
        (A/(name+'.stdout')).write_bytes(q.stdout);(A/(name+'.stderr')).write_bytes(q.stderr)
        r={'name':name,'command':cmd,'exit_code':q.returncode,'seconds':time.monotonic()-st,'stdout_sha256':hashlib.sha256(q.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(q.stderr).hexdigest()};receipt['commands'].append(r)
        assert q.returncode==0 and not q.stderr,r;return q
    try:
        compiler=command('compiler-version',['/usr/bin/c++','--version']);receipt['compiler_sha256']=sha(Path('/usr/bin/c++').resolve())
        for mode in ['release','debug']:
            old=next(q['command']for q in commands if '-std=c++17'in q['command']and ('-O3'in q['command'])==(mode=='release'))
            cmd=[]
            for arg in old:
                if arg.endswith('/local_gate.cpp'):arg=str(A/'probe.cpp')
                elif arg.endswith('/local-'+mode):arg=str(A/('probe-'+mode))
                elif arg.endswith('/'+mode+'.d'):arg=str(A/(mode+'.d'))
                cmd.append(arg)
            assert str(A/'probe.cpp')in cmd and str(A/('probe-'+mode))in cmd
            command('compile-'+mode,cmd)
            text=(A/(mode+'.d')).read_text().replace('\\\n',' ')
            paths=sorted(set(str(Path(v).resolve())for v in shlex.split(text.split(':',1)[1])))
            deps={p:sha(p)for p in paths};exe=A/('probe-'+mode)
            build={'command':cmd,'executable_sha256':sha(exe),'compiler_dependencies':deps,'compiler_dependencies_count':len(deps)}
            (A/('build-'+mode+'.json')).write_text(json.dumps(build,indent=2,allow_nan=False)+'\n')
            for path,h in recipe['inputs'].items():assert sha(ROOT/path)==h,path
            command('run-'+mode,[str(exe)],(HERE/'prepared-input001/input.txt').read_bytes())
            result=analyze(A/('run-'+mode+'.stdout'),HERE/'prepared-input001/expected.json',A/('analysis-'+mode))
            receipt['modes'][mode]={'build_receipt_sha256':sha(A/('build-'+mode+'.json')),'analysis_receipt_sha256':sha(A/('analysis-'+mode+'/receipt.json')),'passed':result['passed'],'maxima':result['maxima']}
            assert result['passed'],result
        receipt['release_debug_equal']=(A/'run-release.stdout').read_bytes()==(A/'run-debug.stdout').read_bytes();assert receipt['release_debug_equal']
        receipt['source_after']={p:sha(ROOT/p)for p in recipe['inputs']};assert receipt['source_after']==receipt['source_before'];receipt['passed']=True
    except Exception as e:receipt['exception']=repr(e)
    receipt['seconds']=time.monotonic()-start;(A/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n');print(json.dumps({'attempt':str(A),'receipt_sha256':sha(A/'receipt.json'),**receipt},indent=2));sys.exit(0 if receipt['passed']else 1)
if __name__=='__main__':main()
