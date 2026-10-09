"""Single-use exact released compile + t0 native seam only. No evolution."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
    auth_path=HERE/'probe-seam-authorization.json';auth=json.loads(auth_path.read_text())
    assert auth['compile_and_seam_authorized'] and not auth['snapshot_or_evolution_authorized']
    assert sha(Path(__file__))==auth['runner_sha256']
    rp=HERE/'probe-build-held/recipe.json';assert sha(rp)==auth['recipe_sha256']
    r=json.loads(rp.read_text());source=HERE/'native_seam_and_snapshot.cpp'
    assert sha(source)==auth['probe_source_sha256']
    cases=HERE/'seam-cases.json';assert sha(cases)==auth['seam_cases_sha256']
    out=HERE/'probe-attempts/compile-and-seam-001';out.mkdir(parents=True,exist_ok=False)
    for name,p in [('runner.py',Path(__file__)),('recipe.json',rp),('authorization.json',auth_path),
                   ('native_seam_and_snapshot.cpp',source),('seam-cases.json',cases)]:
        (out/name).write_bytes(p.read_bytes())
    start=time.monotonic();commands=[];before={};deps={}
    try:
        for name,digest in r['source_before'].items():assert sha(Path(name))==digest,name;before[name]=digest
        for name in subprocess.check_output(['git','ls-files','src','CMakeLists.txt'],cwd=ROOT,text=True).splitlines():
            p=ROOT/name;before[str(p)]=sha(p)
        for mode,entry in r['native_build_receipts'].items():
            assert sha(Path(entry['path']))==entry['sha256']
            assert sha(Path(entry['executable']))==entry['executable_sha256']
            before[entry['path']]=entry['sha256'];before[entry['executable']]=entry['executable_sha256']
        exe=Path(r['executable']);assert not exe.exists()
        for phase,command in [('compile',r['command']),('seam',[str(exe),'--seam'])]:
            stdout=out/(phase+'.stdout');stderr=out/(phase+'.stderr')
            record={'phase':phase,'command':command,'cwd':r['cwd'],
                    'stdout':stdout.name,'stderr':stderr.name}
            dump(out/(phase+'-command.json'),record);ts=time.monotonic()
            with stdout.open('wb') as so,stderr.open('wb') as se:
                result=subprocess.run(command,cwd=r['cwd'],stdout=so,stderr=se)
            record.update(returncode=result.returncode,seconds=time.monotonic()-ts,
                          stdout_sha256=sha(stdout),stderr_sha256=sha(stderr));commands.append(record)
            assert result.returncode==0,(phase,result.returncode)
            assert stderr.read_bytes()==b'',(phase,'stderr')
            if phase=='compile':
                assert stdout.read_bytes()==b''
                for token in shlex.split(Path(r['depfile']).read_text().replace('\\\n',' '))[1:]:
                    p=Path(token);p=p if p.is_absolute() else ROOT/p
                    assert p.is_file();deps[str(p)]=sha(p)
            else:
                result_json=json.loads(stdout.read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
                assert len(result_json['rows'])==18
                assert result_json['max_scaled_error']<=2e-12
                assert result_json['reference_rhs_abs']<=1e-10
                assert result_json['omit_beta_negative_control_abs']>1e-8
                assert result_json['duplicate_alpha_negative_control_abs']>1e-8
        for name,digest in before.items():assert sha(Path(name))==digest,name
        for name,digest in deps.items():assert sha(Path(name))==digest,name
        dump(out/'receipt.json',{'passed_compile_and_fixed_t0_seam':True,
          'launch_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'compiled_implementation':r['compiled_implementation'],'recipe_sha256':sha(rp),
          'authorization_sha256':sha(auth_path),'runner_sha256':sha(Path(__file__)),
          'source_before':before,'source_after':before,'all_compiler_dependency_sha256':deps,
          'commands':commands,'probe_executable':str(exe),'probe_executable_sha256':sha(exe),
          'seam_cases_sha256':sha(cases),'row_count':len(result_json['rows']),
          'max_scaled_error':result_json['max_scaled_error'],
          'reference_rhs_abs':result_json['reference_rhs_abs'],
          'omit_beta_negative_control_abs':result_json['omit_beta_negative_control_abs'],
          'duplicate_alpha_negative_control_abs':result_json['duplicate_alpha_negative_control_abs'],
          'seconds':time.monotonic()-start,'snapshot_executed':False,'native_evolution_executed':False,
          'scope':'Actual CartesianPatch arrays/RHS at t0 only; manual all22 comparator. No full Mesh initializer invocation, task graph time step, snapshot or evolution.'})
        print('PASS native t0 seam',result_json['max_scaled_error'],sha(exe))
    except Exception as error:
        dump(out/'failure.json',{'passed_compile_and_fixed_t0_seam':False,'error':str(error),
           'source_before':before,'commands':commands,'runner_sha256':sha(Path(__file__)),
           'authorization_sha256':sha(auth_path),'recipe_sha256':sha(rp),
           'seconds':time.monotonic()-start,'native_evolution_executed':False})
        raise
if __name__=='__main__':main()
