"""Single-use compile/link executor for the already prepared exact held recipes.

No native process is launched. An exact parent compile authorization is required.
"""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x): p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def check_index(p,digest):
    assert sha(p)==digest,p
    d=json.loads(p.read_text());entries=d['files']
    pairs=entries.items() if isinstance(entries,dict) else ((x['path'],x) for x in entries)
    for name,row in pairs:
        assert sha(p.parent/name)==(row if isinstance(row,str) else row['sha256']),name

def main():
    mode=sys.argv[1]; assert mode in ['wave-map','c0','wave-map-half']
    auth_path=Path(sys.argv[2]).resolve();auth=json.loads(auth_path.read_text())
    assert auth['compilation_authorized'] is True and auth['native_evolution_authorized'] is False
    assert sha(Path(__file__))==auth['executor_sha256']
    p=HERE/'recipes'/mode/'recipe.json'
    assert sha(p)==auth['recipes_sha256'][mode]
    r=json.loads(p.read_text())
    target=HERE/'build-attempts'/(mode+'-001');target.mkdir(parents=True,exist_ok=False)
    (target/'executor-source.py').write_bytes(Path(__file__).read_bytes())
    (target/'exact-recipe.json').write_bytes(p.read_bytes())
    (target/'compile-authorization.json').write_bytes(auth_path.read_bytes())
    started=time.monotonic();before={};commands=[];deps={};outputs=[]
    def remember(path,digest):
        path=Path(path);assert sha(path)==digest,path
        before[str(path)]=digest
    try:
        assert r['compiled_implementation']=='27c19d20696ea6dd4704032c51dfd026218f64f2'
        for name,d in r['base_inputs_sha256'].items(): remember(ROOT/name,d)
        for name,d in r['base_source_sha256'].items(): remember(ROOT/name,d)
        family=ROOT/'build-layer-research/continuum/preferred/native-overlay/spatial-norm-family'
        for name,d in r['base_overlay_sha256'].items(): remember(family/name,d)
        for name,d in r['private_sources_sha256'].items(): remember(ROOT/name,d)
        for name,item in r['base_link_inputs'].items(): remember(name,item['sha256'])
        remember(HERE/'prepare_build.py',r['prepare_script_sha256'])
        additive=HERE/'additive-independent-prerequisites.json'
        remember(additive,auth['additive_independent_prerequisites_sha256'])
        reviews=json.loads(additive.read_text())
        for key in ['independent_principal_review','independent_finite_RHS_readback']:
            item=reviews[key];check_index(ROOT/item['path'],item['sha256'])
        assert len(r['compile_commands'])==6
        assert all(not Path(c['output']).exists() for c in r['compile_commands'])
        assert not Path(r['executable']).exists()
        dump(target/'precompile-pins.json',before)
        compiler=Path(r['compile_commands'][0]['command'][0])
        compiler_version=subprocess.check_output([str(compiler),'--version'],text=True)
        for k,row in enumerate(r['compile_commands']):
            command=row['command'];assert '-include' not in command
            assert not any('spatial-norm-family' in arg for arg in command)
            out=target/f'compile-{k}.stdout';err=target/f'compile-{k}.stderr'
            attempt={'phase':'compile','number':k,'command':command,'cwd':row['cwd'],
                     'stdout':out.name,'stderr':err.name}
            dump(target/f'compile-{k}-command.json',attempt)
            ts=time.monotonic()
            with out.open('wb') as so,err.open('wb') as se:
                result=subprocess.run(command,cwd=row['cwd'],stdout=so,stderr=se)
            attempt.update(returncode=result.returncode,seconds=time.monotonic()-ts,
                           stdout_sha256=sha(out),stderr_sha256=sha(err))
            commands.append(attempt)
            assert result.returncode==0,('compile failed',k,result.returncode)
            assert out.read_bytes()==b'' and err.read_bytes()==b'',('compiler diagnostics',k)
            obj=Path(row['output']);dep=Path(row['depfile'])
            tokens=shlex.split(dep.read_text().replace('\\\n',' '))
            for name in tokens[1:]:
                dp=Path(name)
                if not dp.is_absolute(): dp=(Path(row['cwd'])/dp).resolve()
                assert dp.is_file(),dp
                deps[str(dp)]=sha(dp)
            outputs.append({'object':str(obj),'sha256':sha(obj),
                            'depfile':str(dep),'depfile_sha256':sha(dep)})
            print(mode,'compiled',k+1,'of 6',flush=True)
        out=target/'link.stdout';err=target/'link.stderr'
        attempt={'phase':'link','command':r['link_command'],'cwd':r['link_cwd'],
                 'stdout':out.name,'stderr':err.name}
        dump(target/'link-command.json',attempt);ts=time.monotonic()
        with out.open('wb') as so,err.open('wb') as se:
            result=subprocess.run(r['link_command'],cwd=r['link_cwd'],stdout=so,stderr=se)
        attempt.update(returncode=result.returncode,seconds=time.monotonic()-ts,
                       stdout_sha256=sha(out),stderr_sha256=sha(err));commands.append(attempt)
        assert result.returncode==0,('link failed',result.returncode)
        assert out.read_bytes()==b'' and err.read_bytes()==b'', 'link diagnostics'
        for name,d in before.items(): assert sha(Path(name))==d,('input drift',name)
        for name,d in deps.items(): assert sha(Path(name))==d,('dependency drift',name)
        exe=Path(r['executable'])
        receipt={'passed_compile_link':True,'mode':mode,'half_step':r['half_step'],
          'launch_HEAD':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'compiled_implementation':r['compiled_implementation'],
          'recipe_sha256':sha(p),'authorization_sha256':sha(auth_path),
          'executor_sha256':sha(Path(__file__)),'compiler_sha256':sha(compiler),
          'compiler_version':compiler_version,'source_before':before,'source_after':before,
          'all_compiler_dependency_sha256':deps,'compiled_objects':outputs,'commands':commands,
          'executable':str(exe),'executable_sha256':sha(exe),
          'base_private_inputs_and_dependencies_unchanged':True,
          'seconds':time.monotonic()-started,'native_executed':False,
          'scope':'Compile/link only. Native seam/analyzer/evolution remain held.'}
        dump(target/'receipt.json',receipt)
        print('PASS',mode,'compile/link only',receipt['executable_sha256'],flush=True)
    except Exception as error:
        dump(target/'failure.json',{'passed_compile_link':False,'mode':mode,
          'error':str(error),'source_before':before,'commands':commands,
          'recipe_sha256':sha(p),'authorization_sha256':sha(auth_path),
          'executor_sha256':sha(Path(__file__)),'seconds':time.monotonic()-started,
          'native_executed':False})
        raise

if __name__=='__main__': main()
