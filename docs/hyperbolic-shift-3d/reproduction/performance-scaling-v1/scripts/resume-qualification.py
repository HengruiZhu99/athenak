import pathlib,subprocess,time,json,shlex,shutil,hashlib
r=pathlib.Path('build-performance').resolve();out=r/'qualification';assert out.is_dir();records=json.loads((out/'commands.json').read_text())
def run(name,cmd,timeout=900,cwd=None,env=None):
 start=time.monotonic();rec=dict(name=name,command=list(map(str,cmd)),cwd=str(cwd or pathlib.Path.cwd()),status='started');records.append(rec);(out/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
 with (out/(name+'.log')).open('w') as log:
  p=subprocess.run(rec['command'],cwd=cwd,stdout=log,stderr=subprocess.STDOUT,timeout=timeout,env=env)
 rec.update(returncode=p.returncode,seconds=time.monotonic()-start);(out/'commands.json').write_text(json.dumps(records,indent=2)+'\n');print(name,p.returncode,flush=True);assert p.returncode==0,name
run('configure-components',['cmake','-S',str(pathlib.Path.cwd()),'-B',str(r/'candidate')])
py=r/'analysis-env/bin/python';test=pathlib.Path('tst/hyperbolic_shift_3d')
targets=['hyperbolic_shift_3d_amr_setup','hyperbolic_shift_3d_amr_transfer','hyperbolic_shift_3d_amr_operators','hyperbolic_shift_3d_sparse_exchange','hyperbolic_shift_3d_radial_field_batch','hyperbolic_shift_3d_mesh_plan_bundle','hyperbolic_shift_3d_repartition_transaction','hyperbolic_shift_3d_mpi_continuous_positive_fast','hyperbolic_shift_3d_mpi_continuous_positive_sparse']
run('build-components-reconfigured',['cmake','--build',r/'candidate','-j4','--target',*targets],1800)
run('ctest',['ctest','--test-dir',r/'candidate','--output-on-failure','-R','^(hyperbolic_shift_3d_amr_transfer_mpi[124]|hyperbolic_shift_3d_amr_operators|hyperbolic_shift_3d_radial_field_batch|hyperbolic_shift_3d_mesh_plan_bundle_mpi)$'],1200)
bin=r/'candidate/tst/hyperbolic_shift_3d'
for P in [1,2,4]:run('sparse-'+str(P),['mpiexec','-n',str(P),bin/'hyperbolic_shift_3d_sparse_exchange'],420)
run('transaction',['mpiexec','-n','2',bin/'hyperbolic_shift_3d_repartition_transaction',out/'transaction'],420)
fixture=pathlib.Path('/Users/hz0693/.codex/worktrees/hyperbolic-core-shell/hyperboloidal/output/hyperbolic-core-shell/continuous-positive-local-v1/puncture-n16.txt');shutil.copy2(fixture,out/'puncture-n16.txt');(out/'fixture.sha256').write_text(hashlib.sha256(fixture.read_bytes()).hexdigest()+'\n')
run('sparse-integrated',[py,test/'test_sparse_stage_mpi.py','--binary',bin/'hyperbolic_shift_3d_mpi_continuous_positive_sparse','--control',bin/'hyperbolic_shift_3d_mpi_continuous_positive_fast','--mpiexec',shutil.which('mpiexec'),'--initial',out/'puncture-n16.txt','--output-dir',out/'sparse-integrated'],480)
# The native static archive is the same complete source list, minus main.cpp.
entry=next(x for x in json.loads((r/'control/compile_commands.json').read_text()) if x['file'].endswith('/main.cpp'))
cmd=shlex.split(entry['command']);cmd[cmd.index('-o')+1]=str(out/'main.o');run('main-compile',cmd,cwd=entry['directory'])
link=shlex.split((r/'control/tst/hyperbolic_shift_3d/CMakeFiles/hyperbolic_shift_3d_amr_evolution.dir/link.txt').read_text());link=[str(out/'main.o') if x.endswith('/amr_evolution.cpp.o') else x for x in link];link[link.index('-o')+1]=str(out/'athena');run('main-link',link,cwd=r/'control/tst/hyperbolic_shift_3d')
run('main-z4c',[py,test/'validate_main_amr.py','--binary',out/'athena','--output',out/'main-z4c'],1200)
(out/'completed.json').write_text(json.dumps({'commands':len(records)})+'\n')
