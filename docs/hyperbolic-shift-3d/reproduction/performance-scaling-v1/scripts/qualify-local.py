import pathlib,subprocess,time,json,shlex,shutil,hashlib
r=pathlib.Path('build-performance').resolve();out=r/'qualification';out.mkdir(exist_ok=False);records=[]
def run(name,cmd,timeout=900,cwd=None,env=None):
 start=time.monotonic();rec=dict(name=name,command=list(map(str,cmd)),cwd=str(cwd or pathlib.Path.cwd()),status='started');records.append(rec);(out/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
 with (out/(name+'.log')).open('w') as log:
  p=subprocess.run(rec['command'],cwd=cwd,stdout=log,stderr=subprocess.STDOUT,timeout=timeout,env=env)
 rec.update(returncode=p.returncode,seconds=time.monotonic()-start);(out/'commands.json').write_text(json.dumps(records,indent=2)+'\n');print(name,p.returncode,flush=True);assert p.returncode==0,name
start=time.monotonic()
while not (r/'cpu-m4-bench/completed.json').exists():
 if time.monotonic()-start>600:raise RuntimeError('timing not complete')
 time.sleep(2)
py=r/'analysis-env/bin/python';b=r/'m4-bin/hyperbolic_shift_3d_amr_setup';test=pathlib.Path('tst/hyperbolic_shift_3d')
for suite in ['matrix','controls','restart','equilibrium']:
 run('m4-'+suite,[py,test/'validate_amr.py','--binary',b,'--output',out/('m4-'+suite),'--suite',suite],1800)
run('checkpoint-reject',[py,test/'test_amr_checkpoint.py','--binary',b,'--checkpoint',out/'m4-restart/wave-save.checkpoint','--output',out/'checkpoint-reject'],420)
run('donor-audit',[py,test/'audit_amr_rejection.py','--prefix',out/'m4-controls/boost-rejected','--output',out/'donor-audit.json'])
for name,a,c in [('wave',r/'baseline-matrix/wave-1.checkpoint',out/'m4-matrix/wave-1.checkpoint'),('restart',out/'m4-restart/wave-full.checkpoint',out/'m4-restart/wave-restart.checkpoint'),('boost-restart',out/'m4-controls/boost-fixed.checkpoint',out/'m4-controls/boost-restart.checkpoint')]:
 run('compare-'+name,[py,test/'analyze_amr.py',a,c,'--output',out/('compare-'+name+'.json')])
targets=['hyperbolic_shift_3d_amr_setup','hyperbolic_shift_3d_amr_transfer','hyperbolic_shift_3d_amr_operators','hyperbolic_shift_3d_sparse_exchange','hyperbolic_shift_3d_radial_field_batch','hyperbolic_shift_3d_mesh_plan_bundle','hyperbolic_shift_3d_repartition_transaction','hyperbolic_shift_3d_mpi_continuous_positive_fast','hyperbolic_shift_3d_mpi_continuous_positive_sparse']
run('build-components',['cmake','--build',r/'candidate','-j4','--target',*targets],1800)
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
