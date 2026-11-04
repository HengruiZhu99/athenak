import pathlib,subprocess,json,hashlib,os,time
root=pathlib.Path.cwd();work=root/'build-default-cleanup';prior=root/'build-performance';binary=prior/'candidate/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_amr_evolution'
runs=[];references={}
def add(name,ranks,kind,policy,reference,steps=6,**extras):
 runs.append(dict(name=name,variant='default',ranks=ranks,kind=kind,policy=policy,steps=steps,timeout=300,**extras));references[name]=str(prior/reference)
add('wave-1',1,'wave','dynamic','qualification/m4-matrix/wave-1')
add('wave-4-poison',4,'wave','dynamic','final-check/final-wave',env={'HYPERBOLIC_SPARSE_POISON':'1'})
add('minkowski',2,'minkowski','dynamic','qualification/m4-matrix/minkowski')
add('wave-uniform',1,'wave','uniform','qualification/m4-matrix/wave-uniform')
add('boost-fixed',1,'boost','fixed','final-check/fixed-boost-final')
add('boost-uniform',2,'boost','uniform','qualification/m4-controls/boost-uniform')
add('boost-rejected',2,'boost','dynamic','final-check/rejected-m4',expected=2)
add('wave-save',2,'wave','dynamic','qualification/m4-restart/wave-save',steps=3)
add('wave-restart',4,'wave','dynamic','qualification/m4-restart/wave-restart',restart=str(work/'amr/wave-save.checkpoint'))
plan={'variants':{'default':str(binary)},'runs':runs};(work/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');(work/'references.json').write_text(json.dumps(references,indent=2)+'\n')
records=[]
def run(name,cmd,timeout):
 record={'name':name,'command':list(map(str,cmd)),'status':'started'};records.append(record);save();start=time.monotonic()
 with (work/'evidence'/(name+'.log')).open('w') as f:
  p=subprocess.run(record['command'],stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
 record.update(status='complete',returncode=p.returncode,seconds=time.monotonic()-start);save();print(name,p.returncode,flush=True);assert p.returncode==0,name
def save():(work/'evidence/commands.json').write_text(json.dumps(records,indent=2)+'\n')
run('amr',['python3','tst/hyperbolic_shift_3d/performance/run_plan.py',work/'plan.json',work/'amr'],1800)
checks=[]
for case in runs:
 name=case['name'];suffixes=['history.jsonl','failed-stage.bin','rejected-transfer.bin','failure.json'] if case.get('expected')==2 else ['history.jsonl','checkpoint']
 for suffix in suffixes:
  a=work/'amr'/(name+'.'+suffix);b=pathlib.Path(references[name]+'.'+suffix);ha=hashlib.sha256(a.read_bytes()).hexdigest();hb=hashlib.sha256(b.read_bytes()).hexdigest();checks.append(dict(case=name,artifact=suffix,reference=str(b),reference_sha256=hb,default_sha256=ha,bitwise_equal=ha==hb))
(work/'evidence/bitwise.json').write_text(json.dumps(checks,indent=2)+'\n');assert all(x['bitwise_equal'] for x in checks)
run('checkpoint-rejection',['python3','tst/hyperbolic_shift_3d/test_amr_checkpoint.py','--binary',binary,'--checkpoint',work/'amr/wave-save.checkpoint','--output',work/'checkpoint-rejection'],420)
run('contracts',['ctest','--test-dir',prior/'candidate','--output-on-failure','-R','^(hyperbolic_shift_3d_amr_setup_contract|hyperbolic_shift_3d_mesh_plan_bundle_mpi)$'],900)
bindir=prior/'candidate/tst/hyperbolic_shift_3d'
run('transaction',['mpiexec','-n','2',bindir/'hyperbolic_shift_3d_repartition_transaction',work/'transaction'],420)
run('sparse-integration',['python3','tst/hyperbolic_shift_3d/test_sparse_stage_mpi.py','--binary',bindir/'hyperbolic_shift_3d_mpi_continuous_positive_sparse','--control',work/'frozen/legacy-fast','--mpiexec','/opt/homebrew/bin/mpiexec','--initial',prior/'qualification/puncture-n16.txt','--output-dir',work/'sparse'],480)
(work/'evidence/completed.json').write_text(json.dumps({'commands':len(records),'bitwise_checks':len(checks),'passed':True})+'\n')
