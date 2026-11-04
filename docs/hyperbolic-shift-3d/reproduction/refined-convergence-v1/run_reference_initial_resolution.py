"""Reproduce N12/N14 initial-only controls; refuses existing outputs."""
import pathlib,subprocess,hashlib,json,time
root=next(p for p in pathlib.Path(__file__).resolve().parents if (p/'CMakeLists.txt').exists());out=root/'output/refined-convergence/reference-initial-resolution';out.mkdir(exist_ok=True)
binary=root/'build-refined-convergence/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_refined_amr'
config=root/'inputs/hyperbolic_shift_3d/refined_convergence/central20_stationary_smr025_radial12_reference.cfg'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for n in (12,14):
 prefix=out/f'n{n}';record=out/f'n{n}-command.json';assert not record.exists() and not pathlib.Path(str(prefix)+'.checkpoint').exists()
 command=['mpiexec','-n','1',str(binary),str(n),'boost','0',format(.000125*16/n,'.17g'),str(prefix),'-','smr',str(config)]
 result=dict(command=command,binary_sha256=sha(binary),config_sha256=sha(config),source_commit='b52f09723b9c5a51340401bf91a7bf4426d8f339',started=time.time(),purpose='Initial-data-only N12/N14 resolution control. No evolution steps; N16 t0 row reused from existing admitted two-step control.')
 record.write_text(json.dumps(result,indent=2)+'\n')
 with (out/f'n{n}-stdout.log').open('w') as log:
  p=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=600)
 result.update(returncode=p.returncode,finished=time.time());record.write_text(json.dumps(result,indent=2)+'\n')
 if p.returncode:raise RuntimeError(f'N{n} failed: {p.returncode}')
 print(json.dumps(dict(n=n,seconds=result['finished']-result['started'])),flush=True)
