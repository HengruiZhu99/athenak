from pathlib import Path
import hashlib, json, os, subprocess, sys

ROOT=Path(__file__).resolve().parent
REVIEW=ROOT.parents[1]
NEW=ROOT/'athena-covariant-sources'
OLD=ROOT.parent/'mode-analysis/athena-lapse-scaled'
BASE=REVIEW/'stability-isolation-20260919/gauge-slow/shift_Gamma2/input.athinput'
RST=BASE.parent/'rst/rank_00000000/ks_background.00002.rst'
# The build tree now contains the separately saved mode hook. Never overwrite
# the immutable source-only executable while reproducing its validation.
assert hashlib.sha256(NEW.read_bytes()).hexdigest()==json.loads(
    (ROOT/'manifest.json').read_text())['binaries']['athena-covariant-sources']
base=BASE.read_text().replace('<z4c>','<z4c>\ndamp_lapse_scaled = false\nccz4_covariant_sources = false')
base=base.replace('<problem>','<problem>\nmode_analysis = false')
(ROOT/'validation.athinput').write_text(base)
ENV=dict(os.environ,OMP_NUM_THREADS='2',OMP_PROC_BIND='false',OPENBLAS_NUM_THREADS='1')
COMMON=['time/cfl_number=.15','time/nlim=3','time/tlim=1000','time/ndiag=1',
        'problem/mode_analysis=false','z4c/damp_lapse_scaled=true','z4c/damp_kappa1=.3',
        'z4c/debug_balance=true','z4c/debug_reduction_stride=1',
        'z4c/debug_snapshot_operations=rhs_full_vs_bg,volume_rhs,post_recast',
        'z4c/rhs_term_debug=true','z4c/rhs_term_debug_stride=1']
cases=[
 ('zero',NEW,False,['z4c/ccz4_covariant_sources=true','problem/vacuum_gauge_pulse_amplitude=0']),
 ('off_new',NEW,False,['z4c/ccz4_covariant_sources=false']),
 ('off_old',OLD,False,[]),
 ('matter_off_low',NEW,False,['time/nlim=1','z4c/ccz4_covariant_sources=false',
     'problem/vacuum_gauge_pulse_amplitude=0','problem/zero_tmunu=false',
     'mhd/zero_tmunu_feedback=false','mhd/dfloor=1e-14','mhd/pfloor=1e-17',
     'problem/rho_cut=1e-14','problem/excision_atmo_density=1e-14',
     'problem/excision_atmo_energy=1e-17']),
 ('matter_on_low',NEW,False,['time/nlim=1','z4c/ccz4_covariant_sources=true',
     'problem/vacuum_gauge_pulse_amplitude=0','problem/zero_tmunu=false',
     'mhd/zero_tmunu_feedback=false','mhd/dfloor=1e-14','mhd/pfloor=1e-17',
     'problem/rho_cut=1e-14','problem/excision_atmo_density=1e-14',
     'problem/excision_atmo_energy=1e-17']),
 ('late_off',NEW,True,['time/nlim=6668','z4c/ccz4_covariant_sources=false']),
 ('late_on',NEW,True,['time/nlim=6668','z4c/ccz4_covariant_sources=true'])]
if len(sys.argv)>1:cases=[c for c in cases if c[0] in sys.argv[1:]]
report=json.loads((ROOT/'validation-runs.json').read_text()) if (ROOT/'validation-runs.json').exists() else []
for name,binary,restart,extra in cases:
    folder=ROOT/'validation'/name
    folder.mkdir(parents=True,exist_ok=False)
    cmd=[str(binary),'-i',str(ROOT/'validation.athinput')]
    if restart:cmd+=['-r',str(RST)]
    cmd+=COMMON+extra
    with (folder/'run.log').open('w') as f:
        p=subprocess.run(cmd,cwd=folder,env=ENV,stdout=f,stderr=subprocess.STDOUT,timeout=120)
    log=(folder/'run.log').read_text()
    row=dict(name=name,command=cmd,returncode=p.returncode,invalid='Z4C_INVALID_STATE' in log,
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    report.append(row)
    (ROOT/'validation-runs.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(row),flush=True)
    if p.returncode or row['invalid']:raise RuntimeError(name)
