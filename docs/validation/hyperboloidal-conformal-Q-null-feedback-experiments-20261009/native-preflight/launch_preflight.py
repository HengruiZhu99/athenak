"""Fresh Q/null-feedback preflights with complete as-built source verification."""
import importlib.util,json,subprocess,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
PYTHON='/Users/hz0693/Documents/Codex/2026-10-06/referenced-chatgpt-conversation-this-is-an/work/venv/bin/python'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
mode='physical-inner';stage=sys.argv[1];assert stage in {'reference','short'}
spec=importlib.util.spec_from_file_location('trace_auditor',HERE/'audit_native.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
verified=audit.verify_build(mode);audit.load_helper()
folder=HERE/mode;folder.mkdir(exist_ok=True)
launch=folder/(stage+'-launch.json');assert not launch.exists()
if stage=='short':
 r=json.loads((folder/'audit/reference.json').read_text());assert r['status']=='PASS'
 reference_sha=sha(folder/'audit/reference.json')
else:reference_sha=None
build_path=HERE/'native-build/build-receipt.json';build=json.loads(build_path.read_text());exe=ROOT/build['executable']
command=[PYTHON,str(ROOT/'tst/hyperboloidal/run_layer_validation.py'),str(exe),str(folder/stage),
 '--suite','reference-long' if stage=='reference' else 'long','--overrides',str(HERE/'native.athinput'),
 '--duration','.05' if stage=='reference' else '.02','--output-cadence','.025' if stage=='reference' else '.01']
record={'scope':'Finite-positive-lapse Q/null gauge preflight only; no stable pulse, puncture or scri acceptance.',
 'launch_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
 'mode':mode,'stage':stage,'source_verification':verified,'trace_gate_index_sha256':audit.TRACE_INDEX,
 'build_receipt_sha256':sha(build_path),'executable_sha256':sha(exe),
 'launch_script_sha256':sha(Path(__file__)),'auditor_sha256':sha(HERE/'audit_native.py'),
 'override_sha256':sha(HERE/'native.athinput'),'reference_audit_sha256':reference_sha,'command':command}
launch.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
subprocess.run(command,cwd=ROOT,check=True)
