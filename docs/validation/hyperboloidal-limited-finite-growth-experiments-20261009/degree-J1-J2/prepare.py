"""Fresh degree extension of the pinned N8 contraction; no scientific query/run."""
from pathlib import Path
import hashlib,json,shutil,subprocess

P=Path(__file__).resolve().parent
old=P.parent/'total-j-finite-rb-control-20261009'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(old/'assemble_blas.py')=='a3b140d7ef41bd5d83ffd9e03cba373f1c10f9486b07c815fd5d80c061926fd7'
assert not (P/'assemble_degree.py').exists()
source=(old/'assemble_blas.py').read_text()
source=source.replace("for f in ('assemble_blas.py','energy_coefficients.py','canceled_basis_complex.py'):","for f in ('assemble_degree.py','energy_coefficients.py','canceled_basis_complex.py'):")
oldguard="assert args.J in (0,1,2) and args.N==8 and args.rb==.98,'First-stage release only before gates pass'"
newguard="assert args.J in (0,1,2) and args.N in (12,16) and args.rb==.98,'Released degree-extension controls only'"
assert oldguard in source;source=source.replace(oldguard,newguard)
anchor="    payload=''.join(format(r,'.17g')+'\\n' for r in radii)\n"
assert source.count(anchor)==1
reuse="""    # Cached reference queries are independent of polynomial degree.
    if (folder/'reference.txt').exists():
        receipt=json.loads((folder/'reference-receipt.json').read_text())
        assert receipt['exit_code']==0 and receipt['query_sha256']==hashlib.sha256(payload.encode()).hexdigest()
        assert receipt['output_sha256']==sha(folder/'reference.txt') and receipt['executable_sha256']==sha(exe)
        rows=np.fromstring((folder/'reference.txt').read_text(),sep=' ').reshape(-1,17)
        assert len(rows)==len(radii) and np.isfinite(rows).all()
        return rows
"""
source=source.replace(anchor,anchor+reuse)
(P/'assemble_degree.py').write_text(source)
for name in ('energy_coefficients.py','canceled_basis_complex.py','radial-bridge-release'):
    shutil.copy2(old/name,P/name)
(P/'inputs/basis').mkdir(parents=True)
shutil.copyfile(old/'inputs/basis/basis-data.json',P/'inputs/basis/basis-data.json')
rec={'kind':'source-only fresh degree extension; no compile/query/assembly',
     'preparation_HEAD':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip(),
     'original_assembler_sha256':sha(old/'assemble_blas.py'),'new_assembler_sha256':sha(P/'assemble_degree.py'),
     'only_changes':['copied assembler basename','CLI N admission 8 ->12/16, fixed rb=.98','validate and reuse existing degree-independent reference query rows'],
     'point_executable_bytecopy':{'original':str(old/'radial-bridge-release'),'sha256':sha(P/'radial-bridge-release')},
     'copied_inputs':{name:sha(P/name) for name in ('energy_coefficients.py','canceled_basis_complex.py','inputs/basis/basis-data.json')},
     'scope':'same source/equations/trial family/SAT/tolerances; no spectra or propagation',
     'scientific_run_release':'after N8 freeze and parent authorization already granted for J0 N12 then N16'}
(P/'preparation-receipt.json').write_text(json.dumps(rec,indent=2,allow_nan=False)+'\n')
print(json.dumps(rec,indent=2))
