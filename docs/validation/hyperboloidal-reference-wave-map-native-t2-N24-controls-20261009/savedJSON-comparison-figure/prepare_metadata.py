"""Standard-library-only source/input/runtime pinning before figure execution."""
from pathlib import Path
import ast,hashlib,json,sysconfig
P=Path(__file__).resolve().parent;R=P.parents[2]
def pin(p):
 p=Path(p).resolve();h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return {'path':str(p),'sha256':h.hexdigest(),'bytes':p.stat().st_size}
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
C=R/'build-layer-research/wave-map-native-N24-partial-comparison-root-20261009';report=C/'attempt001/report.json';assert pin(report)['sha256']=='0610afdad653199a50faad87455b241c9450651ba6c760950a0b0e442973fd76'
q=json.loads((C/'recipe.json').read_text());inputs=[pin(report),pin(C/'recipe.json')]
for p,s in q['input_pins'].items():item=pin(p);assert item['sha256']==s;inputs.append(item)
roots=[Path(sysconfig.get_paths()['stdlib']),R/'build-layer-research/plot-deps'];runtime={}
for root in roots:
 for p in sorted(root.rglob('*')):
  if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc','.pyo'] and (root.name=='plot-deps' or 'site-packages' not in p.relative_to(root).parts):runtime[str(p.resolve())]=pin(p)
py=Path('/Library/Developer/CommandLineTools/usr/bin/python3');runtime[str(py)]=pin(py)
dump(P/'runtime-inputs.json',{'scope':'Unimported plotting runtime inventory; all file bytes pinned before source execution.','files':list(runtime.values())})
recipe={'scope':'Scientific figure from completed owner/manual savedJSON/rootreport plus failurestderr only; no arrays/native/probe calls.','comparison_report':str(report),'comparison_recipe':str(C/'recipe.json'),'figure':'2x2 panels: large std/half native RMS; small native RMS; root matched relative differences; native squaredZshell fraction','figure_size_inches':[11.5,8.2],'log_RMS_limits':[8e-6,.6],'relative_percent_limits':[1e-6,.2],'time_axis_limits':[0,1.06],'time0':'near-roundoff t0 omitted from log panels without altering source data','shell_formula':'N_shell/N_active*(RMS_Z_shell/RMS_Z_all)^2, r>=.9','limits':'All original processes failed beforetarget2; no unsaved abort state is reconstructed; no stability/causal/energy inference.','source':pin(P/'plot_saved.py'),'python':str(py),'environment':{'PYTHONDONTWRITEBYTECODE':'1','OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','PYTHONPATH':str(R/'build-layer-research/plot-deps'),'MPLCONFIGDIR':str(P/'matplotlib-runtime-cache')}}
dump(P/'recipe.json',recipe)
files=[pin(P/name) for name in ['plot_saved.py','prepare_metadata.py','recipe.json','runtime-inputs.json']]
for p in [P/'plot_saved.py',Path(__file__).resolve()]:ast.parse(p.read_text(),filename=str(p))
dump(P/'source-index.json',{'source_only_before_plot':True,'files':files,'inputs':inputs,'runtime_files':len(runtime),'no_native_arrays_or_scientific_imports_in_preparation':True})
print(json.dumps({'source_index':pin(P/'source-index.json'),'source':pin(P/'plot_saved.py'),'recipe':pin(P/'recipe.json'),'inputs':len(inputs),'runtime_files':len(runtime)},indent=2))
