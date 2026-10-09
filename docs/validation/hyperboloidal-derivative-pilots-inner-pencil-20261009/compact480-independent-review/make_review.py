"""Standard-library hash/AST/source-only review capture; no science import."""
from pathlib import Path
import ast
import hashlib
import json
import shutil

P=Path(__file__).resolve().parent
B=P.parent
C=B/'native-angular-pulse-flat-derivatives-compact-root-held-20261009'
O=B/'native-angular-pulse-flat-derivatives-compact-root-launch-held-20261009'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def load(p):return json.loads(Path(p).read_text())


def write(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')


expected={str(C/'source-index.json'):'4f5a4fbbfa6c6e4af867963189f4d3a5549116e047849bf9fac8d3c96b6c9607',
          str(O/'source-index.json'):'0e4fcf32687ab11271b6e1a6bf90a5ba3ce9c6877b131a4cddf5564fcedafc1f'}
for path,digest in expected.items():
    if sha(path)!=digest:raise RuntimeError('index changed')
rows=[]
for folder in (C,O):
    for row in load(folder/'source-index.json')['files']:
        if sha(row['path'])!=row['sha256']:raise RuntimeError('source changed')
        rows.append(row)
        if row['path'].endswith('.py'):ast.parse(Path(row['path']).read_text())
recipe=load(C/'comparison-recipe.json')
dependencies={**recipe['dependency_pins'],**recipe['mpmath_python_pins'],
              recipe['python_runtime_path']:recipe['python_runtime_sha256']}
for path,digest in dependencies.items():
    if sha(path)!=digest:raise RuntimeError('dependency drift')
for name,digest in [('derivative_core.py','4ba5538c9443210cf00fc2a6cf37ca19a6d800f7f347ad9ef7b875c1c0b733ec'),
                    ('analytic_jets.py','b2defe4517d4b052851abdff21e6912fa5c4f3814d4cc3efa819c6edfc8fe410'),
                    ('values_context.py','89c96ee729b8cc741757740534566c9557dd6fdce2e66213c2fe8ff1c7e8ffe7')]:
    if sha(C/name)!=digest:raise RuntimeError('inherited math source differs')
target=P/'source-before';target.mkdir(exist_ok=False)
copies=[]
for i,row in enumerate(rows):
    dest=target/('%02d-'%i+Path(row['path']).name)
    shutil.copyfile(row['path'],dest)
    if sha(dest)!=row['sha256']:raise RuntimeError('source copy differs')
    copies.append(dict(row,copy=str(dest)))
write(P/'receipt.json',{'disposition':'PASS_SOURCE_ONLY_NO_EXECUTION',
      'candidate_index':expected[str(C/'source-index.json')],
      'outer_index':expected[str(O/'source-index.json')],
      'source_files':copies,'dependency_pins_rehashed':dependencies,
      'candidate_files':9,'outer_files':6,'protected_paths_with_authorization':138,
      'copied_paths_in_future_outer':137,'ray_rows':480,'group_rows':24,'checks':38344,
      'no_import_query_array_load_or_scientific_execution':True,
      'scope':'Pencil/source/admission review only; local arithmetic pilot, full gate held',
      'blocking_corrections':[],
      'clarifications':['Analytic branch width0 is formula bookkeeping, not interval enclosure',
                        'Root signs use fixed quadrature and are not interval arithmetic',
                        'No full runtime extrapolation or inverse/Jacobian claim'],
      'mechanical_history':{'tool_parser_error':'SyntaxError: Unexpected token )',
                            'stage':'tool-JavaScript parse before metadata shell command dispatch',
                            'candidate_execution':False},
      'review_sha256':sha(P/'REVIEW.md')})
files=[]
for path in sorted(P.rglob('*')):
    if path.is_file():files.append({'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size})
write(P/'index.json',{'source_only_review':True,'files':files,'frozen':True})
print(json.dumps({'index_sha256':sha(P/'index.json'),'receipt_sha256':sha(P/'receipt.json'),
                  'review_sha256':sha(P/'REVIEW.md'),'files':len(files),
                  'dependency_count':len(dependencies),'no_scientific_execution':True},indent=2))
