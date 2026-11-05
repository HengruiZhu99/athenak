import json
from pathlib import Path
e=Path('docs/spherical-optimization/evidence');cases=[]
for n in [128,256,512]:
 old=e/f'outer-lapse025-n{n}-10M-v26';new=e/f'outer-lapse025-penalty05-n{n}-10M-v{27 if n==128 else 28}'
 dt=json.loads((old/'manifest.json').read_text())['config']['fixed_dt']
 for family,ref,overrides in [('conservative',old,dict(fixed_dt=None,experimental_timestep=False,cfl=.2)),('step055',old,dict(fixed_dt=dt*.55/.4,experimental_timestep=True)),('penalty05-step068',new,dict(fixed_dt=dt*.68/.4,experimental_timestep=True))]:
  cases.append(dict(name=f'{family}-n{n}',family=family,reference_run=str(ref),overrides=overrides))
spec=dict(protocol='docs/spherical-optimization/COST-PROTOCOL.md',qualification='Penalty .5/C.68 controls completed before timing; see PENALTY-TRADEOFF.md and penalty05-*-v28 evidence.',cases=cases)
(e/'cost-specification-v28.json').write_text(json.dumps(spec,indent=2)+'\n')
