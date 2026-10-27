"""Run the frozen bounded protocol sequentially; no retries or parameter mutation."""
import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True);p.add_argument('--library',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
protocol=json.loads(a.protocol.read_text());(a.output/'protocol.json').write_text(a.protocol.read_text())
for case in protocol['cases']:
 command=[sys.executable,str(Path(__file__).with_name('run.py')),'--library',a.library,'--output',str(a.output/case['name'])]
 for key,value in case.items():
  if key=='name':continue
  flag='--'+key.replace('_','-')
  if isinstance(value,bool):
   if value:command.append(flag)
  else:command += [flag,str(value)]
 with (a.output/(case['name']+'.log')).open('w') as log:
  result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
 print(case['name'],result.returncode,flush=True)
 if result.returncode:raise SystemExit('Harness failure; inspect log')
