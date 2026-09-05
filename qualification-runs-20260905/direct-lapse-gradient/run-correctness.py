"""Run frozen zero-step production projection oracles; never launch stress."""
import json
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parent
source=root/'source'
sys.path.insert(0,str(source/'analysis/pc_gh_regular_extension'))
from make_inputs import parse,write
from verify_hybrid_projection import check,identities
identities()
for path in sorted((root/'inputs/oracle').glob('*-o6-*.athinput')):
    target=path.with_name(path.name.replace('-o6-','-o4-'))
    p=parse(path.read_text());p['pc_gh']['spatial_order']='4'
    if target.exists():
        assert parse(target.read_text())==p, 'Existing FD4 input differs'
    else:write(target,p)
results=[]
for path in sorted((root/'inputs/oracle').glob('*.athinput')):
    out=root/'oracle'/path.stem
    cmd=[sys.executable,str(source/'analysis/pc_gh_regular_extension/cuda_driver.py'),'run',
         '--build',str(source/'build-direct-cuda'),'--input',str(path),'--output',str(out)]
    subprocess.run(cmd,check=True)
    result=check(out,direct_lapse=True)
    (out/'independent-verification.json').write_text(json.dumps(result,indent=2)+'\n')
    results.append(result)
    (root/'projection-results.json').write_text(json.dumps(results,indent=2)+'\n')
print('PASS',len(results),'production projection cases',flush=True)
