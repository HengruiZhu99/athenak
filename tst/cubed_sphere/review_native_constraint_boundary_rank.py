#!/usr/bin/env python3
"""Rank only: nonlinear constraint value conditions do not complete an IBVP."""
import argparse,ast,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--symbols',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();helper=Path(__file__).with_name('check_native_second_order_sbp_sat.py');source=helper.read_text();nodes=ast.parse(source).body
for node in nodes:
 if isinstance(node,ast.FunctionDef) and node.name in ['symbols','mm']:exec(compile(ast.Module(body=[node],type_ignores=[]),str(helper),'exec'),globals())
records=[]
for path in a.symbols:
 for s in symbols(path):
  M=s.pop('matrix');C=s.pop('constraints');S=-M;I=np.eye(20);incoming=(s['hi']*I-S)/(s['hi']-s['lo']);outgoing=I-incoming;J=C[4:];J=J/np.linalg.norm(J,axis=1)[:,None];si=np.linalg.svd(mm(J,incoming),compute_uv=False);so=np.linalg.svd(mm(J,outgoing),compute_uv=False);ri=int(sum(si>1e-9));ro=int(sum(so>1e-9));records.append(dict(**s,normalized_Theta_Z_incoming_singular_values=si.tolist(),normalized_Theta_Z_outgoing_singular_values=so.tolist(),incoming_constraint_value_rank=ri,outgoing_constraint_value_rank=ro,remaining_incoming_freedoms=10-ri,passed=ri==4 and ro==4))
r=dict(passed=all(x['passed'] for x in records),records=records,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),symbol_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in a.symbols},scope='Theta/Z algebraic value rows only, full nonlinear compatibility not established. Four reflective constraint value conditions leave six physical/gauge incoming freedoms. Not a constraint absorption or well-posedness proof.');a.output.write_text(json.dumps(r,indent=2)+'\n');print(r['passed'],len(records))
