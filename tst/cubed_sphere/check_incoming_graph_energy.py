#!/usr/bin/env python3
import ast,argparse,json,hashlib
from pathlib import Path
import numpy as np
np.seterr(over='raise',invalid='raise',divide='raise',under='ignore')
p=argparse.ArgumentParser();p.add_argument('parent',type=Path);p.add_argument('sbp',type=Path);p.add_argument('output',type=Path);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
s=json.loads(args.sbp.read_text());weights=np.array(s['norm_boundary_weights']);rows=np.array(s['left_boundary_rows']);trace=np.array(s['left_trace'])
for fn,names in [('evolve_live_coefficient_sbp_sat.py',['mm','spatial']),('check_incoming_derivative_energy.py',['operator','dot','mv','quadratic','whiten'])]:
 path=Path(__file__).with_name(fn);tree=ast.parse(path.read_text());exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),str(path),'exec'))

def eigencheck(S,G):
 R=np.linalg.cholesky(G);Z=whiten(S,G);vs,us=np.linalg.eigh(Z);v=np.linalg.solve(R.T,us[:,-1]);rate=float(vs[-1]);direct=quadratic(S,v)/quadratic(G,v);res=mv(S,v)-rate*mv(G,v);scale=np.linalg.norm(mv(S,v))+abs(rate)*np.linalg.norm(mv(G,v));return rate,abs(direct-rate)/max(1.,abs(rate)),float(np.linalg.norm(res)/max(scale,1e-30))
records=[]
for case in json.loads((args.parent/'review.json').read_text())['cases']:
 if not case['mapped']:continue
 model=case['runs']['256'];tau=model['tau'];ell=abs(model['lo'])*tau;t=.4*tau;frac=.02*np.cos(t)/(1+.02*np.sin(t))
 for mapped in [False,True]:
  for n in [32,64,128]:
   x,h,D,L,*_=operator(n,model['k'] if mapped else 0.,model['lo'],t);H=np.diag(h);G1=H+ell*ell*mm(D.T,h[:,None]*D);LT_H_L=mm(L.T,h[:,None]*L);Gg=H+tau*tau*LT_H_L;S=h[:,None]*L+L.T*h[None,:];graphS=mm(Gg,L)+mm(L.T,Gg);rhs=S+tau*tau*mm(L.T,mm(S,L));identity=float(np.linalg.norm(graphS-rhs)/max(np.linalg.norm(graphS),np.linalg.norm(rhs),1e-30));liveS=graphS+2*tau*tau*frac*LT_H_L;rate,rayleigh,res=eigencheck(liveS,Gg);rate1,rayleigh1,res1=eigencheck(mm(G1,L)+mm(L.T,G1),G1);equiv=np.linalg.eigvalsh(whiten(Gg,G1));l2rate=float(np.linalg.eigvalsh(S/np.sqrt(h[:,None]*h[None,:]))[-1]);bound=max(0.,2*frac)+max(0.,l2rate);records.append(dict(name=case['name'],mapped=mapped,n=n,graph_identity_relative_defect=identity,graph_live_energy_rate=rate,graph_rate_upper_bound=bound,graph_bound_pass=bool(rate<=bound+1e-8*max(1.,abs(bound))),graph_Rayleigh_relative_defect=rayleigh,graph_generalized_eigen_residual=res,H1_Rayleigh_relative_defect=rayleigh1,H1_generalized_eigen_residual=res1,H1_energy_rate=rate1,equivalence_min=float(equiv[0]),equivalence_max=float(equiv[-1]),Ggraph_condition=float(np.linalg.cond(Gg)),identity_and_Rayleigh_pass=bool(identity<=1e-10 and max(rayleigh,rayleigh1)<=1e-8)))
r=dict(records=records,all_identity_Rayleigh_pass=all(x['identity_and_Rayleigh_pass'] for x in records),all_graph_bounds_pass=all(x['graph_bound_pass'] for x in records),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(args.output/'graph-review.json').write_text(json.dumps(r,indent=2)+'\n');print(r['all_identity_Rayleigh_pass'],r['all_graph_bounds_pass']);print([(x['name'],x['mapped'],x['n'],x['equivalence_min'],x['equivalence_max'],x['graph_live_energy_rate']) for x in records])
