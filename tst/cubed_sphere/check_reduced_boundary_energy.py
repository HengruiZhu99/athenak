#!/usr/bin/env python3
"""Frozen normal compatible momentum-SAT algebra on analytic native backgrounds.
No native boundary integration or variable-volume-work energy theorem.
"""
import json,math,sys
from pathlib import Path
base=Path(sys.argv[1]);out=base/'boundary-energy';out.mkdir(exist_ok=True)
criteria={'status':'Predeclared before execution','backgrounds':300,'gate_relative_energy_identity':1e-11,'gate_positive_tangent_roundoff':1e-11,'gate_driven_bound_excess':1e-11,'require':'Every sampled outward beta.normal <=0 and c>0; no assumption all modes outgoing.','SAT':'V_t penalty -c(V+w-g), w=N G(n,Q)/sqrt(G(n,n)); U_t=beta.gradU+V unchanged, Q=gradU recomputed/derived, never independently reset.','limits':'Algebraic frozen-normal face energy only. Source/volume-work, coupled time evolution, patch seams and native lift unvalidated; angles are algebraic flux samples, not reflection measurements.'}
(out/'criteria.json').write_text(json.dumps(criteria,indent=2)+'\n')
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def mul(g,q):return [dot(row,q) for row in g]
def inv3(a):
 det=a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])-a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])+a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0])
 return [[(a[(j+1)%3][(i+1)%3]*a[(j+2)%3][(i+2)%3]-a[(j+1)%3][(i+2)%3]*a[(j+2)%3][(i+1)%3])/det for j in range(3)] for i in range(3)]
r={'backgrounds':0,'tests':0,'failures':[],'max_identity':0.,'max_bound_excess':0.,'min_tangent':0.,'beta_normal_range':[math.inf,-math.inf],'incoming_speed_range':[math.inf,-math.inf],'source_only_energy_bound_range':[math.inf,-math.inf]}
for family in ['synthetic','extended','massive']:
 mats=[json.loads(x) for x in (base/(family+'-matrices.jsonl')).read_text().splitlines()];jets=[json.loads(x) for x in (base/(family+'-jets.jsonl')).read_text().splitlines()];assert len(mats)==len(jets)
 for mat,jet in zip(mats,jets):
  r['backgrounds']+=1;N=mat['N'];G=[mat['G'][i*3:(i+1)*3] for i in range(3)];x=jet['coordinate'];norm=math.sqrt(dot(x,x));n=[v/norm for v in x];bn=dot(mat['beta'],n);gn=dot(n,mul(G,n));c=N*math.sqrt(gn)
  assert bn<=0 and c>0
  for key,val in [('beta_normal_range',bn),('incoming_speed_range',-bn-c)]:r[key]=[min(r[key][0],val),max(r[key][1],val)]
  # Conservative instantaneous source-only bound. Excludes coefficient/shift volume work.
  gi=inv3(G);emin=min(1.,N*N/sum(gi[i][i] for i in range(3)));bound=math.sqrt(sum(b*b for b in mat['source_matrix_4x20']))/emin
  r['source_only_energy_bound_range']=[min(r['source_only_energy_bound_range'][0],bound),max(r['source_only_energy_bound_range'][1],bound)]
  tangent=[-n[1],n[0],0.] if abs(n[2])<.9 else [1.,0.,0.];tn=math.sqrt(dot(tangent,tangent));tangent=[v/tn for v in tangent]
  for angle in [0,30,45,60,75]:
   a=angle*math.pi/180;q=[math.cos(a)*n[i]+math.sin(a)*tangent[i] for i in range(3)];w=N*dot(n,mul(G,q))/math.sqrt(gn);qenergy=N*N*dot(q,mul(G,q));tanenergy=qenergy-w*w;r['min_tangent']=min(r['min_tangent'],tanenergy)
   if tanenergy < -1e-11*max(1.,qenergy):r['failures'].append([family,jet['radius'],'tangent'])
   for V,g in [(0.,0.),(.3,0.),(-.2,.01),(.1,-.02)]:
    flux=bn*(V*V+qenergy)/2+c*V*w;actual=flux-c*V*(V+w-g);expected=(bn/2-c)*V*V+bn*(w*w+tanenergy)/2+c*V*g
    scale=max(1.,abs(actual),abs(expected));err=abs(actual-expected)/scale;r['max_identity']=max(r['max_identity'],err)
    A=c-bn/2;upper=-A*V*V/2+bn*(w*w+tanenergy)/2+c*c*g*g/(2*A);excess=max(0.,actual-upper)/max(1.,abs(actual),abs(upper));r['max_bound_excess']=max(r['max_bound_excess'],excess);r['tests']+=1
    if err>1e-11 or excess>1e-11:r['failures'].append([family,jet['radius'],'energy',err,excess])
assert r['backgrounds']==300
r['passed']=not r['failures'];r['source_bound_note']='Very conservative frozen instantaneous source-only estimate; excludes time/space coefficient volume work, not a full Gronwall/stability bound.'
(out/'review.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
