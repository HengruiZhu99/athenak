from pathlib import Path
import json,numpy as np,re
p=Path(__file__).parent;a=json.loads((p/'first-stage-helper.json').read_text())
keys=['FTheta','deltaTheta','deltaKhat','c1_correction_error','c2_correction_error'];o={k:max(a,key=lambda z:abs(z[k])) for k in keys};o['FQ']=max(a,key=lambda z:np.linalg.norm(z['FQ']));print(json.dumps(o,indent=2));(p/'first-stage-maxima.json').write_text(json.dumps(o,indent=2)+'\n')
root=p.parent/'stages-v1/pulse_r1_t1';out=[]
for c in range(3):
 for s in range(1,4):
  stem=f'z4c_snapshot_pre_boundary_rhs_rank0_cycle{c}_stage{s}';m=json.loads((root/(stem+'.json')).read_text());pre=np.fromfile(root/(stem+'.bin')).reshape(m['shape']);post=np.fromfile(root/(stem.replace('pre_boundary_rhs','post_boundary_rhs')+'.bin')).reshape(m['shape']);d=post-pre
  ent={'cycle':c,'stage':s,'cycle_start_time':m['time'],'fields':{}}
  for name,field in [('Theta',17),('Khat',7),('Gamma_x',14),('gxx',1),('alpha',18)]:
   rec={}
   for typ,v in [('pre',pre),('post',post),('correction',d)]:
    aa=v[:,field,4:12,4:12,4:12];idx=np.unravel_index(np.argmax(abs(aa)),aa.shape);bl=m['blocks'][idx[0]];xyz=[bl['xmin'][j]+(idx[3-j]+.5)*bl['dx'][j] for j in range(3)]
    rec[typ]={'max_abs':float(abs(aa[idx])),'gid':bl['gid'],'level':bl['level'],'xyz':xyz,'r':float(np.linalg.norm(xyz)),'physical_face_count':int(sum(abs(x)==1.875 for x in xyz))}
   ent['fields'][name]=rec
  out.append(ent)
(p/'early-stage-localization.json').write_text(json.dumps(out,indent=2)+'\n');print('earlyTheta',out[0]['fields']['Theta'])
