import json
from pathlib import Path
import numpy as np
p=Path(__file__).parent;weight=np.ones(20);weight[6:16]=32
small=np.load(p/'coupled-pde-long-N4096-dt2.npz')['u'];large=np.load(p/'coupled-pde-long-N8192-dt2.npz')['u'];a=small[(len(small)-32)//2:(len(small)+32)//2];b=large[(len(large)-32)//2:(len(large)+32)//2]
u6=np.load(p/'coupled-rk3-final-dt0.6.npz')['u'];u3=np.load(p/'coupled-rk3-final-dt0.3.npz')['u'];i=(len(u3)-32)//2
out={'scope':'Signed-array independent final comparisons, BDF2 domain doubling at exact50000M; RK3 matched50000.4M timesteps.','BDF2_domain_doubling':{'interior_relative_weighted_L2':float(np.linalg.norm((a-b)*weight)/np.linalg.norm(b*weight)),'interior_max_abs':float(abs(a-b).max()),'theta_max_relative':float(abs(abs(a[:,7]).max()-abs(b[:,7]).max())/abs(b[:,7]).max())},'RK3_dt_halving':{'full_relative_weighted_L2':float(np.linalg.norm((u6-u3)*weight)/np.linalg.norm(u3*weight)),'interior_relative_weighted_L2':float(np.linalg.norm((u6[i:i+32]-u3[i:i+32])*weight)/np.linalg.norm(u3[i:i+32]*weight)),'theta_relative_L2':float(np.linalg.norm(u6[:,7]-u3[:,7])/np.linalg.norm(u3[:,7]))}}
rkSmall=np.load(p/'coupled-rk3-N4096-final-dt0.3.npz')['u'];ri=(len(rkSmall)-32)//2;diff=rkSmall[ri:ri+32]-u3[i:i+32]
out['RK3_domain_doubling']={'interior_relative_weighted_L2':float(np.linalg.norm(diff*weight)/np.linalg.norm(u3[i:i+32]*weight)),'interior_max_abs':float(abs(diff).max())}
(p/'long-comparison.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
