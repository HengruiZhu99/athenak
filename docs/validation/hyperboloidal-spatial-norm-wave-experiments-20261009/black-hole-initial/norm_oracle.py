"""100-digit live spatial-norm shift-pole gate, fixed BH ADM data."""
import json
from pathlib import Path
from frozen_outer_oracle import *

def norm_evaluate(O,eta=mp.mpf(6),da1=mp.mpf(0),db1=mp.mpf(0),da2=mp.mpf(0),db2=mp.mpf(0)):
    v=evaluate(O,'baseline',da1,db1,da2,db2,xi_rate=1/a)
    r=mp.sqrt(S*S-2*a*S*O);op=-r/(a*S);L=S/a-O;m=M*O/(2*r);psi=1+m
    alpha=(1-m)/psi*L+da1*O+da2*O*O
    beta=-r/a*(1-m)/psi**3+db1*O+db2*O*O
    wn=-beta*op/alpha
    dbdiv=M/(2*a)*(4+3*m+m*m)/psi**3+db1+db2*O
    chdiv=-M/(2*r)*(4+6*m+4*m*m+m**3)/psi**4
    C=S/a*(1-S/(eta*a*a))
    added=-eta*(dbdiv+C*chdiv)
    v['beta_dot']+=added
    v['Ndot']+=2*wn*op/alpha*added
    v['gauge']+=2*wn*op/alpha*added
    v['shift_manifold_dot']=v['beta_dot']+C*v['geometry']/(op*op)
    v['C']=C
    return v

if __name__=='__main__':
    rows=[];summary=[]
    for eta in [mp.mpf(5),mp.mpf(6),mp.mpf(10)]:
        lead=norm_evaluate(mp.mpf(0),eta)
        for k in ['alpha_dot','beta_dot','Ndot','shift_manifold_dot','geometry']:
            assert abs(lead[k])<mp.mpf('1e-80'),(eta,k,lead[k])
        ncoef=mp.diff(lambda x:norm_evaluate(x,eta)['Ndot'],mp.mpf(0))
        ac=mp.diff(lambda x:norm_evaluate(x,eta,da2=mp.mpf(1))['Ndot'],mp.mpf(0))-ncoef
        bc=mp.diff(lambda x:norm_evaluate(x,eta,db2=mp.mpf(1))['Ndot'],mp.mpf(0))-ncoef
        db2=-ncoef/bc
        corrected=mp.diff(lambda x:norm_evaluate(x,eta,db2=db2)['Ndot'],mp.mpf(0))
        assert abs(corrected)<mp.mpf('1e-75')
        q2=mp.diff(lambda x:norm_evaluate(x,eta,db2=db2)['Ndot'],mp.mpf(0),2)/2
        summary.append({'eta':str(eta),'C':mp.nstr(lead['C'],25),'Ndot_Omega':mp.nstr(ncoef,25),'delta_a2_coefficient':mp.nstr(ac,25),'delta_b2_coefficient':mp.nstr(bc,25),'delta_b2_cancel':mp.nstr(db2,25),'corrected_Ndot_Omega':mp.nstr(corrected,25),'corrected_Ndot_Omega2':mp.nstr(q2,25)})
        for free in [mp.mpf(0),db2]:
            for O in [mp.mpf('.0975'),mp.mpf('1e-4'),mp.mpf('1e-8'),mp.mpf('1e-30'),mp.mpf(0)]:
                val=norm_evaluate(O,eta,db2=free)
                rows.append({'eta':str(eta),'delta_b2':mp.nstr(free,25),'Omega':str(O),**{k:mp.nstr(x,25) for k,x in val.items()}})
    Path(__file__).with_name('norm_oracle.json').write_text(json.dumps({'dps':mp.mp.dps,'summary':summary,'rows':rows},indent=2)+'\n')
    print('PASS live spatial-norm feedback zero-jet and next null-rate gates')
    print(json.dumps(summary,indent=2))
