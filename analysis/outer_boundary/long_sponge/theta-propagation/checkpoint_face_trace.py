#!/usr/bin/env python3
"""Evaluate scalar incoming state functions at a fixed first-active +x face near its center.

Rank-local read complements, but never replaces, all-rank checkpoint validation.
Supported fixture: evolved residual Minkowski, [-2048,2048]^3, uniform 32^3 or 64^3,
8 blocks of 16^3 or 32^3, one block/rank, rank7 location(1,1,1). Uses the source's
active-only second-order derivative, full conformal normal and projected state.
Weighted RHS rate omits frame/coefficient time derivatives; nonlinear state
functions are therefore not claimed to be exact invariants. This is state
measurement, not a reconstruction of individual RK-stage rates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np
from checkpoint_reader import checkpoint


def tensor(vals):
    return np.array([[vals[0],vals[1],vals[2]], [vals[1],vals[3],vals[4]], [vals[2],vals[4],vals[5]]])


def evaluate(path):
    record=checkpoint(path)
    assert len(record['state'])==1
    raw=path.read_bytes();end=raw.index(b'<par_end>\n')+len(b'<par_end>\n')
    params={};section=None
    for line in raw[:end].decode().splitlines():
        line=line.split('#',1)[0].strip()
        if line.startswith('<'): section=line[1:-1];params[section]={}
        elif '=' in line:
            key,value=line.split('=',1);params[section][key.strip()]=value.strip()
    assert float(params['problem']['bh_mass'])==0
    assert params['mesh_refinement']['refinement']=='none'
    assert params['z4c'].get('residual_gauge','standard_subtract')=='background_adapted'
    ng,nx,ny,nz=struct.unpack_from('<4i',raw,end+8+72+76)
    assert ng==4 and nx==ny==nz and nx in (16,32)
    assert record['cells']==(nx+2*ng)**3
    dx=2048/nx
    for axis in (1,2,3):
        assert int(params['mesh']['nx%d'%axis])==2*nx
        assert float(params['mesh']['x%dmin'%axis])==-2048
        assert float(params['mesh']['x%dmax'%axis])==2048
        assert int(params['meshblock']['nx%d'%axis])==nx
    assert record['total']==8
    loc_start=end+8+72+2*76+20
    loc=struct.unpack_from('<4i',raw,loc_start+16*7)
    assert loc[:3]==(1,1,1)
    u=np.asarray(record['state'][0]).reshape(25,*([nx+2*ng]*3))
    k,j,i=ng,ng,ng+nx-1
    xyz=[2048-dx/2,dx/2,dx/2]
    point=u[:,k,j,i]
    g=np.eye(3)+tensor(point[1:7]);gu=np.linalg.inv(g)
    nd=np.array([1.,0.,0.])/np.sqrt(gu[0,0]);nu=gu@nd
    assert np.linalg.eigvalsh(g).min()>0
    def derivative_xyz(n):
        # xyz: backward at the last active i; forward at j=k=ng, exactly source D2.
        return np.array([(3*u[n,k,j,i]-4*u[n,k,j,i-1]+u[n,k,j,i-2])/(2*dx),
                         (-3*u[n,k,j,i]+4*u[n,k,j+1,i]-u[n,k,j+2,i])/(2*dx),
                         (-3*u[n,k,j,i]+4*u[n,k+1,j,i]-u[n,k+2,j,i])/(2*dx)])
    def derivative(n):
        ds=derivative_xyz(n)
        return float(sum(nu[a]*ds[a] for a in range(3) if abs(nu[a])>1e-12))
    chi=1+point[0];alpha=1+point[18];theta=point[17]
    assert chi>0 and alpha>0
    gam=float(nd@point[14:17]);dchi=derivative(0)
    A=tensor(point[8:14]);Ann=float(nu@A@nu-np.trace(gu@A)/3)
    dh=tensor(np.array([derivative(n) for n in range(1,7)]))
    dhnn=float(nu@dh@nu-np.trace(gu@dh)/3)
    c1terms={'sqrtchi_Theta':float(np.sqrt(chi)*theta),
             'halfchi_Gamma_n':float(.5*chi*gam), 'D_n_chi':dchi}
    c2terms={'four_Khat_over_3sqrtchi':float(4*point[7]/(3*np.sqrt(chi))),
             'two_Theta_over_3sqrtchi':float(2*theta/(3*np.sqrt(chi))),
             'minus_two_A_nn_TF_over_sqrtchi':float(-2*Ann/np.sqrt(chi)),
             'minus_Gamma_n':-gam, 'D_n_h_nn_TF':dhnn}
    c1=sum(c1terms.values());c2=sum(c2terms.values())
    flat_dchi=float(derivative_xyz(0)[0])
    flat_c1=float(theta+.5*point[14]+flat_dchi)
    flat_dh=tensor(np.array([derivative_xyz(n)[0] for n in range(1,7)]))
    flat_c2=float(4*point[7]/3+2*theta/3-2*(A[0,0]-np.trace(A)/3)-point[14]+flat_dh[0,0]-np.trace(flat_dh)/3)
    # Longitudinal gauge amplitude from current source, with fixed adapted
    # Minkowski background beta_gauge=0, alpha_gauge=1, f=2 and G=1.
    assert float(params['z4c'].get('shift_Gamma',1))==1
    assert float(params['z4c'].get('residual_lapse_f',1))==1
    assert float(params['z4c'].get('sss_damping_amp',0))==0
    assert float(params['z4c'].get('lapse_oplog',2))*float(params['z4c'].get('lapse_harmonicf',1))==2
    assert float(params['z4c'].get('lapse_harmonic',0))==0
    beta_n=float(nd@point[19:22]);shift_q=4/3
    lam=.5*(beta_n+np.sqrt(beta_n**2+16/3));mu=lam-beta_n;delta=lam
    sep_lapse=2*chi-mu*delta;sep_light=chi*alpha**2-mu**2
    dbetan=float(nd@np.array([derivative(n) for n in range(19,22)]))
    gauge_terms={'Khat':float(alpha*delta**2*sep_light*point[7]),
       'Theta':float(.5*alpha*shift_q*sep_lapse*theta),
       'Gamma_n':float(.25*delta*(4*chi*alpha**2-3*mu**2)*sep_lapse*gam),
       'D_chi':float(.5*alpha**2*delta*sep_lapse*dchi),
       'D_alpha':float(-chi*alpha*delta*sep_light*derivative(18)),
       'D_beta_n':float(sep_lapse*sep_light*dbetan)}
    amplitude=float(params['problem'].get('outer_sponge_test_theta_pulse_amplitude',0))
    width=float(params['problem'].get('outer_sponge_test_theta_pulse_width',0))
    radius=float(params['problem'].get('outer_sponge_test_theta_pulse_radius',0))
    profile=params['problem'].get('outer_sponge_test_theta_pulse_profile','gaussian')
    dipole=int(params['problem'].get('outer_sponge_test_theta_pulse_dipole_axis',0))
    assert profile in ('gaussian','compact') and dipole in range(4)
    assert all(float(params['problem'].get('bh_center_x%d'%a,0))==0 for a in (1,2,3))
    assert float(params['problem'].get('bh_spin',0))==0
    r=float(np.linalg.norm(xyz))
    expected=0.
    if amplitude!=0:
        assert width>0
        q=(r-radius)/width
        shape=(np.exp(1-1/(1-q*q)) if q*q<1 else 0.) if profile=='compact' else np.exp(-.5*q*q)
        expected=float(amplitude*shape*(1. if dipole==0 else xyz[dipole-1]/r))
    return {'time':record['time'],'cycle':record['cycle'],'rank':7,
            'block':7,'relative_level':0,'xyz':xyz,'dx':dx,
            'global_nx':2*nx,'block_nx':nx,'nghost':ng,
            'seed':{'amplitude':amplitude,'profile':profile,'width':width,
                    'radius':radius,'dipole_axis':dipole,'initial_analytic_Theta_at_point':expected},
            'checkpoint_file':path.name,'checkpoint_sha256':hashlib.sha256(raw).hexdigest(),
            'Theta':float(theta),'chi_res':float(point[0]),'alpha_res':float(point[18]),
            'metric_res_frobenius':float(np.linalg.norm(g-np.eye(3))),
            'beta_res_norm':float(np.linalg.norm(point[19:22])),
            'normal_d':nd.tolist(),'normal_u':nu.tolist(),
            'C1_terms':c1terms,'C2_terms':c2terms,'longitudinal_gauge_terms':gauge_terms,
            'C1_in_actual_basis':c1,'C1_out_actual_basis':c1-2*c1terms['sqrtchi_Theta'],
            'C2_in_actual_basis':c2,
            'C2_out_actual_basis':-c2+2*(-gam+dhnn),
            'longitudinal_gauge_in_actual_basis':sum(gauge_terms.values()),
            'C1_in_flat_reference':flat_c1,'C2_in_flat_reference':flat_c2,
            'C1_actual_minus_flat':c1-flat_c1,'C2_actual_minus_flat':c2-flat_c2}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--prefix',default='theta_loweta_coremask')
    p.add_argument('--indices',nargs='+',type=int,help='File sequence indices; otherwise select initial and latest files')
    a=p.parse_args()
    base=a.run/'rst/rank_00000007'
    if a.indices:
        paths=[base/(a.prefix+'.%05d.rst'%n) for n in a.indices]
    else:
        found=sorted(base.glob(a.prefix+'.*.rst'))
        assert found, 'No matching checkpoints'
        paths=list(dict.fromkeys([found[0],found[-1]]))
    rows=[evaluate(path) for path in paths if path.exists()]
    assert rows and rows[0]['time']==0
    for row in rows:
        for field in ('C1_in_actual_basis','C2_in_actual_basis','longitudinal_gauge_in_actual_basis','Theta'):
            row[field+'_over_initial']=row[field]/rows[0][field] if rows[0][field]!=0 else None
            row[field+'_minus_initial']=row[field]-rows[0][field]
    out={'scope':__doc__,'missing_files':[p.name for p in paths if not p.exists()],
         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'rows':rows}
    a.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2))
