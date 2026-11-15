"""Independent conformal-Weyl numerator check; no division by Omega at scri."""
import numpy as np

def curvature(metric,dg,ddg):
 inv=np.linalg.inv(metric)
 h=np.swapaxes(dg,-3,-2)+np.moveaxis(dg,-3,-1)-dg
 gamma=.5*np.einsum('...ad,...dbc->...abc',inv,h)
 di=-np.einsum('...am,...emn,...nd->...ead',inv,dg,inv)
 dh=np.swapaxes(ddg,-3,-2)+np.moveaxis(ddg,-3,-1)-ddg
 dc=.5*(np.einsum('...ead,...dbc->...eabc',di,h)+np.einsum('...ad,...edbc->...eabc',inv,dh))
 # R^a_bcd = partial_c Gamma^a_db - partial_d Gamma^a_cb + Gamma^a_ce Gamma^e_db - Gamma^a_de Gamma^e_cb.
 mixed=np.einsum('...cadb->...abcd',dc)-np.einsum('...dacb->...abcd',dc)+np.einsum('...ace,...edb->...abcd',gamma,gamma)-np.einsum('...ade,...ecb->...abcd',gamma,gamma)
 riemann=np.einsum('...ae,...ebcd->...abcd',metric,mixed)
 ricci=np.einsum('...abad->...bd',mixed)
 scalar=np.einsum('...ab,...ab->...',inv,ricci)
 weyl=riemann-.5*(np.einsum('...ac,...bd->...abcd',metric,ricci)-np.einsum('...ad,...bc->...abcd',metric,ricci)-np.einsum('...bc,...ad->...abcd',metric,ricci)+np.einsum('...bd,...ac->...abcd',metric,ricci))+scalar[...,None,None,None,None]/6*(np.einsum('...ac,...bd->...abcd',metric,metric)-np.einsum('...ad,...bc->...abcd',metric,metric))
 return weyl,ricci,inv

if __name__=='__main__':
 from pathlib import Path
 import importlib.util,sys
 root=Path(sys.argv[1]);sys.path.insert(0,str(root/'vis/python'))
 from scri_geometry import ricci as reference_ricci,derivative_weights
 spec=importlib.util.spec_from_file_location('analytic',root/'tst/z4c_hyperboloidal/scri_geometry.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 theta=np.linspace(.2,2.9,11);phi=np.linspace(.1,5.7,11)
 directions=np.column_stack((np.cos(theta),np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi)))
 dyad=np.column_stack((np.zeros(len(theta)),(np.column_stack((-np.sin(theta),np.cos(theta)*np.cos(phi),np.cos(theta)*np.sin(phi)))-1j*np.column_stack((np.zeros(len(theta)),-np.sin(phi),np.cos(phi))))/np.sqrt(2)))
 n=np.zeros((len(theta),4));n[:,0]=1
 radii=np.linspace(.97,1,7);t=.4
 for amplitude in [0,.02]:
  numerators=[]
  for r in radii:
   g,dg,ddg,_,_=m.radiative_metric(t,r*directions,amplitude=amplitude)
   c,ric,inv=curvature(g,dg,ddg)
   np.testing.assert_allclose(ric,reference_ricci(g,dg,ddg)[0],atol=2e-14)
   np.testing.assert_allclose(np.einsum('...ac,...abcd->...bd',inv,c),0,atol=2e-14)
   numerators.append(np.einsum('qa,qb,qc,qd,qabcd->q',n,dyad,n,dyad,c))
  numerators=np.array(numerators)
  np.testing.assert_allclose(numerators[-1],0,atol=1e-14)
  psi0=2*np.einsum('r,rq->q',derivative_weights(radii,1.,1),numerators)
  expected=-amplitude*1.7**2*np.sin(1.7*t)*np.sin(theta)**2
  print('amplitude',amplitude,'max radiation error',np.max(abs(psi0-expected)),'signal',max(abs(expected)))
  np.testing.assert_allclose(psi0,expected,atol=2e-9)
 print('PASS Weyl trace, Ricci convention, zero boundary numerator and direct-scri radiative limit')
