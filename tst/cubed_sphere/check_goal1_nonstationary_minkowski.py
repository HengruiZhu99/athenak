#!/usr/bin/env python3
"""Validate the exterior physical oracle, then save independent endpoint fixtures."""
import hashlib,json,time
from pathlib import Path
import mpmath as m
import goal1_nonstationary_minkowski_oracle as oracle
m.mp.dps=80
root=Path(__file__).resolve().parents[2];out=root/'docs/validation/goal1-minkowski-20261018/endpoint-dynamics-v1';work=root/'build-goal1-minkowski-20261018/endpoint-dynamics-v1';protocol=json.loads((out/'protocol-v1.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();physical=[];start=time.time()
for e0 in protocol['parameters']['epsilon']:
 e=m.mpf(str(e0))
 for t0 in protocol['parameters']['time']:
  t=m.mpf(str(t0))
  for r0 in protocol['independent_physical_sanity_radii']:
   r=m.mpf(str(r0));o,L,B,Bp,kr,kt,delta,alpha,beta,Tt,TR=oracle.radial(r,t,e);R=r/o
   def physical_T(Rnew):return (t+m.sqrt(m.mpf('.25')+Rnew**2))/(1+e/Rnew)
   tr=m.diff(physical_T,R);trr=m.diff(physical_T,R,2);s=1-tr*tr;sr=-2*tr*trr
   kr0=-trr/s**m.mpf('1.5');kt0=-tr/(R*m.sqrt(s));scalar=2*(1-1/s)/R**2+2*sr/(s*s*R)
   H=scalar+(kr0+2*kt0)**2-kr0**2-2*kt0**2
   def physical_kt(z):
    slope=m.diff(physical_T,z);return -slope/(z*m.sqrt(1-slope*slope))
   M=m.diff(physical_kt,R)-(kr0-kt0)/R
   harmonic=m.diff(lambda z:m.mpf('3.5')*(1+e/z),R,2)+2/R*m.diff(lambda z:m.mpf('3.5')*(1+e/z),R)
   values=[B-s*L*L/(o*o),kr-kr0,kt-kt0,delta-(kr0-kt0)/o,alpha-o/(1+e/R)/m.sqrt(s),TR-tr,Tt-1/(1+e/R)]
   endpointB=oracle.radial(m.mpf(1),t,e)[2]
   physical.append({'epsilon':e0,'time':t0,'radius':r0,'factorization_absolute_max':float(max(abs(v)for v in values)),'physical_H_absolute':float(abs(H)),'physical_M_absolute':float(abs(M)),'harmonic_inverse_map_absolute':float(abs(harmonic)),'B':float(B),'T_t':float(Tt),'endpoint_B_identity_absolute':float(abs(endpointB-(1-8*e*t+8*e*e)))})
keys=['factorization_absolute_max','physical_H_absolute','physical_M_absolute','harmonic_inverse_map_absolute','endpoint_B_identity_absolute'];maxima={k:max(p[k]for p in physical)for k in keys};review={'scope':__doc__,'precision_digits':80,'points':len(physical),'maxima':maxima,'spacelike_and_time_orientation_pass':all(p['B']>0 and p['T_t']>0 for p in physical),'independent_physical_oracle_pass':max(maxima.values())<=1e-60 and all(p['B']>0 and p['T_t']>0 for p in physical),'rows':physical,'source_sha256':sha(__file__),'oracle_source_sha256':sha(Path(oracle.__file__))};(out/'physical-oracle-review.json').write_text(json.dumps(review,indent=2)+'\n');print('physical oracle',maxima,flush=True);assert review['independent_physical_oracle_pass']
fixture=work/'endpoint-fixtures.txt';assert not fixture.exists();lines=[];cases=[]
rows_first=[0,1,2,3,4,5,6,7,17,18,19,20,21]
for e0 in protocol['parameters']['epsilon']:
 e=m.mpf(str(e0))
 for t0 in protocol['parameters']['time']:
  t=m.mpf(str(t0))
  for normal in [(1,0,0),(m.mpf(2)/3,-m.mpf(2)/3,m.mpf(1)/3)]:
   x=list(normal);index=len(lines);data=[]
   for k in range(4):data+=oracle.scalar_jet(lambda y,k=k:oracle.fields(y,t,e)[0][k],x)
   for i in range(3):
    data+=oracle.scalar_jet(lambda y,i=i:oracle.fields(y,t,e)[1][i],x)
    data+=oracle.scalar_jet(lambda y,i=i:oracle.fields(y,t,e)[2][i],x)
   for i,j in oracle.pairs:
    data+=oracle.scalar_jet(lambda y,i=i,j=j:oracle.fields(y,t,e)[3][i][j],x)
    data+=[oracle.fields(x,t,e)[4][i][j]]+[oracle.derivative(lambda y,i=i,j=j:oracle.fields(y,t,e)[4][i][j],x,[d])for d in range(3)]
   time_rhs=[m.diff(lambda z,k=k:oracle.native(x,z,e)[k],t)for k in range(22)]
   space_time=[oracle.derivative(lambda z,k=k:oracle.native(z[:3],z[3],e)[k],x+[t],[i,3])for k in rows_first for i in range(3)]
   Wtime=[]
   for n in range(10):
    qtt=m.diff(lambda z:oracle.configuration(x,z,e)[n],t,2);correction=m.mpf(0)
    for i in range(3):
     qgrad=oracle.derivative(lambda y:oracle.configuration(y,t,e)[n],x,[i]);qrefgrad=2*x[i]if n==6 else (-2 if n==7+i else 0)
     btime=m.diff(lambda z:oracle.fields(x,z,e)[1][i],t)
     qtimegrad=oracle.derivative(lambda z:oracle.configuration(z[:3],z[3],e)[n],x+[t],[i,3]);correction+=btime*(qgrad-qrefgrad)+oracle.fields(x,t,e)[1][i]*qtimegrad
    Wtime.append(qtt-correction)
   data+=time_rhs+space_time+Wtime;lines.append(' '.join(m.nstr(v,45)for v in [index,e,t,*x,*data]));cases.append({'case':index,'epsilon':e0,'time':t0,'x':[float(v)for v in x],'target_time_rhs_max':float(max(abs(v)for v in time_rhs)),'target_W_time_max':float(max(abs(v)for v in Wtime))});print('fixture',index,flush=True)
fixture.write_text('\n'.join(lines)+'\n');receipt={'precision_digits':80,'points':len(lines),'native_state_scalars_per_case':184,'native_time_targets':22,'native_time_spatial_rows':rows_first,'native_time_spatial_targets':39,'velocity_time_targets':10,'fixture_sha256':sha(fixture),'cases':cases,'seconds':time.time()-start,'source_sha256':sha(__file__),'oracle_source_sha256':sha(Path(oracle.__file__))};(out/'fixtures.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
