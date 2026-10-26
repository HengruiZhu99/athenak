#!/usr/bin/env python3
"""Independent physical pullback checks and native-only oracle fixtures."""
from pathlib import Path
import argparse,json,time,hashlib,math
import mpmath as m
from goal1_global_clock_oracle import Clock
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--work',type=Path,required=True);p.add_argument('--group',choices=['physical','fixtures'],required=True);p.add_argument('--batch',type=int,default=1);a=p.parse_args();m.mp.dps=80
cache=json.loads((a.output/'reused-height-cache.json').read_text());EPS=m.mpf('5e-6');start=time.time()
def save(name,j): (a.output/name).write_text(json.dumps(j,indent=2)+'\n')
def zero(shape):
 if not shape:return m.mpf(0)
 return [zero(shape[1:]) for _ in range(shape[0])]
def error(a,b):return abs(a-b)/max(1,abs(a),abs(b))

def physical():
 records=[]
 for rs in ['0','.3','.7','.98','1']:
  clock=Clock(rs,cache);r=m.mpf(rs)
  for ts in ['0','.75']:
   t=m.mpf(ts);e=EPS;c=clock.core(r,t,e);f=clock.radial(r,t,e);o,L,G,q,v=[c[k] for k in ['o','L','G','q','v']];errs=[]
   if r==0:
    T=c['T'];H,HT,HR=clock.wave(T,0);wave=-m.diff(lambda s:clock.wave(s,0)[0],T,2)+3*m.diff(lambda R:clock.wave(T,R)[0],m.mpf(0),2)
    implicit=T+e*H-t;origin=abs(q-1/(1-2*e*clock.F(T,2)));rawH=rawM=m.mpf(0)
    inverse_negative=abs(1-q);slice=1
   elif o!=0:
    R=r/o;T=clock.physical_T(r,t,e);H,HT,HR=clock.wave(T,R)
    wave=-m.diff(lambda z:clock.wave(z,R)[0],T,2)+m.diff(lambda z:clock.wave(T,z)[0],R,2)+2/R*m.diff(lambda z:clock.wave(T,z)[0],R)
    implicit=T+e*H-(t+clock.uref(r)+R);origin=0
    Rp=m.diff(lambda z:z/clock.reference(z)[0],r);Rpp=m.diff(lambda z:z/clock.reference(z)[0],r,2);Tr=m.diff(lambda z:clock.physical_T(z,t,e),r);Trr=m.diff(lambda z:clock.physical_T(z,t,e),r,2);Tt=m.diff(lambda z:clock.physical_T(r,z,e),t)
    slope=Tr/Rp;sr=-2*slope*(Trr*Rp-Tr*Rpp)/Rp**3;s=1-slope*slope
    kr=-(Trr*Rp-Tr*Rpp)/(Rp**3*s**m.mpf('1.5'));kt=-slope/(R*m.sqrt(s));curvature=2*(1-1/s)/R**2+2*sr/(s*s*R);rawH=curvature+(kr+2*kt)**2-kr**2-2*kt**2
    def kt_pull(z):
     Rz=z/clock.reference(z)[0];Rzp=m.diff(lambda y:y/clock.reference(y)[0],z);vz=m.diff(lambda y:clock.physical_T(y,t,e),z)/Rzp
     return -vz/(Rz*m.sqrt(1-vz*vz))
    rawM=m.diff(kt_pull,r)/Rp-(kr-kt)/R
    errs=[G-o*o*(Rp*Rp-Tr*Tr),f[2]-(kr+2*kt),f[8]/(f[0]*G)-(kr-kt)*2/(3*o),f[1]-o*Tt/m.sqrt(s),f[4]+Tt*Tr/(Rp*Rp-Tr*Tr),v-slope,q-Tt]
    slice=s;inverse_negative=abs(1-Tt)
   else:
    U=t+clock.constant;exactG=1-8*e*clock.F(U);wave=implicit=origin=rawH=rawM=0;errs=[G-exactG,f[1]-2/m.sqrt(exactG),f[2]+6/m.sqrt(exactG),f[4]+2/exactG,f[8]/(f[0]*G)-m.mpf(4)/3*e*clock.F(U,1)/m.sqrt(exactG)]
    slice=G;inverse_negative=abs(G-1)
   rec={'r':rs,'t':ts,'wave_absolute':float(abs(wave)),'implicit_absolute':float(abs(implicit)),'origin_identity':float(origin),'pullback_scaled_max':float(max([abs(z) for z in errs]+[m.mpf(0)])),'physical_H_absolute':float(abs(rawH)),'physical_M_absolute':float(abs(rawM)),'q':float(q),'slice_spacelike_measure':float(slice),'omit_inverse_or_leading_change_negative':float(inverse_negative)};records.append(rec);print('physical',rs,ts,flush=True);save('physical-partial.json',records)
 keys=['wave_absolute','implicit_absolute','origin_identity','pullback_scaled_max','physical_H_absolute','physical_M_absolute'];maxima={k:max(x[k] for x in records) for k in keys};passed=max(maxima.values())<1e-50 and all(x['q']>0 and x['slice_spacelike_measure']>0 for x in records)
 review={'scope':'Physical pullback/harmonic inverse and endpoint factors only. Not native or evolution admission. Origin H/M use isotropic analytic limit; finite nonzero radii use independently differentiated physical embedding.','points':len(records),'maxima':maxima,'passed':passed,'records':records,'height_uncertainty':cache['declared_height_uncertainty'],'seconds':time.time()-start};save('physical-review.json',review);assert passed,review;print(json.dumps({k:v for k,v in review.items() if k!='records'}),flush=True)

# Native tensor calculus reconstruction independently implemented, not imported from native radial helper.
def fixture(clock,r,t,e,n):
 f=clock.radial(r,t,e);dr=[m.diff(lambda z,k=k:clock.radial(z,t,e)[k],r) for k in range(10)];dd=[m.diff(lambda z,k=k:clock.radial(z,t,e)[k],r,2) if k not in [3,5,8,9] else m.mpf(0) for k in range(10)]
 if r==0:
  # Exact isotropic/odd origin limits. Lambda'' and A'' are unused and not claimed.
  Gr2=m.diff(lambda z:clock.core(z,t,e)['G'],r,2);dr[5]=m.mpf(5)/3*Gr2
 dn=zero([3,3]);ddn=zero([3,3,3])
 if r:
  for d in range(3):
   for i in range(3):
    dn[d][i]=((d==i)-n[d]*n[i])/r
    for z in range(3):ddn[d][z][i]=(-(d==i)*n[z]-(z==i)*n[d]-(d==z)*n[i]+3*n[d]*n[z]*n[i])/r**2
 def scalar(k):
  d=[dr[k]*z for z in n] if r else [m.mpf(0)]*3
  h=[[dd[k]*n[i]*n[j]+dr[k]*dn[i][j] for j in range(3)]for i in range(3)] if r else [[dd[k]*(i==j)for j in range(3)]for i in range(3)]
  return f[k],d,h
 def vector(k):
  vs=[]
  for i in range(3):
   value=f[k]*n[i]
   d=[dr[k]*n[z]*n[i]+f[k]*dn[z][i] for z in range(3)] if r else [dr[k]*(z==i) for z in range(3)]
   h=[[dd[k]*n[z]*n[y]*n[i]+dr[k]*(dn[z][y]*n[i]+n[z]*dn[y][i]+n[y]*dn[z][i])+f[k]*ddn[z][y][i] for y in range(3)]for z in range(3)] if r else zero([3,3])
   if k==5:h=zero([3,3]) # Native Lambda second derivatives not consumed/claimed.
   vs.append((value,d,h))
  return vs
 def tensor(kr,kt,second):
  vals=[]
  for i,j in [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]:
   val=f[kt]*(i==j)+(f[kr]-f[kt])*n[i]*n[j]
   ds=[dr[kt]*n[z]*(i==j)+(dr[kr]-dr[kt])*n[z]*n[i]*n[j]+(f[kr]-f[kt])*(dn[z][i]*n[j]+n[i]*dn[z][j])for z in range(3)] if r else [m.mpf(0)]*3
   hs=zero([3,3])
   if second:
    for d in range(3):
     for z in range(3):
      if r:hs[d][z]=(dd[kt]*n[d]*n[z]+dr[kt]*dn[d][z])*(i==j)+((dd[kr]-dd[kt])*n[d]*n[z]+(dr[kr]-dr[kt])*dn[d][z])*n[i]*n[j]+(dr[kr]-dr[kt])*(n[d]*(dn[z][i]*n[j]+n[i]*dn[z][j])+n[z]*(dn[d][i]*n[j]+n[i]*dn[d][j]))+(f[kr]-f[kt])*(ddn[d][z][i]*n[j]+dn[d][i]*dn[z][j]+dn[z][i]*dn[d][j]+n[i]*ddn[d][z][j])
      else:hs[d][z]=dd[kt]*(d==z)*(i==j)+(dd[kr]-dd[kt])/2*((d==i)*(z==j)+(d==j)*(z==i))
   vals.append((val,ds,hs))
  return vals
 s=[scalar(k)for k in range(4)];b=vector(4);l=vector(5);g=tensor(6,7,True);A=tensor(8,9,False)
 def serial(q):return [q[0]]+q[1]+[q[2][i][j]for i in range(3)for j in range(i,3)]
 data=[]
 for q in s:data+=serial(q)
 for i in range(3):data+=serial(b[i])+serial(l[i])
 for i in range(6):data+=serial(g[i])+[A[i][0]]+A[i][1]
 assert len(data)==184
 # Reuse linear Cartesian tensor reconstruction on independent radial time jets.
 ft=[m.diff(lambda tt,k=k:clock.radial(r,tt,e)[k],t)for k in range(10)]
 frt=[m.diff(lambda rr,tt,k=k:clock.radial(rr,tt,e)[k],(r,t),(1,1))if k!=5 else m.mpf(0)for k in range(10)]
 targets=[ft[0]]+[ft[7]*(i==j)+(ft[6]-ft[7])*n[i]*n[j]for i,j in [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]]+[ft[2]]+[ft[9]*(i==j)+(ft[8]-ft[9])*n[i]*n[j]for i,j in [(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]]+[ft[5]*z for z in n]+[ft[3],ft[1]]+[ft[4]*z for z in n]
 first=[zero([3])for k in range(22)]
 for row,k in [(0,0),(7,2),(17,3),(18,1)]:first[row]=[frt[k]*z for z in n] if r else zero([3])
 for row,(i,j) in enumerate([(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)],1):
  first[row]=[frt[7]*n[z]*(i==j)+(frt[6]-frt[7])*n[z]*n[i]*n[j]+(ft[6]-ft[7])*(dn[z][i]*n[j]+n[i]*dn[z][j])for z in range(3)] if r else zero([3])
 for i in range(3):first[19+i]=[frt[4]*n[z]*n[i]+ft[4]*dn[z][i]for z in range(3)] if r else [frt[4]*(z==i)for z in range(3)]
 first=[v for k in [0,1,2,3,4,5,6,7,17,18,19,20,21]for v in first[k]]
 return data,targets,first

def fixtures():
 # First source-admission batch spans origin, transition, exterior and exact scri.
 specs=[(r,t,'5e-6')for r,t in [('0','0'),('0','.4'),('.3','0'),('.7','.4'),('.98','.75'),('1','0'),('1','.75')]] if a.batch==1 else [(r,t,'5e-6')for r,t in [('.025','.4'),('.025','.4001'),('.05','0'),('.5','.4'),('.9','.75'),('.95','.75'),('.3','.4'),('1','1.5')]]+[(r,t,'0')for r,t in [('.3','.4'),('.7','0'),('1','.75')]]+[(r,t,'-5e-6')for r,t in [('.3','0'),('.7','.4'),('1','.75')]];manifest=[]
 for rs,ts,en in specs:
  stem=f'fixture-r{rs}-t{ts}' if a.batch==1 else f'fixture-e{en}-r{rs}-t{ts}'
  path=a.work/(stem+'.txt');meta=a.output/(stem+'.json')
  if path.exists() and meta.exists():manifest.append(json.loads(meta.read_text()));continue
  clock=Clock(rs,cache);r=m.mpf(rs);t=m.mpf(ts);e=m.mpf(en);n=[m.mpf(2)/3,-m.mpf(2)/3,m.mpf(1)/3]if r else [m.mpf(0)]*3
  index=(0 if a.batch==1 else 7)+len(manifest);data,target,first=fixture(clock,r,t,e,n);flat=[index,e,t,*[r*z for z in n],*data,*target,*first];mpath=path.with_suffix('.mp.txt');mpath.write_text(' '.join(m.nstr(v,65)for v in flat)+'\n');path.write_text(' '.join(format(float(v),'.17g')for v in flat)+'\n');rec={'case':index,'file':path.name,'r':rs,'t':ts,'epsilon':str(e),'native_serialized_state_scalars':len(data),'native_time_targets':len(target),'native13_time_first_targets':len(first),'MP_source_sha256':hashlib.sha256(mpath.read_bytes()).hexdigest(),'IEEE_input_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'max_IEEE_scaled_roundoff':float(max(error(v,m.mpf(float(v)))for v in flat)),'seconds':time.time()-start};meta.write_text(json.dumps(rec,indent=2)+'\n');manifest.append(rec);print('fixture',rs,ts,flush=True)
 (a.work/('fixtures.txt' if a.batch==1 else 'fixtures-v2.txt')).write_text(''.join((a.work/r['file']).read_text()for r in manifest));save('fixtures.json' if a.batch==1 else 'fixtures-v2.json',{'rows':manifest,'precision_digits':80,'height_uncertainty':cache['declared_height_uncertainty'],'native_added_evolved_fields':0,'unused_inputs':'Native Lambda DD and A DD not consumed/claimed; origin formulas use analytic isotropy. No momenta/reduction targets.','seconds':time.time()-start})
if a.group=='physical':physical()
else:fixtures()
