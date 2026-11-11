#include <iostream>
#include <stdexcept>
#include "utils/circumbinary_geometry.hpp"
#include "utils/circumbinary_accumulator.hpp"
#include "dyn_grmhd/valencia_sources.hpp"
#include "dyn_grmhd/valencia_surface_flux.hpp"
#include "dyn_grmhd/valencia_diagnostics.hpp"
#include "utils/circumbinary_budget.hpp"

void Check(bool yes,const char *message) { if (!yes) throw std::runtime_error(message); }
void Near(Real a,Real b,Real tol=1e-11) { Check(std::abs(a-b)<tol,"numerical mismatch"); }

// These accessors fail if an inactive direction is ever accessed.
struct FakeADM {
  struct Scalar { Real operator()(int,int k,int j,int i) const {
    Check(k==0&&j==0,"collapsed scalar dimension accessed");return 2.0+i*0.25;
  }} alpha;
  struct Vector { Real operator()(int,int a,int k,int j,int i) const {
    Check(k==0&&j==0,"collapsed vector dimension accessed");return a+i*0.25;
  }} beta_u;
  struct Tensor { Real operator()(int,int a,int b,int k,int j,int i) const {
    Check(k==0&&j==0,"collapsed tensor dimension accessed");return a+b+i*0.25;
  }} g_dd;
};

struct RotADM {
  struct Scalar {Real operator()(int,int,int,int)const{return 1;}}alpha;
  struct Vector {Real operator()(int,int a,int,int j,int i)const{
    return a==0 ? -0.03*j : a==1 ? 0.03*i:0;
  }}beta_u;
  struct Tensor {Real operator()(int,int a,int b,int,int,int)const{return a==b ? 1:0;}}g_dd;
};
struct TestPrim {Real operator()(int,int n,int,int,int)const {
  return n==IDN ? 1.0:n==IPR ? 0.1:n==IVX ? 0.2:n==IVY ? -0.1:0.05;
}};
struct TestB {Real operator()(int,int,int,int,int)const{return 0;}};
void KerrSpatialReference(const Real x[3],Real mass,const Real chi[3],const Real v[3],Real g[6]) {
  Real a[3],v2=0,a2=0;for(int i=0;i<3;++i){a[i]=mass*chi[i];v2+=v[i]*v[i];a2+=a[i]*a[i];}
  Real gamma=1/sqrt(1-v2),q=gamma*gamma/(gamma+1),L[3][3],xb[3]={};
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){L[i][j]=(i==j?1.0:0.0)+q*v[i]*v[j];xb[i]+=L[i][j]*x[j];}
  Real x2=0,ad=0;for(int i=0;i<3;++i){x2+=xb[i]*xb[i];ad+=a[i]*xb[i];}
  Real r=sqrt(0.5*(x2-a2+sqrt((x2-a2)*(x2-a2)+4*ad*ad)));
  Real cross[3]={xb[1]*a[2]-xb[2]*a[1],xb[2]*a[0]-xb[0]*a[2],xb[0]*a[1]-xb[1]*a[0]};
  Real ell[3],h[3]={};
  for(int i=0;i<3;++i)ell[i]=(r*xb[i]+cross[i]+ad*a[i]/r)/(r*r+a2);
  for(int i=0;i<3;++i){h[i]=-gamma*v[i];for(int j=0;j<3;++j)h[i]+=ell[j]*L[j][i];}
  Real factor=2*mass*r*r*r/(r*r*r*r+ad*ad);int index=0;
  for(int i=0;i<3;++i)for(int j=i;j<3;++j)g[index++]=(i==j ? 1.0:0.0)+factor*h[i]*h[j];
}
void AreaTests() {
  Real zero[3]={},tilt[3]={0.3,0.4,0.5},boost[3]={0.35,-0.2,0.1};
  for(int configuration=0;configuration<3;++configuration){
    Real *chi=configuration==0 ? zero:tilt,*velocity=configuration==2 ? boost:zero;
    circumbinary::Quadric quad;
    circumbinary::KerrQuadric(1.0,chi,velocity,quad.H,[](Real z){return sqrt(z);});
    Real spin2=0;for(int d=0;d<3;++d)spin2+=chi[d]*chi[d];
    Real rp=1+sqrt(1-spin2),exact=4*M_PI*(rp*rp+spin2),previous=0;
    for(int nt:{40,80,160}){
      FluxAccumulator integral,combined;
      for(int t=0;t<nt;++t)for(int p=0;p<2*nt;++p){
        Real th=(t+0.5)*M_PI/nt,ph=p*M_PI/nt,r=quad.Radius(th,ph);
        Real n[3]={sin(th)*cos(ph),sin(th)*sin(ph),cos(th)};
        Real nth[3]={cos(th)*cos(ph),cos(th)*sin(ph),-sin(th)};
        Real nph[3]={-sin(th)*sin(ph),sin(th)*cos(ph),0};
        Real rt=0,rf=0;
        for(int i=0;i<3;++i)for(int j=0;j<3;++j){rt-=r*r*r*nth[i]*quad.H[i][j]*n[j];rf-=r*r*r*nph[i]*quad.H[i][j]*n[j];}
        Real et[3],ef[3],x[3],g[6];
        for(int d=0;d<3;++d){et[d]=rt*n[d]+r*nth[d];ef[d]=rf*n[d]+r*nph[d];x[d]=r*n[d];}
        KerrSpatialReference(x,1,chi,velocity,g);
        Real det=adm::SpatialDet(g[0],g[1],g[2],g[3],g[4],g[5]);
        Real w=sqrt(det)*M_PI*M_PI/(nt*nt);
        Real area[3]={w*(et[1]*ef[2]-et[2]*ef[1]),w*(et[2]*ef[0]-et[0]*ef[2]),w*(et[0]*ef[1]-et[1]*ef[0])};
        ValenciaSurfacePoint(1,0,5.0/3,zero,zero,g,det,1,zero,x,zero,area,integral);
        if(configuration==1 && nt==160){
          Real other_x[3]={x[0]-6,x[1],x[2]},other[6],full[6],full_area[3];
          KerrSpatialReference(other_x,0.4,zero,zero,other);
          for(int d=0;d<6;++d)full[d]=g[d]+other[d]-((d==0||d==3||d==5)?1.0:0.0);
          Real full_det=adm::SpatialDet(full[0],full[1],full[2],full[3],full[4],full[5]);
          for(int d=0;d<3;++d)full_area[d]=area[d]*sqrt(full_det/det);
          ValenciaSurfacePoint(1,0,5.0/3,zero,zero,full,full_det,1,zero,x,zero,full_area,combined);
        }

      }
      if(configuration==1 && nt==160)
        Check(std::abs(combined.area/exact-1)>0.001,"superposed area incorrectly forced to isolated Kerr value");
      Real error=std::abs(integral.area/exact-1);
      Check(error<0.6/(nt*nt),"isolated Kerr area");
      if(previous>0)Check(previous/error>3.8,"surface quadrature order");previous=error;
      std::cout<<"Kerr case="<<configuration<<" nt="<<nt<<" area relative error="<<error<<"\n";
    }
  }
}

void RegionalConvergenceTests() {
  circumbinary::BudgetGeometry geo;geo.inner=2;geo.outer=3;geo.holes=1;
  geo.center[0][0]=0.7;geo.radius[0]=0.75;
  for(int d=0;d<3;++d)geo.excision_H[0][d][d]=16;
  Real a=0.02,b=0.03,zero[3]={},g[6]={1,0,0,1,0,1};
  auto flux=[&](Real r,Real cx){
    FluxAccumulator sum;int nt=160;
    for(int t=0;t<nt;++t)for(int p=0;p<2*nt;++p){
      Real th=(t+0.5)*M_PI/nt,ph=p*M_PI/nt;
      Real n[3]={sin(th)*cos(ph),sin(th)*sin(ph),cos(th)};
      Real x[3]={cx+r*n[0],r*n[1],r*n[2]};
      Real wv[3]={a*x[0]-b*x[1],a*x[1]+b*x[0],a*x[2]};
      Real ds[3];for(int d=0;d<3;++d)ds[d]=r*r*sin(th)*M_PI*M_PI/(nt*nt)*n[d];
      ValenciaSurfacePoint(1,0,5.0/3,wv,zero,g,1,1,zero,x,zero,ds,sum);
    }
    return sum.moving[13]+sum.moving[16];
  };
  Real h=flux(0.75,0.7),ex=flux(0.25,0.7),inner=flux(2,0),outer=flux(3,0);
  Real boundary[4]={h-ex,0,inner-h,outer-inner};
  Real coarse=0,finest=0;
  for(int cells:{16,32,64}) {
    Real sources[4]={},dx=6.4/cells;
    // Fractional Cartesian samples use the production disjoint-region classifier.
    for(int k=0;k<cells*2;++k)for(int j=0;j<cells*2;++j)for(int i=0;i<cells*2;++i){
      Real x=-3.2+(i+0.5)*dx/2,y=-3.2+(j+0.5)*dx/2,z=-3.2+(k+0.5)*dx/2;
      int r=geo.Region(x,y,z);if(r<0)continue;
      sources[r]+=5*a*b*(x*x+y*y+0*z)*dx*dx*dx/8;
    }
    Real norm=0;
    for(int r:{0,2,3})norm+=std::abs(boundary[r]+sources[r]);
    if(cells==16)coarse=norm;if(cells==64)finest=norm;
    std::cout<<"manufactured regional cells="<<cells<<" absolute residual L1="<<norm<<"\n";
  }
  Check(finest<coarse/3,"regional geometric residual convergence");
}

void SurfaceTests() {
  Real metric[6]={1,0,0,1,0,1},zero[3]={},normal[3]={0.4,0.2,0.7};
  Real x[3]={2,3,1},velocity[3]={0.2,-0.1,0.3};
  Real W=1/sqrt(1-0.14),wv[3]={W*0.2,W*(-0.1),W*0.3};
  FluxAccumulator comoving;
  ValenciaSurfacePoint(1,0.1,5.0/3,wv,zero,metric,1,1,zero,x,velocity,normal,comoving);
  Near(comoving.moving[0],0);Near(comoving.moving[1],0);Near(comoving.moving[2],0);
  Real tangent[3]={normal[1],-normal[0],0},tangential_motion[3];
  for(int d=0;d<3;++d)tangential_motion[d]=velocity[d]+tangent[d];
  FluxAccumulator reparameterized;
  ValenciaSurfacePoint(1,0.1,5.0/3,wv,zero,metric,1,1,zero,x,tangential_motion,normal,reparameterized);
  Near(reparameterized.moving[0],comoving.moving[0]);
  Real outgoing_wv[3]={0.1,0,0},unit_x[3]={1,0,0};FluxAccumulator outgoing;
  ValenciaSurfacePoint(1,0,5.0/3,outgoing_wv,zero,metric,1,1,zero,x,zero,unit_x,outgoing);
  Near(outgoing.moving[0],-0.1);Near(outgoing.moving[1],0);Near(outgoing.moving[2],0.1);
  // Same physical worldtube in a rotating coordinate basis at a common instant.
  Real omega=0.03,shift[3]={-omega*x[1],omega*x[0],0};
  Real wr[3];for(int d=0;d<3;++d)wr[d]=velocity[d]-shift[d];
  FluxAccumulator inertial,rotating;
  Real B[3]={0.1,-0.04,0.2};
  ValenciaSurfacePoint(1,0.1,5.0/3,wv,B,metric,1,1,zero,x,velocity,normal,inertial);
  ValenciaSurfacePoint(1,0.1,5.0/3,wv,B,metric,1,1,shift,x,wr,normal,rotating);
  Near(inertial.moving[0],rotating.moving[0]);
  for(int part=0;part<2;++part){
    Near(inertial.moving[13+3*part],rotating.moving[13+3*part]);
    Near(rotating.moving[17+part],inertial.moving[17+part]-omega*inertial.moving[13+3*part]);
  }
  // Rotation of vector components; Wv rotates as a spatial vector.
  Real angle=0.7,c=cos(angle),ss=sin(angle);
  auto rotate=[&](Real *a){Real ax=a[0],ay=a[1];a[0]=c*ax+ss*ay;a[1]=-ss*ax+c*ay;};
  rotate(x);rotate(wv);rotate(B);rotate(normal);rotate(velocity);
  FluxAccumulator turned;
  ValenciaSurfacePoint(1,0.1,5.0/3,wv,B,metric,1,1,zero,x,velocity,normal,turned);
  Near(turned.moving[0],inertial.moving[0]);
  Near(turned.moving[5],c*inertial.moving[5]+ss*inertial.moving[6]);
  Near(turned.moving[6],-ss*inertial.moving[5]+c*inertial.moving[6]);
  Near(turned.moving[13],inertial.moving[13]);
  // Expanding surface in static fluid: net mass flux equals d(rho*volume)/dt.
  for(int nt:{40,80,160}){
    FluxAccumulator expanding;Real radius=2,rdot=0.3;
    for(int t=0;t<nt;++t)for(int p=0;p<2*nt;++p){
      Real th=(t+0.5)*M_PI/nt,ph=p*M_PI/nt;
      Real n[3]={sin(th)*cos(ph),sin(th)*sin(ph),cos(th)},pos[3],w[3],dA[3];
      for(int d=0;d<3;++d){pos[d]=radius*n[d];w[d]=rdot*n[d];
        dA[d]=radius*radius*sin(th)*M_PI*M_PI/(nt*nt)*n[d];}
      ValenciaSurfacePoint(1,0,5.0/3,zero,zero,metric,1,1,zero,pos,w,dA,expanding);
    }
    Real exact=4*M_PI*radius*radius*rdot;
    Check(std::abs(expanding.moving[0]/exact-1)<0.5/(nt*nt),"expanding mass flux");
    Near(expanding.moving[2],0);
    std::cout<<"surface nt="<<nt<<" expanding relative error="<<expanding.moving[0]/exact-1<<"\n";
  }
  // Densitized magnetic field with a nonunit determinant: half-unsigned flux,
  // one determinant, no extra hemisphere factor or scalar Reynolds correction.
  {
    FluxAccumulator magnetic;Real curved[6]={4,0,0,9,0,16},Btilde[3]={24,0,0};int nt=160;
    for(int t=0;t<nt;++t)for(int p=0;p<2*nt;++p){
      Real th=(t+0.5)*M_PI/nt,ph=p*M_PI/nt;
      Real n[3]={sin(th)*cos(ph),sin(th)*sin(ph),cos(th)},pos[3],ds[3];
      for(int d=0;d<3;++d){pos[d]=2*n[d];ds[d]=24*4*sin(th)*M_PI*M_PI/(nt*nt)*n[d];}
      ValenciaSurfacePoint(1,0,5.0/3,zero,Btilde,curved,576,1,zero,pos,zero,ds,magnetic);
    }
    Check(std::abs(magnetic.phiB/(24*4*M_PI)-1)<1e-4,"magnetic determinant/half-flux convention");
  }
  // Flat rotating torque includes the product-rule term and cancels pointwise.
  for(int ng=2;ng<=4;++ng)for(int dim=2;dim<=3;++dim){
    RotADM adm;TestPrim prim;TestB bcc;Real idx[3]={1,1,1},tau[3];
    ValenciaTorqueCell(adm,prim,bcc,ng,dim,5.0/3,idx,0,0,3,4,4,3,0,tau);
    Near(tau[2],0);
  }
  // Region classification is disjoint, including actual exclusions.
  circumbinary::BudgetGeometry geo;geo.inner=3;geo.outer=5;geo.holes=2;
  geo.center[0][0]=1;geo.center[1][0]=-1;geo.radius[0]=geo.radius[1]=0.8;
  for(int h=0;h<2;++h)for(int d=0;d<3;++d)geo.excision_H[h][d][d]=25;
  Near(geo.Region(1,0,0),-1);Near(geo.Region(1,0.5,0),0);
  Near(geo.Region(-1,0.5,0),1);Near(geo.Region(0,2,0),2);Near(geo.Region(0,4,0),3);
  // A conservative momentum remap need not preserve the angular first moment.
  // S_y=x on a cube: one coarse cell and eight fine cells both store net P_y=0,
  // but their piecewise-constant angular first moments differ by 2.
  Real fine_p=0,fine_j=0;
  for(int k=0;k<2;++k)for(int j=0;j<2;++j)for(int i=0;i<2;++i){
    Real xx=i-0.5;fine_p+=xx;fine_j+=xx*xx;
  }
  Near(fine_p,0);Near(fine_j,2);
  // Stage correction propagation: a correction after RK2 stage 1 contributes half.
  circumbinary::QualifiedMass correction;correction.Begin();
  correction.Stage(0,1,2,true,0);correction.Stage(0.5,1,3,true,0);
  correction.Commit(1,1,1);Near(correction.mass,4);
}

int main() {
  RegionalConvergenceTests();
  AreaTests();
  SurfaceTests();
  using namespace circumbinary;
  Real zero[3]={}, H[3][3];
  KerrQuadric(1.0,zero,zero,H,[](Real x){return sqrt(x);});
  Quadric q;
  for (int i=0;i<3;++i) for (int j=0;j<3;++j) q.H[i][j]=H[i][j];
  Near(q.Radius(0.9,0.7),2.0);
  Real velocity[3]={0.6,0,0};
  KerrQuadric(1.0,zero,velocity,H,[](Real x){return sqrt(x);});
  Near(1/sqrt(H[0][0]),1.6);Near(1/sqrt(H[1][1]),2.0);
  Real chi[3]={0.3,0.4,0.5};
  KerrQuadric(0.7,chi,velocity,H,[](Real x){return sqrt(x);});
  Real rp=0.7*(1+sqrt(0.5));
  for (int n=0;n<100;++n) {
    Real th=(n+0.5)*M_PI/100,ph=n*0.71;
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) q.H[i][j]=H[i][j];
    Real r=q.Radius(th,ph),x[3]={r*sin(th)*cos(ph),r*sin(th)*sin(ph),r*cos(th)};
    // Exact ordering: subtract center, inverse boost, then arbitrary-spin Kerr radius.
    x[0]*=1.25;
    Real a2=0.49*0.5,ad=0,x2=0;
    for (int i=0;i<3;++i) {ad+=0.7*chi[i]*x[i];x2+=x[i]*x[i];}
    Real rks=sqrt(0.5*(x2-a2+sqrt((x2-a2)*(x2-a2)+4*ad*ad)));
    Near(rks,rp);
  }
  // Deforming/rotating quadric kinematics, including translating center.
  q.center_dot[0]=0.3;q.H_dot[0][0]=0.2;q.H_dot[0][1]=q.H_dot[1][0]=0.05;
  for (int n=0;n<100;++n) {
    Real th=(n+0.5)*M_PI/100,ph=n*0.71,r=q.Radius(th,ph);
    Real x[3]={r*sin(th)*cos(ph),r*sin(th)*sin(ph),r*cos(th)},w[3];q.Motion(th,ph,w);
    Real ft=0,adv=0;
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
      ft+=x[i]*q.H_dot[i][j]*x[j]-2*q.center_dot[i]*q.H[i][j]*x[j];
      adv+=2*w[i]*q.H[i][j]*x[j];
    }
    Near(ft+adv,0.0);
  }
  for (int ng=2;ng<=4;++ng) {
    FakeADM adm; Real idx[3]={4,4,4},da[3],db[3][3],dg[3][3][3];
    ValenciaGeometryDerivatives(ng,adm,1,idx,0,0,0,4,da,db,dg);
    Near(da[0],1.0);Near(da[1],0);Near(da[2],0);Near(db[2][1],0);Near(dg[1][0][0],0);
  }
  {
    Real E=1.5,S[3]={0.2,-0.7,0.31},W[3][3],da[3],db[3][3],dg[3][3][3];
    Real old[3]={-1,1,3},shared[3]={-1,1,3},dt=0.37,vol=0.091,alpha=0.8;
    for(int a=0;a<3;++a){da[a]=0.12*(a+1);for(int b=0;b<3;++b){
      W[a][b]=0.01*(a+b+1);db[a][b]=0.02*(a-b);
      for(int c=0;c<3;++c)dg[a][b][c]=0.01*(a+b+c);
    }}
    for(int a=0;a<3;++a){for(int b=0;b<3;++b){
      for(int c=0;c<3;++c)old[a]+=0.5*dt*alpha*vol*W[b][c]*dg[a][b][c];
      old[a]+=dt*vol*S[b]*db[a][b];
    }old[a]-=dt*vol*E*da[a];}
    ValenciaMomentumSources(E,S,W,alpha,da,db,dg,dt,[&](int a,Real term){shared[a]+=term;},vol);
    for(int a=0;a<3;++a)Check(old[a]==shared[a],"evolution source arithmetic changed");
  }
  // Actual fixed-u1 MHD recurrence, stage abscissae from the same scalar update.
  for (int method=1;method<=3;++method) for (int count:{10,20,40}) {
    Real gam[3]={0,0.5,0},beta[3]={1,0.5,0};
    if (method==3) {gam[1]=0.25;gam[2]=2.0/3;beta[1]=0.25;beta[2]=2.0/3;}
    Real a0[3]={1,0.5,0};if(method==3){a0[1]=0.75;a0[2]=1.0/3;}
    QualifiedMass mass,constant;
    for (int n=0;n<count;++n) {
      Real t=Real(n)/count,dt=1.0/count,c=0;
      mass.Begin();constant.Begin();
      for (int s=0;s<method;++s) {
        mass.Stage(gam[s],beta[s]*dt,(t+c)*(t+c),true,0);
        constant.Stage(gam[s],beta[s]*dt,2,true,0);
        c=gam[s]*c+beta[s]*dt;
      }
      mass.Commit(n+1,t+dt,dt);constant.Commit(n+1,t+dt,dt);
      Real stored=constant.mass;constant.Commit(n+1,t+dt,dt);Near(constant.mass,stored);
    }
    Near(constant.mass,2.0);Near(constant.covered,1.0);
    if (method==3) Near(mass.mass,1.0/3);
    if (method==2) Near(mass.mass,1.0/3+1.0/(6*count*count));
    // Reproduce the actual uniform-mesh prescribed-ADM epoch ordering for RK2:
    // stage 1 sees the previous setter epoch, then the stage task refreshes ADM
    // at mesh time; stage 2 sees that epoch. It is not the RK state abscissa.
    Real geometry_integral=0,metric_epoch=0;
    for(int n=0;n<count;++n){
      Real t=Real(n)/count,dt=1.0/count;
      Real trial=dt*metric_epoch;
      metric_epoch=t;
      trial=0.5*trial+0.5*dt*metric_epoch;
      geometry_integral+=trial;
    }
    Near(0.5-geometry_integral,1.0/count-0.5/(count*count));
  }
  QualifiedMass gap;gap.Begin();gap.Stage(0,0.5,2,true,0);gap.Stage(0.5,0.25,NAN,false,32);
  gap.Commit(1,0.5,0.5);Near(gap.mass,0);Near(gap.unavailable,0.5);
  gap.Begin();gap.Stage(0,0.5,2,true,0);gap.Commit(2,1.0,0.5);
  Near(gap.mass,1);Near(gap.covered,0.5);
  std::cout<<"geometry, motion, collapsed derivatives, RK recurrence, idempotence, gaps: passed\n";
}
