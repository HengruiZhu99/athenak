#include <iostream>
#include <stdexcept>
#include "utils/circumbinary_geometry.hpp"
#include "utils/circumbinary_accumulator.hpp"
#include "dyn_grmhd/valencia_sources.hpp"

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

int main() {
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
    // The phase-frozen geometry used by prescribed ADM is only first order in time.
    Real left=0;for (int n=0;n<count;++n) left+=(Real(n)/count)/count;
    Near(0.5-left,0.5/count);
  }
  QualifiedMass gap;gap.Begin();gap.Stage(0,0.5,2,true,0);gap.Stage(0.5,0.25,NAN,false,32);
  gap.Commit(1,0.5,0.5);Near(gap.mass,0);Near(gap.unavailable,0.5);
  gap.Begin();gap.Stage(0,0.5,2,true,0);gap.Commit(2,1.0,0.5);
  Near(gap.mass,1);Near(gap.covered,0.5);
  std::cout<<"geometry, motion, collapsed derivatives, RK recurrence, idempotence, gaps: passed\n";
}
