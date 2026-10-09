// Actual Cartesian C0 tensor and physical-P/spatial-norm gauge boundary jets.
// Linear dual perturbations only; no native or surrogate evolution.
#include "../live-damping-control/profile_helpers.hpp"
struct RS {std::array<D,20>r,s;};
RS Parts(const hyp::LayerPoint<double>&p,const Jet&u,double a,int profile){
 const auto o=Omega(u,p);const auto f=hyp::ConformalRHS(u,o,D(10)/u.alpha.value,
 profile?hyp::ResearchLiveKappa2Profile(u,o,D(p.radius),D(10)):D(0));
 const auto g=GaugeParts(Reference(p),u,a,true);hyp::GaugeRHS<D>gr{},gs{};
 gr.alpha=g.regular.alpha;gs.alpha=g.pole.alpha;for(int i=0;i<3;++i){gr.beta[i]=g.regular.beta[i];gs.beta[i]=g.pole.beta[i];}
 return {Fields(f.regular,gr),Fields(f.pole,gs)};
}
J OmegaJ(const hyp::LayerPoint<double>&p){J o(D(p.omega));for(int i=0;i<3;++i){o.d[i]=p.domega[i];for(int j=0;j<3;++j)o.dd[i][j]=p.omega_hessian[i][j];}return o;}
J Unit(const hyp::LayerPoint<double>&p,int i){const double r=p.radius,n[3]={1.,0.,0.};J y{D(n[i])};
 for(int d=0;d<3;++d){y.d[d]=((d==i?1.:0.)-n[d]*n[i])/r;
 for(int e=0;e<3;++e)y.dd[d][e]=(-(d==i?1.:0.)*n[e]-(e==i?1.:0.)*n[d]-(d==e?1.:0.)*n[i]+3*n[i]*n[d]*n[e])/(r*r);}return y;
}
J Linear(const J&x){J y(D(0,x.v.v));for(int i=0;i<3;++i){y.d[i]=D(0,x.d[i].v);for(int j=0;j<3;++j)y.dd[i][j]=D(0,x.dd[i][j].v);}return y;}
J Power(const hyp::LayerPoint<double>&p,int n){J q(D(1)),o=OmegaJ(p);for(int i=0;i<n;++i)q=q*o;return q;}
Jet Case(const hyp::LayerPoint<double>&p,double a,int which){Jet u=Lift(p.state);
 // which0: finite Q with every leading tensor/gauge pole cancelled by actual
 // radial tracefree A and radial Lambda, retaining their nonfree angular jets.
 // which1: P=Pref+Omega^2, no other perturbation (first scalar rates vanish).
 if(which==0){const double l=4/(30*a*a+2),ar=-2*l/3;
  Seed(u,2,Linear(Power(p,1)));
  const int ti[5]={0,0,0,1,1},tj[5]={0,1,2,1,2};
  for(int c=0;c<5;++c){J f=J(D(1.5))*Unit(p,ti[c])*Unit(p,tj[c]);if(ti[c]==tj[c])f=f-J(D(.5));Seed(u,12+c,Linear(J(D(ar))*f));}
  for(int i=0;i<3;++i)Seed(u,17+i,Linear(J(D(l))*Unit(p,i)));
 }else if(which==1)Seed(u,2,Linear(Power(p,2)));
 else {
  // Exact first actual RHS of which1, differentiated as a Cartesian field.
  // alpha_t=-alpha^2 Omega, chi_t=2alpha Omega/3,
  // P_t=-2Omega^2/a, Theta_t=-2alpha Omega/a,
  // Lambda_t^i=8alpha x^i/(3a); all other components zero.
  J alpha(D(p.alpha));for(int i=0;i<3;++i){alpha.d[i]=p.state.alpha.d[i];for(int j=0;j<3;++j)alpha.dd[i][j]=p.state.alpha.dd[i][j];}
  const J om=Power(p,1);
  Seed(u,0,Linear(J(D(-1))*alpha*alpha*om));
  Seed(u,1,Linear(J(D(2./3.))*alpha*om));
  Seed(u,2,Linear(J(D(-2/a))*om*om));
  Seed(u,3,Linear(J(D(-2/a))*alpha*om));
  for(int i=0;i<3;++i){J x(D(i==0?p.radius:0));x.d[i]=1;Seed(u,17+i,Linear(J(D(8/(3*a)))*alpha*x));}
 }
 Consistent(u);return u;
}
void Vector(const std::array<D,20>&v,bool derivative){std::cout<<'[';for(int i=0;i<20;++i){if(i)std::cout<<',';std::cout<<(derivative?v[i].d:v[i].v);}std::cout<<']';}
#ifndef SCRI_FIRSTJET_ONLY
int main(){std::cout<<std::setprecision(17)<<"{\"boundary\":[";bool first=true;
 for(double a:{.5,.75,1.,2.})for(int profile:{0,1}){const hyp::LayerReference<double>ref(1,a,{true,.05,.95});const auto p=ref.At(1.,0.,0.);
 if(!first)std::cout<<',';first=false;std::cout<<"{\"a\":"<<a<<",\"profile\":"<<profile<<",\"B\":[";
 // Column order u0, u1=partial_Omega u|scri, partial_ny u0, partial_nz u0.
 std::array<std::array<double,80>,20>B{};
 for(int family=0;family<4;++family)for(int col=0;col<20;++col){auto u=Lift(p.state);J seed(D(1));if(family==1)seed=Power(p,1);if(family>=2)seed=Unit(p,family-1);
 Seed(u,col,Linear(seed));Consistent(u);const auto rs=Parts(p,u,a,profile);for(int row=0;row<20;++row)B[row][family*20+col]=rs.s[row].d;}
 for(int row=0;row<20;++row){if(row)std::cout<<',';std::cout<<'[';for(int col=0;col<80;++col){if(col)std::cout<<',';std::cout<<B[row][col];}std::cout<<']';}std::cout<<"]}";
 }
 std::cout<<"],\"cases\":[";first=true;
 for(double a:{.5,.75,1.,2.})for(int profile:{0,1})for(int which:{0,1,2}){const hyp::LayerReference<double>ref(1,a,{true,.05,.95});
 for(double om:{0.,1e-2,5e-3,2.5e-3,1e-3,5e-4,1e-4,1e-5,1e-6}){const auto p=ref.At(std::sqrt(1-2*a*om),0.,0.);const auto u=Case(p,a,which);const auto rs=Parts(p,u,a,profile);const auto c=hyp::EvolvedConstraints(u,Omega(u,p));
 if(!first)std::cout<<',';first=false;std::cout<<"{\"a\":"<<a<<",\"profile\":"<<profile<<",\"which\":"<<which<<",\"Omega\":"<<p.omega<<",\"r\":"<<p.radius<<",\"regular\":";Vector(rs.r,true);std::cout<<",\"pole\":";Vector(rs.s,true);
 if(p.omega>0){std::array<D,20>f{};for(int i=0;i<20;++i)f[i]=rs.r[i]+rs.s[i]/D(p.omega);const double w=Omega(u,p).normal.v,alpha=u.alpha.value.v;
 const double wdot=-(p.domega[0]*f[4].d+w*f[0].d)/alpha;
 const double Qdot=(f[2].d-3*wdot)/p.omega;
 const double Ndot=(f[1].d-f[7].d)*p.domega[0]*p.domega[0]-2*w*wdot;
 std::cout<<",\"RHS\":";Vector(f,true);std::cout<<",\"Qdot\":"<<Qdot<<",\"Nrawdot\":"<<Ndot;
 }
 std::cout<<",\"H\":"<<c.hamiltonian.d<<",\"Mrad\":"<<c.momentum[0].d<<",\"Zrad\":"<<c.z4.z_covector[0].d<<",\"Theta\":"<<u.theta.value.d<<",\"Q\":"<<(which==0?1.:p.omega)<<'}';
 }}std::cout<<"]}\n";
}
#endif
