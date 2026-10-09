#include "inputs/audit_helpers.hpp"
// Actual first-jet singular map, using full analytic angular jets.
J Basis(const hyp::LayerPoint<double>&p,int family){if(family==1)return OmegaJ(p);if(family>=2)return Unit(p,family-1);return J(D(1));}
std::array<double,20> Formula(const Jet&u,double a){
 std::array<double,20>o{};const auto gt=hyp::Geometry(u.metric),gb=hyp::Geometry(hyp::PenroseMetric(u.metric,u.chi));
 const double da=u.alpha.value.d,c=u.chi.value.d,P=u.trace.value.d,T=u.theta.value.d,b=u.beta.value[0].d,x=c-u.metric.g[0][0].d;
 o[0]=(-P+3*da+3*b)/(a*a);o[1]=2*(P+2*T-3*da-3*b)/(3*a);
 o[2]=-2*(P+2*T)/(a*a)-3*x/(a*a*a)+10*T;
 o[3]=-2*P/(a*a)-(1/(a*a)+20)*T-3*x/(a*a*a);
 o[4]=-5*(x+2*a*(da+b))/(a*a*a);
 double H[3][3]{},d[3]{},tr=0;for(int i=0;i<3;++i){d[i]=u.lambda.value[i].d-gt.contracted[i].d;for(int j=0;j<3;++j){H[i][j]=u.metric.g[i][j].d+gb.connection[0][i][j].d;if(i==j)tr+=H[i][j];}}
 const int ti[5]={0,0,0,1,1},tj[5]={0,1,2,1,2};
 for(int k=0;k<5;++k){const int i=ti[k],j=tj[k];const double h=H[i][j]-(i==j?tr/3:0),z=.5*((i==0?d[j]:0)+(j==0?d[i]:0)-(i==j?2*d[0]/3:0));o[12+k]=2*(h-z-u.a.k[i][j].d)/(a*a);}
 for(int i=0;i<3;++i)o[17+i]=-(10-2/(a*a))*d[i]-2*(2*u.trace.d[i].d+u.theta.d[i].d)/(3*a)+4*u.a.k[i][0].d/(a*a);
 return o;
}
// E rows H0,M0(3),Z0(3),Theta0,N0,N1,Qnum0,Ny,Nz,Qy,Qz,Thetay,Thetaz,H1,Theta1,SA(5).
std::array<double,24> Conditions(const Jet&u,const hyp::LayerPoint<double>&p,double a){std::array<double,24>e{};auto c=Constraints(u,p);for(int i=0;i<8;++i)e[i]=c[i];
 const auto gt=hyp::Geometry(u.metric);const auto om=Omega(u,p);D norm{};std::array<D,3>dn{};
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){norm+=u.chi.value*gt.inverse[i][j]*D(p.domega[i]*p.domega[j]);for(int z=0;z<3;++z){D dinv{};for(int l=0;l<3;++l)for(int m=0;m<3;++m)dinv-=gt.inverse[i][l]*u.metric.dg[z][l][m]*gt.inverse[m][j];dn[z]+=(u.chi.d[z]*gt.inverse[i][j]+u.chi.value*dinv)*D(p.domega[i]*p.domega[j])+u.chi.value*gt.inverse[i][j]*D(p.omega_hessian[z][i]*p.domega[j]+p.domega[i]*p.omega_hessian[z][j]);}}
 norm-=om.normal*om.normal;for(int z=0;z<3;++z)dn[z]-=D(2)*om.normal*om.dnormal[z];
 e[8]=norm.d;e[9]=-a*dn[0].d;e[10]=u.trace.value.d-3*om.normal.d;
 for(int j=1;j<3;++j){e[10+j]=dn[j].d;e[12+j]=u.trace.d[j].d-3*om.dnormal[j].d;e[14+j]=u.theta.d[j].d;}
 const double P1=-a*u.trace.d[0].d,T1=-a*u.theta.d[0].d,x0=u.chi.value.d-u.metric.g[0][0].d,x1=-a*(u.chi.d[0].d-u.metric.dg[0][0][0].d);
 double lap=-3*u.chi.value.d/a+.5*u.chi.d[0].d/a;for(int j=0;j<3;++j)lap+=u.metric.dg[j][j][0].d/a;
 e[17]=-4*(P1+2*T1)/a+4*lap-6*(x1/(a*a)-2*x0/a);e[18]=T1;
 auto f=Formula(u,a);for(int i=0;i<5;++i)e[19+i]=f[12+i];return e;
}
void PrintD(const std::array<double,24>&e){std::cout<<'[';for(int i=0;i<24;++i)std::cout<<(i?",":"")<<e[i];std::cout<<']';}
Jet Test(const hyp::LayerPoint<double>&p,int which){auto u=Lift(p.state);const J o=OmegaJ(p);if(which==0){Seed(u,3,Linear(o));Seed(u,2,Linear(J(D(-2))*o));}
 if(which==1){Seed(u,15,Linear(J(D(1))));Seed(u,13,Linear(J(D(-1))*Unit(p,1)));Seed(u,14,Linear(Unit(p,2)));}
 if(which==2)return Witness(p);
 if(which==3)Seed(u,2,Linear(o*o));
 if(which==4){J alpha(D(p.alpha));for(int i=0;i<3;++i){alpha.d[i]=p.state.alpha.d[i];for(int j=0;j<3;++j)alpha.dd[i][j]=p.state.alpha.dd[i][j];}
 Seed(u,0,Linear(J(D(-1))*alpha*alpha*o));Seed(u,1,Linear(J(D(2./3.))*alpha*o));const double a=-3/p.k_physical;
 Seed(u,2,Linear(J(D(-2/a))*o*o));Seed(u,3,Linear(J(D(-2/a))*alpha*o));for(int i=0;i<3;++i){J x(D(i==0?p.radius:0));x.d[i]=1;Seed(u,17+i,Linear(J(D(8/(3*a)))*alpha*x));}}
 Consistent(u);return u;}
int main(){double err=0,identity=0,h1err=0;int cols=0;std::cout<<std::setprecision(17)<<"{\"maps\":[";bool first=true;
 for(double a:{.5,.75,1.,2.}){hyp::LayerReference<double>ref(1,a,{true,.05,.95});auto p=ref.At(1,0,0);std::array<std::array<double,80>,20>B{};std::array<std::array<double,80>,24>E{};
 for(int family=0;family<4;++family)for(int c=0;c<20;++c){auto u=Lift(p.state);Seed(u,c,Linear(Basis(p,family)));Consistent(u);const auto f=Parts(p,u,10,5,true);const auto formula=Formula(u,a);const auto e=Conditions(u,p,a);for(int i=0;i<20;++i){B[i][family*20+c]=f.s[i].d;err=std::max(err,std::abs(f.s[i].d-formula[i]));}for(int i=0;i<24;++i)E[i][family*20+c]=e[i];
 for(int i=0;i<3;++i)identity=std::max(identity,std::abs(e[1+i]-(a*10-2/a)*e[4+i]+u.theta.d[i].d-a*f.s[17+i].d/2));
 double hx=0;const double w[4]={4,-6,4,-1};for(int j=0;j<4;++j){auto pp=ref.At(std::sqrt(1-2*a*(j+1)*1e-4),0,0);auto uu=Lift(pp.state);Seed(uu,c,Linear(Basis(pp,family)));Consistent(uu);hx+=w[j]*(Constraints(uu,pp)[0]-e[0])/pp.omega;}h1err=std::max(h1err,std::abs(hx-e[17]));++cols;}
 if(!first)std::cout<<',';first=false;std::cout<<"{\"a\":"<<a<<",\"B\":[";for(int i=0;i<20;++i){std::cout<<(i?",":"")<<'[';for(int j=0;j<80;++j)std::cout<<(j?",":"")<<B[i][j];std::cout<<']';}std::cout<<"],\"E\":[";for(int i=0;i<24;++i){std::cout<<(i?",":"")<<'[';for(int j=0;j<80;++j)std::cout<<(j?",":"")<<E[i][j];std::cout<<']';}std::cout<<"]}";
 }
 std::cout<<"],\"tests\":[";first=true;for(double a:{.5,.75,1.,2.})for(int which=0;which<5;++which){hyp::LayerReference<double>ref(1,a,{true,.05,.95});for(double oo:{0.,.001,.0001,.00001}){auto p=ref.At(std::sqrt(1-2*a*oo),0,0);const auto u=Test(p,which),f=Parts(p,u,10,5,true);if(!first)std::cout<<',';first=false;std::cout<<"{\"a\":"<<a<<",\"which\":"<<which<<",\"Omega\":"<<p.omega<<",\"R\":";Print(f.r);std::cout<<",\"S\":";Print(f.s);std::cout<<",\"E\":";PrintD(Conditions(u,p,a));if(p.omega>0){std::array<D,20>F{};for(int i=0;i<20;++i)F[i]=f.r[i]+f.s[i]/D(p.omega);std::cout<<",\"F\":";Print(F);}std::cout<<'}';}}
 std::cout<<"],\"summary\":{\"columns\":"<<cols<<",\"formula_error\":"<<err<<",\"M_identity_error\":"<<identity<<",\"H1_formula_error\":"<<h1err<<"}}\n";
}
