#include "audit_helpers.hpp"
int main(){double fd_error[3]{},corner_error=0;int fd_rows=0;std::cout<<std::setprecision(17)<<"{\"poles\":[";bool first=true;
 for(double a:{.5,.75,1.,2.})for(double k:{5.,10.})for(double sigma:{0.,5.}){hyp::LayerReference<double>ref(1,a,{true,.05,.95});auto p=ref.At(1,0,0);
 if(!first)std::cout<<',';first=false;std::cout<<"{\"a\":"<<a<<",\"kappa\":"<<k<<",\"sigma\":"<<sigma<<",\"M\":[";
 std::array<std::array<double,20>,20>m{};for(int c=0;c<20;++c){auto u=Lift(p.state);Seed(u,c,J(D(0,1)));Consistent(u);const auto f=Parts(p,u,k,sigma);for(int r=0;r<20;++r)m[r][c]=f.s[r].d;}
 for(int e=0;e<3;++e){const double eps=std::pow(10.,-3-e);for(int c=0;c<20;++c){std::array<double,20>v[2];for(int sign=0;sign<2;++sign){auto u=Lift(p.state);Seed(u,c,J(D(sign?eps:-eps)));Consistent(u);auto f=Parts(p,u,k,sigma);for(int r=0;r<20;++r)v[sign][r]=f.s[r].v;}for(int r=0;r<20;++r)fd_error[e]=std::max(fd_error[e],std::abs((v[1][r]-v[0][r])/(2*eps)-m[r][c]));++fd_rows;}}

 for(int r=0;r<20;++r){if(r)std::cout<<',';std::cout<<'[';for(int c=0;c<20;++c)std::cout<<(c?",":"")<<m[r][c];std::cout<<']';}std::cout<<"]}";}
 std::cout<<"],\"witnesses\":[";first=true;
 for(double a:{.5,.75,1.,2.})for(double sigma:{0.,5.}){hyp::LayerReference<double>ref(1,a,{true,.05,.95});
 for(double O:{0.,.01,.005,.0025,.001,.0005,.0004,.0003,.0002,.0001,.00001}){auto p=ref.At(std::sqrt(1-2*a*O),0,0);auto u=Witness(p);auto f=Parts(p,u,10,sigma);const auto c=Constraints(u,p);
 if(!first)std::cout<<',';first=false;std::cout<<"{\"a\":"<<a<<",\"sigma\":"<<sigma<<",\"Omega\":"<<p.omega<<",\"regular\":";Print(f.r);std::cout<<",\"pole\":";Print(f.s);
 std::cout<<",\"C\":[";for(int i=0;i<8;++i)std::cout<<(i?",":"")<<c[i];std::cout<<']';
 if(p.omega>0){std::array<D,20>F{};for(int i=0;i<20;++i)F[i]=f.r[i]+f.s[i]/D(p.omega);auto o=Omega(u,p);double wdot=-(p.domega[0]*F[4].d+o.normal.v*F[0].d)/p.alpha;
 const double Ndot=(F[1].d-F[7].d)*p.domega[0]*p.domega[0]-2*o.normal.v*wdot;
 std::cout<<",\"RHS\":";Print(F);std::cout<<",\"Ndot\":"<<Ndot<<",\"Qnumdot\":"<<F[2].d-3*wdot<<",\"Npert\":"<<qnf::NullDifference(Reference(p),u).d;}
 std::cout<<'}';}}
 std::cout<<"],\"finite_Q_counterexample\":[";first=true;
 for(double a:{.5,.75,1.,2.})for(double O:{.001,.0001,.00001}){hyp::LayerReference<double>ref(1,a,{true,.05,.95});auto p=ref.At(std::sqrt(1-2*a*O),0,0);auto u=Lift(p.state);Seed(u,2,Linear(J(D(.01))*OmegaJ(p)));Consistent(u);const auto f=Parts(p,u,10,5,true);std::array<D,20>F{};for(int i=0;i<20;++i)F[i]=f.r[i]+f.s[i]/D(p.omega);auto o=Omega(u,p);const double wdot=-(p.domega[0]*F[4].d+o.normal.v*F[0].d)/p.alpha;
 if(!first)std::cout<<',';first=false;std::cout<<"{\"a\":"<<a<<",\"Omega\":"<<p.omega<<",\"Omega_Qdot\":"<<F[2].d-3*wdot<<",\"ThetaDot\":"<<F[3].d<<",\"Lambda_normal_pole\":"<<f.s[17].d<<'}';}
 std::cout<<"],\"FD\":{\"rows\":"<<fd_rows<<",\"eps\":[0.001,0.0001,0.00001],\"max_abs_error\":["<<fd_error[0]<<','<<fd_error[1]<<','<<fd_error[2]<<"]}}\n";
}
