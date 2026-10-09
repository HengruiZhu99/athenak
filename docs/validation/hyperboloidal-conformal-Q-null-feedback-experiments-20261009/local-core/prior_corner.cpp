#include "audit_helpers.hpp"
#include "nonlinear_values.hpp"
#include "inputs/prior_physical_null_feedback.hpp"
int main(){std::cout<<std::setprecision(17)<<'[';bool first=true;
 for(double a:{.5,.75,1.,2.})for(double xi:{1.5,1./a}){hyp::LayerReference<double>ref(1,a,{true,.05,.95});hyp::LayerGaugeParameters g;g.physical_trace_lapse=true;g.preferred_source=false;g.scri_lapse_damping=xi;
 for(double O:{.004,.003,.002,.001}){auto p=ref.At(std::sqrt(1-2*a*O),0,0);double fd[4]{};const double e=1e-3;
 for(int sign=0;sign<2;++sign){const double eps=sign?e:-e;auto u=Lift(p.state);Seed(u,0,J(D(eps))*OmegaJ(p));for(int i=0;i<3;++i)Seed(u,4+i,J(D(-eps))*OmegaJ(p)*Unit(p,i));Consistent(u);const auto ud=Values(u);const auto f=research::Assemble(research::Gauge(p,ud,g),p.omega);fd[0]+=(sign?1:-1)*f.alpha/(2*e);fd[1]+=(sign?1:-1)*f.beta[0]/(2*e);}
 // Geometric RHS is unchanged by either gauge and is evaluated by an exact dual.
 auto u=Witness(p);const auto parts=Parts(p,u,10,5);const double Pdot=parts.r[2].d+parts.s[2].d/p.omega;
 const auto o=Omega(u,p);const double wdot=-(p.domega[0]*fd[1]+o.normal.v*fd[0])/p.alpha;
 const double Xdot=(parts.r[1].d+parts.s[1].d/p.omega)-(parts.r[7].d+parts.s[7].d/p.omega);
 const double Ndot=Xdot*p.domega[0]*p.domega[0]-2*o.normal.v*wdot,Qdot=Pdot-3*wdot;
 if(!first)std::cout<<',';first=false;std::cout<<"{\"a\":"<<a<<",\"xi\":"<<xi<<",\"Omega\":"<<p.omega<<",\"alphaDot\":"<<fd[0]<<",\"betaDot\":"<<fd[1]<<",\"PDot\":"<<Pdot<<",\"Ndot\":"<<Ndot<<",\"QnumeratorDot\":"<<Qdot<<'}';
 }}std::cout<<"]\n";
}
