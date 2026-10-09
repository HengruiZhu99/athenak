// Fresh finite-Omega reference Fourier matrices. No global stability claim.
#include "inputs/audit_helpers.hpp"
#include "early_feedback.hpp"
hyp::GaugeRHSParts<D> GP(const hyp::LayerPoint<double>&p,const Jet&u,double a,int form){hyp::LayerGaugeParameters g;g.scri_lapse_damping=1/a;
 if(form>=4)return earlynf::Gauge(Reference(p),u,g,{5,form-4});
 if(form<3)return qnf::Gauge(Reference(p),u,g,{.85,.95,form?5.:0.,form==2});
 g.physical_trace_lapse=true;g.preferred_source=false;auto q=hyp::InteriorLayerGauge(Reference(p),u,g);
 const D W=hyp::LayerCoefficients(D(p.radius),u.alpha.value,g).weight;
 if(W>D(0)){const auto l=hyp::Geometry(u.metric),h=hyp::Geometry(Reference(p).state.metric);D norm{},Ghat{},dG{};
 for(int i=0;i<3;++i){norm+=p.domega[i]*p.domega[i];for(int j=0;j<3;++j){const D w=p.domega[i]*p.domega[j];Ghat+=p.state.chi.value*h.inverse[i][j]*w;dG+=((u.chi.value-p.state.chi.value)*h.inverse[i][j]+u.chi.value*(l.inverse[i][j]-h.inverse[i][j]))*w;}}
 const double eta=1.5/(a*a),C=1/(3*a);for(int i=0;i<3;++i)q.pole.beta[i]-=eta*W*((u.beta.value[i]-p.beta[i])-C*p.domega[i]*dG/(Kokkos::sqrt(norm)*Ghat));}
 return q;}
std::array<D,20> F(const hyp::LayerPoint<double>&p,const Jet&u,double a,int form){hyp::Z4cRHS<D>f{};const auto o=Omega(u,p);if(!hyp::AssembleInterior(hyp::ConformalRHS(u,o,D(10)/u.alpha.value,D(0)),o.omega,f))throw std::runtime_error("geometry invalid");hyp::GaugeRHS<D>g{};if(!qnf::Assemble(GP(p,u,a,form),D(p.omega),g))throw std::runtime_error("gauge invalid");return Fields(f,g);}
M Matrix(const hyp::LayerPoint<double>&p,double a,int form,double k,const std::array<double,3>&n){M m{};
 for(int col=0;col<20;++col)for(int phase=0;phase<2;++phase){auto u=Lift(p.state);J seed(D(0,phase?0:1));for(int i=0;i<3;++i){seed.d[i]=D(0,phase?k*n[i]:0);for(int j=0;j<3;++j)seed.dd[i][j]=D(0,phase?0:-k*k*n[i]*n[j]);}Seed(u,col,seed);Consistent(u);const auto f=F(p,u,a,form);for(int row=0;row<20;++row)m[row][col]+=(phase?I:C(1))*f[row].d;}
 return m;}
void PrintM(const M&m){std::cout<<'[';for(int i=0;i<20;++i){if(i)std::cout<<',';std::cout<<'[';for(int j=0;j<20;++j){if(j)std::cout<<',';std::cout<<'['<<m[i][j].real()<<','<<m[i][j].imag()<<']';}std::cout<<']';}std::cout<<']';}
#ifndef Q_FOLLOWUP_NO_MAIN
int main(){std::cout<<std::setprecision(17)<<'[';bool first=true;
 for(double a:{.5,.75,1.,2.}){hyp::LayerReference<double>ref(1,a,{true,.05,.95});for(double r:{.45,.65,.8,.85,.9,.95,.98})for(double k:{0.,4.,16.,64.,256.})for(int dir=0;dir<2;++dir)for(int form=4;form<6;++form){const auto p=ref.At(r,0,0);const std::array<double,3>n=dir?std::array<double,3>{.36,-.48,.8}:std::array<double,3>{1,0,0};const auto m=Matrix(p,a,form,k,n);
 if(!first)std::cout<<',';first=false;std::cout<<"{\"a\":"<<a<<",\"r\":"<<r<<",\"Omega\":"<<p.omega<<",\"k\":"<<k<<",\"dir\":"<<dir<<",\"form\":"<<form<<",\"M\":";PrintM(m);std::cout<<'}';}}
 std::cout<<"]\n";
}
#endif
