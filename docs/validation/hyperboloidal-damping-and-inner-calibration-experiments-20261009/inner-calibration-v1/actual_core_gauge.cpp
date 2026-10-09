// Pointwise actual core gauge check only: no evolution or BH RHS subtraction.
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include "z4c/hyperboloidal/layer_gauge.hpp"
namespace hyp=z4c::hyperboloidal;
int main(int argc,char**argv){if(argc!=10)return 2;double v[9];for(int i=0;i<9;++i)v[i]=std::stod(argv[i+1]);
 const double r=v[0],alpha=v[1],da=v[2],K=v[3],beta=v[4],db=v[5],chi=v[6],dc=v[7],eta=v[8];
 hyp::LayerReference<double> ref(1.,.5,{true,.05,.95});const auto p=ref.At(r,0.,0.);
 if(p.omega!=1||p.alpha!=1||p.k_physical!=0)return 3;
 auto u=p.state;u.alpha.value=alpha;u.alpha.d[0]=da;u.chi.value=chi;u.chi.d[0]=dc;
 u.trace.value=K;u.theta.value=0;u.beta.value[0]=beta;u.beta.d[0][0]=db;
 u.beta.d[1][1]=u.beta.d[2][2]=beta/r;
 hyp::LayerGaugeParameters g;g.preferred_source=false;g.physical_trace_lapse=true;
 g.shift_inner=eta;g.shift_outer=eta+1;g.Validate(1.);
 const auto parts=hyp::InteriorLayerGauge(p,u,g);hyp::GaugeRHS<double>rhs;
 if(!hyp::AssembleGaugeInterior(parts,p.omega,rhs))return 4;
 for(int i=0;i<3;++i)rhs.beta[i]+=parts.pole.beta[i]/p.omega;
 const double slicing=beta*da-alpha*(alpha+2)*K;
 const double driver=beta*(db-eta);
 std::cout<<std::setprecision(17)<<"{\"alpha_rhs\":"<<rhs.alpha<<",\"beta_rhs\":"<<rhs.beta[0]<<",\"slicing_expression\":"<<slicing<<",\"driver_expression\":"<<driver<<",\"weight\":"<<hyp::LayerCoefficients(r,alpha,g).weight<<",\"Minkowski_reference_unchanged\":true}\n";
}
