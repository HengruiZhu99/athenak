#define Point FrozenC0Point
#include "../discrete-bianchi/immutable-discrete-bulk-20261009/dual_helpers.hpp"
#undef Point
#include "bulk_c1_additions.hpp"
double Max(const hyp::C1Parts<D>&q) {
 double m=0;for(const auto*r:{&q.regular,&q.pole,&q.double_pole}){
  for(D v:{r->chi,r->trace,r->theta})m=std::max(m,std::max(std::abs(v.v),std::abs(v.d)));
  for(int i=0;i<3;++i){m=std::max(m,std::max(std::abs(r->lambda[i].v),std::abs(r->lambda[i].d)));
   for(int j=0;j<3;++j)for(D v:{r->metric[i][j],r->a[i][j]})m=std::max(m,std::max(std::abs(v.v),std::abs(v.d)));}
 }return m;
}
int main(){
 double einstein=0,outer=0,flatjet=0,boundmargin=1e99,reference=0;int rows=0;
 for(double a:{.5,.75,1.,2.}){
  hyp::LayerReference<double> ref(1.,a,{true,.05,.95});
  for(int ir=0;ir<=1000;++ir){const double r=ir/1000.;const auto p=ref.At(r,0,0);auto u=Lift(p.state);
   const auto w=hyp::SmoothCutoff(r,.45,.85);const double c=1-w.value;
   if(c>0)boundmargin=std::min(boundmargin,p.omega-std::min(1.,(1-.85*.85)/(2*a)));
   if(r>=.85)flatjet=std::max(flatjet,std::max({std::abs(c),std::abs(w.d),std::abs(w.dd),std::abs(w.ddd)}));
   reference=std::max(reference,Max(hyp::BulkC1Additions(u,Omega(u,p),D(r))));
   for(int col=0;col<20;++col){J seed(D(.01*std::sin(col+1),std::cos(col+1)));
    for(int i=0;i<3;++i){seed.d[i]=D(.01*std::cos((i+1.)*(col+1)),.1);
     for(int j=0;j<3;++j)seed.dd[i][j]=D(.005*std::sin((i+j+1.)*(col+1)),.2);}
    Seed(u,col,seed);}
   Consistent(u);
   if(r>=.85)outer=std::max(outer,Max(hyp::BulkC1Additions(u,Omega(u,p),D(r))));
   u.theta={};const auto gt=hyp::Geometry(u.metric);
   for(int i=0;i<3;++i){u.lambda.value[i]=gt.contracted[i];
    for(int j=0;j<3;++j)u.lambda.d[j][i]=gt.dcontracted[j][i];}
   einstein=std::max(einstein,Max(hyp::BulkC1Additions(u,Omega(u,p),D(r))));++rows;
  }
 }
 std::cout<<std::setprecision(17)<<"{\"rows\":"<<rows<<",\"einstein_max\":"<<einstein
  <<",\"outer_offconstraint_max\":"<<outer<<",\"outer_cutoff_jets_max\":"<<flatjet
  <<",\"support_Omega_bound_margin\":"<<boundmargin<<",\"reference_addition_max\":"<<reference<<"}\n";
 return einstein==0&&outer==0&&flatjet==0&&boundmargin>=0&&reference<1e-12?0:1;
}
