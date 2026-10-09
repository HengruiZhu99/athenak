#include "helpers.hpp"
int main() {
  double identity=0,jacobian=0,reference=0,outer=0,core=0,other=0,nonphysical=0;
  double inner_identity=0,source_max=0,reference_rhs=0,cutoff_jets=0;
  int rows=0;
  for(double a:{.5,.75,1.,2.}) {
    hyp::LayerReference<double> ref(1,a,{true,.05,.95});
    hyp::LayerGaugeParameters g;g.physical_trace_lapse=true;g.preferred_source=false;
    g.scri_lapse_damping=1/a;
    for(int ri=0;ri<=1000;++ri) {
      const double r=ri/1000.;const auto p=ref.At(.36*r,-.48*r,.8*r);
      const auto pd=Reference(p);const auto u=Background(p,.01);
      const D source=hyp::ResearchInnerLapseAdvection(pd,u,g);
      source_max=std::max(source_max,std::abs(source.v));
      const D c=1-hyp::LayerCoefficients(pd.radius,u.alpha.value,g).weight;
      D alternate{};
      for(int i=0;i<3;++i)alternate-=c*(u.alpha.value*u.beta.value[i]/pd.alpha-pd.beta[i])*pd.dalpha[i];
      identity=std::max(identity,std::abs(source.v-alternate.v));
      reference=std::max(reference,std::abs(hyp::ResearchInnerLapseAdvection(pd,Lift(p.state),g).v));
      if(r>=.85)outer=std::max(outer,std::abs(source.v));
      if(r<=.05)core=std::max(core,std::abs(source.v));
      const auto base=GaugeParts(pd,u,a,true,false),candidate=GaugeParts(pd,u,a,true,true);
      identity=std::max(identity,std::abs(candidate.regular.alpha.v-base.regular.alpha.v-source.v));
      other=std::max(other,std::abs(candidate.pole.alpha.v-base.pole.alpha.v));
      for(int i=0;i<3;++i)other=std::max({other,std::abs(candidate.regular.beta[i].v-base.regular.beta[i].v),
                                             std::abs(candidate.pole.beta[i].v-base.pole.beta[i].v)});
      // At W=0, the complete regular lapse term equals the original log-relative one.
      if(r<=.45) {
        D relative{};
        for(int i=0;i<3;++i)relative+=u.beta.value[i]*(u.alpha.d[i]-u.alpha.value*pd.dalpha[i]/pd.alpha);
        inner_identity=std::max(inner_identity,std::abs(candidate.regular.alpha.v-relative.v));
      }
      auto nonphysical_g=g;nonphysical_g.physical_trace_lapse=false;
      nonphysical=std::max(nonphysical,std::abs(hyp::ResearchInnerLapseAdvection(pd,u,nonphysical_g).v));
      // Value Jacobian is independent of live spatial derivative perturbations.
      for(int column=0;column<20;++column) {
        auto seeded=u;Seed(seeded,column,J(D(0,1)));Consistent(seeded);
        const auto v=hyp::ResearchInnerLapseAdvection(pd,seeded,g);
        D expected{};
        if(column==0)for(int i=0;i<3;++i)expected-=c*u.beta.value[i]*pd.dalpha[i]/pd.alpha;
        if(column>=4&&column<=6)expected=-c*u.alpha.value*pd.dalpha[column-4]/pd.alpha;
        jacobian=std::max(jacobian,std::abs(v.d-expected.v));
      }
      if(p.omega>0)for(const auto v:Evaluate(p,Lift(p.state),a,10,1,true))reference_rhs=std::max(reference_rhs,std::abs(v.v));
      ++rows;
    }
  }
  for(double r:{.85,.8500000001,.9,1.}) {
    const auto w=hyp::SmoothCutoff(r,.45,.85);
    cutoff_jets=std::max({cutoff_jets,std::abs(1-w.value),std::abs(w.d),std::abs(w.dd),std::abs(w.ddd)});
  }
  std::cout<<std::setprecision(17)<<"{\"rows\":"<<rows<<",\"nonlinear_identity_max\":"<<identity
    <<",\"value_jacobian_max\":"<<jacobian<<",\"reference_source_max\":"<<reference
    <<",\"outer_offconstraint_source_max\":"<<outer<<",\"exact_Cauchy_core_source_max\":"<<core
    <<",\"all_shift_and_pole_difference_max\":"<<other<<",\"nonphysical_lapse_source_max\":"<<nonphysical
    <<",\"Wzero_log_relative_identity_max\":"<<inner_identity<<",\"source_max\":"<<source_max
    <<",\"reference_full20_rhs_max\":"<<reference_rhs<<",\"outer_cutoff_first_three_jets_max\":"<<cutoff_jets<<"}\n";
  return identity<1e-12&&jacobian<1e-12&&reference==0&&outer==0&&core==0&&other==0&&nonphysical==0
      &&inner_identity<1e-12&&reference_rhs<1e-9&&cutoff_jets==0?0:1;
}
