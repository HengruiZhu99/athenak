#include "helpers.hpp"
int main() {
  double identity=0,jacobian=0,reference=0,outer=0,core=0,other=0,nonphysical=0;
  double original_q_identity=0,reference_rhs=0,branch_identity=0,minimum_omega=1;
  double combined_branch_jacobian=0;
  int rows=0,branch_rows=0;
  for(double a:{.5,.75,1.,2.}) {
    hyp::LayerReference<double>ref(1,a,{true,.05,.95});
    hyp::LayerGaugeParameters g;g.physical_trace_lapse=true;g.preferred_source=false;
    g.scri_lapse_damping=1/a;
    for(int ri=0;ri<=1000;++ri) {
      const double r=ri/1000.;const auto p=ref.At(.36*r,-.48*r,.8*r);
      const auto pd=Reference(p);const auto u=Background(p,.01);
      const D c=1-hyp::SmoothCutoff(pd.radius,D(g.r0),D(g.r1)).value;
      const auto base=GaugeParts(pd,u,a,true,0);
      const D trace=hyp::ResearchInnerConformalTrace(pd,u,g);
      D b{},br{};for(int i=0;i<3;++i){b+=u.beta.value[i]*pd.domega[i];br+=pd.beta[i]*pd.domega[i];}
      if(c>0){const D expected=3*c*(u.alpha.value+2*c)*(u.alpha.value*br/pd.alpha-b)/pd.omega;
        identity=std::max(identity,std::abs(trace.v-expected.v));
        if(a==.5)minimum_omega=std::min(minimum_omega,p.omega);}
      reference=std::max(reference,std::abs(hyp::ResearchInnerConformalTrace(pd,Lift(p.state),g).v));
      for(int mode:{2,3}) {
        const auto candidate=GaugeParts(pd,u,a,true,mode);
        D expected=base.regular.alpha+trace;
        if(mode==3)expected+=hyp::ResearchInnerLapseAdvection(pd,u,g);
        identity=std::max(identity,std::abs(candidate.regular.alpha.v-expected.v));
        other=std::max(other,std::abs(candidate.pole.alpha.v-base.pole.alpha.v));
        for(int i=0;i<3;++i)other=std::max({other,std::abs(candidate.regular.beta[i].v-base.regular.beta[i].v),std::abs(candidate.pole.beta[i].v-base.pole.beta[i].v)});
        if(r>=.85)outer=std::max(outer,std::abs(candidate.regular.alpha.v-base.regular.alpha.v));
        if(r<=.05)core=std::max(core,std::abs(candidate.regular.alpha.v-base.regular.alpha.v));
        if(r<=.45 && mode==3){auto qg=g;qg.physical_trace_lapse=false;
          const auto q=hyp::InteriorLayerGauge(pd,u,qg);
          original_q_identity=std::max(original_q_identity,std::abs(candidate.regular.alpha.v+candidate.pole.alpha.v/p.omega-q.regular.alpha.v-q.pole.alpha.v/p.omega));}
        if(p.omega>0)for(const auto v:Evaluate(p,Lift(p.state),a,10,mode,true))reference_rhs=std::max(reference_rhs,std::abs(v.v));
      }
      for(int column=0;column<20;++column){auto seeded=u;Seed(seeded,column,J(D(0,1)));Consistent(seeded);
        const D f=hyp::ResearchInnerConformalTrace(pd,seeded,g);D expected{};
        if(c>0&&column==0)expected=3*c*((2*u.alpha.value+2*c)*br/pd.alpha-b)/pd.omega;
        if(c>0&&column>=4&&column<=6)expected=-3*c*(u.alpha.value+2*c)*pd.domega[column-4]/pd.omega;
        jacobian=std::max(jacobian,std::abs(f.d-expected.v));}
      auto qg=g;qg.physical_trace_lapse=false;
      nonphysical=std::max(nonphysical,std::abs(hyp::ResearchInnerConformalTrace(pd,u,qg).v));
      ++rows;
    }
    // Exercise both numerically equivalent branches, including tiny lapse.
    for(double r:{.2,.4,.65,.8})for(double ratio:{1e-300,1e-12,.49,.5,.5000000001,.51,1.}){
      auto p=ref.At(.36*r,-.48*r,.8*r);auto pd=Reference(p);auto u=Lift(p.state);u.alpha.value=D(ratio*p.alpha);
      for(int i=0;i<3;++i){u.beta.value[i]=D(.01*(i+1));u.alpha.d[i]=D(ratio*p.dalpha[i]);}
      const D c=1-hyp::SmoothCutoff(pd.radius,D(g.r0),D(g.r1)).value;
      D b{},br{};for(int i=0;i<3;++i){b+=u.beta.value[i]*pd.domega[i];br+=pd.beta[i]*pd.domega[i];}
      const D expected=3*c*(u.alpha.value+2*c)*(u.alpha.value*br/pd.alpha-b)/pd.omega;
      const D f=hyp::ResearchInnerConformalTrace(pd,u,g);
      branch_identity=std::max(branch_identity,std::abs(f.v-expected.v)/(1+std::abs(expected.v)));
      for(int column:{0,4,5,6}){auto seeded=u;Seed(seeded,column,J(D(0,1)));const auto d=hyp::ResearchInnerConformalTrace(pd,seeded,g);
        const D e=column==0?3*c*((2*u.alpha.value+2*c)*br/pd.alpha-b)/pd.omega:-3*c*(u.alpha.value+2*c)*pd.domega[column-4]/pd.omega;
        jacobian=std::max(jacobian,std::abs(d.d-e.v));
        const auto combined=GaugeParts(pd,seeded,a,true,3);
        D expected_combined=e;
        const D W=1-c,nu=g.lapse_inner+(g.lapse_outer-g.lapse_inner)*W;
        if(column==0){expected_combined-=nu*(Kokkos::log(u.alpha.value/pd.alpha)+1);
          for(int i=0;i<3;++i)expected_combined-=c*u.beta.value[i]*pd.dalpha[i]/pd.alpha;}
        else {const int i=column-4;expected_combined+=u.alpha.d[i]-c*u.alpha.value*pd.dalpha[i]/pd.alpha;}
        combined_branch_jacobian=std::max(combined_branch_jacobian,
            std::abs(combined.regular.alpha.d-expected_combined.v)/(1+std::abs(expected_combined.v)));}
      ++branch_rows;
    }
  }
  hyp::LayerReference<double>tiny_ref(1,.5,{true,.05,.95});
  auto tiny_p=tiny_ref.At(.4,0.,0.);auto tiny=Lift(tiny_p.state);tiny.alpha.value=D(1e-300);
  for(int i=0;i<3;++i){tiny.beta.value[i]=D(0);tiny.alpha.d[i]=D(0);}
  hyp::LayerGaugeParameters tiny_g;tiny_g.physical_trace_lapse=true;tiny_g.preferred_source=false;
  const auto tiny_pd=Reference(tiny_p);const auto tiny_trace=hyp::ResearchInnerConformalTrace(tiny_pd,tiny,tiny_g);
  const auto tiny_parts=GaugeParts(tiny_pd,tiny,.5,true,3);
  double contraction=0;for(int i=0;i<3;++i)contraction+=tiny_p.beta[i]*tiny_p.domega[i];
  std::cout<<std::setprecision(17)<<"{\"rows\":"<<rows<<",\"crossbranch_rows\":"<<branch_rows
    <<",\"nonlinear_identity_max\":"<<identity<<",\"value_jacobian_max\":"<<jacobian
    <<",\"reference_source_max\":"<<reference<<",\"outer_source_difference_max\":"<<outer
    <<",\"core_source_difference_max\":"<<core<<",\"shift_pole_difference_max\":"<<other
    <<",\"nonphysical_source_max\":"<<nonphysical<<",\"Wzero_original_Q_identity_max\":"<<original_q_identity
    <<",\"crossbranch_identity_relative_max\":"<<branch_identity
    <<",\"combined_crossbranch_jacobian_relative_max\":"<<combined_branch_jacobian
    <<",\"reference_RHS_max\":"<<reference_rhs
    <<",\"a.5_sampled_support_Omega_min\":"<<minimum_omega
    <<",\"tiny_lapse\":{\"alpha\":1e-300,\"alpha_ref\":"<<tiny_p.alpha<<",\"Omega\":"<<tiny_p.omega
    <<",\"beta_ref_dot_dOmega\":"<<contraction<<",\"trace_addition\":"<<tiny_trace.v
    <<",\"combined_regular_alpha\":"<<tiny_parts.regular.alpha.v<<"}}\n";
  return identity<1e-12&&jacobian<1e-12&&reference==0&&outer==0&&core==0&&other==0&&nonphysical==0
      &&original_q_identity<1e-12&&branch_identity<1e-12&&combined_branch_jacobian<1e-12&&reference_rhs<1e-9?0:1;
}
