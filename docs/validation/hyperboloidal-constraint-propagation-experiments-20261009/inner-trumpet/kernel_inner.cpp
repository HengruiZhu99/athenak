#include <algorithm>
#include <array>
#include <iomanip>
#include <iostream>
#include "local_trumpet.hpp"
namespace hyp = z4c::hyperboloidal;
void Require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
int main(int argc, char** argv) {
 Kokkos::initialize(argc, argv);
 try {
  std::cout << std::setprecision(17);
  double H = 0, Mom = 0, static_error = 0, lapse_error = 0, balance_error = 0;
  double determinant_error = 0, jet1 = 0, jet2 = 0, damp_error = 0;
  double mass_error = 0, areal_error = 0;
  int rows = 0;
  hyp::LayerReference<double> ref(1, .5, {true, .05, .95});
  ref.Validate();
  for (const auto config : std::array<std::array<double, 3>, 5>{
      {{1.4, .75, .5}, {1.4, 1., .5}, {1.4, 1.5, .5},
       {1.4, 2. / 3, .5}, {1.35, 1.25, .5}}}) {
   inner_audit::LocalTrumpet t{config[2], config[0] * config[2], config[1]};
   hyp::LayerGaugeParameters g;
   g.preferred_source = false; g.physical_trace_lapse = true; g.Validate(1);
   auto damped = g; damped.shift_inner = t.V(); damped.Validate(1);
   for (double rf : {.04, .02, .01, .005, .002, .001, .0005, .0002, .0001}) {
    const double r = rf * t.mass;
    for (const auto n : std::array<std::array<double, 3>, 2>{{{1, 0, 0}, {.36, -.48, .8}}}) {
     auto dot = [&](const double v[3]) { double out = 0; for(int i=0;i<3;++i)out += n[i]*v[i]; return out; };
     const auto u = t.At(r*n[0],r*n[1],r*n[2]);
     const auto p = ref.At(r*n[0],r*n[1],r*n[2]);
     Require(p.omega == 1 && p.alpha == 1 && p.k_physical == 0, "not exact reference core");
     hyp::OmegaJet<double> o{}; o.omega = 1;
     const auto con = hyp::EvolvedConstraints(u, o);
     const auto parts = hyp::ConformalRHS(u, o, 10., 0.);
     hyp::Z4cRHS<double> f{}; hyp::GaugeRHS<double> ga{}, gd{};
     Require(con.valid && parts.valid && hyp::AssembleInterior(parts, 1., f) &&
       hyp::AssembleGaugeInterior(hyp::InteriorLayerGauge(p,u,g),1.,ga) &&
       hyp::AssembleGaugeInterior(hyp::InteriorLayerGauge(p,u,damped),1.,gd), "full kernel invalid");
     H = std::max(H, std::abs(con.hamiltonian)*t.mass*t.mass);
     Mom = std::max(Mom, std::sqrt(con.momentum_physical_norm2)*t.mass*t.mass);
     determinant_error = std::max(determinant_error,std::abs(hyp::Geometry(u.metric).determinant-1));
     double gr=0,ar=0,trg=0,tra=0,dgr=0,dtrg=0;
     for(int i=0;i<3;++i){trg+=u.metric.g[i][i];tra+=u.a.k[i][i];
      for(int d=0;d<3;++d)dtrg+=n[d]*u.metric.dg[d][i][i];
      for(int j=0;j<3;++j){gr+=n[i]*n[j]*u.metric.g[i][j];ar+=n[i]*n[j]*u.a.k[i][j];
       for(int d=0;d<3;++d)dgr+=n[d]*n[i]*n[j]*u.metric.dg[d][i][j];}}
     const double gt=(trg-gr)/2,gp=(dtrg-dgr)/2;
     const double areal=r*std::sqrt(gt/u.chi.value),kt=(tra-ar)/(2*gt)+u.trace.value/3;
     const double Rp=areal/r+areal/2*(gp/gt-dot(u.chi.d)/u.chi.value);
     const double mMS=areal/2*(1-u.chi.value/gr*Rp*Rp+areal*areal*kt*kt);
     mass_error=std::max(mass_error,std::abs(mMS/t.mass-1));
     areal_error=std::max(areal_error,std::abs(areal-(t.radius0+t.epsilon*t.mass*std::pow(r/t.mass,t.exponent)))/t.mass);
     double serr = std::max({std::abs(f.trace)*t.mass*t.mass,
       std::abs(f.theta)*t.mass*t.mass,std::abs(f.chi)*t.mass/(1+u.chi.value)});
     for (int i=0;i<3;++i) {
      // Lambda is a connection: use r to normalize its 1/r derivative scale.
      serr = std::max(serr,std::abs(f.lambda[i])*r*t.mass);
      for (int j=0;j<3;++j)serr=std::max({serr,
        std::abs(f.metric[i][j])*t.mass/(1+std::abs(u.metric.g[i][j])),
        std::abs(f.a[i][j])*t.mass*t.mass/(1+std::abs(u.a.k[i][j])*t.mass)});
     }
     static_error=std::max(static_error,serr);
     lapse_error=std::max(lapse_error,std::abs(ga.alpha)*t.mass/u.alpha.value);
     const double beta=dot(u.beta.value), lam=dot(u.lambda.value);
     double bp=0;for(int i=0;i<3;++i)for(int j=0;j<3;++j)bp+=n[i]*n[j]*u.beta.d[j][i];
     const double driver=.375*u.alpha.value*u.alpha.value*u.chi.value*lam;
     balance_error=std::max(balance_error,std::abs(dot(ga.beta)-beta*bp-driver)*t.mass*t.mass/r);
     damp_error=std::max(damp_error,std::abs(dot(gd.beta)-dot(ga.beta)+t.V()*beta)*t.mass*t.mass/r);
     std::cout << "{\"M\":"<<t.mass<<",\"R0_over_M\":"<<config[0]<<",\"p\":"<<t.exponent
       <<",\"r_over_M\":"<<rf<<",\"oblique\":"<<(n[0]!=1)<<",\"alpha\":"<<u.alpha.value
       <<",\"K_M\":"<<u.trace.value*t.mass<<",\"v_M\":"<<t.V()*t.mass
       <<",\"beta_over_r_M\":"<<beta/r*t.mass<<",\"rLambda\":"<<r*lam
       <<",\"lambda_limit\":"<<2*(std::pow(t.Zeta(),2./3)-std::pow(t.Zeta(),-4./3))
       <<",\"driver_over_r_M2\":"<<driver/r*t.mass*t.mass
       <<",\"betaDot_over_r_M2\":"<<dot(ga.beta)/r*t.mass*t.mass
       <<",\"betaDot_limit_M2\":"<<t.V()*t.V()*t.mass*t.mass
       <<",\"dampedBetaDot_over_r_M2\":"<<dot(gd.beta)/r*t.mass*t.mass
       <<",\"alphaDot_over_alpha_M\":"<<ga.alpha/u.alpha.value*t.mass<<"}\n";
     ++rows;
    }
   }
   // Independent off-axis value finite differences of all consumed jets.
   const double r=.01*t.mass, x[3]={.36*r,-.48*r,.8*r};
   const auto u=t.At(x[0],x[1],x[2]); const double h=1e-5*r;
   for(int d=0;d<3;++d) {
    std::array<double,3> xp{x[0],x[1],x[2]},xm=xp;xp[d]+=h;xm[d]-=h;
    const auto up=t.At(xp[0],xp[1],xp[2]),um=t.At(xm[0],xm[1],xm[2]);
    jet1=std::max({jet1,std::abs((up.alpha.value-um.alpha.value)/(2*h)-u.alpha.d[d])*r/u.alpha.value,
      std::abs((up.chi.value-um.chi.value)/(2*h)-u.chi.d[d])*r/u.chi.value,
      std::abs((up.trace.value-um.trace.value)/(2*h)-u.trace.d[d])*r*t.mass});
    for(int i=0;i<3;++i) {
     jet1=std::max({jet1,std::abs((up.beta.value[i]-um.beta.value[i])/(2*h)-u.beta.d[d][i])*t.mass,
       std::abs((up.lambda.value[i]-um.lambda.value[i])/(2*h)-u.lambda.d[d][i])*r*r});
     for(int j=0;j<3;++j) jet1=std::max({jet1,
       std::abs((up.metric.g[i][j]-um.metric.g[i][j])/(2*h)-u.metric.dg[d][i][j])*r,
       std::abs((up.a.k[i][j]-um.a.k[i][j])/(2*h)-u.a.dk[d][i][j])*r*t.mass});
    }
    for(int e=0;e<3;++e) {
     std::array<double,3> pp{x[0],x[1],x[2]},pm=pp,mp=pp,mm=pp;
     pp[d]+=h;pp[e]+=h;pm[d]+=h;pm[e]-=h;mp[d]-=h;mp[e]+=h;mm[d]-=h;mm[e]-=h;
     const auto upp=t.At(pp[0],pp[1],pp[2]),upm=t.At(pm[0],pm[1],pm[2]);
     const auto ump=t.At(mp[0],mp[1],mp[2]),umm=t.At(mm[0],mm[1],mm[2]);
     jet2=std::max({jet2,
       std::abs((upp.alpha.value-upm.alpha.value-ump.alpha.value+umm.alpha.value)/(4*h*h)-u.alpha.dd[d][e])*r*r/u.alpha.value,
       std::abs((upp.chi.value-upm.chi.value-ump.chi.value+umm.chi.value)/(4*h*h)-u.chi.dd[d][e])*r*r/u.chi.value});
     for(int i=0;i<3;++i) {
      jet2=std::max(jet2,std::abs((upp.beta.value[i]-upm.beta.value[i]-ump.beta.value[i]+umm.beta.value[i])/(4*h*h)-u.beta.dd[d][e][i])*r*t.mass);
      for(int j=0;j<3;++j)jet2=std::max(jet2,std::abs((upp.metric.g[i][j]-upm.metric.g[i][j]-ump.metric.g[i][j]+umm.metric.g[i][j])/(4*h*h)-u.metric.ddg[d][e][i][j])*r*r);
     }
    }
   }
  }
  Require(H<1e-9 && Mom<1e-9 && static_error<1e-8 && lapse_error<1e-9 &&
    balance_error<1e-12 && determinant_error<1e-12 && damp_error<1e-12 && jet1<1e-6 && jet2<1e-3 && mass_error<1e-12 && areal_error<1e-12,
    "local trumpet constraints/geometry/gauge/consumed-jet gate failed");
  std::cerr<<std::setprecision(17)<<"PASS rows="<<rows<<" H_M2="<<H<<" M_M2="<<Mom
    <<" staticScaled="<<static_error<<" alphaDot_over_alpha_M="<<lapse_error
    <<" shiftBalance="<<balance_error<<" dampingIdentity="<<damp_error
    <<" detError="<<determinant_error<<" firstJets="<<jet1<<" secondJets="<<jet2
    <<" relativeMSMass="<<mass_error<<" arealRadiusOverM="<<areal_error<<"\n";
 } catch(std::exception const& e) {std::cerr<<e.what()<<"\n";Kokkos::finalize();return 1;}
 Kokkos::finalize(); return 0;
}
