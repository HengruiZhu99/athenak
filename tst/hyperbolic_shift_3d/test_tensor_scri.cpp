#include <cmath>
#include <iostream>
#include "coupled_diagnostics.hpp"
#include "z4c/hyperboloidal/spherical_tensor.hpp"
namespace h=z4c::hyperboloidal;
int main(int argc,char**argv){
  Kokkos::initialize(argc,argv);int result=0;
  try{
    // Unconstrained nonspherical jets: test the complete combined pole, not
    // just values that vanish by the boundary constraints.
    for(int seed=1;seed<=12;++seed){
      const Real x[3]={.36,.48,.8};h::Z4cJet<Real> u{};
      u.chi.value=.8;u.alpha.value=1.03;u.trace.value=-2.98;u.theta.value=.002;
      for(int i=0;i<3;++i){u.beta.value[i]=-x[i]+.003*(i+1);u.lambda.value[i]=.002*(i+1);
        u.chi.d[i]=.003*(seed+i);u.alpha.d[i]=x[i]+.002*seed;
        u.trace.d[i]=.004*(i+1);u.theta.d[i]=.001*(i+1);
        for(int j=0;j<3;++j){u.metric.g[i][j]=(i==j?1.:.01);u.a.k[i][j]=.001*(i+j+1);
          u.beta.d[i][j]=(i==j?-1.:0)+.001*seed;u.chi.dd[i][j]=.001*(i+j+1);
          u.alpha.dd[i][j]=(i==j?1.:0);u.trace.dd[i][j]=.001;u.theta.dd[i][j]=.0001;
          for(int a=0;a<3;++a){u.metric.dg[a][i][j]=.001*(a+i+j+seed);
            u.a.dk[a][i][j]=.0002*(a+i+j+1);
            for(int b=0;b<3;++b)u.metric.ddg[a][b][i][j]=.0001*(a+b+i+j+1);}}}
      auto o=h::HyperbolicComposite::Omega(x,u,true);
      auto parts=h::ConformalRHS(u,o,4.,-.99);h::HyperbolicDamping(x,u,o,parts);
      Real beta_ref[3]={-x[0],-x[1],-x[2]};
      auto pole=h::TensorPoleSources(u,o,Real(1),beta_ref,4.,-.99,1.,1.,true,true);
      Real a[31]{},b[31]{};h::HyperbolicComposite::StoreGeometry(parts.pole,a);
      h::HyperbolicComposite::StoreGeometry(pole.geometry,b);
      for(int f=0;f<18;++f)if(std::abs(a[f]-b[f])>2e-13)
        throw std::runtime_error("scri tensor pole differs from native geometric RHS");
      h::VectorJet<Real>B{};Real D[3]{};auto gauge=h::HyperbolicGauge(x,u,B,D);
      if(std::abs(gauge.lapse_pole-pole.lapse)>2e-13)
        throw std::runtime_error("scri lapse pole differs from native gauge");
      const auto dual=h::TensorPoleNormalJet(u,o,x,x,4.,-.99,1.,1.,true,true);
      Real derivatives[31]{};
      h::Z4cRHS<Real> dr{};dr.chi=dual.geometry.chi.derivative;dr.trace=dual.geometry.trace.derivative;
      dr.theta=dual.geometry.theta.derivative;
      for(int i=0;i<3;++i){dr.lambda[i]=dual.geometry.lambda[i].derivative;
        for(int j=0;j<3;++j)dr.a[i][j]=dual.geometry.a[i][j].derivative;}
      h::HyperbolicComposite::StoreGeometry(dr,derivatives);
      Real sampled[2][31]{},lapse[2]{};const Real eps=1e-5;
      for(int s=0;s<2;++s){const Real t=(2*s-1)*eps;auto v=u;Real y[3];
        auto scalar=[&](auto&v,const auto&original){
          for(int a=0;a<3;++a){v.value+=t*x[a]*original.d[a];
            for(int i=0;i<3;++i)v.d[i]+=t*x[a]*original.dd[a][i];}};
        scalar(v.chi,u.chi);scalar(v.alpha,u.alpha);scalar(v.trace,u.trace);scalar(v.theta,u.theta);
        for(int i=0;i<3;++i){y[i]=(1+t)*x[i];
          for(int a=0;a<3;++a){v.beta.value[i]+=t*x[a]*u.beta.d[a][i];v.lambda.value[i]+=t*x[a]*u.lambda.d[a][i];
            for(int j=0;j<3;++j){v.metric.g[i][j]+=t*x[a]*u.metric.dg[a][i][j];
              v.a.k[i][j]+=t*x[a]*u.a.dk[a][i][j];
              for(int b=0;b<3;++b)v.metric.dg[b][i][j]+=t*x[a]*u.metric.ddg[a][b][i][j];}}}
        auto om=h::HyperbolicComposite::Omega(y,v);Real ref[3]={-y[0],-y[1],-y[2]};
        const auto pp=h::TensorPoleSources(v,om,(1+h::HyperbolicComposite::Radius(y)*h::HyperbolicComposite::Radius(y))/2,ref,4.,-.99,1.,1.,true,true);
        h::HyperbolicComposite::StoreGeometry(pp.geometry,sampled[s]);lapse[s]=pp.lapse;
      }
      for(int f=0;f<18;++f)if(std::abs((sampled[1][f]-sampled[0][f])/(2*eps)-derivatives[f])>2e-8)
        throw std::runtime_error("scri normal AD derivative disagrees with finite difference");
      if(std::abs((lapse[1]-lapse[0])/(2*eps)-dual.lapse.derivative)>2e-8)
        throw std::runtime_error("scri lapse AD derivative disagrees with finite difference");
    }
    for(Real r:{.03,.2,.6,.98}){
      Real q[8]={-.2,.1,.03,.02,.001,.003,-.01,.02};
      Real d[8]={.1,.03,.02,.01,.002,.01,.02,.03};
      Real dd[8]={.05,.02,.01,.01,.002,.01,.01,.02};
      const h::CMCReference<Real> reference{1,1};const auto u=h::SphericalJet(r,q,d,dd,reference);
      Real x[3]={r,0,0};const auto o=h::HyperbolicComposite::Omega(x,u);
      const auto sphere=coupled_diagnostics::Sphere(x,u,o);const auto expected=h::SphereDiagnostics(r,u,o);
      if(std::abs(sphere[0]-expected.expansion_out)>2e-12||std::abs(sphere[1]-expected.areal_radius)>2e-12)
        throw std::runtime_error("general sphere diagnostics fail spherical reduction");
    }
    h::HyperbolicComposite e(cubed_sphere::CompositeMesh(.125,{.45,.75,1},1,6,3),false);
    for(size_t p=0;p<e.scri.rows.size();++p){const auto*x=e.scri.rows[p].geometry.x;
      e.scri.values[p*e.nf]=1+.002*x[0]*x[1];
      e.scri.values[p*e.nf+2]=.001*x[2];e.scri.values[p*e.nf+15]=.003;
    }
    const auto before=e.scri.values;e.BoundaryValues();
    for(size_t p=0;p<e.scri.rows.size();++p)for(int f=0;f<18;++f)
      if(e.scri.values[p*e.nf+f]!=before[p*e.nf+f])throw std::runtime_error("boundary value closure resets geometry");
    auto padded=e.Padded(e.state);auto rates=e.BoundaryRates(padded,.0001);padded=e.Padded(e.state);
    Real ghost_error=0;
    for(size_t p=0;p<e.scri.rows.size();++p)for(int f:{0,2,15,22,23,24,25,26,27})
      ghost_error=std::max(ghost_error,std::abs(e.scri.ReconstructFromGhost(p,padded,f)-e.scri.values[p*e.nf+f]));
    Real ghost_null=0;
    for(const auto&row:e.grid.halos)if(row.boundary_weight!=0){Real trace[31]{};
      for(int f:{0,1,2,3,4,5,6,22,23,24}){const int base=f*e.grid.count;
        const Real anchor=padded[base+row.entries[0].point];Real lifted=padded[base+row.target]-anchor;
        for(const auto&term:row.entries)lifted-=term.weight*(padded[base+term.point]-anchor);
        trace[f]=anchor+lifted/row.boundary_weight;}
      ghost_null=std::max(ghost_null,e.NullResidual(row.boundary.data(),trace));
    }
    if(ghost_null>3e-12)throw std::runtime_error("angular ghost-lift null relation");
    if(e.null_max>3e-14||e.compatibility_max>3e-14||e.interpolated_null_max>3e-14||ghost_error>3e-12)
      throw std::runtime_error("live nonspherical boundary/ghost closure");
    h::HyperbolicComposite::V input(e.state.size()),diss(input.size(),0);
    for(size_t i=0;i<input.size();++i)input[i]=std::sin(.137*i);
    e.Dissipate(input,diss);Real energy=0;
    for(size_t p=0;p<e.grid.active.size();++p)for(int f=0;f<e.nf;++f)
      energy+=e.grid.volumes[p]*input[p*e.nf+f]*diss[p*e.nf+f];
    if(!(energy<0))throw std::runtime_error("KO8 increases weighted energy");
    h::HyperbolicComposite flat(cubed_sphere::CompositeMesh(.125,{.45,.75,1},1,6,3),false);
    h::HyperbolicComposite::V edge1,edge2;
    // A nonspherical smooth perturbation exercises nonzero transport and sources.
    for(size_t p=0;p<flat.grid.active.size();++p){const auto*x=flat.grid.geometry[p].x;
      flat.state[p*flat.nf+18]=.0001*x[0]*x[1]*std::pow(1-flat.Radius(x),4);}
    auto serial=flat.RHS(flat.state,.0001,edge1);flat.workers=2;auto threaded=flat.RHS(flat.state,.0001,edge2);
    if(serial!=threaded||edge1!=edge2)throw std::runtime_error("threaded RHS differs from serial");
    for(size_t p=0;p<flat.grid.active.size();++p)flat.state[p*flat.nf+18]=0;
    // Restore all initial traces after the audit's materialization of dependent D.
    std::fill(flat.scri.values.begin(),flat.scri.values.end(),0);
    for(size_t p=0;p<flat.scri.rows.size();++p)flat.scri.values[p*flat.nf]=1;
    flat.Step(.0001);Real equilibrium_error=0;
    for(size_t p=0;p<flat.grid.active.size();++p)for(int f=0;f<flat.nf;++f)
      equilibrium_error=std::max(equilibrium_error,std::abs(flat.state[p*flat.nf+f]-(f==0?1.:0.)));
    if(equilibrium_error>1e-11)throw std::runtime_error("coupled RK4 fails Minkowski equilibrium");
    const auto accepted=flat.state,accepted_edge=flat.scri.values,accepted_comp=flat.compensation;
    bool failed=false;try{flat.Step(.1);}catch(const std::exception&){failed=true;}
    if(!failed||flat.failed_state.empty()||flat.failed_stage!=1||flat.state!=accepted||flat.scri.values!=accepted_edge||flat.compensation!=accepted_comp)
      throw std::runtime_error("failure did not preserve accepted state and rejected stage");
    std::cout<<"coupled Minkowski step error="<<equilibrium_error<<"\n";
    std::cout<<"live closure null="<<e.null_max<<" compatibility="<<e.compatibility_max<<" ghost="<<ghost_error<<" interpolated_null="<<e.interpolated_null_max<<" ghost_lift_null="<<ghost_null<<" KO_energy="<<energy<<"\n";
    std::cout<<"12 nonspherical pole and normal-derivative checks passed\n";
  }catch(const std::exception&e){std::cerr<<e.what()<<'\n';result=1;}
  Kokkos::finalize();return result;
}
