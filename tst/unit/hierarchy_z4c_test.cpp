#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstring>
#include "z4c/hierarchy_physics.hpp"
#include "driver/hierarchy_physical_maximum.hpp"
struct TestPhysics: z4c::HierarchyPhysics<3> {
      using z4c::HierarchyPhysics<3>::HierarchyPhysics;
      bool skip_projection=false;
      void Project(const subcycling::StepContext &s,int stage,const subcycling::BlockBatches &b,
                   const DvceArray5D<Real> &u) {
        z4c::HierarchyPhysics<3>::Project(s,skip_projection ? 3 : stage,b,u);
      }
    };
int main(int argc,char **argv) {
 Kokkos::initialize(argc,argv);
 {
  using Z=z4c::Z4c;
  auto flag=[&](const std::string &name) {for(int a=1;a<argc;++a) if(argv[a]==name) return true;return false;};
  std::vector<subcycling::BlockKey> leaves{{0,1,0,0},{0,0,1,0},{0,1,1,0},
    {1,0,0,0},{1,1,0,0},{1,0,1,0},{1,1,1,0}};
  if(flag("--uniform")) leaves={{0,0,0,0},{0,1,0,0},{0,0,1,0},{0,1,1,0}};
  const int roots=flag("--interior-patch") ? 4 : 2;
  if(flag("--interior-patch")) {
    leaves.clear();
    for(int x=0;x<roots;++x) for(int y=0;y<roots;++y) {
      if(x==1 && y==1) for(int a=0;a<2;++a) for(int b=0;b<2;++b)
        leaves.push_back({1,2+a,2+b,0});
      else leaves.push_back({0,x,y,0});
    }
  }
  if(flag("--three-level")) {
    bool refined=false;
    for(auto it=leaves.begin();it!=leaves.end();++it) if((*it)[0]==1) {
      const auto parent=*it;leaves.erase(it);
      for(int a=0;a<2;++a) for(int b=0;b<2;++b)
        leaves.push_back({2,2*parent[1]+a,2*parent[2]+b,0});
      refined=true;break;
    }
    if(!refined) throw std::runtime_error("three-level test needs a refined parent");
  }
  subcycling::Hierarchy tree(leaves,0,2);
  const int nx=flag("--nx32") ? 32 : (flag("--nx16") ? 16 : 8),nn=nx+9;
  z4c::Z4cGridLayout l;l.centering=z4c::Z4cGridCentering::vertex;
  l.nx1=l.nx2=nx;l.nx3=l.n3=1;l.ks=l.ke=0;l.is=l.js=4;l.ie=l.je=nx+4;l.n1=l.n2=nn;
  RegionSize domain{};domain.x1min=0;domain.x1max=roots;domain.x2min=-roots/2.;domain.x2max=roots/2.;
  domain.x3min=-.5;domain.x3max=.5;
  subcycling::HierarchyGeometry geometry;
  geometry.Initialize(tree,domain,0,roots,roots,l,{{BoundaryFlag::axis,BoundaryFlag::outflow,
    BoundaryFlag::outflow,BoundaryFlag::outflow,BoundaryFlag::periodic,BoundaryFlag::periodic}});
  Z::Options opt{};opt.chi_psi_power=-4;opt.chi_div_floor=opt.chi_min_floor=1e-12;
  opt.lapse_oplog=2;opt.lapse_harmonicf=1;opt.sss_damping_time=opt.ssl_damping_time=1;
  opt.shift_mode=z4c::Z4cShiftMode::prescribed_zero;opt.use_z4c=true;opt.extrap_order=4;
  opt.vertex_axis_correction_tolerance=1e-10;
  opt.boundary_rhs_mode=flag("--cpbc") ? z4c::Z4cBoundaryRHSMode::full_constraint_bjorhus :
    z4c::Z4cBoundaryRHSMode::sommerfeld;
  opt.telegraph_lapse=true;opt.telegraph_tau=1;opt.telegraph_kappa=1;
  opt.telegraph_damping_prescription=z4c::TelegraphDampingPrescription::max_domain_abs_K;
  std::vector<int> parity;for(int v=0;v<Z::nz4c;++v) parity.push_back(z4c::Z4cStateAxisParitySignFromPackedIndex(v));
  std::vector<std::vector<double>> results;
  for(int steps:{4,8,16,32}) {
    DvceArray5D<Real> leaf("Z4c leaves",leaves.size(),Z::nz4c,1,nn,nn);
    auto h=Kokkos::create_mirror_view(leaf);
    for(int m=0;m<leaf.extent_int(0);++m) for(int v=0;v<Z::nz4c;++v)
      for(int j=0;j<nn;++j) for(int i=0;i<nn;++i) {
        const auto key=leaves[m];const double w=std::ldexp(1.,-key[0]);
        const double x=w*(key[1]+(i-4)/static_cast<double>(nx)),z=domain.x2min+w*(key[2]+(j-4)/static_cast<double>(nx));
        double value=(v==Z::I_Z4C_CHI || v==Z::I_Z4C_GXX || v==Z::I_Z4C_GYY || v==Z::I_Z4C_GZZ || v==Z::I_Z4C_ALPHA) ? 1 : 0;
        if(v==Z::I_Z4C_ALPHA) value+=(flag("--strong-gauge") ? .1 : .001)*(flag("--polynomial-lapse") ? x*x+z*z : std::exp(-x*x-z*z));
        h(m,v,0,j,i)=value;
      }
    Kokkos::deep_copy(leaf,h);
    subcycling::VertexParentStates storage;storage.InitializeAll(tree,leaf,l);
    storage.test_restriction_margin=flag("--interior-restriction") ? 1 :
      (flag("--deep-restriction") ? 3 : 0);
    subcycling::HierarchyRK4 engine;engine.Initialize<6>(tree,l,0,roots,roots,parity,4,{{false,true,true,true}});
    if(!flag("--unreconciled-initial")) engine.ReconcileInitialState(storage);
    engine.test_corrector_passes=flag("--corrector") ? 5 :
      (flag("--corrector2") ? 2 : (flag("--corrector3") ? 3 : 1));
    engine.test_skip_restriction=flag("--no-restriction");
    TestPhysics physics(storage,geometry,l,opt,0,roots,roots,flag("--no-ko") ? 0 : .02/64,
      {{false,true,true,true}},[vary=flag("--time-dependent")](double t){return vary ? 1.+.5*t : 1.;},[](double){return 0.;},[](double){return 0.;});
    physics.skip_projection=flag("--no-projection");
    subcycling::HierarchyPhysicalMaximum global_K;
    global_K.Build(tree,Z::I_Z4C_KHAT,Z::I_Z4C_THETA);
    const subcycling::HierarchyRK4::Histories *gauge_history=nullptr;
    Real initial_K=0,largest_gauge=0;int gauge_samples=0;
    if(flag("--global-gauge")) physics.max_K=[&](double t) {
      const Real value=gauge_history ? global_K.Evaluate(*gauge_history,t) : initial_K;
      largest_gauge=std::max(largest_gauge,value);++gauge_samples;return value;
    };
    const auto gauge_pass=[&](const subcycling::HierarchyRK4::Histories &histories) {
      gauge_history=histories.empty()?nullptr:&histories;
    };
    const double dt=.04/steps;
    const unsigned ratio=flag("--synchronous-hierarchy") ? 1 :
      (flag("--three-level") && !flag("--coarse-group") ? 4 : 2);
    int minimum_passes=100,maximum_passes=0;double worst_change=0;
    for(int n=0;n<steps;++n) {
      if(flag("--rollback-failure") && n==0) {
        const auto before=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),storage.Values());
        subcycling::CorrectorControl tight;tight.maximum_passes=3;
        tight.absolute_tolerance=1e-30;tight.relative_tolerance=0;
        bool failed=false;
        try {engine.RunCorrected(n*dt,dt,storage,physics,ratio,tight);}
        catch(const subcycling::CorrectorFailure &) {failed=true;}
        const auto after=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),storage.Values());
        if(!failed || engine.LastCorrectorReport().converged ||
           std::memcmp(before.data(),after.data(),before.size()*sizeof(Real))!=0)
          throw std::runtime_error("corrector rollback did not preserve interval start");
      }
      if(flag("--adaptive") || flag("--rollback-failure") || flag("--global-gauge")) {
        subcycling::CorrectorReport report;subcycling::CorrectorControl control;
        if(flag("--extra-passes")) control.maximum_passes=12;
        if(flag("--tight-corrector")) {
          control.maximum_passes=16;control.absolute_tolerance=1e-13;control.relative_tolerance=1e-11;
        }
        try {report=engine.RunCorrected(n*dt,dt,storage,physics,ratio,control,
          flag("--global-gauge") ? std::function<void(const subcycling::HierarchyRK4::Histories &)>(gauge_pass) : nullptr,
          flag("--global-gauge") ? std::function<Real(const subcycling::HierarchyRK4::Histories &,const subcycling::HierarchyRK4::Histories &)>(
            [&](const auto &a,const auto &b){return global_K.Difference(a,b,control);}) : nullptr); }
        catch(const subcycling::CorrectorFailure &) {
          report=engine.LastCorrectorReport();
          std::cerr << "corrector failure steps=" << steps << " interval=" << n
                    << " passes=" << report.passes << " endpoint=" << report.endpoint_change
                    << " history=" << report.history_change << " feedback=" << report.feedback_change << std::endl;
          throw;
        }
        if(flag("--global-gauge")) {
          initial_K=global_K.Evaluate(engine.AcceptedHistories(),(n+1)*dt);
          gauge_history=nullptr;
        }
        if(!report.converged) throw std::runtime_error("accepted unconverged interval");
        minimum_passes=std::min(minimum_passes,report.passes);
        maximum_passes=std::max(maximum_passes,report.passes);
        worst_change=std::max(worst_change,std::max(report.feedback_change,std::max(report.endpoint_change,report.history_change)));
      } else engine.Run(n*dt,dt,storage,physics,ratio);
    }
    if(flag("--global-gauge")) {
      if(gauge_samples==0 || !(largest_gauge>0)) throw std::runtime_error("global gauge was not exercised");
      std::cout << "global gauge samples=" << gauge_samples << " max=" << largest_gauge << std::endl;
    }
    if(maximum_passes>0) std::cout << "corrector passes=" << minimum_passes << ".." << maximum_passes
                                 << " worst normalized change=" << worst_change << std::endl;
    auto out=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),storage.Values());
    std::vector<double> values;
    for(int n=0;n<out.extent_int(0);++n) if(!tree.Nodes()[n].Covered())
      for(int v=0;v<Z::nz4c;++v) for(int j=l.js;j<=l.je;++j) for(int i=l.is;i<=l.ie;++i) {
        const double value=out(n,v,0,j,i);
        if(!std::isfinite(value)) throw std::runtime_error("nonfinite hierarchy Z4c result");
        values.push_back(value);
      }
    if(const char *prefix=std::getenv("ATHENA_HIERARCHY_TEST_DUMP_PREFIX")) {
      std::ofstream output(std::string(prefix)+std::to_string(steps)+".txt");
      if(!output) throw std::runtime_error("cannot write hierarchy test fields");
      output << std::setprecision(17);
      for(double value:values) output << value << '\n';
    }
    results.push_back(values);std::cout << "completed steps=" << steps << '\n';
  }
  std::vector<double> errors;
  for(int r=0;r<3;++r) {
    double sum=0;for(std::size_t n=0;n<results[r].size();++n) sum+=std::pow(results[r][n]-results[r+1][n],2);
    double maximum=0;std::size_t location=0;
    for(std::size_t n=0;n<results[r].size();++n) {
      const double d=std::abs(results[r][n]-results[r+1][n]);
      if(d>maximum) {maximum=d;location=n;}
    }
    const int points=(nx+1)*(nx+1);
    std::cout << "max difference=" << maximum << " leaf=" << location/(Z::nz4c*points)
              << " component=" << (location/points)%Z::nz4c << " j=" << (location%(points))/(nx+1)
              << " i=" << location%(nx+1) << std::endl;
    errors.push_back(std::sqrt(sum/results[r].size()));
    std::cout << "successive rms=" << errors.back() << '\n';
  }
  bool converged=true;
  for(int r=0;r<2;++r) {
    const double ratio=errors[r]/errors[r+1];
    std::cout << "ratio=" << ratio << std::endl;
    converged &= ratio>10 && ratio<24;
  }
  if(!converged && !flag("--diagnostic")) throw std::runtime_error("coupled Z4c temporal convergence gate failed");
 }
 Kokkos::finalize();
}
