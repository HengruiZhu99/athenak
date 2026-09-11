#include <cmath>
#include <iostream>
#include "z4c/hierarchy_physics.hpp"
int main(int argc,char **argv) {
 Kokkos::initialize(argc,argv);
 {
  using Z=z4c::Z4c;
  std::vector<subcycling::BlockKey> leaves{{0,1,0,0},{0,0,1,0},{0,1,1,0},
    {1,0,0,0},{1,1,0,0},{1,0,1,0},{1,1,1,0}};
  if(argc>1 && std::string(argv[1])=="--uniform") leaves={{0,0,0,0},{0,1,0,0},{0,0,1,0},{0,1,1,0}};
  subcycling::Hierarchy tree(leaves,0,2);
  z4c::Z4cGridLayout l;l.centering=z4c::Z4cGridCentering::vertex;
  l.nx1=l.nx2=8;l.nx3=l.n3=1;l.ks=l.ke=0;l.is=l.js=4;l.ie=l.je=12;l.n1=l.n2=17;
  RegionSize domain{};domain.x1min=0;domain.x1max=2;domain.x2min=-1;domain.x2max=1;
  domain.x3min=-.5;domain.x3max=.5;
  subcycling::HierarchyGeometry geometry;
  geometry.Initialize(tree,domain,0,2,2,l,{{BoundaryFlag::axis,BoundaryFlag::outflow,
    BoundaryFlag::outflow,BoundaryFlag::outflow,BoundaryFlag::periodic,BoundaryFlag::periodic}});
  Z::Options opt{};opt.chi_psi_power=-4;opt.chi_div_floor=opt.chi_min_floor=1e-12;
  opt.lapse_oplog=2;opt.lapse_harmonicf=1;opt.sss_damping_time=opt.ssl_damping_time=1;
  opt.shift_mode=z4c::Z4cShiftMode::prescribed_zero;opt.use_z4c=true;opt.extrap_order=4;
  opt.vertex_axis_correction_tolerance=1e-10;
  opt.boundary_rhs_mode=z4c::Z4cBoundaryRHSMode::sommerfeld;
  opt.telegraph_lapse=true;opt.telegraph_tau=1;opt.telegraph_kappa=1;
  opt.telegraph_damping_prescription=z4c::TelegraphDampingPrescription::max_domain_abs_K;
  std::vector<int> parity;for(int v=0;v<Z::nz4c;++v) parity.push_back(z4c::Z4cStateAxisParitySignFromPackedIndex(v));
  std::vector<std::vector<double>> results;
  for(int steps:{4,8,16,32}) {
    DvceArray5D<Real> leaf("Z4c leaves",leaves.size(),Z::nz4c,1,17,17);
    auto h=Kokkos::create_mirror_view(leaf);
    for(int m=0;m<leaf.extent_int(0);++m) for(int v=0;v<Z::nz4c;++v)
      for(int j=0;j<17;++j) for(int i=0;i<17;++i) {
        const auto key=leaves[m];const double w=std::ldexp(1.,-key[0]);
        const double x=w*(key[1]+(i-4)/8.),z=-1+w*(key[2]+(j-4)/8.);
        double value=(v==Z::I_Z4C_CHI || v==Z::I_Z4C_GXX || v==Z::I_Z4C_GYY || v==Z::I_Z4C_GZZ || v==Z::I_Z4C_ALPHA) ? 1 : 0;
        if(v==Z::I_Z4C_ALPHA) value+=.001*std::exp(-x*x-z*z);
        h(m,v,0,j,i)=value;
      }
    Kokkos::deep_copy(leaf,h);
    subcycling::VertexParentStates storage;storage.InitializeAll(tree,leaf,l);
    subcycling::HierarchyRK4 engine;engine.Initialize<6>(tree,l,0,2,2,parity,4,{{false,true,true,true}});
    z4c::HierarchyPhysics<3> physics(storage,geometry,l,opt,0,2,2,.02/64,
      {{false,true,true,true}},[](double){return 1.;},[](double){return 0.;},[](double){return 0.;});
    const double dt=.04/steps;
    for(int n=0;n<steps;++n) engine.Run(n*dt,dt,storage,physics);
    auto out=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),storage.Values());
    std::vector<double> values;
    for(int n=0;n<out.extent_int(0);++n) if(!tree.Nodes()[n].Covered())
      for(int v=0;v<Z::nz4c;++v) for(int j=l.js;j<=l.je;++j) for(int i=l.is;i<=l.ie;++i) {
        const double value=out(n,v,0,j,i);
        if(!std::isfinite(value)) throw std::runtime_error("nonfinite hierarchy Z4c result");
        values.push_back(value);
      }
    results.push_back(values);std::cout << "completed steps=" << steps << '\n';
  }
  std::vector<double> errors;
  for(int r=0;r<3;++r) {
    double sum=0;for(std::size_t n=0;n<results[r].size();++n) sum+=std::pow(results[r][n]-results[r+1][n],2);
    errors.push_back(std::sqrt(sum/results[r].size()));
    std::cout << "successive rms=" << errors.back() << '\n';
  }
  bool converged=true;
  for(int r=0;r<2;++r) {
    const double ratio=errors[r]/errors[r+1];
    std::cout << "ratio=" << ratio << std::endl;
    converged &= ratio>10 && ratio<24;
  }
  if(!converged) throw std::runtime_error("coupled Z4c temporal convergence gate failed");
 }
 Kokkos::finalize();
}
