// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Experimental bounded CPU composite evolution. No production guards relaxed.
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include "z4c/hyperboloidal/composite_shift.hpp"
#include "z4c/hyperboloidal/reference_gauge.hpp"
#include "mesh/cubed_sphere/composite_checkpoint.hpp"
#include "mesh/cubed_sphere/scri_surface.hpp"
#include "z4c/hyperboloidal/tensor_scri.hpp"
#include "z4c/hyperboloidal/composite_state.hpp"
#include "z4c/hyperboloidal/puncture_log_plans.hpp"
#include "z4c/hyperboloidal/puncture_refinement.hpp"
#include "z4c/hyperboloidal/composite_evolution.hpp"

namespace h = z4c::hyperboloidal;
using V = multigrid::Vector;
constexpr int fields = 22;
const int tensor_i[6] = {0,0,0,1,1,2}, tensor_j[6] = {0,1,2,1,2,2};

cubed_sphere::CompositeMesh PilotMesh(int n,const V &radii,int bulk,int refinement=0){
  auto mesh=cubed_sphere::CompositeMesh(.25,radii,1,n,2,bulk);
  return h::RefinePunctureNeighborhood(std::move(mesh),h::PuncturePosition{},refinement,50000).mesh;
}

h::OmegaJet<Real> Omega(const Real x[3], const h::Z4cJet<Real> &u, bool boundary = false) {
  return h::CompositeOmegaAt(x,u,boundary);
}

using Pilot = h::CompositeTensorEvolution;

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  try {
    if(argc>1&&std::string(argv[1])=="--coordinates"){
      if(argc!=7)throw std::invalid_argument("coordinates: N radii bulk refinement output.csv");
      const int n=std::stoi(argv[2]),bulk=std::stoi(argv[4]),refinement=std::stoi(argv[5]);
      V radii;std::stringstream list(argv[3]);std::string part;
      while(std::getline(list,part,','))radii.push_back(std::stod(part));
      if(n<6||n>16||bulk<1||bulk>4||radii.empty()||radii.size()>16||radii.back()!=1)
        throw std::invalid_argument("bounded coordinate descriptor");
      const auto mesh=PilotMesh(n,radii,bulk,refinement);
      std::ofstream out(argv[6]);out<<std::setprecision(17)<<"kind,id,x,y,z,r\n";size_t id=0;
      for(size_t m=0;m<mesh.blocks.size();++m)for(int k=0;k<n;++k)for(int j=0;j<n;++j)for(int i=0;i<n;++i){
        const Real q[3]={(i+.5)/n,(j+.5)/n,(k+.5)/n};Real x[3];mesh.Forward(m,q,x);
        out<<0<<','<<id++<<','<<x[0]<<','<<x[1]<<','<<x[2]<<','<<std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2])<<'\n';}
      if(!out)throw std::runtime_error("coordinate output failed");
      std::cout<<"{\"cells\":"<<id<<",\"revision\":"<<mesh.revision<<"}\n";
      Kokkos::finalize();return 0;
    }
    const int n = argc > 1 ? std::stoi(argv[1]) : 6;
    const Real amplitude = argc > 2 ? std::stod(argv[2]) : 0.;
    const Real dt = argc > 3 ? std::stod(argv[3]) : .0005;
    const int steps = argc > 4 ? std::stoi(argv[4]) : 2;
    const bool angular = argc > 5 ? std::stoi(argv[5]) : false;
    const int shells = argc > 6 ? std::stoi(argv[6]) : 1;
    const Real tolerance = argc > 7 ? std::stod(argv[7]) : 1e-11;
    const std::string profile = argc > 8 ? argv[8] : "";
    const std::string checkpoint = argc > 9 ? argv[9] : "";
    const std::string source_tag = argc > 10 ? argv[10] : "";
    const int spatial_order = argc > 11 ? std::stoi(argv[11]) : 4;
    const bool regular_scri = argc > 12 ? std::stoi(argv[12]) : false;
    const bool wormhole=argc>15?std::stoi(argv[15]):false;
    const std::string initial=argc>16?argv[16]:"";
    const int bulk=argc>17?std::stoi(argv[17]):1;
    const int isotropic_reference=argc>18?std::stoi(argv[18]):0;
    V radii=shells==1?V{1}:V{.65,1};
    if(argc>19&&std::string(argv[19]).size()){radii.clear();std::stringstream list(argv[19]);std::string value;
      while(std::getline(list,value,','))radii.push_back(std::stod(value));}
    const int surface_points=argc>20?std::stoi(argv[20]):6;
    const bool radial_stencils=argc>21?std::stoi(argv[21]):false;
    const int positive_storage=argc>22?std::stoi(argv[22]):0;
    const Real telegraph_width=argc>23?std::stod(argv[23]):.24;
    const bool enriched=argc>24?std::stoi(argv[24]):false;
    const bool radial_lines=argc>25?std::stoi(argv[25]):false;
    const bool topology=argc>26?std::stoi(argv[26]):false;
    const int refinement=argc>27?std::stoi(argv[27]):0;
    if (n < 6 || n > 16 || steps < 0 || steps > 200 || dt <= 0 || dt*steps > .1
        || std::abs(amplitude) > 1e-3 || shells<1||shells>16||radii.size()!=size_t(shells)||radii.back()!=1||bulk<1||bulk>4
        ||size_t(n)*n*n*(bulk*bulk*bulk+6*shells)>50000
        ||surface_points<6||surface_points>8||surface_points>n+1
        ||refinement<0||refinement>30|| (refinement>0&&!topology)
        ||positive_storage<0||positive_storage>3||(!wormhole&&positive_storage)
        ||!(telegraph_width>0&&telegraph_width<.395)|| (enriched&&positive_storage!=2)
        ||isotropic_reference<0||isotropic_reference>2||(!wormhole&&isotropic_reference)
        ||(wormhole&&(!regular_scri||n%2||initial.empty())))
      throw std::invalid_argument("bounded coupled pilot budget");
    const auto start = std::chrono::steady_clock::now();
    Pilot pilot(PilotMesh(n,radii,bulk,refinement),true,tolerance,spatial_order,regular_scri,wormhole,isotropic_reference,surface_points,radial_stencils,positive_storage,telegraph_width,enriched,radial_lines,topology);
    if(argc>13)pilot.pole_sat=std::stod(argv[13]);if(argc>14)pilot.regularity_sat=std::stod(argv[14]);
    if(pilot.pole_sat<0||pilot.regularity_sat<0)throw std::invalid_argument("negative scri weak-boundary coefficient");
    pilot.Initialize(amplitude, angular);
    if(wormhole)pilot.InitializeWormhole(initial,amplitude,angular);
    const int saved_step = steps/2;
    for (int i = 0; i < steps; ++i) {
      pilot.Step(dt);
      if (!checkpoint.empty() && i+1 == saved_step) {
        cubed_sphere::CompositeSnapshot snapshot{pilot.shift.grid.mesh, source_tag, dt, tolerance,
            static_cast<std::uint64_t>(i+1), fields, pilot.state, pilot.beta_delta};
        snapshot.spatial_order = spatial_order;
        if(pilot.scri){snapshot.boundary_scheme=1;snapshot.boundary_fields=pilot.scri->values;}
        snapshot.boundary_pole_sat=pilot.pole_sat;snapshot.boundary_regularity_sat=pilot.regularity_sat;
        snapshot.wormhole_chart=wormhole;
        snapshot.isotropic_reference=isotropic_reference;
        snapshot.scri_radial_points=surface_points;snapshot.radial_interface_stencils=radial_stencils;
        snapshot.positive_storage=positive_storage;snapshot.telegraph_width=telegraph_width;snapshot.enriched_log_derivatives=enriched;snapshot.radial_line_smoothing=radial_lines;snapshot.topology_coarsening=topology;
        cubed_sphere::CompositeCheckpoint::Write(checkpoint, snapshot);
      }
    }
    bool restart_equal = checkpoint.empty();
    if (!checkpoint.empty()) {
      const auto snapshot = cubed_sphere::CompositeCheckpoint::Read(checkpoint, source_tag);
      Pilot restarted(snapshot.mesh, snapshot.telegraph, snapshot.tolerance, snapshot.spatial_order,snapshot.boundary_scheme==1,snapshot.wormhole_chart,snapshot.isotropic_reference,snapshot.scri_radial_points,snapshot.radial_interface_stencils,snapshot.positive_storage,snapshot.telegraph_width,snapshot.enriched_log_derivatives,snapshot.radial_line_smoothing,snapshot.topology_coarsening);
      restarted.state = snapshot.fields; restarted.beta_delta = snapshot.shift_warm_start;
      restarted.pole_sat=snapshot.boundary_pole_sat;restarted.regularity_sat=snapshot.boundary_regularity_sat;
      if(restarted.scri)restarted.scri->values=snapshot.boundary_fields;
      for (std::uint64_t i = snapshot.step; i < static_cast<std::uint64_t>(steps); ++i)
        restarted.Step(snapshot.dt);
      restart_equal = restarted.state == pilot.state && restarted.beta_delta == pilot.beta_delta;
      if(pilot.scri)restart_equal=restart_equal&&restarted.scri->values==pilot.scri->values;
      if (!restart_equal) throw std::runtime_error("coupled restart is not bitwise reproducible");
    }
    pilot.RHS(pilot.state, dt);  // accepted-state diagnostics, with a fresh stage solve
    const auto &grid = pilot.shift.grid;
    Real H2 = 0, M2 = 0, theta2 = 0, Z2 = 0, volume = 0;
    Real maxH = 0, maxM = 0, drift = 0, min_alpha = 1, min_chi = 1;
    std::ofstream profiles;
    if (!profile.empty()) {
      profiles.open(profile); profiles << std::setprecision(17)
          << "x,y,z,r,weight,alpha,chi,g_rr,A_rr,Khat,Theta,Lambda_r,beta_r,H,M_norm,det_error";
      for (int v = 0; v < fields; ++v) profiles << ",state_" << v;
      profiles << ",delta_beta_x,delta_beta_y,delta_beta_z\n";
    }
    for (size_t p = 0; p < grid.active.size(); ++p) {
      const auto &u = pilot.jets[p];
      const auto c = h::EvolvedConstraints(u, Omega(grid.geometry[p].x, u));
      if (!c.valid) throw std::runtime_error("invalid coupled constraint diagnostic");
      const Real weight = grid.volumes[p]*std::sqrt(pilot.shift.geometry[p].determinant)/std::pow(u.chi.value, 1.5);
      volume += weight; H2 += weight*c.hamiltonian*c.hamiltonian;
      M2 += weight*c.momentum_conformal_norm2; theta2 += weight*u.theta.value*u.theta.value;
      Z2 += weight*c.z4.z_conformal_norm2;
      maxH = std::max(maxH, std::abs(c.hamiltonian));
      maxM = std::max(maxM, std::sqrt(c.momentum_conformal_norm2));
      min_alpha = std::min(min_alpha, u.alpha.value); min_chi = std::min(min_chi, u.chi.value);
      for (int v = 0; v < fields; ++v) drift = std::max(drift, std::abs(pilot.state[p*fields+v]));
      if (profiles) {
        const auto &x = grid.geometry[p].x;
        const Real r = std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
        Real grr = 0, arr = 0, lambda = 0, beta_r = 0;
        if (r > 0) for (int a = 0; a < 3; ++a) {
          lambda += u.lambda.value[a]*x[a]/r; beta_r += u.beta.value[a]*x[a]/r;
          for (int b = 0; b < 3; ++b) {
            grr += u.metric.g[a][b]*x[a]*x[b]/(r*r);
            arr += u.a.k[a][b]*x[a]*x[b]/(r*r);
          }
        }
        profiles << x[0] << ',' << x[1] << ',' << x[2] << ',' << r << ',' << weight << ','
                 << u.alpha.value << ',' << u.chi.value << ',' << grr << ',' << arr << ','
                 << u.trace.value << ',' << u.theta.value << ',' << lambda << ',' << beta_r << ','
                 << c.hamiltonian << ',' << std::sqrt(c.momentum_conformal_norm2) << ','
                 << c.z4.determinant_residual;
        for (int v = 0; v < fields; ++v) profiles << ',' << pilot.state[p*fields+v];
        for (int v = 0; v < 3; ++v) profiles << ',' << pilot.beta_delta[3*p+v];
        profiles << '\n';
      }
    }
    const V padded = pilot.Padded(pilot.state);
    Real scri_clock=0,scri_null=0,scri_tangent=0,ghost_clock=0,ghost_null=0,ghost_tangent=0;
    if(pilot.scri)for(size_t p=0;p<pilot.scri->rows.size();++p){
      const auto &x=pilot.scri->rows[p].geometry.x;
      const auto sample=pilot.scri->Sample(p,padded,pilot.scri->values,fields);
      const auto u=h::UnpackComposite(x,sample.value,sample.d,sample.dd,wormhole,isotropic_reference,positive_storage);const auto geometry=h::Geometry(u.metric);
      if(!geometry.valid)throw std::runtime_error("invalid accepted scri metric");
      Real bn=0,bghost=0,norm=0,ghost[3];
      for(int i=0;i<3;++i){bn+=(-x[i]+pilot.scri->beta[3*p+i])*x[i];
        ghost[i]=-x[i]+pilot.scri->ReconstructFromGhost(p,pilot.beta_padded,i);bghost+=ghost[i]*x[i];
        for(int j=0;j<3;++j)norm+=u.chi.value*geometry.inverse[i][j]*x[i]*x[j];}
      const Real ag=pilot.scri->ReconstructFromGhost(p,padded,18);
      const Real alpha_ghost=positive_storage?pilot.DecodePositive(18,ag,x):(wormhole?std::exp(ag):(1+x[0]*x[0]+x[1]*x[1]+x[2]*x[2])/2+ag);
      scri_clock=std::max(scri_clock,std::abs(u.alpha.value-1));ghost_clock=std::max(ghost_clock,std::abs(alpha_ghost-1));
      scri_null=std::max(scri_null,std::abs(bn+u.alpha.value*std::sqrt(norm)));
      ghost_null=std::max(ghost_null,std::abs(bghost+alpha_ghost*std::sqrt(norm)));
      for(int i=0;i<3;++i){scri_tangent=std::max(scri_tangent,std::abs(-x[i]+pilot.scri->beta[3*p+i]-bn*x[i]));
        ghost_tangent=std::max(ghost_tangent,std::abs(ghost[i]-bghost*x[i]));}
    }
    if (!profile.empty()) {
      std::ofstream spheres(profile+".spheres.csv");
      spheres << std::setprecision(17) << "r,direction,alpha,chi,g_rr,A_rr,Khat,Theta,Lambda_r,beta_r\n";
      constexpr int directions = 64;
      for (int radial = 1; radial < 20; ++radial)
      for (int direction = 0; direction < directions; ++direction) {
        const Real radius = radial/20.;
        const Real z = 1-2*(direction+.5)/directions;
        const Real phi = direction*2.39996322972865332;
        const Real normal[3] = {std::sqrt(1-z*z)*std::cos(phi), std::sqrt(1-z*z)*std::sin(phi), z};
        const Real x[3] = {radius*normal[0], radius*normal[1], radius*normal[2]};
        V value(fields, 0); Real delta[3]{};
        for (const auto &entry : grid.Interpolate(x, std::min(n, 6))) {
          for (int v = 0; v < fields; ++v) value[v] += entry.weight*padded[v*grid.count+entry.point];
          // Convert the padded donor index back to its active lexicographic ID.
          const int i = entry.point%grid.padded-grid.ng;
          const int j = entry.point/grid.padded%grid.padded-grid.ng;
          const int k = entry.point/(grid.padded*grid.padded)%grid.padded-grid.ng;
          const int m = entry.point/(grid.padded*grid.padded*grid.padded);
          const size_t active = ((m*n+k)*n+j)*n+i;
          for (int v = 0; v < 3; ++v) delta[v] += entry.weight*pilot.beta_delta[3*active+v];
        }
        Real grr = 1, arr = 0, lambda = 0, beta_r = -radius;
        for (int t = 0; t < 6; ++t) {
          const int i = tensor_i[t], j = tensor_j[t];
          const Real weight = normal[i]*normal[j]*(i == j ? 1 : 2);
          grr += weight*value[1+t]; arr += weight*value[7+t];
        }
        for (int v = 0; v < 3; ++v) {lambda += value[15+v]*normal[v]; beta_r += delta[v]*normal[v];}
        std::vector<std::array<Real,3>> zeros(fields);std::vector<std::array<std::array<Real,3>,3>> zz(fields);
        const auto extracted=h::UnpackComposite(x,value,zeros,zz,wormhole,isotropic_reference,positive_storage);
        arr=0;for(int i=0;i<3;++i)for(int j=0;j<3;++j)arr+=extracted.a.k[i][j]*normal[i]*normal[j];
        spheres << radius << ',' << direction << ',' << extracted.alpha.value << ',' << extracted.chi.value << ',' << grr << ',' << arr << ','
                << extracted.trace.value << ',' << extracted.theta.value << ',' << lambda << ',' << beta_r << '\n';
      }
    }
    V beta(3*grid.count, 0);
    for (size_t p = 0; p < grid.active.size(); ++p)
      for (int v = 0; v < 3; ++v) beta[v*grid.count+grid.active[p]] = pilot.beta_delta[3*p+v];
    Real pole = 0, clock = 0, null = 0, tangential = 0;
    const h::CMCReference<Real> reference{1, 1};
    for (size_t m = 0; m < grid.mesh.blocks.size(); ++m) {
      if (grid.mesh.blocks[m].layer != shells-1) continue;
      for (int k = 0; k < n; ++k) for (int j = 0; j < n; ++j) {
        const Real q[3] = {1, (j+.5)/n, (k+.5)/n};
        Real x[3]; grid.mesh.Forward(m, q, x);
        V values; std::vector<std::array<Real, 3>> d;
        std::vector<std::array<std::array<Real, 3>, 3>> dd;
        grid.SampleJets(padded, fields, x, values, d, dd);
        auto u = h::UnpackComposite(x, values, d, dd,wormhole,isotropic_reference,positive_storage);
        grid.SampleJets(beta, 3, x, values, d, dd);
        for (int v = 0; v < 3; ++v) {
          u.beta.value[v] += values[v];
          for (int a = 0; a < 3; ++a) {
            u.beta.d[a][v] += d[v][a];
            for (int b = 0; b < 3; ++b) u.beta.dd[a][b][v] += dd[v][a][b];
          }
        }
        const auto omega = Omega(x, u, true);
        auto parts = h::ConformalRHS(u, omega, 4., -.99,true);
          const auto gauge = h::UnfactoredReferenceGauge(reference, reference.At(x[0], x[1], x[2]),
            u, h::GaugeParameters<Real>{1, 0, wormhole?2.:1., 0}, wormhole, true);
        pole = std::max({pole, std::abs(parts.pole.chi), std::abs(parts.pole.trace),
                        std::abs(parts.pole.theta), std::abs(gauge.pole.alpha)});
        for (int a = 0; a < 3; ++a) {
          pole = std::max(pole, std::abs(parts.pole.lambda[a]));
          for (int b = 0; b < 3; ++b) pole = std::max(pole, std::abs(parts.pole.a[a][b]));
        }
        const auto geometry = h::Geometry(u.metric);
        Real bn = 0, norm = 0;
        for (int a = 0; a < 3; ++a) {
          bn += u.beta.value[a]*x[a];
          for (int b = 0; b < 3; ++b) norm += u.chi.value*geometry.inverse[a][b]*x[a]*x[b];
        }
        null = std::max(null, std::abs(bn+u.alpha.value*std::sqrt(norm)));
        clock = std::max(clock, std::abs(u.alpha.value-1));
        for (int a = 0; a < 3; ++a) tangential = std::max(tangential, std::abs(u.beta.value[a]-bn*x[a]));
      }
    }
    std::cout << std::setprecision(17) << "{\"n\":" << n << ",\"shells\":" << shells
              << ",\"cells\":" << grid.active.size() << ",\"amplitude\":" << amplitude
              << ",\"angular\":" << angular << ",\"dt\":" << dt << ",\"steps\":" << steps
              << ",\"time\":" << dt*steps << ",\"seconds\":"
              << std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()
              << ",\"elliptic_iterations\":" << pilot.elliptic_iterations
              << ",\"shift_residual_max\":" << pilot.max_shift_residual
              << ",\"shift_residual_point_max\":"<<pilot.max_shift_point_residual
              << ",\"H_rms\":" << std::sqrt(H2/volume) << ",\"M_rms\":" << std::sqrt(M2/volume)
              << ",\"Theta_rms\":" << std::sqrt(theta2/volume) << ",\"Z_rms\":" << std::sqrt(Z2/volume)
              << ",\"H_max\":" << maxH << ",\"M_max\":" << maxM << ",\"max_deviation\":" << drift
              << ",\"min_alpha\":" << min_alpha << ",\"min_chi\":" << min_chi
              << ",\"scri_pole\":" << pole << ",\"scri_clock_extrapolation\":" << clock
              << ",\"scri_null_extrapolation\":" << null << ",\"scri_tangent_extrapolation\":" << tangential
              << ",\"restart_checked\":" << !checkpoint.empty() << ",\"restart_equal\":" << restart_equal
              << ",\"spatial_order\":" << spatial_order
              << ",\"wormhole_chart\":"<<wormhole<<",\"bulk_subdivisions\":"<<bulk
              << ",\"isotropic_reference\":"<<isotropic_reference
              << ",\"scri_radial_points\":"<<surface_points<<",\"radial_interface_stencils\":"<<radial_stencils
              << ",\"topology_coarsening\":"<<topology
              << ",\"mesh_revision\":"<<grid.mesh.revision
              << ",\"radial_line_smoothing\":"<<radial_lines
              << ",\"enriched_log_derivatives\":"<<enriched
              << ",\"enriched_cells\":"<<(pilot.log_plans?pilot.log_plans->rows.size():0)
              << ",\"telegraph_width\":"<<telegraph_width
              << ",\"positive_storage\":"<<positive_storage
              << ",\"regular_scri\":"<<regular_scri<<",\"scri_nodal_clock\":"<<scri_clock<<",\"scri_nodal_null\":"<<scri_null
              << ",\"scri_nodal_tangent\":"<<scri_tangent<<",\"scri_ghost_clock\":"<<ghost_clock<<",\"scri_ghost_null\":"<<ghost_null
              << ",\"scri_ghost_tangent\":"<<ghost_tangent<<",\"scri_tensor_pole\":"<<pilot.boundary_pole
              << ",\"scri_raw_clock_rate\":"<<pilot.raw_clock_rate<<",\"scri_regularity\":"<<pilot.boundary_regularity
              << ",\"closure\":\""<<(regular_scri?"evolved_tensor_limit":"experimental_active_only_extrapolation")<<"\"}\n";
    // Report original scientific gates without weakening them for a coarse grid.
    if(regular_scri){if(pilot.boundary_pole>1e-7||ghost_clock>1e-13||ghost_null>1e-12||ghost_tangent>1e-12
        ||(!wormhole&&amplitude==0&&drift>1e-9))status=1;}
    else if (pole > 1e-7 || clock > 1e-13 || null > 1e-12 || (amplitude == 0 && drift > 1e-9)) status = 1;
  } catch (const std::exception &error) {std::cerr << error.what() << "\n"; status = 2;}
  Kokkos::finalize();
  return status;
}
