// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <fstream>
#include <iostream>
#include "mesh/cubed_sphere/composite_checkpoint.hpp"
#include "mesh/cubed_sphere/composite_difference.hpp"
#include "mesh/cubed_sphere/scri_surface.hpp"
#include "multigrid/flexible_gmres.hpp"

void Require(bool okay, const char *message) {if (!okay) throw std::runtime_error(message);}
Real Polynomial(const Real x[3]) {
  return 1+.2*x[0]-.1*x[1]+.05*x[2]+x[0]*x[1]+.1*x[0]*x[1]*x[2]+x[2]*x[2];
}
template <typename F> void Reject(F action, const char *message) {
  bool rejected = false;
  try {action();} catch (const std::exception &) {rejected = true;}
  Require(rejected, message);
}
int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  try {
    using namespace cubed_sphere;
    const bool radial_stencils=argc>2?std::stoi(argv[2]):false;
    CompositeMesh mesh(.25, {.65, 1}, 2, 6, 2, 2);
    const auto parent = mesh.blocks[0];
    CompositeDifference initial(mesh,4,radial_stencils);
    std::vector<Real> field(initial.count, 0);
    for (size_t p = 0; p < initial.active.size(); ++p)
      field[initial.active[p]] = Polynomial(initial.geometry[p].x);
    mesh.RefineCartesian(0);
    Require(mesh.blocks.size() == 63 && mesh.revision == 1, "mixed-level leaf refinement");
    CompositeDifference refined(mesh,4,radial_stencils);
    Real transfer_error = 0;
    std::vector<Real> transferred(refined.count, 0);
    for (size_t p = 0; p < refined.active.size(); ++p) {
      const auto &x = refined.geometry[p].x;
      Real value = 0;
      for (const auto &entry : initial.Interpolate(x, 6)) value += entry.weight*field[entry.point];
      transferred[refined.active[p]] = value;
      transfer_error = std::max(transfer_error, std::abs(value-Polynomial(x)));
    }
    mesh.CoarsenCartesian(parent);
    Require(mesh.blocks.size() == 56 && mesh.revision == 2, "Cartesian coarsening revision");
    CompositeDifference coarsened(mesh,4,radial_stencils);
    for (const auto &point : coarsened.geometry) {
      Real value = 0;
      for (const auto &entry : refined.Interpolate(point.x, 6)) value += entry.weight*transferred[entry.point];
      transfer_error = std::max(transfer_error, std::abs(value-Polynomial(point.x)));
    }
    Require(transfer_error < 2e-12, "polynomial refine/coarsen transfer");
    const std::string path = argc > 1 ? argv[1] : "composite-checkpoint-test.bin";
    CompositeSnapshot snapshot{refined.mesh, std::string(64, 'a'), .0005, 1e-11, 2, 22};
    snapshot.radial_interface_stencils=radial_stencils;
    snapshot.telegraph_width=.24;
    snapshot.log_halo_radius=.24;
    snapshot.fields.resize(22*refined.active.size()); snapshot.shift_warm_start.resize(3*refined.active.size());
    for (size_t p = 0; p < snapshot.fields.size(); ++p) snapshot.fields[p] = .01*(p%7419);
    for (size_t p = 0; p < snapshot.shift_warm_start.size(); ++p) snapshot.shift_warm_start[p] = -.001*(p%71);
    CompositeCheckpoint::Write(path, snapshot);
    const auto loaded = CompositeCheckpoint::Read(path, snapshot.source_tag);
    Require(loaded.fields == snapshot.fields && loaded.shift_warm_start == snapshot.shift_warm_start
            && loaded.mesh.blocks.size() == 63 && loaded.mesh.radii == snapshot.mesh.radii
            && loaded.mesh.revision == 1 && loaded.step == 2
            && loaded.telegraph_width==snapshot.telegraph_width
            && loaded.log_halo_radius==snapshot.log_halo_radius
            && loaded.puncture_derivative_scheme==0&&loaded.boundary_derivative_scheme==0, "checkpoint exact payload");
    // Version13 fixed its optional halo support at .12. Removing only the new
    // trailing radius and rebuilding the receipt tests real backward decoding.
    std::ifstream raw_in(path,std::ios::binary);std::vector<char> raw((std::istreambuf_iterator<char>(raw_in)),{});
    auto seal=[](std::vector<char> &bytes){std::uint64_t sum=14695981039346656037ull;
      for(size_t i=0;i<bytes.size()-8;++i){sum^=static_cast<unsigned char>(bytes[i]);sum*=1099511628211ull;}
      std::memcpy(bytes.data()+bytes.size()-8,&sum,8);};
    auto version15=raw;version15.erase(version15.end()-12,version15.end()-8);const std::uint32_t v15=15;
    std::memcpy(version15.data()+8,&v15,4);seal(version15);
    const std::string v15_path=path+".v15";
    {std::ofstream out(v15_path,std::ios::binary);out.write(version15.data(),version15.size());}
    const auto supported_only=CompositeCheckpoint::Read(v15_path,snapshot.source_tag);
    Require(supported_only.boundary_derivative_scheme==0&&supported_only.log_halo_radius==.24,"v15 boundary baseline backward decode");
    std::filesystem::remove(v15_path);
    auto version14=version15;version14.erase(version14.end()-12,version14.end()-8);const std::uint32_t v14=14;
    std::memcpy(version14.data()+8,&v14,4);seal(version14);
    const std::string v14_path=path+".v14";
    {std::ofstream out(v14_path,std::ios::binary);out.write(version14.data(),version14.size());}
    const auto radius_only=CompositeCheckpoint::Read(v14_path,snapshot.source_tag);
    Require(radius_only.log_halo_radius==.24&&radius_only.puncture_derivative_scheme==0,"v14 baseline scheme backward decode");
    std::filesystem::remove(v14_path);
    auto old=version14;old.erase(old.end()-16,old.end()-8);const std::uint32_t old_version=13;
    std::memcpy(old.data()+8,&old_version,4);seal(old);
    const std::string old_path=path+".v13";
    {std::ofstream out(old_path,std::ios::binary);out.write(old.data(),old.size());}
    const auto legacy=CompositeCheckpoint::Read(old_path,snapshot.source_tag);
    Require(legacy.fields==snapshot.fields&&legacy.log_halo_radius==.12,"v13 fixed-support backward decode");
    std::filesystem::remove(old_path);
    auto bad_radius=raw;const Real invalid_radius=std::numeric_limits<Real>::quiet_NaN();
    std::memcpy(bad_radius.data()+bad_radius.size()-24,&invalid_radius,8);seal(bad_radius);
    {std::ofstream out(old_path,std::ios::binary);out.write(bad_radius.data(),bad_radius.size());}
    Reject([&]{CompositeCheckpoint::Read(old_path,snapshot.source_tag);},"checksummed nonfinite halo radius accepted");
    std::filesystem::remove(old_path);
    auto bad_scheme=raw;const int unsupported_scheme=2;
    std::memcpy(bad_scheme.data()+bad_scheme.size()-16,&unsupported_scheme,4);seal(bad_scheme);
    {std::ofstream out(old_path,std::ios::binary);out.write(bad_scheme.data(),bad_scheme.size());}
    Reject([&]{CompositeCheckpoint::Read(old_path,snapshot.source_tag);},"checksummed unsupported derivative scheme accepted");
    std::filesystem::remove(old_path);
    auto bad_boundary=raw;
    std::memcpy(bad_boundary.data()+bad_boundary.size()-12,&unsupported_scheme,4);seal(bad_boundary);
    {std::ofstream out(old_path,std::ios::binary);out.write(bad_boundary.data(),bad_boundary.size());}
    Reject([&]{CompositeCheckpoint::Read(old_path,snapshot.source_tag);},"checksummed unsupported boundary derivative scheme accepted");
    std::filesystem::remove(old_path);
    auto incompatible_boundary=snapshot;incompatible_boundary.boundary_derivative_scheme=1;
    Reject([&]{CompositeCheckpoint::Write(old_path,incompatible_boundary);},"Hermite scheme without tensor scri accepted");
    auto advanced=snapshot;advanced.wormhole_chart=true;advanced.boundary_scheme=1;
    advanced.positive_storage=2;advanced.enriched_log_derivatives=true;advanced.puncture_derivative_scheme=1;advanced.boundary_derivative_scheme=1;
    size_t boundary_points=0;for(const auto &b:advanced.mesh.blocks)
      if(b.layer==int(advanced.mesh.radii.size())-1)boundary_points+=advanced.mesh.cells*advanced.mesh.cells;
    advanced.boundary_fields.assign(boundary_points*advanced.components,0);
    CompositeCheckpoint::Write(old_path,advanced);
    const auto advanced_loaded=CompositeCheckpoint::Read(old_path,snapshot.source_tag);
    Require(advanced_loaded.puncture_derivative_scheme==1&&advanced_loaded.boundary_derivative_scheme==1,"supported and boundary derivative scheme roundtrip");
    advanced.positive_storage=1;
    Reject([&]{CompositeCheckpoint::Write(old_path,advanced);},"supported derivative with incompatible storage accepted");
    std::filesystem::remove(old_path);
    CompositeDifference restored(loaded.mesh,4,loaded.radial_interface_stencils);
    Require(restored.active == refined.active && restored.halos.size() == refined.halos.size(),
            "checkpoint rebuild geometry/connectivity");
    ScriSurface boundary(refined,1);
    std::fill(boundary.values.begin(),boundary.values.end(),1.);
    std::vector<Real> radial(refined.count,0);
    for(size_t p=0;p<refined.active.size();++p)radial[refined.active[p]]=(p%refined.n+.5)/refined.n;
    refined.Fill(radial,1,[&](int,const auto &x){return boundary.Interpolate(boundary.values,1,x.data())[0];});
    Real surface_error=0;
    for(size_t p=0;p<boundary.rows.size();++p){const auto jet=boundary.Sample(p,radial,boundary.values,1);
      surface_error=std::max(surface_error,std::abs(jet.value[0]-1));
      surface_error=std::max(surface_error,std::abs(boundary.ReconstructFromGhost(p,radial,0)-1));
      for(int a=0;a<3;++a){surface_error=std::max(surface_error,std::abs(jet.d[0][a]-boundary.rows[p].geometry.J[0][a]));
        for(int b=0;b<3;++b)surface_error=std::max(surface_error,std::abs(jet.dd[0][a][b]-boundary.rows[p].geometry.H[0][a][b]));}}
    Require(surface_error<3e-10,"boundary first/second chain rule or independent ghost reconstruction");
    for(size_t p=0;p<refined.active.size();++p)radial[refined.active[p]]=std::pow((p%refined.n+.5)/refined.n,6);
    refined.Fill(radial,1,[&](int,const auto &){return 1.;});
    Real active_only_error=0,ghost_boundary_error=0;
    for(size_t p=0;p<boundary.rows.size();++p){const auto &row=boundary.rows[p];Real trace=0;
      for(int i=0;i<6;++i){Real weight=1;for(int j=0;j<6;++j)if(i!=j)weight*=(refined.n-.5-(refined.n-6+j))/Real(i-j);
        trace+=weight*radial[refined.Index(row.block,refined.n-6+i,row.j,row.k)];}
      active_only_error=std::max(active_only_error,std::abs(trace-1));
      ghost_boundary_error=std::max(ghost_boundary_error,std::abs(boundary.ReconstructFromGhost(p,radial,0)-1));}
    Require(active_only_error>1e-5&&ghost_boundary_error<1e-12,"independent actual boundary residual vs extrapolation audit");
    Reject([&] {CompositeCheckpoint::Read(path, std::string(64, 'b'));}, "source mismatch accepted");
    {std::fstream corrupted(path, std::ios::binary|std::ios::in|std::ios::out);
      corrupted.seekp(180); const char byte = 0x55; corrupted.write(&byte, 1);}
    Reject([&] {CompositeCheckpoint::Read(path, snapshot.source_tag);}, "corrupted checkpoint accepted");
    std::filesystem::remove(path);
    // Non-diagonal system and changing right preconditioner verify flexible
    // Arnoldi semantics; deliberately insufficient iterations must not pass.
    const multigrid::Operator apply = [](const auto &x, auto &y) {
      y = {4*x[0]+x[1], x[0]+3*x[1]+x[2], x[1]+2*x[2]};
    };
    const multigrid::InnerProduct dot = [](const auto &x, const auto &y) {
      return x[0]*y[0]+x[1]*y[1]+x[2]*y[2];
    };
    int cycles = 0;
    const multigrid::Operator changing = [&](const auto &x, auto &y) {
      const Real a = ++cycles%2 ? .9 : 1.1;
      y = {a*x[0]/4, x[1]/3, x[2]/(2*a)};
    };
    multigrid::Vector rhs{1, 2, 3}, solution(3, 0);
    const auto report = multigrid::FlexibleGMRES(apply, changing, dot, rhs, solution, 1e-13, 0, 9, 3);
    Require(report.converged && report.residual.back() < 1e-13, "flexible original residual");
    solution.assign(3, 0);
    const auto failed = multigrid::FlexibleGMRES(apply, changing, dot, rhs, solution, 1e-14, 0, 1, 1);
    Require(!failed.converged, "Krylov failure falsely reported convergence");
    std::cout << "{\"mixed_leaf_blocks\":63,\"layers\":2,\"face_subdivisions\":2,"
              << "\"transfer_error\":" << transfer_error << ",\"surface_jet_error\":"<<surface_error
              << ",\"active_only_trace_error\":"<<active_only_error<<",\"ghost_boundary_residual\":"<<ghost_boundary_error<<",\"checkpoint_exact\":true,"
              << "\"corruption_and_source_mismatch_rejected\":true,\"status\":\"passed\"}\n";
  } catch (const std::exception &error) {std::cerr << error.what() << '\n'; status = 1;}
  Kokkos::finalize();
  return status;
}
