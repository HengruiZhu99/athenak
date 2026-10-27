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
    CompositeMesh mesh(.25, {.65, 1}, 2, 6, 2, 2);
    const auto parent = mesh.blocks[0];
    CompositeDifference initial(mesh);
    std::vector<Real> field(initial.count, 0);
    for (size_t p = 0; p < initial.active.size(); ++p)
      field[initial.active[p]] = Polynomial(initial.geometry[p].x);
    mesh.RefineCartesian(0);
    Require(mesh.blocks.size() == 63 && mesh.revision == 1, "mixed-level leaf refinement");
    CompositeDifference refined(mesh);
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
    CompositeDifference coarsened(mesh);
    for (const auto &point : coarsened.geometry) {
      Real value = 0;
      for (const auto &entry : refined.Interpolate(point.x, 6)) value += entry.weight*transferred[entry.point];
      transfer_error = std::max(transfer_error, std::abs(value-Polynomial(point.x)));
    }
    Require(transfer_error < 2e-12, "polynomial refine/coarsen transfer");
    const std::string path = argc > 1 ? argv[1] : "composite-checkpoint-test.bin";
    CompositeSnapshot snapshot{refined.mesh, std::string(64, 'a'), .0005, 1e-11, 2, 22};
    snapshot.fields.resize(22*refined.active.size()); snapshot.shift_warm_start.resize(3*refined.active.size());
    for (size_t p = 0; p < snapshot.fields.size(); ++p) snapshot.fields[p] = .01*(p%7419);
    for (size_t p = 0; p < snapshot.shift_warm_start.size(); ++p) snapshot.shift_warm_start[p] = -.001*(p%71);
    CompositeCheckpoint::Write(path, snapshot);
    const auto loaded = CompositeCheckpoint::Read(path, snapshot.source_tag);
    Require(loaded.fields == snapshot.fields && loaded.shift_warm_start == snapshot.shift_warm_start
            && loaded.mesh.blocks.size() == 63 && loaded.mesh.radii == snapshot.mesh.radii
            && loaded.mesh.revision == 1 && loaded.step == 2, "checkpoint exact payload");
    CompositeDifference restored(loaded.mesh);
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
