// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <chrono>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include "z4c/hyperboloidal/composite_shift.hpp"
#include "manufactured.hpp"

namespace h = z4c::hyperboloidal;
double Cotton(bool curved, Real delta) {
  Real x[3] = {.13, .21, .17};
  const auto state = manufactured::State(x, curved);
  const auto geometry = h::Geometry(state.metric);
  Real schouten[3][3], derivative[3][3][3]{};
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j)
    schouten[i][j] = geometry.ricci[i][j]-.25*geometry.scalar*state.metric.g[i][j];
  for (int k = 0; k < 3; ++k) {
    x[k] += delta; const auto up = manufactured::State(x, curved); const auto gp = h::Geometry(up.metric);
    x[k] -= 2*delta; const auto um = manufactured::State(x, curved); const auto gm = h::Geometry(um.metric);
    x[k] += delta;
    for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      derivative[k][i][j] = (gp.ricci[i][j]-.25*gp.scalar*up.metric.g[i][j]
          -gm.ricci[i][j]+.25*gm.scalar*um.metric.g[i][j])/(2*delta);
      for (int a = 0; a < 3; ++a)
        derivative[k][i][j] -= geometry.connection[a][k][i]*schouten[a][j]
                              +geometry.connection[a][k][j]*schouten[i][a];
    }
  }
  double result = 0;
  for (int i = 0; i < 3; ++i)
  for (int j = 0; j < 3; ++j)
  for (int k = 0; k < 3; ++k) result = std::max(result, std::abs(derivative[k][i][j]-derivative[j][i][k]));
  return result;
}
int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  int status = 0;
  try {
    const int n = argc > 1 ? std::stoi(argv[1]) : 6;
    const int shells = argc > 2 ? std::stoi(argv[2]) : 1;
    const int subdivisions = argc > 3 ? std::stoi(argv[3]) : 1;
    const bool curved = argc > 4 ? std::stoi(argv[4]) : true;
    const bool refine = argc > 5 ? std::stoi(argv[5]) : false;
    const Real tolerance = argc > 6 ? std::stod(argv[6]) : 1e-9;
    const int spatial_order = argc > 7 ? std::stoi(argv[7]) : 4;
    if (n > 24 || (shells != 1 && shells != 2) || subdivisions > 2)
      throw std::invalid_argument("bounded test cell budget");
    std::vector<Real> radii = shells == 1 ? std::vector<Real>{1} : std::vector<Real>{.65, 1};
    cubed_sphere::CompositeMesh mesh(.25, radii, subdivisions, n, 2, refine ? 2 : 1);
    if (refine) mesh.RefineCartesian(0);
    const auto start = std::chrono::steady_clock::now();
    h::CompositeShiftStage stage(mesh, spatial_order);
    const double setup = std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
    auto &grid = stage.grid;
    std::cerr << "built " << grid.active.size() << " cells, " << mesh.blocks.size()
              << " blocks, " << stage.preconditioner.Levels() << " global MG levels in "
              << setup << " seconds\n";
    std::vector<Real> constant(grid.count, 1);
    grid.Fill(constant, 1, [](int, const auto &) {return 1.;});
    double constant_error = 0;
    for (Real value : constant) constant_error = std::max(constant_error, std::abs(value-1));
    std::vector<Real> field(grid.count, 0);
    for (size_t p = 0; p < grid.active.size(); ++p) {
      const auto &x = grid.geometry[p].x;
      field[grid.active[p]] = std::exp(.2*x[0]-.3*x[1]+.1*x[2]);
    }
    grid.Fill(field, 1, [](int, const auto &x) {return std::exp(.2*x[0]-.3*x[1]+.1*x[2]);});
    double derivative_error = 0, hessian_error = 0, map_error = 0, max_curvature = 0;
    std::vector<h::MetricJet<Real>> metric;
    multigrid::Vector source, exact, solution(3*grid.active.size(), 0);
    const Real exponent[3] = {.2, -.3, .1};
    for (size_t p = 0; p < grid.active.size(); ++p) {
      Real d[3], dd[3][3]; grid.Derivatives(field, 0, p, d, dd);
      for (int a = 0; a < 3; ++a) {
        derivative_error = std::max(derivative_error, std::abs(d[a]-exponent[a]*field[grid.active[p]]));
        for (int b = 0; b < 3; ++b)
          hessian_error = std::max(hessian_error, std::abs(dd[a][b]-exponent[a]*exponent[b]*field[grid.active[p]]));
      }
      const auto state = manufactured::State(grid.geometry[p].x, curved);
      metric.push_back(state.metric);
      const auto geometry = h::Geometry(state.metric);
      max_curvature = std::max(max_curvature, std::abs(geometry.scalar));
      Real rhs[3]; h::EllipticShiftSource(state, geometry, 0., rhs);
      for (int v = 0; v < 3; ++v) {source.push_back(rhs[v]); exact.push_back(state.beta.value[v]);}
      const int m = p/(n*n*n);
      Real q[3], back[3]; mesh.Inverse(m, grid.geometry[p].x, q); mesh.Forward(m, q, back);
      for (int a = 0; a < 3; ++a) map_error = std::max(map_error, std::abs(back[a]-grid.geometry[p].x[a]));
    }
    stage.SetGeometry(metric);
    const h::CompositeShiftStage::Boundary boundary = [](const auto &x) {
      return std::array<Real, 3>{-x[0], -x[1], -x[2]};
    };
    multigrid::Vector analytic_discrete;
    stage.Apply(exact, analytic_discrete, boundary);
    multigrid::Vector truncation(source.size());
    for (size_t i = 0; i < truncation.size(); ++i) truncation[i] = analytic_discrete[i]+source[i];
    const auto solve_start = std::chrono::steady_clock::now();
    const auto report = stage.Solve(source, boundary, solution, tolerance, 0., 120);
    const double solve_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-solve_start).count();
    multigrid::Vector result, error(solution.size()), residual(solution.size());
    stage.Apply(solution, result, boundary);
    double max_error = 0, max_residual = 0, bulk_error = 0, seam_error = 0, scri_error = 0;
    for (size_t p = 0; p < grid.active.size(); ++p) {
      const int m = p/(n*n*n), i = p%n, j = p/n%n, k = p/(n*n)%n;
      for (int v = 0; v < 3; ++v) {
        const size_t a = 3*p+v;
        error[a] = solution[a]-exact[a]; residual[a] = result[a]+source[a];
        const double local_error = std::abs(error[a]);
        max_error = std::max(max_error, local_error);
        max_residual = std::max(max_residual, std::abs(residual[a]));
        if (mesh.blocks[m].face < 0) bulk_error = std::max(bulk_error, local_error);
        if (i < 2 || j < 2 || k < 2 || i >= n-2 || j >= n-2 || k >= n-2)
          seam_error = std::max(seam_error, local_error);
        if (mesh.blocks[m].layer == shells-1 && i >= n-2)
          scri_error = std::max(scri_error, local_error);
      }
    }
    const double residual_rms = std::sqrt(stage.Dot(residual, residual));
    const double error_rms = std::sqrt(stage.Dot(error, error));
    const double truncation_rms = std::sqrt(stage.Dot(truncation, truncation));
    const double volume = std::accumulate(grid.volumes.begin(), grid.volumes.end(), 0.);
    const double cotton = Cotton(curved, 1e-4), cotton_fine = Cotton(curved, 5e-5);
    std::cout << std::setprecision(17) << "{\"n\":" << n << ",\"shells\":" << shells
              << ",\"subdivisions\":" << subdivisions << ",\"curved\":" << curved
              << ",\"refined\":" << refine << ",\"blocks\":" << mesh.blocks.size()
              << ",\"cells\":" << grid.active.size() << ",\"mg_levels\":" << stage.preconditioner.Levels()
              << ",\"coarse_cells\":" << stage.preconditioner.CoarseCells()
              << ",\"setup_seconds\":" << setup << ",\"solve_seconds\":" << solve_seconds
              << ",\"iterations\":" << report.iterations << ",\"applications\":" << report.applications
              << ",\"converged\":" << report.converged << ",\"map_error\":" << map_error
              << ",\"constant_error\":" << constant_error << ",\"derivative_error\":" << derivative_error
              << ",\"hessian_error\":" << hessian_error << ",\"volume\":" << volume
              << ",\"curvature_max\":" << max_curvature << ",\"truncation_rms\":" << truncation_rms
              << ",\"error_rms\":" << error_rms << ",\"error_max\":" << max_error
              << ",\"bulk_error\":" << bulk_error << ",\"interface_error\":" << seam_error
              << ",\"scri_collar_error\":" << scri_error << ",\"residual_rms\":" << residual_rms
              << ",\"residual_max\":" << max_residual << ",\"residual_history\":[";
    for (size_t i = 0; i < report.residual.size(); ++i) std::cout << (i ? "," : "") << report.residual[i];
    std::cout << "],\"elliptic_atol\":" << tolerance << ",\"spatial_order\":" << spatial_order
              << ",\"cotton_max\":" << cotton_fine << ",\"cotton_difference\":" << std::abs(cotton-cotton_fine) << "}\n";
    if (!report.converged || constant_error != 0 || map_error > 1e-12 || residual_rms > 1e-7) status = 1;
    if (curved && (!(cotton_fine > 1e-4) || std::abs(cotton-cotton_fine) > 1e-6)) status = 1;
  } catch (const std::exception &error) {
    std::cerr << error.what() << "\n";
    status = 2;
  }
  Kokkos::finalize();
  return status;
}
