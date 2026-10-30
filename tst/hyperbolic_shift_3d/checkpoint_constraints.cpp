// Diagnostic-only use of the native DT1 loader, halo closure and volume kernels.
// No time stepping, horizon search, or checkpoint output occurs here.
#define main hyperbolic_unused_evolution_main
#include "coupled_mpi.cpp"
#undef main

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &global_variable::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &global_variable::nranks);
  Kokkos::initialize(argc, argv);
  int code = 0;
  try {
    if (argc != 6) throw std::invalid_argument(
        "usage: checkpoint_constraints n checkpoint dt cutoff output.json");
    const int n = std::stoi(argv[1]);
    const Real dt = std::stod(argv[3]), cutoff = std::stod(argv[4]);
    if (n < 6 || n % 2 || !std::isfinite(dt) || dt <= 0 ||
        !std::isfinite(cutoff) || cutoff < 0 || cutoff >= 1)
      throw std::invalid_argument("invalid resolution/dt/cutoff");
    CollectiveLocal([&] {
      if (global_variable::my_rank == 0 && std::filesystem::exists(argv[5]))
        throw std::runtime_error("refusing to overwrite diagnostic output");
    });
    std::unique_ptr<Evolution> evolution;
    CollectiveLocal([&] { evolution = std::make_unique<Evolution>(n, 2, 2, ""); });
    auto &e = *evolution;
    e.LoadCheckpoint(argv[2], 2, dt);
    e.RefreshBoundary(dt);
    h::HyperbolicCompositeExchange::Summary halo{};
    CollectiveLocal([&] { halo = e.exchange()->Fill(e.state); });
    e.SynchronizeError(halo.invalid, "invalid diagnostic halo");
    Diagnostic d{};
    CollectiveLocal([&] { d = e.kernel()->Diagnostics(e.exchange()->padded, cutoff, e.surface); });
    d = ReduceDiagnostic(d);
    e.SynchronizeError(d.invalid_cells, "invalid diagnostic volume geometry");
    CollectiveLocal([&] {
      if (e.rank != 0) return;
      std::ofstream out(argv[5]);
      out << std::setprecision(17) << "{\"diagnostic_only\":true,\"n\":" << n
          << ",\"t\":" << e.time << ",\"t_over_M\":" << 3*e.time
          << ",\"steps\":" << e.steps << ",\"cutoff\":" << cutoff
          << ",\"checkpoint_schema\":" << JsonString(checkpoint_magic)
          << ",\"horizon_acceptance_assumed\":true,\"norms\":{";
      auto emit = [&](const char *name, const auto &s) {
        out << JsonString(name) << ":{\"volume\":" << s.volume
            << ",\"H_L2\":" << JsonNumber(Norm(s.h2, s.volume))
            << ",\"M_L2\":" << JsonNumber(Norm(s.m2, s.volume))
            << ",\"Mr_L2\":" << JsonNumber(Norm(s.mr2, s.volume))
            << ",\"Theta_L2\":" << JsonNumber(Norm(s.theta2, s.volume))
            << ",\"Zr_L2\":" << JsonNumber(Norm(s.zr2, s.volume)) << '}';
      };
      emit("total", d.total); out << ','; emit("exterior", d.exterior);
      out << "}}\n";
      out.close();
      if (!out) throw std::runtime_error("diagnostic output write failed");
    });
  } catch (const std::exception &ex) {
    if (!global_variable::my_rank) std::cerr << ex.what() << '\n';
    code = 1;
  }
  Kokkos::finalize(); MPI_Finalize(); return code;
}
