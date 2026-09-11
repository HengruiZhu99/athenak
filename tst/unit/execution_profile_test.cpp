#include <stdexcept>
#include <cmath>
#include <iostream>
#include <Kokkos_Core.hpp>
#include "driver/execution_profile.hpp"

void Check(bool ok) { if (!ok) throw std::runtime_error("profile regression failed"); }

int main(int argc, char **argv) {
  Kokkos::initialize(argc, argv);
  {
    namespace ep = execution_profile;
    { ep::Scope disabled("disabled"); }
    Check(ep::entries.empty());
    ep::enabled = true;
    Kokkos::View<double*> values("values", 32768);
    {
      ep::Scope parent("parent");
      {
        ep::Scope child("child");
        Kokkos::parallel_for("profile test fill", values.extent(0),
            KOKKOS_LAMBDA(int i) { values(i) = i*0.25; });
      }
      {
        ep::Scope child("child");
        Kokkos::parallel_for("profile test update", values.extent(0),
            KOKKOS_LAMBDA(int i) { values(i) += 1.0; });
      }
    }
    const auto p = ep::entries.at("parent"), c = ep::entries.at("parent/child");
    Check(p.calls == 1 && c.calls == 2);
    Check(c.inclusive > 0 && p.exclusive >= 0);
    Check(std::abs(p.inclusive-p.exclusive-c.inclusive) < 1.e-12);
    Check(c.exclusive == c.inclusive);
    const auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
    for (int i=0; i<32768; ++i) Check(host(i) == i*0.25+1.0);
    ep::enabled = false;
    { ep::Scope disabled("disabled"); }
    Check(ep::entries.count("disabled") == 0);
    std::cout << "PASS: nested exclusive accounting, disabled path, GPU completion\n";
  }
  Kokkos::finalize();
}
