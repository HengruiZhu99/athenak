// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef HYPERBOLIC_REFINED_CONFIGURATION_HPP_
#define HYPERBOLIC_REFINED_CONFIGURATION_HPP_
#include <fstream>
#include <set>
#include <limits>
#include <mpi.h>
#include <iomanip>
#include <map>
#include <sstream>
#include "fixed_composite_mesh.hpp"
#include "z4c/hyperboloidal/hyperbolic_shift.hpp"
#include "z4c/hyperboloidal/native_wormhole.hpp"
#ifndef HYPERBOLIC_CONFIGURED_FORMULATION
#define HYPERBOLIC_CONFIGURED_FORMULATION 0
#endif
namespace hyperbolic_reproduction {
inline constexpr bool configured_formulation = HYPERBOLIC_CONFIGURED_FORMULATION;

struct RefinedConfiguration {
  z4c::hyperboloidal::HyperbolicShiftParameters<Real> gauge;
  std::vector<Real> shell_radii;
  std::array<Real, 3> lower{}, upper{};
  Real inscribed_radius = 0, inner_map_scale = 0, affine_radius = 0;
  Real cfl_target = .2, buffer_cells = 4, largest_transition_spacing = 0;
  std::string identity;

  static std::string Trim(std::string s) {
    const auto a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
  }
  void Load(const std::string& filename, const cubed_sphere::CompositeMesh& mesh) {
    std::ifstream input(filename);
    if (!input)
      throw std::invalid_argument("cannot open refined configuration");
    std::map<std::string, std::string> values;
    const std::set<std::string> allowed = {
        "omega_profile", "omega_flat_radius", "omega_outer_radius", "omega_plateau", "gauge_begin",
        "gauge_end",     "buffer_cells",      "cfl_target",         "shell_radii"};
    std::string line;
    while (std::getline(input, line)) {
      line = Trim(line.substr(0, line.find('#')));
      if (line.empty())
        continue;
      const auto equal = line.find('=');
      if (equal == std::string::npos)
        throw std::invalid_argument("configuration requires key=value");
      const auto key = Trim(line.substr(0, equal)), value = Trim(line.substr(equal + 1));
      if (!allowed.count(key) || value.empty() || !values.emplace(key, value).second)
        throw std::invalid_argument("unknown, duplicate or empty configuration key: " + key);
    }
    auto number = [&](const std::string& key, Real fallback) {
      const auto found = values.find(key);
      if (found == values.end())
        return fallback;
      size_t used = 0;
      const Real x = std::stod(found->second, &used);
      if (used != found->second.size() || !std::isfinite(x))
        throw std::invalid_argument("invalid finite configuration value: " + key);
      return x;
    };
    lower.fill(std::numeric_limits<Real>::infinity());
    upper.fill(-std::numeric_limits<Real>::infinity());
    Real core_spacing = 0;
    for (const auto& b : mesh.blocks)
      if (b.face < 0)
        for (int a = 0; a < 3; ++a) {
          lower[a] = std::min(lower[a], b.lo[a]);
          upper[a] = std::max(upper[a], b.hi[a]);
          core_spacing = std::max(core_spacing, (b.hi[a] - b.lo[a]) / mesh.cells);
        }
    inscribed_radius = std::numeric_limits<Real>::infinity();
    for (int a = 0; a < 3; ++a) {
      if (!(lower[a] < 0 && upper[a] > 0))
        throw std::invalid_argument("origin must be inside Cartesian domain");
      inscribed_radius = std::min(inscribed_radius, std::min(-lower[a], upper[a]));
    }
    const auto profile = values.count("omega_profile") ? values.at("omega_profile") : "flat";
    if (profile != "flat" && profile != "legacy")
      throw std::invalid_argument("omega_profile must be flat or legacy");
    gauge.profile.flat = profile == "flat";
    gauge.enforce_cfl = true;
    if (values.count("omega_flat_radius") && values.at("omega_flat_radius") == "auto")
      values.erase("omega_flat_radius");
    gauge.profile.flat_radius = number("omega_flat_radius", inscribed_radius);
    gauge.profile.outer_radius = number("omega_outer_radius", gauge.profile.flat_radius + .04);
    gauge.profile.plateau = number("omega_plateau", .5);
    gauge.profile.Validate();
    gauge.inner_begin =
        number("gauge_begin", gauge.profile.flat ? gauge.profile.outer_radius + .06 : .2);
    gauge.inner_end = number("gauge_end", .56);
    buffer_cells = number("buffer_cells", 4);
    cfl_target = number("cfl_target", .2);
    shell_radii = mesh.radii;
    if (values.count("shell_radii")) {
      shell_radii.clear();
      std::istringstream list(values.at("shell_radii"));
      std::string word;
      while (std::getline(list, word, ',')) {
        word = Trim(word);
        size_t used = 0;
        const Real r = std::stod(word, &used);
        if (used != word.size() || !std::isfinite(r))
          throw std::invalid_argument("invalid shell radius");
        shell_radii.push_back(r);
      }
    }
    if (shell_radii.size() != 6 || shell_radii.back() != 1 || shell_radii[4] != .875)
      throw std::invalid_argument(
          "six shells with final interfaces .875 and 1 are required for unchanged scri support");
    Real previous = mesh.inner_radius;
    largest_transition_spacing = core_spacing;
    for (Real radius : shell_radii) {
      if (!(radius > previous))
        throw std::invalid_argument("shell radii must increase from the overlap boundary");
      if (previous < gauge.inner_begin && radius > gauge.profile.flat_radius)
        largest_transition_spacing =
            std::max(largest_transition_spacing, (radius - previous) / mesh.cells);
      previous = radius;
    }
    using Native = z4c::hyperboloidal::NativeWormhole;
    const Native native(1. / 3., 2e-13, 4, 12, Native::TransitionProfile::polynomial_c6);
    inner_map_scale = native.IsotropicScale();
    affine_radius = native.InnerRadius();
    // Omega changes the conformal frame, not the native physical coordinate map.
    // Recompute the normalization anyway, rather than importing the BN1 literal.
    gauge.inner_gamma = gauge.profile.flat ? inner_map_scale * inner_map_scale
                                           : z4c::hyperboloidal::InnerTelegrapher::shift_gamma;
    if (!(cfl_target > 0 && cfl_target <= .4 && buffer_cells >= 4 &&
          gauge.inner_begin < gauge.inner_end && gauge.inner_end <= .56))
      throw std::invalid_argument("invalid gauge/CFL/buffer parameters");
    if (gauge.profile.flat && (!(gauge.inner_begin - gauge.profile.outer_radius >=
                                 buffer_cells * largest_transition_spacing) ||
                               !(gauge.inner_begin < affine_radius) ||
                               !(gauge.profile.outer_radius - gauge.profile.flat_radius >=
                                 3 * largest_transition_spacing)))
      throw std::invalid_argument(
          "Omega/gauge transitions need resolved widths, an exterior buffer and affine-map gauge "
          "onset");
    if (!gauge.profile.flat && (gauge.inner_begin != .2 || gauge.inner_end != .56))
      throw std::invalid_argument("legacy profile preserves BN1 gauge transitions");
    std::ostringstream out;
    out << std::setprecision(17)
        << "RF1;flat-inner-Cartesian-Z4c-v1;native-C6-areal4M-12M;M=1/3;v=.1;restoration4:.6-.9;"
        << "KO8;scri-original;penalty=.6510416666660156;chi-floor=1e-8:.03;lapse-floor=1e-10;"
        << "flat=" << gauge.profile.flat << ";omega=" << gauge.profile.flat_radius << ','
        << gauge.profile.outer_radius << ',' << gauge.profile.plateau
        << ";gauge=" << gauge.inner_begin << ',' << gauge.inner_end << ',' << gauge.inner_gamma
        << ";eta=6;k=" << inner_map_scale << ";cfl=" << cfl_target << ";buffer=" << buffer_cells
        << ";box=";
    for (int a = 0; a < 3; ++a)
      out << lower[a] << ',' << upper[a] << ';';
    out << "shells=";
    for (auto r : shell_radii)
      out << r << ',';
    identity = out.str();
  }
  void CheckCollective() const {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int length = identity.size();
    MPI_Bcast(&length, 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::string reference = rank == 0 ? identity : std::string(length, '\0');
    MPI_Bcast(reference.data(), length, MPI_CHAR, 0, MPI_COMM_WORLD);
    int bad = identity != reference;
    MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (bad)
      throw std::invalid_argument("refined configuration differs across ranks");
  }
  void InitialFields(const Real x[3], std::array<double, 31>& fields) const {
    if (!gauge.profile.flat)
      return;
    const Real r = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
    gauge.profile.TransformFields(r, fields);
    const Real weight = z4c::hyperboloidal::Smoothstep9((r - gauge.inner_begin) /
                                                        (gauge.inner_end - gauge.inner_begin));
    for (int a = 0; a < 3; ++a)
      fields[22 + a] = (1 - weight) * x[a];
  }
  void WriteIdentity(std::ostream& out) const {
    if (identity.empty())
      throw std::logic_error("unresolved formulation identity");
    const uint64_t size = identity.size();
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    out.write(identity.data(), size);
  }
  void ReadIdentity(std::istream& in) const {
    uint64_t size = 0;
    in.read(reinterpret_cast<char*>(&size), sizeof(size));
    if (!in || size != identity.size())
      throw std::runtime_error("checkpoint formulation identity mismatch");
    std::string text(size, '\0');
    in.read(text.data(), size);
    if (!in || text != identity)
      throw std::runtime_error("checkpoint formulation identity mismatch");
  }
};
inline RefinedConfiguration refined_configuration;
}  // namespace hyperbolic_reproduction
#endif
