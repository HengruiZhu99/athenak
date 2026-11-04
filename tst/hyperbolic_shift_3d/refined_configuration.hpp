// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef HYPERBOLIC_REFINED_CONFIGURATION_HPP_
#define HYPERBOLIC_REFINED_CONFIGURATION_HPP_
#include <fstream>
#include <set>
#include <limits>
#include <mpi.h>
#include <iomanip>
#include <map>
#include <memory>
#include <sstream>
#include "fixed_composite_mesh.hpp"
#include "z4c/hyperboloidal/hyperbolic_shift.hpp"
#include "z4c/hyperboloidal/native_wormhole.hpp"
#include "z4c/hyperboloidal/stationary_gauge_reference.hpp"
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
  Real core_half_width=.18, shell_inner_radius=.16, initial_start_M=4, initial_end_M=12, boost_velocity=.1;
  int core_subdivisions=4;
  bool core_bounds_overridden=false;
  int smr_levels=0;Real smr_radius=.05;
  std::string identity;
  std::shared_ptr<z4c::hyperboloidal::StationaryGaugeReference> stationary_reference;
  cubed_sphere::CompositeMesh Mesh(int n) const {
    cubed_sphere::CompositeMesh mesh(core_half_width,shell_radii,1,n,3,core_subdivisions,shell_inner_radius);
    mesh.evolved_overlap=3;mesh.continuous_cartesian_core=true;return mesh;
  }

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
        "gauge_end",     "buffer_cells",      "cfl_target",         "shell_radii", "core_half_width", "shell_inner_radius", "core_subdivisions", "initial_start_M", "initial_end_M", "boost_velocity", "smr_levels", "smr_radius", "halo_frame", "shift_speed_policy", "outer_gauge"};
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
    core_half_width=number("core_half_width",mesh.half_width);
    shell_inner_radius=number("shell_inner_radius",core_half_width*(.16/.18));
    const Real subdivisions=number("core_subdivisions",4);
    if(subdivisions!=2 && subdivisions!=4 && subdivisions!=8)
      throw std::invalid_argument("core subdivisions must be 2, 4 or 8");
    core_subdivisions=int(subdivisions);
    if(!(core_half_width>0 && shell_inner_radius>0 && shell_inner_radius<core_half_width &&
         std::sqrt(3.)*core_half_width*(1+6./(core_subdivisions*mesh.cells))<.875))
      throw std::invalid_argument("invalid core/overlap or Cartesian evolved cells reach outer scri shell");
    core_bounds_overridden=values.count("core_half_width");
    if(core_bounds_overridden){lower.fill(-core_half_width);upper.fill(core_half_width);}
    core_spacing=2*core_half_width/(core_subdivisions*mesh.cells);
    initial_start_M=number("initial_start_M",4);initial_end_M=number("initial_end_M",12);
    boost_velocity=number("boost_velocity",.1);
    const Real levels=number("smr_levels",0);smr_radius=number("smr_radius",.05);
    if(levels!=std::floor(levels) || levels<0 || levels>3 || !(smr_radius>0 && smr_radius<shell_inner_radius))
      throw std::invalid_argument("invalid static refinement levels/radius");
    smr_levels=int(levels);
    if(std::abs(boost_velocity)>=.3)throw std::invalid_argument("boost outside audited mild range");
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
    const auto halo_frame=values.count("halo_frame")?values.at("halo_frame"):"stored";
    if(halo_frame!="stored"&&halo_frame!="legacy")throw std::invalid_argument("halo_frame must be stored or legacy");
    gauge.conformal_frame_halos=halo_frame=="legacy";
    const auto shift_policy=values.count("shift_speed_policy")?values.at("shift_speed_policy"):"constant";
    if(shift_policy!="constant"&&shift_policy!="separated")throw std::invalid_argument("shift_speed_policy must be constant or separated");
    gauge.separated_shift_speeds=shift_policy=="separated";
    const auto outer_gauge=values.count("outer_gauge")?values.at("outer_gauge"):"original";
    if(outer_gauge!="original"&&outer_gauge!="stationary-reference")throw std::invalid_argument("outer_gauge must be original or stationary-reference");
    gauge.stationary_outer_reference=outer_gauge=="stationary-reference";
    gauge.reference_start_M=initial_start_M;gauge.reference_end_M=initial_end_M;
    if(gauge.conformal_frame_halos&&!gauge.profile.flat)throw std::invalid_argument("legacy-frame halo control requires flat profile");
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
    // All radial interfaces, including any refinement of the outermost shell,
    // are identity-bound. The same original scri operator is evaluated on the
    // selected grid; its spacing and penalty CFL must be recorded separately.
    // The six-shell compile flag selects the legacy baseline, not this runtime
    // descriptor's number of independently specified radial layers.
    if (shell_radii.size() < 6 || shell_radii.size() > 24 || shell_radii.back() != 1 ||
        shell_radii[shell_radii.size()-2] < .875)
      throw std::invalid_argument(
          "6 to 24 shells ending at 1, with the last inner interface at least .875, are required");
    Real previous = shell_inner_radius;
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
    const Native native(1. / 3., 2e-13, initial_start_M, initial_end_M, Native::TransitionProfile::polynomial_c6);
    inner_map_scale = native.IsotropicScale();
    affine_radius = native.InnerRadius();
    // Omega changes the conformal frame, not the native physical coordinate map.
    // Recompute the normalization anyway, rather than importing the BN1 literal.
    gauge.inner_gamma = gauge.profile.flat ? inner_map_scale * inner_map_scale
                                           : z4c::hyperboloidal::InnerTelegrapher::shift_gamma;
    if (!(cfl_target > 0 && cfl_target <= .4 && buffer_cells >= 4 &&
          gauge.inner_begin < gauge.inner_end && gauge.inner_end <= .8))
      throw std::invalid_argument("invalid gauge/CFL/buffer parameters");
    if (gauge.profile.flat && (!(gauge.inner_begin - gauge.profile.outer_radius >=
                                 buffer_cells * largest_transition_spacing) ||
                               !(gauge.inner_begin < affine_radius) ||
                               !(gauge.profile.outer_radius - gauge.profile.flat_radius >=
                                 3 * largest_transition_spacing)))
      throw std::invalid_argument(
          "Omega/gauge transitions need resolved widths, an exterior buffer and affine-map gauge "
          "onset");
    if (!gauge.profile.flat && (gauge.inner_begin != .2 || gauge.inner_end != .56 || initial_start_M!=4 || initial_end_M!=12))
      throw std::invalid_argument("legacy profile preserves BN1 gauge transitions");
    std::ostringstream out;
    out << std::setprecision(17)
        << "RF2;flat-inner-Cartesian-Z4c-v1;native-C6-areal=" << initial_start_M << "," << initial_end_M << ";M=1/3;v=" << boost_velocity << ";restoration4:.6-.9;"
        << "KO8;scri-original;penalty=.6510416666660156;chi-floor=1e-8:.03;lapse-floor=1e-10;"
        << "flat=" << gauge.profile.flat << ";omega=" << gauge.profile.flat_radius << ','
        << gauge.profile.outer_radius << ',' << gauge.profile.plateau
        << ";gauge=" << gauge.inner_begin << ',' << gauge.inner_end << ',' << gauge.inner_gamma
        << ";eta=6;k=" << inner_map_scale << ";cfl=" << cfl_target << ";buffer=" << buffer_cells
        << ";smr=" << smr_levels << "," << smr_radius << ";subdivisions=" << core_subdivisions << ";shell_inner=" << shell_inner_radius << ";box=";
    for (int a = 0; a < 3; ++a)
      out << lower[a] << ',' << upper[a] << ';';
    out << "shells=";
    for (auto r : shell_radii)
      out << r << ',';
    if(gauge.conformal_frame_halos)out<<";halo_frame=legacy-cartesian-to-shell-v1";
    if(gauge.separated_shift_speeds)out<<";shift_speed_policy=separated-inner-v1";
    if(gauge.stationary_outer_reference)out<<";outer_gauge=stationary-reference-v2;initial-lapse-blend=flat-to-inner-begin";
    identity = out.str();
    stationary_reference.reset();
    if(gauge.stationary_outer_reference)stationary_reference=std::make_shared<z4c::hyperboloidal::StationaryGaugeReference>(gauge);
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
    if(stationary_reference && r>gauge.profile.flat_radius){
      const auto reference=stationary_reference->At(r);
      const Real blend=z4c::hyperboloidal::Smoothstep9((r-gauge.profile.flat_radius)/(gauge.inner_begin-gauge.profile.flat_radius));
      const Real A=(1+r*r)/2;
      fields[18]=(1-blend)*(fields[18]+A)+blend*reference.alpha-A;
      for(int a=0;a<3;++a)fields[22+a]=reference.shift_ratio*x[a];
    }
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
