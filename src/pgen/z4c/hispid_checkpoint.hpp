// Portable explicit-field reader; never deserialize native structure padding.
#ifndef PGEN_Z4C_HISPID_CHECKPOINT_HPP_
#define PGEN_Z4C_HISPID_CHECKPOINT_HPP_
#include <cmath>
#include <fstream>
#include <locale>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "HiSpID.h"

namespace hispid_import {
struct Checkpoint {
  HiSpID_Config config{};
  int seed_family=HISPID_SEED_QI;
  std::string library_sha, acceptance;
  std::vector<double> unknowns;
};
inline Checkpoint Read(const std::string &filename, const std::string &expected_sha) {
  std::ifstream in(filename);in.imbue(std::locale::classic());
  if (!in) throw std::runtime_error("Cannot open HiSpID checkpoint");
  auto label=[&](const char *expected) {
    std::string name;if (!(in>>name) || name!=expected)
      throw std::runtime_error(std::string("Expected checkpoint field: ")+expected);
  };
  auto real=[&](double &value) {
    if (!(in>>value) || !std::isfinite(value)) throw std::runtime_error("Nonfinite/invalid checkpoint value");
  };
  auto integer=[&](int &value) { if (!(in>>value)) throw std::runtime_error("Invalid checkpoint integer"); };
  Checkpoint result;auto &c=result.config;
  label("HISPID_CHECKPOINT");int version;integer(version);
  if (version!=1 && version!=2) throw std::runtime_error("Unsupported HiSpID checkpoint version");
  label("parameterization");std::string parameterization;in>>parameterization;
  if (parameterization!=HiSpID_unknown_parameterization()) throw std::runtime_error("Unsupported HiSpID unknown parameterization");
  if (version==2) {
    label("seed_family");std::string family;in>>family;
    if (family=="qi") result.seed_family=HISPID_SEED_QI;
    else if (family=="trumpet_r0_m") result.seed_family=HISPID_SEED_TRUMPET_R0_M;
    else throw std::runtime_error("Unsupported HiSpID seed family");
  }
  label("library_sha256");in>>result.library_sha;
  if (result.library_sha.size()!=64 || result.library_sha.find_first_not_of("0123456789abcdef")!=std::string::npos ||
      result.library_sha!=expected_sha) throw std::runtime_error("Checkpoint source-library SHA does not match input");
  label("acceptance");in>>result.acceptance;
  if (result.acceptance!="analytic_seed" && result.acceptance!="preliminary" &&
      result.acceptance!="strong" && result.acceptance!="diagnostic") throw std::runtime_error("Invalid acceptance label");
  for (int h=0;h<2;++h) {
    label(h==0?"hole0":"hole1");real(c.hole[h].mass);
    for (double &x:c.hole[h].center) real(x);
    for (double &x:c.hole[h].spin) real(x);
    for (double &x:c.hole[h].velocity) real(x);
  }
  label("n");for (int &x:c.n) integer(x);
  label("conformal_choice");integer(c.conformal_choice);
  label("inner_flatten");integer(c.inner_flatten);
  label("omega");for (double &x:c.omega) real(x);
  label("attenuation_power");integer(c.attenuation_power);
  label("inner_min");for (double &x:c.inner_min) real(x);
  label("inner_max");for (double &x:c.inner_max) real(x);
  label("far_radius");real(c.far_radius);
  label("tolerance");real(c.tolerance);
  label("max_newton");integer(c.max_newton);
  label("max_krylov");integer(c.max_krylov);
  label("krylov_restart");integer(c.krylov_restart);
  label("memory_limit_mib");integer(c.memory_limit_mib);
  std::size_t count=4;
  for (int n:c.n) {
    if (n<4 || n>256) throw std::runtime_error("Invalid checkpoint grid extent");
    count*=static_cast<std::size_t>(n);
  }
  label("unknowns");std::size_t saved_count;if (!(in>>saved_count) || saved_count!=count)
    throw std::runtime_error("Checkpoint unknown count mismatch");
  // Preserve the producer's requested budget. The linked native sampler
  // validates its own supported cap and allocation formula independently.
  if (c.memory_limit_mib<16 || c.memory_limit_mib>65536 ||
      double(count/4)*128>double(c.memory_limit_mib)*1024*1024)
    throw std::runtime_error("Checkpoint exceeds sampling-context allocation budget");
  result.unknowns.resize(count);for (double &x:result.unknowns) real(x);
  label("END");std::string trailing;if (in>>trailing) throw std::runtime_error("Extra checkpoint data");
  return result;
}
struct Destroy { void operator()(HiSpID_Data *data) const { HiSpID_destroy(data); } };
using Context=std::unique_ptr<HiSpID_Data,Destroy>;
}  // namespace hispid_import
#endif  // PGEN_Z4C_HISPID_CHECKPOINT_HPP_
