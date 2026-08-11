#ifndef PGEN_PGEN_DEFAULTS_HPP_
#define PGEN_PGEN_DEFAULTS_HPP_
//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file pgen_defaults.hpp
//! \brief Input-selected pgen defaults retained for historical custom builds.

#include <string_view>

inline const char *DefaultInputSelectedPgen(
    const std::string_view compiled_problem) noexcept {
  return compiled_problem == "z4c_irisk_xcts" ? "z4c_irisk_xcts" : "none";
}

#endif  // PGEN_PGEN_DEFAULTS_HPP_
