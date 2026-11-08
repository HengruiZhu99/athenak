// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_MAPPED_GEOMETRY_HPP_
#define Z4C_HYPERBOLOIDAL_MAPPED_GEOMETRY_HPP_
#include "mesh/chart_atlas.hpp"

// The native mesh atlas is the single geometry source. Physics only uses its
// Cartesian derivative jet algebra and the geometry attached to a native leaf.
namespace z4c_hyperboloidal::mapped {
using Point=chart_atlas::Point;
using Jet=chart_atlas::detail::Jet;
using chart_atlas::detail::Power;
using chart_atlas::detail::Atan;
}  // namespace z4c_hyperboloidal::mapped
#endif
