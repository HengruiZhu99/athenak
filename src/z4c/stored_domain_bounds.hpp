//========================================================================================
// AthenaK astrophysical fluid dynamics & numerical relativity code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the AthenaK collaboration
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file stored_domain_bounds.hpp
//! \brief Trivially copyable bounds for allocated cell-centered storage.

#ifndef Z4C_STORED_DOMAIN_BOUNDS_HPP_
#define Z4C_STORED_DOMAIN_BOUNDS_HPP_

#include <type_traits>

namespace z4c {

//! Inclusive bounds and extents of storage that actually exists on a MeshBlock.
//!
//! Active dimensions include their ghost zones. Collapsed dimensions contain exactly
//! the active singleton and never fabricate ghost planes that were not allocated.
struct StoredDomainBounds {
  int is, ie;
  int js, je;
  int ks, ke;
  int n1, n2, n3;
};

static_assert(std::is_trivially_copyable_v<StoredDomainBounds>);

template <typename RegionIndices>
constexpr StoredDomainBounds MakeStoredDomainBounds(const RegionIndices &indcs) {
  const int is = indcs.is - indcs.ng;
  const int ie = indcs.ie + indcs.ng;
  const int js = indcs.nx2 > 1 ? indcs.js - indcs.ng : indcs.js;
  const int je = indcs.nx2 > 1 ? indcs.je + indcs.ng : indcs.je;
  const int ks = indcs.nx3 > 1 ? indcs.ks - indcs.ng : indcs.ks;
  const int ke = indcs.nx3 > 1 ? indcs.ke + indcs.ng : indcs.ke;
  return {is, ie, js, je, ks, ke, ie - is + 1, je - js + 1, ke - ks + 1};
}

template <typename RegionIndices>
constexpr StoredDomainBounds MakeCoarseStoredDomainBounds(
    const RegionIndices &indcs) {
  const int is = indcs.cis - indcs.ng;
  const int ie = indcs.cie + indcs.ng;
  const int js = indcs.cnx2 > 1 ? indcs.cjs - indcs.ng : indcs.cjs;
  const int je = indcs.cnx2 > 1 ? indcs.cje + indcs.ng : indcs.cje;
  const int ks = indcs.cnx3 > 1 ? indcs.cks - indcs.ng : indcs.cks;
  const int ke = indcs.cnx3 > 1 ? indcs.cke + indcs.ng : indcs.cke;
  return {is, ie, js, je, ks, ke, ie - is + 1, je - js + 1, ke - ks + 1};
}

}  // namespace z4c

#endif  // Z4C_STORED_DOMAIN_BOUNDS_HPP_
