#ifndef ATHENA_TENSOR_HPP_
#define ATHENA_TENSOR_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file athena_tensor.hpp
//  \brief provides classes for tensor-like fields
//
//  Convention: indices a,b,c,d are tensor indices. Indices n,i,j,k are grid indices.

#include <type_traits>
#include <cassert> // assert
#include <utility>
#include "athena.hpp"

// tensor symmetries
enum class TensorSymm {
  NONE,     // no symmetries
  SYM2,     // symmetric in the last 2 indices
  ISYM2,    // symmetric in the first 2 indices
  SYM22,    // symmetric in the last 2 pairs of indices
};


using sub_DvceArray5D_2D = decltype(Kokkos::subview(
                           std::declval<DvceArray5D<Real>>(),
                           Kokkos::ALL,std::make_pair(0,6),
                           Kokkos::ALL,Kokkos::ALL,Kokkos::ALL));
using sub_DvceArray5D_1D = decltype(Kokkos::subview(
                           std::declval<DvceArray5D<Real>>(),
                           Kokkos::ALL,std::make_pair(0,3),
                           Kokkos::ALL,Kokkos::ALL,Kokkos::ALL));
using sub_DvceArray5D_0D = decltype(Kokkos::subview(
                           std::declval<DvceArray5D<Real>>(),
                           Kokkos::ALL,1,
                           Kokkos::ALL,Kokkos::ALL,Kokkos::ALL));

using sub_HostArray5D_2D = decltype(Kokkos::subview(
                           std::declval<HostArray5D<Real>>(),
                           Kokkos::ALL,std::make_pair(0,6),
                           Kokkos::ALL,Kokkos::ALL,Kokkos::ALL));
using sub_HostArray5D_1D = decltype(Kokkos::subview(
                           std::declval<HostArray5D<Real>>(),
                           Kokkos::ALL,std::make_pair(0,3),
                           Kokkos::ALL,Kokkos::ALL,Kokkos::ALL));
using sub_HostArray5D_0D = decltype(Kokkos::subview(
                           std::declval<HostArray5D<Real>>(),
                           Kokkos::ALL,1,
                           Kokkos::ALL,Kokkos::ALL,Kokkos::ALL));

// this is the abstract base class
// Tensor index dimensions are compile-time parameters. Grid data are still stored in
// AthenaK's standard five-dimensional views.
template<typename T, TensorSymm sym, int ndim, int rank>
class AthenaHostTensor;

//----------------------------------------------------------------------------------------
// rank 0 AthenaHostTensor: 3D scalar fields
// This is simply a DvceArray3D
template<typename T, TensorSymm sym, int ndim>
class AthenaHostTensor<T, sym, ndim, 0> {
 public:
  // the default constructor/destructor/copy operators are sufficient
  AthenaHostTensor() = default;
  ~AthenaHostTensor() = default;
  AthenaHostTensor(AthenaHostTensor<T, sym, ndim, 0> const &) = default;
  AthenaHostTensor<T, sym, ndim, 0> & operator=
  (AthenaHostTensor<T, sym, ndim, 0> const &) = default;
  // operators to access the data
  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator() (int const m, int const k, int const j, int const i) const {
    return data_(m,k,j,i);
  }
  //KOKKOS_INLINE_FUNCTION
  void InitWithShallowSlice(HostArray5D<Real> src, const int indx) {
    data_ = Kokkos::subview(src,Kokkos::ALL,indx,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
  }

 private:
  sub_HostArray5D_0D data_;
};
//----------------------------------------------------------------------------------------
// rank 1 AthenaTensor, e.g., the lapse
template<typename T, TensorSymm sym, int ndim>
class AthenaHostTensor<T, sym, ndim, 1> {
 public:
  // the default constructor/destructor/copy operators are sufficient
  AthenaHostTensor() = default;
  ~AthenaHostTensor() = default;
  AthenaHostTensor(AthenaHostTensor<T, sym, ndim, 1> const &) = default;
  AthenaHostTensor<T, sym, ndim, 1> & operator =
  (AthenaHostTensor<T, sym, ndim, 1> const &) = default;
  // operators to access the data
  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator() (int const m, int const a,
                             int const k, int const j, int const i) const {
    return data_(m,a,k,j,i);
  }
  void InitWithShallowSlice(HostArray5D<Real> src, const int indx1, const int indx2) {
    data_ = Kokkos::subview(src, Kokkos::ALL, std::make_pair(indx1, indx2+1),
                                 Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  }

 private:
  sub_HostArray5D_1D data_;
};
//----------------------------------------------------------------------------------------
// rank 2 AthenaTensor, e.g., the metric or the extrinsic curvature
template<typename T, TensorSymm sym, int ndim>
class AthenaHostTensor<T, sym, ndim, 2> {
 public:
  AthenaHostTensor();
  // the default destructor/copy operators are sufficient
  ~AthenaHostTensor() = default;
  AthenaHostTensor(AthenaHostTensor<T, sym, ndim, 2> const &) = default;
  AthenaHostTensor<T, sym, ndim, 2> & operator=
  (AthenaHostTensor<T, sym, ndim, 2> const &) = default;

  int idxmap(int const a, int const b) const {
    return idxmap_[a][b];
  }
  // operators to access the data
  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator() (int const m, int const a, int const b,
                             int const k, int const j, int const i) const {
    return data_(m,idxmap_[a][b],k,j,i);
  }
  //KOKKOS_INLINE_FUNCTION
  void InitWithShallowSlice(HostArray5D<Real> src, const int indx1, const int indx2) {
    data_ = Kokkos::subview(src, Kokkos::ALL, std::make_pair(indx1, indx2+1),
                                 Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  }

 private:
  sub_HostArray5D_2D data_;
  int idxmap_[ndim][ndim];
  int ndof_;
};

//----------------------------------------------------------------------------------------
// Implementation details
// They are all duplicated to account for dim 0
template<typename T, TensorSymm sym, int ndim>
AthenaHostTensor<T, sym, ndim, 2>::AthenaHostTensor() {
  switch(sym) {
    case TensorSymm::NONE:
      ndof_ = 0;
      for(int a = 0; a < ndim; ++a)
      for(int b = 0; b < ndim; ++b) {
        idxmap_[a][b] = ndof_++;
      }
      break;
    case TensorSymm::SYM2:
    case TensorSymm::ISYM2:
      ndof_ = 0;
      for(int a = 0; a < ndim; ++a)
      for(int b = a; b < ndim; ++b) {
        idxmap_[a][b] = ndof_++;
        idxmap_[b][a] = idxmap_[a][b];
      }
      break;
  }
}


// this is the abstract base class
// Tensor index dimensions are compile-time parameters. Grid data are still stored in
// AthenaK's standard five-dimensional views.
template<typename T, TensorSymm sym, int ndim, int rank>
class AthenaTensor;

//----------------------------------------------------------------------------------------
// rank 0 AthenaTensor: 3D scalar fields
// This is simply a DvceArray3D
template<typename T, TensorSymm sym, int ndim>
class AthenaTensor<T, sym, ndim, 0> {
 public:
  // the default constructor/destructor/copy operators are sufficient
  AthenaTensor() = default;
  ~AthenaTensor() = default;
  AthenaTensor(AthenaTensor<T, sym, ndim, 0> const &) = default;
  AthenaTensor<T, sym, ndim, 0> & operator=
  (AthenaTensor<T, sym, ndim, 0> const &) = default;
  // operators to access the data
  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator() (int const m, int const k, int const j, int const i) const {
    return data_(m,k,j,i);
  }
  //KOKKOS_INLINE_FUNCTION
  void InitWithShallowSlice(DvceArray5D<Real> src, const int indx) {
    data_ = Kokkos::subview(src,Kokkos::ALL,indx,Kokkos::ALL,Kokkos::ALL,Kokkos::ALL);
  }
 private:
  sub_DvceArray5D_0D data_;
};

//----------------------------------------------------------------------------------------
// rank 1 AthenaTensor: 3D vector and co-vector fields
// This is a 4D AthenaTensor
template<typename T, TensorSymm sym, int ndim>
class AthenaTensor<T, sym, ndim, 1> {
 public:
  // the default constructor/destructor/copy operators are sufficient
  AthenaTensor() = default;
  ~AthenaTensor() = default;
  AthenaTensor(AthenaTensor<T, sym, ndim, 1> const &) = default;
  AthenaTensor<T, sym, ndim, 1> & operator=
  (AthenaTensor<T, sym, ndim, 1> const &) = default;
  // operators to access the data
  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator() (int const m, int const a,
                             int const k, int const j, int const i) const {
    return data_(m,a,k,j,i);
  }
  //KOKKOS_INLINE_FUNCTION
  void InitWithShallowSlice(DvceArray5D<Real> src, const int indx1, const int indx2) {
    data_ = Kokkos::subview(src, Kokkos::ALL, std::make_pair(indx1, indx2+1),
                                 Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  }
 private:
  sub_DvceArray5D_1D data_;
};

//----------------------------------------------------------------------------------------
// rank 2 AthenaTensor, e.g., the metric or the extrinsic curvature
template<typename T, TensorSymm sym, int ndim>
class AthenaTensor<T, sym, ndim, 2> {
 public:
  AthenaTensor();
  // the default destructor/copy operators are sufficient
  ~AthenaTensor() = default;
  AthenaTensor(AthenaTensor<T, sym, ndim, 2> const &) = default;
  AthenaTensor<T, sym, ndim, 2> & operator=
  (AthenaTensor<T, sym, ndim, 2> const &) = default;

  int idxmap(int const a, int const b) const {
    return idxmap_[a][b];
  }
  // operators to access the data
  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator() (int const m, int const a, int const b,
                             int const k, int const j, int const i) const {
    return data_(m,idxmap_[a][b],k,j,i);
  }
  //KOKKOS_INLINE_FUNCTION
  void InitWithShallowSlice(DvceArray5D<Real> src, const int indx1, const int indx2) {
    data_ = Kokkos::subview(src, Kokkos::ALL, std::make_pair(indx1, indx2+1),
                                 Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
  }

 private:
  sub_DvceArray5D_2D data_;
  int idxmap_[ndim][ndim];
  int ndof_;
};

//----------------------------------------------------------------------------------------
// Implementation details
// They are all duplicated to account for dim 0
template<typename T, TensorSymm sym, int ndim>
AthenaTensor<T, sym, ndim, 2>::AthenaTensor() {
  switch(sym) {
    case TensorSymm::NONE:
      ndof_ = 0;
      for(int a = 0; a < ndim; ++a)
      for(int b = 0; b < ndim; ++b) {
        idxmap_[a][b] = ndof_++;
      }
      break;
    case TensorSymm::SYM2:
    case TensorSymm::ISYM2:
      ndof_ = 0;
      for(int a = 0; a < ndim; ++a)
      for(int b = a; b < ndim; ++b) {
        idxmap_[a][b] = ndof_++;
        idxmap_[b][a] = idxmap_[a][b];
      }
      break;
  }
}


// Here tensors are defined as static 1D arrays, with compile-time dimension calculated as
// dim**rank
// this is the abstract base class
template<typename T, TensorSymm sym, int ndim, int rank>
class AthenaPointTensor;

//----------------------------------------------------------------------------------------
// rank 1 AthenaPointTensor: spatially 0D vector and co-vector fields
// This is a 1D AthenaPointTensor
template<typename T, TensorSymm sym, int ndim>
class AthenaPointTensor<T, sym, ndim, 1> {
 public:
  // the default constructor/destructor/copy operators are sufficient
  AthenaPointTensor() = default;
  ~AthenaPointTensor() = default;
  AthenaPointTensor(AthenaPointTensor<T, sym, ndim, 1> const &) = default;
  AthenaPointTensor<T, sym, ndim, 1> & operator=
  (AthenaPointTensor<T, sym, ndim, 1> const &) = default;

  KOKKOS_INLINE_FUNCTION
  T operator()(int const a) const {
    return data_[a];
  }
  KOKKOS_INLINE_FUNCTION
  T & operator()(int const a) {
    return data_[a];
  }
  KOKKOS_INLINE_FUNCTION
  void ZeroClear() {
    for (int i = 0; i < ndim; ++i) {
      data_[i] = 0;
    }
  }

 private:
  T data_[ndim]; // NOLINT(runtime/arrays)
};

//----------------------------------------------------------------------------------------
// Tensor degrees of freedom, base case assumes zero
template<TensorSymm sym, int ndim, int rank>
constexpr int TensorDOF = -1;

//----------------------------------------------------------------------------------------
// Rank 2 tensor degrees of freedom
template<int ndim>
constexpr int TensorDOF<TensorSymm::NONE, ndim, 2> = ndim*ndim;

template<int ndim>
constexpr int TensorDOF<TensorSymm::SYM2, ndim, 2> = ndim*(ndim+1)/2;

//----------------------------------------------------------------------------------------
// Rank 3 tensor degrees of freedom
template<int ndim>
constexpr int TensorDOF<TensorSymm::NONE, ndim, 3> = ndim*ndim*ndim;

template<int ndim>
constexpr int TensorDOF<TensorSymm::SYM2, ndim, 3> = ndim*ndim*(ndim+1)/2;

template<int ndim>
constexpr int TensorDOF<TensorSymm::ISYM2, ndim, 3> = ndim*ndim*(ndim+1)/2;

//----------------------------------------------------------------------------------------
// Rank 4 tensor degrees of freedom
template<int ndim>
constexpr int TensorDOF<TensorSymm::NONE, ndim, 4> = ndim*ndim*ndim*ndim;

template<int ndim>
constexpr int TensorDOF<TensorSymm::SYM2, ndim, 4> = ndim*ndim*ndim*(ndim+1)/2;

template<int ndim>
constexpr int TensorDOF<TensorSymm::ISYM2, ndim, 4> = ndim*ndim*ndim*(ndim+1)/2;

template<int ndim>
constexpr int TensorDOF<TensorSymm::SYM22, ndim, 4> = ndim*ndim*(ndim+1)*(ndim+1)/4;

//----------------------------------------------------------------------------------------
//! \class AthenaPointMixedTensor
//! \brief Point tensor with one index dimension independent of a symmetric index pair.
//!
//! This is used, for example, by first-order generalized harmonic evolution for
//! Phi_iab: i has spatial dimension 3 while a,b have spacetime dimension 4. Storage is
//! compile-time sized, contains no mapping table or allocation, and is symmetric only in
//! the final pair.
template<typename T, int outer_dim, int pair_dim>
class AthenaPointMixedTensor {
 public:
  static_assert(outer_dim > 0, "Mixed tensor outer dimension must be positive.");
  static_assert(pair_dim > 0, "Mixed tensor pair dimension must be positive.");

  static constexpr int ndof = outer_dim*pair_dim*(pair_dim + 1)/2;

  KOKKOS_INLINE_FUNCTION
  AthenaPointMixedTensor() = default;
  ~AthenaPointMixedTensor() = default;
  AthenaPointMixedTensor(AthenaPointMixedTensor const &) = default;
  AthenaPointMixedTensor & operator=(AthenaPointMixedTensor const &) = default;

  KOKKOS_INLINE_FUNCTION
  T operator()(int const i, int const a, int const b) const {
    return data_[Index(i, a, b)];
  }

  KOKKOS_INLINE_FUNCTION
  T & operator()(int const i, int const a, int const b) {
    return data_[Index(i, a, b)];
  }

  KOKKOS_INLINE_FUNCTION
  void ZeroClear() {
    for (int n = 0; n < ndof; ++n) {
      data_[n] = T{};
    }
  }

 private:
  KOKKOS_INLINE_FUNCTION
  static constexpr int Index(int const i, int const a, int const b) {
    const int lo = (a < b ? a : b);
    const int hi = (a < b ? b : a);
    return i*(pair_dim*(pair_dim + 1)/2)
           + lo*(2*pair_dim - lo + 1)/2 + hi - lo;
  }

  T data_[ndof]; // NOLINT
};

// Concise generic name for code that does not need to distinguish storage location.
template<typename T, int outer_dim, int pair_dim>
using MixedTensor = AthenaPointMixedTensor<T, outer_dim, pair_dim>;


//----------------------------------------------------------------------------------------
// rank 2 AthenaPointTensor
// This is a 0D AthenaPointTensor
template<typename T, TensorSymm sym, int ndim>
class AthenaPointTensor<T, sym, ndim, 2> {
 public:
  KOKKOS_INLINE_FUNCTION
  AthenaPointTensor() = default;
  // the default destructor/copy operators are sufficient
  ~AthenaPointTensor() = default;
  AthenaPointTensor(AthenaPointTensor<T, sym, ndim, 2> const &) = default;
  AthenaPointTensor<T, sym, ndim, 2> & operator=
  (AthenaPointTensor<T, sym, ndim, 2> const &) = default;
  KOKKOS_INLINE_FUNCTION
  Real operator()(int const a, int const b) const {
    if constexpr (sym == TensorSymm::NONE) {
      return data_[b + ndim*a];
    } else if (sym == TensorSymm::SYM2) {
      if (b < a) {
        return data_[b*(2*ndim - b + 1)/2+a-b];
      } else {
        return data_[a*(2*ndim - a + 1)/2+b-a];
      }
    } else {
      static_assert(AthenaPointTensor<T, sym, ndim, 2>::allowed_sym,
                    "Undefined symmetry for rank-2 AthenaPointTensor.");
      return data_[0];
    }
    //return data_[idxmap_[a][b]];
  }
  KOKKOS_INLINE_FUNCTION
  Real & operator()(int const a, int const b) {
    if constexpr (sym == TensorSymm::NONE) {
      return data_[b + ndim*a];
    } else if (sym == TensorSymm::SYM2) {
      if (b < a) {
        return data_[b*(2*ndim - b + 1)/2+a-b];
      } else {
        return data_[a*(2*ndim - a + 1)/2+b-a];
      }
    } else {
      static_assert(AthenaPointTensor<T, sym, ndim, 2>::allowed_sym,
                    "Undefined symmetry for rank-2 AthenaPointTensor.");
      return data_[0];
    }
    //return data_[idxmap_[a][b]];
  }
  KOKKOS_INLINE_FUNCTION
  void ZeroClear() {
    for (int i = 0; i < TensorDOF<sym, ndim, 2>; ++i) {
      data_[i] = 0.0;
    }
  }

 private:
  Real data_[TensorDOF<sym, ndim, 2>]; // NOLINT
  static constexpr bool allowed_sym = (sym == TensorSymm::NONE ||
                                       sym == TensorSymm::SYM2);
};

//----------------------------------------------------------------------------------------
// rank 3 AthenaPointTensor
// This is a 0D AthenaPointTensor
template<typename T, TensorSymm sym, int ndim>
class AthenaPointTensor<T, sym, ndim, 3> {
 public:
  KOKKOS_INLINE_FUNCTION
  AthenaPointTensor() = default;
  // the default destructor/copy operators are sufficient
  ~AthenaPointTensor() = default;
  AthenaPointTensor(AthenaPointTensor<T, sym, ndim, 3> const &) = default;
  AthenaPointTensor<T, sym, ndim, 3> & operator=
  (AthenaPointTensor<T, sym, ndim, 3> const &) = default;
  KOKKOS_INLINE_FUNCTION
  Real operator()(int const a, int const b, int const c) const {
    if constexpr (sym == TensorSymm::NONE) {
      return data_[c + ndim*(b + ndim*a)];
    } else if (sym == TensorSymm::SYM2) {
      constexpr int ndof2_ = TensorDOF<TensorSymm::SYM2, ndim, 2>;
      if (c < b) {
        return data_[c*(2*ndim - c + 1)/2 + b - c + ndof2_*a];
      } else {
        return data_[b*(2*ndim - b + 1)/2 + c - b + ndof2_*a];
      }
    } else if (sym == TensorSymm::ISYM2) {
      if (b < a) {
        return data_[c + ndim*(b*(2*ndim - b + 1)/2 + a - b)];
      } else {
        return data_[c + ndim*(a*(2*ndim - a + 1)/2 + b - a)];
      }
    } else {
      static_assert(AthenaPointTensor<T, sym, ndim, 3>::allowed_sym,
                    "Undefined symmetry for rank-3 AthenaPointTensor.");
      return data_[0];
    }
  }
  KOKKOS_INLINE_FUNCTION
  Real & operator()(int const a, int const b, int const c) {
    if constexpr (sym == TensorSymm::NONE) {
      return data_[c + ndim*(b + ndim*a)];
    } else if (sym == TensorSymm::SYM2) {
      constexpr int ndof2_ = TensorDOF<TensorSymm::SYM2, ndim, 2>;
      if (c < b) {
        return data_[c*(2*ndim - c + 1)/2 + b - c + ndof2_*a];
      } else {
        return data_[b*(2*ndim - b + 1)/2 + c - b + ndof2_*a];
      }
    } else if (sym == TensorSymm::ISYM2) {
      if (b < a) {
        return data_[c + ndim*(b*(2*ndim - b + 1)/2 + a - b)];
      } else {
        return data_[c + ndim*(a*(2*ndim - a + 1)/2 + b - a)];
      }
    } else {
      static_assert(AthenaPointTensor<T, sym, ndim, 3>::allowed_sym,
                    "Undefined symmetry for rank-3 AthenaPointTensor.");
      return data_[0];
    }
  }
  KOKKOS_INLINE_FUNCTION
  void ZeroClear() {
    for (int i = 0; i < TensorDOF<sym,ndim,3>; ++i) {
      data_[i] = 0.0;
    }
  }

 private:
  Real data_[TensorDOF<sym,ndim,3>];
  static constexpr bool allowed_sym = (sym == TensorSymm::NONE ||
                                       sym == TensorSymm::SYM2 ||
                                       sym == TensorSymm::ISYM2);
};

//----------------------------------------------------------------------------------------
// rank 4 AthenaPointTensor
// This is a 0D AthenaPointTensor
template<typename T, TensorSymm sym, int ndim>
class AthenaPointTensor<T, sym, ndim, 4> {
 public:
  KOKKOS_INLINE_FUNCTION
  AthenaPointTensor() {
    switch(sym) {
      case TensorSymm::NONE:
        ndof_ = ndim*ndim*ndim*ndim;
        break;
      case TensorSymm::SYM22:
        ndof_ = (ndim + 1)*ndim/2 * (ndim + 1)*ndim/2;
        break;
    }
  }
  // the default destructor/copy operators are sufficient
  ~AthenaPointTensor() = default;
  AthenaPointTensor(AthenaPointTensor<T, sym, ndim, 4> const &) = default;
  AthenaPointTensor<T, sym, ndim, 4> & operator=
  (AthenaPointTensor<T, sym, ndim, 4> const &) = default;

  KOKKOS_INLINE_FUNCTION
  Real operator()(int a, int b, int c, int d) const {
    if constexpr (sym == TensorSymm::NONE) {
      return data_[ndim * ndim * ndim * a + ndim * ndim * b + ndim * c + d];
    } else if constexpr (sym == TensorSymm::SYM22) {
      constexpr int ndof2_ = TensorDOF<TensorSymm::SYM2, ndim, 2>;
      if (a < b) {
        Kokkos::kokkos_swap(a, b);
      }
      if (c < d) {
        Kokkos::kokkos_swap(c, d);
      }
      return data_[(b*( 2*ndim - b +1)/2 + a - b)*ndof2_ + d*( 2*ndim - d +1)/2 + c - d];
    } else {
      static_assert(AthenaPointTensor<T, sym, ndim, 4>::allowed_sym,
                    "Undefined symmetry for rank-4 AthenaPointTensor.");
      return data_[0];
    }
  }


  KOKKOS_INLINE_FUNCTION
  Real & operator()(int a, int b, int c, int d) {
    if constexpr (sym == TensorSymm::NONE) {
      return data_[ndim * ndim * ndim * a + ndim * ndim * b + ndim * c + d];
    } else if constexpr (sym == TensorSymm::SYM22) {
      constexpr int ndof2_ = TensorDOF<TensorSymm::SYM2, ndim, 2>;
      if (a < b) {
        Kokkos::kokkos_swap(a, b);
      }
      if (c < d) {
        Kokkos::kokkos_swap(c, d);
      }
      return data_[(b*( 2*ndim - b +1)/2 + a - b)*ndof2_ + d*( 2*ndim - d +1)/2 + c - d];
    } else {
      static_assert(AthenaPointTensor<T, sym, ndim, 4>::allowed_sym,
                    "Undefined symmetry for rank-4 AthenaPointTensor.");
      return data_[0];
    }
  }

  KOKKOS_INLINE_FUNCTION
  void ZeroClear() {
    for (int i = 0; i < TensorDOF<sym,ndim,4>; ++i) {
      data_[i] = 0.0;
    }
  }

 private:
  Real data_[TensorDOF<sym,ndim,4>];
  int ndof_;
  static constexpr bool allowed_sym = (sym == TensorSymm::NONE ||
                                       sym == TensorSymm::SYM22);
};

// Here tensors are defined as static 1D arrays, with compile-time dimension calculated as
// dim**rank
// this is the abstract base class
template<typename T, TensorSymm sym, int ndim, int rank>
class AthenaScratchTensor;

//----------------------------------------------------------------------------------------
// rank 0 AthenaScratchTensor: spatially 0D vector and co-vector fields
// This is a 1D AthenaScratchTensor
template<typename T, TensorSymm sym, int ndim>
class AthenaScratchTensor<T, sym, ndim, 0> {
 public:
  // the default constructor/destructor/copy operators are sufficient
  AthenaScratchTensor() = default;
  ~AthenaScratchTensor() = default;
  AthenaScratchTensor(AthenaScratchTensor<T, sym, ndim, 0> const &) = default;
  AthenaScratchTensor<T, sym, ndim, 0> & operator=
  (AthenaScratchTensor<T, sym, ndim, 0> const &) = default;

  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(int const i) const {
    return data_(i);
  }
  KOKKOS_INLINE_FUNCTION
  void NewAthenaScratchTensor(const TeamMember_t &member, int scr_level, int nx) {
    data_ = ScrArray1D<T>(member.team_scratch(scr_level), nx);
  }
  KOKKOS_INLINE_FUNCTION
  void ZeroClear() {
    Kokkos::Experimental::local_deep_copy(data_, 0.);
  }
 private:
  ScrArray1D<T> data_;
};

//----------------------------------------------------------------------------------------
// rank 1 AthenaScratchTensor: spatially 0D vector and co-vector fields
// This is a 1D AthenaScratchTensor
template<typename T, TensorSymm sym, int ndim>
class AthenaScratchTensor<T, sym, ndim, 1> {
 public:
  // the default constructor/destructor/copy operators are sufficient
  AthenaScratchTensor() = default;
  ~AthenaScratchTensor() = default;
  AthenaScratchTensor(AthenaScratchTensor<T, sym, ndim, 1> const &) = default;
  AthenaScratchTensor<T, sym, ndim, 1> & operator=
  (AthenaScratchTensor<T, sym, ndim, 1> const &) = default;

  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(int const a, int const i) const {
    return data_(a, i);
  }
  KOKKOS_INLINE_FUNCTION
  void NewAthenaScratchTensor(const TeamMember_t & member, int scr_level, int nx) {
    data_ = ScrArray2D<T>(member.team_scratch(scr_level), ndim, nx);
  }
  KOKKOS_INLINE_FUNCTION
  void ZeroClear() {
    Kokkos::Experimental::local_deep_copy(data_, 0.);
  }
 private:
  ScrArray2D<T> data_;
};

//----------------------------------------------------------------------------------------
// rank 2 AthenaScratchTensor
// This is a 1D AthenaScratchTensor
template<typename T, TensorSymm sym, int ndim>
class AthenaScratchTensor<T, sym, ndim, 2> {
 public:
  KOKKOS_INLINE_FUNCTION
  AthenaScratchTensor() {
    switch(sym) {
      case TensorSymm::NONE:
        ndof_ = ndim * ndim;
        break;
      case TensorSymm::SYM2:
      case TensorSymm::ISYM2:
        ndof_ = (ndim + 1)*ndim/2;
        break;
    }
  }
  // the default destructor/copy operators are sufficient
  ~AthenaScratchTensor() = default;
  AthenaScratchTensor(AthenaScratchTensor<T, sym, ndim, 2> const &) = default;
  AthenaScratchTensor<T, sym, ndim, 2> & operator=
  (AthenaScratchTensor<T, sym, ndim, 2> const &) = default;

  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(int a, int b, int i) const {
    if constexpr (sym == TensorSymm::NONE) {
      return data_(ndim * a + b, i);
    } else if (sym == TensorSymm::SYM2) {
      if (a < b) {
        Kokkos::kokkos_swap(a, b);
      }
      return data_(b*( 2*ndim - b +1)/2 + a - b, i);
    } else {
      static_assert(AthenaScratchTensor<T, sym, ndim, 2>::allowed_sym,
                    "Undefined symmetry for rank-2 AthenaScratchTensor.");
      return data_[0];
    }
  }
  KOKKOS_INLINE_FUNCTION
  void NewAthenaScratchTensor(const TeamMember_t & member, int scr_level, int nx) {
    data_ = ScrArray2D<T>(member.team_scratch(scr_level), ndof_, nx);
  }
  KOKKOS_INLINE_FUNCTION
  void ZeroClear() {
    Kokkos::Experimental::local_deep_copy(data_, 0);
  }

 private:
  ScrArray2D<T> data_;
  int ndof_;
  static constexpr bool allowed_sym = (sym == TensorSymm::NONE ||
                                       sym == TensorSymm::SYM2);
};

//----------------------------------------------------------------------------------------
// rank 3 AthenaScratchTensor
// This is a 1D AthenaScratchTensor
template<typename T, TensorSymm sym, int ndim>
class AthenaScratchTensor<T, sym, ndim, 3> {
 public:
  KOKKOS_INLINE_FUNCTION
  AthenaScratchTensor() {
    switch(sym) {
      case TensorSymm::NONE:
        ndof_ = ndim * ndim * ndim;
        break;
      case TensorSymm::SYM2:
      case TensorSymm::ISYM2:
        ndof_ = ndim * (ndim + 1)*ndim/2;
        break;
    }
  }
  // the default destructor/copy operators are sufficient
  ~AthenaScratchTensor() = default;
  AthenaScratchTensor(AthenaScratchTensor<T, sym, ndim, 3> const &) = default;
  AthenaScratchTensor<T, sym, ndim, 3> & operator=
  (AthenaScratchTensor<T, sym, ndim, 3> const &) = default;

  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(int a, int b, int c, int const i) const {
    if constexpr (sym == TensorSymm::NONE) {
      return data_(ndim * ndim * a + ndim * b + c, i);
    } else if constexpr (sym == TensorSymm::SYM2) {
      if (b < c) {
        Kokkos::kokkos_swap(b, c);
      }
      return data_(a*(ndim + 1)*ndim/2 + c*( 2*ndim - c +1)/2 + b - c,i);
    } else if constexpr (sym == TensorSymm::ISYM2) {
      if (a < b) {
        Kokkos::kokkos_swap(a, b);
      }
      return data_((b*(2*ndim - b +1)/2 + a - b)*ndim + c,i);
    } else {
      static_assert(AthenaScratchTensor<T, sym, ndim, 3>::allowed_sym,
                    "Undefined symmetry for rank-3 AthenaScratchTensor.");
      return data_[0];
    }
  }
  KOKKOS_INLINE_FUNCTION
  void NewAthenaScratchTensor(const TeamMember_t & member, int scr_level, int nx) {
    data_ = ScrArray2D<T>(member.team_scratch(scr_level), ndof_, nx);
  }
  KOKKOS_INLINE_FUNCTION
  void ZeroClear() {
    Kokkos::Experimental::local_deep_copy(data_, 0);
  }

 private:
  ScrArray2D<T> data_;
  int ndof_;
  static constexpr bool allowed_sym = (sym == TensorSymm::NONE ||
                                sym == TensorSymm::SYM2 || sym == TensorSymm::ISYM2);
};

//----------------------------------------------------------------------------------------
// rank 4 AthenaScratchTensor
// This is a 1D AthenaScratchTensor
template<typename T, TensorSymm sym, int ndim>
class AthenaScratchTensor<T, sym, ndim, 4> {
 public:
  KOKKOS_INLINE_FUNCTION
  AthenaScratchTensor() {
    switch(sym) {
      case TensorSymm::NONE:
        ndof_ = ndim*ndim*ndim*ndim;
        break;
      case TensorSymm::SYM22:
        ndof_ = (ndim + 1)*ndim/2 * (ndim + 1)*ndim/2;
        break;
    }
  }
  // the default destructor/copy operators are sufficient
  ~AthenaScratchTensor() = default;
  AthenaScratchTensor(AthenaScratchTensor<T, sym, ndim, 4> const &) = default;
  AthenaScratchTensor<T, sym, ndim, 4> & operator=
  (AthenaScratchTensor<T, sym, ndim, 4> const &) = default;

  KOKKOS_INLINE_FUNCTION
  decltype(auto) operator()(int a, int b,
                            int c, int d, int const i) const {
    if constexpr (sym == TensorSymm::NONE) {
      return data_(ndim * ndim * ndim * a + ndim * ndim * b + ndim * c + d, i);
    } else if constexpr (sym == TensorSymm::SYM22) {
      if (a < b) {
        Kokkos::kokkos_swap(a, b);
      }
      if (c < d) {
        Kokkos::kokkos_swap(c, d);
      }
      return data_((b*( 2*ndim - b +1)/2 + a - b)*(ndim + 1)*ndim/2 +
                    d*( 2*ndim - d +1)/2 + c - d,i);
    } else {
      static_assert(AthenaScratchTensor<T, sym, ndim, 4>::allowed_sym,
                    "Undefined symmetry for rank-4 AthenaScratchTensor.");
      return data_[0];
    }
  }

  KOKKOS_INLINE_FUNCTION
  void NewAthenaScratchTensor(const TeamMember_t & member, int scr_level, int nx) {
    data_ = ScrArray2D<T>(member.team_scratch(scr_level), ndof_, nx);
  }
  KOKKOS_INLINE_FUNCTION
  void ZeroClear() {
    Kokkos::Experimental::local_deep_copy(data_, 0);
  }

 private:
  ScrArray2D<T> data_;
  int ndof_;
  static constexpr bool allowed_sym = (sym == TensorSymm::NONE ||
                                       sym == TensorSymm::SYM22);
};

#endif // ATHENA_TENSOR_HPP_
