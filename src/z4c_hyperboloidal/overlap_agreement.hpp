// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_OVERLAP_AGREEMENT_HPP_
#define Z4C_HYPERBOLOIDAL_OVERLAP_AGREEMENT_HPP_
#include <array>
#include "bvals/mapped_exchange.hpp"
#include "z4c_hyperboloidal/mapped_halo.hpp"

namespace z4c_hyperboloidal {
class Z4cHyperboloidal;
struct AgreementParameters {
  Real kappa=0, inner=0, outer=0;
  bool full_overlap=false;
  Real taper_width=0.015;
};

// Rows are evaluated identically on every rank owning a transpose recipient.
// Only sparse donor values travel. Each recipient has a precomputed gather of
// transpose entries, so no reverse exchange, matrix assembly or atomics occurs.
class OverlapAgreementPlan {
 public:
  void Rebuild(MeshBlockPack *pack,const Z4cHyperboloidal &module);
  TaskStatus StartReceives() {return exchange_.StartReceives();}
  TaskStatus PackAndSend(const DvceArray5D<Real> &residual) {
    return exchange_.PackAndSend(residual);
  }
  TaskStatus Receive() {return exchange_.Receive();}
  TaskStatus ClearSend() {return exchange_.ClearSend();}
  TaskStatus ClearRecv() {return exchange_.ClearRecv();}
  void Evaluate();
  void AddRHS(const Z4cHyperboloidal &module) const;
  std::array<Real,8> History() const;
  Real RateBound() const {return rate_bound_;}
  const bvals::MappedExchange &Exchange() const {return exchange_;}

 private:
  friend struct OverlapAgreementAudit;
  struct Row {int begin,end;Real weight;bool owned;};
  struct Entry {int ordinal;Real coefficient;};
  struct Target {HaloPoint point;int begin,end;Real chi_weight;};
  struct Transpose {int row;Real coefficient;};
  bvals::MappedExchange exchange_;
  DvceArray1D<Row> rows_;
  DvceArray1D<Entry> entries_;
  DvceArray1D<Target> targets_;
  DvceArray1D<Transpose> transpose_;
  DvceArray1D<int> fields_;
  DvceArray2D<Real> discrepancy_;
  Real kappa_=0,rate_bound_=0;
};
}  // namespace z4c_hyperboloidal
#endif  // Z4C_HYPERBOLOIDAL_OVERLAP_AGREEMENT_HPP_
