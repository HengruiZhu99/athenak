// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_OVERLAP_AGREEMENT_HPP_
#define Z4C_HYPERBOLOIDAL_OVERLAP_AGREEMENT_HPP_
#include <array>
#include <map>
#include <functional>
#include <set>
#include "bvals/mapped_exchange.hpp"
#include "z4c_hyperboloidal/mapped_halo.hpp"

namespace z4c_hyperboloidal {
class Z4cHyperboloidal;
struct AgreementParameters {
  Real kappa=1, inner=0;
  Real taper_width=0.015;
};

// Stream the same constructed discrepancy stencils used by Rebuild. A mesh audit
// can visit them without allocating field storage or all transpose entries.
using AgreementStencilVisitor=std::function<void(const std::vector<WeightedDonor>&,
    const bvals::DonorPoint&,Real taper,Real radius,bool cartesian)>;
std::array<Real,6> VisitAgreementStencils(const Mesh &,const NativeDonorSelector &,
    const AgreementParameters &,const AgreementStencilVisitor &);

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
  struct BatchRows {
    DvceArray1D<int> rows, targets;
  };
  // The caller only caches complete local stage groups with fixed membership.
  const BatchRows &RowsForBatch(const std::vector<int> &slots, const DvceArray1D<int> &active);
  void Evaluate(bvals::MappedExchange::ConstValues values={}, const BatchRows *batch=nullptr);
  const std::vector<std::pair<int,int>> &DependencyEdges() const {return dependency_edges_;}
  using DonorsByRecipient=std::map<int,std::set<bvals::DonorPoint>>;
  const DonorsByRecipient &MultirateDonors() const { return multirate_donors_; }
  const std::vector<int> &AgreementBlocks() const {return agreement_blocks_;}
  void AddRHS(const Z4cHyperboloidal &module, const BatchRows *batch=nullptr) const;
  std::array<Real,8> History() const;
  Real RateBound() const {return rate_bound_;}
  const bvals::MappedExchange &Exchange() const {return exchange_;}

 private:
  friend struct OverlapAgreementAudit;
  std::map<std::vector<int>,BatchRows> batches_;
  std::vector<std::pair<int,int>> dependency_edges_;
  std::vector<int> agreement_blocks_;
  DonorsByRecipient multirate_donors_;
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
