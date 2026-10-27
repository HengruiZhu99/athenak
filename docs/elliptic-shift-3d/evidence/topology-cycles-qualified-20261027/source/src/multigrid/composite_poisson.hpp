// Copyright(C) 2014 James M. Stone and Athena code contributors.
// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef MULTIGRID_COMPOSITE_POISSON_HPP_
#define MULTIGRID_COMPOSITE_POISSON_HPP_

#include <memory>
#include "mesh/cubed_sphere/composite_difference.hpp"
#include "multigrid/composite_radial_line.hpp"
#include "multigrid/composite_topology_transfer.hpp"

namespace multigrid {

// CPU reference hierarchy for mapped-patch correction equations. The 2^3
// restriction / 3:1 trilinear correction stencils follow AthenaK's Multigrid
// Restrict/ProlongateAndCorrect, with mapped-volume restriction replacing the
// equal-volume average. Every level has its own GLOBAL active-only donor plan.
// The optional fourth constructor control instead uses general nested logical
// transfers (including odd N) and complete Cartesian sibling coarsening. It is
// a lower-order preconditioner; the fine operator is identical on both paths.
// This backend is deliberately separate from the Cartesian-octree MG driver:
// assigning shell blocks fictitious Cartesian neighbors would be incorrect.
class CompositePoisson {
 public:
  explicit CompositePoisson(cubed_sphere::CompositeMesh mesh,bool radial_stencils=false,bool radial_lines=false,bool topology=false)
      : grid_(mesh, 2,radial_stencils) {
    InitializeSmoother(radial_lines);
    if(topology){
      const auto plan=CompositeCoarseningPlan::Build(mesh);
      if(!plan.dense_supported)throw std::runtime_error(plan.unsupported_reason);
      topology_=true;
      if(plan.levels.size()>1){
        transfer_=std::make_unique<CompositeLevelTransfer>(mesh,plan.levels[1],grid_.volumes);
        coarse_.reset(new CompositePoisson(plan,1,radial_stencils,radial_lines,transfer_->coarse_volume));
      }else FactorCoarse();
      return;
    }
    if (mesh.cells >= 4 && mesh.cells%2 == 0) {
      mesh.cells /= 2;
      coarse_ = std::make_unique<CompositePoisson>(std::move(mesh),radial_stencils,radial_lines);
    } else FactorCoarse();
  }
  const cubed_sphere::CompositeDifference &Grid() const {return grid_;}
  int Levels() const {return coarse_ ? 1+coarse_->Levels() : 1;}
  size_t CoarseCells() const {return coarse_ ? coarse_->CoarseCells() : grid_.active.size();}
  bool UsesRadialLines()const{return bool(line_);}
  bool UsesTopologyCoarsening()const{return topology_;}
  std::vector<size_t> HierarchyCells()const{auto cells=coarse_?coarse_->HierarchyCells():std::vector<size_t>{};cells.insert(cells.begin(),grid_.active.size());return cells;}
  std::vector<size_t> HierarchyHaloRows()const{auto rows=coarse_?coarse_->HierarchyHaloRows():std::vector<size_t>{};rows.insert(rows.begin(),grid_.halos.size());return rows;}
  CompositePoisson(const CompositePoisson&)=delete;
  CompositePoisson& operator=(const CompositePoisson&)=delete;
  CompositePoisson(CompositePoisson&&)=delete;
  CompositePoisson& operator=(CompositePoisson&&)=delete;

  void Operator(const std::vector<Real> &u, std::vector<Real> &out) const {
    if (u.size() != grid_.active.size()) throw std::invalid_argument("Poisson active shape");
    std::vector<Real> field(grid_.count, 0);
    for (size_t p = 0; p < u.size(); ++p) field[grid_.active[p]] = u[p];
    grid_.Fill(field, 1);
    out.resize(u.size());
    for (size_t p = 0; p < u.size(); ++p) {
      Real d[3], dd[3][3];
      grid_.Derivatives(field, 0, p, d, dd);
      out[p] = -(dd[0][0]+dd[1][1]+dd[2][2]);
    }
  }
  std::vector<Real> Cycle(const std::vector<Real> &rhs) const {
    if(rhs.size()!=grid_.active.size())throw std::invalid_argument("Poisson cycle rhs shape");
    if (!coarse_) return CoarseSolve(rhs);
    std::vector<Real> u(rhs.size(), 0), au;
    Smooth(rhs, u, 3);
    Operator(u, au);
    if(transfer_){
      std::vector<Real> residual(rhs.size());for(size_t p=0;p<rhs.size();++p)residual[p]=rhs[p]-au[p];
      const auto correction=transfer_->Prolong(coarse_->Cycle(transfer_->Restrict(residual)));
      for(size_t p=0;p<u.size();++p)u[p]+=correction[p];
      Smooth(rhs,u,3);return u;
    }
    const int nc = coarse_->grid_.n, nf = grid_.n;
    std::vector<Real> restricted(coarse_->grid_.active.size(), 0);
    for (size_t p = 0; p < restricted.size(); ++p) {
      const int m = p/(nc*nc*nc), i = p%nc, j = p/nc%nc, k = p/(nc*nc)%nc;
      Real volume = 0;
      for (int dk = 0; dk < 2; ++dk)
      for (int dj = 0; dj < 2; ++dj)
      for (int di = 0; di < 2; ++di) {
        const size_t f = ((m*nf+2*k+dk)*nf+2*j+dj)*nf+2*i+di;
        restricted[p] += grid_.volumes[f]*(rhs[f]-au[f]);
        volume += grid_.volumes[f];
      }
      restricted[p] /= volume;
    }
    const auto correction = coarse_->Cycle(restricted);
    const auto &cg = coarse_->grid_;
    std::vector<Real> field(cg.count, 0);
    for (size_t p = 0; p < correction.size(); ++p) field[cg.active[p]] = correction[p];
    cg.Fill(field, 1);
    for (size_t p = 0; p < u.size(); ++p) {
      const int m = p/(nf*nf*nf), indices[3] = {int(p%nf), int(p/nf%nf), int(p/(nf*nf)%nf)};
      int first[3]; Real fraction[3];
      for (int a = 0; a < 3; ++a) {
        const Real q = (indices[a]+.5)/2-.5;
        first[a] = static_cast<int>(std::floor(q)); fraction[a] = q-first[a];
      }
      for (int k = 0; k < 2; ++k)
      for (int j = 0; j < 2; ++j)
      for (int i = 0; i < 2; ++i)
        u[p] += (i ? fraction[0] : 1-fraction[0])
              *(j ? fraction[1] : 1-fraction[1])*(k ? fraction[2] : 1-fraction[2])
              *field[cg.Index(m, first[0]+i, first[1]+j, first[2]+k)];
    }
    Smooth(rhs, u, 3);
    return u;
  }

 private:
  cubed_sphere::CompositeDifference grid_;
  std::vector<Real> diagonal_, lu_;
  std::vector<size_t> pivots_;
  std::unique_ptr<CompositePoisson> coarse_;
  std::unique_ptr<CompositeRadialLine> line_;
  std::unique_ptr<CompositeLevelTransfer> transfer_;
  bool topology_=false;

  CompositePoisson(const CompositeCoarseningPlan &plan,size_t level,bool radial_stencils,bool radial_lines,
                   const std::vector<Real> &physical_volume)
      :grid_(plan.levels.at(level),2,radial_stencils),topology_(true){
    // Each descriptor constructs a fresh GLOBAL ghost plan. Transferring
    // active corrections alone is not a substitute for coarse patch coupling.
    InitializeSmoother(radial_lines);
    if(level+1<plan.levels.size()){
      transfer_=std::make_unique<CompositeLevelTransfer>(grid_.mesh,plan.levels[level+1],physical_volume);
      coarse_.reset(new CompositePoisson(plan,level+1,radial_stencils,radial_lines,transfer_->coarse_volume));
    }else FactorCoarse();
  }
  void InitializeSmoother(bool radial_lines){
    if(radial_lines)line_=std::make_unique<CompositeRadialLine>(grid_);
    for(const auto &point:grid_.geometry){Real diagonal=0;
      for(int A=0;A<3;++A)for(int a=0;a<3;++a)diagonal+=point.J[A][a]*point.J[A][a];
      diagonal_.push_back(2*grid_.n*grid_.n*diagonal);
    }
  }

  void Smooth(const std::vector<Real> &rhs, std::vector<Real> &u, int sweeps) const {
    if(line_){line_->Smooth(rhs,u,sweeps,[this](const auto &a,auto &b){Operator(a,b);});return;}
    std::vector<Real> au;
    for (int sweep = 0; sweep < sweeps; ++sweep) {
      Operator(u, au);
      for (size_t p = 0; p < u.size(); ++p) u[p] += .45*(rhs[p]-au[p])/diagonal_[p];
    }
  }
  void FactorCoarse() {
    const size_t size = grid_.active.size();
    if (size > 1024) throw std::runtime_error("bounded composite coarse solve exceeds 1024 cells");
    lu_.resize(size*size);
    pivots_.resize(size);
    std::vector<Real> basis(size, 0), column;
    for (size_t j = 0; j < size; ++j) {
      basis[j] = 1;
      Operator(basis, column);
      for (size_t i = 0; i < size; ++i) lu_[i*size+j] = column[i];
      basis[j] = 0;
    }
    for (size_t k = 0; k < size; ++k) {
      size_t pivot = k;
      for (size_t i = k+1; i < size; ++i)
        if (std::abs(lu_[i*size+k]) > std::abs(lu_[pivot*size+k])) pivot = i;
      if (!(std::abs(lu_[pivot*size+k]) > 1e-12))
        throw std::runtime_error("singular global composite coarse solve");
      pivots_[k] = pivot;
      if (pivot != k) for (size_t j = 0; j < size; ++j)
        std::swap(lu_[k*size+j], lu_[pivot*size+j]);
      for (size_t i = k+1; i < size; ++i) {
        lu_[i*size+k] /= lu_[k*size+k];
        for (size_t j = k+1; j < size; ++j)
          lu_[i*size+j] -= lu_[i*size+k]*lu_[k*size+j];
      }
    }
  }
  std::vector<Real> CoarseSolve(std::vector<Real> rhs) const {
    const size_t n = grid_.active.size();
    if (rhs.size() != n) throw std::invalid_argument("coarse rhs shape");
    for (size_t i = 0; i < n; ++i) if (pivots_[i] != i) std::swap(rhs[i], rhs[pivots_[i]]);
    for (size_t i = 0; i < n; ++i)
      for (size_t j = 0; j < i; ++j) rhs[i] -= lu_[i*n+j]*rhs[j];
    for (size_t ii = n; ii-- > 0;) {
      for (size_t j = ii+1; j < n; ++j) rhs[ii] -= lu_[ii*n+j]*rhs[j];
      rhs[ii] /= lu_[ii*n+ii];
    }
    return rhs;
  }
};

}  // namespace multigrid
#endif  // MULTIGRID_COMPOSITE_POISSON_HPP_
