// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef MULTIGRID_COMPOSITE_RADIAL_LINE_HPP_
#define MULTIGRID_COMPOSITE_RADIAL_LINE_HPP_

#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <tuple>
#include "mesh/cubed_sphere/composite_difference.hpp"

namespace multigrid {

// Host-only approximate block inverse for the second-order scalar Poisson
// PRECONDITIONER. Every shell line spans all radial layers at fixed face and
// angular node. The full operator/residual remains the caller's responsibility.
// B = -(G00 D00 + H0 D0) + 2*n^2*(G11+G22) I. Same-line active donor entries
// in radial ghost plans are retained, including non-nearest-layer entries.
// All angular/mixed and other-line/interface couplings are lagged in rhs-Au.
// No physical source, ghost plan, boundary data or final operator is changed.
class CompositeRadialLine {
 public:
  using Values = std::vector<Real>;
  using ApplyOperator = std::function<void(const Values &,Values &)>;
  struct Diagnostics {
    size_t lines=0, maximum_length=0, shell_points=0, cartesian_points=0;
    size_t radial_ghost_rows=0, same_line_ghost_entries=0, lagged_ghost_entries=0;
    size_t cross_layer_entries=0, nonadjacent_layer_entries=0;
    Real minimum_relative_pivot=1;
  };

  explicit CompositeRadialLine(const cubed_sphere::CompositeDifference &grid,
                              size_t maximum_line_length=128)
      : grid_(grid),revision_(grid.mesh.revision),radial_stencils_(grid.radial_interface_stencils),
        halo_rows_(grid.halos.size()),diagonal_(grid.active.size()),
        point_line_(grid.count,-1),point_offset_(grid.count,-1),
        active_ordinal_(grid.count,-1),halo_ordinal_(grid.count,-1) {
    if(grid.order!=2)throw std::invalid_argument("radial-line smoother requires second-order preconditioner grid");
    if(maximum_line_length<2)throw std::invalid_argument("radial-line length cap below2");
    using Key=std::tuple<int,Real,Real,Real,Real,int,int>;
    std::map<Key,std::vector<size_t>> groups;
    const int n=grid.n;const size_t block_cells=size_t(n)*n*n;
    for(size_t p=0;p<grid.active.size();++p){
      active_ordinal_[grid.active[p]]=int(p);
      const auto &block=grid.mesh.blocks[p/block_cells];
      Real norm=0;for(int A=0;A<3;++A)for(int a=0;a<3;++a)norm+=grid.geometry[p].J[A][a]*grid.geometry[p].J[A][a];
      diagonal_[p]=2*n*n*norm;
      if(!(diagonal_[p]>0)||!std::isfinite(diagonal_[p]))throw std::runtime_error("invalid radial-line point diagonal");
      if(block.face<0){++diagnostics_.cartesian_points;continue;}
      const int j=p/n%n,k=p/(n*n)%n;
      groups[{block.face,block.lo[1],block.hi[1],block.lo[2],block.hi[2],j,k}].push_back(p);
      ++diagnostics_.shell_points;
    }
    for(size_t h=0;h<grid.halos.size();++h)halo_ordinal_[grid.halos[h].target]=int(h);
    for(auto &group:groups){
      auto &points=group.second;
      if(points.size()>maximum_line_length)throw std::runtime_error("global radial line exceeds bounded dense-factor cap");
      std::sort(points.begin(),points.end(),[&](size_t a,size_t b){return Radius(a)<Radius(b);});
      std::set<int> layers;
      for(size_t p:points)layers.insert(grid.mesh.blocks[p/block_cells].layer);
      if(layers.size()!=grid.mesh.radii.size())throw std::runtime_error("global radial line lacks matching angular blocks across every shell layer");
      // The current shell hierarchy has no hanging angular faces. Reject an
      // incomplete radial partition explicitly rather than silently splitting it.
      for(int layer:layers){std::vector<std::pair<Real,Real>> intervals;
        for(size_t m=0;m<grid.mesh.blocks.size();++m){const auto &b=grid.mesh.blocks[m];
          const auto &key=group.first;
          if(b.face==std::get<0>(key)&&b.layer==layer&&b.lo[1]==std::get<1>(key)&&b.hi[1]==std::get<2>(key)
              &&b.lo[2]==std::get<3>(key)&&b.hi[2]==std::get<4>(key))intervals.push_back({b.lo[0],b.hi[0]});}
        std::sort(intervals.begin(),intervals.end());Real end=0;
        for(const auto &interval:intervals){if(std::abs(interval.first-end)>1e-13)throw std::runtime_error("radial-line shell partition has gap/overlap");end=interval.second;}
        if(std::abs(end-1)>1e-13)throw std::runtime_error("radial-line shell partition does not reach layer boundary");
      }
      const int line=int(lines_.size());
      for(size_t i=0;i<points.size();++i){point_line_[grid.active[points[i]]]=line;point_offset_[grid.active[points[i]]]=int(i);}
      Line data;data.points=std::move(points);lines_.push_back(std::move(data));
    }
    diagnostics_.lines=lines_.size();
    for(size_t l=0;l<lines_.size();++l){auto &line=lines_[l];const size_t size=line.points.size();
      diagnostics_.maximum_length=std::max(diagnostics_.maximum_length,size);
      line.matrix.assign(size*size,0);
      for(size_t row=0;row<size;++row){const size_t p=line.points[row];const auto &geometry=grid.geometry[p];
        Real G[3]{},H0=0;for(int a=0;a<3;++a){for(int A=0;A<3;++A)G[A]+=geometry.J[A][a]*geometry.J[A][a];H0+=geometry.H[0][a][a];}
        line.matrix[row*size+row]+=2*n*n*(G[0]+G[1]+G[2]);
        for(int direction:{-1,1}){
          const int donor=grid.active[p]+grid.Offset(0,direction);const Real coefficient=-n*n*G[0]-direction*n*H0/2;
          if(active_ordinal_[donor]>=0)AddEntry(l,row,donor,coefficient,false);
          else {const int h=halo_ordinal_[donor];if(h<0)throw std::runtime_error("radial stencil has no active/halo donor");
            ++diagnostics_.radial_ghost_rows;const auto &plan=grid.halos[h];
            if(plan.entries.empty())throw std::runtime_error("empty radial ghost interpolation plan");
            // Fill uses an anchor form. Retain its tiny partition-of-unity
            // correction too; do not assume floating point weights sum to1.
            Real sum=plan.boundary_weight;
            for(const auto &entry:plan.entries){sum+=entry.weight;AddEntry(l,row,entry.point,coefficient*entry.weight,true);}
            AddEntry(l,row,plan.entries.front().point,coefficient*(1-sum),true);
          }
        }
      }
      line.lu=line.matrix;Factor(line);
    }
  }
  const Diagnostics &GetDiagnostics() const{return diagnostics_;}
  const cubed_sphere::CompositeDifference &Grid() const{return grid_;}

  // Simultaneous damped block-Jacobi defect correction. Off-line, angular,
  // mixed, Cartesian/shell and physical-boundary effects enter the FULL residual.
  void Smooth(const Values &rhs,Values &u,int sweeps,const ApplyOperator &apply,
              Real damping=.65) const {
    Check();
    if(rhs.size()!=grid_.active.size()||u.size()!=rhs.size()||sweeps<0||!(damping>0&&damping<=1))
      throw std::invalid_argument("radial-line smoother shape/sweep/damping");
    Values au,residual(rhs.size());
    for(int sweep=0;sweep<sweeps;++sweep){apply(u,au);
      if(au.size()!=rhs.size())throw std::runtime_error("radial-line full operator shape");
      for(size_t p=0;p<u.size();++p)residual[p]=rhs[p]-au[p];
      for(size_t p=0;p<u.size();++p)if(point_line_[grid_.active[p]]<0)u[p]+=damping*residual[p]/diagonal_[p];
      for(const auto &line:lines_){Values correction(line.points.size());
        for(size_t i=0;i<correction.size();++i)correction[i]=residual[line.points[i]];
        Solve(line,correction);
        for(size_t i=0;i<correction.size();++i){if(!std::isfinite(correction[i]))throw std::runtime_error("nonfinite radial-line correction");u[line.points[i]]+=damping*correction[i];}
      }
    }
  }

  // Numerical LU verification, independent of the full Poisson residual.
  Real MaximumFactorResidual() const {
    Check();Real maximum=0;
    for(const auto &line:lines_){const size_t size=line.points.size();Values rhs(size),solution;
      for(size_t i=0;i<size;++i)rhs[i]=std::sin(Real(i+1)*.731)+Real(i%3)-1;solution=rhs;Solve(line,solution);
      Real numerator=0,denominator=0;
      for(size_t i=0;i<size;++i){Real value=0,scale=std::abs(rhs[i]);for(size_t j=0;j<size;++j){value+=line.matrix[i*size+j]*solution[j];scale+=std::abs(line.matrix[i*size+j]*solution[j]);}
        numerator=std::max(numerator,std::abs(value-rhs[i]));denominator=std::max(denominator,scale);}
      maximum=std::max(maximum,numerator/std::max(denominator,std::numeric_limits<Real>::min()));
    }return maximum;
  }

  // Independent stencil evaluation on a bounded subset of lines: zero every
  // other active line, run the unchanged ghost fill, then use First/Second.
  Real MaximumAssemblyResidual(size_t samples=6) const {
    Check();Real maximum=0;if(lines_.empty())return maximum;
    samples=std::min(samples,lines_.size());
    for(size_t sample=0;sample<samples;++sample){const auto &line=lines_[sample*lines_.size()/samples];
      const size_t size=line.points.size();Values field(grid_.count,0),values(size);
      for(size_t i=0;i<size;++i){values[i]=std::sin((i+1)*.719)+.1*i;field[grid_.active[line.points[i]]]=values[i];}
      grid_.Fill(field,1);
      for(size_t i=0;i<size;++i){const size_t p=line.points[i];const auto &g=grid_.geometry[p];Real G[3]{},H0=0;
        for(int a=0;a<3;++a){for(int A=0;A<3;++A)G[A]+=g.J[A][a]*g.J[A][a];H0+=g.H[0][a][a];}
        const Real stencil=-G[0]*grid_.Second(field,grid_.active[p],0,0)-H0*grid_.First(field,grid_.active[p],0)
            +2*grid_.n*grid_.n*(G[1]+G[2])*values[i];
        Real matrix=0,scale=std::abs(stencil);for(size_t j=0;j<size;++j){matrix+=line.matrix[i*size+j]*values[j];scale+=std::abs(line.matrix[i*size+j]*values[j]);}
        maximum=std::max(maximum,std::abs(matrix-stencil)/std::max(scale,std::numeric_limits<Real>::min()));
      }
    }return maximum;
  }

 private:
  struct Line{std::vector<size_t> points,pivots;Values matrix,lu;};
  const cubed_sphere::CompositeDifference &grid_;
  unsigned long revision_;bool radial_stencils_;size_t halo_rows_;
  Values diagonal_;std::vector<int> point_line_,point_offset_,active_ordinal_,halo_ordinal_;
  std::vector<Line> lines_;Diagnostics diagnostics_;
  Real Radius(size_t p) const{const auto &x=grid_.geometry[p].x;return std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);}
  void Check() const{
    if(grid_.mesh.revision!=revision_||grid_.radial_interface_stencils!=radial_stencils_||grid_.halos.size()!=halo_rows_||grid_.active.size()!=diagonal_.size())
      throw std::runtime_error("stale radial-line factors after regrid/stencil change; rebuild hierarchy");
  }
  void AddEntry(size_t l,size_t row,int donor,Real coefficient,bool ghost){
    if(coefficient==0)return;
    if(point_line_[donor]!=int(l)){if(ghost)++diagnostics_.lagged_ghost_entries;return;}
    auto &line=lines_[l];const size_t column=point_offset_[donor];line.matrix[row*line.points.size()+column]+=coefficient;
    if(ghost)++diagnostics_.same_line_ghost_entries;
    const size_t block_cells=size_t(grid_.n)*grid_.n*grid_.n;
    const int a=grid_.mesh.blocks[line.points[row]/block_cells].layer,b=grid_.mesh.blocks[line.points[column]/block_cells].layer;
    if(a!=b){++diagnostics_.cross_layer_entries;if(std::abs(a-b)>1)++diagnostics_.nonadjacent_layer_entries;}
  }
  void Factor(Line &line){const size_t size=line.points.size();line.pivots.resize(size);Real scale=0;
    for(Real a:line.lu)scale=std::max(scale,std::abs(a));
    for(size_t k=0;k<size;++k){size_t pivot=k;
      for(size_t i=k+1;i<size;++i)if(std::abs(line.lu[i*size+k])>std::abs(line.lu[pivot*size+k]))pivot=i;
      const Real magnitude=std::abs(line.lu[pivot*size+k]);
      if(!(magnitude>0)||!std::isfinite(magnitude))throw std::runtime_error("singular/nonfinite global radial-line block");
      diagnostics_.minimum_relative_pivot=std::min(diagnostics_.minimum_relative_pivot,magnitude/scale);
      line.pivots[k]=pivot;if(pivot!=k)for(size_t j=0;j<size;++j)std::swap(line.lu[k*size+j],line.lu[pivot*size+j]);
      for(size_t i=k+1;i<size;++i){line.lu[i*size+k]/=line.lu[k*size+k];for(size_t j=k+1;j<size;++j)line.lu[i*size+j]-=line.lu[i*size+k]*line.lu[k*size+j];}
    }
  }
  static void Solve(const Line &line,Values &rhs){const size_t size=line.points.size();
    for(size_t i=0;i<size;++i)if(line.pivots[i]!=i)std::swap(rhs[i],rhs[line.pivots[i]]);
    for(size_t i=0;i<size;++i)for(size_t j=0;j<i;++j)rhs[i]-=line.lu[i*size+j]*rhs[j];
    for(size_t i=size;i-->0;){for(size_t j=i+1;j<size;++j)rhs[i]-=line.lu[i*size+j]*rhs[j];rhs[i]/=line.lu[i*size+i];}
  }
};
} // namespace multigrid
#endif // MULTIGRID_COMPOSITE_RADIAL_LINE_HPP_
