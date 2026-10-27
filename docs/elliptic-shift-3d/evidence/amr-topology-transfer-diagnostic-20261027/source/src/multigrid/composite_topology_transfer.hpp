// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef MULTIGRID_COMPOSITE_TOPOLOGY_TRANSFER_HPP_
#define MULTIGRID_COMPOSITE_TOPOLOGY_TRANSFER_HPP_

#include <cstdint>
#include <map>
#include <numeric>
#include <queue>
#include <string>
#include "mesh/cubed_sphere/composite.hpp"

namespace multigrid {

struct CompositeTopologyRound {
  cubed_sphere::CompositeMesh coarse;
  size_t merged_families=0;
};

// PRECONDITIONER descriptor coarsening only: never alters the evolution mesh.
// Complete globally finest Cartesian sibling families merge simultaneously.
// Parent bounds are dyadic and anchored at -half_width; shell descriptors are
// copied unchanged. Base Cartesian blocks (refinement0) are not merged.
inline CompositeTopologyRound CoarsenCompositeCartesian(const cubed_sphere::CompositeMesh &fine) {
  using Block=cubed_sphere::CompositeBlock;
  using Key=std::array<std::int64_t,4>;
  struct Family {Block parent;std::vector<size_t> children;unsigned octants=0;};
  int finest=0;std::array<Real,3> base{};bool have_base=false;
  for(const auto &b:fine.blocks)if(b.face<0){
    if(b.layer!=-1||b.refinement<0)throw std::invalid_argument("invalid Cartesian AMR refinement metadata");
    finest=std::max(finest,b.refinement);
    for(int a=0;a<3;++a){const Real width=b.hi[a]-b.lo[a],root=std::ldexp(width,b.refinement);
      if(!(width>0)||!std::isfinite(root)||b.lo[a]<-fine.half_width-1e-13||b.hi[a]>fine.half_width+1e-13)
        throw std::invalid_argument("invalid Cartesian AMR bounds");
      if(have_base&&std::abs(root-base[a])>2e-12*base[a])throw std::invalid_argument("inconsistent Cartesian AMR base widths");
      if(!have_base)base[a]=root;}
    have_base=true;
  }
  CompositeTopologyRound out{fine,0};if(finest==0)return out;
  std::map<Key,Family> families;
  for(size_t m=0;m<fine.blocks.size();++m){const auto &b=fine.blocks[m];if(b.face>=0||b.refinement!=finest)continue;
    Block parent=b;--parent.refinement;Key key{};key[0]=parent.refinement;unsigned octant=0;
    for(int a=0;a<3;++a){const Real width=std::ldexp(base[a],-b.refinement),coordinate=(b.lo[a]+fine.half_width)/width;
      if(!std::isfinite(coordinate)||coordinate<-.01||coordinate>Real(1ull<<50))throw std::invalid_argument("Cartesian dyadic AMR index range");
      const auto index=static_cast<std::int64_t>(std::llround(coordinate));
      if(std::abs(coordinate-index)>2e-10||std::abs((b.hi[a]-b.lo[a])-width)>2e-12*width)
        throw std::invalid_argument("Cartesian AMR leaf is not on its dyadic lattice");
      key[1+a]=index/2;if(index%2)octant|=1u<<a;
      parent.lo[a]=-fine.half_width+key[1+a]*(2*width);parent.hi[a]=parent.lo[a]+2*width;
    }
    auto &family=families[key];if(family.children.empty())family.parent=parent;
    if(family.octants&(1u<<octant))throw std::invalid_argument("duplicate Cartesian sibling octant");
    family.octants|=1u<<octant;family.children.push_back(m);
  }
  std::vector<bool> removed(fine.blocks.size(),false);
  for(const auto &entry:families)if(entry.second.children.size()==8&&entry.second.octants==255){
    ++out.merged_families;for(size_t m:entry.second.children)removed[m]=true;}
  if(!out.merged_families)return out;
  out.coarse.blocks.clear();for(size_t m=0;m<fine.blocks.size();++m)if(!removed[m])out.coarse.blocks.push_back(fine.blocks[m]);
  for(const auto &entry:families)if(entry.second.children.size()==8&&entry.second.octants==255)out.coarse.blocks.push_back(entry.second.parent);
  ++out.coarse.revision;return out;
}

struct CompositeCoarseningPlan {
  enum class Kind {LogicalCells,CartesianSiblings};
  std::vector<cubed_sphere::CompositeMesh> levels;
  std::vector<Kind> transitions;
  std::vector<size_t> merged_families;
  bool dense_supported=false;
  std::string unsupported_reason;
  size_t TerminalCells() const {const auto &m=levels.back();return m.blocks.size()*m.cells*m.cells*m.cells;}
  // Odd3->2 and5->3 logical transfers are intentional lower-order MG controls.
  // They do not alter the final discretized vector operator or its tolerance.
  static CompositeCoarseningPlan Build(cubed_sphere::CompositeMesh mesh,size_t dense_cap=1024) {
    CompositeCoarseningPlan out;out.levels.push_back(mesh);
    while(mesh.cells>2){mesh.cells=std::max(2,(mesh.cells+1)/2);++mesh.revision;
      out.transitions.push_back(Kind::LogicalCells);out.merged_families.push_back(0);out.levels.push_back(mesh);}
    while(true){bool refined=false;for(const auto &b:mesh.blocks)if(b.face<0&&b.refinement>0)refined=true;
      if(!refined)break;
      auto round=CoarsenCompositeCartesian(mesh);
      if(!round.merged_families){out.unsupported_reason="refined Cartesian leaves have no complete finest sibling family";return out;}
      mesh=std::move(round.coarse);out.transitions.push_back(Kind::CartesianSiblings);out.merged_families.push_back(round.merged_families);out.levels.push_back(mesh);
    }
    out.dense_supported=out.TerminalCells()<=dense_cap;
    if(!out.dense_supported)out.unsupported_reason="unchanged shell/base topology exceeds bounded dense coarse cap at N2; shell/face coarsening or iterative coarse solve is still required";
    return out;
  }
};

// General nested-level restriction and GLOBAL ACTIVE-ONLY prolongation.
// Restriction weights are exact logical overlap fractions times fine physical
// mapped-cell volumes, normalized per coarse cell. Optional supplied positive
// volumes allow a caller's scaled physical metric measure. No discarded cells.
// For odd logical ratios, the Jacobian is piecewise constant over each fine
// cell: this preserves constants and total represented mass, not exact analytic
// subcell integrals. It is a lower-order preconditioner transfer.
class CompositeLevelTransfer {
 public:
  using Vector=std::vector<Real>;
  struct Entry {size_t point;Real weight;};
  cubed_sphere::CompositeMesh fine,coarse;
  std::vector<std::vector<Entry>> restriction,prolongation;
  Vector fine_volume,coarse_volume;
  size_t quadratic_rows=0,linear_rows=0,identity_rows=0,cross_patch_rows=0;
  Real maximum_prolongation_l1=0,maximum_moment_error=0,maximum_coverage_error=0;
  CompositeLevelTransfer(cubed_sphere::CompositeMesh f,cubed_sphere::CompositeMesh c,const Vector &physical_volume={})
      :fine(std::move(f)),coarse(std::move(c)) {
    if(fine.half_width!=coarse.half_width||fine.radii!=coarse.radii||fine.face_subdivisions!=coarse.face_subdivisions||coarse.cells>fine.cells)
      throw std::invalid_argument("incompatible nested composite transfer geometry");
    MakePoints(fine,fine_points_,fine_volume);Vector geometric_coarse;MakePoints(coarse,coarse_points_,geometric_coarse);
    if(!physical_volume.empty()){if(physical_volume.size()!=fine_volume.size())throw std::invalid_argument("fine physical measure shape");fine_volume=physical_volume;}
    for(Real w:fine_volume)if(!(w>0)||!std::isfinite(w))throw std::invalid_argument("fine physical measure must be positive finite/scaled");
    BuildRestriction();BuildProlongation();
  }
  Vector Restrict(const Vector &values,int components=1) const {
    Shape(values,fine_volume.size(),components);Vector out(coarse_volume.size()*components);
    for(size_t p=0;p<restriction.size();++p)for(int c=0;c<components;++c){const Real anchor=values[components*restriction[p][0].point+c];Real sum=0;
      for(const auto &e:restriction[p])sum+=e.weight*(values[components*e.point+c]-anchor);out[components*p+c]=anchor+sum/coarse_volume[p];}
    return out;
  }
  Vector Prolong(const Vector &values,int components=1) const {
    Shape(values,coarse_volume.size(),components);Vector out(fine_volume.size()*components);
    for(size_t p=0;p<prolongation.size();++p)for(int c=0;c<components;++c){const Real anchor=values[components*prolongation[p][0].point+c];Real sum=0;
      for(const auto &e:prolongation[p])sum+=e.weight*(values[components*e.point+c]-anchor);out[components*p+c]=anchor+sum;}
    return out;
  }
  const std::vector<std::array<Real,3>> &FinePoints() const {return fine_points_;}
  const std::vector<std::array<Real,3>> &CoarsePoints() const {return coarse_points_;}
 private:
  std::vector<std::array<Real,3>> fine_points_,coarse_points_;
  struct Node {size_t point;int left=-1,right=-1,axis=0;};
  std::vector<Node> tree_;
  static void Shape(const Vector &v,size_t points,int components){if(components<1||v.size()!=points*components)throw std::invalid_argument("composite transfer field shape");
    for(Real x:v)if(!std::isfinite(x))throw std::invalid_argument("nonfinite composite transfer field");}
  static bool Near(Real a,Real b){return std::abs(a-b)<=2e-12*std::max({Real(1),std::abs(a),std::abs(b)});}
  static void MakePoints(const cubed_sphere::CompositeMesh &m,std::vector<std::array<Real,3>> &points,Vector &volumes){
    const int n=m.cells;points.reserve(m.blocks.size()*n*n*n);volumes.reserve(points.capacity());
    for(size_t b=0;b<m.blocks.size();++b)for(int k=0;k<n;++k)for(int j=0;j<n;++j)for(int i=0;i<n;++i){
      const Real q[3]={(i+.5)/n,(j+.5)/n,(k+.5)/n};const auto g=m.Geometry(b,q);points.push_back({g.x[0],g.x[1],g.x[2]});volumes.push_back(cubed_sphere::PhysicalCellVolume(g,n));}}
  void BuildRestriction(){
    const int nf=fine.cells,nc=coarse.cells;restriction.resize(coarse_points_.size());coarse_volume.assign(coarse_points_.size(),0);Vector coverage(fine_points_.size(),0);
    for(size_t fb=0;fb<fine.blocks.size();++fb){const auto &f=fine.blocks[fb];int owner=-1;
      for(size_t cb=0;cb<coarse.blocks.size();++cb){const auto &c=coarse.blocks[cb];if(f.face!=c.face||f.layer!=c.layer)continue;bool contains=true;
        for(int a=0;a<3;++a)if(f.face<0){if(f.lo[a]<c.lo[a]-2e-13||f.hi[a]>c.hi[a]+2e-13)contains=false;}
          else if(!Near(f.lo[a],c.lo[a])||!Near(f.hi[a],c.hi[a]))contains=false;
        if(contains){if(owner>=0)throw std::invalid_argument("overlapping coarse block ownership");owner=cb;}}
      if(owner<0)throw std::invalid_argument("fine leaf has no nested coarse owner");const auto &c=coarse.blocks[owner];
      for(int k=0;k<nf;++k)for(int j=0;j<nf;++j)for(int i=0;i<nf;++i){const int ijk[3]={i,j,k};const size_t p=((fb*nf+k)*nf+j)*nf+i;
        Real flo[3],fhi[3],clo[3],dc[3];int begin[3],end[3];
        for(int a=0;a<3;++a){const Real lower=f.face<0?f.lo[a]:0,upper=f.face<0?f.hi[a]:1;
          flo[a]=lower+(upper-lower)*ijk[a]/nf;fhi[a]=lower+(upper-lower)*(ijk[a]+1)/nf;clo[a]=f.face<0?c.lo[a]:0;dc[a]=(f.face<0?c.hi[a]-c.lo[a]:1)/nc;
          begin[a]=std::clamp(int(std::floor((flo[a]-clo[a])/dc[a]+1e-11)),0,nc-1);end[a]=std::clamp(int(std::ceil((fhi[a]-clo[a])/dc[a]-1e-11)),1,nc);}
        for(int ck=begin[2];ck<end[2];++ck)for(int cj=begin[1];cj<end[1];++cj)for(int ci=begin[0];ci<end[0];++ci){const int index[3]={ci,cj,ck};Real fraction=1;
          for(int a=0;a<3;++a){const Real lo=clo[a]+index[a]*dc[a],hi=lo+dc[a];fraction*=std::max(Real(0),std::min(fhi[a],hi)-std::max(flo[a],lo))/(fhi[a]-flo[a]);}
          if(!(fraction>0))continue;const size_t q=((size_t(owner)*nc+ck)*nc+cj)*nc+ci;const Real weight=fine_volume[p]*fraction;
          restriction[q].push_back({p,weight});coarse_volume[q]+=weight;coverage[p]+=fraction;}
      }
    }
    for(size_t p=0;p<coverage.size();++p){maximum_coverage_error=std::max(maximum_coverage_error,std::abs(coverage[p]-1));if(std::abs(coverage[p]-1)>2e-10)throw std::runtime_error("composite restriction has a gap/overlap");}
    for(size_t p=0;p<restriction.size();++p)if(restriction[p].empty()||!(coarse_volume[p]>0)||!std::isfinite(coarse_volume[p]))throw std::runtime_error("empty/overflowing composite coarse volume");
  }
  int BuildTree(std::vector<size_t> &ids,size_t begin,size_t end,int depth){if(begin==end)return -1;const size_t middle=(begin+end)/2;const int axis=depth%3;
    std::nth_element(ids.begin()+begin,ids.begin()+middle,ids.begin()+end,[&](size_t a,size_t b){return coarse_points_[a][axis]<coarse_points_[b][axis];});
    const int node=tree_.size();tree_.push_back({ids[middle],-1,-1,axis});const int left=BuildTree(ids,begin,middle,depth+1),right=BuildTree(ids,middle+1,end,depth+1);tree_[node].left=left;tree_[node].right=right;return node;}
  using Distance=std::pair<Real,size_t>;
  void Nearest(int node,const std::array<Real,3> &x,size_t wanted,std::priority_queue<Distance> &heap) const {if(node<0)return;const auto &n=tree_[node];Real distance=0;
    for(int a=0;a<3;++a)distance+=std::pow(coarse_points_[n.point][a]-x[a],2);
    if(heap.size()<wanted)heap.push({distance,n.point});else if(distance<heap.top().first){heap.pop();heap.push({distance,n.point});}
    const Real difference=x[n.axis]-coarse_points_[n.point][n.axis];const int near=difference<0?n.left:n.right,far=difference<0?n.right:n.left;
    Nearest(near,x,wanted,heap);if(heap.size()<wanted||difference*difference<=heap.top().first)Nearest(far,x,wanted,heap);}
  static std::array<Real,10> Basis(const std::array<Real,3> &x){return {1,x[0],x[1],x[2],x[0]*x[0],x[1]*x[1],x[2]*x[2],x[0]*x[1],x[0]*x[2],x[1]*x[2]};}
  static bool Solve(Real a[10][10],Real b[10],int size){Real maximum=0;for(int i=0;i<size;++i)maximum=std::max(maximum,std::abs(a[i][i]));
    for(int k=0;k<size;++k){int pivot=k;for(int i=k+1;i<size;++i)if(std::abs(a[i][k])>std::abs(a[pivot][k]))pivot=i;
      if(!(std::abs(a[pivot][k])>1e-12*maximum))return false;
      if(pivot!=k){for(int j=k;j<size;++j)std::swap(a[k][j],a[pivot][j]);std::swap(b[k],b[pivot]);}
      for(int i=k+1;i<size;++i){const Real factor=a[i][k]/a[k][k];for(int j=k;j<size;++j)a[i][j]-=factor*a[k][j];b[i]-=factor*b[k];}}
    for(int i=size;i-->0;){for(int j=i+1;j<size;++j)b[i]-=a[i][j]*b[j];b[i]/=a[i][i];}return true;}
  bool MomentRow(size_t p,const std::vector<Distance> &neighbors,int degree,std::vector<Entry> &row,Real &moment_error) const {
    const int moments=degree==2?10:4;const Real scale=std::sqrt(neighbors.back().first);if(!(scale>0))return false;
    std::vector<std::array<Real,10>> basis;Vector preference;Real matrix[10][10]{},rhs[10]{};rhs[0]=1;
    for(const auto &neighbor:neighbors){std::array<Real,3> local;for(int a=0;a<3;++a)local[a]=(coarse_points_[neighbor.second][a]-fine_points_[p][a])/scale;
      basis.push_back(Basis(local));preference.push_back(1/(Real(.1)+neighbor.first/(scale*scale)));const auto &v=basis.back();
      for(int i=0;i<moments;++i)for(int j=0;j<moments;++j)matrix[i][j]+=preference.back()*v[i]*v[j];}
    if(!Solve(matrix,rhs,moments))return false;row.clear();std::array<Real,10> actual{};
    for(size_t j=0;j<neighbors.size();++j){Real weight=0;for(int i=0;i<moments;++i)weight+=preference[j]*basis[j][i]*rhs[i];row.push_back({neighbors[j].second,weight});
      for(int i=0;i<moments;++i)actual[i]+=weight*basis[j][i];}
    moment_error=0;for(int i=0;i<moments;++i)moment_error=std::max(moment_error,std::abs(actual[i]-(i==0?1:0)));return moment_error<2e-9;
  }
  void BuildProlongation(){std::vector<size_t> ids(coarse_points_.size());std::iota(ids.begin(),ids.end(),0);tree_.reserve(ids.size());BuildTree(ids,0,ids.size(),0);prolongation.resize(fine_points_.size());
    const size_t coarse_cell_block=coarse.cells*coarse.cells*coarse.cells;
    for(size_t p=0;p<fine_points_.size();++p){bool built=false;Real error=0;int used_degree=2;
      for(size_t wanted:{size_t(24),size_t(48),size_t(96)}){std::priority_queue<Distance> heap;Nearest(0,fine_points_[p],std::min(wanted,coarse_points_.size()),heap);std::vector<Distance> neighbors;
        while(!heap.empty()){neighbors.push_back(heap.top());heap.pop();}std::reverse(neighbors.begin(),neighbors.end());
        if(neighbors.front().first<1e-28*coarse.half_width*coarse.half_width){prolongation[p]={{neighbors.front().second,1}};++identity_rows;built=true;used_degree=0;break;}
        if(MomentRow(p,neighbors,2,prolongation[p],error)){built=true;break;}
        if(wanted==96&&MomentRow(p,neighbors,1,prolongation[p],error)){built=true;used_degree=1;}
      }
      if(!built)throw std::runtime_error("global active prolongation has deficient physical moments");
      if(used_degree==2)++quadratic_rows;else if(used_degree==1)++linear_rows;maximum_moment_error=std::max(maximum_moment_error,error);
      Real l1=0;const auto &first=coarse.blocks[prolongation[p][0].point/coarse_cell_block];bool cross=false;
      for(const auto &e:prolongation[p]){l1+=std::abs(e.weight);const auto &b=coarse.blocks[e.point/coarse_cell_block];if(b.face!=first.face||b.layer!=first.layer)cross=true;}
      maximum_prolongation_l1=std::max(maximum_prolongation_l1,l1);if(cross)++cross_patch_rows;
    }
  }
};

}  // namespace multigrid
#endif
