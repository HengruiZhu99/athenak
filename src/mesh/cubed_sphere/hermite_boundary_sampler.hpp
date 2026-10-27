// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef MESH_CUBED_SPHERE_HERMITE_BOUNDARY_SAMPLER_HPP_
#define MESH_CUBED_SPHERE_HERMITE_BOUNDARY_SAMPLER_HPP_
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <type_traits>
#include "mesh/cubed_sphere/scri_surface.hpp"
#include "mesh/cubed_sphere/surface_polynomial_weights.hpp"
namespace cubed_sphere {

// Host sampler for arbitrary global Cartesian stored components. Geometry is
// cached, never stage fields. Prepared owns an immutable stage snapshot. The
// radial derivative is with respect to compactified coordinate r=|x|, not the
// metric-unit normal or uncompactified areal radius. Metric contractions and
// physical boundary/source equations remain the caller's responsibility.
class HermiteBoundarySampler {
 private:
  struct GeometryFrame;
  struct EpochState {bool has_epoch=false;std::uint64_t epoch=0;};
  struct StageData {
    std::weak_ptr<const EpochState> owner;
    std::shared_ptr<const GeometryFrame> geometry;
    std::uint64_t epoch=0;int components=0;
    std::vector<Real> trace,G,H,q0,q00;
  };
 public:
  using Jets=ScriSurface::Jets;
  class Prepared {
   public:
    Prepared(const Prepared&)=default;Prepared(Prepared&&)=default;
    Prepared&operator=(const Prepared&)=default;Prepared&operator=(Prepared&&)=default;
    int components()const{return Get().components;}
    std::uint64_t epoch()const{return Get().epoch;}
    const std::vector<Real>&trace()const{return Get().trace;}
    const std::vector<Real>&radial_first()const{return Get().G;}
    const std::vector<Real>&radial_second()const{return Get().H;}
    const std::vector<Real>&raw_q0()const{return Get().q0;}
    const std::vector<Real>&raw_q00()const{return Get().q00;}
   private:
    friend class HermiteBoundarySampler;
    explicit Prepared(std::shared_ptr<const StageData> data):data_(std::move(data)){}
    const StageData&Get()const{if(!data_)throw std::logic_error("empty Hermite Prepared");return *data_;}
    std::shared_ptr<const StageData> data_;
  };
  struct RadialMapJet {Real rs=0,rss=0,rs_A[3]{};};
  struct Diagnostics {Real value_weight_l1=0,factor_condition_indicator=0,polynomial_reproduction=0;size_t angular_plans=0;};

  explicit HermiteBoundarySampler(const ScriSurface&s)
      :surface_(s),grid_(s.grid),mesh_(grid_.mesh),n_(grid_.n),ng_(grid_.ng),padded_(grid_.padded),count_(grid_.count),order_(grid_.order),
       radial_stencils_(grid_.radial_interface_stencils),nodes_(s.radial_nodes),surface_components_(s.components),
       surface_size_(s.values.size()),beta_size_(s.beta.size()),active_(grid_.active),geometry_(grid_.geometry),base_(s.base),
       angular_(Coordinates(s)),epoch_(std::make_shared<EpochState>()) {
    active_storage_=grid_.active.data();geometry_storage_=grid_.geometry.data();row_storage_=surface_.rows.data();base_storage_=surface_.base.data();block_storage_=grid_.mesh.blocks.data();
    if(std::numeric_limits<Real>::digits<50)throw std::invalid_argument("Hermite host sampler requires double precision");
    ValidateInitial();BuildRadialWeights();rows_.reserve(s.rows.size());
    for(const auto&r:s.rows)rows_.push_back({r.block,r.j,r.k,r.geometry,r.radial_spacing,r.terms.size()});
    for(size_t p=0;p<rows_.size();++p){const auto&r=rows_[p];const Real q[3]={1,(r.j+.5)/n_,(r.k+.5)/n_};native_.push_back(BuildQuery(r.block,q,p));}
    frame_=std::make_shared<const GeometryFrame>(GeometryFrame{native_,angular_rows_,mesh_.revision});
  }
  HermiteBoundarySampler(ScriSurface&&)=delete;
  HermiteBoundarySampler(const ScriSurface&&)=delete;
  HermiteBoundarySampler(const HermiteBoundarySampler&)=delete;
  HermiteBoundarySampler&operator=(const HermiteBoundarySampler&)=delete;
  HermiteBoundarySampler(HermiteBoundarySampler&&)=delete;
  HermiteBoundarySampler&operator=(HermiteBoundarySampler&&)=delete;

  // Same epoch may prepare several component sets (e.g. metric22 and beta3).
  // Only a successful greater-epoch preparation invalidates older snapshots.
  Prepared Prepare(const std::vector<Real>&padded,const std::vector<Real>&surface_values,int components,std::uint64_t epoch) {
    CheckGeometry();if(epoch_->has_epoch&&epoch<epoch_->epoch)throw std::logic_error("Hermite epoch moved backwards");
    if(components<1||components>64||padded.size()!=size_t(count_)*components||surface_values.size()!=rows_.size()*components)
      throw std::invalid_argument("Hermite Prepare component/shape mismatch");
    for(Real v:padded)if(!std::isfinite(v))throw std::invalid_argument("nonfinite Hermite padded data");
    for(Real v:surface_values)if(!std::isfinite(v))throw std::invalid_argument("nonfinite Hermite surface data");
    auto data=std::make_shared<StageData>();data->owner=epoch_;data->geometry=frame_;data->epoch=epoch;data->components=components;data->trace=surface_values;
    data->G.resize(surface_values.size());data->H.resize(surface_values.size());data->q0.resize(surface_values.size());data->q00.resize(surface_values.size());
    for(size_t p=0;p<rows_.size();++p){const auto&r=rows_[p];const auto&m=native_[p].radial;
      for(int f=0;f<components;++f){const size_t target=p*components+f;const Real anchor=surface_values[target];Real first=0,second=0;
        for(int i=0;i<nodes_-1;++i){const Real difference=padded[size_t(f)*count_+Index(r.block,n_-(nodes_-1)+i,r.j,r.k)]-anchor;first+=dr_[i]*difference;second+=ddr_[i]*difference;}
        data->q0[target]=first;data->q00[target]=second;const auto normal=CoordinateRadialNormal(first,second,m);data->G[target]=normal[0];data->H[target]=normal[1];
        if(!std::isfinite(first)||!std::isfinite(second)||!std::isfinite(data->G[target])||!std::isfinite(data->H[target]))throw std::runtime_error("nonfinite Hermite prepared normal jet");
      }
    }
    epoch_->has_epoch=true;epoch_->epoch=epoch;return Prepared(std::move(data));
  }
  Jets Sample(size_t index,const Prepared&prepared)const {
    CheckPrepared(prepared);if(index>=native_.size())throw std::out_of_range("Hermite boundary sample index");CheckRow(index);const auto&data=prepared.Get();return Evaluate(data.geometry->native[index],data,data.geometry->angular);
  }
  // Additional boundary query used for exact-seam/corner validation. Native
  // points route through Sample and retain their exact trace/q0/q00 data.
  Jets SampleAt(int block,const Real q[3],const Prepared&prepared)const {
    CheckPrepared(prepared);ValidateQuery(block,q);CheckBlock(block);int j=std::lround(n_*q[1]-.5),k=std::lround(n_*q[2]-.5);
    if(j>=0&&j<n_&&k>=0&&k<n_&&std::abs(n_*q[1]-.5-j)<1e-12&&std::abs(n_*q[2]-.5-k)<1e-12)
      {const size_t index=base_[block]+j+n_*k;CheckRow(index);const auto&data=prepared.Get();return Evaluate(data.geometry->native[index],data,data.geometry->angular);}
    return Evaluate(BuildQuery(block,q,NoNative()),prepared.Get(),angular_rows_);
  }
  Diagnostics diagnostics()const{return diagnostic_;}
  // Full O(grid) integrity audit. Prepare always calls this. Sample checks only
  // epoch/owner, storage/shape/revision and the queried row/block. Derived-grid
  // mutations made illegally without revision require this explicit audit;
  // legitimate regridding must rebuild the sampler before preparing new data.
  void ValidateGeometry()const{CheckGeometry();}

  static std::array<Real,2> CoordinateRadialNormal(Real q0,Real q00,const RadialMapJet&map){
    if(!(map.rs>0)||!std::isfinite(map.rs)||!std::isfinite(map.rss)||!std::isfinite(q0)||!std::isfinite(q00))throw std::invalid_argument("invalid Hermite radial normal input");
    const Real G=q0/map.rs,H=(q00-map.rss*G)/(map.rs*map.rs);if(!std::isfinite(G)||!std::isfinite(H))throw std::runtime_error("nonfinite Hermite coordinate normal jet");return {G,H};
  }

  // General inverse-map chain rule. For x(s,A)=r(s,A)n(A), includes nonzero
  // r_ss. The current CompositeMesh maps are independently checked below.
  static RadialMapJet RadialGeometry(const Point&p) {
    for(Real v:p.x)if(!std::isfinite(v))throw std::invalid_argument("nonfinite Hermite map position");
    Real radius=0;for(Real v:p.x)radius+=v*v;radius=std::sqrt(radius);if(!(radius>0&&std::isfinite(radius)))throw std::invalid_argument("invalid Hermite map radius");
    Real D[3][3]{},X[3][3][3]{},normal[3];const auto &J=p.J;
    const Real det=J[0][0]*(J[1][1]*J[2][2]-J[1][2]*J[2][1])-J[0][1]*(J[1][0]*J[2][2]-J[1][2]*J[2][0])+J[0][2]*(J[1][0]*J[2][1]-J[1][1]*J[2][0]);
    if(det==0||!std::isfinite(det))throw std::invalid_argument("singular Hermite inverse map");
    for(int A=0;A<3;++A)for(int i=0;i<3;++i){D[i][A]=(J[(A+1)%3][(i+1)%3]*J[(A+2)%3][(i+2)%3]-J[(A+1)%3][(i+2)%3]*J[(A+2)%3][(i+1)%3])/det;if(!std::isfinite(D[i][A]))throw std::invalid_argument("nonfinite Hermite forward map");}
    for(int i=0;i<3;++i){normal[i]=p.x[i]/radius;for(int A=0;A<3;++A)for(int B=0;B<3;++B)for(int C=0;C<3;++C)for(int j=0;j<3;++j)for(int k=0;k<3;++k){if(!std::isfinite(p.H[C][j][k]))throw std::invalid_argument("nonfinite Hermite inverse Hessian");X[i][A][B]-=D[i][C]*p.H[C][j][k]*D[j][A]*D[k][B];}}
    RadialMapJet out;Real rA[3]{},norm0=0;
    for(int i=0;i<3;++i){for(int A=0;A<3;++A)rA[A]+=normal[i]*D[i][A];norm0+=D[i][0]*D[i][0];}out.rs=rA[0];
    for(int i=0;i<3;++i)out.rss+=normal[i]*X[i][0][0];out.rss+=(norm0-out.rs*out.rs)/radius;
    for(int A=1;A<3;++A){Real dot=0;for(int i=0;i<3;++i){out.rs_A[A]+=normal[i]*X[i][0][A];dot+=D[i][0]*D[i][A];}out.rs_A[A]+=(dot-out.rs*rA[A])/radius;}
    const Real tolerance=1e-10*std::max(Real(1),std::sqrt(norm0));
    if(!(out.rs>0&&std::isfinite(out.rs)&&std::isfinite(out.rss)))throw std::invalid_argument("Hermite radial map orientation");
    for(int A=1;A<3;++A)if(std::abs(rA[A])>tolerance||!std::isfinite(out.rs_A[A]))throw std::invalid_argument("Hermite map boundary is not a constant-radius surface");
    for(int i=0;i<3;++i)if(std::abs(D[i][0]-out.rs*normal[i])>tolerance)throw std::invalid_argument("Hermite radial direction is not a coordinate ray");
    Real normal_acceleration=0;for(int i=0;i<3;++i)normal_acceleration+=normal[i]*X[i][0][0];
    for(int i=0;i<3;++i){if(std::abs(X[i][0][0]-normal_acceleration*normal[i])>1e-10*std::max(Real(1),std::abs(out.rss)+norm0/radius))throw std::invalid_argument("Hermite radial coordinate curves off its ray");for(int A=1;A<3;++A)if(std::abs(X[i][0][A]-out.rs_A[A]*normal[i]-out.rs*D[i][A]/radius)>1e-10*std::max(Real(1),std::abs(out.rs_A[A])+std::abs(out.rs*D[i][A]/radius)))throw std::invalid_argument("Hermite mixed map jet is not a radial ray family");}
    return out;
  }
 private:
  struct RowStamp {int block,j,k;Point geometry;Real spacing;size_t terms;};
  struct Query {Point geometry;RadialMapJet radial;std::array<size_t,25> angular;size_t native;};
  struct GeometryFrame {std::vector<Query> native;std::vector<SurfacePolynomialWeights::Row> angular;unsigned long revision;};
  const ScriSurface &surface_;const CompositeDifference &grid_;const CompositeMesh mesh_;
  int n_,ng_,padded_,count_,order_;bool radial_stencils_;int nodes_,surface_components_;size_t surface_size_,beta_size_;
  std::vector<int> active_,base_;std::vector<Point> geometry_;std::vector<RowStamp> rows_;
  SurfacePolynomialWeights angular_;std::shared_ptr<EpochState> epoch_;
  Real dr_[8]{},ddr_[8]{},radial_condition_=0;
  mutable Diagnostics diagnostic_;
  mutable std::vector<SurfacePolynomialWeights::Row> angular_rows_;
  mutable std::map<std::array<Real,3>,size_t> angular_cache_;
  std::vector<Query> native_;
  std::shared_ptr<const GeometryFrame> frame_;
  const int*active_storage_=nullptr;const Point*geometry_storage_=nullptr;const ScriSurface::Row*row_storage_=nullptr;const int*base_storage_=nullptr;const CompositeBlock*block_storage_=nullptr;
  static size_t NoNative(){return std::numeric_limits<size_t>::max();}
  static std::vector<std::array<Real,3>> Coordinates(const ScriSurface&s){std::vector<std::array<Real,3>> out;for(const auto&r:s.rows)out.push_back({r.geometry.x[0],r.geometry.x[1],r.geometry.x[2]});return out;}
  int Index(int m,int i,int j,int k)const{return ((m*padded_+k+ng_)*padded_+j+ng_)*padded_+i+ng_;}
  static bool EqualPoint(const Point&a,const Point&b){static_assert(std::is_trivially_copyable<Point>::value&&sizeof(Point)==39*sizeof(Real),"Point geometry comparison layout");return std::memcmp(&a,&b,sizeof(Point))==0;}
  void ValidateInitial()const {
    if(n_<6||ng_<1||padded_!=n_+2*ng_||nodes_<6||nodes_>8||nodes_>n_+1||count_<=0||mesh_.cells!=n_||mesh_.ghosts!=ng_||!(order_==2||order_==4)||ng_<order_/2)throw std::invalid_argument("Hermite grid/radial contract");
    CompositeMesh valid(mesh_.half_width,mesh_.radii,mesh_.face_subdivisions,n_,ng_);(void)valid;
    const size_t cells=mesh_.blocks.size()*size_t(n_)*n_*n_,padded_count=mesh_.blocks.size()*size_t(padded_)*padded_*padded_;
    if(active_.size()!=cells||geometry_.size()!=cells||padded_count!=size_t(count_)||base_.size()!=mesh_.blocks.size())throw std::invalid_argument("Hermite grid descriptor/active shape");
    for(size_t p=0;p<cells;++p){const int m=p/(n_*n_*n_),i=p%n_,j=p/n_%n_,k=p/(n_*n_)%n_;const Real q[3]={(i+.5)/n_,(j+.5)/n_,(k+.5)/n_};if(active_[p]!=Index(m,i,j,k)||!EqualPoint(geometry_[p],mesh_.Geometry(m,q)))throw std::invalid_argument("Hermite active IDs/positions disagree with map");}
    size_t expected=0;for(size_t m=0;m<mesh_.blocks.size();++m){const auto&b=mesh_.blocks[m];if(b.face< -1||b.face>5||b.layer< -1||b.layer>=int(mesh_.radii.size())||(b.face<0)!=(b.layer<0))throw std::invalid_argument("Hermite patch identity");for(int A=0;A<3;++A)if(!(std::isfinite(b.lo[A])&&std::isfinite(b.hi[A])&&b.hi[A]>b.lo[A]))throw std::invalid_argument("Hermite patch bounds");if(b.layer==int(mesh_.radii.size())-1){if(base_[m]!=int(expected))throw std::invalid_argument("Hermite surface base ordering");expected+=size_t(n_)*n_;}else if(base_[m]!=-1)throw std::invalid_argument("Hermite nonboundary surface base");}
    if(expected!=surface_.rows.size())throw std::invalid_argument("Hermite surface row shape");
    for(size_t p=0;p<expected;++p){const auto&r=surface_.rows[p];if(r.block<0||r.block>=int(mesh_.blocks.size())||r.j<0||r.j>=n_||r.k<0||r.k>=n_||base_[r.block]+r.j+n_*r.k!=int(p)||!(r.radial_spacing>0))throw std::invalid_argument("Hermite surface row metadata");const Real q[3]={1,(r.j+.5)/n_,(r.k+.5)/n_};if(!EqualPoint(r.geometry,mesh_.Geometry(r.block,q)))throw std::invalid_argument("Hermite surface position/J/H disagree with map");RadialGeometry(r.geometry);}
  }
  void CheckGeometry()const {
    const auto&m=grid_.mesh;
    if(grid_.n!=n_||grid_.ng!=ng_||grid_.padded!=padded_||grid_.count!=count_||grid_.order!=order_||grid_.radial_interface_stencils!=radial_stencils_||m.revision!=mesh_.revision||m.half_width!=mesh_.half_width||m.radii!=mesh_.radii||m.cells!=mesh_.cells||m.ghosts!=mesh_.ghosts||m.face_subdivisions!=mesh_.face_subdivisions||m.blocks.size()!=mesh_.blocks.size()||grid_.active!=active_||grid_.geometry.size()!=geometry_.size()||surface_.radial_nodes!=nodes_||surface_.components!=surface_components_||surface_.base!=base_||surface_.rows.size()!=rows_.size()||surface_.values.size()!=surface_size_||surface_.beta.size()!=beta_size_)throw std::logic_error("stale Hermite geometry/descriptor");
    for(size_t b=0;b<m.blocks.size();++b){const auto&a=m.blocks[b];const auto&old=mesh_.blocks[b];if(a.face!=old.face||a.layer!=old.layer||a.refinement!=old.refinement)throw std::logic_error("stale Hermite block identity");for(int A=0;A<3;++A)if(a.lo[A]!=old.lo[A]||a.hi[A]!=old.hi[A])throw std::logic_error("stale Hermite block position");}
    static_assert(sizeof(Point)==39*sizeof(Real),"Point snapshot layout");if(!geometry_.empty()&&std::memcmp(grid_.geometry.data(),geometry_.data(),geometry_.size()*sizeof(Point)))throw std::logic_error("stale Hermite active position/J/H");
    for(size_t p=0;p<rows_.size();++p){const auto&r=surface_.rows[p];const auto&old=rows_[p];if(r.block!=old.block||r.j!=old.j||r.k!=old.k||r.radial_spacing!=old.spacing||r.terms.size()!=old.terms||!EqualPoint(r.geometry,old.geometry))throw std::logic_error("stale Hermite surface position/J/H");}
  }
  void CheckCheapGeometry()const {
    const auto&m=grid_.mesh;
    if(grid_.n!=n_||grid_.ng!=ng_||grid_.padded!=padded_||grid_.count!=count_||grid_.order!=order_||grid_.radial_interface_stencils!=radial_stencils_||m.revision!=mesh_.revision||m.half_width!=mesh_.half_width||m.radii!=mesh_.radii||m.cells!=mesh_.cells||m.ghosts!=mesh_.ghosts||m.face_subdivisions!=mesh_.face_subdivisions||m.blocks.size()!=mesh_.blocks.size()||grid_.active.size()!=active_.size()||grid_.geometry.size()!=geometry_.size()||surface_.radial_nodes!=nodes_||surface_.components!=surface_components_||surface_.base.size()!=base_.size()||surface_.rows.size()!=rows_.size()||surface_.values.size()!=surface_size_||surface_.beta.size()!=beta_size_||grid_.active.data()!=active_storage_||grid_.geometry.data()!=geometry_storage_||surface_.rows.data()!=row_storage_||surface_.base.data()!=base_storage_||m.blocks.data()!=block_storage_)throw std::logic_error("stale Hermite geometry/shape/revision");
  }
  void CheckBlock(int b)const {const auto&now=grid_.mesh.blocks.at(b);const auto&old=mesh_.blocks.at(b);if(now.face!=old.face||now.layer!=old.layer||now.refinement!=old.refinement||surface_.base[b]!=base_[b])throw std::logic_error("stale Hermite queried block");for(int A=0;A<3;++A)if(now.lo[A]!=old.lo[A]||now.hi[A]!=old.hi[A])throw std::logic_error("stale Hermite queried block position");}
  void CheckRow(size_t p)const {const auto&now=surface_.rows.at(p);const auto&old=rows_.at(p);CheckBlock(old.block);if(now.block!=old.block||now.j!=old.j||now.k!=old.k||now.radial_spacing!=old.spacing||now.terms.size()!=old.terms||!EqualPoint(now.geometry,old.geometry))throw std::logic_error("stale Hermite queried row");}
  void CheckPrepared(const Prepared&p)const {CheckCheapGeometry();const auto&data=p.Get();auto owner=data.owner.lock();if(!owner||owner.get()!=epoch_.get())throw std::logic_error("foreign Hermite Prepared");if(data.geometry.get()!=frame_.get())throw std::logic_error("foreign Hermite geometry frame");if(!epoch_->has_epoch||data.epoch!=epoch_->epoch)throw std::logic_error("stale Hermite Prepared epoch");}
  static Real RadialCondition(const std::vector<Real>&z){const int n=z.size();Real a[8][8]{},inverse[8][8]{},norm=0;for(int i=0;i<n;++i){Real sum=0;for(int k=0;k<n;++k){a[i][k]=std::pow(z[i],k);sum+=std::abs(a[i][k]);}norm=std::max(norm,sum);inverse[i][i]=1;}for(int k=0;k<n;++k){int pivot=k;for(int i=k+1;i<n;++i)if(std::abs(a[i][k])>std::abs(a[pivot][k]))pivot=i;if(std::abs(a[pivot][k])<1e-12)throw std::runtime_error("Hermite radial rank guard");for(int j=0;j<n;++j){std::swap(a[k][j],a[pivot][j]);std::swap(inverse[k][j],inverse[pivot][j]);}Real v=a[k][k];for(int j=0;j<n;++j){a[k][j]/=v;inverse[k][j]/=v;}for(int i=0;i<n;++i)if(i!=k){Real v=a[i][k];for(int j=0;j<n;++j){a[i][j]-=v*a[k][j];inverse[i][j]-=v*inverse[k][j];}}}Real inverse_norm=0;for(int i=0;i<n;++i){Real sum=0;for(int j=0;j<n;++j)sum+=std::abs(inverse[i][j]);inverse_norm=std::max(inverse_norm,sum);}return norm*inverse_norm;}
  void BuildRadialWeights(){Real radial[8];std::vector<Real> scaled;for(int r=0;r<nodes_;++r){radial[r]=r==nodes_-1?1:Real(n_-(nodes_-1)+r+.5)/n_;Jet basis(1);for(int k=0;k<nodes_;++k)if(k!=r){Real other=k==nodes_-1?1:Real(n_-(nodes_-1)+k+.5)/n_;basis=basis*(Jet::Variable(1,0)-Jet(other))/Jet(radial[r]-other);}dr_[r]=basis.d[0];ddr_[r]=basis.h[0][0];scaled.push_back((radial[r]-1)/(1-radial[0]));}const Real span=1-radial[0];for(int k=0;k<nodes_;++k){Real first=0,second=0;for(int r=0;r<nodes_-1;++r){Real v=k?std::pow((radial[r]-1)/span,k):0;first+=span*dr_[r]*v;second+=span*span*ddr_[r]*v;}diagnostic_.polynomial_reproduction=std::max({diagnostic_.polynomial_reproduction,std::abs(first-(k==1?1.:0)),std::abs(second-(k==2?2.:0))});}radial_condition_=RadialCondition(scaled);if(radial_condition_>=1e8||diagnostic_.polynomial_reproduction>1e-10)throw std::runtime_error("Hermite radial polynomial/condition guard");}
  void ValidateQuery(int block,const Real*q)const {if(block<0||block>=int(base_.size())||base_[block]<0||q[0]!=1||!std::isfinite(q[1])||!std::isfinite(q[2])||q[1]<0||q[1]>1||q[2]<0||q[2]>1)throw std::invalid_argument("Hermite boundary query coordinates");}
  size_t AngularRow(int block,const Real*q)const {std::array<Real,3>key{Real(block),q[1],q[2]};for(int A=1;A<3;++A){Real lattice=std::round(2*n_*q[A]);if(std::abs(2*n_*q[A]-lattice)<1e-12)key[A]=lattice/(2*n_);}auto found=angular_cache_.find(key);if(found!=angular_cache_.end())return found->second;Real local[3]={1,q[1],q[2]},x[3];mesh_.Forward(block,local,x);auto row=angular_.Build(x);diagnostic_.value_weight_l1=std::max(diagnostic_.value_weight_l1,row.l1);diagnostic_.factor_condition_indicator=std::max(diagnostic_.factor_condition_indicator,row.condition*radial_condition_);diagnostic_.polynomial_reproduction=std::max(diagnostic_.polynomial_reproduction,row.reproduction);if(diagnostic_.factor_condition_indicator>=1e8)throw std::runtime_error("Hermite angular/radial factor-condition guard");const size_t id=angular_rows_.size();angular_rows_.push_back(std::move(row));angular_cache_.emplace(key,id);diagnostic_.angular_plans=angular_rows_.size();return id;}
  Query BuildQuery(int block,const Real*q,size_t native)const {Query out;out.geometry=mesh_.Geometry(block,q);out.radial=RadialGeometry(out.geometry);out.native=native;const Real h=1./n_;for(int j=0;j<5;++j)for(int k=0;k<5;++k){Real point[3]={1,q[1]+(j-2)*h,q[2]+(k-2)*h};out.angular[5*j+k]=AngularRow(block,point);}return out;}
  Jets Evaluate(const Query&q,const StageData&data,const std::vector<SurfacePolynomialWeights::Row>&plans)const {
    const int fields=data.components;std::vector<Real> sample[5][5];for(int j=0;j<5;++j)for(int k=0;k<5;++k){const auto&row=plans[q.angular[5*j+k]];sample[j][k].resize(3*fields);for(int which=0;which<3;++which){const auto&values=which==0?data.trace:(which==1?data.G:data.H);for(int f=0;f<fields;++f){Real anchor=values[row.ids[0]*fields+f],value=anchor;for(size_t i=0;i<row.ids.size();++i)value+=row.weights[i]*(values[row.ids[i]*fields+f]-anchor);sample[j][k][which*fields+f]=value;}}}
    Jets out;out.value.resize(fields);out.d.resize(fields);out.dd.resize(fields);const Real h=1./n_,w[5]={1./12,-2./3,0,2./3,-1./12},w2[5]={-1./12,4./3,-2.5,4./3,-1./12};
    for(int f=0;f<fields;++f){const bool native=q.native!=NoNative();const Real F=native?data.trace[q.native*fields+f]:sample[2][2][f],G=native?data.G[q.native*fields+f]:sample[2][2][fields+f],H=native?data.H[q.native*fields+f]:sample[2][2][2*fields+f];out.value[f]=F;Real dq[3]={native?data.q0[q.native*fields+f]:q.radial.rs*G,0,0},ddq[3][3]{};ddq[0][0]=native?data.q00[q.native*fields+f]:q.radial.rs*q.radial.rs*H+q.radial.rss*G;
      Real dG[3]{};for(int j=0;j<5;++j){dq[1]+=w[j]*(sample[j][2][f]-F)/h;dq[2]+=w[j]*(sample[2][j][f]-F)/h;dG[1]+=w[j]*(sample[j][2][fields+f]-G)/h;dG[2]+=w[j]*(sample[2][j][fields+f]-G)/h;ddq[1][1]+=w2[j]*(sample[j][2][f]-F)/(h*h);ddq[2][2]+=w2[j]*(sample[2][j][f]-F)/(h*h);for(int k=0;k<5;++k)ddq[1][2]+=w[j]*w[k]*(sample[j][k][f]-F)/(h*h);}for(int A=1;A<3;++A)ddq[0][A]=ddq[A][0]=q.radial.rs*dG[A]+q.radial.rs_A[A]*G;ddq[2][1]=ddq[1][2];
      for(int i=0;i<3;++i)for(int A=0;A<3;++A){out.d[f][i]+=q.geometry.J[A][i]*dq[A];for(int j=0;j<3;++j){out.dd[f][i][j]+=q.geometry.H[A][i][j]*dq[A];for(int B=0;B<3;++B)out.dd[f][i][j]+=q.geometry.J[A][i]*q.geometry.J[B][j]*ddq[A][B];}}
      if(!std::isfinite(out.value[f]))throw std::runtime_error("nonfinite Hermite sampled value");for(int i=0;i<3;++i){if(!std::isfinite(out.d[f][i]))throw std::runtime_error("nonfinite Hermite sampled gradient");for(int j=0;j<3;++j)if(!std::isfinite(out.dd[f][i][j]))throw std::runtime_error("nonfinite Hermite sampled Hessian");}
    }return out;
  }
};
} // namespace cubed_sphere
#endif
