// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include "outputs/scri.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fcntl.h>
#include <filesystem>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

#include "globals.hpp"
#include "scri_provenance.hpp"
#include "z4c_hyperboloidal/restart.hpp"
#include "mesh/mesh.hpp"
#include "parameter_input.hpp"
#include "z4c_hyperboloidal/mapped_halo.hpp"
#include "z4c_hyperboloidal/surface_exchange.hpp"
#include "z4c_hyperboloidal/scri_quadrature.hpp"
#include "z4c_hyperboloidal/scri_rate_diagnostic.hpp"

namespace hyp=z4c_hyperboloidal;
namespace {
void CollectiveCheck(bool ok,const std::string &why) {
  int bad=ok?0:1;
#if MPI_PARALLEL_ENABLED
  MPI_Allreduce(MPI_IN_PLACE,&bad,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
#endif
  if(bad)throw std::runtime_error("Scri output: "+(ok?"failure on another rank":why));
}
// Explicit scalars, never native struct layouts. Endian marker permits byte swapping.
class Archive {
 public:
  explicit Archive(const std::string &path) {
    int fd=open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL,0666);
    if(fd<0)throw std::runtime_error("cannot exclusively create "+path);
    file_=fdopen(fd,"wb");
    if(!file_) {close(fd);throw std::runtime_error("fdopen "+path);}
    Write("SCRI0001",8);Scalar<std::uint32_t>(0x01020304);
  }
  ~Archive() {if(file_)fclose(file_);}
  template<class T> void Scalar(T v) {Write(&v,sizeof(v));}
  void Write(const void *p,size_t n) {
    if(n && fwrite(p,1,n,file_)!=n)throw std::runtime_error("archive write failed");
  }
  template<class T> void Array(const std::string &name,std::uint32_t type,
                              const std::vector<std::uint64_t> &shape,
                              const std::vector<T> &v) {
    std::uint64_t count=1;for(auto d:shape)count*=d;
    if(count!=v.size())throw std::logic_error("scri array shape mismatch: "+name);
    Scalar<std::uint32_t>(name.size());Scalar(type);
    Scalar<std::uint32_t>(shape.size());
    for(auto d:shape)Scalar(d);
    Write(name.data(),name.size());Write(v.data(),v.size()*sizeof(T));
  }
  void Text(const std::string &name,const std::string &s) {
    Array(name,3,{s.size()},std::vector<char>(s.begin(),s.end()));
  }
  void Finish() {
    Scalar<std::uint32_t>(0);
    const int status=fclose(file_);file_=nullptr;
    if(status)throw std::runtime_error("archive close failed");
  }
 private:
  FILE *file_=nullptr;
};
struct Arrays {
  struct Entry {std::string name;std::vector<std::uint64_t> shape;std::vector<double> data;};
  std::vector<Entry> entries;
  void Add(std::string name,std::vector<std::uint64_t> shape,std::vector<double> values) {
    entries.push_back({std::move(name),std::move(shape),std::move(values)});
  }
  void Write(Archive &a) const {for(const auto &e:entries)a.Array(e.name,1,e.shape,e.data);}
};
void Jet(std::vector<double> &v,const hyp::mapped::Jet &j) {
  v.push_back(j.f);
  for(auto x:j.d)v.push_back(x);
  for(auto &r:j.h)for(auto x:r)v.push_back(x);
}
void Geometry(std::vector<double> &v,const hyp::mapped::Point &p) {
  for(auto x:p.x)v.push_back(x);
  for(auto &r:p.J)for(auto x:r)v.push_back(x);
  for(auto &s:p.H)for(auto &r:s)for(auto x:r)v.push_back(x);
}
void Map(std::vector<double> &v,const chart_atlas::BlockMap &m) {
  for(int x:{m.chart_id,m.face,m.layer})v.push_back(x);
  for(auto x:m.lo)v.push_back(x);for(auto x:m.hi)v.push_back(x);
  for(auto x:{m.half_width,m.inner_radius,m.outer_radius})v.push_back(x);
  for(bool x:{m.cube_transition,m.equiangular,m.outer_surface})v.push_back(x);
  for(auto x:m.node_offset)v.push_back(x);
  for(auto x:m.overlap_sides)v.push_back(x);
}
std::string Number(int n) {std::ostringstream s;s<<std::setw(8)<<std::setfill('0')<<n;return s.str();}
}  // namespace

struct ScriOutput::Impl {
  bool nodal=true,modes=true,static_due=false,skip=false,rate_diagnostics=false;
  int lmax=8,sequence=-1,last_cycle=-1;
  Real last_time=-1;
  std::string epoch,parameters,identity,units;
  std::unique_ptr<hyp::SurfaceExchange> boundary;
  std::unique_ptr<bvals::MappedExchange> bulk,edge;
  DvceArray5D<Real> staging;
  DvceArray2D<Real> bulk_ref,edge_ref,bulk_res,edge_res,harmonics;
  DvceArray1D<Real> bulk_weight;
  Arrays geometry,data;
  std::vector<double> coefficients;
  double prepare_seconds=0;

  void Rebuild(Mesh *pm,const std::string &prefix) {
    auto *pack=pm->pmb_pack;auto &p=*pack->pz4c_hyperboloidal;
    boundary=std::make_unique<hyp::SurfaceExchange>(pack,&p,true);
    bulk=std::make_unique<bvals::MappedExchange>();
    edge=std::make_unique<bvals::MappedExchange>();
    std::vector<int> fields(hyp::Fields::count);std::iota(fields.begin(),fields.end(),0);
    bulk->Rebuild(pack,boundary->BulkExchange().UniqueDonors(),fields);
    edge->Rebuild(pack,boundary->EdgeExchange().UniqueDonors(),fields);
    // MappedExchange sorts/deduplicates. These lists were already canonical.
    Kokkos::realloc(staging,p.nblocks,hyp::Fields::count,p.n[2],p.n[1],1);
    geometry.entries.clear();
    hyp::NativeDonorSelector selector(*pm,p.origin,p.activity);
    hyp::SphericalReference3D reference(p.parameters);
    const auto support=[&](const std::string &name,const bvals::MappedExchange &ex,
                            bool surface,DvceArray2D<Real> &ref) {
      const auto &donors=ex.UniqueDonors();const std::uint64_t count=donors.size();
      std::vector<double> ids,maps,coords,refs,weights,omega;
      Kokkos::realloc(ref,count,hyp::Fields::count);
      auto hr=Kokkos::create_mirror_view(ref);
      for(size_t i=0;i<count;++i) {
        const auto d=donors[i];const auto &map=selector.Map(d.gid);
        const Real q[3]={surface?1:(d.i-p.origin+map.node_offset[0])/p.n[0],
          (d.j-(surface?0:p.origin)+map.node_offset[1])/p.n[1],
          (d.k-(surface?0:p.origin)+map.node_offset[2])/p.n[2]};
        const auto g=map.Geometry(q);const auto r=reference.At(g.x,surface);
        for(int x:{d.gid,d.k,d.j,d.i,pm->rank_eachmb[d.gid]})ids.push_back(x);
        Map(maps,map);for(auto x:q)coords.push_back(x);Geometry(coords,g);
        for(int f=0;f<hyp::Fields::count;++f) {Jet(refs,r[f]);hr(i,f)=r[f].f;}
        weights.push_back(surface?1:p.parameters.residual_weight.Value(g.x));
        const auto o=p.parameters.profile.At(g.x);
        hyp::mapped::Jet oj(o.omega);
        for(int a=0;a<3;++a) {oj.d[a]=o.gradient[a];for(int b=0;b<3;++b)oj.h[a][b]=o.hessian[a][b];}
        Jet(omega,oj);
      }
      geometry.Add(name+"_ids",{count,5},std::move(ids));
      geometry.Add(name+"_maps",{count,24},std::move(maps));
      geometry.Add(name+"_geometry",{count,42},std::move(coords));
      geometry.Add(name+"_reference",{count,31,13},std::move(refs));
      geometry.Add(name+"_omega",{count,13},std::move(omega));
      geometry.Add(name+"_residual_weight",{count},weights);
      Kokkos::deep_copy(ref,hr);
      if(!surface) {
        Kokkos::realloc(bulk_weight,count);auto hw=Kokkos::create_mirror_view(bulk_weight);
        for(size_t i=0;i<count;++i)hw(i)=weights[i];Kokkos::deep_copy(bulk_weight,hw);
      }
    };
    support("bulk",*bulk,false,bulk_ref);support("edge",*edge,true,edge_ref);
    Kokkos::realloc(bulk_res,bulk_ref.extent(0),hyp::Fields::independent_count);
    Kokkos::realloc(edge_res,edge_ref.extent(0),hyp::Fields::independent_count);
    const auto rows=Kokkos::create_mirror_view_and_copy(HostMemSpace(),boundary->Rows());
    const auto terms=Kokkos::create_mirror_view_and_copy(HostMemSpace(),boundary->Terms());
    const auto sg=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.surface_geometry);
    const auto sr=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.surface_reference);
    std::vector<double> row_data,term_ids,term_jets,ids,coordinates,refs,weights,face_data;
    int resolution=1000000000;const int n=p.n[0],nm=modes?(lmax+1)*(lmax+1):0;
    for(int face=0;face<p.nfaces;++face) {
      int gid=pack->gids+p.face_blocks.h_view(face);const auto &map=selector.Map(gid);
      face_data.push_back(gid);
      face_data.push_back((map.outer_radius-map.inner_radius)*(map.hi[0]-map.lo[0])/n);
      face_data.push_back(1-map.node_offset[0]);Map(face_data,map);
      for(int a=1;a<3;++a)resolution=std::min(resolution,int(std::lround(2*n/(map.hi[a]-map.lo[a]))));
    }
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE,&resolution,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
#endif
    if(modes && lmax>=resolution)throw std::invalid_argument("scri lmax must be less than coarsest angular cells per cube face");
    Kokkos::realloc(harmonics,p.nsurface_points,modes?2*nm:0);
    auto yh=Kokkos::create_mirror_view(harmonics);
    for(int i=0;i<p.nsurface_points;++i) {
      const auto c=p.surface_points.h_view(i);int gid=pack->gids+p.face_blocks.h_view(c.face);
      for(int x:{gid,c.face,c.k,c.j,global_variable::my_rank})ids.push_back(x);
      Geometry(coordinates,sg(i));for(const auto &j:sr(i))Jet(refs,j);
      const auto row=rows(i);for(double x:{double(row.begin),double(row.end),double(row.spacing)})row_data.push_back(x);
      const auto &map=selector.Map(gid);double weight=hyp::ScriSolidAngleWeight(map,n,c.j,c.k);
      weights.push_back(weight);
      if(modes)for(int l=0;l<=lmax;++l)for(int m=-l;m<=l;++m) {
        const auto y=hyp::ScriScalarHarmonic(l,m,sg(i).x);
        const int k=l*l+l+m;yh(i,2*k)=weight*y[0];yh(i,2*k+1)=-weight*y[1];
      }
    }
    for(size_t i=0;i<terms.extent(0);++i) {
      const auto t=terms(i);term_ids.push_back(t.boundary?1:0);term_ids.push_back(t.ordinal);Jet(term_jets,t.weight);
    }
    geometry.Add("surface_ids",{std::uint64_t(p.nsurface_points),5},std::move(ids));
    geometry.Add("surface_geometry",{std::uint64_t(p.nsurface_points),39},std::move(coordinates));
    geometry.Add("surface_reference",{std::uint64_t(p.nsurface_points),31,13},std::move(refs));
    geometry.Add("surface_weights",{std::uint64_t(p.nsurface_points)},std::move(weights));
    geometry.Add("faces",{std::uint64_t(p.nfaces),27},std::move(face_data));
    geometry.Add("rows",{rows.extent(0),3},std::move(row_data));
    geometry.Add("term_ids",{terms.extent(0),2},std::move(term_ids));
    geometry.Add("term_weights",{terms.extent(0),13},std::move(term_jets));
    geometry.Add("configuration",{9},{double(n),double(p.origin),double(p.overlap),double(p.nghost),
        double(p.parameters.scri_treatment),p.parameters.residual_weight.inner,p.parameters.residual_weight.outer,
        double(sizeof(Real)),double(resolution)});
    const auto &source=p.boundary_source;
    geometry.Add("boundary_source",{7},{source.alpha,source.shift_ratio,source.lapse_offset,
        source.lapse_linear,source.lapse_quadratic,source.shift_offset,source.scri_obstruction_reference});
    const auto &o=p.parameters.profile;
    geometry.Add("conformal_profile",{4},{double(o.flat),o.flat_radius,o.outer_radius,o.plateau});
    Kokkos::deep_copy(harmonics,yh);
    sequence=pm->GetAMRLoadBalanceUpdateSeq();epoch=prefix+".epoch.scri";static_due=true;
    identity=hyp::restart::Identity(p);
  }
};

ScriOutput::ScriOutput(ParameterInput *pin,Mesh *pm,OutputParameters op)
    :BaseTypeOutput(pin,pm,op),impl_(std::make_unique<Impl>()) {
  if(!pm->pmb_pack->pz4c_hyperboloidal)throw std::invalid_argument("scri output requires z4c_hyperboloidal");
  auto &s=*impl_;s.nodal=pin->GetOrAddBoolean(op.block_name,"nodal",true);
  s.modes=pin->GetOrAddBoolean(op.block_name,"modes",true);
  s.lmax=pin->GetOrAddInteger(op.block_name,"lmax",8);
  s.rate_diagnostics=pin->GetOrAddBoolean(op.block_name,"boundary_diagnostics",false);
  if(s.rate_diagnostics && !s.nodal)throw std::invalid_argument("boundary diagnostics require nodal output");
  s.units=pin->GetOrAddString(op.block_name,"unit_description","geometric code units; c=G=1; physical mass scale unspecified");
  if(!s.nodal && !s.modes)throw std::invalid_argument("scri output requires nodal or modes");
  if(s.modes && (s.lmax<0 || s.lmax>128))throw std::invalid_argument("scri lmax must lie in [0,128]");
  if(op.gid>=0 || op.slice1 || op.slice2 || op.slice3 || op.include_gzs)
    throw std::invalid_argument("scri output does not support gid, slices or ghost_zones");
  s.last_cycle=pin->GetOrAddInteger(op.block_name,"scri_last_cycle",-1);
  s.last_time=pin->GetOrAddReal(op.block_name,"scri_last_time",-1);
  std::ostringstream params;pin->ParameterDump(params);s.parameters=params.str();
}
ScriOutput::~ScriOutput()=default;

void ScriOutput::LoadOutputData(Mesh *pm) {
  auto &s=*impl_;s.skip=pm->ncycle==s.last_cycle && pm->time==s.last_time
      && (!s.boundary || s.sequence==pm->GetAMRLoadBalanceUpdateSeq());
  if(s.skip)return;
  Kokkos::Timer timer;auto &p=*pm->pmb_pack->pz4c_hyperboloidal;
  const std::string prefix=out_params.file_basename+"."+out_params.file_id+".o"+
      std::to_string(out_params.block_number)+"."+Number(out_params.file_number)+".r"+Number(global_variable::my_rank);
  if(!s.boundary || s.sequence!=pm->GetAMRLoadBalanceUpdateSeq())s.Rebuild(pm,prefix);
  const auto points=p.surface_points.d_view;const auto owners=p.face_blocks.d_view;
  const auto state=p.s0;const auto staging=s.staging;
  Kokkos::parallel_for("scri output pack accepted surface",Kokkos::RangePolicy<>(0,p.nsurface_points),
    KOKKOS_LAMBDA(int i) {const auto c=points(i);for(int f=0;f<31;++f)staging(owners(c.face),f,c.k,c.j,0)=state(c.face,f,c.k,c.j);});
  s.bulk->StartReceives();s.edge->StartReceives();
  s.bulk->PackAndSend(p.u0);s.edge->PackAndSend(staging);
  while(s.bulk->Receive()!=TaskStatus::complete) {}
  while(s.edge->Receive()!=TaskStatus::complete) {}
  const auto residual=[&](const bvals::MappedExchange &ex,DvceArray2D<Real> ref,
                          DvceArray2D<Real> dst,bool surface) {
    const auto values=ex.Values();const auto weight=s.bulk_weight;
    Kokkos::parallel_for("scri output residual support",Kokkos::RangePolicy<>(0,values.extent_int(0)),
      KOKKOS_LAMBDA(int i) {int slot=0;for(int f=0;f<31;++f)if(hyp::Fields::Independent(f))
        dst(i,slot++)=(values(i,f)-ref(i,f))/(!surface && f==0?weight(i):1);});
  };
  residual(*s.bulk,s.bulk_ref,s.bulk_res,false);residual(*s.edge,s.edge_ref,s.edge_res,true);
  bool valid=true;std::string error;
  try {s.boundary->Sample(s.bulk_res,s.edge_res);s.boundary->Solve();}
  catch(const std::exception &e) {valid=false;error=e.what();}
  while(s.bulk->ClearSend()!=TaskStatus::complete) {}
  while(s.edge->ClearSend()!=TaskStatus::complete) {}
  while(s.bulk->ClearRecv()!=TaskStatus::complete) {}
  while(s.edge->ClearRecv()!=TaskStatus::complete) {}
  CollectiveCheck(valid,error);
  s.data.entries.clear();
  if(s.nodal) {
    const auto save=[&](std::string name,bvals::MappedExchange::ConstValues values) {
      auto host=Kokkos::create_mirror_view_and_copy(HostMemSpace(),values);std::vector<double> v;
      v.reserve(host.size());for(size_t i=0;i<host.extent(0);++i)for(int f=0;f<31;++f)v.push_back(host(i,f));
      s.data.Add(name,{host.extent(0),31},std::move(v));
    };
    save("bulk_state",s.bulk->Values());save("edge_state",s.edge->Values());
    // Only the small surface is mirrored, never the volume.
    auto h=Kokkos::create_mirror_view_and_copy(HostMemSpace(),state);
    auto correction=Kokkos::create_mirror_view_and_copy(HostMemSpace(),s.boundary->Correction());
    auto sampled=Kokkos::create_mirror_view_and_copy(HostMemSpace(),s.boundary->SampledJets());
    auto sg=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.surface_geometry);
    auto cj=Kokkos::create_mirror_view_and_copy(HostMemSpace(),s.boundary->CorrectionJets());
    auto kernel=s.boundary->BindKernel();
    auto dr=Kokkos::create_mirror_view_and_copy(HostMemSpace(),kernel.face_spacing);
    auto gap=Kokkos::create_mirror_view_and_copy(HostMemSpace(),kernel.face_gap);
    auto refs=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.surface_reference);
    auto rows=Kokkos::create_mirror_view_and_copy(HostMemSpace(),s.boundary->Rows());
    std::vector<double> values,lifts,jets,lift_jets,metric_rates,lapse_terms,raw_rates,penalty_rates;
    const int nc=correction.extent_int(1);
    for(int i=0;i<p.nsurface_points;++i) {
      const auto c=p.surface_points.h_view(i);
      for(int f=0;f<31;++f) {values.push_back(h(c.face,f,c.k,c.j));Jet(jets,sampled(i)[f]);}
      for(int a=0;a<nc;++a) {
        lifts.push_back(correction(c.face,a,c.k,c.j));
        const auto lift=hyp::NormalLift(sg(i),cj(c.face,a,c.k,c.j),dr(c.face),gap(c.face));
        Jet(lift_jets,lift);
      }
    }
    if(s.rate_diagnostics) {
      for(int i=0;i<p.nsurface_points;++i) {
        auto restored=sampled(i);const auto c=p.surface_points.h_view(i);
        if(!hyp::SphericalPolicy::Restore(sg(i).x,restored,refs(i)))
          throw std::runtime_error("invalid diagnostic restoration");
        for(int a=0;a<nc;++a)restored[a<3?22+a:18]=restored[a<3?22+a:18]+
          hyp::NormalLift(sg(i),cj(c.face,a,c.k,c.j),dr(c.face),gap(c.face));
        const auto d=hyp::DiagnoseScriRate(sg(i),rows(i).spacing,restored,p.parameters,p.boundary_source,p.penalty);
        metric_rates.insert(metric_rates.end(),d.metric_rate.begin(),d.metric_rate.end());
        lapse_terms.insert(lapse_terms.end(),d.lapse.begin(),d.lapse.end());
        raw_rates.insert(raw_rates.end(),d.unprojected.begin(),d.unprojected.end());
        penalty_rates.insert(penalty_rates.end(),d.penalty_rate.begin(),d.penalty_rate.end());
      }
      const auto nq=std::uint64_t(p.nsurface_points);
      s.data.Add("projected_metric_rates",{nq,11},std::move(metric_rates));
      s.data.Add("lapse_identity_terms",{nq,12},std::move(lapse_terms));
      s.data.Add("unprojected_boundary_rates",{nq,31},std::move(raw_rates));
      s.data.Add("boundary_penalty_rates",{nq,31},std::move(penalty_rates));
    }
    s.data.Add("surface_state",{std::uint64_t(p.nsurface_points),31},std::move(values));
    s.data.Add("normal_lift_jets",{std::uint64_t(p.nsurface_points),std::uint64_t(nc),13},std::move(lift_jets));
    s.data.Add("normal_correction",{std::uint64_t(p.nsurface_points),std::uint64_t(nc)},std::move(lifts));
    // Redundant second jets audit the exact saved donor reconstruction.
    s.data.Add("sampled_residual_jets",{std::uint64_t(p.nsurface_points),31,13},std::move(jets));
  }
  if(s.modes) {
    const int nm=(s.lmax+1)*(s.lmax+1),nq=p.nsurface_points;
    const auto harmonics=s.harmonics;
    DvceArray1D<Real> coefficients("scri modes",31*nm*2);
    using Policy=Kokkos::TeamPolicy<>;using Member=Policy::member_type;
    Kokkos::parallel_for("scri native modes",Policy(31*nm*2,Kokkos::AUTO),KOKKOS_LAMBDA(const Member &team) {
      const int index=team.league_rank(),f=index/(nm*2),k=index%(nm*2);Real value=0;
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team,nq),[&](int i,Real &sum) {
        const auto c=points(i);sum+=harmonics(i,k)*state(c.face,f,c.k,c.j);
      },value);
      Kokkos::single(Kokkos::PerTeam(team),[&]() {coefficients(index)=value;});
    });
    auto h=Kokkos::create_mirror_view_and_copy(HostMemSpace(),coefficients);
    s.coefficients.assign(h.data(),h.data()+h.size());
#if MPI_PARALLEL_ENABLED
    MPI_Allreduce(MPI_IN_PLACE,s.coefficients.data(),s.coefficients.size(),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
#endif
  }
  Kokkos::fence();s.prepare_seconds=timer.seconds();
}

void ScriOutput::WriteOutputFile(Mesh *pm,ParameterInput *pin) {
  auto &s=*impl_;if(s.skip)return;
  Kokkos::Timer timer;bool valid=true;std::string error;
  try {
    if(s.static_due) {
      Archive a(s.epoch);a.Text("kind","native scri topology epoch");
      a.Text("parameters",s.parameters);a.Text("initial_data_identity",s.identity);
      a.Text("execution_space",DevExeSpace::name());a.Text("units",s.units);a.Text("source_manifest",SCRI_SOURCE_MANIFEST);
      a.Text("basis","common Cartesian; scalar harmonics; Condon-Shortley phase; native simulation coordinates and time");
      a.Text("build",std::string(__VERSION__)+"; Real="+std::to_string(sizeof(Real))+"; MPI="+std::to_string(MPI_PARALLEL_ENABLED)+"; compiled="+__DATE__+" "+__TIME__);
      std::string fields;for(auto name:hyp::Fields::names)fields+=std::string(name)+"\n";
      a.Text("fields",fields);s.geometry.Write(a);a.Finish();
    }
    const std::string file=out_params.file_basename+"."+out_params.file_id+".o"+
      std::to_string(out_params.block_number)+"."+Number(out_params.file_number)+".r"+Number(global_variable::my_rank)+".scri";
    Archive a(file);a.Text("kind","native scri accepted state");a.Text("epoch",std::filesystem::path(s.epoch).filename().string());
    a.Array<std::int64_t>("record",2,{8},{pm->ncycle,out_params.file_number,s.sequence,global_variable::my_rank,
        global_variable::nranks,s.nodal?1:0,s.modes?1:0,s.lmax});
    a.Array<double>("time",1,{1},{pm->time});s.data.Write(a);
    if(s.modes && global_variable::my_rank==0)a.Array("native_modes",1,{31,std::uint64_t((s.lmax+1)*(s.lmax+1)),2},s.coefficients);
    a.Array<double>("preparation_seconds",1,{1},{s.prepare_seconds});a.Finish();
  } catch(const std::exception &e) {valid=false;error=e.what();}
  CollectiveCheck(valid,error);
  s.static_due=false;s.last_cycle=pm->ncycle;s.last_time=pm->time;
  ++out_params.file_number;
  if(out_params.last_time<0)out_params.last_time=pm->time;else out_params.last_time+=out_params.dt;
  pin->SetInteger(out_params.block_name,"file_number",out_params.file_number);
  pin->SetRealExact(out_params.block_name,"last_time",out_params.last_time);
  pin->SetInteger(out_params.block_name,"scri_last_cycle",s.last_cycle);
  pin->SetRealExact(out_params.block_name,"scri_last_time",s.last_time);
  if(global_variable::my_rank==0)std::cout<<"scri output "<<out_params.block_name<<" cycle="<<pm->ncycle
      <<" prepare_seconds="<<s.prepare_seconds<<" write_seconds="<<timer.seconds()<<std::endl;
}
