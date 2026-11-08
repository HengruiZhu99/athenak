//========================================================================================
// Native AthenaK restart payload for hyperboloidal Z4c.
// Licensed under the 3-clause BSD License (the "LICENSE").
//========================================================================================
#include "z4c_hyperboloidal/restart.hpp"

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>

#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "mesh/chart_atlas.hpp"
#include "z4c_hyperboloidal/z4c_hyperboloidal.hpp"

namespace z4c_hyperboloidal {
namespace restart {
namespace {
constexpr std::uint64_t magic=0x4859504552525354ULL;  // HYPERRST
constexpr std::uint64_t version=1;
constexpr int record_fields=Fields::count+8;
constexpr int header_words=8;

void RequireAll(bool good, const char *message) {
  int failed=good ? 0 : 1;
#if MPI_PARALLEL_ENABLED
  int any=0;
  MPI_Allreduce(&failed,&any,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
  failed=any;
#endif
  if (failed) throw std::runtime_error(std::string("Hyperboloidal restart: ")+message);
}

// Per-rank files carry independent copies of native mesh metadata. Compare
// those copies exactly before restoring any payload, so mixed checkpoint
// generations cannot silently supply different times, trees, or policies.
void RequireSameBytes(const void *data, IOWrapperSizeT size) {
#if MPI_PARALLEL_ENABLED
  IOWrapperSizeT root_size=size;
  MPI_Bcast(&root_size,sizeof(root_size),MPI_BYTE,0,MPI_COMM_WORLD);
  RequireAll(size==root_size,"mixed per-rank checkpoint metadata size");
  std::array<unsigned char,65536> root{};
  const auto *local=static_cast<const unsigned char*>(data);
  bool same=true;
  for (IOWrapperSizeT at=0;at<size;at+=root.size()) {
    const int count=static_cast<int>(std::min<IOWrapperSizeT>(root.size(),size-at));
    if (global_variable::my_rank==0) std::memcpy(root.data(),local+at,count);
    MPI_Bcast(root.data(),count,MPI_BYTE,0,MPI_COMM_WORLD);
    same=(std::memcmp(root.data(),local+at,count)==0) && same;
  }
  RequireAll(same,"mixed per-rank checkpoint metadata");
#else
  (void)data;
  (void)size;
#endif
}

void CheckPerRankConsistency(const Mesh &mesh, const std::string &identity) {
  // No struct padding is serialized into this comparison buffer.
  std::array<unsigned char,2*sizeof(Real)+3*sizeof(int)> header{};
  std::size_t at=0;
  const auto append=[&](const auto &value) {
    std::memcpy(header.data()+at,&value,sizeof(value));
    at+=sizeof(value);
  };
  append(mesh.time); append(mesh.dt); append(mesh.ncycle);
  append(mesh.nmb_total); append(mesh.root_level);
  RequireSameBytes(header.data(),header.size());
  const auto atlas=mesh.atlas->Identity();
  RequireSameBytes(atlas.data(),atlas.size());
  RequireSameBytes(identity.data(),identity.size());
  // LogicalLocation has an asserted contiguous five-int32 representation,
  // identical to native version-2 location records. These are mesh metadata,
  // never global field replicas; only a fixed 64 KiB comparison buffer is used.
  RequireSameBytes(mesh.lloc_eachmb,Multiply(mesh.nmb_total,sizeof(LogicalLocation)));
  RequireSameBytes(mesh.cost_eachmb,Multiply(mesh.nmb_total,sizeof(float)));
}

IOWrapperSizeT RecordScalars(const Z4cHyperboloidal &p) {
  IOWrapperSizeT bulk=1;
  for (int a=0;a<3;++a) bulk=Multiply(bulk,p.n[a]+2*p.overlap);
  const auto count=Multiply(Add(bulk,Multiply(p.n[1],p.n[2])),record_fields);
  if (count>INT_MAX) throw std::overflow_error("Hyperboloidal restart MPI count overflow");
  return count;
}

KOKKOS_INLINE_FUNCTION
void PackChi(const ChiRecord &r, Real *out) {
  out[0]=r.minimum; out[1]=r.accepted_minimum; out[2]=r.floor_count;
  out[3]=r.nonpositive_count; out[4]=r.nonfinite_count; out[5]=r.max_correction;
  out[6]=r.sum_correction; out[7]=r.first_floor;
}

KOKKOS_INLINE_FUNCTION
ChiRecord UnpackChi(const Real *in) {
  ChiRecord r;
  r.minimum=in[0]; r.accepted_minimum=in[1]; r.floor_count=in[2];
  r.nonpositive_count=in[3]; r.nonfinite_count=in[4]; r.max_correction=in[5];
  r.sum_correction=in[6]; r.first_floor=in[7];
  return r;
}

void ReadHeaderBytes(IOWrapper &file, void *data, std::size_t size, bool per_rank) {
  bool good=true;
  if (global_variable::my_rank==0 || per_rank) {
    good=file.Read_bytes(data,1,size,per_rank)==size;
  }
  RequireAll(good,"truncated metadata");
#if MPI_PARALLEL_ENABLED
  if (!per_rank) MPI_Bcast(data,size,MPI_BYTE,0,MPI_COMM_WORLD);
#endif
}
}  // namespace

IOWrapperSizeT Add(IOWrapperSizeT a, IOWrapperSizeT b) {
  constexpr auto limit=static_cast<IOWrapperSizeT>(std::numeric_limits<std::int64_t>::max());
  if (a>limit || b>limit-a) throw std::overflow_error("Restart byte offset overflow");
  return a+b;
}

IOWrapperSizeT Multiply(IOWrapperSizeT a, IOWrapperSizeT b) {
  constexpr auto limit=static_cast<IOWrapperSizeT>(std::numeric_limits<std::int64_t>::max());
  if (b && a>limit/b) throw std::overflow_error("Restart byte count overflow");
  return a*b;
}

IOWrapperSizeT RecordBytes(const Z4cHyperboloidal &p) {
  return Multiply(RecordScalars(p),sizeof(Real));
}

std::string Identity(const Z4cHyperboloidal &p) {
  std::ostringstream s;
  s.imbue(std::locale::classic());
  s << "native-hyperboloidal-v1;fields=" << Fields::count << ";chi-record=8"
    << ";formulation=n12-configured-constant-shift-v1;integrator=rk4_classical"
    << ";real=" << sizeof(Real) << ";n=" << p.n[0] << ',' << p.n[1] << ',' << p.n[2]
    << ";overlap=" << p.overlap << ";nghost=" << p.nghost << std::hexfloat;
#define HYP_ID(value) s << ";" #value "=" << value
  HYP_ID(p.ko); HYP_ID(p.chi_floor); HYP_ID(p.penalty); HYP_ID(p.requested_dt);
  const auto &g=p.parameters;
  HYP_ID(g.profile.flat); HYP_ID(g.profile.flat_radius); HYP_ID(g.profile.outer_radius);
  HYP_ID(g.profile.plateau); HYP_ID(g.enforce_cfl); HYP_ID(g.conformal_frame_halos);
  HYP_ID(g.separated_shift_speeds); HYP_ID(g.stationary_outer_reference);
  HYP_ID(g.reference_start_M); HYP_ID(g.reference_end_M); HYP_ID(g.reference_compact_end);
  HYP_ID(g.inner_begin); HYP_ID(g.inner_end); HYP_ID(g.inner_gamma);
  HYP_ID(g.inner_chi_guard); HYP_ID(g.cartesian_chi_guard);
  HYP_ID(g.coefficient_begin); HYP_ID(g.coefficient_end); HYP_ID(g.restoration_begin);
  HYP_ID(g.restoration_end); HYP_ID(g.bulk_mu); HYP_ID(g.ratio); HYP_ID(g.damping);
  HYP_ID(g.restoring); HYP_ID(g.outer_lapse_damping); HYP_ID(g.kappa); HYP_ID(g.tau);
  HYP_ID(g.epsilon); HYP_ID(g.inner_z4); HYP_ID(g.kappa1); HYP_ID(g.kappa2);
#undef HYP_ID
  return s.str();
}

IOWrapperSizeT WriteMetadata(IOWrapper &file, const Mesh &mesh, bool per_rank) {
  const auto &p=*mesh.pmb_pack->pz4c_hyperboloidal;
  const auto identity=Identity(p);
  const std::array<std::uint64_t,header_words> header={magic,version,identity.size(),
      RecordScalars(p),static_cast<std::uint64_t>(global_variable::nranks),
      static_cast<std::uint64_t>(mesh.gids_eachrank[global_variable::my_rank]),
      static_cast<std::uint64_t>(mesh.nmb_thisrank),static_cast<std::uint64_t>(per_rank)};
  bool good=true;
  if (global_variable::my_rank==0 || per_rank) {
    good=file.Write_any_type(header.data(),sizeof(header),"byte",per_rank)==sizeof(header);
    good=(file.Write_any_type(identity.data(),identity.size(),"byte",per_rank)
          ==identity.size()) && good;
  }
  RequireAll(good,"metadata write failed");
  return sizeof(header)+identity.size();
}

void ReadMetadata(IOWrapper &file, const Mesh &mesh, bool per_rank) {
  std::array<std::uint64_t,header_words> header{};
  ReadHeaderBytes(file,header.data(),sizeof(header),per_rank);
  RequireAll(header[0]==magic && header[1]==version && header[2]<=16384,
             "unsupported or corrupt metadata header");
  const auto &p=*mesh.pmb_pack->pz4c_hyperboloidal;
  RequireAll(header[3]==RecordScalars(p) && header[7]==static_cast<std::uint64_t>(per_rank),
             "record shape or file mode mismatch");
  RequireAll(!per_rank || (header[4]==static_cast<std::uint64_t>(global_variable::nranks) &&
      header[5]==static_cast<std::uint64_t>(mesh.gids_eachrank[global_variable::my_rank]) &&
      header[6]==static_cast<std::uint64_t>(mesh.nmb_thisrank)),
      "per-rank restart ownership changed; use a shared restart file to repartition");
  std::string identity(header[2],'\0');
  ReadHeaderBytes(file,identity.data(),identity.size(),per_rank);
  RequireAll(identity==Identity(p),"mathematical parameters or formulation mismatch");
  if (per_rank) CheckPerRankConsistency(mesh,identity);
}

IOWrapperSizeT ReadStride(IOWrapper &file, bool per_rank) {
  IOWrapperSizeT stride=0;
  ReadHeaderBytes(file,&stride,sizeof(stride),per_rank);
  return stride;
}

void Pack(const Z4cHyperboloidal &p, HostArray2D<Real> &host) {
  const auto count=RecordScalars(p);
  DvceArray2D<Real> packed("hyperboloidal restart pack",p.nblocks,count);
  Kokkos::deep_copy(packed,0.0);
  const int nx=p.n[0]+2*p.overlap, ny=p.n[1]+2*p.overlap;
  const int base=p.origin-p.overlap, snx=p.n[1];
  const auto bulk_slots=Multiply(Multiply(nx,ny),p.n[2]+2*p.overlap);
  auto points=p.points.d_view; auto surface=p.surface_points.d_view;
  auto face_blocks=p.face_blocks.d_view; auto u=p.u0; auto s=p.s0;
  auto chi=p.chi_records; auto schi=p.surface_chi_records;
  Kokkos::parallel_for("pack restart evolved bulk",Kokkos::RangePolicy<>(0,p.npoints),
    KOKKOS_LAMBDA(int q) {
      const auto c=points(q);
      const auto slot=((c.k-base)*ny+c.j-base)*nx+c.i-base;
      Real *out=&packed(c.m,record_fields*slot);
      for (int f=0;f<Fields::count;++f) out[f]=u(c.m,f,c.k,c.j,c.i);
      PackChi(chi(q),out+Fields::count);
    });
  Kokkos::parallel_for("pack restart owned scri",Kokkos::RangePolicy<>(0,p.nsurface_points),
    KOKKOS_LAMBDA(int q) {
      const auto c=surface(q);
      Real *out=&packed(face_blocks(c.face),record_fields*(bulk_slots+c.k*snx+c.j));
      for (int f=0;f<Fields::count;++f) out[f]=s(c.face,f,c.k,c.j);
      PackChi(schi(q),out+Fields::count);
    });
  Kokkos::realloc(host,p.nblocks,count);
  Kokkos::deep_copy(host,packed);
}

void Unpack(Z4cHyperboloidal &p, const HostArray2D<Real> &host) {
  p.InvalidateDiagnostics();
  const auto count=RecordScalars(p);
  bool good=host.extent(0)==static_cast<std::size_t>(p.nblocks) && host.extent(1)==count;
  if (good) for (std::size_t i=0;i<host.size();++i) {
    if (!std::isfinite(host.data()[i])) {good=false; break;}
  }
  RequireAll(good,"invalid shape or nonfinite checkpoint state");
  DvceArray2D<Real> packed("hyperboloidal restart unpack",p.nblocks,count);
  Kokkos::deep_copy(packed,host);
  const int nx=p.n[0]+2*p.overlap, ny=p.n[1]+2*p.overlap;
  const int base=p.origin-p.overlap, snx=p.n[1];
  const auto bulk_slots=Multiply(Multiply(nx,ny),p.n[2]+2*p.overlap);
  auto points=p.points.d_view; auto surface=p.surface_points.d_view;
  auto face_blocks=p.face_blocks.d_view; auto u=p.u0; auto s=p.s0;
  auto chi=p.chi_records; auto schi=p.surface_chi_records;
  Kokkos::deep_copy(u,std::numeric_limits<Real>::quiet_NaN());
  Kokkos::parallel_for("unpack restart evolved bulk",Kokkos::RangePolicy<>(0,p.npoints),
    KOKKOS_LAMBDA(int q) {
      const auto c=points(q);
      const auto slot=((c.k-base)*ny+c.j-base)*nx+c.i-base;
      const Real *in=&packed(c.m,record_fields*slot);
      for (int f=0;f<Fields::count;++f) u(c.m,f,c.k,c.j,c.i)=in[f];
      chi(q)=UnpackChi(in+Fields::count);
    });
  Kokkos::parallel_for("unpack restart owned scri",Kokkos::RangePolicy<>(0,p.nsurface_points),
    KOKKOS_LAMBDA(int q) {
      const auto c=surface(q);
      const Real *in=&packed(face_blocks(c.face),record_fields*(bulk_slots+c.k*snx+c.j));
      for (int f=0;f<Fields::count;++f) s(c.face,f,c.k,c.j)=in[f];
      schi(q)=UnpackChi(in+Fields::count);
    });
  Kokkos::fence();
  p.FormResidual();
}

void CheckFileExtent(IOWrapper &file, IOWrapperSizeT offset, IOWrapperSizeT stride,
                     IOWrapperSizeT expected_stride, int records, bool per_rank) {
  RequireAll(stride==expected_stride,"block record byte count mismatch");
  IOWrapperSizeT end=0;
  bool good=true;
  try {end=Add(offset,Multiply(stride,records));}
  catch (const std::overflow_error &) {good=false;}
  RequireAll(good,"block payload byte extent overflow");
  RequireAll(file.GetSize(per_rank)>=end,"truncated block payload");
}

void WriteRecords(IOWrapper &file, const Mesh &mesh, const HostArray2D<Real> &host,
                  IOWrapperSizeT offset, IOWrapperSizeT stride, bool per_rank) {
  const auto count=RecordScalars(*mesh.pmb_pack->pz4c_hyperboloidal);
  RequireAll(host.extent(0)==static_cast<std::size_t>(mesh.nmb_thisrank) &&
             host.extent(1)==count && stride>=Multiply(count,sizeof(Real)),
             "invalid output record shape");
  Add(offset,Multiply(stride,mesh.nmb_thisrank));
  int minimum=mesh.nmb_eachrank[0], maximum=minimum;
  for (int r=1;r<global_variable::nranks;++r) {
    minimum=std::min(minimum,mesh.nmb_eachrank[r]);
    maximum=std::max(maximum,mesh.nmb_eachrank[r]);
  }
  bool good=true;
  for (int m=0;m<maximum;++m) {
    if (m>=mesh.nmb_thisrank) continue;
    const auto at=Add(offset,Multiply(stride,m));
    const auto written=m<minimum ?
        file.Write_any_type_at_all(&host(m,0),count,at,"Real",per_rank) :
        file.Write_any_type_at(&host(m,0),count,at,"Real",per_rank);
    good=(written==count) && good;
  }
  RequireAll(good,"block payload write failed");
}

void ReadRecords(IOWrapper &file, Mesh &mesh, IOWrapperSizeT offset,
                 IOWrapperSizeT stride, bool per_rank) {
  auto &p=*mesh.pmb_pack->pz4c_hyperboloidal;
  const auto count=RecordScalars(p);
  HostArray2D<Real> host("hyperboloidal restart input",p.nblocks,count);
  int minimum=mesh.nmb_eachrank[0], maximum=minimum;
  for (int r=1;r<global_variable::nranks;++r) {
    minimum=std::min(minimum,mesh.nmb_eachrank[r]);
    maximum=std::max(maximum,mesh.nmb_eachrank[r]);
  }
  bool good=true;
  for (int m=0;m<maximum;++m) {
    if (m>=mesh.nmb_thisrank) continue;
    const auto at=Add(offset,Multiply(stride,m));
    const auto read=m<minimum ? file.Read_Reals_at_all(&host(m,0),count,at,per_rank) :
                               file.Read_Reals_at(&host(m,0),count,at,per_rank);
    good=(read==count) && good;
  }
  RequireAll(good,"truncated or unreadable block payload");
  // References are established before restoring state; no reference initializer is
  // allowed to run after unpacking the accepted solution and cumulative records.
  p.InitializeReference();
  Unpack(p,host);
}
}  // namespace restart
}  // namespace z4c_hyperboloidal
