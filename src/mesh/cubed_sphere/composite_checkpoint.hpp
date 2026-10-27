// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef MESH_CUBED_SPHERE_COMPOSITE_CHECKPOINT_HPP_
#define MESH_CUBED_SPHERE_COMPOSITE_CHECKPOINT_HPP_

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <type_traits>
#include "mesh/cubed_sphere/composite.hpp"

namespace cubed_sphere {
struct CompositeSnapshot {
  CompositeMesh mesh;
  std::string source_tag;
  double dt = 0, tolerance = 0;
  std::uint64_t step = 0;
  int components = 0;
  std::vector<double> fields, shift_warm_start;
  int spatial_order = 4;
  bool telegraph = true;
  int boundary_scheme = 0;
  double boundary_pole_sat = 4, boundary_regularity_sat = 8;
  std::vector<double> boundary_fields;
  bool wormhole_chart = false;
  int isotropic_reference = 0;
  int scri_radial_points = 6;
  bool radial_interface_stencils = false;
  int positive_storage=0;
  bool enriched_log_derivatives=false;
  bool radial_line_smoothing=false;
  bool topology_coarsening=false;
  bool log_halo_enrichment=false;
  int puncture_derivative_scheme=0;
  int boundary_derivative_scheme=0;
  bool affine_shift_derivatives=false;
  Real log_halo_radius=.12; // v13 support was fixed at .12; v14 binds the radius
  Real telegraph_width=.08; // versions2..8 used the historical narrow default
};

// Explicit version/endian/layout contract; shell, bulk and new flux fields all
// live in the same active-state payload. Hierarchies/halos are NEVER serialized:
// the reader rebuilds them from the checked leaf descriptor before stage use.
class CompositeCheckpoint {
  struct Buffer {
    std::vector<char> bytes;
    size_t cursor = 0;
    template <typename T> void Put(const T &value) {
      static_assert(std::is_arithmetic_v<T>);
      const auto *data = reinterpret_cast<const char *>(&value);
      bytes.insert(bytes.end(), data, data+sizeof(T));
    }
    template <typename T> T Get() {
      if (cursor+sizeof(T) > bytes.size()) throw std::runtime_error("truncated composite checkpoint");
      T value; std::memcpy(&value, bytes.data()+cursor, sizeof(T)); cursor += sizeof(T);
      return value;
    }
    void PutVector(const std::vector<double> &v) {
      Put<std::uint64_t>(v.size()); for (double value : v) Put(value);
    }
    std::vector<double> GetVector(size_t expected) {
      if (Get<std::uint64_t>() != expected) throw std::runtime_error("checkpoint field layout mismatch");
      std::vector<double> v(expected);
      for (auto &value : v) {
        value = Get<double>();
        if (!std::isfinite(value)) throw std::runtime_error("checkpoint nonfinite field");
      }
      return v;
    }
  };
  static std::uint64_t Hash(const std::vector<char> &bytes, size_t length) {
    std::uint64_t hash = 14695981039346656037ull;
    for (size_t i = 0; i < length; ++i) {hash ^= static_cast<unsigned char>(bytes[i]); hash *= 1099511628211ull;}
    return hash;
  }
 public:
  static void Write(const std::string &path, const CompositeSnapshot &snapshot) {
    const auto &mesh = snapshot.mesh;
    const size_t count = mesh.blocks.size()*mesh.cells*mesh.cells*mesh.cells;
    size_t face_cells=0;for(const auto &block:mesh.blocks)
      if(block.layer==static_cast<int>(mesh.radii.size())-1)face_cells+=mesh.cells*mesh.cells;
    if (snapshot.source_tag.size() != 64 || snapshot.components < 1
        || snapshot.fields.size() != count*snapshot.components || snapshot.shift_warm_start.size() != 3*count
        || !(snapshot.dt > 0 && snapshot.tolerance > 0)
        || (snapshot.boundary_scheme!=0&&snapshot.boundary_scheme!=1)
        || snapshot.boundary_fields.size()!=(snapshot.boundary_scheme?face_cells*snapshot.components:0)
        || !(snapshot.boundary_pole_sat>=0&&snapshot.boundary_regularity_sat>=0)
        || !std::isfinite(snapshot.boundary_pole_sat+snapshot.boundary_regularity_sat)
        || (snapshot.wormhole_chart&&snapshot.boundary_scheme!=1)
        || (snapshot.isotropic_reference&&!snapshot.wormhole_chart)
        || snapshot.isotropic_reference<0||snapshot.isotropic_reference>2
        || snapshot.scri_radial_points<6||snapshot.scri_radial_points>8||snapshot.scri_radial_points>mesh.cells+1
        ||snapshot.positive_storage<0||snapshot.positive_storage>3||(!snapshot.wormhole_chart&&snapshot.positive_storage)
        ||!(snapshot.telegraph_width>0&&snapshot.telegraph_width<.395)
        ||(snapshot.enriched_log_derivatives&&snapshot.positive_storage!=2)
        ||(snapshot.log_halo_enrichment&&(!snapshot.enriched_log_derivatives||snapshot.positive_storage!=2))
        ||!(snapshot.log_halo_radius>0)||!std::isfinite(snapshot.log_halo_radius)
        ||(snapshot.log_halo_enrichment&&snapshot.log_halo_radius>mesh.half_width)
        ||snapshot.puncture_derivative_scheme<0||snapshot.puncture_derivative_scheme>1
        ||(snapshot.puncture_derivative_scheme&&(!snapshot.enriched_log_derivatives||snapshot.positive_storage!=2||!snapshot.wormhole_chart))
        ||snapshot.boundary_derivative_scheme<0||snapshot.boundary_derivative_scheme>1
        ||(snapshot.boundary_derivative_scheme&&snapshot.boundary_scheme!=1)
        ||(snapshot.affine_shift_derivatives&&snapshot.boundary_scheme!=1))
      throw std::invalid_argument("invalid composite checkpoint state");
    Buffer b;
    b.Put<std::uint64_t>(0x4353484946543031ull); b.Put<std::uint32_t>(17); b.Put<std::uint32_t>(0x01020304);
    for (char c : snapshot.source_tag) b.Put<char>(c);
    b.Put(mesh.half_width); b.Put(mesh.face_subdivisions); b.Put(mesh.cells); b.Put(mesh.ghosts);
    b.Put<std::uint64_t>(mesh.revision); b.PutVector(mesh.radii);
    b.Put<std::uint64_t>(mesh.blocks.size());
    for (const auto &block : mesh.blocks) {
      b.Put(block.face); b.Put(block.layer); b.Put(block.refinement);
      for (int a = 0; a < 3; ++a) {b.Put(block.lo[a]); b.Put(block.hi[a]);}
    }
    b.Put(snapshot.dt); b.Put(snapshot.tolerance); b.Put(snapshot.step); b.Put(snapshot.components);
    b.Put(snapshot.spatial_order); b.Put<int>(snapshot.telegraph ? 1 : 0);
    b.PutVector(snapshot.fields); b.PutVector(snapshot.shift_warm_start);
    b.Put(snapshot.boundary_scheme); b.Put(snapshot.boundary_pole_sat); b.Put(snapshot.boundary_regularity_sat);
    b.PutVector(snapshot.boundary_fields);
    b.Put<int>(snapshot.wormhole_chart?1:0);
    b.Put<int>(snapshot.isotropic_reference);
    b.Put(snapshot.scri_radial_points);b.Put<int>(snapshot.radial_interface_stencils?1:0);
    b.Put(snapshot.positive_storage);b.Put(snapshot.telegraph_width);b.Put<int>(snapshot.enriched_log_derivatives?1:0);b.Put<int>(snapshot.radial_line_smoothing?1:0);b.Put<int>(snapshot.topology_coarsening?1:0);b.Put<int>(snapshot.log_halo_enrichment?1:0);b.Put(snapshot.log_halo_radius);b.Put(snapshot.puncture_derivative_scheme);
    b.Put(snapshot.boundary_derivative_scheme);
    b.Put<int>(snapshot.affine_shift_derivatives?1:0);
    b.Put(Hash(b.bytes, b.bytes.size()));
    const std::string temporary = path+".tmp";
    {std::ofstream out(temporary, std::ios::binary|std::ios::trunc);
      out.write(b.bytes.data(), b.bytes.size());
      out.close(); if (!out) throw std::runtime_error("composite checkpoint write failed");}
    std::filesystem::rename(temporary, path);
  }
  static CompositeSnapshot Read(const std::string &path, const std::string &source_tag) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("missing composite checkpoint");
    Buffer b; b.bytes.assign(std::istreambuf_iterator<char>(in), {});
    if (b.bytes.size() < 128 || b.bytes.size() > 512ull*1024*1024)
      throw std::runtime_error("checkpoint length outside bounded contract");
    std::uint64_t stored; std::memcpy(&stored, b.bytes.data()+b.bytes.size()-8, 8);
    if (stored != Hash(b.bytes, b.bytes.size()-8)) throw std::runtime_error("composite checkpoint checksum mismatch");
    if (b.Get<std::uint64_t>() != 0x4353484946543031ull) throw std::runtime_error("checkpoint magic mismatch");
    const auto version=b.Get<std::uint32_t>();
    if ((version<2||version>17)||b.Get<std::uint32_t>()!=0x01020304)
      throw std::runtime_error("checkpoint format/endian mismatch");
    std::string tag;
    for (int i = 0; i < 64; ++i) tag += b.Get<char>();
    if (tag != source_tag) throw std::runtime_error("checkpoint executable/source binding mismatch");
    const double half_width = b.Get<double>();
    const int subdivisions = b.Get<int>(), cells = b.Get<int>(), ghosts = b.Get<int>();
    const auto revision = b.Get<std::uint64_t>();
    const auto shell_count = b.Get<std::uint64_t>();
    if (shell_count == 0 || shell_count > 64) throw std::runtime_error("checkpoint shell-count bound");
    std::vector<Real> radii;
    for (size_t i = 0; i < shell_count; ++i) radii.push_back(b.Get<double>());
    if (cells < 2 || cells > 64 || ghosts < 1 || ghosts > 8 || subdivisions < 1 || subdivisions > 16)
      throw std::runtime_error("checkpoint cell/halo/subdivision bound");
    CompositeMesh mesh(half_width, radii, subdivisions, cells, ghosts);
    const auto blocks = b.Get<std::uint64_t>();
    if (blocks == 0 || blocks > 4096) throw std::runtime_error("checkpoint block-count bound");
    mesh.blocks.clear(); mesh.revision = revision;
    for (size_t m = 0; m < blocks; ++m) {
      CompositeBlock block;
      block.face = b.Get<int>(); block.layer = b.Get<int>(); block.refinement = b.Get<int>();
      for (int a = 0; a < 3; ++a) {
        block.lo[a] = b.Get<double>(); block.hi[a] = b.Get<double>();
        if (!std::isfinite(block.lo[a]) || !std::isfinite(block.hi[a]) || !(block.hi[a] > block.lo[a]))
          throw std::runtime_error("checkpoint invalid leaf bounds");
      }
      if (block.face < -1 || block.face > 5 || block.layer < -1
          || block.layer >= static_cast<int>(radii.size()) || (block.face < 0) != (block.layer < 0))
        throw std::runtime_error("checkpoint invalid patch identity");
      mesh.blocks.push_back(block);
    }
    CompositeSnapshot out{std::move(mesh), tag};
    out.dt = b.Get<double>(); out.tolerance = b.Get<double>();
    out.step = b.Get<std::uint64_t>(); out.components = b.Get<int>();
    out.spatial_order = b.Get<int>();
    const int telegraph = b.Get<int>(); out.telegraph = telegraph == 1;
    if ((out.spatial_order != 2 && out.spatial_order != 4) || (telegraph != 0 && telegraph != 1))
      throw std::runtime_error("checkpoint operator/gauge configuration mismatch");
    const size_t count = blocks*cells*cells*cells;
    if (out.components < 1 || out.components > 64 || count*out.components > 32000000
        || !(out.dt > 0 && out.tolerance > 0) || !std::isfinite(out.dt+out.tolerance))
      throw std::runtime_error("checkpoint state/epoch bound");
    out.fields = b.GetVector(count*out.components); out.shift_warm_start = b.GetVector(3*count);
    if(version>=3){
      out.boundary_scheme=b.Get<int>();out.boundary_pole_sat=b.Get<double>();out.boundary_regularity_sat=b.Get<double>();
      size_t face_cells=0;for(const auto &block:out.mesh.blocks)
        if(block.layer==static_cast<int>(out.mesh.radii.size())-1)face_cells+=cells*cells;
      if((out.boundary_scheme!=0&&out.boundary_scheme!=1)||out.boundary_pole_sat<0||out.boundary_regularity_sat<0
          ||!std::isfinite(out.boundary_pole_sat+out.boundary_regularity_sat))
        throw std::runtime_error("checkpoint scri closure configuration mismatch");
      out.boundary_fields=b.GetVector(out.boundary_scheme==1?face_cells*out.components:0);
    }
    if(version>=4){const int chart=b.Get<int>();if((chart!=0&&chart!=1)||(chart&&out.boundary_scheme!=1))
      throw std::runtime_error("checkpoint puncture chart configuration mismatch");out.wormhole_chart=chart==1;}
    if(version>=5){const int ref=b.Get<int>();if((ref<0||ref>(version>=6?2:1))||(ref&&!out.wormhole_chart))
      throw std::runtime_error("checkpoint isotropic reference configuration mismatch");out.isotropic_reference=ref;}
    if(version>=7){out.scri_radial_points=b.Get<int>();const int radial=b.Get<int>();
      if(out.scri_radial_points<6||out.scri_radial_points>8||out.scri_radial_points>cells+1||(radial!=0&&radial!=1))
        throw std::runtime_error("checkpoint surface/halo stencil mismatch");out.radial_interface_stencils=radial==1;}
    if(version>=8){out.positive_storage=b.Get<int>();if(out.positive_storage<0||out.positive_storage>3||(!out.wormhole_chart&&out.positive_storage))
      throw std::runtime_error("checkpoint positive storage mismatch");}
    if(version>=9){out.telegraph_width=b.Get<Real>();if(!(out.telegraph_width>0&&out.telegraph_width<.395))
      throw std::runtime_error("checkpoint telegraph taper mismatch");}
    if(version>=10){const int enriched=b.Get<int>();if((enriched!=0&&enriched!=1)||(enriched&&out.positive_storage!=2))
      throw std::runtime_error("checkpoint enrichment mismatch");out.enriched_log_derivatives=enriched;}
    if(version>=11){const int lines=b.Get<int>();if(lines!=0&&lines!=1)throw std::runtime_error("checkpoint smoother mismatch");out.radial_line_smoothing=lines;}
    if(version>=12){const int topology=b.Get<int>();if(topology!=0&&topology!=1)throw std::runtime_error("checkpoint hierarchy mismatch");out.topology_coarsening=topology;}
    if(version>=13){const int halos=b.Get<int>();if((halos!=0&&halos!=1)||(halos&&(!out.enriched_log_derivatives||out.positive_storage!=2)))throw std::runtime_error("checkpoint log halo mismatch");out.log_halo_enrichment=halos;}
    if(version>=14){out.log_halo_radius=b.Get<Real>();if(!(out.log_halo_radius>0)||!std::isfinite(out.log_halo_radius)
      ||(out.log_halo_enrichment&&out.log_halo_radius>out.mesh.half_width))throw std::runtime_error("checkpoint log halo radius mismatch");}
    if(version>=15){out.puncture_derivative_scheme=b.Get<int>();if(out.puncture_derivative_scheme<0||out.puncture_derivative_scheme>1
      ||(out.puncture_derivative_scheme&&(!out.enriched_log_derivatives||out.positive_storage!=2||!out.wormhole_chart)))throw std::runtime_error("checkpoint puncture derivative scheme mismatch");}
    if(version>=16){out.boundary_derivative_scheme=b.Get<int>();if(out.boundary_derivative_scheme<0||out.boundary_derivative_scheme>1
      ||(out.boundary_derivative_scheme&&out.boundary_scheme!=1))throw std::runtime_error("checkpoint boundary derivative scheme mismatch");}
    if(version>=17){const int affine=b.Get<int>();if(affine<0||affine>1||(affine&&out.boundary_scheme!=1))throw std::runtime_error("checkpoint affine shift derivative mismatch");out.affine_shift_derivatives=affine;}
    if (b.cursor+8 != b.bytes.size()) throw std::runtime_error("checkpoint unexpected trailing payload");
    return out;
  }
};
}  // namespace cubed_sphere
#endif  // MESH_CUBED_SPHERE_COMPOSITE_CHECKPOINT_HPP_
