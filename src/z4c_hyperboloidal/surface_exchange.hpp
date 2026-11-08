#ifndef Z4C_HYPERBOLOIDAL_SURFACE_EXCHANGE_HPP_
#define Z4C_HYPERBOLOIDAL_SURFACE_EXCHANGE_HPP_
//========================================================================================
// AthenaK astrophysical plasma code
// Copyright(C) 2026 AthenaK contributors
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================

#include "bvals/mapped_exchange.hpp"
#include "z4c_hyperboloidal/endpoint.hpp"
#include "z4c_hyperboloidal/z4c_hyperboloidal.hpp"

class MeshBlockPack;

namespace z4c_hyperboloidal {

// A scri patch belongs to its outermost native MeshBlock. Only cross-patch
// derivative donors communicate; the normal solve and its polynomial extension
// use that block's own surface patch and require no correction all-gather.
class SurfaceExchange {
 public:
  struct Row {int begin, end; Real spacing;};
  struct Term {int ordinal; bool boundary; mapped::Jet weight;};
  struct RadialTerm {
    int ordinal;
    bool boundary;
    BlockPoint local;
    mapped::Jet weight;
  };
  struct FieldSlots {int index[Fields::count];};

  struct Kernel {
    DvceArray1D<int> radial_index;
    DvceArray1D<Row> radial_rows;
    DvceArray1D<RadialTerm> radial_terms;
    DvceArray1D<BlockPoint> points;
    DvceArray1D<int> block_faces;
    DvceArray4D<Real> correction;
    DvceArray1D<Real> face_spacing, face_gap;
    bvals::MappedExchange::ConstValues surface_values;
    FieldSlots fields;
    int n, origin;

    KOKKOS_INLINE_FUNCTION bool HasRadialRow(int point) const {
      return radial_index(point)>=0;
    }
    // Bulk endpoint rows have their value at the central bulk node. Cartesian
    // derivative rows use six interior radial nodes and the actual endpoint.
    // The angular bulk ghosts must already have been interpolated by MappedHalo.
    KOKKOS_INLINE_FUNCTION mapped::Jet ResidualJet(int point,int field,
        const DvceArray5D<Real> &residual,bool second=true) const {
      mapped::Jet out;
      if (!Fields::Independent(field)) return out;
      const auto c=points(point);
      out.f=residual(c.m,field,c.k,c.j,c.i);
      const auto row=radial_rows(radial_index(point));
      for (int t=row.begin;t<row.end;++t) {
        const auto term=radial_terms(t); const auto a=term.local;
        const Real delta=(term.boundary ? surface_values(term.ordinal,fields.index[field])
            : residual(a.m,field,a.k,a.j,a.i))-out.f;
        for (int i=0;i<3;++i) {
          out.d[i]+=term.weight.d[i]*delta;
          if (second)for (int j=0;j<3;++j)out.h[i][j]+=term.weight.h[i][j]*delta;
        }
      }
      return out;
    }
    KOKKOS_INLINE_FUNCTION mapped::Jet EndpointResidualJet(int point,int field,
        const DvceArray5D<Real> &residual) const {
      return ResidualJet(point,field,residual,
          field<=6||field==18||(field>=22&&field<25));
    }
    KOKKOS_INLINE_FUNCTION mapped::Jet NormalCorrection(int point,
        const mapped::Point &map,int axis) const {
      const auto c=points(point); const int face=block_faces(c.m);
      if (face<0 || c.i-origin<n-2) return {};
      return Correction(face,c.j-origin,c.k-origin,map,axis);
    }
    KOKKOS_INLINE_FUNCTION mapped::Jet Correction(int face,int j,int k,
        const mapped::Point &map,int axis) const {
      const int first[2]={j-2<0 ? 0 : (j-2>n-6 ? n-6 : j-2),
                          k-2<0 ? 0 : (k-2>n-6 ? n-6 : k-2)};
      const Real q[2]={(j+.5)/n,(k+.5)/n};
      mapped::Jet basis[2][6];
      for (int A=0;A<2;++A)for (int i=0;i<6;++i) {
        basis[A][i]=mapped::Jet(1);
        for (int l=0;l<6;++l)if (l!=i)
          basis[A][i]=basis[A][i]*(mapped::Jet::Variable(q[A],A+1)
              -mapped::Jet((first[A]+l+.5)/n))/mapped::Jet(Real(i-l)/n);
      }
      mapped::Jet logical;
      for (int v=0;v<6;++v)for (int u=0;u<6;++u)
        logical=logical+basis[0][u]*basis[1][v]
            *mapped::Jet(correction(face,axis,first[1]+v,first[0]+u));
      return NormalLift(map,logical,face_spacing(face),face_gap(face));
    }
  };

  // Explicit evolution-only view. BindKernel retains direct sampling for
  // diagnostics and other consumers that have not refreshed this stage cache.
  struct CachedKernel : Kernel {
    DvceArray2D<mapped::Jet> radial_jets;
    KOKKOS_INLINE_FUNCTION mapped::Jet EndpointResidualJet(int point,int field,
        const DvceArray5D<Real> &) const {
      if (!Fields::Independent(field))return mapped::Jet{};
      return radial_jets(radial_index(point),fields.index[field]);
    }
  };

  SurfaceExchange(MeshBlockPack *pack,Z4cHyperboloidal *module);
  void Rebuild();
  TaskStatus Project();
  TaskStatus StartReceives();
  TaskStatus PackAndSend();
  TaskStatus Receive();
  TaskStatus Sample();
  TaskStatus Solve();
  TaskStatus Evaluate();
  TaskStatus ClearSend();
  TaskStatus ClearRecv();
  Kernel BindKernel() const;
  // Requires current interpolated bulk ghosts and received scri residuals.
  // Refresh on every Surface RHS use; the returned view is stage-local data.
  CachedKernel PrepareRadialJets();

  const DvceArray1D<CartesianChartJets> &SampledJets() const {return sampled_;}
  const DvceArray4D<Real> &Correction() const {return correction_;}
  const DvceArray1D<Row> &Rows() const {return rows_;}
  const DvceArray1D<Term> &Terms() const {return terms_;}
  const bvals::MappedExchange &BulkExchange() const {return bulk_exchange_;}
  const bvals::MappedExchange &EdgeExchange() const {return surface_exchange_;}
  Real MaxCompatibility() const {return max_compatibility_;}
  Real MaxClock() const {return max_clock_;}

 private:
  MeshBlockPack *pack_;
  Z4cHyperboloidal *module_;
  bvals::MappedExchange bulk_exchange_, surface_exchange_;
  FieldSlots fields_;
  DvceArray1D<Row> rows_, radial_rows_;
  DvceArray1D<Term> terms_;
  DvceArray1D<RadialTerm> radial_terms_;
  DvceArray1D<int> radial_index_;
  DvceArray1D<int> radial_points_, radial_fields_;
  DvceArray2D<mapped::Jet> radial_jets_;
  DvceArray1D<Real> face_spacing_, face_gap_;
  DvceArray1D<CartesianChartJets> sampled_;
  DvceArray4D<Real> correction_;
  // MappedExchange addresses native GID-local-block indices. This thin surface
  // staging view has one radial slot per local block, never a global state copy.
  DvceArray5D<Real> surface_staging_;
  Real max_compatibility_=0,max_clock_=0;
  bool ready_=false;
  void CheckReady() const;
};

}  // namespace z4c_hyperboloidal
#endif  // Z4C_HYPERBOLOIDAL_SURFACE_EXCHANGE_HPP_
