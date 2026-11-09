// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Full N12 scri/adjacent-radial operators against the independent legacy model.
// Global reference arrays exist only in this regression, never in native tasks.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "z4c_hyperboloidal/mapped_halo.hpp"
#include "z4c_hyperboloidal/surface_exchange.hpp"
#include "z4c_hyperboloidal/endpoint_state.hpp"
#include "z4c/hyperboloidal/central_z4c_configuration.hpp"
#include "z4c/hyperboloidal/spherical_transfer_3d.hpp"

namespace native=z4c_hyperboloidal;
namespace old=z4c::hyperboloidal;
constexpr int n=12,nf=31,nedge=6*n*n,jet_width=13;
void Require(bool yes,const char *what) {if (!yes)throw std::runtime_error(what);}

double Perturb(const double *x,int f,int scenario) {
  if (!scenario)return 0;
  return 2e-6*(f+1)/nf*(x[0]+.3*x[1]*x[1]-.2*x[2]*x[2]*x[2]
      +.1*x[0]*x[1]+sin(3*x[2]));
}
template<class J>
void WriteJet(std::vector<Real> &values,size_t index,const J &jet) {
  values[jet_width*index]=jet.f;
  for (int a=0;a<3;++a) {
    values[jet_width*index+1+a]=jet.d[a];
    for (int b=0;b<3;++b)values[jet_width*index+4+3*a+b]=jet.h[a][b];
  }
}
// Exercise the explicit evolution cache without changing the default direct
// endpoint path used by Diagnostics' separate SurfaceExchange instance.
std::uint64_t CheckEndpointCache(native::SurfaceExchange &surface,
                                native::Z4cHyperboloidal &p,std::uint64_t &cache_rows,
                                std::uint64_t &cache_bytes) {
  auto cached=surface.PrepareRadialJets();
  Require(cached.correction_jets.size()==3*std::size_t(p.nsurface_points),
          "endpoint view did not bind refreshed normal correction jets");
  cache_rows=cached.radial_jets.extent(0);
  cache_bytes=cached.radial_jets.span()*sizeof(native::mapped::Jet);
  native::mapped::Jet poison(std::numeric_limits<Real>::quiet_NaN());
  for (int a=0;a<3;++a) {
    poison.d[a]=poison.f;
    for (int b=0;b<3;++b)poison.h[a][b]=poison.f;
  }
  Kokkos::deep_copy(cached.radial_jets,poison);
  Kokkos::deep_copy(cached.correction_jets,poison);
  const auto direct=surface.BindKernel();
  const auto cells=p.points.d_view;const auto geometry=p.geometry;
  const auto reference=p.reference;const auto residual=p.residual;
  const auto parameters=p.parameters;const int n=p.n[0];
  DvceArray1D<native::PointState> expected("direct endpoint with poisoned cache",
                                         direct.radial_rows.extent(0));
  Kokkos::parallel_for("oracle direct endpoint ignores cache",
    Kokkos::RangePolicy<>(0,p.npoints),KOKKOS_LAMBDA(int point) {
      if (!direct.HasRadialRow(point))return;
      const auto c=cells(point);const Real inv_h[3]={Real(n),Real(n),Real(n)};
      expected(direct.radial_index(point))=native::EndpointState(point,
          native::BlockPointReader{residual,c.m,c.k,c.j,c.i},geometry(point),
          inv_h,reference(point),parameters,direct);
    });
  surface.Solve();
  cached=surface.PrepareRadialJets();
  DvceArray1D<native::PointState> actual("cached endpoint",direct.radial_rows.extent(0));
  Kokkos::parallel_for("oracle cached endpoint after poison",
    Kokkos::RangePolicy<>(0,p.npoints),KOKKOS_LAMBDA(int point) {
      if (!cached.HasRadialRow(point))return;
      const auto c=cells(point);const Real inv_h[3]={Real(n),Real(n),Real(n)};
      actual(cached.radial_index(point))=native::EndpointState(point,
          native::BlockPointReader{residual,c.m,c.k,c.j,c.i},geometry(point),
          inv_h,reference(point),parameters,cached);
    });
  const auto a=Kokkos::create_mirror_view_and_copy(HostMemSpace(),actual);
  const auto b=Kokkos::create_mirror_view_and_copy(HostMemSpace(),expected);
  std::uint64_t compared=0;
  const auto equal=[&](Real x,Real y) {
    Require(std::isfinite(x)&&std::isfinite(y)&&std::memcmp(&x,&y,sizeof(Real))==0,
            "cached/direct endpoint state differs after cache poison");++compared;
  };
  for (size_t row=0;row<a.extent(0);++row) {
    Require(a(row).status==native::PointStatus::success && a(row).status==b(row).status,
            "endpoint state failed after cache poison");
    for (int f=0;f<nf;++f) {
      equal(a(row).jets[f].f,b(row).jets[f].f);
      equal(a(row).transport[f],b(row).transport[f]);
      for (int i=0;i<3;++i) {
        equal(a(row).jets[f].d[i],b(row).jets[f].d[i]);
        for (int j=0;j<3;++j)equal(a(row).jets[f].h[i][j],b(row).jets[f].h[i][j]);
      }
    }
  }
  return compared;
}

struct Error {
  double scaled=0,absolute=0;
  std::uint64_t count=0,failures=0;
  int worst_field=-1;
  void Compare(double actual,double expected,int field) {
    Require(std::isfinite(actual) && std::isfinite(expected),"nonfinite surface comparison");
    const double difference=std::abs(actual-expected),e=difference/(1+std::abs(expected));
    if (e>scaled) {scaled=e;worst_field=field;}
    absolute=std::max(absolute,difference);++count;
    if (difference>1e-12+5e-10*std::abs(expected))++failures;
  }
  template<class J>
  void CompareJet(const J &actual,const std::vector<Real> &expected,size_t index,int field) {
    Compare(actual.f,expected[jet_width*index],field);
    for (int a=0;a<3;++a) {
      Compare(actual.d[a],expected[jet_width*index+1+a],field);
      for (int b=0;b<3;++b)Compare(actual.h[a][b],expected[jet_width*index+4+3*a+b],field);
    }
  }
  void Global() {
    MPI_Allreduce(MPI_IN_PLACE,&scaled,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&absolute,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&count,1,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,&failures,1,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
  }
};

int main(int argc,char **argv) {
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);
  Kokkos::initialize(argc,argv);
  try {
    {
      ParameterInput pin;IOWrapper file;file.Open(argv[1],IOWrapper::FileMode::read);
      pin.LoadFromFile(file);file.Close();
      Mesh mesh(&pin);mesh.BuildTreeFromScratch(&pin);mesh.AddCoordinatesAndPhysics(&pin);
      mesh.pgen=std::make_unique<ProblemGenerator>(&pin,&mesh);
      auto &p=*mesh.pmb_pack->pz4c_hyperboloidal;
      native::SurfaceExchange surface(mesh.pmb_pack,&p);
      // Fixed weights must be rebuilt with the plan, including same-size
      // rebuilds and ranks with no outer face. The endpoint oracle below still
      // compares against the original on-demand angular polynomial.
      native::mapped::Jet weight_poison(std::numeric_limits<Real>::quiet_NaN());
      for (int a=0;a<3;++a) {
        weight_poison.d[a]=weight_poison.f;
        for (int b=0;b<3;++b)weight_poison.h[a][b]=weight_poison.f;
      }
      Kokkos::deep_copy(surface.CorrectionWeights(),weight_poison);
      surface.Rebuild();
      const auto weights=Kokkos::create_mirror_view_and_copy(
          HostMemSpace(),surface.CorrectionWeights());
      Require(weights.size()==(p.nfaces>0 ? std::size_t(n*n*36) : 0),
              "fixed angular weight allocation differs");
      for (std::size_t i=0;i<weights.size();++i) {
        const auto &w=weights.data()[i];
        Require(std::isfinite(w.f),"fixed weight value not refreshed");
        for (int a=0;a<3;++a) {
          Require(std::isfinite(w.d[a]),"fixed weight gradient not refreshed");
          for (int b=0;b<3;++b)
            Require(std::isfinite(w.h[a][b]),"fixed weight Hessian not refreshed");
        }
      }
      native::MappedHalo halo;halo.Rebuild(mesh.pmb_pack,p.origin);
      std::unique_ptr<old::SphericalTransfer3D> reference;
      if (!global_variable::my_rank) {
        cubed_sphere::CompositeMesh oldmesh(.1,
            {.16,.24,.32,.40,.48,.56,.64,.72,.80,.88,.94,1},1,n,4,3,.08);
        oldmesh.evolved_overlap=3;oldmesh.RefineCartesian(13);
        oldmesh.amr_cartesian_core=true;oldmesh.outer_radial_endpoint_grid=true;
        old::CentralZ4cConfiguration config;config.inner_z4c_radius_M=20;
        config.compact_slicing_end=.97;auto parameters=config.Resolve().gauge;
        parameters.outer_lapse_damping=.25;
        reference=std::make_unique<old::SphericalTransfer3D>(std::move(oldmesh),parameters);
        Require(reference->grid.active.size()==229392 && reference->scri.rows.size()==nedge,
                "legacy oracle is not the pinned N12 mesh");
      }
      Error projection_error,sample_error,radial_error,correction_error,rate_error;
      double maximum_defect=0;std::uint64_t zero_faces=p.nfaces==0,cache_compared=0,
          cache_rows=0,cache_bytes=0;
      MPI_Allreduce(MPI_IN_PLACE,&zero_faces,1,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
      for (int scenario=0;scenario<2;++scenario) {
        auto state=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.u0);
        auto edge=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.s0);
        const auto geom=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.geometry);
        const auto refs=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.reference);
        const auto sgeom=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.surface_geometry);
        const auto srefs=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.surface_reference);
        Kokkos::deep_copy(state,std::numeric_limits<Real>::quiet_NaN());
        for (int point=0;point<p.npoints;++point) {
          const auto c=p.points.h_view(point);const auto &x=geom(point).x;
          const double r2=x[0]*x[0]+x[1]*x[1]+x[2]*x[2];
          for (int f=0;f<nf;++f)state(c.m,f,c.k,c.j,c.i)=refs(point)[f].f
              +(native::Fields::Independent(f) ? Perturb(x,f,scenario)*(f==0 ? r2 : 1) : 0);
        }
        for (int point=0;point<p.nsurface_points;++point) {
          const auto c=p.surface_points.h_view(point);
          for (int f=0;f<nf;++f)edge(c.face,f,c.k,c.j)=srefs(point)[f].f
              +(native::Fields::Independent(f) ? Perturb(sgeom(point).x,f,scenario) : 0);
        }
        Kokkos::deep_copy(p.u0,state);Kokkos::deep_copy(p.s0,edge);
        Kokkos::deep_copy(p.residual,std::numeric_limits<Real>::quiet_NaN());
        p.EnforceAlgebraicConstraints();surface.Project();p.FormResidual();
        surface.StartReceives();halo.StartReceives();
        surface.PackAndSend();halo.PackAndSend(p.residual,p.u0);
        halo.FillLocal(p.residual,p.u0);
        while (surface.Receive()!=TaskStatus::complete) {}
        surface.Sample();
        // Diagnostic-only normal solves must preserve the module RHS. Final
        // evaluation must subsequently overwrite every poisoned output field.
        Kokkos::deep_copy(p.s_rhs,std::numeric_limits<Real>::quiet_NaN());
        native::SpatialGeometry<Real> poison{};
        poison.valid=true;
        poison.determinant=poison.scalar=std::numeric_limits<Real>::quiet_NaN();
        for (int a=0;a<3;++a) {
          poison.contracted[a]=poison.scalar;
          for (int b=0;b<3;++b) {
            poison.inverse[a][b]=poison.ricci[a][b]=poison.dcontracted[a][b]=poison.scalar;
            for (int c=0;c<3;++c)poison.connection[a][b][c]=poison.scalar;
          }
        }
        Require(surface.MetricGeometry().extent(0)==p.nsurface_points &&
                surface.PenroseGeometry().extent(0)==p.nsurface_points,
                "surface geometry cache allocation differs");
        Kokkos::deep_copy(surface.MetricGeometry(),poison);
        Kokkos::deep_copy(surface.PenroseGeometry(),poison);
        native::mapped::Jet correction_poison(poison.scalar);
        for (int a=0;a<3;++a) {
          correction_poison.d[a]=poison.scalar;
          for (int b=0;b<3;++b)correction_poison.h[a][b]=poison.scalar;
        }
        Kokkos::deep_copy(surface.CorrectionJets(),correction_poison);
        surface.Solve();
        const auto untouched=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.s_rhs);
        for (std::size_t i=0;i<untouched.size();++i)
          Require(std::isnan(untouched.data()[i]),"normal solve overwrote module RHS");
        surface.Evaluate();
        maximum_defect=std::max(maximum_defect,surface.MaxCompatibility());
        while (halo.Receive()!=TaskStatus::complete) {}
        halo.FillRemote(p.residual,p.u0);
        cache_compared+=CheckEndpointCache(surface,p,cache_rows,cache_bytes);

        std::vector<Real> projected(nedge*nf),samples(nedge*nf*jet_width),
            radial(3*nedge*nf*jet_width),correction(3*nedge),rate(nedge*nf);
        if (reference) {
          auto &model=*reference;
          for (size_t point=0;point<model.grid.active.size();++point) {
            const auto &x=model.grid.geometry[point].x;
            const double r2=x[0]*x[0]+x[1]*x[1]+x[2]*x[2];
            for (int f=0;f<nf;++f)model.state[nf*point+f]=model.refs[point][f].f
                +(old::CartesianIndependent(f) ? Perturb(x,f,scenario)*(f==0 ? r2 : 1) : 0);
          }
          for (int point=0;point<nedge;++point)for (int f=0;f<nf;++f)
            model.scri.values[nf*point+f]=model.edge_refs[point][f].f
                +(old::CartesianIndependent(f)
                  ? Perturb(model.scri.rows[point].geometry.x,f,scenario) : 0);
          model.Materialize(model.state);model.MaterializeBoundary(model.state);
          auto residual=model.Residuals(model.state);
          model.SolveBoundary(residual.first,residual.second);
          for (int point=0;point<nedge;++point) {
            const auto &row=model.scri.rows[point];
            const int face=model.grid.mesh.blocks[row.block].face;
            const int canonical=(face*n+row.k)*n+row.j;
            const auto jets=model.scri.Sample(point,residual.first,residual.second,nf);
            const auto rhs=model.Boundary(point,model.EdgeJet(point,residual.first,residual.second,true));
            for (int f=0;f<nf;++f) {
              projected[nf*canonical+f]=model.scri.values[nf*point+f];
              rate[nf*canonical+f]=rhs.rate[f];
              cubed_sphere::Jet jet(jets.value[f]);
              for (int a=0;a<3;++a) {
                jet.d[a]=jets.d[f][a];
                for (int b=0;b<3;++b)jet.h[a][b]=jets.dd[f][a][b];
              }
              WriteJet(samples,nf*canonical+f,jet);
              for (int r=0;r<3;++r) {
                const int active=model.grid.ActiveIndex(row.block,n-3+r,row.j,row.k);
                cubed_sphere::Jet bulk(residual.first[f*model.grid.count+model.grid.active[active]]);
                model.radial_plan->Derivatives(residual.first,residual.second,f,active,
                                                bulk.d,bulk.h,true);
                WriteJet(radial,nf*(3*canonical+r)+f,bulk);
              }
            }
            for (int a=0;a<3;++a)correction[3*canonical+a]=model.correction[3*point+a];
          }
        }
        for (auto *v:{&projected,&samples,&radial,&correction,&rate})
          MPI_Bcast(v->data(),v->size(),MPI_DOUBLE,0,MPI_COMM_WORLD);
        const auto current=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.s0);
        const auto sampled=Kokkos::create_mirror_view_and_copy(HostMemSpace(),surface.SampledJets());
        const auto corrected=Kokkos::create_mirror_view_and_copy(HostMemSpace(),surface.Correction());
        const auto rhs=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.s_rhs);
        for (int point=0;point<p.nsurface_points;++point) {
          const auto c=p.surface_points.h_view(point);const int m=p.face_blocks.h_view(c.face);
          const int face=mesh.pmb_pack->pmb->mb_map.h_view(m).face;
          const int canonical=(face*n+c.k)*n+c.j;
          for (int f=0;f<nf;++f) {
            projection_error.Compare(current(c.face,f,c.k,c.j),projected[nf*canonical+f],f);
            sample_error.CompareJet(sampled(point)[f],samples,nf*canonical+f,f);
            rate_error.Compare(rhs(c.face,f,c.k,c.j),rate[nf*canonical+f],f);
          }
          for (int a=0;a<3;++a)
            correction_error.Compare(corrected(c.face,a,c.k,c.j),correction[3*canonical+a],a);
        }
        const auto kernel=surface.BindKernel();const auto bulk_residual=p.residual;
        DvceArray2D<native::mapped::Jet> bulk_jets("radial jets",kernel.radial_rows.extent(0),nf);
        Kokkos::parallel_for("oracle native radial jet",Kokkos::RangePolicy<>(0,p.npoints),
          KOKKOS_LAMBDA(int point) {
            if (kernel.HasRadialRow(point))for (int f=0;f<nf;++f)
              bulk_jets(kernel.radial_index(point),f)=kernel.ResidualJet(point,f,bulk_residual,true);
          });
        const auto bulk_host=Kokkos::create_mirror_view_and_copy(HostMemSpace(),bulk_jets);
        const auto radial_index=Kokkos::create_mirror_view_and_copy(HostMemSpace(),kernel.radial_index);
        for (int point=0;point<p.npoints;++point) {
          const auto c=p.points.h_view(point);const auto map=mesh.pmb_pack->pmb->mb_map.h_view(c.m);
          if (!map.outer_surface || c.i-p.origin<n-3)continue;
          const int canonical=(map.face*n+c.k-p.origin)*n+c.j-p.origin;
          const int r=c.i-p.origin-(n-3);
          for (int f=0;f<nf;++f)
            radial_error.CompareJet(bulk_host(radial_index(point),f),radial,nf*(3*canonical+r)+f,f);
        }
        while (surface.ClearSend()!=TaskStatus::complete) {}
        while (surface.ClearRecv()!=TaskStatus::complete) {}
        while (halo.ClearSend()!=TaskStatus::complete) {}
        while (halo.ClearRecv()!=TaskStatus::complete) {}
      }
      for (auto *e:{&projection_error,&sample_error,&radial_error,&correction_error,&rate_error})
        e->Global();
      MPI_Allreduce(MPI_IN_PLACE,&maximum_defect,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE,&cache_compared,1,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE,&cache_rows,1,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE,&cache_bytes,1,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
      Require(cache_rows==3*nedge && cache_bytes==cache_rows*26*sizeof(native::mapped::Jet),
              "endpoint cache allocation coverage changed");
      Require(cache_compared==2*3*nedge*nf*14,"endpoint cache comparison coverage changed");
      if (!global_variable::my_rank) {
        std::cout<<std::setprecision(17);
        std::cout<<"cache_layout rows="<<cache_rows<<" fields=26 jet_bytes="
            <<sizeof(native::mapped::Jet)<<" bytes="<<cache_bytes<<"\n";
        std::cout<<"endpoint_cache comparisons="<<cache_compared
            <<" poison_refresh=passed default_direct=passed bitwise=passed\n";
        const char *names[] = {"projection","surface_jets","radial_jets","normal_correction","surface_rhs"};
        Error *errors[]={&projection_error,&sample_error,&radial_error,&correction_error,&rate_error};
        for (int i=0;i<5;++i)std::cout<<names[i]<<" comparisons="<<errors[i]->count
            <<" max_scaled="<<errors[i]->scaled<<" max_absolute="<<errors[i]->absolute
            <<" failures="<<errors[i]->failures<<"\n";
        std::cout<<"normal_compatibility="<<maximum_defect<<" zero_face_ranks="<<zero_faces<<"\n";
      }
      for (auto *e:{&projection_error,&sample_error,&radial_error,&correction_error,&rate_error})
        Require(e->failures==0,"surface oracle exceeds adopted short-operator gate");
      if (!global_variable::my_rank)std::cout<<"PASS native scri and radial oracle, two scenarios\n";
    }
  } catch (const std::exception &e) {
    std::cerr<<"rank "<<global_variable::my_rank<<": "<<e.what()<<"\n";
    MPI_Abort(MPI_COMM_WORLD,1);
  }
  Kokkos::finalize();MPI_Finalize();
}
