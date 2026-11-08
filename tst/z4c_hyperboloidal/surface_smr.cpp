// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Manufactured operators on a genuinely refined outer-shell native topology.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "z4c_hyperboloidal/surface_exchange.hpp"
#include "z4c_hyperboloidal/dissipation_plan.hpp"

namespace h=z4c_hyperboloidal;
void Require(bool yes,const char *what) {if (!yes)throw std::runtime_error(what);}
double Value(double r,int scenario) {return scenario ? 2+.3*r+.2*r*r : 2;}
double FieldScale(int f) {return .2+.01*f;}
std::array<double,13> Exact(const double *x,int scenario) {
  std::array<double,13> out{};
  const double r=std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
  out[0]=Value(r,scenario);
  if (scenario)for (int a=0;a<3;++a) {
    out[1+a]=(.3/r+.4)*x[a];
    for (int b=0;b<3;++b)
      out[4+3*a+b]=(.3/r+.4)*(a==b)-.3*x[a]*x[b]/(r*r*r);
  }
  return out;
}
std::array<double,13> Components(const h::mapped::Jet &q) {
  std::array<double,13> out{};out[0]=q.f;
  for (int a=0;a<3;++a) {
    out[1+a]=q.d[a];
    for (int b=0;b<3;++b)out[4+3*a+b]=q.h[a][b];
  }
  return out;
}

// Exact cache-storage checks supplement, without replacing, the independent
// polynomial gates below. The second scenario deliberately leaves finite jets
// from the first scenario until the refresh is invoked.
void CheckRadialCache(h::SurfaceExchange &surface,h::Z4cHyperboloidal &p,
    h::SurfaceExchange::CachedKernel &cached,bool stale,
    std::uint64_t &compared,std::uint64_t &stale_changed) {
  const auto direct=surface.BindKernel();const auto data=p.residual;
  DvceArray2D<h::mapped::Jet> expected("direct radial cache reference",
      direct.radial_rows.extent(0),h::Fields::independent_count);
  Kokkos::parallel_for("SMR direct radial cache reference",Kokkos::RangePolicy<>(0,p.npoints),
    KOKKOS_LAMBDA(int point) {
      if (!direct.HasRadialRow(point))return;
      for (int f=0;f<h::Fields::count;++f)if (h::Fields::Independent(f))
        expected(direct.radial_index(point),direct.fields.index[f])=
            direct.ResidualJet(point,f,data,f<=6||f==18||(f>=22&&f<25));
    });
  const auto truth=Kokkos::create_mirror_view_and_copy(HostMemSpace(),expected);
  if (stale) {
    const auto old=Kokkos::create_mirror_view_and_copy(HostMemSpace(),cached.radial_jets);
    Require(old.extent(0)==truth.extent(0)&&old.extent(1)==truth.extent(1),
            "stale radial cache shape changed without rebuild");
    for (size_t row=0;row<old.extent(0);++row)for (int f=0;f<h::Fields::independent_count;++f) {
      Require(std::isfinite(old(row,f).f)&&old(row,f).f!=truth(row,f).f,
              "manufactured state did not expose stale radial cache");++stale_changed;
    }
  }
  cached=surface.PrepareRadialJets();
  Require(cached.radial_jets.extent(0)==truth.extent(0)&&
          cached.radial_jets.extent(1)==h::Fields::independent_count,
          "radial cache does not cover the current refined rows and fields");
  h::mapped::Jet poison(std::numeric_limits<Real>::quiet_NaN());
  for (int a=0;a<3;++a) {
    poison.d[a]=poison.f;
    for (int b=0;b<3;++b)poison.h[a][b]=poison.f;
  }
  Kokkos::deep_copy(cached.radial_jets,poison);
  cached=surface.PrepareRadialJets();
  const auto actual=Kokkos::create_mirror_view_and_copy(HostMemSpace(),cached.radial_jets);
  for (size_t row=0;row<truth.extent(0);++row)for (int f=0;f<h::Fields::independent_count;++f) {
    const auto a=Components(actual(row,f)),b=Components(truth(row,f));
    for (int c=0;c<13;++c) {
      Require(std::isfinite(a[c])&&std::isfinite(b[c])&&
              std::memcmp(&a[c],&b[c],sizeof(a[c]))==0,
              "refined radial cache did not refresh bitwise after poison");++compared;
    }
  }
}

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
      h::SurfaceExchange surface(mesh.pmb_pack,&p);
      const auto maps=mesh.pmb_pack->pmb->mb_map.h_view;
      const auto geometry=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.geometry);
      const auto edge_geometry=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.surface_geometry);
      const auto refs=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.surface_reference);
      const int n=p.n[0],origin=p.origin;
      std::uint64_t counts[4]={std::uint64_t(p.nblocks),std::uint64_t(p.nfaces),0,0};
      for (int m=0;m<p.nblocks;++m) {
        const auto map=maps(m);
        if (map.outer_surface) {
          Require(map.node_offset[0]==0,"outer refined leaf lost endpoint node offset");
          if (map.chart_id==1) {
            Require(map.hi[0]-map.lo[0]==.5,"outer SMR radial split missing");
            ++counts[2];
          }
        } else if (map.chart_id==1 && map.layer==11) {
          Require(map.node_offset[0]==0,"inner radial sibling lost inherited nodal grid");
          ++counts[3];
        }
      }
      MPI_Allreduce(MPI_IN_PLACE,counts,4,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
      Require(counts[0]==113 && counts[1]==9 && counts[2]==4 && counts[3]==4,
              "outer SMR fixture topology changed");

      // Independently evaluate the spherical volume Jacobian. The shared
      // r=.97 refined interface owns a full nodal cell; only r=.94, the inner
      // endpoint of the original nodal layer, owns a half cell. A test of KO
      // rates alone cannot expose this error because its taper vanishes here.
      h::DissipationPlan dissipation;dissipation.Rebuild(*mesh.pmb_pack,p);
      const auto volumes=Kokkos::create_mirror_view_and_copy(HostMemSpace(),
                                                          dissipation.Bind().volumes);
      std::uint64_t volume_checks=0,interior_starts=0;
      for (int point=0;point<p.npoints;++point) {
        const auto c=p.points.h_view(point);const auto map=maps(c.m);
        if (map.face<0 || map.layer!=11)continue;
        const double q1=map.lo[1]+(c.j-origin+.5)*(map.hi[1]-map.lo[1])/n;
        const double q2=map.lo[2]+(c.k-origin+.5)*(map.hi[2]-map.lo[2])/n;
        const double a=std::acos(-1.)/4,u=std::tan(a*q1),v=std::tan(a*q2);
        const double angular=a*a*(1+u*u)*(1+v*v)/std::pow(1+u*u+v*v,1.5);
        const double r=h::SphericalPolicy::Radius(geometry(point).x);
        const double dr=(map.outer_radius-map.inner_radius)*(map.hi[0]-map.lo[0])/n;
        double expected=r*r*dr*angular*(map.hi[1]-map.lo[1])/n
            *(map.hi[2]-map.lo[2])/n;
        if (std::abs(r-.94)<1e-14)expected*=.5;
        Require(std::abs(volumes(point)-expected)<=128*std::numeric_limits<Real>::epsilon()
            *expected,"refined nodal volume disagrees with analytic spherical Jacobian");
        ++volume_checks;
        if (map.chart_id==1 && c.i==origin && std::abs(r-.97)<1e-14)++interior_starts;
      }
      MPI_Allreduce(MPI_IN_PLACE,&volume_checks,1,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE,&interior_starts,1,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
      Require(volume_checks==13*1728 && interior_starts==4*144,
              "refined nodal-volume coverage changed");
      if (!global_variable::my_rank)std::cout<<"nodal_volume comparisons="<<volume_checks
          <<" internal_radial_starts="<<interior_starts<<" analytic_jacobian=passed\n";

      std::uint64_t compared=0,failed=0,cache_compared=0,stale_changed=0;
      h::SurfaceExchange::CachedKernel cached;
      double worst_ratio=0,max_absolute=0;
      const auto check=[&](double actual,double expected,double row_norm) {
        Require(std::isfinite(actual),"manufactured surface value is nonfinite");
        // A priori rounding allowance from the absolute coefficient row sum.
        // Donor values are O(1); 256 accounts for tensor interpolation and row
        // merging operations. This is a polynomial consistency test, not the
        // stricter native-vs-legacy evolving-state comparison.
        const double bound=256*std::numeric_limits<double>::epsilon()
            *std::max(1.,row_norm)*4;
        const double delta=std::abs(actual-expected);
        max_absolute=std::max(max_absolute,delta);
        worst_ratio=std::max(worst_ratio,delta/bound);
        ++compared;if (delta>bound)++failed;
      };
      for (int scenario=0;scenario<2;++scenario) {
        auto residual=Kokkos::create_mirror_view(p.residual);
        auto edge=Kokkos::create_mirror_view(p.s0);
        Kokkos::deep_copy(residual,std::numeric_limits<Real>::quiet_NaN());
        for (int point=0;point<p.npoints;++point) {
          const auto c=p.points.h_view(point);
          const double r=h::SphericalPolicy::Radius(geometry(point).x);
          for (int f=0;f<31;++f)residual(c.m,f,c.k,c.j,c.i)=h::Fields::Independent(f)
              ? FieldScale(f)*Value(r,scenario) : 0;
        }
        // Supply exact manufactured angular ghost values for the local bulk
        // tensor rows. Bulk donor-plan correctness has its own legacy oracle;
        // this test isolates surface transport and the endpoint reconstruction.
        for (int face=0;face<p.nfaces;++face) {
          const int m=p.face_blocks.h_view(face);const auto map=maps(m);
          for (int k=-2;k<n+2;++k)for (int j=-2;j<n+2;++j)
          for (int i=n-6;i<n;++i) {
            const Real q[3]={(i+map.node_offset[0])/n,(j+.5)/n,(k+.5)/n};
            Real x[3];map.Forward(q,x);const Real r=h::SphericalPolicy::Radius(x);
            for (int f=0;f<31;++f)residual(m,f,origin+k,origin+j,origin+i)=
                h::Fields::Independent(f) ? FieldScale(f)*Value(r,scenario) : 0;
          }
        }
        for (int point=0;point<p.nsurface_points;++point) {
          const auto c=p.surface_points.h_view(point);
          for (int f=0;f<31;++f)edge(c.face,f,c.k,c.j)=refs(point)[f].f
              +(h::Fields::Independent(f) ? FieldScale(f)*Value(1,scenario) : 0);
        }
        Kokkos::deep_copy(p.residual,residual);Kokkos::deep_copy(p.s0,edge);
        surface.StartReceives();surface.PackAndSend();
        while (surface.Receive()!=TaskStatus::complete) {}
        surface.Sample();
        CheckRadialCache(surface,p,cached,scenario!=0,cache_compared,stale_changed);
        const auto samples=Kokkos::create_mirror_view_and_copy(HostMemSpace(),surface.SampledJets());
        const auto rows=Kokkos::create_mirror_view_and_copy(HostMemSpace(),surface.Rows());
        const auto terms=Kokkos::create_mirror_view_and_copy(HostMemSpace(),surface.Terms());
        for (int point=0;point<p.nsurface_points;++point) {
          std::array<double,13> norms{};norms[0]=1;
          for (int t=rows(point).begin;t<rows(point).end;++t) {
            const auto q=Components(terms(t).weight);
            for (int a=0;a<13;++a)norms[a]+=std::abs(q[a]);
          }
          const auto exact=Exact(edge_geometry(point).x,scenario);
          for (int f=0;f<31;++f) {
            const auto actual=Components(samples(point)[f]);
            for (int a=0;a<13;++a)check(actual[a],h::Fields::Independent(f)
                ? FieldScale(f)*exact[a] : 0,norms[a]);
          }
        }
        const auto kernel=surface.BindKernel();const auto data=p.residual;
        DvceArray2D<h::mapped::Jet> jets("SMR radial test jets",kernel.radial_rows.extent(0),31);
        Kokkos::parallel_for("manufactured native radial operator",Kokkos::RangePolicy<>(0,p.npoints),
          KOKKOS_LAMBDA(int point) {
            if (kernel.HasRadialRow(point))for (int f=0;f<31;++f)
              jets(kernel.radial_index(point),f)=kernel.ResidualJet(point,f,data);
          });
        const auto radial=Kokkos::create_mirror_view_and_copy(HostMemSpace(),jets);
        const auto index=Kokkos::create_mirror_view_and_copy(HostMemSpace(),kernel.radial_index);
        const auto rrows=Kokkos::create_mirror_view_and_copy(HostMemSpace(),kernel.radial_rows);
        const auto rterms=Kokkos::create_mirror_view_and_copy(HostMemSpace(),kernel.radial_terms);
        for (int point=0;point<p.npoints;++point)if (index(point)>=0) {
          const int row=index(point);std::array<double,13> norms{};norms[0]=1;
          for (int t=rrows(row).begin;t<rrows(row).end;++t) {
            const auto q=Components(rterms(t).weight);
            for (int a=0;a<13;++a)norms[a]+=std::abs(q[a]);
          }
          const auto exact=Exact(geometry(point).x,scenario);
          for (int f=0;f<31;++f) {
            const auto actual=Components(radial(row,f));
            for (int a=0;a<13;++a)check(actual[a],h::Fields::Independent(f)
                ? FieldScale(f)*exact[a] : 0,norms[a]);
          }
        }
        while (surface.ClearSend()!=TaskStatus::complete) {}
        while (surface.ClearRecv()!=TaskStatus::complete) {}
      }
      // Each patch receives a distinct constant correction. At scri its
      // physical normal derivative is that constant, independent of the radial
      // refinement spacing, and angular derivatives stay local to this patch.
      auto correction=Kokkos::create_mirror_view(surface.Correction());
      for (int face=0;face<p.nfaces;++face)for (int a=0;a<3;++a)
      for (int k=0;k<n;++k)for (int j=0;j<n;++j)
        correction(face,a,k,j)=.001*(mesh.pmb_pack->gids+p.face_blocks.h_view(face)+1)*(a+1);
      Kokkos::deep_copy(surface.Correction(),correction);
      const auto kernel=surface.BindKernel();const auto points=p.surface_points.d_view;
      const auto geometries=p.surface_geometry;
      DvceArray2D<h::mapped::Jet> lifts("SMR surface lifts",p.nsurface_points,3);
      Kokkos::parallel_for("manufactured native normal lift",Kokkos::RangePolicy<>(0,p.nsurface_points),
        KOKKOS_LAMBDA(int point) {
          const auto c=points(point);
          for (int a=0;a<3;++a)lifts(point,a)=kernel.Correction(c.face,c.j,c.k,geometries(point),a);
        });
      const auto host_lifts=Kokkos::create_mirror_view_and_copy(HostMemSpace(),lifts);
      for (int point=0;point<p.nsurface_points;++point) {
        const auto c=p.surface_points.h_view(point);const int m=p.face_blocks.h_view(c.face);
        const auto map=maps(m);const auto &x=edge_geometry(point).x;
        const double dr=(map.outer_radius-map.inner_radius)*(map.hi[0]-map.lo[0])/n;
        const double radius=h::SphericalPolicy::Radius(x);
        const long double z=(radius-1)/dr;
        long double polynomial=1,first=0,second=0;
        for (int factor=0;factor<7;++factor) {
          second=second*(z+factor)+2*first;
          first=first*(z+factor)+polynomial;
          polynomial*=z+factor;
        }
        polynomial*=dr/720;first/=720;second/=720*dr;
        for (int a=0;a<3;++a) {
          const double value=correction(c.face,a,c.k,c.j);const auto q=host_lifts(point,a);
          check(q.f,value*polynomial,1);
          for (int b=0;b<3;++b) {
            check(q.d[b],value*first*x[b]/radius,1);
            for (int d=0;d<3;++d)
              check(q.h[b][d],value*((second-first/radius)*x[b]*x[d]/(radius*radius)
                  +first/radius*(b==d)),1/dr);
          }
        }
      }

      // Corrupt metadata on exactly one surface-owning rank. Every rank must
      // reject before entering exchange-plan collectives, including zero-face
      // ranks. The failed object must also reject subsequent stage use.
      int owner=p.nfaces ? global_variable::my_rank : global_variable::nranks;
      MPI_Allreduce(MPI_IN_PLACE,&owner,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
      int saved=-1;
      if (global_variable::my_rank==owner) {
        saved=p.face_blocks.h_view(0);p.face_blocks.h_view(0)=mesh.nmb_total;
      }
      int rejected=0;
      try {surface.Rebuild();}catch (const std::runtime_error &) {++rejected;}
      try {surface.StartReceives();}catch (const std::logic_error &) {++rejected;}
      try {surface.Project();}catch (const std::logic_error &) {++rejected;}
      try {surface.PrepareRadialJets();}catch (const std::logic_error &) {++rejected;}
      Require(rejected==4,"failed surface rebuild remained usable or was not collective");
      if (global_variable::my_rank==owner)p.face_blocks.h_view(0)=saved;
      surface.Rebuild();
      surface.StartReceives();surface.PackAndSend();
      while (surface.Receive()!=TaskStatus::complete) {}
      CheckRadialCache(surface,p,cached,false,cache_compared,stale_changed);
      while (surface.ClearSend()!=TaskStatus::complete) {}
      while (surface.ClearRecv()!=TaskStatus::complete) {}
      std::uint64_t cache_rows=cached.radial_jets.extent(0),
          cache_bytes=cached.radial_jets.span()*sizeof(h::mapped::Jet);
      MPI_Allreduce(MPI_IN_PLACE,&cache_rows,1,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE,&cache_bytes,1,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
      Require(cache_rows==3*counts[1]*n*n && cache_bytes==cache_rows*26*sizeof(h::mapped::Jet),
              "refined radial cache allocation coverage changed");
      if (!global_variable::my_rank)std::cout<<"cache_layout rows="<<cache_rows
          <<" fields=26 jet_bytes="<<sizeof(h::mapped::Jet)<<" bytes="<<cache_bytes<<"\n";
      MPI_Allreduce(MPI_IN_PLACE,&cache_compared,1,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE,&stale_changed,1,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
      Require(cache_compared==3*3*counts[1]*n*n*h::Fields::independent_count*13 &&
              stale_changed==3*counts[1]*n*n*h::Fields::independent_count,
              "refined radial cache comparison coverage changed");
      if (!global_variable::my_rank)std::cout<<"radial_cache comparisons="<<cache_compared
          <<" stale_changed="<<stale_changed
          <<" poison_refresh=passed rebuild_refresh=passed bitwise=passed\n";
      MPI_Allreduce(MPI_IN_PLACE,&compared,1,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE,&failed,1,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE,&worst_ratio,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
      MPI_Allreduce(MPI_IN_PLACE,&max_absolute,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
      if (!global_variable::my_rank)std::cout<<std::setprecision(17)
          <<"outer_smr blocks="<<counts[0]<<" surface_patches="<<counts[1]
          <<" comparisons="<<compared<<" failures="<<failed
          <<" max_error_over_bound="<<worst_ratio<<" max_absolute="<<max_absolute
          <<" collective_rebuild_rejection=passed\n";
      Require(failed==0,"refined surface polynomial consistency gate failed");
      if (!global_variable::my_rank)std::cout<<"PASS native refined scri operators\n";
    }
  } catch (const std::exception &error) {
    std::cerr<<"rank "<<global_variable::my_rank<<": "<<error.what()<<"\n";
    MPI_Abort(MPI_COMM_WORLD,1);
  }
  Kokkos::finalize();MPI_Finalize();
}
