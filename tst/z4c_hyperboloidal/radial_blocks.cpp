// Native MPI regression for radial-only shell subdivision and scri rows.
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "pgen/pgen.hpp"
#include "z4c_hyperboloidal/mapped_halo.hpp"
#include "z4c_hyperboloidal/surface_exchange.hpp"

namespace h=z4c_hyperboloidal;
void Check(bool ok,const char *why) {if (!ok)throw std::runtime_error(why);}
double Radius(const Real *x) {return std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);}

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
      auto &p=*mesh.pmb_pack->pz4c_hyperboloidal;const int n=p.n[0],o=p.origin;
      // Materialize every bulk stencil dependency, including the subdivided
      // first overlap region, rather than checking selected coordinates only.
      h::MappedHalo halo;halo.Rebuild(mesh.pmb_pack,o,p.activity);
      h::NativeDonorSelector selector(mesh,o,p.activity);
      int checks=0;
      // Test both sides of nonuniform shell and sub-block interfaces. The
      // radial degree-five polynomial is exactly represented by six donors.
      for (int face=0;face<6;++face) for (double r:{.1332,.1335,.1399,.1401,.1599,.1601,
          .8799,.8801,.8999,.9001,
          .9199,.9201,.9399,.9401,.9549,.9551,.9699,.9701,.9849,.9851,.998}) {
        Real x[3]{};const int a=face/2,b=(a+1)%3,c=(a+2)%3;
        const double scale=r/std::sqrt(1+.2*.2+.4*.4);
        x[a]=(face%2 ? -1 : 1)*scale;x[b]=.2*scale;x[c]=-.4*scale;
        const auto row=selector.Interpolate(x);
        for (int degree=0;degree<=5;++degree) {
          double value=0,norm=0;
          for (const auto &d:row) {
            Check(selector.IsEvolved(d.point),"radial row reads unevolved donor");
            const auto &map=selector.Map(d.point.gid);
            const Real q[3]={(d.point.i-o+map.node_offset[0])/n,
                            (d.point.j-o+.5)/n,(d.point.k-o+.5)/n};
            Real y[3];map.Forward(q,y);
            value+=d.weight*std::pow(Radius(y),degree);norm+=std::abs(d.weight);
          }
          Check(std::abs(value-std::pow(r,degree))<2e-12*std::max(1.,norm),
                "radial interface polynomial not reproduced");++checks;
        }
      }
      // Exercise actual distributed scri sampling, including angular seams,
      // with an analytic radial quadratic residual in g_xx.
      h::SurfaceExchange surface(mesh.pmb_pack,&p);
      auto bulk=Kokkos::create_mirror_view(p.residual);
      auto edge=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.s0);
      Kokkos::deep_copy(bulk,0.);
      const auto maps=mesh.pmb_pack->pmb->mb_map.h_view;
      for (int m=0;m<p.nblocks;++m) for (int k=0;k<p.storage[2];++k)
      for (int j=0;j<p.storage[1];++j) for (int i=0;i<p.storage[0];++i) {
        const auto map=maps(m);const Real q[3]={(i-o+map.node_offset[0])/n,
                                              (j-o+.5)/n,(k-o+.5)/n};
        Real x[3];map.Forward(q,x);const double r=Radius(x);
        bulk(m,1,k,j,i)=2+.3*r+.2*r*r;
      }
      const auto refs=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.surface_reference);
      for (int point=0;point<p.nsurface_points;++point) {
        const auto c=p.surface_points.h_view(point);edge(c.face,1,c.k,c.j)=refs(point)[1].f+2.5;
      }
      Kokkos::deep_copy(p.residual,bulk);Kokkos::deep_copy(p.s0,edge);
      surface.StartReceives();surface.PackAndSend();
      while (surface.Receive()!=TaskStatus::complete) {}
      surface.Sample();
      const auto sampled=Kokkos::create_mirror_view_and_copy(HostMemSpace(),surface.SampledJets());
      const auto geometry=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.surface_geometry);
      const auto rows=Kokkos::create_mirror_view_and_copy(HostMemSpace(),surface.Rows());
      const auto terms=Kokkos::create_mirror_view_and_copy(HostMemSpace(),surface.Terms());
      int surfaces=p.nsurface_points;
      for (int point=0;point<surfaces;++point) {
        const auto jet=sampled(point)[1];const auto &x=geometry(point).x;
        Check(std::abs(jet.f-2.5)<1e-13,"scri residual value");
        for (int a=0;a<3;++a) {
          double norm=0;for (int t=rows(point).begin;t<rows(point).end;++t)
            norm+=std::abs(terms(t).weight.d[a]);
          Check(std::abs(jet.d[a]-.7*x[a])<1024*std::numeric_limits<Real>::epsilon()*std::max(1.,norm),
                "scri radial gradient");
          for (int b=0;b<3;++b) {
            norm=0;for (int t=rows(point).begin;t<rows(point).end;++t)
              norm+=std::abs(terms(t).weight.h[a][b]);
            Check(std::abs(jet.h[a][b]-(.7*(a==b)-.3*x[a]*x[b]))
                  <1024*std::numeric_limits<Real>::epsilon()*std::max(1.,norm),
                  "scri radial Hessian");
          }
        }
      }
      while (surface.ClearSend()!=TaskStatus::complete) {}
      while (surface.ClearRecv()!=TaskStatus::complete) {}
      MPI_Allreduce(MPI_IN_PLACE,&surfaces,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
      Check(surfaces==6*n*n,"radial subdivision changed angular surface count");
      if (!global_variable::my_rank)std::cout<<"PASS radial polynomials="<<checks
          <<" scri_points="<<surfaces<<" blocks="<<mesh.nmb_total<<"\n";
    }
  } catch (const std::exception &error) {
    std::cerr<<"rank "<<global_variable::my_rank<<": "<<error.what()<<'\n';MPI_Abort(MPI_COMM_WORLD,1);
  }
  Kokkos::finalize();MPI_Finalize();return 0;
}
