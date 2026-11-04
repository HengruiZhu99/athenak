// Nontrivial polynomial frame oracle and CPU/device halo equivalence.
#include <mpi.h>
#include <iostream>
#include <iomanip>
#include "globals.hpp"
#include "refined_configuration.hpp"
#include "z4c/hyperboloidal/hyperbolic_composite_exchange.hpp"
namespace h=z4c::hyperboloidal;
void Require(bool b,const char*m){if(!b)throw std::runtime_error(m);}
int main(int argc,char**argv){
 MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
 MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);Kokkos::initialize(argc,argv);int code=0;
 try{
  Require(argc==2,"supply frame-control configuration");
  hyperbolic_reproduction::RefinedConfiguration cfg;cfg.Load(argv[1],hyperbolic_reproduction::ContinuousCoreMesh(12));
  Require(cfg.gauge.conformal_frame_halos,"frame control is disabled");
  h::HyperbolicComposite cpu(cfg.Mesh(12));cpu.Configure(cfg.gauge);
  const auto&g=cpu.grid;const auto plan=cpu.frame_halos;Require(plan&&plan->corrected_rows>0,"no corrected interface rows");
  auto exact=[&](const Real*x,int f){const Real r2=x[0]*x[0]+x[1]*x[1]+x[2]*x[2],r=std::sqrt(r2);
   const Real rho=r>=cfg.gauge.profile.outer_radius?1:cfg.gauge.profile.Radial(r).value/((1-r2)/2);
   const Real poly=1+x[0]+2*x[1]*x[1]+x[2]*x[2]+x[0]*x[1];
   if(f==0)return (1+.01*poly)/(rho*rho);
   if(f>=7&&f<=12)return .001*(f-5)*poly/rho;
   if(f==18)return rho*(1+.03*poly)-(1+r2)/2;
   return .0001*(1+f%5)*poly;
  };
  for(size_t p=0;p<g.active.size();++p)for(int f=0;f<31;++f)cpu.state[31*p+f]=exact(g.geometry[p].x,f);
  for(size_t p=0;p<cpu.scri.rows.size();++p)for(int f=0;f<31;++f)cpu.scri.values[31*p+f]=exact(cpu.scri.rows[p].geometry.x,f);
  cpu.BoundaryValues();
  cpu.frame_halos.reset();const auto baseline=cpu.Padded(cpu.state);cpu.frame_halos=plan;
  const auto corrected=cpu.Padded(cpu.state);Real analytic=0,baseline_error=0,unchanged=0,device_error=0;size_t affected=0;
  const int w=g.padded,bs=w*w*w,o=g.storage_halo;
  for(const auto&row:g.halos){const int block=row.target/bs,offset=row.target%bs;
   const Real q[3]={(offset%w-o+.5)/g.n,((offset/w)%w-o+.5)/g.n,(offset/(w*w)-o+.5)/g.n};Real x[3];g.mesh.Forward(block,q,x);
   for(int f=0;f<31;++f){const int index=f*g.count+row.target;
    if(plan->enabled[row.target]&&h::ConformalHaloPoint::Transformed(f)){
     analytic=std::max(analytic,std::abs(corrected[index]-exact(x,f)));baseline_error=std::max(baseline_error,std::abs(baseline[index]-exact(x,f)));++affected;
    }else unchanged=std::max(unchanged,std::abs(corrected[index]-baseline[index]));
   }
  }
  Require(analytic<2e-12,"polynomial canonical frame not reproduced");Require(unchanged==0,"untargeted halos changed");Require(baseline_error>1e-7,"oracle did not exercise nontrivial frame error");
  using Buffer=h::HyperbolicCompositeDevice::Buffer;Buffer state("manufactured state",cpu.state.size());
  auto host=Kokkos::create_mirror_view(state);for(size_t i=0;i<cpu.state.size();++i)host(i)=cpu.state[i];Kokkos::deep_copy(state,host);
  const int blocks=g.mesh.blocks.size();size_t dependency_counts[2]{};int rejected_surface_rows=0;
  for(int part=0;part<2;++part){const int lo=part*blocks/2,hi=(part+1)*blocks/2;
   h::HyperbolicCompositeExchange exchange(cpu,lo,hi,{},true,false);exchange.SetSurface(cpu.scri.values);
   const auto summary=exchange.Fill(state);Require(summary.invalid==0,"device halo rejected geometry");dependency_counts[part]=exchange.ActiveDependencies().size();
   auto got=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),exchange.padded);
   for(const auto&row:g.halos)if(row.target/bs>=lo&&row.target/bs<hi)for(int f=0;f<31;++f)
    device_error=std::max(device_error,std::abs(got(f*g.count+row.target)-corrected[f*g.count+row.target]));
   auto invalid_surface=cpu.scri.values;
   for(size_t p=0;p<invalid_surface.size();p+=31)invalid_surface[p]=-1;
   exchange.SetSurface(invalid_surface);const auto rejected=exchange.Fill(state);
   Require(rejected.invalid==rejected.invalid_halos+rejected.invalid_surface,"invalid summary decomposition");
   rejected_surface_rows+=rejected.invalid_surface;
  }
  Require(rejected_surface_rows>0,"invalid surface was not identified separately");
  Require(device_error<2e-12,"CPU/device halo disagreement");
  std::cout<<std::setprecision(17)<<"{\"passed\":true,\"backend\":\""<<Kokkos::DefaultExecutionSpace::name()<<"\",\"corrected_rows\":"<<plan->corrected_rows<<",\"affected_components\":"<<affected<<",\"exact_polynomial_max_error\":"<<analytic<<",\"baseline_max_error\":"<<baseline_error<<",\"untargeted_max_change\":"<<unchanged<<",\"cpu_device_max_error\":"<<device_error<<",\"partition_dependencies\":["<<dependency_counts[0]<<','<<dependency_counts[1]<<"]}\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';code=1;}
 Kokkos::finalize();MPI_Finalize();return code;
}
