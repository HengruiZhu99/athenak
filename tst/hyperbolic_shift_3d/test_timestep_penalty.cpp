// Reuse the scri fixture to check capped rates and dependent null closure.
#define main scri_fixture_main
#include "test_scri_device.cpp"
#undef main
int main(int argc,char**argv) {
  Kokkos::initialize(argc,argv);int code=0;
  try {
    h::HyperbolicComposite cpu(cubed_sphere::CompositeMesh(.125,{.45,.75,1},1,6,3));
    Seed(cpu,true);cpu.BoundaryValues();
    const auto active=Upload(cpu.state),surface=Upload(cpu.scri.values);
    Device device(cpu,0,cpu.scri.rows.size());
    for(Real dt:{Real(.0001),Real(.0045),Real(.00225)}) {
      const auto q=device.Evaluate(active,surface,dt);
      Require(q.bad_rows==0,"capped scri evaluation failed");
      Require(q.max_cfl<=.5,"applied penalty exceeds its bound");
      Require(q.null_after<1e-12&&q.compatibility<1e-10,"dependent null closure lost precision");
      if(dt==.0001)Require(q.min_penalty_scale==1,"uncapped step was changed");
      else Require(q.min_penalty_scale<1&&q.max_unlimited_cfl>.5,"large step failed to cap");
      for(Real value:Download(device.LocalRHS()))Require(std::isfinite(value),"nonfinite capped rate");
      std::cout<<std::setprecision(17)<<"dt="<<dt<<" applied="<<q.max_cfl
        <<" unlimited="<<q.max_unlimited_cfl<<" scale="<<q.min_penalty_scale
        <<" null="<<q.null_after<<" compatibility="<<q.compatibility<<'\n';
    }
  }catch(const std::exception&error){std::cerr<<error.what()<<'\n';code=1;}
  Kokkos::finalize();return code;
}
