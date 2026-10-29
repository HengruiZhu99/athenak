#include <sstream>
#include <iostream>
#include "coupled_diagnostics.hpp"
#include "fixed_composite_mesh.hpp"
int main(int argc,char**argv){
  Kokkos::initialize(argc,argv);int result=0;
  try{
    if(argc!=4&&argc!=5)throw std::runtime_error("usage: reanalyze n input_prefix output_prefix [donor_seven|inner_transition|core2]");
    const int n=std::stoi(argv[1]);
    const bool radial=argc==5;
    const bool core2=radial&&std::string(argv[4])=="core2";
    const bool inner=core2||(radial&&std::string(argv[4])=="inner_transition");
    if(radial&&!inner&&std::string(argv[4])!="donor_seven")throw std::runtime_error("unknown diagnostic radial operator");
    using Model=z4c::hyperboloidal::HyperbolicComposite;
    Model e(hyperbolic_reproduction::FixedMesh(n,core2?2:1),false,radial,0,
      radial?Model::NormalDerivative::sampled_numerator:Model::NormalDerivative::analytic_jets,
      radial?Model::RadialClosure::donor_seven:Model::RadialClosure::ghost_lift);
    if(inner)e.grid.SetInnerTransitionLogical(true);
    auto read=[](const std::string&file,auto&v){std::ifstream input(file);std::string line;size_t i=0;
      if(!input)throw std::runtime_error("missing state file");
      while(std::getline(input,line)){std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);Real value;
        while(row>>value){if(i>=v.size())throw std::runtime_error("too many values");v[i++]=value;}}
      if(i!=v.size())throw std::runtime_error("wrong state size");};
    read(std::string(argv[2])+".state.csv",e.state);read(std::string(argv[2])+".boundary.csv",e.scri.values);
    coupled_diagnostics::Write(e,argv[3]);
    std::cout<<"bulk_diagnostic_operator="<<(radial?"donor_seven_last3":"legacy_grid")
             <<" ray_diagnostic_operator=interpolated_jets";
    if(inner)std::cout<<" interpatch_interpolation=logical_inner_transition";
    if(core2)std::cout<<" mesh_topology=cartesian_2x2x2_shells3";
    std::cout<<'\n';
  }catch(const std::exception&ex){std::cerr<<ex.what()<<'\n';result=1;}
  Kokkos::finalize();return result;
}
