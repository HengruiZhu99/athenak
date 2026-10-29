#include <sstream>
#include <iostream>
#include "coupled_diagnostics.hpp"
int main(int argc,char**argv){
  Kokkos::initialize(argc,argv);int result=0;
  try{
    if(argc!=4)throw std::runtime_error("usage: reanalyze n input_prefix output_prefix");
    const int n=std::stoi(argv[1]);
    z4c::hyperboloidal::HyperbolicComposite e(cubed_sphere::CompositeMesh(.125,{.45,.75,1},1,n,3),false);
    auto read=[](const std::string&file,auto&v){std::ifstream input(file);std::string line;size_t i=0;
      if(!input)throw std::runtime_error("missing state file");
      while(std::getline(input,line)){std::replace(line.begin(),line.end(),',',' ');std::istringstream row(line);Real value;
        while(row>>value){if(i>=v.size())throw std::runtime_error("too many values");v[i++]=value;}}
      if(i!=v.size())throw std::runtime_error("wrong state size");};
    read(std::string(argv[2])+".state.csv",e.state);read(std::string(argv[2])+".boundary.csv",e.scri.values);
    coupled_diagnostics::Write(e,argv[3]);
  }catch(const std::exception&ex){std::cerr<<ex.what()<<'\n';result=1;}
  Kokkos::finalize();return result;
}
