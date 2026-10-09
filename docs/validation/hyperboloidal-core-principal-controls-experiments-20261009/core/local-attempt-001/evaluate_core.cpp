// Local exact-core polynomial envelope evaluator; no spatial/radial grid.
#include "core_envelope.hpp"
#include "all_m_data.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
using X3=std::array<double,3>;
using Raw=std::array<double,22>;
constexpr int ti[6]={0,0,0,1,1,2},tj[6]={0,1,2,1,2,2};
Raw Materialize(int J,int m,int phase,const X3&x,const std::array<double,20>&a) {
  Raw out{};
  for(int c=0;c<core_envelope::Channels(J);++c) {
    const auto ch=allm::ChannelAt(J,c);
    const auto v=allm::Evaluate(J,m,ch.spin,ch.L,phase==1,x,{a[c],0,0});
    if(ch.kind==0)out[18]+=v.component[0].value;
    if(ch.kind==1)out[0]-=v.component[0].value/std::sqrt(3.);
    if(ch.kind==2)out[7]+=v.component[0].value;
    if(ch.kind==3)out[17]+=v.component[0].value;
    if(ch.kind==4||ch.kind==5)for(int i=0;i<3;++i)out[(ch.kind==4?19:14)+i]+=v.component[i].value;
    if(ch.kind==6||ch.kind==7)for(int q=0;q<6;++q)out[(ch.kind==6?1:8)+q]+=v.component[3*ti[q]+tj[q]].value;
  }
  return out;
}
int main() {try {
  std::cout<<std::setprecision(17);
  int J,m,c,phase,rot;X3 x;std::array<double,3>w;
  while(std::cin>>J>>m>>c>>phase>>rot>>x[0]>>x[1]>>x[2]>>w[0]>>w[1]>>w[2]) {
    if(rot!=0||c<0||c>=core_envelope::Channels(J))throw std::invalid_argument("query");
    const double rho=x[0]*x[0]+x[1]*x[1]+x[2]*x[2];
    std::array<std::array<double,20>,3> jets{};for(int k=0;k<3;++k)jets[k][c]=w[k];
    std::array<double,20> input{};input[c]=w[0];
    const auto a=Materialize(J,m,phase,x,input);
    const auto f=Materialize(J,m,phase,x,core_envelope::Apply(J,rho,jets));
    for(double value:a)std::cout<<value<<' ';
    for(double value:f)std::cout<<value<<' ';
    std::cout<<'\n';
  }
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
