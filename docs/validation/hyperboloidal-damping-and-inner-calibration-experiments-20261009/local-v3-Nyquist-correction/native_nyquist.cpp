// Supplemental actual native span2.1 Nyquist check; frozen v2 stays unchanged.
#include "profile_helpers.hpp"
int main(){std::cout<<std::setprecision(17)<<'[';bool first=true;
 hyp::LayerReference<double>ref(1.,.5,{true,.05,.95});
 const int ns[3]={24,36,48};const double os[3]={.0142578125,.0038368055555556,.003251953125};
 const double dt[3]={.000427734375,.0001151041666666667,.00009755859375};
 for(int ni=0;ni<3;++ni){const double k=std::acos(-1.)*ns[ni]/2.1;const auto p=ref.At(std::sqrt(1-os[ni]),0.,0.);
 for(int mode:{0,1})for(bool oblique:{false,true}){const std::array<double,3>n=oblique?std::array<double,3>{.36,-.48,.8}:std::array<double,3>{1,0,0};
 if(!first)std::cout<<',';first=false;std::cout<<"{\"N\":"<<ns[ni]<<",\"span\":2.1,\"k\":"<<k<<",\"Omega\":"<<p.omega<<",\"dt\":"<<dt[ni]<<",\"profile\":"<<mode<<",\"oblique\":"<<oblique<<",\"L\":";Print(Matrix(p,.5,10,mode,true,0.,k,n));std::cout<<'}';}}
 std::cout<<"]\n";
}
