#include "taylor.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
int main(){double worst=0;
 for(double v:{.05,1.,20.}){const auto x=Variable<4>(v);const auto a=Taylor<4>(1)+x*x/Taylor<4>(4);const auto inverse=a*Inverse(a),back=Exp(Log(a));for(int n=0;n<=4;++n){worst=std::max(worst,std::abs(inverse[n]-(n==0?1.:0.))/std::max(1.,std::abs(a[n])));worst=std::max(worst,std::abs(back[n]-a[n])/std::max(1.,std::abs(a[n])));}}
 Taylor<4>a(-750);a[1]=1e6;const auto tail=Exp(a);const double exact=std::exp(-750+std::log(1e24/24));const double relative=std::abs(tail[4]-exact)/exact;
 Taylor<4>q;for(int k=0;k<=4;++k)q[k]=k+1;const auto d=Derivative<3>(q);double derivative=0;for(int k=0;k<=3;++k)derivative=std::max(derivative,std::abs(d[k]-(k+1)*(k+2)));
 std::cout<<std::setprecision(17)<<"{\"algebra_scaled\":"<<worst<<",\"tail_relative\":"<<relative<<",\"tail_zero_value\":"<<(tail[0]==0?"true":"false")<<",\"tail_fourth_coefficient\":"<<tail[4]<<",\"derivative_coefficients\":"<<derivative<<"}\n";
 return !(worst<=5e-13&&relative<=5e-13&&tail[0]==0&&tail[4]>0&&derivative==0);
}
