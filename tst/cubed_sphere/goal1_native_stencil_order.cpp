// Read-only formal-order audit of the actual native templates (no evolution).
#include "mesh/mesh.hpp"
#include "utils/finite_diff.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>
struct Polynomial{int p=0,q=0;Real operator()(int,int,int j,int i)const{return std::pow(Real(i),p)*std::pow(Real(j),q);}};
int main(){Real idx[3]={1,1,1};std::cout<<std::setprecision(17);for(int p=0;p<=8;++p){Polynomial f{p,0};std::cout<<"{\"power\":"<<p<<",\"Dx3\":"<<Dx<3>(0,idx,f,0,0,0,0)<<",\"Dxx3\":"<<Dxx<3>(0,idx,f,0,0,0,0)<<",\"Dx4\":"<<Dx<4>(0,idx,f,0,0,0,0)<<",\"Dxx4\":"<<Dxx<4>(0,idx,f,0,0,0,0)<<"}\n";}
for(int p=1;p<=6;++p){Polynomial f{p,1};std::cout<<"{\"mixed_x_power\":"<<p<<",\"mixed_y_power\":1,\"Dxy3\":"<<Dxy<3>(0,1,idx,f,0,0,0,0)<<"}\n";}}
