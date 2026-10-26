#include "goal1_native_endpoint.hpp"
#include "goal1_tensor_runtime.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cstdlib>
namespace cs=cubed_sphere;namespace h=z4c::hyperboloidal;
Real Read(){std::string s;if(!(std::cin>>s))throw std::runtime_error("truncated runtime fixture");char*end;Real x=std::strtod(s.c_str(),&end);if(*end||!std::isfinite(x))throw std::runtime_error("bad runtime fixture");return x;}
std::array<Real,22> Values(const h::Z4cJet<Real>&u){std::array<Real,22>v{};const int i[6]={0,0,0,1,1,2},j[6]={0,1,2,1,2,2};v[0]=u.chi.value;v[7]=u.trace.value;v[17]=u.theta.value;v[18]=u.alpha.value;for(int n=0;n<6;++n){v[n+1]=u.metric.g[i[n]][j[n]];v[n+8]=u.a.k[i[n]][j[n]];}for(int d=0;d<3;++d){v[d+14]=u.lambda.value[d];v[d+19]=u.beta.value[d];}return v;}
Real Error(Real a,Real b){return std::abs(a-b)/std::max({Real(1),std::abs(a),std::abs(b)});}
int main(int argc,char**argv){Kokkos::initialize(argc,argv);int status=0;try{if(argc!=2)throw std::runtime_error("runtime output path argument");std::ofstream generated(argv[1]);generated<<std::setprecision(17);std::cout<<std::setprecision(17);cs::clock_oracle::Height height(48);while(std::cin.peek()!=EOF){std::cin>>std::ws;if(std::cin.peek()==EOF)break;int id=Read();Real time=Read(),x[3],r2=0;for(auto&v:x){v=Read();r2+=v*v;}Real radius=std::sqrt(r2),n[3]{};if(radius)for(int d=0;d<3;++d)n[d]=x[d]/radius;std::array<Real,184>target{};for(auto&v:target)v=Read();Real dt[22];for(auto&v:dt)v=Read();int sector=((id>=4&&id<=7)||id==9)?1:0;auto co=cs::tensor_oracle::Evaluate(radius,time,height);auto native=cs::tensor_oracle::NativeJet(co,radius,n,sector);cs::tensor_oracle::Coeff timeco{};for(int k=0;k<15;++k)timeco.c[k]=cs::tensor_oracle::Dt(co.c[k]);auto timeNative=cs::tensor_oracle::NativeJet(timeco,radius,n,sector);auto nativeDt=Values(timeNative);Real error=0,timeerror=0;int worst=-1;generated<<id<<" "<<time;for(Real v:x)generated<<" "<<v;
 cs::typed_support::Map<Real>(native,[&](int k,Real v){Real e=Error(v,target[k]);if(e>error){error=e;worst=k;}generated<<" "<<v;return v;});for(int k=0;k<22;++k){timeerror=std::max(timeerror,Error(nativeDt[k],dt[k]));generated<<" "<<nativeDt[k];}generated<<"\n";std::cout<<"{\"id\":"<<id<<",\"radius\":"<<radius<<",\"native184_error\":"<<error<<",\"worst_slot\":"<<worst<<",\"native22_time_error\":"<<timeerror<<"}"<<std::endl;}}
catch(const std::exception&e){std::cerr<<e.what()<<std::endl;status=2;}Kokkos::finalize();return status;}
