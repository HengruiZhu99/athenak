// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <fstream>
#include <iomanip>
#include <iostream>
#include "z4c/hyperboloidal/affine_shift_remainder.hpp"
#include "manufactured.hpp"
namespace h=z4c::hyperboloidal;
using P=h::AffineShiftPoint;using J=cubed_sphere::Jet;using V=h::VectorJet<Real>;
int Index(int i,int j,int k){return (k*5+j)*5+i;}
h::MetricJet<Real> Metric(const Real x[3],bool curved){
 auto out=manufactured::State(x,curved).metric;if(!curved)return out;
 J y[3];for(int a=0;a<3;++a)y[a]=J::Variable(x[a],a);
 const auto scale=manufactured::Exp(J(.03)*y[0]+J(.02)*y[1]*y[2]);
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){J f(out.g[i][j]);
   for(int a=0;a<3;++a){f.d[a]=out.dg[a][i][j];for(int b=0;b<3;++b)f.h[a][b]=out.ddg[a][b][i][j];}
   f=f*scale;out.g[i][j]=f.f;for(int a=0;a<3;++a){out.dg[a][i][j]=f.d[a];for(int b=0;b<3;++b)out.ddg[a][b][i][j]=f.h[a][b];}}
 return out;
}
V RemainderJet(const Real x[3],const h::PuncturePosition &point,Real nu){
 V out{};if(nu<0)return out;J y[3];for(int a=0;a<3;++a)y[a]=J::Variable(x[a]-point.position[a],a);
 const auto factor=cubed_sphere::Power(y[0]*y[0]+y[1]*y[1]+y[2]*y[2],nu/2);
 for(int i=0;i<3;++i){const auto u=J((i+1)/16.)*factor*y[i]+J((i+1)/32.)*y[0]*y[1]
     +J(1./64)*y[0]*y[2]*y[2]+J(1./128)*y[0]*y[0]*y[1]*y[2];
   out.value[i]=u.f;for(int a=0;a<3;++a){out.d[a][i]=u.d[a];for(int b=0;b<3;++b)out.dd[a][b][i]=u.h[a][b];}}
 return out;
}
std::array<long double,3> RemainderLong(const Real x[3],const h::PuncturePosition &point,Real nu){
 std::array<long double,3> out{};if(nu<0)return out;long double y[3],r2=0;
 for(int a=0;a<3;++a){y[a]=x[a]-point.position[a];r2+=y[a]*y[a];}
 for(int i=0;i<3;++i)out[i]=(i+1)/16.L*std::pow(r2,static_cast<long double>(nu)/2)*y[i]
   +(i+1)/32.L*y[0]*y[1]+1/64.L*y[0]*y[2]*y[2]+1/128.L*y[0]*y[0]*y[1]*y[2];return out;
}
template<typename T>V FD(const std::vector<std::array<T,3>>&data,Real spacing){
 V out{};const long double c[5]={1/12.L,-2/3.L,0,2/3.L,-1/12.L};
 const long double c2[5]={-1/12.L,4/3.L,-2.5L,4/3.L,-1/12.L};
 for(int component=0;component<3;++component){const long double anchor=data[Index(2,2,2)][component];out.value[component]=static_cast<Real>(anchor);
   for(int a=0;a<3;++a){long double d=0,dd=0;
     for(int s=0;s<5;++s){int index[3]={2,2,2};index[a]=s;
       const long double difference=static_cast<long double>(data[Index(index[0],index[1],index[2])][component])-anchor;
       d+=c[s]*difference;dd+=c2[s]*difference;}
     out.d[a][component]=static_cast<Real>(d/spacing);out.dd[a][a][component]=static_cast<Real>(dd/(static_cast<long double>(spacing)*spacing));}
   for(int a=0;a<3;++a)for(int b=a+1;b<3;++b){long double dd=0;
     for(int s=0;s<5;++s)for(int t=0;t<5;++t){int index[3]={2,2,2};index[a]=s;index[b]=t;
       dd+=c[s]*c[t]*(static_cast<long double>(data[Index(index[0],index[1],index[2])][component])-anchor);}
     out.dd[a][b][component]=out.dd[b][a][component]=static_cast<Real>(dd/(static_cast<long double>(spacing)*spacing));}}
 return out;
}
P Apply(const h::MetricJet<Real>&metric,const V&beta){P result;h::ApplyEllipticShift(metric,h::Geometry(metric),beta,result.data());return result;}
Real Difference(const P&a,const P&b){Real d=0;for(int i=0;i<3;++i)d=std::max(d,std::abs(a[i]-b[i]));return d;}
Real MaxValue(const std::vector<P>&v){Real x=0;for(const auto &row:v)for(Real a:row)x=std::max(x,std::abs(a));return x;}
int main(int argc,char **argv){Kokkos::initialize(argc,argv);int status=0;
try{std::ostream *out=&std::cout;std::ofstream file;if(argc>1){file.open(argv[1]);if(!file)throw std::runtime_error("output file");out=&file;}
 *out<<std::setprecision(17)<<"{\"cases\":[";bool first=true;int plans=0,tests=0;Real max_condition=0,max_moment_relative=0,max_reconstruction=0,max_recenter=0,max_boundary=0,max_affine_curved=0,max_recentered_state=0,max_recentered_operator=0;
 for(bool translated:{false,true})for(bool curved:{false,true})for(bool near:{true,false})for(int exponent=2;exponent<=9;++exponent){
   const Real spacing=std::pow(10.,-exponent);h::PuncturePosition point;
   if(translated)point.position={.037,.013,-.021};point.velocity={-.17,.08,-.11};point.time=.2;point.stage=2;point.revision=7;
   const P offset={near?.5:8.5,.5,.5};P query;std::vector<P> points;std::vector<Real> volume;
   for(int a=0;a<3;++a)query[a]=point.position[a]+spacing*offset[a];
   for(int k=0;k<5;++k)for(int j=0;j<5;++j)for(int i=0;i<5;++i){P x;const int delta[3]={i-2,j-2,k-2};
     for(int a=0;a<3;++a)x[a]=point.position[a]+spacing*(offset[a]+delta[a]);points.push_back(x);
     const auto geometry=h::Geometry(Metric(x.data(),curved));if(!geometry.valid)throw std::runtime_error("invalid nonflat SPD metric");
     volume.push_back(spacing*spacing*spacing*std::sqrt(geometry.determinant));}
   h::AffineShiftMoments moments(points,volume,point,(near?4:12)*spacing);++plans;max_condition=std::max(max_condition,moments.Condition());
   auto next=point;next.position[0]+=.2*spacing;next.position[1]-=.1*spacing;next.position[2]+=.15*spacing;++next.revision;
   h::AffineShiftMoments moved_moments(points,volume,next,(near?4:12)*spacing);++plans;max_condition=std::max(max_condition,moved_moments.Condition());
   for(int family=0;family<4;++family){const Real nu=family<2?-1:(family==2?.5:1.09);h::AffineShiftRemainder chart;chart.puncture=point;chart.b={.17,-.08,.11};
     if(family)chart.B={P{.13,.02,-.04},P{-.03,.07,.05},P{.01,-.06,-.09}};
     std::vector<P> beta;std::vector<std::array<long double,3>> ideal_remainder;
     for(const auto &x:points){const auto u=RemainderJet(x.data(),point,nu);P stored;
       for(int i=0;i<3;++i)stored[i]=u.value[i];chart.u.push_back(stored);
       const auto full=chart.ComposeJet(x.data(),u);beta.push_back({full.value[0],full.value[1],full.value[2]});
       ideal_remainder.push_back(RemainderLong(x.data(),point,nu));}
     const auto original=chart;const auto exact=chart.ComposeJet(query.data(),RemainderJet(query.data(),point,nu));
     const auto metric=Metric(query.data(),curved);const auto exact_l=Apply(metric,exact);
     const auto affine_l=Apply(metric,chart.AffineJet(query.data()));if(curved)for(Real x:affine_l)max_affine_curved=std::max(max_affine_curved,std::abs(x));
     const auto oracle=original.ComposeJet(query.data(),FD(ideal_remainder,spacing));const auto oracle_l=Apply(metric,oracle);
     moments.Rebalance(chart);const Real size=MaxValue(chart.u);const Real normalized_moment=moments.MaximumMoment(chart)/(size?size:1);
     max_moment_relative=std::max(max_moment_relative,normalized_moment);
     for(size_t p=0;p<points.size();++p){V u{};for(int i=0;i<3;++i)u.value[i]=chart.u[p][i];const auto reconstructed=chart.ComposeJet(points[p].data(),u);
       for(int i=0;i<3;++i)max_reconstruction=std::max(max_reconstruction,std::abs(reconstructed.value[i]-beta[p][i]));}
     const auto separate=chart.ComposeJet(query.data(),FD(chart.u,spacing));const auto separate_l=Apply(metric,separate);
     const auto direct_l=Apply(metric,FD(beta,spacing));const auto nodal=moments.FromNodal(beta,point);
     const auto recovered_l=Apply(metric,nodal.ComposeJet(query.data(),FD(nodal.u,spacing)));
     // Recenter the physical affine field, then canonically rebalance moments
     // around the moved support without reconstructing large nodal beta.
     auto moved=chart;
     const auto affine_before=moved.AffineJet(query.data());moved.Recenter(next);const auto affine_after=moved.AffineJet(query.data());
     for(int i=0;i<3;++i)max_recenter=std::max(max_recenter,std::abs(affine_before.value[i]-affine_after.value[i]));
     moved_moments.Rebalance(moved);
     const Real movedsize=MaxValue(moved.u),movedmoment=moved_moments.MaximumMoment(moved)/(movedsize?movedsize:1);
     max_moment_relative=std::max(max_moment_relative,movedmoment);
     if(movedmoment>1.e-10)throw std::runtime_error("recentered canonical moments not zero");
     for(size_t p=0;p<points.size();++p){V u{};for(int i=0;i<3;++i)u.value[i]=moved.u[p][i];const auto full=moved.ComposeJet(points[p].data(),u);
       for(int i=0;i<3;++i)max_recentered_state=std::max(max_recentered_state,std::abs(full.value[i]-beta[p][i]));}
     const auto moved_l=Apply(metric,moved.ComposeJet(query.data(),FD(moved.u,spacing)));
     max_recentered_operator=std::max(max_recentered_operator,Difference(moved_l,separate_l));
     if(moved.u.size()!=125||chart.u.size()!=125)throw std::runtime_error("compact moments removed physical samples");
     for(int axis=0;axis<3;++axis)for(int sign:{-1,1}){Real x[3]{};x[axis]=sign;const auto g=h::Geometry(Metric(x,curved));Real boundary[3];
       if(!h::FixedAngleScriShift(g,1.,1.,x,boundary))throw std::runtime_error("null boundary oracle");
       const auto boundary_u=chart.BoundaryRemainder(x,boundary);const auto affine=chart.AffineJet(x);
       for(int i=0;i<3;++i)max_boundary=std::max(max_boundary,std::abs(affine.value[i]+boundary_u[i]-boundary[i]));}
     if(family<2&&Difference(separate_l,exact_l)!=0)throw std::runtime_error("constant/anisotropic affine mode not exact");
     if(normalized_moment>1.e-10)throw std::runtime_error("normalized remainder moments not zero");
     if(!first)*out<<',';first=false;++tests;
     *out<<"{\"translated\":"<<translated<<",\"curved\":"<<curved<<",\"near\":"<<near<<",\"h\":"<<spacing<<",\"family\":"<<family
       <<",\"nu\":"<<nu<<",\"gram_condition\":"<<moments.Condition()<<",\"support_points\":"<<moments.SupportPoints()<<",\"all_samples\":"<<moments.Size()
       <<",\"full_original_L_scale\":"<<Difference(exact_l,P{})<<",\"direct_original_residual\":"<<Difference(direct_l,exact_l)
       <<",\"separate_original_residual\":"<<Difference(separate_l,exact_l)<<",\"nodal_recovery_original_residual\":"<<Difference(recovered_l,exact_l)
       <<",\"truncation_coordinate_control\":"<<Difference(oracle_l,exact_l)<<",\"direct_arithmetic_defect\":"<<Difference(direct_l,oracle_l)
       <<",\"separate_arithmetic_defect\":"<<Difference(separate_l,oracle_l)<<",\"normalized_moment\":"<<normalized_moment<<"}";
   }
 }
 // Negative contracts independent of the tiny-grid family.
 h::PuncturePosition p;std::vector<P> line={{-.1,0,0},{0,0,0},{.1,0,0},{.2,0,0}};std::vector<Real> vol(4,1);int rejected=0;
 auto reject=[&](auto action){bool fail=false;try{action();}catch(const std::exception&){fail=true;}if(!fail)throw std::runtime_error("invalid moments accepted");++rejected;};
 reject([&]{h::AffineShiftMoments invalid(line,vol,p,1);});reject([&]{h::AffineShiftMoments invalid(line,vol,p,0);});
 std::vector<P> cube;for(int i=0;i<8;++i)cube.push_back({i&1?.1:-.1,i&2?.1:-.1,i&4?.1:-.1});vol.assign(8,1);
 auto bad=vol;bad[0]=0;reject([&]{h::AffineShiftMoments invalid(cube,bad,p,1);});
 h::AffineShiftMoments valid(cube,vol,p,1);h::AffineShiftRemainder c;c.puncture=p;c.u.assign(8,P{});++c.puncture.revision;
 reject([&]{valid.Rebalance(c);});c.puncture=p;c.u.pop_back();reject([&]{valid.Rebalance(c);});
 if(max_reconstruction>2.e-15||max_recenter>2.e-15||max_recentered_state>2.e-15||max_boundary>2.e-15||!(max_affine_curved>1.e-6))
   throw std::runtime_error("reconstruction/recenter/boundary/fullmetric correction contract");
 *out<<"],\"summary\":{\"real_mantissa_bits\":"<<std::numeric_limits<Real>::digits
   <<",\"long_double_mantissa_bits\":"<<std::numeric_limits<long double>::digits<<",\"moment_plans\":"<<plans<<",\"local_vector_tests\":"<<tests<<",\"maximum_gram_condition\":"<<max_condition
   <<",\"maximum_relative_moment\":"<<max_moment_relative<<",\"maximum_reconstruction_change\":"<<max_reconstruction
   <<",\"maximum_recenter_change\":"<<max_recenter<<",\"maximum_recentered_state_change\":"<<max_recentered_state
   <<",\"maximum_recentered_operator_change\":"<<max_recentered_operator<<",\"maximum_full_boundary_reconstruction_error\":"<<max_boundary
   <<",\"maximum_curved_affine_operator\":"<<max_affine_curved<<",\"negative_contract_rejections\":"<<rejected
   <<",\"implementation_checks_passed\":true,\"qualification\":\"local representation/arithmetic only; augmented elliptic solve and evolution not implemented\"}}\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';status=1;}
Kokkos::finalize();return status;}
