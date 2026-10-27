// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <tuple>
#include "z4c/hyperboloidal/enriched_log_value.hpp"
namespace h=z4c::hyperboloidal;
using Plan=h::EnrichedLogValue;using Point=Plan::Point;
std::vector<Point> Donors(const Point &query,Real spacing){
 Point anchor;for(int a=0;a<3;++a)anchor[a]=spacing*(std::floor(query[a]/spacing)+.5);
 std::vector<Point> points;
 for(int k=-3;k<=3;++k)for(int j=-3;j<=3;++j)for(int i=-3;i<=3;++i)
   points.push_back({anchor[0]+spacing*i,anchor[1]+spacing*j,anchor[2]+spacing*k});
 std::sort(points.begin(),points.end(),[&](const Point&a,const Point&b){
   Real da=0,db=0;for(int d=0;d<3;++d){da+=std::pow(a[d]-query[d],2);db+=std::pow(b[d]-query[d],2);}
   return std::make_tuple(da,a)<std::make_tuple(db,b);});points.resize(125);return points;
}
Real Field(const Point &x,const h::PuncturePosition &point,Real power){
 const auto y=point.Relative(x.data());
 const Real polynomial=.2+.3*y[0]+.13*y[0]*y[1]+.1*y[0]*y[2]*y[2]
   +1.7*std::pow(y[0],5)-.9*y[1]*y[1]*y[2]*y[2]*y[2];
 return polynomial+power*std::log(std::hypot(y[0],std::hypot(y[1],y[2])));
}
void Require(bool condition,const char *message){if(!condition)throw std::runtime_error(message);}
int main(int argc,char **argv){Kokkos::initialize(argc,argv);int status=0;
try{int plans=0,fields=0,rejections=0,identity_checks=0,constant_checks=0;Real error=0,maximum_l1=0;
 const unsigned long revision=13;const Real sentinel=-314.159;
 for(Point center:{Point{},Point{.037,.013,-.021},Point{.1123,-.0312,.0771}})
 for(Real spacing:{.03,.015})for(Point offset:{Point{.31,-.24,.43},Point{1.27,-.67,.19}}){
   h::PuncturePosition point;point.position=center;point.velocity={-.17,.08,.03};point.time=.2;point.stage=2;point.revision=7;
   Point query;for(int a=0;a<3;++a)query[a]=center[a]+spacing*offset[a];
   const auto donors=Donors(query,spacing);for(const auto &x:donors)Require(x!=query,"off-grid query accidentally coincident");
   Plan plan;Require(plan.Build(query,donors,spacing,point,revision),"off-grid value plan rejected");
   Require(!plan.diagnostics().coincident_active_identity,"off-grid query used identity");++plans;
   maximum_l1=std::max(maximum_l1,plan.diagnostics().value_weight_l1);
   std::vector<Real> values;
   for(Real power:{0.,4.,2.,1.09,.5}){
     values.clear();for(const auto &x:donors)values.push_back(Field(x,point,power));const auto original=values;
     Real result=sentinel;Require(plan.Evaluate(values,point,revision,result),"fresh off-grid evaluation rejected");
     const Real defect=std::abs(result-Field(query,point,power));error=std::max(error,defect);
     Require(defect<1.e-10,"independent polynomial/log value reproduction failed");
     Require(values==original,"value interpolation mutated input");++fields;
   }
   values.assign(donors.size(),1.23456789);Real result=sentinel;
   Require(plan.Evaluate(values,point,revision,result)&&result==values[0],"constant value not preserved exactly");++constant_checks;
   Plan identity;const size_t selected=17;
   Require(identity.Build(donors[selected],donors,spacing,point,revision),"coincident active identity rejected");
   Require(identity.diagnostics().coincident_active_identity,"coincident active query did not use identity");
   for(size_t i=0;i<values.size();++i)values[i]=std::sin(.7*i)+.01*i;
   result=sentinel;Require(identity.Evaluate(values,point,revision,result)&&result==values[selected],"identity changed active value");++identity_checks;
   auto reject_evaluation=[&](const Plan&candidate,const std::vector<Real>&data,const h::PuncturePosition&position,unsigned long mesh_revision){
     Real untouched=sentinel;Require(!candidate.Evaluate(data,position,mesh_revision,untouched),"invalid/stale evaluation accepted");
     Require(untouched==sentinel,"failed evaluation mutated output");++rejections;};
   for(int change=0;change<5;++change){auto stale=point;
     if(change==0)stale.position[0]+=.001;if(change==1)stale.velocity[1]+=.001;
     if(change==2)stale.time+=.001;if(change==3)++stale.stage;if(change==4)++stale.revision;
     reject_evaluation(plan,values,stale,revision);}
   reject_evaluation(plan,values,point,revision+1);
   auto invalid=values;invalid[3]=std::numeric_limits<Real>::quiet_NaN();reject_evaluation(plan,invalid,point,revision);
   // Zero-weight identity donors still obey the finite-input contract.
   reject_evaluation(identity,invalid,point,revision);
   invalid=values;invalid.pop_back();reject_evaluation(plan,invalid,point,revision);
   Plan rejected;
   auto reject_build=[&](const Point&q,const std::vector<Point>&x,Real dx,Real maximum_l1=8){
     Require(!rejected.Build(q,x,dx,point,revision,maximum_l1),"invalid value geometry/basis accepted");
     reject_evaluation(rejected,values,point,revision);};
   reject_build(query,donors,0);reject_build(query,donors,-spacing);
   reject_build(query,donors,std::numeric_limits<Real>::infinity());reject_build(query,donors,spacing,.5);
   reject_build(point.position,donors,spacing);
   auto bad_points=donors;bad_points[0]=point.position;reject_build(query,bad_points,spacing);
   bad_points.resize(56);reject_build(query,bad_points,spacing);
   bad_points.assign(125,Point{.3,.4,.2});reject_build(query,bad_points,spacing);
   Require(rejected.diagnostics().status.find("basis_")==0,"rank-deficient geometry did not report basis rejection");
   reject_build(query,donors,spacing,1);Require(rejected.diagnostics().status=="value_weight_amplification","value L1 guard not exercised");
 }
 Require(error<1.e-10,"value reproduction acceptance");
 std::cout<<std::setprecision(17)<<"{\"offgrid_plans\":"<<plans<<",\"independent_field_checks\":"<<fields
   <<",\"identity_checks\":"<<identity_checks<<",\"constant_checks\":"<<constant_checks<<",\"rejections_without_output_mutation\":"<<rejections
   <<",\"maximum_value_error\":"<<error<<",\"maximum_value_weight_l1\":"<<maximum_l1
   <<",\"implementation_checks_passed\":true,\"qualification\":\"self-contained local scalar value contracts only\"}\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';status=1;}
Kokkos::finalize();return status;}
