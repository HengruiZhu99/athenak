#define main snapshot_symbol_main
#include "principal_snapshot.cpp"
#undef main
#include <algorithm>
int main() {
 std::cout<<std::setprecision(17)<<'[';
 bool first=true; double difference=0;int cases=0;
 for(double alpha:{.01,.1,1.})for(double chi:{.0001,.1,1.})
  for(double r:{.01,.2,.6,.9})for(bool oblique:{false,true})for(bool phys:{false,true}) {
   double m[20][20]{},d[20][20]{};
   Extract(alpha,chi,r,oblique,phys,m,0);
   Extract(alpha,chi,r,oblique,phys,d,.5);
   for(int i=0;i<20;++i)for(int j=0;j<20;++j)difference=std::max(difference,std::abs(m[i][j]-d[i][j]));
   const auto c=hyp::LayerCoefficients(r,alpha,hyp::LayerGaugeParameters{});
   if(!first)std::cout<<',';first=false;
   std::cout<<"{\"alpha\":"<<alpha<<",\"chi\":"<<chi<<",\"r\":"<<r
    <<",\"oblique\":"<<oblique<<",\"physical_trace_lapse\":"<<phys
    <<",\"W\":"<<c.weight<<",\"f\":"<<c.f<<",\"q\":"<<c.q<<",\"mu\":"<<c.mu<<",\"M\":[";
   for(int i=0;i<20;++i){if(i)std::cout<<',';std::cout<<'[';for(int j=0;j<20;++j){if(j)std::cout<<',';std::cout<<d[i][j];}std::cout<<']';}
   std::cout<<"]}";++cases;
  }
 std::cout<<"]\n";
 std::cerr<<"cases="<<cases<<" eta_principal_difference="<<difference<<"\n";
 return difference==0 ? 0:1;
}
