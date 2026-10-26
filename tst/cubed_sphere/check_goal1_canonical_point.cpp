// Compare live-frame canonical formulas with actual native RWM principal rows.
#define main archived_principal_main
#include "tst/cubed_sphere/check_radial_principal_symbol.cpp"
#undef main
#include "mesh/cubed_sphere/research_canonical_ko.hpp"
namespace ck=cubed_sphere::canonical_ko;
void Chart(const Jet&u,const double raw[11],double chart[10]){
  chart[0]=raw[0];double g[3][3]{};
  for(int k=0;k<6;++k)g[ia[k]][ib[k]]=g[ib[k]][ia[k]]=raw[1+k];
  ck::MetricChart(ck::MetricFrame(u),g,chart+1);chart[6]=raw[7];for(int i=0;i<3;++i)chart[7+i]=raw[8+i];
}
void TangentSeed(Jet&u,const Jet&base,int col,const double n[3],double e){
  Seed(u,col,col<11?1:0,n,e);
  if((col>=1&&col<=6)||(col>=11&&col<=16)){
    const int k=col<11?col-1:col-11;const auto g=h::Geometry(base.metric);
    const double trace=(ia[k]==ib[k]?1:2)*g.inverse[ia[k]][ib[k]];
    for(int i=0;i<3;++i)for(int j=0;j<3;++j){
      if(col<11)for(int a=0;a<3;++a)u.metric.dg[a][i][j]-=e*trace*base.metric.g[i][j]*n[a]/3;
      else u.a.k[i][j]-=e*trace*base.metric.g[i][j]/3;
    }
  }
}
int main(int argc,char**argv){Kokkos::initialize(argc,argv);std::cout<<std::setprecision(17);
 if(argc!=2)throw std::runtime_error("need native samples");std::ifstream in(argv[1]);
 int family,index=0;double curvature,r0,r1,x[3],v[25],n[3];
 std::cout<<"case,principal_chart_error,inverse_native_action_error,trace_penalty_error,roundtrip_metric_chart_error,Omega\n";
 while(in>>family>>curvature>>r0>>r1>>x[0]>>x[1]>>x[2]){
  for(auto &q:v)if(!(in>>q))throw std::runtime_error("truncated values");for(auto &q:n)if(!(in>>q))throw std::runtime_error("missing covector");
  h::LayerReference<double> ref(1.,curvature,{true,r0,r1});auto p=ref.At(x[0],x[1],x[2]);Jet u{};
  u.chi.value=v[0];u.trace.value=v[7];u.theta.value=v[17];u.alpha.value=v[18];
  for(int k=0;k<6;++k){u.metric.g[ia[k]][ib[k]]=u.metric.g[ib[k]][ia[k]]=v[1+k];u.a.k[ia[k]][ib[k]]=u.a.k[ib[k]][ia[k]]=v[8+k];}
  double beta=0;for(int i=0;i<3;++i){u.lambda.value[i]=v[14+i];u.beta.value[i]=v[19+i];beta+=n[i]*u.beta.value[i];}
  double err=0,inverse_err=0,trerr=0,frameerr=0;
  for(int column=0;column<22;++column){Jet plus=u,minus=u;TangentSeed(plus,u,column,n,1);TangentSeed(minus,u,column,n,-1);
   double pp[10],pm[10],fp[22],fm[22],raw[11],normalq[11]{};ck::Momentum(plus,p.state,n,p.omega,pp);ck::Momentum(minus,p.state,n,p.omega,pm);
   Evaluate(plus,p,x,false,fp);Evaluate(minus,p,x,false,fm);
   for(int i=0;i<11;++i)raw[i]=(fp[i]-fm[i])/2;
   for(int a=0;a<3;++a){normalq[0]+=n[a]*(plus.chi.d[a]-minus.chi.d[a])/2;normalq[7]+=n[a]*(plus.alpha.d[a]-minus.alpha.d[a])/2;
    for(int k=0;k<6;++k)normalq[1+k]+=n[a]*(plus.metric.dg[a][ia[k]][ib[k]]-minus.metric.dg[a][ia[k]][ib[k]])/2;
    for(int k=0;k<3;++k)normalq[8+k]+=n[a]*(plus.beta.d[a][k]-minus.beta.d[a][k])/2;}
   double expected[10],qn[10];Chart(u,raw,expected);Chart(u,normalq,qn);
   for(int i=0;i<10;++i){expected[i]-=beta*qn[i];err=std::max(err,std::abs((pp[i]-pm[i])/2-expected[i])/std::max(1.,std::abs(expected[i])));}
  }
  for(int col=0;col<10;++col){double desired[10]{};desired[col]=1;h::Z4cRHS<double> penalty{};ck::Lift(u,p.omega,desired,penalty);Jet plus=u,minus=u;
   plus.trace.value+=penalty.trace;minus.trace.value-=penalty.trace;plus.theta.value+=penalty.theta;minus.theta.value-=penalty.theta;
   const auto g=h::Geometry(u.metric);double trace=0;
   for(int i=0;i<3;++i){plus.lambda.value[i]+=penalty.lambda[i];minus.lambda.value[i]-=penalty.lambda[i];for(int j=0;j<3;++j){plus.a.k[i][j]+=penalty.a[i][j];minus.a.k[i][j]-=penalty.a[i][j];trace+=g.inverse[i][j]*penalty.a[i][j];}}
   trerr=std::max(trerr,std::abs(trace));double fp[22],fm[22],raw[11],result[10];Evaluate(plus,p,x,false,fp);Evaluate(minus,p,x,false,fm);
   for(int i=0;i<11;++i)raw[i]=(fp[i]-fm[i])/2;Chart(u,raw,result);
   for(int i=0;i<10;++i)inverse_err=std::max(inverse_err,std::abs(result[i]-desired[i]));
   if(col>=1&&col<=5){double m[3][3]{},back[5];ck::InverseMetricChart(ck::MetricFrame(u),desired+1,m);ck::MetricChart(ck::MetricFrame(u),m,back);
    for(int i=0;i<5;++i)frameerr=std::max(frameerr,std::abs(back[i]-desired[1+i]));}
  }
  std::cout<<index++<<","<<err<<","<<inverse_err<<","<<trerr<<","<<frameerr<<","<<p.omega<<"\n";
 }
 Kokkos::finalize();
}
