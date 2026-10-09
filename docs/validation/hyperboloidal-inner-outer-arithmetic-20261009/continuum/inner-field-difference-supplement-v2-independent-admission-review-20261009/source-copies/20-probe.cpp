// HELD additive scalar arithmetic gate; no actual gauge/kernel/grid queries.
#include "dual_helpers.hpp"
#include "inner_gauge.hpp"
#include <string>
namespace inner {template<> struct Number<D>{
  static double Value(D x){return x.v;}static double Derivative(D x){return x.d;}
  static D Make(double x,double d){return D(x,d);}
};}
namespace {
constexpr int seeds[6][3]={{0,0,0},{1,0,0},{0,1,0},{1,1,0},{1,-2,0},{0,0,1}};
void Value(D x){
  if(!std::isfinite(x.v)||!std::isfinite(x.d))throw std::runtime_error("nonfinite supplement output");
  std::cout<<"{\"value_hex\":\""<<std::hexfloat<<x.v<<"\",\"dual_hex\":\""<<x.d<<"\"}";
}
void Audit(const inner::Audit&q){std::cout<<"{\"dA_near\":"<<q.dA_near<<",\"dA_far\":"<<q.dA_far
 <<",\"dc_near\":"<<q.dc_near<<",\"dc_far\":"<<q.dc_far
 <<",\"dal_near\":"<<q.dal_near<<",\"dal_far\":"<<q.dal_far<<'}';}
void Context(const char*kind,int witness,int seed){
  std::cout<<"{\"kind\":\""<<kind<<"\",\"witness\":"<<witness<<",\"seed\":"<<seed
    <<",\"xi\":["<<seeds[seed][0]<<','<<seeds[seed][1]<<','<<seeds[seed][2]<<']';
}
}
namespace inner {
// The three marked expression lines are copied byte-for-byte from source002.
template<class T>T OldNearSquareDifference(T a,T chi,T h,T ch,Audit*audit){
  const T da=a-h,dchi=chi-ch;
  const T dA=Product<T>({a+h,da,chi},audit)+Product<T>({h,h,dchi},audit);
  return dA;
}
template<class T>T OldNearChiDifference(T chi,T gradient,T ch,T reference_gradient,Audit*audit){
  hyp::Z4cJet<T>u{};hyp::LayerPoint<T>p{};u.chi.d[0]=gradient;p.state.chi.d[0]=reference_gradient;
  const T dchi=chi-ch;T dc[1]{};const int i=0;
    dc[i]=(u.chi.d[i]-p.state.chi.d[i])-Product<T>({dchi,p.state.chi.d[i]/ch},audit);
  return dc[0];
}
template<class T>T OldNearLapseDifference(T a,T gradient,T h,T reference_gradient,Audit*audit){
  hyp::Z4cJet<T>u{};hyp::LayerPoint<T>p{};u.alpha.d[0]=gradient;p.dalpha[0]=reference_gradient;
  const T da=a-h;T dal[1]{};const int i=0;
    dal[i]=(u.alpha.d[i]-p.dalpha[i])-Product<T>({da,p.dalpha[i]/h},audit);
  return dal[0];
}
}
namespace {
void Witnesses(bool negative){
  for(int witness=1;witness<=3;++witness)for(int seed=0;seed<(negative?1:6);++seed){
    const double av=std::ldexp(1.,witness==2?300:-300);
    const double cv=std::ldexp(1.,witness==1?601:witness==2?-600:600);
    const D a(av,av*seeds[seed][0]),chi(cv,cv*seeds[seed][1]),h(1,0),ch(1,0);
    const D grad(0,(witness==2?cv:av)*seeds[seed][2]);inner::Audit audit{};D answer;
    if(witness==1)answer=negative?inner::OldNearSquareDifference(a,chi,h,ch,&audit):inner::FieldSquareDifference(a,chi,h,ch,&audit);
    if(witness==2){const D d=negative?inner::OldNearChiDifference(chi,grad,ch,D(1),&audit):inner::FieldLogGradientDifference(chi,grad,ch,D(1),inner::LogGradientField::Chi,&audit);answer=inner::Product<D>({a,a,d},&audit);}
    if(witness==3){const D d=negative?inner::OldNearLapseDifference(a,grad,h,D(1),&audit):inner::FieldLogGradientDifference(a,grad,h,D(1),inner::LogGradientField::Alpha,&audit);answer=inner::Product<D>({D(-1),a,chi,d},&audit);}
    Context(negative?"negative-old-near":"witness",witness,seed);
    std::cout<<",\"alpha\":";Value(a);std::cout<<",\"chi\":";Value(chi);std::cout<<",\"gradient\":";Value(grad);
    std::cout<<",\"answer\":";Value(answer);std::cout<<",\"audit\":";Audit(audit);std::cout<<"}\n";
  }
}
void Bounds(){
  const double delta=std::ldexp(1.,-40),ratios[6]={.5-delta,.5,.5+delta,2-delta,2,2+delta};
  for(int exponent:{-400,0,400})for(int ratio=0;ratio<6;++ratio)for(int seed=0;seed<6;++seed){
    const double hv=std::ldexp(1.,exponent),av=hv*ratios[ratio];
    const D a(av,av*seeds[seed][0]),chi(1,seeds[seed][1]),h(hv,0),ch(1,0),grad(av/8,av*seeds[seed][2]),reference_gradient(hv/4,0);
    inner::Audit audit{};const D da=inner::FieldSquareDifference(a,chi,h,ch,&audit);
    const D dc=inner::FieldLogGradientDifference(a,grad,h,reference_gradient,inner::LogGradientField::Chi,&audit);
    const D dal=inner::FieldLogGradientDifference(a,grad,h,reference_gradient,inner::LogGradientField::Alpha,&audit);
    Context("near-bound",0,seed);std::cout<<",\"reference_exponent\":"<<exponent<<",\"ratio_index\":"<<ratio
      <<",\"near\":"<<(inner::NearFieldValue(av,hv)?"true":"false")<<",\"alpha\":";Value(a);
    std::cout<<",\"chi\":";Value(chi);std::cout<<",\"reference\":";Value(h);std::cout<<",\"gradient\":";Value(grad);
    std::cout<<",\"dA\":";Value(da);std::cout<<",\"dc\":";Value(dc);std::cout<<",\"dal\":";Value(dal);
    std::cout<<",\"audit\":";Audit(audit);std::cout<<"}\n";
  }
}
}
int main(int argc,char**argv){try{
  if(argc!=2||std::string(argv[1])!="--all")throw std::runtime_error("exact fixed --all mode required");
  Witnesses(false);Witnesses(true);Bounds();return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
