// Exploratory, constant-coefficient interior stencil audit. No native runtime edit.
#include "athena.hpp"
#include "dual_helpers.hpp"
#include "z4c/hyperboloidal/layer_gauge.hpp"
#include "utils/finite_diff.hpp"
#include "z4c/hyperboloidal/interior_dissipation.hpp"

struct Plane {
  double theta[3]{}, phase=0;
  double operator()(int,int k,int j,int i) const {
    return std::cos(theta[0]*i+theta[1]*j+theta[2]*k+phase);
  }
};
struct Composed {
  Plane plane; int axis=0; double idx[3]{1,1,1};
  double operator()(int m,int k,int j,int i) {
    return Dx<3>(axis,idx,plane,m,k,j,i);
  }
};
struct Velocity {
  double v[3]{};
  double operator()(int,int a,int,int,int) const { return v[a]; }
};
struct FlatPlane {
  Plane plane;
  double operator()(int s) const {
    const int i=s%21-10,j=(s/21)%21-10,k=s/(21*21)-10;
    return plane(0,k,j,i);
  }
};
struct AllActive { bool operator()(int) const { return true; } };
struct Manufactured {
  double h;
  double operator()(int,int k,int j,int i) const {
    return std::exp(.7*(.3+h*i)-.4*(-.2+h*j)+.2*(.15+h*k));
  }
};
struct ManufacturedFirst {
  Manufactured field; int axis; double idx[3];
  double operator()(int m,int k,int j,int i) {
    return Dx<3>(axis,idx,field,m,k,j,i);
  }
};

struct Symbols {
  C d[3]{}, native[3][3]{}, compatible[3][3]{}, upwind{},ko{};
};
Symbols Stencils(const std::array<double,3>&theta,double h,
                 const std::array<double,3>&velocity) {
  Symbols s; double idx[3]={1/h,1/h,1/h};
  Velocity vel{{velocity[0],velocity[1],velocity[2]}};
  for(int phase=0;phase<2;++phase) {
    Plane p{{theta[0],theta[1],theta[2]},phase?-std::acos(-1.)/2:0};
    // Real cosine + i sine gives exp(i theta.x).
    const C factor=phase?I:C(1);
    for(int i=0;i<3;++i) {
      s.d[i]+=factor*Dx<3>(i,idx,p,0,0,0,0);
      s.upwind+=factor*Lx<3>(i,idx,vel,p,0,i,0,0,0);
      Composed q{p,i,{1/h,1/h,1/h}};
      for(int j=0;j<3;++j) {
        s.compatible[i][j]+=factor*Dx<3>(j,idx,q,0,0,0,0);
        s.native[i][j]+=factor*(i==j?Dxx<3>(i,idx,p,0,0,0,0):
                                    Dxy<3>(i,j,idx,p,0,0,0,0));
      }
    }
    const int strides[3]={1,21,441}; const double hs[3]={h,h,h};
    s.ko+=factor*hyp::InteriorKOSixth(FlatPlane{p},AllActive{},
                                    10+21*10+441*10,strides,hs);
  }
  return s;
}

struct Kernel {
  M l{},q{};
};
Kernel ExtractDiscrete(const Symbols&s,bool compatible,double alpha,double chi,
                       bool spd,double radius,const std::array<double,3>&velocity) {
  hyp::LayerPoint<double> pd{}; pd.omega=1;pd.alpha=alpha;pd.radius=radius;
  pd.state.alpha.value=alpha;pd.state.chi.value=chi;
  const double b[3][3]={{1.3,.2,-.1},{0,.8,.12},{0,0,1/1.04}};
  for(int i=0;i<3;++i) {
    pd.beta[i]=pd.state.beta.value[i]=velocity[i];
    for(int j=0;j<3;++j)
      for(int k=0;k<3;++k)
        pd.state.metric.g[i][j]+=(spd?b[k][i]*b[k][j]:double(i==k&&j==k));
  }
  hyp::LayerPoint<D> p{};p.omega=1;p.alpha=alpha;p.radius=radius;
  p.state=Lift(pd.state);for(int i=0;i<3;++i)p.beta[i]=velocity[i];
  hyp::LayerGaugeParameters gauge;gauge.preferred_source=false;
  gauge.physical_trace_lapse=true;gauge.lapse_outer=0;gauge.shift_outer=0;
  gauge.scri_lapse_damping=0;
  Kernel result;
  for(int col=0;col<20;++col)for(int phase=0;phase<2;++phase) {
    const auto part=[phase](C z){return phase?z.imag():z.real();};
    Jet u=Lift(pd.state);J seed(D(0,phase?0:1));
    for(int i=0;i<3;++i) {
      seed.d[i]=D(0,part(s.d[i]));
      for(int j=0;j<3;++j)
        seed.dd[i][j]=D(0,part(compatible?s.compatible[i][j]:s.native[i][j]));
    }
    Seed(u,col,seed);Consistent(u);
    auto o=Omega(u,pd);hyp::Z4cRHS<D> f;
    if(!hyp::AssembleInterior(hyp::ConformalRHS(u,o,D(0),D(0)),o.omega,f))
      throw std::runtime_error("invalid geometric extraction");
    hyp::GaugeRHS<D> g;
    if(!hyp::AssembleGaugeInterior(hyp::InteriorLayerGauge(p,u,gauge),D(1),g))
      throw std::runtime_error("invalid gauge extraction");
    double v[20]{};v[0]=g.alpha.d;v[1]=f.chi.d;v[2]=f.trace.d;v[3]=f.theta.d;
    const int ti[5]={0,0,0,1,1},tj[5]={0,1,2,1,2};
    for(int j=0;j<3;++j){v[4+j]=g.beta[j].d;v[17+j]=f.lambda[j].d;}
    for(int j=0;j<5;++j){v[7+j]=f.metric[ti[j]][tj[j]].d;v[12+j]=f.a[ti[j]][tj[j]].d;}
    const C factor=phase?I:C(1);
    for(int row=0;row<20;++row)result.l[row][col]+=factor*v[row];
    const auto q=Constraints(u,pd);
    for(int row=0;row<8;++row)result.q[row][col]+=factor*q[row];
  }
  return result;
}
void Matrix(const M&m,int rows=20) {
  std::cout<<'[';for(int i=0;i<rows;++i){if(i)std::cout<<',';std::cout<<'[';
  for(int j=0;j<20;++j){if(j)std::cout<<',';std::cout<<'['<<m[i][j].real()<<','<<m[i][j].imag()<<']';}
  std::cout<<']';}std::cout<<']';
}
void Complex(C z){std::cout<<'['<<z.real()<<','<<z.imag()<<']';}
int main(int argc,char**) {
  if(argc>1) {
    std::cout<<std::setprecision(17)<<'[';bool first=true;
    const double a[3]={.7,-.4,.2};
    for(double h:{.2,.1,.05,.025}) {
      const double idx[3]={1/h,1/h,1/h};Manufactured f{h};
      if(!first)std::cout<<',';first=false;
      std::cout<<"{\"h\":"<<h<<",\"exact_value\":"<<f(0,0,0,0)
               <<",\"D\":[";
      for(int i=0;i<3;++i){if(i)std::cout<<',';std::cout<<Dx<3>(i,idx,f,0,0,0,0);}
      std::cout<<"],\"S\":[";
      for(int i=0;i<3;++i){if(i)std::cout<<',';std::cout<<'[';
        ManufacturedFirst df{f,i,{1/h,1/h,1/h}};
        for(int j=0;j<3;++j){if(j)std::cout<<',';std::cout<<Dx<3>(j,idx,df,0,0,0,0);}
        std::cout<<']';}
      std::cout<<"]}";
    }
    std::cout<<"]\n";return 0;
  }
  std::cout<<std::setprecision(17)<<'[';bool first=true;
  const double pi=std::acos(-1.);
  const std::array<double,3> angles[]={
    {0,0,0},{.07,0,0},{.07,.11,-.09},{.4,.7,1.1},
    {1.5,0,0},{2.6,1.7,-2.1},{pi,0,0},{pi,pi,pi},
    {pi,.2,0},{pi-1e-3,pi-2e-3,pi-3e-3}};
  for(auto theta:angles)for(bool compatible:{false,true})
    for(bool spd:{false,true})for(double alpha:{.2,1.,3.})
      for(double chi:{.4,1.,2.})for(double radius:{0.,.65,.85}) {
        const std::array<double,3> velocity={.3,-.2,.1};const double h=.125;
        const auto s=Stencils(theta,h,velocity);
        const auto k=ExtractDiscrete(s,compatible,alpha,chi,spd,radius,velocity);
        if(!first)std::cout<<',';first=false;
        std::cout<<"{\"theta\":["<<theta[0]<<','<<theta[1]<<','<<theta[2]
          <<"],\"compatible\":"<<compatible<<",\"spd\":"<<spd
          <<",\"alpha\":"<<alpha<<",\"chi\":"<<chi<<",\"r\":"<<radius
          <<",\"h\":"<<h<<",\"D\":[";
        for(int i=0;i<3;++i){if(i)std::cout<<',';Complex(s.d[i]);}
        std::cout<<"],\"S\":[";for(int i=0;i<3;++i){if(i)std::cout<<',';std::cout<<'[';
          for(int j=0;j<3;++j){if(j)std::cout<<',';Complex(compatible?s.compatible[i][j]:s.native[i][j]);}std::cout<<']';}
        std::cout<<"],\"upwind\":";Complex(s.upwind);
        std::cout<<",\"KO\":";Complex(s.ko);
        std::cout<<",\"L\":";Matrix(k.l);std::cout<<",\"Q\":";Matrix(k.q,8);std::cout<<'}';
      }
  std::cout<<"]\n";
}
