#ifndef PRIVATE_FINITE_RB_RADIAL_NORMALIZATION_HPP_
#define PRIVATE_FINITE_RB_RADIAL_NORMALIZATION_HPP_
#include "point_energy.hpp"
struct R2 {
  double v=0,d=0,dd=0;R2()=default;R2(double a):v(a){}R2(double a,double b,double c):v(a),d(b),dd(c){}
  R2&operator+=(const R2&a){v+=a.v;d+=a.d;dd+=a.dd;return *this;}
  R2&operator-=(const R2&a){v-=a.v;d-=a.d;dd-=a.dd;return *this;}
  R2&operator*=(const R2&a){const auto old=*this;v=old.v*a.v;d=old.d*a.v+old.v*a.d;dd=old.dd*a.v+2*old.d*a.d+old.v*a.dd;return *this;}
  friend R2 operator+(R2 a,const R2&b){return a+=b;}friend R2 operator-(R2 a,const R2&b){return a-=b;}
  friend R2 operator*(R2 a,const R2&b){return a*=b;}friend R2 operator-(R2 a){return {-a.v,-a.d,-a.dd};}
};
R2 Inverse(const R2&a){return {1/a.v,-a.d/(a.v*a.v),2*a.d*a.d/(a.v*a.v*a.v)-a.dd/(a.v*a.v)};}
R2 operator/(const R2&a,const R2&b){return a*Inverse(b);}
R2 SquareRoot(const R2&a){const double z=std::sqrt(a.v);return {z,a.d/(2*z),a.dd/(2*z)-a.d*a.d/(4*z*z*z)};}
using RadialRaw=std::array<R2,22>;using RadialTensor=std::array<std::array<R2,3>,3>;
R2 Along(const TJ&a,const X3&n){R2 r(a.value);for(int i=0;i<3;++i){r.d+=n[i]*a.d[i];for(int j=0;j<3;++j)r.dd+=n[i]*n[j]*a.dd[i][j];}return r;}
RadialRaw RadialVariation(const Jet&u,const X3&n){const auto a=VariationJets(u);RadialRaw r{};for(int f=0;f<22;++f)r[f]=Along(a[f],n);return r;}
RadialTensor MatrixInverse(const RadialTensor&a){RadialTensor co{},out{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)
  co[i][j]=a[(i+1)%3][(j+1)%3]*a[(i+2)%3][(j+2)%3]-a[(i+1)%3][(j+2)%3]*a[(i+2)%3][(j+1)%3];
  R2 det{};for(int j=0;j<3;++j)det+=a[0][j]*co[0][j];for(int i=0;i<3;++i)for(int j=0;j<3;++j)out[i][j]=co[j][i]/det;return out;}
struct ReferenceRadial {
  R2 alpha{},chi{},omega{},c{},beta_n{};RadialTensor metric{},inverse{},a{},raised_a{};
  std::array<std::array<R2,3>,3> frame{},coframe{};
};
ReferenceRadial MakeReferenceRadial(const hyp::LayerPoint<double>&p,const X3&n,const X3&t,const X3&v){
  ReferenceRadial b{};
  auto scalar=[&](const auto&a){TJ j{};j.value=a.value;for(int i=0;i<3;++i){j.d[i]=a.d[i];for(int k=0;k<3;++k)j.dd[i][k]=a.dd[i][k];}return Along(j,n);};
  b.alpha=scalar(p.state.alpha);b.chi=scalar(p.state.chi);
  TJ oj{};oj.value=p.omega;for(int i=0;i<3;++i){oj.d[i]=p.domega[i];for(int j=0;j<3;++j)oj.dd[i][j]=p.omega_hessian[i][j];}b.omega=Along(oj,n);
  const auto bar=hyp::PenroseMetric(p.state.metric,p.state.chi);R2 gamma_nn{};
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){TJ g{};g.value=p.state.metric.g[i][j];TJ ar{};ar.value=p.state.a.k[i][j];TJ bg{};bg.value=bar.g[i][j];
    for(int d=0;d<3;++d){g.d[d]=p.state.metric.dg[d][i][j];ar.d[d]=p.state.a.dk[d][i][j];bg.d[d]=bar.dg[d][i][j];for(int e=0;e<3;++e){g.dd[d][e]=p.state.metric.ddg[d][e][i][j];bg.dd[d][e]=bar.ddg[d][e][i][j];}}
    b.metric[i][j]=Along(g,n);b.a[i][j]=Along(ar,n);gamma_nn+=R2(n[i]*n[j])*Along(bg,n);
  }
  b.c=Inverse(SquareRoot(gamma_nn));b.inverse=MatrixInverse(b.metric);
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int k=0;k<3;++k)for(int l=0;l<3;++l)b.raised_a[i][j]+=b.inverse[i][k]*b.inverse[j][l]*b.a[k][l];
  for(int i=0;i<3;++i){b.frame[0][i]=b.c*R2(n[i]);b.frame[1][i]=R2(t[i]);b.frame[2][i]=R2(v[i]);b.coframe[0][i]=R2(n[i])/b.c;b.coframe[1][i]=R2(t[i]);b.coframe[2][i]=R2(v[i]);
    TJ bj{};bj.value=p.state.beta.value[i];for(int d=0;d<3;++d){bj.d[d]=p.state.beta.d[d][i];for(int e=0;e<3;++e)bj.dd[d][e]=p.state.beta.dd[d][e][i];}b.beta_n+=R2(n[i])*Along(bj,n)/b.c;
  }
  return b;
}
struct R1 {double v=0,d=0;};
// No V.dd accessor: second A-reference jets are neither supplied nor consumed.
struct NormalizedRadial {std::array<R2,10>u{};std::array<R1,10>v{};};
NormalizedRadial Normalize(const RadialRaw&d,const ReferenceRadial&b){
  NormalizedRadial out{};std::array<R2,10>velocity{};RadialTensor g{},a{},h{},s{};for(int k=0;k<6;++k){g[ti[k]][tj[k]]=g[tj[k]][ti[k]]=d[1+k];a[ti[k]][tj[k]]=a[tj[k]][ti[k]]=d[8+k];}
  R2 required{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)required+=b.raised_a[i][j]*g[i][j];
  for(int aa=0;aa<3;++aa)for(int bb=0;bb<3;++bb)for(int i=0;i<3;++i)for(int j=0;j<3;++j){h[aa][bb]+=b.frame[aa][i]*b.frame[bb][j]*g[i][j]/b.chi;
    s[aa][bb]+=b.frame[aa][i]*b.frame[bb][j]*(a[i][j]-b.metric[i][j]*required/R2(3))/b.chi;}
  out.u[0]=d[18]/b.alpha;out.u[1]=d[0]/b.chi;
  out.u[2]=h[0][0];out.u[3]=h[0][1];out.u[4]=h[0][2];out.u[5]=(h[1][1]-h[2][2])/R2(2);out.u[6]=h[1][2];
  velocity[0]=d[7]/b.omega;velocity[1]=d[17]/b.omega;
  velocity[2]=s[0][0];velocity[3]=s[0][1];velocity[4]=s[0][2];velocity[5]=(s[1][1]-s[2][2])/R2(2);velocity[6]=s[1][2];
  for(int aa=0;aa<3;++aa)for(int i=0;i<3;++i){out.u[7+aa]+=b.coframe[aa][i]*d[19+i]/b.alpha;velocity[7+aa]+=b.chi*b.coframe[aa][i]*d[14+i];}
  for(int k=0;k<10;++k)out.v[k]={velocity[k].v,velocity[k].d};
  return out;
}
void MakeScreen(const X3&n,X3&t,X3&v){X3 a=std::abs(n[2])<.8?X3{0,0,1}:X3{0,1,0};t={a[1]*n[2]-a[2]*n[1],a[2]*n[0]-a[0]*n[2],a[0]*n[1]-a[1]*n[0]};double z=0;for(auto x:t)z+=x*x;z=std::sqrt(z);for(auto&x:t)x/=z;v={n[1]*t[2]-n[2]*t[1],n[2]*t[0]-n[0]*t[2],n[0]*t[1]-n[1]*t[0]};}
#endif
