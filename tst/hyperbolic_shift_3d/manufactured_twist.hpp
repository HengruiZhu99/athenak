// Test-only third-order Cartesian jets for a nonconstant regular scri metric.
// y=Rot_z(epsilon*z/r)x preserves radius, homogeneity and volume. Pullback of
// flat space gives g=J^T J, det(g)=1, chi=1, A=0, K=-3, Lambda=contracted Gamma.
// No field interpolation or finite differences define these reference jets.
#ifndef HYP3D_MANUFACTURED_TWIST_HPP_
#define HYP3D_MANUFACTURED_TWIST_HPP_
namespace manufactured_twist {
struct Jet3 {
 double f=0,d[3]{},dd[3][3]{},ddd[3][3][3]{};
 Jet3()=default;Jet3(double v):f(v){}
 static Jet3 Variable(double v,int a){Jet3 q(v);q.d[a]=1;return q;}
};
inline Jet3 operator+(const Jet3&a,const Jet3&b){Jet3 c(a.f+b.f);
 for(int i=0;i<3;++i){c.d[i]=a.d[i]+b.d[i];for(int j=0;j<3;++j){c.dd[i][j]=a.dd[i][j]+b.dd[i][j];
  for(int k=0;k<3;++k)c.ddd[i][j][k]=a.ddd[i][j][k]+b.ddd[i][j][k];}}return c;}
inline Jet3 operator*(const Jet3&a,const Jet3&b){Jet3 c(a.f*b.f);
 for(int i=0;i<3;++i){c.d[i]=a.d[i]*b.f+a.f*b.d[i];for(int j=0;j<3;++j){
  c.dd[i][j]=a.dd[i][j]*b.f+a.d[i]*b.d[j]+a.d[j]*b.d[i]+a.f*b.dd[i][j];
  for(int k=0;k<3;++k)c.ddd[i][j][k]=a.ddd[i][j][k]*b.f+a.dd[i][j]*b.d[k]+a.dd[i][k]*b.d[j]+a.dd[j][k]*b.d[i]
   +a.d[i]*b.dd[j][k]+a.d[j]*b.dd[i][k]+a.d[k]*b.dd[i][j]+a.f*b.ddd[i][j][k];}}return c;}
inline Jet3 Unary(const Jet3&a,double f,double d,double dd,double ddd){Jet3 c(f);
 for(int i=0;i<3;++i){c.d[i]=d*a.d[i];for(int j=0;j<3;++j){c.dd[i][j]=d*a.dd[i][j]+dd*a.d[i]*a.d[j];
  for(int k=0;k<3;++k)c.ddd[i][j][k]=d*a.ddd[i][j][k]+dd*(a.dd[i][j]*a.d[k]+a.dd[i][k]*a.d[j]+a.dd[j][k]*a.d[i])
   +ddd*a.d[i]*a.d[j]*a.d[k];}}return c;}
inline Jet3 Power(const Jet3&a,double p){return Unary(a,std::pow(a.f,p),p*std::pow(a.f,p-1),p*(p-1)*std::pow(a.f,p-2),p*(p-1)*(p-2)*std::pow(a.f,p-3));}
inline Jet3 Sin(const Jet3&a){return Unary(a,std::sin(a.f),std::cos(a.f),-std::sin(a.f),-std::cos(a.f));}
inline Jet3 Cos(const Jet3&a){return Unary(a,std::cos(a.f),-std::sin(a.f),-std::cos(a.f),std::sin(a.f));}
inline NativeJets Fields(const Real*x){
 Jet3 xx[3],r2{};for(int i=0;i<3;++i){xx[i]=Jet3::Variable(x[i],i);r2=r2+xx[i]*xx[i];}
 const auto angle=Jet3(.15)*xx[2]*Power(r2,-.5),c=Cos(angle),s=Sin(angle);
 const Jet3 y[3]={c*xx[0]+Jet3(-1)*s*xx[1],s*xx[0]+c*xx[1],xx[2]};
 h::MetricJet<Real> metric{};
 for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int b=0;b<3;++b){
  metric.g[i][j]+=y[b].d[i]*y[b].d[j];
  for(int a=0;a<3;++a){metric.dg[a][i][j]+=y[b].dd[a][i]*y[b].d[j]+y[b].d[i]*y[b].dd[a][j];
   for(int k=0;k<3;++k)metric.ddg[a][k][i][j]+=y[b].ddd[a][k][i]*y[b].d[j]+y[b].dd[a][i]*y[b].dd[k][j]
     +y[b].dd[k][i]*y[b].dd[a][j]+y[b].d[i]*y[b].ddd[a][k][j];}
 }
 const auto geometry=h::Geometry(metric);
 if(!geometry.valid||std::abs(geometry.determinant-1)>2e-13)throw std::runtime_error("invalid twist reference metric");
 NativeJets out;out.v[0]=1;
 for(int f=0;f<6;++f){const int i=Model::ti[f],j=Model::tj[f];out.v[1+f]=metric.g[i][j]-(i==j);
  for(int a=0;a<3;++a){out.d[1+f][a]=metric.dg[a][i][j];for(int b=0;b<3;++b)out.dd[1+f][a][b]=metric.ddg[a][b][i][j];}}
 for(int i=0;i<3;++i){out.v[15+i]=geometry.contracted[i];for(int a=0;a<3;++a)out.d[15+i][a]=geometry.dcontracted[a][i];}
 return out;
}
}
#endif
