// Truncated analytic r/time series for oracle evaluation only. Never evolved.
#pragma once
#include <array>
#include <cmath>
#include <stdexcept>
namespace cubed_sphere {namespace clock_oracle {
struct Series {
 static constexpr int N=3;double a[4][4]{};
 Series()=default;Series(double x){a[0][0]=x;}
 double value()const{return a[0][0];}
 static Series R(double x){Series z(x);z.a[1][0]=1;return z;}
 static Series Time(double x){Series z(x);z.a[0][1]=1;return z;}
};
inline Series operator+(const Series&x,const Series&y){Series z;for(int i=0;i<=3;++i)for(int j=0;j<=3-i;++j)z.a[i][j]=x.a[i][j]+y.a[i][j];return z;}
inline Series operator-(const Series&x,const Series&y){Series z;for(int i=0;i<=3;++i)for(int j=0;j<=3-i;++j)z.a[i][j]=x.a[i][j]-y.a[i][j];return z;}
inline Series operator-(const Series&x){return Series(0)-x;}
inline Series operator*(const Series&x,const Series&y){Series z;for(int i=0;i<=3;++i)for(int j=0;j<=3-i;++j)for(int k=0;k<=i;++k)for(int l=0;l<=j;++l)z.a[i][j]+=x.a[k][l]*y.a[i-k][j-l];return z;}
inline Series Scale(const Series&x,double s){Series z;for(int i=0;i<=3;++i)for(int j=0;j<=3-i;++j)z.a[i][j]=x.a[i][j]*s;return z;}
inline Series Power(const Series&x,double p){if(!(x.value()>0))throw std::runtime_error("oracle positive analytic power domain");Series d=Scale(x-Series(x.value()),1/x.value()),term(1),z(1);double c=1;for(int k=1;k<=3;++k){term=term*d;c*= (p-k+1)/k;z=z+Scale(term,c);}return Scale(z,std::pow(x.value(),p));}
inline Series Inverse(const Series&x){if(x.value()==0)throw std::runtime_error("oracle zero denominator");Series d=Scale(x-Series(x.value()),1/x.value()),term(1),z(1);for(int k=1;k<=3;++k){term=term*d;z=z+Scale(term,k%2?-1:1);}return Scale(z,1/x.value());}
inline Series operator/(const Series&x,const Series&y){return x*Inverse(y);}
inline Series Exp(const Series&x){Series d=x-Series(x.value()),term(1),z(1);double c=1;for(int k=1;k<=3;++k){term=term*d;c/=k;z=z+Scale(term,c);}return Scale(z,std::exp(x.value()));}
inline Series Dr(const Series&x){Series z;for(int i=0;i<3;++i)for(int j=0;j<3-i;++j)z.a[i][j]=(i+1)*x.a[i+1][j];return z;}
inline Series DivideRAtOrigin(const Series&x){Series z;for(int i=0;i<3;++i)for(int j=0;j<3-i;++j)z.a[i][j]=x.a[i+1][j];return z;}
}}
