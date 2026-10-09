#ifndef RWM_WIDE_ARITHMETIC_HPP_
#define RWM_WIDE_ARITHMETIC_HPP_
#include <cmath>
// Independent embedding oracle only. FMA double-double, about106 significand
// bits for ordinary nonoverflowing inputs. Not interval/certified arithmetic.
namespace wide {
struct DD {
 double hi=0,lo=0;
 DD()=default;DD(double v):hi(v){}DD(double h,double l):hi(h),lo(l){}
 explicit operator double()const{return hi+lo;}
 DD&operator+=(DD);DD&operator-=(DD);DD&operator*=(DD);DD&operator/=(DD);
};
inline DD Sum(double a,double b){const double s=a+b,bb=s-a;return {s,(a-(s-bb))+(b-bb)};}
inline DD Normalize(double a,double b){return Sum(a,b);}
inline DD operator+(DD a,DD b){const DD s=Sum(a.hi,b.hi);return Normalize(s.hi,s.lo+a.lo+b.lo);}
inline DD operator-(DD a){return {-a.hi,-a.lo};}
inline DD operator-(DD a,DD b){return a+(-b);}
inline DD operator*(DD a,DD b){const double h=a.hi*b.hi;const double l=std::fma(a.hi,b.hi,-h)+a.hi*b.lo+a.lo*b.hi+a.lo*b.lo;return Normalize(h,l);}
inline DD operator/(DD a,DD b){const double q0=a.hi/b.hi;const DD r=a-b*DD(q0);const double q1=r.hi/b.hi;const DD r2=r-b*DD(q1);return DD(q0)+DD(q1)+DD(r2.hi/b.hi);}
inline DD&DD::operator+=(DD a){return *this=*this+a;}inline DD&DD::operator-=(DD a){return *this=*this-a;}inline DD&DD::operator*=(DD a){return *this=*this*a;}inline DD&DD::operator/=(DD a){return *this=*this/a;}
inline bool operator<(DD a,DD b){return a.hi<b.hi||(a.hi==b.hi&&a.lo<b.lo);}inline bool operator>(DD a,DD b){return b<a;}inline bool operator<=(DD a,DD b){return !(b<a);}inline bool operator==(DD a,DD b){return a.hi==b.hi&&a.lo==b.lo;}
inline DD Sqrt(DD x){DD y(std::sqrt(x.hi));y+=(x-y*y)/(DD(2)*y);y+=(x-y*y)/(DD(2)*y);return y;}
}
#endif
