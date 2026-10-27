// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef TST_TRANSITION_BACKGROUND_PROJECTION_HPP_
#define TST_TRANSITION_BACKGROUND_PROJECTION_HPP_
#include "transition_background_jet.hpp"
namespace z4c { namespace hyperboloidal {
// Known background column only is projected; q and physical values never are.
// All P5 modes remain. This is a basis transformation, not a change of span/PDE.
struct TransitionBackgroundProjection {
 using Point=std::array<Real,3>;
 std::string status="unbuilt";int rank=0;Real condition=0,residual_norm=0,relative_residual=0;
 std::vector<Real> residual;ScalarJet<Real> query{};
 bool Build(const Point&at,const std::vector<Point>&points,const std::vector<Real>&background,const ScalarJet<Real>&exact,Real h){
  rank=0;condition=0;residual_norm=0;relative_residual=0;residual.clear();query={};
  constexpr int p=56;const int n=points.size();auto fail=[&](const char*s){status=s;return false;};if(n<p||n>256||background.size()!=points.size()||!std::isfinite(exact.value)||!(h>0&&std::isfinite(h)))return fail("invalid_geometry");
  std::array<std::array<int,3>,p>powers{};int next=0;for(int degree=0;degree<=5;++degree)for(int a=0;a<=degree;++a)for(int b=0;b<=degree-a;++b)powers[next++]={a,b,degree-a-b};
  std::vector<Real>phi(n*p),sw(n),norm(p),v(n*p),Q(n*p),R(p*p);std::array<int,p>perm{};
  for(int i=0;i<n;++i){if(!std::isfinite(background[i]))return fail("invalid_background");Point t{};Real d2=0;for(int a=0;a<3;++a){t[a]=(points[i][a]-at[a])/h;d2+=t[a]*t[a];}sw[i]=std::exp(-d2/16);for(int j=0;j<p;++j){Real f=1;for(int a=0;a<3;++a)f*=std::pow(t[a],powers[j][a]);phi[i*p+j]=f;v[j*n+i]=sw[i]*f;norm[j]+=v[j*n+i]*v[j*n+i];}}
  for(int j=0;j<p;++j){norm[j]=std::sqrt(norm[j]);if(!(norm[j]>0))return fail("P5_rank_deficient");perm[j]=j;for(int i=0;i<n;++i)v[j*n+i]/=norm[j];}
  Real largest=0;for(int k=0;k<p;++k){int pivot=k;Real best=-1;for(int j=k;j<p;++j){Real a=0;for(int i=0;i<n;++i)a+=v[j*n+i]*v[j*n+i];if(a>best){best=a;pivot=j;}}if(pivot!=k){for(int i=0;i<n;++i)std::swap(v[k*n+i],v[pivot*n+i]);std::swap(perm[k],perm[pivot]);for(int j=0;j<k;++j)std::swap(R[j*p+k],R[j*p+pivot]);}R[k*p+k]=std::sqrt(best);largest=std::max(largest,R[k*p+k]);if(!(R[k*p+k]>1e-11*largest))return fail("P5_rank_deficient");++rank;for(int i=0;i<n;++i)Q[k*n+i]=v[k*n+i]/R[k*p+k];for(int j=k+1;j<p;++j)for(int pass=0;pass<2;++pass){Real dot=0;for(int i=0;i<n;++i)dot+=Q[k*n+i]*v[j*n+i];R[k*p+j]+=dot;for(int i=0;i<n;++i)v[j*n+i]-=dot*Q[k*n+i];}}
  Real rnorm=0;std::array<Real,p>inverse_rows{};for(int i=0;i<p;++i){Real sum=0;for(int j=i;j<p;++j)sum+=std::abs(R[i*p+j]);rnorm=std::max(rnorm,sum);}for(int col=0;col<p;++col){std::array<Real,p>x{};for(int i=p-1;i>=0;--i){Real sum=i==col?1:0;for(int j=i+1;j<p;++j)sum-=R[i*p+j]*x[j];x[i]=sum/R[i*p+i];}for(int i=0;i<p;++i)inverse_rows[i]+=std::abs(x[i]);}condition=rnorm*(*std::max_element(inverse_rows.begin(),inverse_rows.end()));if(!(condition<=1e8))return fail("P5_condition");
  std::array<Real,p>rhs{},solution{},coeff{};Real bnorm=0;for(int k=0;k<p;++k)for(int i=0;i<n;++i)rhs[k]+=Q[k*n+i]*sw[i]*(background[i]-exact.value);for(int k=p-1;k>=0;--k){Real sum=rhs[k];for(int j=k+1;j<p;++j)sum-=R[k*p+j]*solution[j];solution[k]=sum/R[k*p+k];coeff[perm[k]]=solution[k]/norm[perm[k]];}
  residual.resize(n);for(int i=0;i<n;++i){Real fit=0;for(int j=0;j<p;++j)fit+=coeff[j]*phi[i*p+j];residual[i]=(background[i]-exact.value)-fit;residual_norm+=sw[i]*sw[i]*residual[i]*residual[i];bnorm+=sw[i]*sw[i]*std::pow(background[i]-exact.value,2);}
  residual_norm=std::sqrt(residual_norm);relative_residual=residual_norm/std::sqrt(bnorm);if(!(residual_norm>0&&std::isfinite(residual_norm)&&relative_residual>1e-11))return fail("background_residual_rank_deficient");query=exact;query.value=-coeff[0];for(int j=0;j<p;++j){const auto&power=powers[j];const int degree=power[0]+power[1]+power[2];if(degree==1)for(int a=0;a<3;++a)if(power[a]==1)query.d[a]-=coeff[j]/h;if(degree==2)for(int a=0;a<3;++a)for(int b=a;b<3;++b)if((a==b&&power[a]==2)||(a!=b&&power[a]==1&&power[b]==1)){const Real c=(a==b?2:1)*coeff[j]/(h*h);query.dd[a][b]-=c;if(a!=b)query.dd[b][a]-=c;}}
  query.value/=residual_norm;for(int a=0;a<3;++a){query.d[a]/=residual_norm;for(int b=0;b<3;++b)query.dd[a][b]/=residual_norm;}for(auto&v:residual)v/=residual_norm;status="ready";return true;
 }
};
}}
#endif
