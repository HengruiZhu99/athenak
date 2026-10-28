#include <cmath>
#include "z4c/hyperboloidal/hyperbolic_shift.hpp"
#include "z4c/hyperboloidal/spherical_tensor.hpp"
namespace h=z4c::hyperboloidal;
namespace {
constexpr int N=26;
const int ii[5]={0,0,0,1,1},jj[5]={0,1,2,1,2};
void basis(int a,const h::Z4cJet<double> &base,double b[3][3]) {
  b[ii[a]][jj[a]]=b[jj[a]][ii[a]]=1;
  const auto g=h::Geometry(base.metric);
  double tr=0;
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)tr+=g.inverse[i][j]*b[i][j];
  b[2][2]=-tr/g.inverse[2][2];
}

extern "C" double hyperbolic_metric_max(const double *g) {
  double metric[3][3];
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)metric[i][j]=g[3*i+j];
  return h::LargestMetricEigenvalue(metric);
}
}

// Interior pseudodifferential reduction on the determinant/tracefree manifold.
// No spherical reduction. Every geometric entry comes from ConformalRHS.
extern "C" int hyperbolic_symbol(double alpha,double chi,const double *metric,
    const double *n,double F,double eta,double w,double kappa,double tau,
    double mu,double *P) {
  h::Z4cJet<double> base{};base.alpha.value=alpha;base.chi.value=chi;
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)base.metric.g[i][j]=metric[3*i+j];
  if(!h::Geometry(base.metric).valid || alpha<=0 || chi<=0 || tau<=0) return 1;
  h::OmegaJet<double> o{};o.omega=1;
  const double eps=1.e-4;
  for(int col=0;col<N;++col)for(int sign:{-1,1}) {
    const double e=sign*eps;
    auto u=base;
    // Rows with a differentiated configuration variable.
    if(col>=6 && col<11) {
      double b[3][3]{};basis(col-6,base,b);
      for(int i=0;i<3;++i)for(int j=0;j<3;++j)u.a.k[i][j]=e*b[i][j];
    }
    if(col==11)u.trace.value=e;
    if(col==12)u.theta.value=e;
    if(col>=17 && col<20)for(int j=0;j<3;++j)u.beta.d[j][col-17]=e*n[j];
    auto R=h::ConformalRHS(u,o,0.,0.);
    if(!R.valid)return 2;
    double upper[N]{};
    upper[0]=R.regular.chi+R.pole.chi;
    for(int a=0;a<5;++a)upper[1+a]=R.regular.metric[ii[a]][jj[a]];
    if(col==11)upper[16]=-F*e;
    if(col>=20 && col<23)upper[16]=eta*chi*w*n[col-20]*e;
    if(col>=23)upper[17+col-23]=mu*e;
    if(col==16)for(int i=0;i<3;++i)upper[20+i]=kappa*w/tau*n[i]*e;

    // Rows whose state variable has differential order zero.
    u=base;
    if(col==0 || col==16) for(int j=0;j<3;++j)for(int k=0;k<3;++k)
      (col==0 ? u.chi.dd[j][k] : u.alpha.dd[j][k])=e*n[j]*n[k];
    if(col>=1 && col<6) {
      double b[3][3]{};basis(col-1,base,b);
      for(int j=0;j<3;++j)for(int k=0;k<3;++k)
      for(int a=0;a<3;++a)for(int bidx=0;bidx<3;++bidx)
        u.metric.ddg[j][k][a][bidx]=e*n[j]*n[k]*b[a][bidx];
    }
    if(col==11 || col==12)for(int j=0;j<3;++j)
      (col==11 ? u.trace.d[j] : u.theta.d[j])=e*n[j];
    if(col>=13 && col<16)for(int j=0;j<3;++j)u.lambda.d[j][col-13]=e*n[j];
    if(col>=17 && col<20)for(int j=0;j<3;++j)for(int k=0;k<3;++k)
      u.beta.dd[j][k][col-17]=e*n[j]*n[k];
    R=h::ConformalRHS(u,o,0.,0.);if(!R.valid)return 3;
    for(int a=0;a<5;++a)upper[6+a]=R.regular.a[ii[a]][jj[a]]+R.pole.a[ii[a]][jj[a]];
    upper[11]=R.regular.trace+R.pole.trace;
    upper[12]=R.regular.theta+R.pole.theta;
    for(int i=0;i<3;++i)upper[13+i]=upper[23+i]=R.regular.lambda[i]+R.pole.lambda[i];
    for(int row=0;row<N;++row) {
      if(sign==-1)P[N*row+col]=0;
      P[N*row+col]+=sign*upper[row]/(2*eps);
    }
  }
  return 0;
}

extern "C" int hyperbolic_radial_gauge(double r,const double *q,const double *d,
    const double *dd,double B,double dB,double D,double *out) {
  const h::CMCReference<double> ref{1,1};
  auto u=h::SphericalJet(r,q,d,dd,ref);
  u.beta=h::RadialVector(-r+q[7],-1+d[7],dd[7],r);
  auto flux=h::RadialVector(B,dB,0.,r);
  double x[3]={r,0,0},driver[3]={D,0,0};
  const auto g=h::HyperbolicGauge(x,u,flux,driver);
  out[0]=g.lapse_regular;out[1]=g.lapse_pole;
  out[2]=g.shift_regular[0];out[3]=g.shift_pole[0];out[4]=g.flux_rate[0];
  const auto c=h::ShiftCoefficients(r,u.alpha.value,u.chi.value);
  out[5]=c.mu;out[6]=c.w;out[7]=c.dw_dr;
  h::OmegaJet<double> o{};o.omega=(1-r*r)/2;o.gradient[0]=-r;
  for(int i=0;i<3;++i)o.hessian[i][i]=-1;
  h::SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,o);
  auto parts=h::ConformalRHS(u,o,4.,-.99);const auto original=parts;
  if(!h::HyperbolicDamping(x,u,o,parts))return 1;
  out[8]=parts.pole.trace-original.pole.trace;
  out[9]=parts.pole.theta-original.pole.theta;
  out[10]=parts.pole.lambda[0]-original.pole.lambda[0];
  return g.valid ? 0 : 1;
}

extern "C" int hyperbolic_scri(double omega,double alpha,const double *g,
    const double *n,const double *trial,double at,const double *gt,
    const double *a,double mu,double *D,double *result) {
  double metric[3][3],rate[3][3];
  for(int i=0;i<3;++i)for(int j=0;j<3;++j) {
    metric[i][j]=g[3*i+j];rate[i][j]=gt[3*i+j];
  }
  auto b=h::ReconstructScriShift(omega,alpha,metric,n,trial);
  if(!h::ReconstructScriAuxiliary(b,at,rate,a,mu,D))return 1;
  for(int i=0;i<3;++i)result[i]=b.beta[i];
  result[3]=b.normal_rate;return 0;
}
