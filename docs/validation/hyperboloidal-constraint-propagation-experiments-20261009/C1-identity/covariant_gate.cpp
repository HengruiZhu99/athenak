// Scratch exact full-tensor C_Z4c additions and physical-ADM comparison.
#include "../discrete-bianchi/immutable-discrete-bulk-20261009/dual_helpers.hpp"

#include "c1_additions.hpp"
hyp::Z4cRHS<D> Delta(const Jet&u,const hyp::OmegaJet<D>&o,bool connection) {
  hyp::Z4cRHS<D> f;
  if(!hyp::AssembleC1Interior(hyp::TensorC1Additions(u,o,D(1),connection),o.omega,f))
    throw std::runtime_error("interior C1 assembly invalid");
  return f;
}
hyp::Z4cRHS<D> Sum(hyp::Z4cRHS<D> a,const hyp::Z4cRHS<D>&b) {
  a.chi+=b.chi;a.trace+=b.trace;a.theta+=b.theta;
  for(int i=0;i<3;++i){a.lambda[i]+=b.lambda[i];for(int j=0;j<3;++j){a.metric[i][j]+=b.metric[i][j];a.a[i][j]+=b.a[i][j];}}
  return a;
}
J ScalarJ(const hyp::ScalarJet<D>&a) {
  J x(a.value);for(int i=0;i<3;++i){x.d[i]=a.d[i];for(int j=0;j<3;++j)x.dd[i][j]=a.dd[i][j];}return x;
}
J OmegaJ(const hyp::OmegaJet<D>&o) {
  J x(o.omega);for(int i=0;i<3;++i){x.d[i]=o.gradient[i];for(int j=0;j<3;++j)x.dd[i][j]=o.hessian[i][j];}return x;
}
double Abs(D x){return std::abs(x.v);}
void Max(double&out,D error,D scale=D(1)){out=std::max(out,Abs(error)/std::max(1.,Abs(scale)));}

struct Check {double metric=0,k0=0,k1=0,theta=0,z_extra=0,z_cov=0,
                    metric_abs=0,k0_abs=0,k1_abs=0,z_non_covariant=0,
                    extra_shift=0,einstein=0,zero_switch=0;};
double Norm(const hyp::Z4cRHS<D>&f) {
  double n=std::max({Abs(f.chi),Abs(f.trace),Abs(f.theta)});
  for(int i=0;i<3;++i){n=std::max(n,Abs(f.lambda[i]));for(int j=0;j<3;++j)n=std::max({n,Abs(f.metric[i][j]),Abs(f.a[i][j])});}
  return n;
}
Check Audit(const Jet&u,const hyp::OmegaJet<D>&o,double kappa,double kappa2) {
  Check result;hyp::Z4cRHS<D> f0;
  if(!hyp::AssembleInterior(hyp::ConformalRHS(u,o,D(kappa)/u.alpha.value,D(kappa2)),o.omega,f0))throw std::runtime_error("C0 kernel invalid");
  const auto f1=Sum(f0,Delta(u,o,false)),fcov=Sum(f0,Delta(u,o,true));
  Jet on=u;on.theta.value=0;const auto ong=hyp::Geometry(on.metric);
  for(int i=0;i<3;++i)on.lambda.value[i]=ong.contracted[i];
  result.einstein=std::max(Norm(Delta(on,o,false)),Norm(Delta(on,o,true)));
  hyp::Z4cRHS<D> zero;
  if(!hyp::AssembleC1Interior(hyp::TensorC1Additions(u,o,D(0),true),o.omega,zero))throw std::runtime_error("zero-switch invalid");
  result.zero_switch=Norm(zero);
  const auto gt=hyp::Geometry(u.metric);const auto c=hyp::EvolvedConstraints(u,o);
  const J O=OmegaJ(o),chi=ScalarJ(u.chi),alpha=ScalarJ(u.alpha);
  const J al=alpha/O,K=ScalarJ(u.trace)+J(D(2))*ScalarJ(u.theta);
  hyp::MetricJet<D> pm{};hyp::CurvatureJet<D> pk{};J gam[3][3],kj[3][3];
  for(int i=0;i<3;++i)for(int j=0;j<3;++j) {
    gam[i][j]=Metric(u,i,j)/chi/O/O;
    kj[i][j]=A(u,i,j)/chi/O+gam[i][j]*K/J(D(3));
    pm.g[i][j]=gam[i][j].v;pk.k[i][j]=kj[i][j].v;
    for(int d=0;d<3;++d){pm.dg[d][i][j]=gam[i][j].d[d];pk.dk[d][i][j]=kj[i][j].d[d];for(int e=0;e<3;++e)pm.ddg[d][e][i][j]=gam[i][j].dd[d][e];}
  }
  const auto pg=hyp::Geometry(pm);D z[3]{},zt[3]{},zp[3]{},dz[3][3]{},dpc[3][3]{};
  for(int i=0;i<3;++i) {
    z[i]=c.z4.z_covector[i];zt[i]=(u.lambda.value[i]-gt.contracted[i])/2;
    for(int j=0;j<3;++j) {
      zp[i]+=pg.inverse[i][j]*c.z4.z_covector[j];
      for(int d=0;d<3;++d)dz[d][i]+=(u.metric.dg[d][i][j]*(u.lambda.value[j]-gt.contracted[j])
                          +u.metric.g[i][j]*(u.lambda.d[d][j]-gt.dcontracted[d][j]))/2;
    }
  }
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){dpc[i][j]=dz[i][j];for(int k=0;k<3;++k)dpc[i][j]-=pg.connection[k][i][j]*z[k];}
  D B=u.alpha.value*o.omega*D(0);
  for(int i=0;i<3;++i)B+=u.alpha.value*o.omega*zt[i]*u.chi.d[i]+2*u.alpha.value*u.chi.value*zt[i]*o.gradient[i];
  B-=(6*u.alpha.value*o.normal+3*(1+kappa2)*D(kappa))*u.theta.value/o.omega;
  for(int i=0;i<3;++i)for(int j=0;j<3;++j) {
    D lie{},hess=al.dd[i][j],kk{};
    for(int a=0;a<3;++a){hess-=pg.connection[a][i][j]*al.d[a];lie+=u.beta.value[a]*pk.dk[a][i][j]+pk.k[a][j]*u.beta.d[i][a]+pk.k[i][a]*u.beta.d[j][a];
      for(int b=0;b<3;++b)kk+=pk.k[i][a]*pg.inverse[a][b]*pk.k[b][j];}
    const D adm=-hess+al.v*(pg.ricci[i][j]+K.v*pk.k[i][j]-2*kk)+lie;
    const D S0=al.v*(dpc[i][j]+dpc[j][i])+pm.g[i][j]*B/3;
    const D S1=al.v*(dpc[i][j]+dpc[j][i]-2*u.theta.value*pk.k[i][j])
               -(1+kappa2)*D(kappa)*u.theta.value*pm.g[i][j]/o.omega;
    const D gd=(f0.metric[i][j]-u.metric.g[i][j]*f0.chi/u.chi.value)/(o.omega*o.omega*u.chi.value);
    D lieg{};for(int a=0;a<3;++a)lieg+=u.beta.value[a]*pm.dg[a][i][j]+pm.g[a][j]*u.beta.d[i][a]+pm.g[i][a]*u.beta.d[j][a];
    const D metric_residual=gd-(-2*al.v*pk.k[i][j]+lieg);
    Max(result.metric,metric_residual,D(1+Abs(gd)+Abs(2*al.v*pk.k[i][j])+Abs(lieg)));
    Max(result.metric_abs,metric_residual);
    const auto kd=[&](const hyp::Z4cRHS<D>&f){return (f.a[i][j]-u.a.k[i][j]*f.chi/u.chi.value)/(o.omega*u.chi.value)
                  +gd*K.v/3+pm.g[i][j]*(f.trace+2*f.theta)/3;};
    const D common_scale=D(1+Abs(hess)+Abs(al.v*pg.ricci[i][j])
      +Abs(al.v*K.v*pk.k[i][j])+Abs(2*al.v*kk)+Abs(lie));
    Max(result.k0,kd(f0)-adm-S0,common_scale+D(Abs(kd(f0))+Abs(S0)));
    Max(result.k1,kd(f1)-adm-S1,common_scale+D(Abs(kd(f1))+Abs(S1)));
    Max(result.k0_abs,kd(f0)-adm-S0);Max(result.k1_abs,kd(f1)-adm-S1);
  }
  D divz{},zda{},advtheta{};
  for(int i=0;i<3;++i){zda+=zp[i]*al.d[i];advtheta+=u.beta.value[i]*u.theta.d[i];for(int j=0;j<3;++j)divz+=pg.inverse[i][j]*dpc[i][j];}
  const D th=advtheta+al.v*(c.hamiltonian/2+divz-K.v*u.theta.value)
              -zda-D(kappa)*(2+kappa2)*u.theta.value/o.omega;
  Max(result.theta,f1.theta-th,f1.theta);
  // Differentiate the actual metric equation, not a presumed Gamma identity.
  Jet time=u;D divbeta{};for(int i=0;i<3;++i)divbeta+=u.beta.d[i][i];
  for(int i=0;i<3;++i)for(int j=0;j<3;++j) {
    time.metric.g[i][j].d=f0.metric[i][j].v;
    for(int d=0;d<3;++d) {
      D df=-2*(u.alpha.d[d]*u.a.k[i][j]+u.alpha.value*u.a.dk[d][i][j])
            -D(2./3.)*u.metric.dg[d][i][j]*divbeta;
      for(int k=0;k<3;++k) {
        df-=D(2./3.)*u.metric.g[i][j]*u.beta.dd[d][k][k];
        df+=u.beta.d[d][k]*u.metric.dg[k][i][j]+u.beta.value[k]*u.metric.ddg[d][k][i][j]
           +u.metric.dg[d][k][j]*u.beta.d[i][k]+u.metric.g[k][j]*u.beta.dd[d][i][k]
           +u.metric.dg[d][i][k]*u.beta.d[j][k]+u.metric.g[i][k]*u.beta.dd[d][j][k];
      }
      time.metric.dg[d][i][j].d=df.v;
    }
  }
  const auto timed=hyp::Geometry(time.metric);
  for(int i=0;i<3;++i) {
    D rate{},ratecov{},wanted=al.v*(c.momentum[i]+u.theta.d[i]-D(kappa)*z[i]/u.alpha.value)-u.theta.value*al.d[i],extra{};
    for(int j=0;j<3;++j) {
      rate+=(f1.metric[i][j]*(u.lambda.value[j]-gt.contracted[j])
            +u.metric.g[i][j]*(f1.lambda[j]-D(timed.contracted[j].d)))/2;
      ratecov+=(fcov.metric[i][j]*(u.lambda.value[j]-gt.contracted[j])
            +u.metric.g[i][j]*(fcov.lambda[j]-D(timed.contracted[j].d)))/2;
      wanted+=u.beta.value[j]*dz[j][i]+z[j]*u.beta.d[i][j];
      for(int k=0;k<3;++k){wanted-=2*al.v*pk.k[i][j]*pg.inverse[j][k]*z[k];extra+=u.metric.g[i][j]*zt[k]*u.beta.d[k][j];}
    }
    Max(result.z_extra,rate-wanted-extra,rate);
    Max(result.z_cov,ratecov-wanted,ratecov);
    Max(result.z_non_covariant,rate-wanted);Max(result.extra_shift,extra);
  }
  return result;
}

int main(int argc,char**) {
  if(argc>1) {
    Jet u{};u.alpha.value=u.chi.value=1;hyp::OmegaJet<D> o{};o.omega=1;
    for(int i=0;i<3;++i){u.metric.g[i][i]=1;u.lambda.value[i]=2*(i+1);}
    u.beta.d[0][0]=1;const auto c=Audit(u,o,5.,0.);
    std::cout<<std::setprecision(17)<<"{\"AppendixB_C1_addition_norm\":"<<Norm(Delta(u,o,false))
      <<",\"noncovariant_Z_rate\":"<<c.z_non_covariant
      <<",\"extra_shift_rate\":"<<c.extra_shift
      <<",\"repaired_Z_error\":"<<c.z_cov<<"}\n";return 0;
  }
  std::cout<<std::setprecision(17)<<'[';bool first=true;
  for(double a:{.5,.75,1.,2.}) {
    const hyp::LayerReference<double> ref(1.,a,{true,.05,.95});
    for(double r:{0.,.3,.5,.75,.85,.95,.98,.99995})for(double perturb:{0.,.01,.1})for(double kappa:{5.,10.})for(double kappa2:{0.,.3}) {
      const auto p=ref.At(r,0.,0.);Jet u=Lift(p.state);
      for(int col=0;col<20;++col) {
        J seed(D(perturb*std::sin(col+1.)));
        for(int d=0;d<3;++d){seed.d[d]=perturb*std::cos((col+1.)*(d+1.))/3;
          for(int e=0;e<3;++e)seed.dd[d][e]=perturb*std::sin((col+1.)*(d+e+2.))/5;}
        Seed(u,col,seed);
      }
      Consistent(u);const auto o=Omega(u,p);const auto c=Audit(u,o,kappa,kappa2);
      if(!first)std::cout<<',';first=false;
      std::cout<<"{\"a\":"<<a<<",\"r\":"<<r<<",\"Omega\":"<<p.omega
        <<",\"perturb\":"<<perturb<<",\"kappa\":"<<kappa<<",\"kappa2\":"<<kappa2
        <<",\"metric_error\":"<<c.metric<<",\"C0_ADM_error\":"<<c.k0
        <<",\"C1_ADM_error\":"<<c.k1<<",\"C1_Theta_error\":"<<c.theta
        <<",\"C1_Z_extra_shift_error\":"<<c.z_extra
        <<",\"covariant_connection_Z_error\":"<<c.z_cov
        <<",\"metric_absolute_residual\":"<<c.metric_abs
        <<",\"C0_ADM_absolute_residual\":"<<c.k0_abs
        <<",\"C1_ADM_absolute_residual\":"<<c.k1_abs
        <<",\"C1_Z_noncovariant_rate\":"<<c.z_non_covariant
        <<",\"extra_shift_rate\":"<<c.extra_shift
        <<",\"Einstein_sector_addition\":"<<c.einstein
        <<",\"C0_switch_addition\":"<<c.zero_switch<<'}';
    }
  }
  std::cout<<"]\n";
}
