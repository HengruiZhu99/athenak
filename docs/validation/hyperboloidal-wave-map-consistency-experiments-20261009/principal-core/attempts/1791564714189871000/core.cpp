#include "dual_helpers.hpp"
#include "reference_wave_map.hpp"
#include <algorithm>
namespace {
using Coeff=std::array<double,4>;
struct Field {J alpha,chi,trace,theta,beta[3],lambda[3],metric[3][3],a[3][3];};
J Linear(const J&x){J y(D(0,x.v.v));for(int i=0;i<3;++i){y.d[i]=D(0,x.d[i].v);for(int j=0;j<3;++j)y.dd[i][j]=D(0,x.dd[i][j].v);}return y;}
J Poly(const Coeff&c,int deriv,const J&rho){J y;for(int k=3;k>=deriv;--k){double q=c[k];for(int j=0;j<deriv;++j)q*=k-j;y=y*rho+J(D(q));}return y;}
Field Fields(const J&t,const J&tp,const J&tpp,const J&z,const J&zp,const J&zpp,const J&v,const J&w,const J&rho,const J xx[3]){
 Field f{};f.alpha=v;f.chi=J(D(-2))*z-J(D(4./3.))*rho*zp;f.trace=J(D(-6))*tp-J(D(4))*rho*tpp;
 for(int i=0;i<3;++i){f.beta[i]=xx[i]*(w-J(D(2))*tp);f.lambda[i]=J(D(8./3.))*xx[i]*(J(D(5))*zp+J(D(2))*rho*zpp);
  for(int j=0;j<3;++j){const J stf=xx[i]*xx[j]-(i==j?rho/J(D(3)):J{});f.metric[i][j]=J(D(4))*zp*stf;f.a[i][j]=J(D(-4))*tpp*stf;}}
 return f;
}
Jet LiftField(const hyp::LayerPoint<double>&p,const Field&f){auto u=Lift(p.state);Scalar(u.alpha,Linear(f.alpha));Scalar(u.chi,Linear(f.chi));Scalar(u.trace,Linear(f.trace));Scalar(u.theta,Linear(f.theta));
 for(int i=0;i<3;++i){Vector(u.beta,i,Linear(f.beta[i]));Vector(u.lambda,i,Linear(f.lambda[i]));for(int j=i;j<3;++j){SetMetric(u,i,j,Metric(u,i,j)+Linear(f.metric[i][j]));SetA(u,i,j,A(u,i,j)+Linear(f.a[i][j]));}}return u;
}
hyp::LayerPoint<D> Cast(const hyp::LayerPoint<double>&p){hyp::LayerPoint<D>q{};q.state=Lift(p.state);q.alpha=p.alpha;q.radius=p.radius;q.omega=p.omega;q.L=p.L;q.b=p.b;q.k_bar=p.k_bar;q.k_physical=p.k_physical;for(int i=0;i<3;++i){q.beta[i]=p.beta[i];q.dalpha[i]=p.dalpha[i];q.domega[i]=p.domega[i];for(int j=0;j<3;++j)q.omega_hessian[i][j]=p.omega_hessian[i][j];}return q;}
std::array<double,22> Raw(const hyp::Z4cRHS<D>&f,const hyp::GaugeRHS<D>&g){std::array<double,22>a{};a[0]=g.alpha.d;a[1]=f.chi.d;a[2]=f.trace.d;a[3]=f.theta.d;const int ti[6]={0,0,0,1,1,2},tj[6]={0,1,2,1,2,2};for(int i=0;i<3;++i){a[4+i]=g.beta[i].d;a[19+i]=f.lambda[i].d;}for(int j=0;j<6;++j){a[7+j]=f.metric[ti[j]][tj[j]].d;a[13+j]=f.a[ti[j]][tj[j]].d;}return a;}
std::array<double,22> Raw(const Field&f){std::array<double,22>a{};a[0]=f.alpha.v.v;a[1]=f.chi.v.v;a[2]=f.trace.v.v;a[3]=f.theta.v.v;const int ti[6]={0,0,0,1,1,2},tj[6]={0,1,2,1,2,2};for(int i=0;i<3;++i){a[4+i]=f.beta[i].v.v;a[19+i]=f.lambda[i].v.v;}for(int j=0;j<6;++j){a[7+j]=f.metric[ti[j]][tj[j]].v.v;a[13+j]=f.a[ti[j]][tj[j]].v.v;}return a;}
void Err(double&ab,double&sc,double a,double b){if(!std::isfinite(a)||!std::isfinite(b))throw std::runtime_error("nonfinite core comparator");const double e=std::abs(a-b);ab=std::max(ab,e);sc=std::max(sc,e/std::max({1.,std::abs(a),std::abs(b)}));}
}
int main(){double all_abs=0,all_scaled=0,constraints=0,input_normals=0,output_normals=0,accel_abs=0,accel_scaled=0,origin_w_limit=0;int count=0;
 const std::array<std::array<double,3>,3> directions={{{1,0,0},{.36,-.48,.8},{-.48,.64,.6}}};
 hyp::LayerReference<double>ref(1.,.5,{true,.05,.95});
 for(int witness=0;witness<17;++witness)for(double r:{0.,.025,.049,.05})for(const auto&n:directions){
  Coeff t{},z{},v{},w{};if(witness<16){auto*cs=witness/4==0?&t:witness/4==1?&z:witness/4==2?&v:&w;(*cs)[witness%4]=1;}
  else{t={1,1,-.2,0};z={1./3.,-1./7.,0,1./11.};v={-.25,1./9.,0,0};w={1./6.,0,-1./13.,0};}
  const double x[3]={r*n[0],r*n[1],r*n[2]};const auto p=ref.At(x[0],x[1],x[2]);J xx[3],rho;for(int i=0;i<3;++i){xx[i].v=x[i];xx[i].d[i]=1;rho=rho+xx[i]*xx[i];}
  const auto T=Poly(t,0,rho),Tp=Poly(t,1,rho),Tpp=Poly(t,2,rho),Z=Poly(z,0,rho),Zp=Poly(z,1,rho),Zpp=Poly(z,2,rho),V=Poly(v,0,rho),W=Poly(w,0,rho);
  const auto seed=Fields(T,Tp,Tpp,Z,Zp,Zpp,V,W,rho,xx);const auto u=LiftField(p,seed);const auto pd=Cast(p);const D xd[3]={x[0],x[1],x[2]};const auto c=rwm::ReferenceConnection(pd,xd);
  hyp::GaugeRHS<D>g{};hyp::Z4cRHS<D>f{};if(!rwm::Assemble(rwm::Gauge(pd,u,c),D(p.omega),g)||!hyp::AssembleInterior(hyp::ConformalRHS(u,Omega(u,p),D(10)/u.alpha.value,D(0)),D(p.omega),f))throw std::runtime_error("core actual RHS invalid");
  const J Vt=J(D(6))*Tp+J(D(4))*rho*Tpp,Wt=J(D(10))*Zp+J(D(4))*rho*Zpp;
  const auto expected=Fields(V,Poly(v,1,rho),Poly(v,2,rho),W,Poly(w,1,rho),Poly(w,2,rho),Vt,Wt,rho,xx);
  const auto actual=Raw(f,g),target=Raw(expected);for(int i=0;i<22;++i)Err(all_abs,all_scaled,actual[i],target[i]);
  const auto cc=hyp::EvolvedConstraints(u,Omega(u,p));if(!cc.valid)throw std::runtime_error("core constraints invalid");
  constraints=std::max({constraints,std::abs(cc.hamiltonian.d),std::abs(cc.z4.theta_physical.d)});for(int i=0;i<3;++i)constraints=std::max({constraints,std::abs(cc.momentum[i].d),std::abs(cc.z4.z_covector[i].d)});
  input_normals=std::max({input_normals,std::abs(cc.z4.determinant_residual.d),std::abs(cc.z4.tracefree_residual.d)});
  double trh=0,tra=0;for(int i=0;i<3;++i){trh+=f.metric[i][i].d;tra+=f.a[i][i].d;}output_normals=std::max({output_normals,std::abs(trh),std::abs(tra)});
  Err(accel_abs,accel_scaled,g.alpha.d,Vt.v.v);
  double recover_w=0;for(int i=0;i<3;++i){const double Xtt=g.beta[i].d+V.d[i].v;Err(accel_abs,accel_scaled,Xtt,x[i]*Wt.v.v);recover_w+=x[i]*Xtt;}
  if(r>0)Err(accel_abs,accel_scaled,recover_w/rho.v.v,Wt.v.v);
  else { // Exact scalar origin limit of the already-validated harmonic core rows.
   // d_i Xtt^i = d_i Lambda^i + .5 Delta chi. All consumed input jets complete.
   double limit=0;for(int i=0;i<3;++i)limit+=(u.lambda.d[i][i].d+.5*u.chi.dd[i][i].d)/3;
   origin_w_limit=std::max(origin_w_limit,std::abs(limit-Wt.v.v));
  }
  ++count;
 }
 std::cout<<std::setprecision(17)<<"{\"core_cases\":"<<count<<",\"witnesses\":17,\"all22_max_absolute\":"<<all_abs<<",\"all22_max_scaled\":"<<all_scaled<<",\"physical8_initial_constraint_max\":"<<constraints<<",\"input_algebraic_normal_max\":"<<input_normals<<",\"output_algebraic_normal_max\":"<<output_normals<<",\"coordinate_acceleration_max_absolute\":"<<accel_abs<<",\"coordinate_acceleration_max_scaled\":"<<accel_scaled<<",\"analytic_origin_scalar_w_limit_error\":"<<origin_w_limit<<"}\n";
}
