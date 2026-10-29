#ifndef HYPERBOLIC_SHIFT_COUPLED_DIAGNOSTICS_HPP_
#define HYPERBOLIC_SHIFT_COUPLED_DIAGNOSTICS_HPP_
#include "z4c/hyperboloidal/hyperbolic_composite.hpp"
namespace coupled_diagnostics {
namespace h=z4c::hyperboloidal;
// Expansion of coordinate spheres in the full nonspherical geometry. A zero
// on one ray is NOT a general apparent-horizon solve. Angular spread reports
// when a spherical comparison has ceased to be meaningful.
inline std::array<Real,2> Sphere(const Real*x,const h::Z4cJet<Real>&u,const h::OmegaJet<Real>&o){
  const Real r=h::HyperbolicComposite::Radius(x);const auto g=h::Geometry(u.metric);
  Real n[3],v[3]{},dv[3][3]{},dq[3]{},Q=0;
  for(int i=0;i<3;++i)n[i]=x[i]/r;
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)v[i]+=g.inverse[i][j]*n[j];
  for(int i=0;i<3;++i)Q+=n[i]*v[i];
  for(int a=0;a<3;++a)for(int i=0;i<3;++i){
    for(int j=0;j<3;++j){dv[a][i]+=g.inverse[i][j]*((a==j)-n[a]*n[j])/r;
      for(int b=0;b<3;++b)dv[a][i]-=g.inverse[i][j]*u.metric.dg[a][j][b]*v[b];}
    dq[a]+=n[i]*dv[a][i]+((a==i)-n[a]*n[i])*v[i]/r;
  }
  Real divergence=0,ann=0;
  for(int i=0;i<3;++i){Real connection=0;for(int a=0;a<3;++a)connection+=g.connection[a][a][i];
    divergence+=o.omega*(dv[i][i]+v[i]*(connection-dq[i]/(2*Q)-u.chi.d[i]/u.chi.value))-2*v[i]*o.gradient[i];
    for(int j=0;j<3;++j)ann+=u.a.k[i][j]*v[i]*v[j];}
  const Real expansion=std::sqrt(u.chi.value/Q)*divergence+o.omega*ann/Q-2*(u.trace.value+2*u.theta.value)/3;
  const Real area_radius=r*std::pow(g.determinant*Q,.25)/(o.omega*std::sqrt(u.chi.value));
  return {expansion,area_radius};
}
inline void Header(std::ostream&out){out<<"x,y,z,r,chi,g_rr,alpha,beta_r,A_rr,P,Theta,Lambda_r,B_r,D_r,J_r,H,M_r,M_norm,Z_r,Z_norm,det_error,tracefree_error,expansion,area_radius\n";}
inline void Row(std::ostream&out,const Real*x,const h::Z4cJet<Real>&u,const h::OmegaJet<Real>&o,const std::vector<Real>&values){
  const Real r=h::HyperbolicComposite::Radius(x);const auto c=h::EvolvedConstraints(u,o);const auto sphere=Sphere(x,u,o);
  Real grr=0,arr=0,beta=0,lambda=0,B=0,D=0,J=0,M=0,Z=0;
  for(int i=0;i<3;++i){const Real n=x[i]/r;
    beta+=n*u.beta.value[i];lambda+=n*u.lambda.value[i];B+=n*values[19+i];D+=n*values[25+i];J+=n*values[28+i];
    M+=n*c.momentum[i];Z+=n*c.z4.z_covector[i];
    for(int j=0;j<3;++j){grr+=n*u.metric.g[i][j]*x[j]/r;arr+=n*u.a.k[i][j]*x[j]/r;}}
  out<<x[0]<<','<<x[1]<<','<<x[2]<<','<<r<<','<<u.chi.value<<','<<grr<<','<<u.alpha.value<<','<<beta<<','<<arr<<','<<u.trace.value<<','<<u.theta.value<<','<<lambda<<','<<B<<','<<D<<','<<J<<','<<c.hamiltonian<<','<<M<<','<<std::sqrt(std::max(Real(0),c.momentum_conformal_norm2))<<','<<Z<<','<<std::sqrt(std::max(Real(0),c.z4.z_conformal_norm2))<<','<<c.z4.determinant_residual<<','<<c.z4.tracefree_residual<<','<<sphere[0]<<','<<sphere[1]<<'\n';
}
inline void Write(h::HyperbolicComposite&e,const std::string&prefix){
  const auto padded=e.Padded(e.state);std::ofstream out(prefix+".diagnostics.csv");out<<std::setprecision(17);Header(out);
  std::vector<Real>v(e.nf);std::vector<std::array<Real,3>>d(e.nf);
  std::vector<std::array<std::array<Real,3>,3>>dd(e.nf);
  for(size_t p=0;p<e.grid.active.size();++p){const auto*x=e.grid.geometry[p].x;
    for(int f=0;f<e.nf;++f){v[f]=e.state[p*e.nf+f];Real df[3],ddf[3][3];e.BulkDerivatives(padded,f,p,df,ddf);
      for(int a=0;a<3;++a){d[f][a]=df[a];for(int b=0;b<3;++b)dd[f][a][b]=ddf[a][b];}}
    const auto u=e.Jet(x,v,d,dd);Row(out,x,u,e.Omega(x,u),v);
  }
  std::ofstream rays(prefix+".rays.csv");rays<<std::setprecision(17);Header(rays);
  const Real directions[14][3]={{1,0,0},{-1,0,0},{0,1,0},{0,-1,0},{0,0,1},{0,0,-1},
    {1,1,1},{1,1,-1},{1,-1,1},{1,-1,-1},{-1,1,1},{-1,1,-1},{-1,-1,1},{-1,-1,-1}};
  for(auto&direction:directions)for(int i=1;i<200;++i){Real x[3];
    const Real r=i/200.,length=e.Radius(direction);for(int a=0;a<3;++a)x[a]=r*direction[a]/length;
    e.grid.SampleJets(padded,e.nf,x,v,d,dd,true);
    try{const auto u=e.Jet(x,v,d,dd);Row(rays,x,u,e.Omega(x,u),v);}
    catch(const std::exception&){for(int a=0;a<3;++a)rays<<x[a]<<',';rays<<r;
      for(int f=4;f<24;++f)rays<<",nan";rays<<'\n';}
  }
}
}
#endif
