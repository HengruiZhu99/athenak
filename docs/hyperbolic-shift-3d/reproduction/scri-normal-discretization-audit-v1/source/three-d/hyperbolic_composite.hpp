// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#ifndef Z4C_HYPERBOLOIDAL_HYPERBOLIC_COMPOSITE_HPP_
#define Z4C_HYPERBOLOIDAL_HYPERBOLIC_COMPOSITE_HPP_
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <thread>
#include <exception>
#include "mesh/cubed_sphere/scri_surface.hpp"
#include "mesh/cubed_sphere/angular_dissipation.hpp"
#include "z4c/hyperboloidal/hyperbolic_shift.hpp"
#include "z4c/hyperboloidal/tensor_scri.hpp"

namespace z4c::hyperboloidal {
// CPU coupled evolution on actual Cartesian/core and mapped shell cells. No
// angular averaging or radial reconstruction enters the evolution operator.
// Layout: chi, metric deviation[6], A[6], P+3, Theta, Lambda[3], alpha-A,
// telegraph B[3], beta+x[3], D[3], adaptive J[3].
class HyperbolicComposite {
 public:
  using V=std::vector<Real>;
  static constexpr int nf=31;
  inline static constexpr int ti[6]={0,0,0,1,1,2},tj[6]={0,1,2,1,2,2};
  cubed_sphere::CompositeDifference grid;
  cubed_sphere::ScriSurface scri;
  V state,compensation,edge_compensation,failed_state,failed_boundary;
  int failed_stage=0;
  bool adaptive;
  Real ko=.5;
  Real angular_epsilon=0,angular_cfl=0,angular_cfl_max=0;
  std::shared_ptr<const cubed_sphere::AngularDissipationPlan> angular_plan;
  int workers=1;
  Real time=0,null_max=0,compatibility_max=0,pole_max=0,clock_max=0;
  mutable Real interpolated_null_max=0,interpolated_null_before_max=0;
  Real gap_min=std::numeric_limits<Real>::infinity(),cfl_max=0;
  size_t steps=0;
  HyperbolicComposite(cubed_sphere::CompositeMesh mesh,bool source=false,bool cross_face_scri=false,
                      Real angular_strength=0)
      :grid(std::move(mesh),4,true),scri(grid,nf,6,cross_face_scri),
       state(nf*grid.active.size(),0),compensation(state.size(),0),
       edge_compensation(scri.values.size(),0),adaptive(source),angular_epsilon(angular_strength) {
    if(!std::isfinite(angular_epsilon)||angular_epsilon<0)
      throw std::invalid_argument("invalid angular dissipation strength");
    if(angular_epsilon>0){
      if(!cross_face_scri)throw std::invalid_argument("angular dissipation requires cross-face scri jets");
      angular_plan=std::make_shared<const cubed_sphere::AngularDissipationPlan>(scri,grid.n,7);
    }
    for(size_t p=0;p<grid.active.size();++p)state[nf*p]=1;
    for(size_t p=0;p<scri.rows.size();++p)scri.values[nf*p]=1;
  }
  static Real Radius(const Real*x){return std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);}
  static Z4cJet<Real> Jet(const Real*x,const V&v,
      const std::vector<std::array<Real,3>>&d,
      const std::vector<std::array<std::array<Real,3>,3>>&dd) {
    Z4cJet<Real> u{};
    auto scalar=[&](int f,ScalarJet<Real>&s){s.value=v[f];
      for(int a=0;a<3;++a){s.d[a]=d[f][a];for(int b=0;b<3;++b)s.dd[a][b]=dd[f][a][b];}};
    scalar(0,u.chi);scalar(13,u.trace);scalar(14,u.theta);scalar(18,u.alpha);
    u.trace.value-=3;u.alpha.value+=(1+Radius(x)*Radius(x))/2;
    for(int a=0;a<3;++a){u.alpha.d[a]+=x[a];u.alpha.dd[a][a]+=1;
      u.beta.value[a]=v[22+a]-x[a];u.lambda.value[a]=v[15+a];u.beta.d[a][a]=-1;
      for(int b=0;b<3;++b){u.beta.d[b][a]+=d[22+a][b];u.lambda.d[b][a]=d[15+a][b];
        for(int c=0;c<3;++c){u.beta.dd[b][c][a]=dd[22+a][b][c];u.lambda.dd[b][c][a]=dd[15+a][b][c];}}}
    for(int t=0;t<6;++t){int i=ti[t],j=tj[t];u.metric.g[i][j]=u.metric.g[j][i]=v[1+t]+(i==j);
      u.a.k[i][j]=u.a.k[j][i]=v[7+t];
      for(int a=0;a<3;++a){u.metric.dg[a][i][j]=u.metric.dg[a][j][i]=d[1+t][a];
        u.a.dk[a][i][j]=u.a.dk[a][j][i]=d[7+t][a];
        for(int b=0;b<3;++b)u.metric.ddg[a][b][i][j]=u.metric.ddg[a][b][j][i]=dd[1+t][a][b];}}
    if(!(u.chi.value>0&&u.alpha.value>0)||!Geometry(u.metric).valid)
      throw std::runtime_error("hyperboloidal stage: nonpositive geometry/lapse");
    return u;
  }
  static OmegaJet<Real> Omega(const Real*x,const Z4cJet<Real>&u,bool boundary=false){
    OmegaJet<Real> o{};o.omega=boundary?0:(1-Radius(x)*Radius(x))/2;
    for(int a=0;a<3;++a){o.gradient[a]=-x[a];o.hessian[a][a]=-1;}
    SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,o);return o;
  }
  static VectorJet<Real> Vector(int offset,const V&v,const std::vector<std::array<Real,3>>&d){
    VectorJet<Real> q{};for(int i=0;i<3;++i){q.value[i]=v[offset+i];
      for(int a=0;a<3;++a)q.d[a][i]=d[offset+i][a];}return q;
  }
  static void StoreGeometry(const Z4cRHS<Real>&r,Real*out){
    out[0]=r.chi;out[13]=r.trace;out[14]=r.theta;
    for(int t=0;t<6;++t){out[1+t]=r.metric[ti[t]][tj[t]];out[7+t]=r.a[ti[t]][tj[t]];}
    for(int a=0;a<3;++a)out[15+a]=r.lambda[a];
  }
  // Fixed angular coordinates supply the two incoming tangential gauge data.
  // The normal value is eliminated from the live metric, not from a fixed jet.
  static Real NullResidual(const Real*x,const Real*v){
    MetricJet<Real> m{};for(int t=0;t<6;++t)m.g[ti[t]][tj[t]]=m.g[tj[t]][ti[t]]=v[1+t]+(ti[t]==tj[t]);
    const auto g=Geometry(m);if(!g.valid||!(v[0]>0))throw std::runtime_error("invalid scri geometry");
    Real q=0,normal=0;for(int i=0;i<3;++i){normal+=x[i]*(v[22+i]-x[i]);
      for(int j=0;j<3;++j)q+=g.inverse[i][j]*x[i]*x[j];}
    return std::abs(normal*normal-v[0]*q)/(v[0]*q);
  }
  static void EnforceBoundaryValues(const Real*x,Real*v){
    v[18]=0;MetricJet<Real> m{};
    for(int t=0;t<6;++t)m.g[ti[t]][tj[t]]=m.g[tj[t]][ti[t]]=v[1+t]+(ti[t]==tj[t]);
    const auto g=Geometry(m);if(!g.valid||!(v[0]>0))throw std::runtime_error("invalid scri geometry");
    Real q=0;for(int i=0;i<3;++i)for(int j=0;j<3;++j)q+=g.inverse[i][j]*x[i]*x[j];
    const Real speed=std::sqrt(v[0]*q);
    for(int a=0;a<3;++a){v[22+a]=(1-speed)*x[a];v[28+a]=0;}
  }
  void BoundaryValues(){
    for(size_t p=0;p<scri.rows.size();++p){const auto*x=scri.rows[p].geometry.x;auto*v=&scri.values[p*nf];
      EnforceBoundaryValues(x,v);null_max=std::max(null_max,NullResidual(x,v));}
  }
  V ActivePadded(const V&values)const{
    V out(nf*grid.count,0);
    for(size_t p=0;p<grid.active.size();++p)for(int f=0;f<nf;++f)out[f*grid.count+grid.active[p]]=values[p*nf+f];
    return out;
  }
  void FillGhosts(V&out)const{
    // Build the surface cache before threading; ghost interpolation reads only
    // active donors. Component partitions therefore have no exchange ordering.
    std::map<std::array<Real,3>,V> cache;
    for(const auto&row:grid.halos)if(row.boundary_weight!=0&&cache.find(row.boundary)==cache.end())
      {auto values=scri.Interpolate(scri.values,nf,row.boundary.data());
       interpolated_null_before_max=std::max(interpolated_null_before_max,NullResidual(row.boundary.data(),values.data()));
       // A nonlinear null relation does not commute with angular interpolation.
       // Apply it to the interpolated live geometry at EVERY ghost evaluation.
       EnforceBoundaryValues(row.boundary.data(),values.data());
       interpolated_null_max=std::max(interpolated_null_max,NullResidual(row.boundary.data(),values.data()));
       cache.emplace(row.boundary,std::move(values));}
    const int count=std::max(1,std::min(workers,nf));std::vector<std::thread> pool;
    auto fill=[&](int worker){grid.Fill(out,nf,[&](int f,const auto&x){return cache.at(x)[f];},nf*worker/count,nf*(worker+1)/count);};
    for(int w=1;w<count;++w)pool.emplace_back(fill,w);fill(0);for(auto&t:pool)t.join();
  }
  V Padded(const V&values)const{auto out=ActivePadded(values);FillGhosts(out);return out;}
  // Conservative isolated-Q bound only; this is not a combined PDE/RK proof.
  // Shared by the CPU reference and callers orchestrating the device operator.
  void CheckAngularCFL(Real dt){
    angular_cfl=0;
    if(angular_epsilon==0)return;
    if(!angular_plan||!std::isfinite(angular_epsilon)||angular_epsilon<0||!(dt>0)||!std::isfinite(dt))
      throw std::invalid_argument("invalid angular dissipation configuration/timestep");
    angular_cfl=dt*angular_epsilon*grid.n*angular_plan->SpectralUpperBound();
    angular_cfl_max=std::max(angular_cfl_max,angular_cfl);
    if(!std::isfinite(angular_cfl)||angular_cfl>2.7)
      throw std::runtime_error("isolated angular dissipation CFL exceeded");
  }
  // Geometry receives its regular limit and weak numerator/regularity rates.
  // No geometric field or independently evolved Lambda is algebraically reset.
  V BoundaryRates(const V&padded,Real dt){
    V out(scri.values.size(),0);
    CheckAngularCFL(dt);
    // BoundaryValues must already have supplied the projected actual surface.
    // Compute once before any dependent-D row writes; only 21 free fields enter Q.
    const V angular=angular_epsilon>0?angular_plan->Apply(scri.values,angular_epsilon):V{};
    for(size_t p=0;p<scri.rows.size();++p){const auto&row=scri.rows[p];const auto*x=row.geometry.x;
      const auto sample=scri.Sample(p,padded,scri.values,nf);const auto u=Jet(x,sample.value,sample.d,sample.dd);
      const auto o=Omega(x,u,true);const auto g=Geometry(u.metric);
      auto parts=ConformalRHS(u,o,4.,-.99);if(!HyperbolicDamping(x,u,o,parts))throw std::runtime_error("scri source invalid");
      const auto pole=TensorPoleNormalJet(u,o,x,x,4.,-.99,1.,1.,true,true);
      auto rate=TensorScriLimit(parts.regular,pole,-1.);const auto trace=ScriRegularityTraces(u,o);
      const Real h=row.radial_spacing;if(dt*12/h>.5)throw std::runtime_error("scri explicit penalty CFL");
      Real q=0;for(int i=0;i<3;++i)for(int j=0;j<3;++j)q+=g.inverse[i][j]*x[i]*x[j];
      rate.chi+=4/h*parts.pole.chi-8/h*(u.chi.value-1/q);
      rate.trace+=4/h*parts.pole.trace-8/h*(u.trace.value+3);
      rate.theta+=4/h*parts.pole.theta-8/h*u.theta.value;
      pole_max=std::max({pole_max,std::abs(parts.pole.chi),std::abs(parts.pole.trace),std::abs(parts.pole.theta)});
      for(int i=0;i<3;++i){rate.lambda[i]+=4/h*parts.pole.lambda[i]-8/h*trace.lambda[i];
        pole_max=std::max(pole_max,std::abs(parts.pole.lambda[i]));
        for(int j=0;j<3;++j){rate.a[i][j]+=4/h*parts.pole.a[i][j]-8/h*trace.a[i][j];
          pole_max=std::max(pole_max,std::abs(parts.pole.a[i][j]));}}
      if(!angular.empty()){
        rate.chi+=angular[p*nf];rate.trace+=angular[p*nf+13];rate.theta+=angular[p*nf+14];
        for(int t=0;t<6;++t){const int i=ti[t],j=tj[t];
          rate.metric[i][j]+=angular[p*nf+1+t];rate.a[i][j]+=angular[p*nf+7+t];
          if(i!=j){rate.metric[j][i]+=angular[p*nf+1+t];rate.a[j][i]+=angular[p*nf+7+t];}}
        for(int i=0;i<3;++i)rate.lambda[i]+=angular[p*nf+15+i];
      }
      StoreGeometry(rate,&out[p*nf]);
      const auto B=Vector(19,sample.value,sample.d);
      const auto gauge=HyperbolicGauge(x,u,B,&sample.value[25]);
      clock_max=std::max(clock_max,std::abs(gauge.lapse_regular-pole.lapse.derivative));
      pole_max=std::max(pole_max,std::abs(gauge.lapse_pole));
      for(int a=0;a<3;++a)out[p*nf+19+a]=gauge.flux_rate[a];
      if(!angular.empty())for(int a=0;a<3;++a)out[p*nf+19+a]+=angular[p*nf+19+a];
      Real qdot=0;Real invn[3]{};
      for(int i=0;i<3;++i)for(int j=0;j<3;++j)invn[i]+=g.inverse[i][j]*x[j];
      for(int i=0;i<3;++i)for(int j=0;j<3;++j)qdot-=invn[i]*rate.metric[i][j]*invn[j];
      const Real speed=std::sqrt(u.chi.value*q),speed_t=(rate.chi*q+u.chi.value*qdot)/(2*speed);
      const Real mu=ShiftCoefficients(Real(1),Real(1),u.chi.value).mu;
      for(int a=0;a<3;++a){Real regular=-x[a],restoring_limit=0;
        for(int b=0;b<3;++b){regular+=u.beta.value[b]*u.beta.d[b][a];restoring_limit+=4*x[b]*sample.d[22+a][b];}
        const Real target=-speed_t*x[a];
        scri.values[p*nf+25+a]=(target-regular-restoring_limit)/mu;
        compatibility_max=std::max(compatibility_max,std::abs(regular+restoring_limit+mu*scri.values[p*nf+25+a]-target));
      }
    }return out;
  }
  // Fourth-order transport biased in the upwind direction of u_t=beta.grad(u).
  // Analytic background jets remain exact; only stored deviations are differenced.
  Real TransportCorrection(const V &padded,int f,size_t p,const Real beta[3])const{
    const int point=f*grid.count+grid.active[p];Real correction=0;
    for(int A=0;A<3;++A){Real b=0;for(int i=0;i<3;++i)b+=grid.geometry[p].J[A][i]*beta[i];
      const int sign=b>=0?1:-1;const Real anchor=padded[point];
      auto value=[&](int j){return padded[point+grid.Offset(A,sign*j)]-anchor;};
      const Real biased=sign*grid.n*(-3*value(-1)+18*value(1)-6*value(2)+value(3))/12;
      correction+=b*(biased-grid.First(padded,point,A));
    }return correction;
  }
  // Block-local KO8 in conservative -V^-1 B^T W B form. Positive physical
  // volume weights give a nonpositive discrete energy contribution, including
  // core/shell block edges. It vanishes on the analytic constant background.
  // Unlike the 1D stencil it does not couple dissipation across block seams;
  // the transport and geometric derivatives DO exchange across every seam.
  void Dissipate(const V&v,V&rhs)const{
    if(ko==0)return;
    constexpr int b[5]={1,-4,6,-4,1};const int n=grid.n;
    for(size_t m=0;m<grid.mesh.blocks.size();++m)
    for(int k=0;k<n;++k)for(int j=0;j<n;++j)for(int i=0;i<n;++i)
    for(int A=0;A<3;++A){const int pos[3]={i,j,k};if(pos[A]+4>=n)continue;
      const size_t start=m*n*n*n+(k*n+j)*n+i;
      const int stride=A==0?1:(A==1?n:n*n);Real weight=0;
      for(int a=0;a<5;++a){const size_t p=start+a*stride;Real inverse_length=0;
        for(int d=0;d<3;++d)inverse_length+=grid.geometry[p].J[A][d]*grid.geometry[p].J[A][d];
        const Real r=Radius(grid.geometry[p].x),taper=std::pow(std::max(Real(0),1-r*r/.64),6);
        weight+=grid.volumes[p]*std::sqrt(inverse_length)*taper/5;}
      for(int f=0;f<nf;++f){Real fourth=0;const Real anchor=v[nf*start+f];
        for(int a=0;a<5;++a)fourth+=b[a]*(v[nf*(start+a*stride)+f]-anchor);
        for(int a=0;a<5;++a){const size_t p=start+a*stride;
          rhs[nf*p+f]-=ko*n*b[a]*weight*fourth/(256*grid.volumes[p]);}}
    }
  }
  V RHS(const V&values,Real dt,V&edge_rate){
    try{return EvaluateRHS(values,dt,edge_rate);}
    catch(...){failed_state=values;failed_boundary=scri.values;throw;}
  }
  V EvaluateRHS(const V&values,Real dt,V&edge_rate){
    BoundaryValues();auto padded=ActivePadded(values);edge_rate=BoundaryRates(padded,dt);
    // Dependent D traces must be present before all volume derivative evaluations.
    FillGhosts(padded);V out(values.size(),0);
    const int count=std::max(1,std::min(workers,int(grid.active.size())));
    struct Worker {Real gap=std::numeric_limits<Real>::infinity(),cfl=0;std::exception_ptr error;};
    std::vector<Worker> diagnostics(count);
    auto evaluate=[&](int worker){
      auto&local=diagnostics[worker];
      V v(nf);std::vector<std::array<Real,3>> d(nf);
      std::vector<std::array<std::array<Real,3>,3>> dd(nf);
      try{for(size_t p=grid.active.size()*worker/count;p<grid.active.size()*(worker+1)/count;++p){const auto*x=grid.geometry[p].x;
      for(int f=0;f<nf;++f){v[f]=values[p*nf+f];Real df[3],ddf[3][3];grid.Derivatives(padded,f,p,df,ddf,f<=6||f==18||(f>=22&&f<25));
        for(int a=0;a<3;++a){d[f][a]=df[a];for(int b=0;b<3;++b)dd[f][a][b]=ddf[a][b];}}
      const auto u=Jet(x,v,d,dd);const auto o=Omega(x,u);auto parts=ConformalRHS(u,o,4.,-.99);
      if(!HyperbolicDamping(x,u,o,parts))throw std::runtime_error("invalid bulk source");
      Z4cRHS<Real> rate{};if(!AssembleInterior(parts,o.omega,rate))throw std::runtime_error("invalid interior rate");
      StoreGeometry(rate,&out[p*nf]);
      const auto B=Vector(19,v,d),D=Vector(25,v,d);const auto gauge=HyperbolicGauge(x,u,B,D.value);
      if(!gauge.valid)throw std::runtime_error("invalid hyperbolic gauge");
      out[p*nf+18]=gauge.lapse_regular+gauge.lapse_pole/o.omega;
      const Real r=Radius(x);const auto c=ShiftCoefficients(r,u.alpha.value,u.chi.value);
      const Real w=adaptive?1-c.collar:0;
      Real driver[3];if(!HyperbolicAuxiliaryRate(u,D,rate.lambda,driver))throw std::runtime_error("invalid D rate");
      for(int a=0;a<3;++a){out[p*nf+19+a]=gauge.flux_rate[a];out[p*nf+25+a]=driver[a];
        out[p*nf+22+a]=gauge.shift_regular[a]+gauge.shift_pole[a]/o.omega-w*v[28+a];
        Real target=-x[a]-v[28+a];for(int b=0;b<3;++b)target+=u.beta.value[b]*u.beta.d[b][a];
        out[p*nf+28+a]=.1*w*target;}
      for(int f=0;f<28;++f)out[p*nf+f]+=TransportCorrection(padded,f,p,u.beta.value);
      // D_t contains Lambda_t - beta.grad Lambda. The transport correction to
      // Lambda cancels there; D itself receives its own upwind transport above.
      if(adaptive)for(int a=0;a<3;++a)
        out[p*nf+28+a]+=.1*w*TransportCorrection(padded,22+a,p,u.beta.value);
      const auto g=Geometry(u.metric);local.gap=std::min(local.gap,AllDirectionShiftLapseGap(u.metric.g,u.chi.value,c));
      Real speed=0;for(int A=0;A<3;++A){Real transport=0,Q=0,E=0;
        for(int i=0;i<3;++i){transport+=grid.geometry[p].J[A][i]*u.beta.value[i];E+=grid.geometry[p].J[A][i]*grid.geometry[p].J[A][i];
          for(int j=0;j<3;++j)Q+=g.inverse[i][j]*grid.geometry[p].J[A][i]*grid.geometry[p].J[A][j];}
        speed+=std::abs(transport)+std::sqrt(std::max(4*c.mu*Q/3,u.chi.value*(c.F*Q+c.eta*.25*c.w*c.w*E)));}
      local.cfl=std::max(local.cfl,dt*grid.n*speed);
      if(dt*grid.n*speed>.4||dt/.02>.25)throw std::runtime_error("hyperbolic composite CFL exceeded");
    }
      }catch(...){local.error=std::current_exception();}
    };
    std::vector<std::thread> pool;for(int w=1;w<count;++w)pool.emplace_back(evaluate,w);
    evaluate(0);for(auto&t:pool)t.join();
    for(const auto&w:diagnostics){gap_min=std::min(gap_min,w.gap);cfl_max=std::max(cfl_max,w.cfl);
      if(w.error)std::rethrow_exception(w.error);}
    V diss(out.size(),0);Dissipate(values,diss);
    for(size_t p=0;p<grid.active.size();++p){
      for(int f=0;f<nf;++f)out[p*nf+f]+=diss[p*nf+f];
      // The donor drives D with the numerical Lambda rate, including its KO.
      for(int a=0;a<3;++a)out[p*nf+25+a]+=diss[p*nf+15+a];
    }
    for(Real x:out)if(!std::isfinite(x))throw std::runtime_error("nonfinite bulk RHS");
    for(Real x:edge_rate)if(!std::isfinite(x))throw std::runtime_error("nonfinite scri RHS");
    return out;
  }
  void Step(Real dt){
    const V begin=state,edge=scri.values,old_comp=compensation,old_edge_comp=edge_compensation;
    V b1,b2,b3,b4,tmp(state.size());failed_state.clear();failed_boundary.clear();
    try{failed_stage=1;
    auto k1=RHS(begin,dt,b1);
    auto stage=[&](const V&k,const V&b,Real h){for(size_t i=0;i<tmp.size();++i)tmp[i]=begin[i]+h*k[i];
      for(size_t i=0;i<edge.size();++i)scri.values[i]=edge[i]+h*b[i];};
    stage(k1,b1,dt/2);failed_stage=2;auto k2=RHS(tmp,dt,b2);stage(k2,b2,dt/2);failed_stage=3;auto k3=RHS(tmp,dt,b3);
    stage(k3,b3,dt);failed_stage=4;auto k4=RHS(tmp,dt,b4);
    auto update=[&](V&v,V&c,const V&base,const V&a,const V&b,const V&d,const V&e){
      for(size_t i=0;i<v.size();++i){Real inc=dt*(a[i]+2*b[i]+2*d[i]+e[i])/6-c[i];
        v[i]=base[i]+inc;c[i]=(v[i]-base[i])-inc;}};
    update(state,compensation,begin,k1,k2,k3,k4);update(scri.values,edge_compensation,edge,b1,b2,b3,b4);
    failed_stage=5;
    for(size_t p=0;p<grid.active.size();++p){const auto*v=&state[p*nf];
      const Real a=1+v[1],b=v[2],c=v[3],d=1+v[4],e=v[5],f=1+v[6];
      const Real det=a*(d*f-e*e)-b*(b*f-c*e)+c*(b*e-c*d);
      if(!(v[0]>0&&v[18]+(1+std::pow(Radius(grid.geometry[p].x),2))/2>0&&a>0&&a*d-b*b>0&&det>0))
        throw std::runtime_error("nonpositive accepted geometry/lapse");
      for(int field=0;field<nf;++field)if(!std::isfinite(v[field]))throw std::runtime_error("nonfinite accepted state");
    }
    BoundaryValues();BoundaryRates(ActivePadded(state),dt);
    for(size_t p=0;p<scri.rows.size();++p){edge_compensation[p*nf+18]=0;
      for(int f=22;f<nf;++f)edge_compensation[p*nf+f]=0;}
    time+=dt;++steps;failed_stage=0;
    }catch(...){if(failed_state.empty()){failed_state=state;failed_boundary=scri.values;}
      state=begin;scri.values=edge;compensation=old_comp;edge_compensation=old_edge_comp;throw;}

  }
};
} // namespace z4c::hyperboloidal
#endif
