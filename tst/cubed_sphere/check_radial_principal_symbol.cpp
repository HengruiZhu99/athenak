// Frozen-coefficient pseudo-differential principal symbol of implemented RHS.
// Diagnostic only; no solver changes. q=(chi,g6,alpha,beta3), p=(A6,P,Theta,Lambda3).
// Reduction variable is d_r q. Coordinate speeds are minus eigenvalues.
#include <iomanip>
#include <iostream>
#include <fstream>
#include <cmath>
#include <stdexcept>
#include "mesh/cubed_sphere/reference_wave_map.hpp"
namespace h=z4c::hyperboloidal;
using Jet=h::Z4cJet<double>;
const int ia[6]={0,0,0,1,1,2},ib[6]={0,1,2,1,2,2};

void Seed(Jet &u,int column,int order,const double n[3],double e) {
  for(int a=0;a<3;++a) {
    if(column==0 && order==1) u.chi.d[a]+=e*n[a];
    if(column==7 && order==1) u.alpha.d[a]+=e*n[a];
    if(column>=8 && column<11 && order==1) u.beta.d[a][column-8]+=e*n[a];
    if(column>=1 && column<=6 && order==1) {
      int i=ia[column-1],j=ib[column-1];u.metric.dg[a][i][j]+=e*n[a];
      if(i!=j)u.metric.dg[a][j][i]+=e*n[a];
    }
    if(column>=11 && column<17 && order==1) {
      int i=ia[column-11],j=ib[column-11];u.a.dk[a][i][j]+=e*n[a];
      if(i!=j)u.a.dk[a][j][i]+=e*n[a];
    }
    if(column==17 && order==1)u.trace.d[a]+=e*n[a];
    if(column==18 && order==1)u.theta.d[a]+=e*n[a];
    if(column>=19 && order==1)u.lambda.d[a][column-19]+=e*n[a];
    for(int b=0;b<3;++b) {
      if(column==0 && order==2)u.chi.dd[a][b]+=e*n[a]*n[b];
      if(column==7 && order==2)u.alpha.dd[a][b]+=e*n[a]*n[b];
      if(column>=8 && column<11 && order==2)u.beta.dd[a][b][column-8]+=e*n[a]*n[b];
      if(column>=1 && column<=6 && order==2) {
        int i=ia[column-1],j=ib[column-1];u.metric.ddg[a][b][i][j]+=e*n[a]*n[b];
        if(i!=j)u.metric.ddg[a][b][j][i]+=e*n[a]*n[b];
      }
    }
  }
  if(order==0) {
    if(column>=11 && column<17) {
      int i=ia[column-11],j=ib[column-11];u.a.k[i][j]+=e;
      if(i!=j)u.a.k[j][i]+=e;
    }
    if(column==17)u.trace.value+=e;
    if(column==18)u.theta.value+=e;
    if(column>=19)u.lambda.value[column-19]+=e;
  }
}
void Evaluate(const Jet &u,const h::LayerPoint<double> &p,const double x[3],bool native,double out[22]) {
  h::OmegaJet<double> o{};o.omega=p.omega;
  for(int a=0;a<3;++a){o.gradient[a]=p.domega[a];for(int b=0;b<3;++b)o.hessian[a][b]=p.omega_hessian[a][b];}
  h::SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,o);
  h::Z4cRHS<double> f{};h::GaugeRHS<double> g{};
  h::LayerGaugeParameters gp;gp.r0=.35;gp.r1=.75;gp.physical_trace_lapse=true;gp.preferred_source=false;
  if(!h::AssembleInterior(h::ConformalRHS(u,o,0.,0.),p.omega,f) ||
     !(native ? h::AssembleGaugeInterior(h::InteriorLayerGauge(p,u,gp),p.omega,g) :
       rwm::Assemble(rwm::Gauge(p,u,rwm::ReferenceConnection(p,x)),p.omega,g)))
    throw std::runtime_error("invalid RHS in symbol extraction");
  out[0]=f.chi;out[7]=g.alpha;out[17]=f.trace;out[18]=f.theta;
  for(int k=0;k<6;++k){out[1+k]=f.metric[ia[k]][ib[k]];out[11+k]=f.a[ia[k]][ib[k]];}
  for(int k=0;k<3;++k){out[8+k]=g.beta[k];out[19+k]=f.lambda[k];}
}
void Constraints(const Jet &u,const h::LayerPoint<double> &p,double out[8]) {
 h::OmegaJet<double> o{};o.omega=p.omega;
 for(int a=0;a<3;++a){o.gradient[a]=p.domega[a];for(int b=0;b<3;++b)o.hessian[a][b]=p.omega_hessian[a][b];}
 h::SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,o);
 const auto c=h::EvolvedConstraints(u,o);if(!c.valid)throw std::runtime_error("invalid constraints");
 out[0]=c.hamiltonian;out[4]=u.theta.value;
 for(int a=0;a<3;++a){out[1+a]=c.momentum[a];out[5+a]=c.z4.z_covector[a];}
}
void Emit(const Jet &u,const h::LayerPoint<double> &p,const double x[3],int family,int label,const double *direction=nullptr) {
 double radius=0,n[3];for(int a=0;a<3;++a)radius+=x[a]*x[a];radius=std::sqrt(radius);for(int a=0;a<3;++a)n[a]=x[a]/radius;
    if(direction)for(int a=0;a<3;++a)n[a]=direction[a];
    const auto geo=h::Geometry(u.metric);double beta=0,norm=0;
    for(int a=0;a<3;++a){beta+=n[a]*u.beta.value[a];for(int b=0;b<3;++b)norm+=n[a]*n[b]*geo.inverse[a][b];}
    const double light=u.alpha.value*std::sqrt(u.chi.value*norm);
    std::cout<<"case "<<family<<" "<<radius<<" "<<label<<" "<<p.omega<<" "<<-beta-light<<" "<<-beta+light<<"\n";
    std::cout<<"inverse";for(int i=0;i<3;++i)for(int j=0;j<3;++j)std::cout<<" "<<geo.inverse[i][j];std::cout<<"\n";
    std::cout<<"normal";for(double v:n)std::cout<<" "<<v;std::cout<<"\n";
    double constraint[8][22]{};
    for(int column=0;column<22;++column)for(int group=0;group<2;++group) {
      Jet plus=u,minus=u;const int order=column<11?(group==0?2:1):(group==0?1:0);
      Seed(plus,column,order,n,1.);Seed(minus,column,order,n,-1.);
      double cp[8],cm[8];Constraints(plus,p,cp);Constraints(minus,p,cm);
      for(int row=group?4:0;row<(group?8:4);++row) {
        // H leading symbol uses ddq only; M uses dp only. Theta,Z are
        // algebraic in reduction variables. Discard lower differential order.
        if(group==0 && ((row==0 && column>=11)||(row>0 && column<11)))continue;
        constraint[row][column]=(cp[row]-cm[row])/2;
      }
    }
    std::cout<<"constraints\n";for(auto &row:constraint){for(double v:row)std::cout<<v<<" ";std::cout<<"\n";}
    for(double step:{1.,.5}) {
      double matrix[22][22]{};
      for(int c=0;c<22;++c)for(int group=0;group<2;++group) {
        Jet plus=u,minus=u;int order=c<11?(group==0?1:2):(group==0?0:1);
        Seed(plus,c,order,n,step);Seed(minus,c,order,n,-step);
        double fp[22],fm[22];Evaluate(plus,p,x,family,fp);Evaluate(minus,p,x,family,fm);
        for(int r=group?11:0;r<(group?22:11);++r)matrix[r][c]=(fp[r]-fm[r])/(2*step);
      }
      std::cout<<"matrix "<<step<<"\n";
      for(auto &row:matrix){for(double v:row)std::cout<<v<<" ";std::cout<<"\n";}
    }
}
int main(int argc,char **argv) {
  Kokkos::initialize(argc,argv);std::cout<<std::setprecision(17);
  if(argc==2 || argc==3) {
    bool logical=argc==3;if(logical && std::string(argv[2])!="--logical-normal")throw std::runtime_error("unknown normal option");
    std::ifstream input(argv[1]);if(!input)throw std::runtime_error("missing sample input");
    int family,label=0;double curvature,r0,r1,x[3],values[25];
    while(input>>family>>curvature>>r0>>r1>>x[0]>>x[1]>>x[2]) {
      for(double &value:values)if(!(input>>value))throw std::runtime_error("truncated sample");
      h::LayerReference<double> ref(1.,curvature,{true,r0,r1});auto p=ref.At(x[0],x[1],x[2]);Jet u{};
      u.chi.value=values[0];u.trace.value=values[7];u.theta.value=values[17];u.alpha.value=values[18];
      for(int k=0;k<6;++k){u.metric.g[ia[k]][ib[k]]=u.metric.g[ib[k]][ia[k]]=values[1+k];u.a.k[ia[k]][ib[k]]=u.a.k[ib[k]][ia[k]]=values[8+k];}
      for(int k=0;k<3;++k){u.lambda.value[k]=values[14+k];u.beta.value[k]=values[19+k];}
      double direction[3]{};if(logical){double norm=0;for(double &v:direction){if(!(input>>v))throw std::runtime_error("missing normal");norm+=v*v;}if(std::abs(norm-1)>1e-12)throw std::runtime_error("nonunit normal");}
      Emit(u,p,x,family,label++,logical?direction:nullptr);
    }
    Kokkos::finalize();return 0;
  }
  // Synthetic admissible live-state controls; not evolved checkpoint evidence.
  for(int family=0;family<2;++family)for(double radius:{.96,.99,.999})
  for(int perturb=0;perturb<3;++perturb) {
    h::LayerReference<double> ref(1.,family?1.:.5,{true,family?.35:.05,family?.75:.95});
    double n[3]={.8,-.48,.36},x[3];for(int a=0;a<3;++a)x[a]=radius*n[a];
    const auto p=ref.At(x[0],x[1],x[2]);Jet u{};
    u.chi.value=p.state.chi.value;u.alpha.value=p.alpha*(perturb==1?1.2:1.);
    u.trace.value=p.state.trace.value;
    for(int a=0;a<3;++a){u.beta.value[a]=p.beta[a]+(perturb==2?.02*n[a]:0.);
      for(int b=0;b<3;++b){u.metric.g[a][b]=p.state.metric.g[a][b];u.a.k[a][b]=p.state.a.k[a][b];}}
    Emit(u,p,x,family,perturb);
  }
  Kokkos::finalize();
}
