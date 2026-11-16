// Assessment-only spherically symmetric reduction of unchanged scheme A.
//
// The state is the production 31-field Cartesian layout sampled on the +x axis,
// x=(r,0,0). Every equation is evaluated by the production point kernels
// (EvaluateBulkPoint, SphericalPolicy::Boundary, NormalLift, the algebraic
// chart, the analytic reference and stationary source). Only the derivative
// supply differs from the 3D code: angular derivatives are exact consequences
// of spherical symmetry instead of finite differences, and the radial grid is
// a single uniform nodal line that ends in the production outer closure
// (nodal last layer, separately evolved scri node, seven-node endpoint rows on
// the last three nodes, shift normal solve and lift, penalties, projections).
// There are no shells, overlaps, agreement penalties or angular dissipation.
// This is a diagnostic model; it is not part of the production build.
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "z4c_hyperboloidal/point_rhs.hpp"
#include "z4c_hyperboloidal/spherical_reference_3d.hpp"
#include "z4c_hyperboloidal/central_z4c_configuration.hpp"
#include "z4c_hyperboloidal/endpoint.hpp"
#include "z4c_hyperboloidal/dissipation.hpp"
#include "z4c_hyperboloidal/conformal_constraints.hpp"

namespace h=z4c_hyperboloidal;
using J=h::mapped::Jet;
using Jets=h::CartesianChartJets;
constexpr int NF=31;
using Row=std::array<double,NF>;
struct R3 {double v=0,d1=0,d2=0;};

static bool NeedSecond(int f) {return f<=6 || f==18 || (f>=22 && f<25);}
// Radial structure functions carried by the axis values of these fields.
static const int kRadialFields[12]={0,1,4,7,10,13,14,15,18,19,22,25};

static J ScalarJet(const R3 &q,const J &r) {
  J o(q.v);
  for (int a=0;a<3;++a) {
    o.d[a]=q.d1*r.d[a];
    for (int b=0;b<3;++b) o.h[a][b]=q.d2*r.d[a]*r.d[b]+q.d1*r.h[a][b];
  }
  return o;
}

// Cartesian jets at x of spherically symmetric fields with radial jets q[f].
static Jets CartesianFromRadial(const double *x,const std::array<R3,NF> &q,bool mask) {
  J xx[3],r2;
  for (int a=0;a<3;++a) {xx[a]=J::Variable(x[a],a);r2=r2+xx[a]*xx[a];}
  const J r=h::mapped::Power(r2,.5);
  Jets v{};
  for (int f:{0,13,14,18}) v[f]=ScalarJet(q[f],r);
  const J g1=ScalarJet(q[1],r),g2=ScalarJet(q[4],r);
  const J a1=ScalarJet(q[7],r),a2=ScalarJet(q[10],r);
  const int ti[6]={0,0,0,1,1,2},tj[6]={0,1,2,1,2,2};
  for (int t=0;t<5;++t) {
    const int a=ti[t],b=tj[t];const J nn=xx[a]*xx[b]/r2;
    v[1+t]=(g1-g2)*nn+(a==b ? g2 : J(0));
    v[7+t]=(a1-a2)*nn+(a==b ? a2 : J(0));
  }
  for (int f:{15,19,22,25}) {
    const J s=ScalarJet(q[f],r);
    for (int a=0;a<3;++a) v[f+a]=s*xx[a]/r;
  }
  if (mask) for (int f=0;f<NF;++f) if (!NeedSecond(f))
    for (int a=0;a<3;++a) for (int b=0;b<3;++b) v[f].h[a][b]=0;
  return v;
}

static bool Invert4(const double g[4][4],double inv[4][4]) {
  double m[4][8];
  for (int a=0;a<4;++a) for (int b=0;b<4;++b) {m[a][b]=g[a][b];m[a][4+b]=a==b;}
  for (int c=0;c<4;++c) {
    int p=c;for (int r=c+1;r<4;++r) if (fabs(m[r][c])>fabs(m[p][c])) p=r;
    if (fabs(m[p][c])<1e-300) return false;
    for (int k=0;k<8;++k) std::swap(m[c][k],m[p][k]);
    const double s=m[c][c];for (int k=0;k<8;++k) m[c][k]/=s;
    for (int r=0;r<4;++r) if (r!=c) {const double f=m[r][c];for (int k=0;k<8;++k) m[r][k]-=f*m[c][k];}
  }
  for (int a=0;a<4;++a) for (int b=0;b<4;++b) inv[a][b]=m[a][4+b];
  return true;
}

struct TResult {double T[4][4]{},box=0,nt=0,max=0;bool valid=false;};
// T_ab = nabla_a nabla_b Omega - g_ab box(Omega)/4 for Omega=(1-r^2)/2 at scri,
// from restored (lifted) Cartesian jets and projected rates. Mirrors
// vis/python/scri_frame.generator_geometry and nonradiative_native_analysis.
static TResult ScriTensor(const double *x,const Jets &j,const Row &rate) {
  TResult out;
  const int ti[6]={0,0,0,1,1,2},tj[6]={0,1,2,1,2,2};
  Jets tangent{};
  for (int f=0;f<NF;++f) {tangent[f].f=j[f].f;tangent[f].d[0]=rate[f];}
  if (!h::TryExpandCartesianChart(tangent)) return out;
  double r2=0;for (int a=0;a<3;++a) r2+=x[a]*x[a];
  const double chi=j[0].f,alpha=j[18].f+(1+r2)/2;
  double beta[3],gt[3][3],dchi[4],dalpha[4],dbeta[4][3],dgt[4][3][3];
  for (int a=0;a<3;++a) beta[a]=j[22+a].f-x[a];
  dchi[0]=rate[0];dalpha[0]=rate[18];
  for (int a=0;a<3;++a) dbeta[0][a]=rate[22+a];
  for (int k=0;k<3;++k) {
    dchi[1+k]=j[0].d[k];dalpha[1+k]=j[18].d[k]+x[k];
    for (int a=0;a<3;++a) dbeta[1+k][a]=j[22+a].d[k]-(a==k);
  }
  for (int t=0;t<6;++t) {
    const int a=ti[t],b=tj[t];
    gt[a][b]=gt[b][a]=j[1+t].f+(a==b);
    dgt[0][a][b]=dgt[0][b][a]=tangent[1+t].d[0];
    for (int k=0;k<3;++k) dgt[1+k][a][b]=dgt[1+k][b][a]=j[1+t].d[k];
  }
  double g[4][4]{},dg[4][4][4]{};
  double gam[3][3],lower[3]{};
  for (int a=0;a<3;++a) for (int b=0;b<3;++b) gam[a][b]=gt[a][b]/chi;
  for (int a=0;a<3;++a) for (int b=0;b<3;++b) lower[a]+=gam[a][b]*beta[b];
  g[0][0]=-alpha*alpha;
  for (int a=0;a<3;++a) {
    g[0][0]+=lower[a]*beta[a];g[0][1+a]=g[1+a][0]=lower[a];
    for (int b=0;b<3;++b) g[1+a][1+b]=gam[a][b];
  }
  for (int c=0;c<4;++c) {
    double dgam[3][3],dlower[3]{};
    for (int a=0;a<3;++a) for (int b=0;b<3;++b)
      dgam[a][b]=dgt[c][a][b]/chi-gam[a][b]*dchi[c]/chi;
    for (int a=0;a<3;++a) for (int b=0;b<3;++b)
      dlower[a]+=dgam[a][b]*beta[b]+gam[a][b]*dbeta[c][b];
    dg[c][0][0]=-2*alpha*dalpha[c];
    for (int a=0;a<3;++a) {
      dg[c][0][0]+=dlower[a]*beta[a]+lower[a]*dbeta[c][a];
      dg[c][0][1+a]=dg[c][1+a][0]=dlower[a];
      for (int b=0;b<3;++b) dg[c][1+a][1+b]=dgam[a][b];
    }
  }
  double inv[4][4];
  if (!Invert4(g,inv)) return out;
  const double grad[4]={0,-x[0],-x[1],-x[2]};
  double H[4][4];
  for (int a=0;a<4;++a) for (int b=0;b<4;++b) {
    H[a][b]=(a==b && a>0) ? -1 : 0;
    for (int c=0;c<4;++c) {
      double conn=0;
      for (int d=0;d<4;++d) conn+=.5*inv[c][d]*(dg[a][d][b]+dg[b][d][a]-dg[d][a][b]);
      H[a][b]-=conn*grad[c];
    }
  }
  for (int a=0;a<4;++a) for (int b=0;b<4;++b) out.box+=inv[a][b]*H[a][b];
  for (int b=0;b<4;++b) out.nt+=inv[0][b]*grad[b];
  for (int a=0;a<4;++a) for (int b=0;b<4;++b) {
    out.T[a][b]=H[a][b]-g[a][b]*out.box/4;
    out.max=fmax(out.max,fabs(out.T[a][b]));
  }
  out.valid=true;
  return out;
}

struct Options {
  double hh=.005,rin=.5,tend=.24,dtfac=.0938884074712349,amp=0,center=.9555,width=.009;
  double restoring=4;
  double kappa=.25,penalty=.5,ko=.1,taper=.02;
  int endpoint_points=7;
  int profile=0,stride=1,threads=8,lift_layers=2;  // production lifts the last two bulk layers
  std::string out="out";
  std::vector<double> profile_times;
  bool test_jets=false;
};

struct Diagnostic {
  double clock=0,speed_t=0,defect=0,lift=0,alpha_n=0,P_n=0,beta_nn=0,speed=0;
  double alpha_nn=0,pen_chi=0,pen_P=0,pen_theta=0,chi=0,P=0,theta=0;
  double raw_alpha_n=0,raw_alpha_nn=0,raw_beta_nn=0;
  TResult tensor;
};

class Sim {
 public:
  Options o;
  int N=0;                      // bulk nodes 0..N-1; node N is scri
  static constexpr int frozen=5;
  h::HyperbolicShiftParameters<double> p;
  std::vector<double> r,mask;
  std::vector<Jets> refj;
  std::vector<Row> ref;
  std::vector<h::StationaryGaugeSource<double>> src;
  std::vector<Row> u;
  double w1[4][9]{},w2[4][9]{};     // endpoint rows for nodes N-3..N
  std::vector<double> ko_taper;

  explicit Sim(const Options &options):o(options) {
    h::CentralZ4cConfiguration c;c.inner_z4c_radius_M=20;c.compact_slicing_end=.97;
    p=c.Resolve().gauge;p.enforce_cfl=true;p.outer_lapse_damping=o.kappa;
    p.restoring=o.restoring;
    p.scri_treatment=h::ScriTreatment::existing;p.Validate();
    N=static_cast<int>(std::lround((1-o.rin)/o.hh));
    if (N<40) throw std::invalid_argument("too few radial nodes");
    r.resize(N+1);mask.resize(N+1);refj.resize(N+1);ref.resize(N+1);src.resize(N+1);u.resize(N+1);
    h::SphericalReference3D reference(p);
    for (int i=0;i<=N;++i) {
      r[i]=i==N ? 1. : 1-(N-i)*o.hh;
      const double x[3]={r[i],0,0};
      refj[i]=reference.At(x,i==N);src[i]=reference.source.At(r[i]);
      for (int f=0;f<NF;++f) ref[i][f]=refj[i][f].f;
      const double z=fmax(0.,fmin(1.,(r[i]-r[frozen])/o.taper));
      mask[i]=i<frozen ? 0 : z*z*z*(10-15*z+6*z*z);
      u[i]=ref[i];
    }
    mask[N]=1;
    if(o.endpoint_points!=7 && o.endpoint_points!=9) throw std::invalid_argument("endpoint points must be 7 or 9");
    double nodes[9];for (int k=0;k<o.endpoint_points;++k) nodes[k]=1-(o.endpoint_points-1-k)*o.hh;
    for (int row=0;row<4;++row) for (int k=0;k<o.endpoint_points;++k) {
      const J basis=h::LagrangeJet(nodes[o.endpoint_points-4+row],nodes,o.endpoint_points,k,0);
      w1[row][k]=basis.d[0];w2[row][k]=basis.h[0][0];
    }
    ko_taper.resize(N+1);
    for (int i=0;i<=N;++i) ko_taper[i]=std::pow(fmax(0.,1-r[i]*r[i]/.64),6);
  }

  void AddPulse() {
    if (o.amp==0) return;
    for (int i=frozen;i<N;++i) {
      const double z=(r[i]-o.center)/o.width;
      if (fabs(z)>=1) continue;
      u[i][h::Fields::lapse]+=o.amp*(o.profile==0 ? exp(1-1/(1-z*z)) : exp(-6*z*z/(1-z*z)));
    }
  }

  static void Enforce(Row &v) {
    Jets j{};for (int f=0;f<NF;++f) j[f].f=v[f];
    if (!h::TryExpandCartesianChart(j)) throw std::runtime_error("invalid metric chart");
    v[6]=j[6].f;v[12]=j[12].f;
  }

  void Project(std::vector<Row> &s) const {
    Row &e=s[N];Enforce(e);
    Jets j{};for (int f=0;f<NF;++f) j[f].f=e[f];
    const double x[3]={1,0,0};
    const auto physical=h::SphericalPolicy::Physical(x,j);
    const auto geometry=h::Geometry(physical.metric);
    if (!geometry.valid || !(e[0]>0)) throw std::runtime_error("invalid scri geometry");
    double Q=0;for (int a=0;a<3;++a) for (int b=0;b<3;++b) Q+=geometry.inverse[a][b]*x[a]*x[b];
    const double speed=sqrt(e[0]*Q);
    e[18]=0;
    for (int a=0;a<3;++a) {
      e[22+a]=(1-speed)*x[a];e[28+a]=0;
      double C=0;
      for (int k=0;k<5;++k) {
        double weight=1;
        for (int l=0;l<5;++l) if (k!=l) weight*=(5.-l)/double(k-l);
        C+=weight*(s[N-5+k][25+a]-s[N-5+k][15+a]);
      }
      e[25+a]=e[15+a]+C;
    }
  }

  double Residual(const std::vector<Row> &s,int i,int f) const {
    if (!h::Fields::Independent(f)) return 0;
    const double value=s[i][f]-ref[i][f];
    return (f==0 && i<N) ? value/(r[i]*r[i]) : value;
  }

  R3 RadialJet(const std::vector<Row> &s,int i,int f) const {
    R3 q;q.v=Residual(s,i,f);
    if (i>=N-3) {
      const int row=i-(N-3);
      for (int k=0;k<o.endpoint_points;++k) {
        const double delta=Residual(s,N-(o.endpoint_points-1)+k,f)-q.v;
        q.d1+=w1[row][k]*delta;q.d2+=w2[row][k]*delta;
      }
      return q;
    }
    const double m2=Residual(s,i-2,f),m1=Residual(s,i-1,f),p1=Residual(s,i+1,f),p2=Residual(s,i+2,f);
    q.d1=(m2-8*m1+8*p1-p2)/(12*o.hh);
    q.d2=(-m2+16*m1-30*q.v+16*p1-p2)/(12*o.hh*o.hh);
    return q;
  }

  J LiftJet(const double *x,double correction,int a) const {
    h::mapped::Point map{};
    for (int k=0;k<3;++k) {map.x[k]=x[k];map.J[k][k]=1;}
    J xx[3],r2;
    for (int k=0;k<3;++k) {xx[k]=J::Variable(x[k],k);r2=r2+xx[k]*xx[k];}
    const J logical=J(correction)*xx[a]/h::mapped::Power(r2,.5);
    return h::NormalLift(map,logical,o.hh,1.);
  }

  // Production stage order: algebraic closure, scri projection, residuals,
  // scri sample and normal solve, scri rate, then bulk rates with the lift.
  void RHS(std::vector<Row> &s,std::vector<Row> &ds,Diagnostic *diagnostic=nullptr) const {
    for (int i=0;i<=N;++i) Enforce(s[i]);
    Project(s);
    const double xs[3]={1,0,0};
    h::mapped::Point smap{};smap.x[0]=1;for (int a=0;a<3;++a) smap.J[a][a]=1;
    std::array<R3,NF> rq{};
    for (int f=0;f<NF;++f) if (h::Fields::Independent(f)) rq[f]=RadialJet(s,N,f);
    Jets sampled=CartesianFromRadial(xs,rq,false);
    if (!h::SphericalPolicy::Restore(xs,sampled,refj[N])) throw std::runtime_error("scri restore");
    const auto us=h::SphericalPolicy::Physical(xs,sampled);
    const auto mg=h::Geometry(us.metric);
    const auto pg=h::Geometry(h::PenroseMetric(us.metric,us.chi));
    double defect[4][3];
    for (int which=0;which<4;++which) {
      Jets jets=sampled;
      if (which>0) for (int a=0;a<3;++a) jets[21+which].d[a]+=xs[a];
      const auto result=h::EvaluateScriPoint(smap,o.hh,jets,p,src[N],o.penalty,&mg,&pg);
      if (!result.valid) throw std::runtime_error("invalid scri normal case");
      for (int a=0;a<3;++a) defect[which][a]=result.defect[a];
    }
    double matrix[3][4];
    for (int a=0;a<3;++a) {
      matrix[a][3]=-defect[0][a];
      for (int b=0;b<3;++b) matrix[a][b]=defect[b+1][a]-defect[0][a];
    }
    for (int col=0;col<3;++col) {
      int pivot=col;
      for (int row=col+1;row<3;++row) if (fabs(matrix[row][col])>fabs(matrix[pivot][col])) pivot=row;
      for (int k=col;k<4;++k) std::swap(matrix[col][k],matrix[pivot][k]);
      if (!(fabs(matrix[col][col])>=1e-12)) throw std::runtime_error("singular normal solve");
      const double scale=matrix[col][col];
      for (int k=col;k<4;++k) matrix[col][k]/=scale;
      for (int row=0;row<3;++row) if (row!=col) {
        const double multiple=matrix[row][col];
        for (int k=col;k<4;++k) matrix[row][k]-=multiple*matrix[col][k];
      }
    }
    const double correction=matrix[0][3];
    Jets lifted=sampled;
    for (int a=0;a<3;++a) lifted[22+a]=lifted[22+a]+LiftJet(xs,a==0 ? correction : matrix[a][3],a);
    const auto rate=h::EvaluateScriPoint(smap,o.hh,lifted,p,src[N],o.penalty,&mg,&pg);
    if (!rate.valid) throw std::runtime_error("invalid scri rate");
    for (int f=0;f<NF;++f) ds[N][f]=h::Fields::Independent(f) ? rate.rate[f] : 0;
    if (diagnostic) {
      auto &d=*diagnostic;
      const auto bare=h::EvaluateScriPoint(smap,o.hh,lifted,p,src[N],0.,&mg,&pg);
      d.clock=rate.clock;d.speed_t=rate.speed_t;d.lift=correction;
      d.defect=fmax(fabs(rate.defect[0]),fmax(fabs(rate.defect[1]),fabs(rate.defect[2])));
      d.alpha_n=lifted[18].d[0]+1;d.alpha_nn=lifted[18].h[0][0]+1;
      d.P_n=lifted[13].d[0];d.beta_nn=lifted[22].d[0]-1;
      d.raw_alpha_n=sampled[18].d[0]+1;d.raw_alpha_nn=sampled[18].h[0][0]+1;
      d.raw_beta_nn=sampled[22].d[0]-1;
      d.speed=-(s[N][22]-1);
      d.pen_chi=rate.rate[0]-bare.rate[0];d.pen_P=rate.rate[13]-bare.rate[13];
      d.pen_theta=rate.rate[14]-bare.rate[14];
      d.chi=s[N][0];d.P=s[N][13]-3;d.theta=s[N][14];
      Row full{};for (int f=0;f<NF;++f) full[f]=rate.rate[f];
      d.tensor=ScriTensor(xs,lifted,full);
    }
    for (int i=0;i<frozen;++i) ds[i].fill(0);
    const auto work=[&](int begin,int end) {
      for (int i=begin;i<end;++i) Point(s,ds,i,correction);
    };
    const int count=N-frozen,threads=std::max(1,std::min(o.threads,count/16));
    if (threads==1) {work(frozen,N);return;}
    std::vector<std::thread> pool;
    for (int t=0;t<threads;++t)
      pool.emplace_back(work,frozen+count*t/threads,frozen+count*(t+1)/threads);
    for (auto &t:pool) t.join();
  }

  Jets BulkJets(const std::vector<Row> &s,int i,double correction,double *transport) const {
    const double x[3]={r[i],0,0};
    std::array<R3,NF> rq{};
    for (int f=0;f<NF;++f) if (h::Fields::Independent(f)) rq[f]=RadialJet(s,i,f);
    if (transport) {
      for (int f=0;f<NF;++f) transport[f]=0;
      if (i<N-3) {
        const double b=s[i][22]-r[i];const int sign=b>=0 ? 1 : -1;
        for (int f=0;f<NF;++f) if (h::Fields::Independent(f)) {
          const double c=rq[f].v;
          const double vm=Residual(s,i-sign,f)-c,v1=Residual(s,i+sign,f)-c;
          const double v2=Residual(s,i+2*sign,f)-c,v3=Residual(s,i+3*sign,f)-c;
          const double biased=sign*(-3*vm+18*v1-6*v2+v3)/(12*o.hh);
          transport[f]=b*(biased-rq[f].d1);
        }
      }
    }
    Jets jets=CartesianFromRadial(x,rq,true);
    const double w=h::SphericalPolicy::CompatibleWeight(r[i],p);
    if (w>0) {
      if (i>=N-4 || i<4) throw std::runtime_error("compatible lapse window outside stencil");
      double composed=0;
      for (int k=-2;k<=2;++k) if (k) {
        const double c=(k==1||k==-1) ? 2.*k/3 : -k/24.;
        const int q=i+k;
        const double first=(Residual(s,q-2,18)-8*Residual(s,q-1,18)+8*Residual(s,q+1,18)
                            -Residual(s,q+2,18))/(12*o.hh);
        composed+=c*first/o.hh;
      }
      const double delta=w*(composed-rq[18].d2);
      jets[18].h[0][0]+=delta;
    }
    if (!h::SphericalPolicy::Restore(x,jets,refj[i],p.residual_weight))
      throw std::runtime_error("bulk restore");
    if (i>=N-o.lift_layers) for (int a=0;a<3;++a) jets[22+a]=jets[22+a]+LiftJet(x,a==0 ? correction : 0.,a);
    if (transport) transport[0]*=p.residual_weight.Value(x);
    return jets;
  }

  void Point(const std::vector<Row> &s,std::vector<Row> &ds,int i,double correction) const {
    h::mapped::Point map{};map.x[0]=r[i];for (int a=0;a<3;++a) map.J[a][a]=1;
    h::PointForcing forcing;
    const Jets jets=BulkJets(s,i,correction,forcing.transport);
    for (int f=28;f<NF;++f) forcing.transport[f]=0;
    if (o.ko>0 && ko_taper[std::min(N,i+4)]>0) {
      const int c[5]={1,-4,6,-4,1};
      for (int target=0;target<5;++target) {
        const int start=i-target;
        if (start<0 || start+4>=N) continue;
        double weight=0;
        for (int a=0;a<5;++a) weight+=ko_taper[start+a]/5;
        if (weight==0) continue;
        for (int f=0;f<NF;++f) {
          double difference=0;
          for (int a=0;a<5;++a) difference+=c[a]*(s[start+a][f]-s[start][f]);
          forcing.dissipation[f]+=-o.ko*c[target]*weight*difference/(256*o.hh);
        }
      }
    }
    const auto result=h::EvaluateBulkPoint(map,jets,p,src[i],forcing);
    if (result.status!=h::PointStatus::success) {
      std::fprintf(stderr,"bulk status %d at r=%.6f\n",int(result.status),r[i]);
      throw std::runtime_error("invalid bulk point");
    }
    for (int f=0;f<NF;++f) ds[i][f]=h::Fields::Independent(f) ? mask[i]*result.rate[f] : 0;
  }

  void Step(double dt) {
    std::vector<Row> k1(N+1),k2(N+1),k3(N+1),k4(N+1),stage=u;
    RHS(u,k1);
    for (int i=0;i<=N;++i) for (int f=0;f<NF;++f) stage[i][f]=u[i][f]+.5*dt*k1[i][f];
    RHS(stage,k2);
    for (int i=0;i<=N;++i) for (int f=0;f<NF;++f) stage[i][f]=u[i][f]+.5*dt*k2[i][f];
    RHS(stage,k3);
    for (int i=0;i<=N;++i) for (int f=0;f<NF;++f) stage[i][f]=u[i][f]+dt*k3[i][f];
    RHS(stage,k4);
    for (int i=0;i<=N;++i) for (int f=0;f<NF;++f)
      u[i][f]+=dt*(k1[i][f]+2*k2[i][f]+2*k3[i][f]+k4[i][f])/6;
    for (int i=0;i<=N;++i) Enforce(u[i]);
    Project(u);
  }

  void WriteProfile(double time,int index) const {
    std::vector<Row> s=u,ds(N+1);
    RHS(s,ds);
    char name[64];std::snprintf(name,sizeof name,"/profile-%03d.csv",index);
    std::ofstream out(o.out+name);out.precision(17);
    out<<"# time="<<time<<"\nr";
    for (int f:kRadialFields) out<<",u"<<f;
    for (int f:kRadialFields) out<<",res"<<f;
    for (int f:kRadialFields) out<<",rate"<<f;
    out<<",hamiltonian,momentum\n";
    for (int i=frozen;i<=N;++i) {
      out<<r[i];
      for (int f:kRadialFields) out<<','<<s[i][f];
      for (int f:kRadialFields) out<<','<<s[i][f]-ref[i][f];
      for (int f:kRadialFields) out<<','<<ds[i][f];
      double ham=NAN,mom=NAN;
      if (i<N) {
        const double x[3]={r[i],0,0};
        const Jets jets=BulkJets(s,i,0.,nullptr);
        const auto physical=h::SphericalPolicy::Physical(x,jets);
        auto omega=p.profile.At(x);
        h::SetStationaryOmegaNormal(physical.alpha.value,physical.beta.value,
                                    physical.alpha.d,physical.beta.d,omega);
        const auto constraints=h::EvolvedConstraints(physical,omega);
        ham=constraints.hamiltonian;mom=constraints.momentum[0];
      }
      out<<','<<ham<<','<<mom<<'\n';
    }
  }

  // Differentiate the reference's own radial functions numerically and compare
  // the composed Cartesian jets with the analytic reference jets.
  void TestJets() const {
    double error[3]{},where[3]{};int field[3]{};
    for (int i=8;i<=N;++i) {
      if (i>N-8 && i<N-3) continue;
      const double x[3]={r[i],0,0};
      std::array<R3,NF> rq{};
      const auto value=[&](int q,int f) {return f==0 && q<N ? ref[q][f]/(r[q]*r[q]) : ref[q][f];};
      for (int f:kRadialFields) {
        R3 q;q.v=value(i,f);
        if (i>=N-3) {
          for (int k=0;k<o.endpoint_points;++k) {
            const double delta=value(N-(o.endpoint_points-1)+k,f)-q.v;
            q.d1+=w1[i-(N-3)][k]*delta;q.d2+=w2[i-(N-3)][k]*delta;
          }
        } else {
          q.d1=(value(i-2,f)-8*value(i-1,f)+8*value(i+1,f)-value(i+2,f))/(12*o.hh);
          q.d2=(-value(i-2,f)+16*value(i-1,f)-30*q.v+16*value(i+1,f)-value(i+2,f))/(12*o.hh*o.hh);
        }
        rq[f]=q;
      }
      Jets jets=CartesianFromRadial(x,rq,false);
      J r2;for (int a=0;a<3;++a) {const J z=J::Variable(x[a],a);r2=r2+z*z;}
      jets[0]=jets[0]*r2;
      if (!h::TryExpandCartesianChart(jets)) throw std::runtime_error("test chart");
      for (int f=0;f<28;++f) {
        const auto note=[&](int k,double e) {if (e>error[k]) {error[k]=e;where[k]=r[i];field[k]=f;}};
        note(0,fabs(jets[f].f-refj[i][f].f));
        for (int a=0;a<3;++a) {
          note(1,fabs(jets[f].d[a]-refj[i][f].d[a]));
          for (int b=0;b<3;++b) note(2,fabs(jets[f].h[a][b]-refj[i][f].h[a][b]));
        }
      }
    }
    std::printf("jet-test h=%g value=%.3e first=%.3e (f%d r=%.4f) second=%.3e (f%d r=%.4f)\n",o.hh,
                error[0],error[1],field[1],where[1],error[2],field[2],where[2]);
  }
};

int main(int argc,char **argv) {
  Options o;
  for (int a=1;a<argc;++a) {
    const std::string key=argv[a];
    const auto next=[&]() {if (a+1>=argc) throw std::invalid_argument("missing value");return std::string(argv[++a]);};
    if (key=="--h") o.hh=std::stod(next());
    else if (key=="--endpoint-points") o.endpoint_points=std::stoi(next());
    else if (key=="--rin") o.rin=std::stod(next());
    else if (key=="--tend") o.tend=std::stod(next());
    else if (key=="--dtfac") o.dtfac=std::stod(next());
    else if (key=="--amp") o.amp=std::stod(next());
    else if (key=="--center") o.center=std::stod(next());
    else if (key=="--width") o.width=std::stod(next());
    else if (key=="--profile") o.profile=std::stoi(next());
    else if (key=="--restoring") o.restoring=std::stod(next());
    else if (key=="--kappa") o.kappa=std::stod(next());
    else if (key=="--penalty") o.penalty=std::stod(next());
    else if (key=="--ko") o.ko=std::stod(next());
    else if (key=="--stride") o.stride=std::stoi(next());
    else if (key=="--threads") o.threads=std::stoi(next());
    else if (key=="--lift-layers") o.lift_layers=std::stoi(next());  // diagnostic only
    else if (key=="--out") o.out=next();
    else if (key=="--test-jets") o.test_jets=true;
    else if (key=="--profiles") {
      std::stringstream list(next());std::string item;
      while (std::getline(list,item,',')) o.profile_times.push_back(std::stod(item));
    } else throw std::invalid_argument("unknown option "+key);
  }
  Sim sim(o);
  if (o.test_jets) {sim.TestJets();return 0;}
  sim.AddPulse();
  const double dt=o.dtfac*o.hh;
  const int steps=static_cast<int>(std::ceil(o.tend/dt-1e-9));
  std::ofstream series(o.out+"/scri.csv");series.precision(17);
  series<<"step,time,clock,speed_t,defect,lift,alpha_n,P_n,beta_nn,speed,alpha_nn,"
          "pen_chi,pen_P,pen_theta,chi,P,theta,raw_alpha_n,raw_alpha_nn,raw_beta_nn,"
          "T_max,T00,T0x,Txx,Tyy,Tzz,box,nt\n";
  {
    std::ofstream meta(o.out+"/meta.txt");meta.precision(17);
    meta<<"h="<<o.hh<<"\nN="<<sim.N<<"\nrin="<<sim.r[0]<<"\ndt="<<dt<<"\nsteps="<<steps
        <<"\namp="<<o.amp<<"\ncenter="<<o.center<<"\nwidth="<<o.width<<"\nprofile="<<o.profile
        <<"\nendpoint_points="<<o.endpoint_points<<"\nthreads="<<o.threads<<"\nstride="<<o.stride<<"\nlift_layers="<<o.lift_layers
        <<"\nkappa="<<o.kappa<<"\npenalty="<<o.penalty<<"\nko="<<o.ko
        <<"\ninner_begin="<<sim.p.inner_begin<<"\ninner_end="<<sim.p.inner_end
        <<"\nflat_radius="<<sim.p.profile.flat_radius<<"\nrestoring="<<sim.p.restoring
        <<"\nratio="<<sim.p.ratio<<"\nbulk_mu="<<sim.p.bulk_mu<<"\nkappa1="<<sim.p.kappa1
        <<"\nkappa2="<<sim.p.kappa2<<'\n';
  }
  size_t next_profile=0;
  std::vector<Row> scratch,rates(sim.N+1);
  for (int step=0;step<=steps;++step) {
    const double time=step*dt;
    if (step%o.stride==0 || step==steps) {
      Diagnostic d;scratch=sim.u;sim.RHS(scratch,rates,&d);
      const auto &T=d.tensor.T;
      series<<step<<','<<time<<','<<d.clock<<','<<d.speed_t<<','<<d.defect<<','<<d.lift<<','
            <<d.alpha_n<<','<<d.P_n<<','<<d.beta_nn<<','<<d.speed<<','<<d.alpha_nn<<','
            <<d.pen_chi<<','<<d.pen_P<<','<<d.pen_theta<<','<<d.chi<<','<<d.P<<','<<d.theta<<','
            <<d.raw_alpha_n<<','<<d.raw_alpha_nn<<','<<d.raw_beta_nn<<','
            <<d.tensor.max<<','<<T[0][0]<<','<<T[0][1]<<','<<T[1][1]<<','<<T[2][2]<<','<<T[3][3]<<','
            <<d.tensor.box<<','<<d.tensor.nt<<'\n';
    }
    while (next_profile<o.profile_times.size() && time>=o.profile_times[next_profile]-.5*dt) {
      sim.WriteProfile(time,static_cast<int>(next_profile));++next_profile;
    }
    if (step<steps) sim.Step(dt);
  }
  std::printf("done N=%d steps=%d t=%.6f\n",sim.N,steps,steps*dt);
  return 0;
}
