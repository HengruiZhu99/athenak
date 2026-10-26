// Compiled copy of the audited g=1 regular-center RHS. No fast-math,
// constraint projection, equilibrium subtraction, or Omega floor.
#include <algorithm>
#include <array>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <chrono>
#include <limits>
#include <string>
using Vec=std::vector<double>;
struct Engine {
  int n,nf,w,qw; double h; Vec r,d1,d2,qweights,params;
  std::vector<int> index,qindex;
  std::vector<int> aleft,aright;Vec awleft,awright;
  Vec d,dd,R,S,beta,v,phase,U,akr,z,k1,k2,k3,k4,stage,lapse_source,primitive,total_chi,total_alpha,af_reference,chart_d,total_khat;
  long calls=0; double solve_seconds=0;
  bool puncture=false,af=false;
  std::string error;
  std::array<double,22> diagnostics{};
  Vec ode_lu,ode_matrix;int ode_band=0,ode_skip=0,ode_refinements=0;double ode_scale=1;
  Engine(int N,int F,int W,int QW,const double* rr,const int* ix,
      const double* D1,const double* D2,const int* qi,const double* qwt,
      const int* al,const double* alw,const int* ar,const double* arw,const double* p):n(N),nf(F),w(W),qw(QW),h(rr[1]-rr[0]),r(rr,rr+N),
      d1(D1,D1+N*W),d2(D2,D2+N*W),qweights(qwt,qwt+((rr[0]==0||p[12]!=0)?N-1:N)*QW),
      params(p,p+27),index(ix,ix+N*W),qindex(qi,qi+((rr[0]==0||p[12]!=0)?N-1:N)*QW),
      aleft(al,al+N*W),aright(ar,ar+N*W),awleft(alw,alw+N*W),awright(arw,arw+N*W) {
    for(Vec* a:{&d,&dd,&R,&S,&z,&k1,&k2,&k3,&k4,&stage})a->resize(n*nf);
    for(Vec* a:{&beta,&v,&phase,&U,&akr})a->resize(n);
    lapse_source.resize(n,0.);
    primitive.resize(n*nf);total_chi.resize(n);total_alpha.resize(n);
    chart_d.resize(n*nf);total_khat.resize(n);af=p[24]!=0;
    if(nf!=8&&nf!=10)throw std::runtime_error("invalid field count");
    diagnostics[3]=diagnostics[4]=std::numeric_limits<double>::infinity();
    puncture=p[12]!=0;if(puncture&&!af)diagnostics[11]=-std::numeric_limits<double>::infinity();
  }
  int reflected(int k)const{return k<w?(r[0]==0?w-k:w-1-k):k-w;}
  int parity(int f)const{return f==5?-1:1;}
  void derivatives(const double* q) {
    std::fill(d.begin(),d.end(),0);std::fill(dd.begin(),dd.end(),0);
    for(int i=0;i<n;i++)for(int j=0;j<w;j++) {
      int k=index[i*w+j],ii=reflected(k);
      for(int f=0;f<nf;f++) {
        double val=q[ii*nf+f]*(k<w?parity(f):1);
        if(puncture||params[23]!=0)val-=q[i*nf+f];
        d[i*nf+f]+=d1[i*w+j]*val;dd[i*nf+f]+=d2[i*w+j]*val;
      }
    }
    if(r[0]==0&&!puncture)for(int f=0;f<nf;f++) {
      if(parity(f)>0)d[f]=0;else dd[f]=0;
    }
  }
  double endpoint_derivative(const Vec& a,int f)const {
    double sum=0;
    for(int j=0;j<w;j++)sum+=d1[(n-1)*w+j]*(a[reflected(index[(n-1)*w+j])*nf+f]-((puncture||params[23]!=0)?a[(n-1)*nf+f]:0));
    return sum;
  }
  double ref(int i,int field,int derivative=0)const {return af_reference[(i*6+field)*3+derivative];}
  double ratio(double a,int i,double limit=0)const{return r[i]>0?a/r[i]:limit;}
  Vec ode_triangular(Vec x)const {
    const int b=ode_band,stride=2*b+1;
    for(int i=0;i<n;i++)for(int j=std::max(0,i-b);j<i;j++)x[i]-=ode_lu[i*stride+b+j-i]*x[j];
    for(int i=n-1;i>=0;i--) {
      for(int j=i+1;j<std::min(n,i+b+1);j++)x[i]-=ode_lu[i*stride+b+j-i]*x[j];
      x[i]/=ode_lu[i*stride+b];
    }
    return x;
  }
  Vec ode_residual(const Vec& rhs,const Vec& x)const {
    Vec residual(n);const int b=ode_band,stride=2*b+1;
    for(int i=0;i<n;i++) {
      long double value=rhs[i];
      for(int j=std::max(0,i-b);j<std::min(n,i+b+1);j++)value-=static_cast<long double>(ode_matrix[i*stride+b+j-i])*x[j];
      residual[i]=static_cast<double>(value);
    }
    return residual;
  }
  Vec solve_shift_ode(double outer) {
    Vec rhs(n);for(int i=0;i<n-1;i++)rhs[i]=ode_scale*z[i+ode_skip];rhs.back()=outer;
    Vec x=ode_triangular(rhs);
    for(int pass=0;pass<ode_refinements;pass++) {
      Vec correction=ode_triangular(ode_residual(rhs,x));for(int i=0;i<n;i++)x[i]+=correction[i];
    }
    Vec residual=ode_residual(rhs,x);
    for(int i=0;i<n-1;i++)diagnostics[21]=std::max(diagnostics[21],std::abs(residual[i])/ode_scale);
    return x;
  }
  double lapse_coefficient(double x,double alpha)const {
    double O=(1-x*x)/2,A=(1+x*x)/2,nu=params[2];int family=static_cast<int>(params[18]);
    if(family==1)return alpha*alpha+4*nu*O*O*alpha*(A+params[19])/(alpha+params[19]);
    if(family==2)return (1-4*O*O)*alpha*alpha+8*nu*O*O*O*alpha;
    return alpha*alpha+4*nu*O*O*alpha;
  }
  double rhs(const double* state,double* out,bool accepted=false) {
    derivatives(state);calls++;
    const bool logchi=params[16]!=0,logalpha=params[17]!=0;
    const double* q=state;
    if(af) {
      if(af_reference.size()!=static_cast<size_t>(n*18))throw std::runtime_error("AF reference not supplied");
      chart_d=d;std::copy(state,state+n*nf,primitive.begin());q=primitive.data();
      for(int i=0;i<n;i++) {
        int k=i*nf;double ec=std::exp(state[k]),ea=std::exp(state[k+6]);
        primitive[k]=ref(i,3)+ref(i,0)*std::expm1(state[k]);
        primitive[k+6]=ref(i,4)+ref(i,2)*std::expm1(state[k+6]);
        total_khat[i]=ref(i,1)*(state[k+3]-3);
        for(int f:{0,6}) {
          int j=f==0?0:2;double ex=f==0?ec:ea;double first=d[k+f],second=dd[k+f];
          if(f==0) {d[k+f]=ex*(ref(i,j,1)+ref(i,j)*first);
            dd[k+f]=ex*(ref(i,j,2)+2*ref(i,j,1)*first+ref(i,j)*(second+first*first));}
          else {double small=std::expm1(state[k+f]);
            d[k+f]=ref(i,4,1)+ref(i,j,1)*small+ref(i,j)*ex*first;
            dd[k+f]=ref(i,4,2)+ref(i,j,2)*small+ex*(2*ref(i,j,1)*first+ref(i,j)*(second+first*first));}
        }
        for(int f:{2,3,4}) {
          double value=state[k+f]-(f==3?3:0),first=d[k+f];
          primitive[k+f]=f==3?3*ref(i,5)+ref(i,1)*state[k+f]:ref(i,1)*state[k+f];
          dd[k+f]=ref(i,1,2)*value+2*ref(i,1,1)*first+ref(i,1)*dd[k+f];
          d[k+f]=ref(i,1,1)*value+ref(i,1)*first;
        }
        total_chi[i]=ref(i,0)*ec;total_alpha[i]=ref(i,2)*ea;
        if(!std::isfinite(ec)||!std::isfinite(ea))throw std::runtime_error("nonfinite AF logarithm");
      }
    } else if(logchi||logalpha) {
      std::copy(state,state+n*nf,primitive.begin());q=primitive.data();
      for(int i=0;i<n;i++) {
        int k=i*nf;double A=(1+r[i]*r[i])/2;
        if(logchi) {
          double chi=std::exp(state[k]);primitive[k]=std::expm1(state[k]);
          dd[k]=chi*(dd[k]+d[k]*d[k]);d[k]*=chi;
        }
        if(logalpha) {
          double factor=std::exp(state[k+6]),small=std::expm1(state[k+6]);primitive[k+6]=A*small;
          dd[k+6]=small+factor*(2*r[i]*d[k+6]+A*(dd[k+6]+d[k+6]*d[k+6]));
          d[k+6]=r[i]*small+A*factor*d[k+6];
        }
      }
    }
    if(!af)for(int i=0;i<n;i++) {
      total_chi[i]=logchi?std::exp(state[i*nf]):1+q[i*nf];
      total_alpha[i]=logalpha?(1+r[i]*r[i])/2*std::exp(state[i*nf+6]):(1+r[i]*r[i])/2+q[i*nf+6];
    }
    double kap=params[0],kap2=params[1],xi=params[3];
    double sat=params[4],tau=params[5],ko=params[6],radius=params[7];
    bool appendix=params[8]!=0;
    auto solve_start=std::chrono::steady_clock::now();
    for(int i=0;i<n;i++) {
      double chi=total_chi[i],alpha=total_alpha[i];
      if(af&&i==0){z[i]=0;continue;}
      diagnostics[3]=std::min(diagnostics[3],chi);diagnostics[4]=std::min(diagnostics[4],alpha);
      if(q[i*nf+1]!=0||chi<=0||alpha<=0||!std::isfinite(chi)||!std::isfinite(alpha))
        throw std::runtime_error("g=1 gauge or positivity violation");
      z[i]=1.5*alpha*ratio(q[i*nf+2],i);
    }
    Vec integral(n,0);double sum=0;bool first_node=r[0]==0||puncture;int intervals=(first_node?n-1:n);
    Vec increments(intervals,0);
    for(int i=0;i<intervals;i++) {
      double increment=0;
      for(int j=0;j<qw;j++) {
        int k=qindex[i*qw+j];double value=(first_node?z[k]:(k==0?0:z[k-1]));
        increment+=qweights[i*qw+j]*value;
      }
      increments[i]=increment;sum+=increment;integral[first_node?i+1:i]=sum;
    }
    bool inward=params[20]!=0;
    if(inward) {
      integral.back()=0;
      for(int i=n-2;i>=0;i--)integral[i]=integral[i+1]+increments[i+(first_node?0:1)];
    }
    int e=(n-1)*nf;double cS=q[e],uS=q[e+6],alphaS=total_alpha.back();
    double outer=-(cS+(2*uS+uS*uS)*total_chi.back())/(1+alphaS*std::sqrt(total_chi.back()));
    if(ode_band)phase=solve_shift_ode(outer);
    for(int i=0;i<n;i++) {
      if(!ode_band)phase[i]=inward?outer-integral[i]:outer+integral[i]-integral.back();
      v[i]=r[i]*phase[i];beta[i]=-r[i]+v[i];
      double alpha=total_alpha[i];
      U[i]=alpha*(q[i*nf+3]+2*q[i*nf+4])/3-q[i*nf+6]-r[i]*v[i];
    }
    solve_seconds+=std::chrono::duration<double>(std::chrono::steady_clock::now()-solve_start).count();
    if(puncture&&!af) {
      double alpha=total_alpha[0],chi=total_chi[0];
      double F=lapse_coefficient(r[0],alpha);
      diagnostics[11]=std::max(diagnostics[11],std::max(-beta[0]+alpha*std::sqrt(chi),-beta[0]+std::sqrt(F*chi)));
      if(params[14]!=0&&diagnostics[11]>0)throw std::runtime_error("puncture inner characteristic became incoming");
    }
    double dU=0;for(int j=0;j<w;j++)dU+=d1[(n-1)*w+j]*(U[reflected(index[(n-1)*w+j])]-((puncture||params[23]!=0)?U.back():0));
    double analytic_dU=(1+d[e+6])*(q[e+3]+2*q[e+4])/3+alphaS*(d[e+3]+2*d[e+4])/3
        -d[e+6]-2*v.back()-1.5*alphaS*q[e+2];
    if(params[9]!=0)dU=analytic_dU;
    std::fill(R.begin(),R.end(),0);std::fill(S.begin(),S.end(),0);
    std::array<double,4> normsum{};double volume=0;
    double bulkH=0,bulkM=0,endH=0,endM=0;
    for(int i=af?1:0;i<n;i++) {
      int k=i*nf;double x=r[i],O=(1-x*x)/2,A0=(1+x*x)/2;
      double c=q[k],chi=total_chi[i],Ar=q[k+2],p=af?total_khat[i]:q[k+3]-3,T=q[k+4],lam=q[k+5],u=q[k+6];
      double cp=d[k],cpp=dd[k],ap=d[k+2],pp=d[k+3],Tp=d[k+4],lp=d[k+5],up=d[k+6],upp=dd[k+6];
      double alpha=total_alpha[i],alphap=x+up,K=p+2*T,dk=q[k+3]+2*T,b=beta[i],vv=v[i],vr=phase[i];
      double ar=ratio(Ar,i),cr=ratio(cp,i,cpp),ur=ratio(up,i,upp),lr=ratio(lam,i,lp);
      double ww=b*x/alpha,divb=-3+3*vr+1.5*alpha*Ar;
      R[k]=b*cp-2*chi*vr-alpha*chi*Ar;
      S[k]=2*chi/3*(alpha*dk-3*u-3*x*vv);
      R[k+2]=b*ap-2*chi/3*(upp-ur)-2*cp*alphap/3+alpha/3*(cpp-cr)-alpha*cp*cp/(6*chi)
        +2*alpha*chi/3*(lp-lr)+2*alpha*cp*lam/3;
      S[k+2]=-4*alpha*x/3*(cp+chi*lam)+alpha*Ar*(K-ww);
      R[k+3]=b*pp+1.5*O*alpha*Ar*Ar-O*chi*(upp+2*ur)+.5*O*cp*alphap
        -3*chi*x*up-6*A0*c+3*u-3*c*u+.5*alpha*x*cp;
      S[k+3]=alpha*(dk*(K-3)/3-3*c*x*x+kap*(1-kap2)*T);
      R[k+4]=b*Tp+alpha*x*cp+O*alpha/2*(2*cpp+4*cr-2.5*cp*cp/chi-1.5*Ar*Ar+chi*(lp+2*lr));
      S[k+4]=alpha*(dk*(K-3)/3-3*c-kap*(2+kap2)*T);
      if(!appendix)S[k+4]-=3*alpha*ww*T;
      R[k+5]=b*lp+2*lam*divb/3+2*alpha*(ap+3*ar-1.5*Ar*cp/chi);
      S[k+5]=-alpha*(2*K/3+kap)*lam-2*alpha/3*(2*pp+Tp)+4*alpha*Ar*x;
      double F=lapse_coefficient(x,alpha);
      double local_xi=xi;
      if(params[22]!=0) {
        double nu_inner=params[2];if(static_cast<int>(params[18])==1)nu_inner*=(.5+params[19])/params[19];
        local_xi+=(3*nu_inner-xi)*16*O*O*O*O;
      }
      R[k+6]=b*up+vv*x;
      S[k+6]=-F*q[k+3]+x*(A0*vv-x*u+u*vv)-local_xi*(params[13]!=0?2*alpha:alpha+A0)*u;
      if(af&&x<.5) {
        S[k]=2*chi/3*(alpha*K+3*A0-3*x*vv);
        R[k+3]=b*pp+1.5*O*alpha*Ar*Ar-O*chi*(upp+2*ur)+.5*O*cp*alphap
          -3*chi*x*up-3*chi*(A0+alpha)+.5*alpha*x*cp;
        S[k+3]=alpha*(K*K/3-3*chi*x*x+kap*(1-kap2)*T);
        S[k+4]=alpha*(K*K/3-3*chi-kap*(2+kap2)*T);
        if(!appendix)S[k+4]-=3*alpha*ww*T;
        R[k+6]=b*alphap;
        S[k+6]=-F*q[k+3]+x*alpha*b+x*x-2*local_xi*alpha*u;
      }
      if(params[21]!=0)S[k+6]+=16*O*O*O*O*x*x*u/A0;
      if(params[15]!=0) {
        double D=-kap*u;S[k+3]+=D*(1-kap2)*T;S[k+4]-=D*(2+kap2)*T;S[k+5]-=D*lam;
      }
      double H=2*dk*(K-3)/3-6*c+2*O*x*cp+O*O*(2*cpp+4*cr-2.5*cp*cp/chi-1.5*Ar*Ar);
      if(af&&x<.5)H=2*K*K/3-6*chi+2*O*x*cp+O*O*(2*cpp+4*cr-2.5*cp*cp/chi-1.5*Ar*Ar);
      double M=O*(ap+3*ar-1.5*Ar*cp/chi)-2*(pp+2*Tp)/3+2*x*Ar;
      if(nf==10) {
        constexpr double pi=3.1415926535897932384626433832795;
        double f=q[k+8],P=q[k+9],fp=d[k+8],Pp=d[k+9],fpp=dd[k+8];
        double phir=std::sqrt(chi)*(O*fp+(-x+.5*O*cp/chi)*f);
        double normal=std::sqrt(chi)*(O*P+(K/3-O*Ar/2)*f);
        R[k+2]-=16*pi/3*alpha*chi*phir*phir;
        R[k+3]+=8*pi*alpha*O*normal*normal;
        R[k+4]-=4*pi*alpha*O*(normal*normal+chi*phir*phir);
        R[k+5]+=16*pi*alpha*normal*phir;
        H-=8*pi*O*O*(normal*normal+chi*phir*phir);M+=8*pi*O*normal*phir;
        double ak=-1+alpha*Ar+(i==n-1?-dU:U[i]/O);
        double N=dk*(K-3)/9-c+O*(-K*Ar/3+A0*cr)+O*O*(Ar*Ar/4-cp*cp/(4*chi));
        double B=N-kap*(1+kap2)*T+2*O*x*chi*lam/3;
        if(params[15]!=0)B-=kap*(1+kap2)*(-u)*(T/alpha);
        if(!appendix)B-=2*ww*T;
        double V=i==n-1?0:B/(O*O)+chi*lr-cp*lam/3;
        double br=-1+vr,mix=.5*alpha*cp+chi*alphap;
        double mixlimit=.5*alpha*cpp+chi*(1+upp);
        R[k+8]=b*fp+br*f+alpha*P;
        R[k+9]=b*Pp+(br+ak)*P+alpha*chi*(fpp+2*ratio(fp,i,fpp))+mix*fp+(ratio(mix,i,mixlimit)-alpha*V)*f;
      }
      normsum[0]+=x*x*H*H;normsum[1]+=x*x*M*M;
      normsum[2]+=x*x*T*T;normsum[3]+=x*x*lam*lam/4;volume+=x*x;
      if(accepted&&x<.8){bulkH=std::max(bulkH,std::abs(H));bulkM=std::max(bulkM,std::abs(M));}
      if(accepted&&x>.9){endH=std::max(endH,std::abs(H));endM=std::max(endM,std::abs(M));}
    }
    for(int f=0;f<4;f++)diagnostics[5+f]=std::max(diagnostics[5+f],std::sqrt(normsum[f]/volume));
    if(accepted) {
      for(int f=0;f<4;f++)diagnostics[13+f]=std::max(diagnostics[13+f],std::sqrt(normsum[f]/volume));
      diagnostics[17]=std::max(diagnostics[17],bulkH);diagnostics[18]=std::max(diagnostics[18],bulkM);
      diagnostics[19]=std::max(diagnostics[19],endH);diagnostics[20]=std::max(diagnostics[20],endM);
    }
    if(params[11]!=0&&!af)for(int i=0;i<n;i++) {
      const auto& ai=beta[i]<0?aleft:aright;const auto& aw=beta[i]<0?awleft:awright;
      for(int f=0;f<nf;f++)if(f!=7) {
        double adv=0;for(int j=0;j<w;j++) {
          int k=ai[i*w+j],ii=reflected(k);double value=q[ii*nf+f]*(k<w?parity(f):1);
          if(puncture||params[23]!=0)value-=q[i*nf+f];adv+=aw[i*w+j]*value;
        }
        R[i*nf+f]+=beta[i]*(adv-d[i*nf+f]);
      }
    }
    for(int i=0;i<n;i++)S[i*nf+6]+=lapse_source[i];
    for(int i=0;i<n-1;i++)for(int f=0;f<nf;f++)out[i*nf+f]=R[i*nf+f]+S[i*nf+f]/((1-r[i]*r[i])/2);
    std::array<double,10> trace{};
    trace[0]=q[e];trace[2]=q[e+2]+2*d[e]/3;trace[3]=q[e+3];trace[4]=q[e+4];trace[5]=q[e+5];
    for(int f=0;f<6;f++) {
      diagnostics[0]=std::max(diagnostics[0],std::abs(S[e+f]));
      diagnostics[2]=std::max(diagnostics[2],std::abs(trace[f]));
    }
    diagnostics[1]=std::max(diagnostics[1],std::abs(S[e+6]));
    diagnostics[9]=std::max(diagnostics[9],std::abs(1+cS-beta.back()*beta.back()/(alphaS*alphaS)));
    diagnostics[10]=std::max(diagnostics[10],std::abs(uS));
    std::array<double,10> ds{};
    for(int f=0;f<nf;f++)ds[f]=endpoint_derivative(S,f);
    auto sampled_ds=ds;
    if(params[9]!=0||params[10]!=0) {
      double c=q[e],chi=total_chi.back(),Ar=q[e+2],T=q[e+4],lam=q[e+5],u=q[e+6];
      double cp=d[e],ap=d[e+2],pp=d[e+3],Tp=d[e+4],lp=d[e+5],up=d[e+6];
      double alpha=total_alpha.back(),alphap=1+up,K=q[e+3]-3+2*T,dk=q[e+3]+2*T,Kp=pp+2*Tp;
      double vv=v.back(),vp=vv+1.5*alpha*Ar,b=-1+vv,bp=-1+vp,ww=b/alpha;
      double wp=(bp+b)/alpha-ww*alphap/alpha;
      ds[0]=2*(cp*U.back()+chi*analytic_dU);
      ds[2]=-4./3*((alphap+alpha)*(cp+chi*lam)+alpha*(dd[e]+cp*lam+chi*lp))
        +(alphap*Ar+alpha*ap)*(K-ww)+alpha*Ar*(Kp-wp);
      double Y=dk*(K-3)/3-3*c+kap*(1-kap2)*T;
      double Yp=2*K*Kp/3-3*cp-6*c+kap*(1-kap2)*Tp;
      ds[3]=alphap*Y+alpha*Yp;
      double Z=dk*(K-3)/3-3*c-kap*(2+kap2)*T;
      double Zp=2*K*Kp/3-3*cp-kap*(2+kap2)*Tp;
      if(!appendix){Z-=3*ww*T;Zp-=3*(wp*T+ww*Tp);}
      ds[4]=alphap*Z+alpha*Zp;
      ds[5]=-alphap*(2*K/3+kap)*lam-alpha*(2*Kp*lam/3+(2*K/3+kap)*lp)
        -2./3*(alphap*(2*pp+Tp)+alpha*(2*dd[e+3]+dd[e+4]))+4*(alphap*Ar+alpha*ap+alpha*Ar);
      ds[6]=-2*alpha*alphap*q[e+3]-alpha*alpha*pp+(vv-u+u*vv)
        +(vv+vp-u-up+up*vv+u*vp)-xi*((alphap+1)*u+(alpha+1)*up);
      if(params[13]!=0)ds[6]-=2*xi*u*up;
      if(params[15]!=0) {
        ds[3]-=kap*(1-kap2)*(up*T+u*Tp);ds[4]+=kap*(2+kap2)*(up*T+u*Tp);
        ds[5]+=kap*(up*lam+u*lp);
      }
    }
    if(params[10]!=0) {
      double rawchi=R[e]-ds[0],rawclock=R[e+6]-ds[6];
      double Jchi=beta.back()-2*U.back(),Jp=-2*(1+cS)*alphaS/3;
      double prate=-params[10]/h*rawclock/(alphaS*alphaS);
      double crate=(-params[10]/h*rawchi-Jp*prate)/Jchi;
      double lift=1/d1[(n-1)*w+w-2];
      out[(n-2)*nf]+=crate*lift;out[(n-2)*nf+3]+=prate*lift;
    }
    if(params[9]==0)ds=sampled_ds;
    for(int f=0;f<nf;f++)out[e+f]=R[e+f]-ds[f]+sat/h*S[e+f]-tau/h*trace[f];
    if(logchi||logalpha)for(int i=0;i<n;i++) {
      if(logchi)out[i*nf]/=total_chi[i];
      if(logalpha)out[i*nf+6]/=total_alpha[i];
    }
    if(af) {
      for(int i=1;i<n;i++) {
        out[i*nf]/=total_chi[i];out[i*nf+6]/=total_alpha[i];
        for(int f:{2,3,4})out[i*nf+f]/=ref(i,1);
      }
      double b1=-1+phase[0],nu=params[2],inner_xi=xi;
      if(static_cast<int>(params[18])==1)nu*=(.5+params[19])/params[19];
      if(params[22]!=0)inner_xi=3*nu;
      out[0]=2*b1;out[6]=2*b1-6*nu+2*inner_xi;
      if(params[21]==0)out[6]+=2/(8*std::exp(state[6]));
      out[2]=2*b1*state[2];out[3]=2*b1*(state[3]-3);out[4]=2*b1*state[4];out[5]=2*b1*state[5];
      if(params[15]!=0){out[3]+=kap*(1-kap2)*state[4];out[4]-=kap*(2+kap2)*state[4];out[5]-=kap*state[5];}
      if(params[11]!=0)for(int i=0;i<n;i++) {
        const auto& ai=beta[i]<0?aleft:aright;const auto& aw=beta[i]<0?awleft:awright;
        for(int f=0;f<nf;f++) {
          double adv=0;for(int j=0;j<w;j++)adv+=aw[i*w+j]*(state[reflected(ai[i*w+j])*nf+f]-state[i*nf+f]);
          out[i*nf+f]+=beta[i]*(adv-chart_d[i*nf+f]);
        }
      }
      if(params[25]!=0) {
        std::fill(z.begin(),z.end(),0);
        for(int i=1;i<n-3;i++) {
          double midpoint=(r[i]+r[i+3])/2,taper=std::pow(std::max(1-midpoint*midpoint/(params[26]*params[26]),0.),6);
          for(int f=0;f<nf;f++) {
            double anchor=state[i*nf+f];
            double third=3*(state[(i+1)*nf+f]-anchor)-3*(state[(i+2)*nf+f]-anchor)+(state[(i+3)*nf+f]-anchor);
            double value=taper*third;z[i*nf+f]+=value;z[(i+1)*nf+f]-=3*value;z[(i+2)*nf+f]+=3*value;z[(i+3)*nf+f]-=value;
          }
        }
        for(int k=0;k<n*nf;k++)out[k]+=params[25]*z[k]/(64*h);
      }
      for(int i=0;i<n;i++)out[i*nf+1]=0;
    }
    if(ko!=0) {
      std::copy(state,state+n*nf,z.begin());
      for(int i=0;i<n;i++)if(r[i]>0) {z[i*nf+2]/=r[i]*r[i];z[i*nf+5]/=r[i];}
      if(r[0]==0){z[2]=dd[2]/2;z[5]=d[5];}
      constexpr int coefficients[7]={1,-6,15,-20,15,-6,1};
      for(int i=0;i<n;i++) {
        double taper=std::pow(std::max(1-(r[i]/radius)*(r[i]/radius),0.),6);
        if(taper==0)continue;
        for(int f=0;f<nf;f++) {
          if(f==1||f==7)continue;
          double val=0;for(int j=0;j<7;j++) {
            int ii=i+j-3;double sign=1;
            if(ii<0){ii=r[0]==0?-ii:-ii-1;if(f==5)sign=1;else sign=parity(f);}
            if(ii<n)val+=coefficients[j]*sign*z[ii*nf+f];
          }
          if(f==2)val*=r[i]*r[i];if(f==5)val*=r[i];
          out[i*nf+f]+=ko*taper*val/(64*h);
        }
      }
    }
    out[e+6]=0;for(int i=0;i<n;i++)out[i*nf+7]=0;
    const double chiS=total_chi.back(),rootChi=std::sqrt(chiS);
    const double dbS=beta.back()+1.5*alphaS*q[e+2];
    const double affinity=out[e]*((logchi||af)?chiS:1)/(2*alphaS*chiS)-d[e]/(2*rootChi)
        -rootChi*(1+d[e+6])/alphaS-dbS/alphaS;
    diagnostics[12]=std::max(diagnostics[12],std::abs(affinity));
    for(int k=0;k<n*nf;k++)if(!std::isfinite(out[k]))throw std::runtime_error("nonfinite RHS");
    return nf==10?4*3.14159265358979323846*R[e+8]*R[e+8]:0;
  }
};
extern "C" {
int spherical_fast_abi(){return 26;}
void spherical_fast_set_shift_ode(void* handle,int bandwidth,int skip,int refinements,double scale,const double* lu,const double* matrix){
  auto& e=*static_cast<Engine*>(handle);e.ode_band=bandwidth;e.ode_skip=skip;e.ode_refinements=refinements;e.ode_scale=scale;
  const int size=e.n*(2*bandwidth+1);e.ode_lu.assign(lu,lu+size);e.ode_matrix.assign(matrix,matrix+size);
}
void spherical_fast_set_af_reference(void* handle,const double* source){
  auto& e=*static_cast<Engine*>(handle);e.af_reference.assign(source,source+e.n*18);
}
void spherical_fast_set_lapse_source(void* handle,const double* source){
  auto& e=*static_cast<Engine*>(handle);std::copy(source,source+e.n,e.lapse_source.begin());
}
const char* spherical_fast_last_error(void* handle){return static_cast<Engine*>(handle)->error.c_str();}
void* spherical_fast_create(int n,int nf,int w,int qw,const double* r,const int* ix,
    const double* d1,const double* d2,const int* qi,const double* qwt,
    const int* al,const double* alw,const int* ar,const double* arw,const double* p) {
  try{return new Engine(n,nf,w,qw,r,ix,d1,d2,qi,qwt,al,alw,ar,arw,p);}catch(...){return nullptr;}
}
void spherical_fast_destroy(void* handle){delete static_cast<Engine*>(handle);}
void spherical_fast_stats(void* handle,double* stats) {
  auto& e=*static_cast<Engine*>(handle);stats[0]=e.calls;stats[1]=e.solve_seconds;
  for(int i=0;i<22;i++)stats[i+2]=e.diagnostics[i];
}
int spherical_fast_rhs(void* handle,const double* q,double* result,double* flux) {
  auto& engine=*static_cast<Engine*>(handle);
  try{*flux=engine.rhs(q,result,true);return 0;}catch(const std::exception& failure){engine.error=failure.what();return -1;}
}
// q is committed only after a complete RK4 step, just as in the Python driver.
int spherical_fast_advance(void* handle,double* q,int steps,double dt,double* energy) {
  auto& e=*static_cast<Engine*>(handle);int size=e.n*e.nf,done=0;
  try {
    for(;done<steps;done++) {
      double f1=e.rhs(q,e.k1.data(),true);
      for(int j=0;j<size;j++)e.stage[j]=q[j]+dt*e.k1[j]/2;
      double f2=e.rhs(e.stage.data(),e.k2.data());
      for(int j=0;j<size;j++)e.stage[j]=q[j]+dt*e.k2[j]/2;
      double f3=e.rhs(e.stage.data(),e.k3.data());
      for(int j=0;j<size;j++)e.stage[j]=q[j]+dt*e.k3[j];
      double f4=e.rhs(e.stage.data(),e.k4.data());
      for(int j=0;j<size;j++)q[j]+=dt*(e.k1[j]+2*e.k2[j]+2*e.k3[j]+e.k4[j])/6;
      *energy+=dt*(f1+2*f2+2*f3+f4)/6;
    }
  }catch(const std::exception& failure){e.error=failure.what();return -done-1;}
  return done;
}
}
