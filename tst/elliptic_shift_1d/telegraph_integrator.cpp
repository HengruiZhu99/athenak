// Reuse the frozen eight-field geometry kernel without changing its ABI/state.
#include "fast_integrator.cpp"
struct Telegraph {
  Engine& e;double kappa,tau,start;int nr;Vec w,wp,comp;
  Vec q1,q2,q3,q4,qs,b1,b2,b3,b4,bs,j1,j2,j3,j4,js,speeds;
  std::array<double,5> stats{};
  Telegraph(Engine& E,double K,double T,const double* W,const double* WP,int NR,double START):
      e(E),kappa(K),tau(T),start(START),nr(NR),w(W,W+E.n),wp(WP,WP+E.n),comp(9*E.n+2*NR,0) {
    if(!e.af_endpoint||e.nf!=8||kappa<0||tau<=0||!std::isfinite(kappa+tau))throw std::runtime_error("invalid telegraph setup");
    for(Vec* v:{&q1,&q2,&q3,&q4,&qs})v->resize(e.n*8);
    for(Vec* v:{&b1,&b2,&b3,&b4,&bs})v->resize(e.n);
    for(Vec* v:{&j1,&j2,&j3,&j4,&js})v->resize(2*nr);
    speeds.resize(e.n*2);
  }
  void rhs(const double* q,const double* B,double* dq,double* dB,bool accepted=false) {
    e.rhs(q,dq,accepted);const int n=e.n;
    for(int i=0;i<n;i++) {
      double derivative=0,adv=0;
      const auto& ai=e.beta[i]<0?e.aleft:e.aright;const auto& aw=e.beta[i]<0?e.awleft:e.awright;
      for(int j=0;j<e.w;j++) {
        derivative+=e.d1[i*e.w+j]*(B[e.reflected(e.index[i*e.w+j])]-B[i]);
        adv+=aw[i*e.w+j]*(B[e.reflected(ai[i*e.w+j])]-B[i]);
      }
      const double r=e.r[i],chi=e.total_chi[i],alpha=e.total_alpha[i];
      // Interior flat-reference radial contraction. Support is zero in both collars.
      const double correction=chi*(w[i]*derivative+(wp[i]+(r>0?2*w[i]/r:0))*B[i]);
      if(i>0&&i<n-1)dq[8*i+6]+=correction/alpha;
      dB[i]=e.beta[i]*adv+(kappa*w[i]*e.d[8*i+6]-B[i])/tau;
      const double c=std::sqrt(chi*(e.lapse_coefficient(r,alpha)+kappa*w[i]*w[i]/tau));
      speeds[2*i]=-e.beta[i]-c;speeds[2*i+1]=-e.beta[i]+c;
      stats[0]=std::max(stats[0],std::abs(e.beta[i])+c);
      if(r>.03&&r<.3) {
        const double O=(1-r*r)/2;
        stats[3]=std::max(stats[3],std::abs((r+e.d[8*i+6])/O+alpha*r/(O*O)));
      }
    }
    // Identical negative-semidefinite puncture KO on the extra field.
    for(int i=1;i<n-3;i++) {
      const double midpoint=(e.r[i]+e.r[i+3])/2;
      const double taper=std::pow(std::max(1-midpoint*midpoint/(e.params[26]*e.params[26]),0.),6);
      const double third=3*(B[i+1]-B[i])-3*(B[i+2]-B[i])+(B[i+3]-B[i]);
      const double value=e.params[25]*taper*third/(64*e.h);
      dB[i]+=value/e.jac(i);dB[i+1]-=3*value/e.jac(i+1);dB[i+2]+=3*value/e.jac(i+2);dB[i+3]-=value/e.jac(i+3);
    }
    for(int i=0;i<n-1;i++)if(e.r[i]>.03&&e.r[i]<.3)
      stats[4]=std::max(stats[4],-(speeds[2*i+2]-speeds[2*i])/(e.r[i+1]-e.r[i]));
    for(int i=0;i<n;i++)if(!std::isfinite(dB[i])||!std::isfinite(dq[8*i+6]))throw std::runtime_error("nonfinite telegraph rate");
  }
  void rays(const double* state,double* rate,bool active)const {
    std::fill(rate,rate+2*nr,0.);
    if(!active)return;
    for(int j=0;j<nr;j++) {
      const double x=state[2*j];if(x<=e.r.front()||x>=e.r.back())continue;
      int i=std::min(e.n-2,std::max(0,static_cast<int>((x-e.r[0])/e.h)));
      if(!e.grid_jacobian.empty())i=std::min(e.n-2,std::max(0,static_cast<int>(std::upper_bound(e.r.begin(),e.r.end(),x)-e.r.begin())-1));
      const int family=j%2;double left=speeds[2*i+family],slope=(speeds[2*(i+1)+family]-left)/(e.r[i+1]-e.r[i]);
      rate[2*j]=left+slope*(x-e.r[i]);rate[2*j+1]=slope;
    }
  }
  void check_dt(double dt) {
    double speed=0;for(int i=0;i<e.n;i++)for(int f=0;f<2;f++)speed=std::max(speed,std::abs(speeds[2*i+f])/e.jac(i));
    stats[1]=std::max(stats[1],dt*speed/e.h);stats[2]=std::max(stats[2],dt/tau);
    if(dt*speed/e.h>.4||dt/tau>.4)throw std::runtime_error("telegraph coupled-speed/relaxation timestep guard");
    for(int i=0;i<e.n;i++)if(dt*e.params[25]/(e.h*e.jac(i))>2.)throw std::runtime_error("mapped-grid dissipation timestep guard");
  }
  void update(double* q,const Vec& a,const Vec& b,const Vec& c,const Vec& d,int size,int offset,double dt) {
    for(int j=0;j<size;j++) {
      double inc=dt*(a[j]+2*b[j]+2*c[j]+d[j])/6;
      if(e.params[27]!=0){double v=inc-comp[offset+j],next=q[j]+v;comp[offset+j]=(next-q[j])-v;q[j]=next;}
      else q[j]+=inc;
    }
  }
};
extern "C" {
void* telegraph_create(void* handle,double kappa,double tau,const double* w,const double* wp,int nr,double start) {
 try{return new Telegraph(*static_cast<Engine*>(handle),kappa,tau,w,wp,nr,start);}catch(...){return nullptr;}
}
void telegraph_destroy(void* handle){delete static_cast<Telegraph*>(handle);}
void telegraph_stats(void* handle,double* out){auto& t=*static_cast<Telegraph*>(handle);std::copy(t.stats.begin(),t.stats.end(),out);}
void telegraph_get_compensation(void* handle,double* out){auto& t=*static_cast<Telegraph*>(handle);std::copy(t.comp.begin(),t.comp.end(),out);}
void telegraph_set_compensation(void* handle,const double* in){auto& t=*static_cast<Telegraph*>(handle);std::copy(in,in+t.comp.size(),t.comp.begin());}
int telegraph_rhs(void* handle,const double* q,const double* B,double* dq,double* dB,double* speeds) {
 auto& t=*static_cast<Telegraph*>(handle);try{t.rhs(q,B,dq,dB,true);std::copy(t.speeds.begin(),t.speeds.end(),speeds);return 0;}catch(const std::exception& x){t.e.error=x.what();return -1;}
}
int telegraph_advance(void* handle,double* q,double* B,double* rays,int count,double dt,double time) {
 auto& t=*static_cast<Telegraph*>(handle);int done=0,n=t.e.n,nq=8*n,nj=2*t.nr;
 try {
  for(;done<count;done++) {
   double now=time+done*dt;const bool active=now>=t.start-8*std::numeric_limits<double>::epsilon()*std::max(1.,t.start);
   t.rhs(q,B,t.q1.data(),t.b1.data(),true);t.check_dt(dt);t.rays(rays,t.j1.data(),active);
   for(int i=0;i<nq;i++)t.qs[i]=q[i]+dt*t.q1[i]/2;
   for(int i=0;i<n;i++)t.bs[i]=B[i]+dt*t.b1[i]/2;
   for(int i=0;i<nj;i++)t.js[i]=rays[i]+dt*t.j1[i]/2;
   t.rhs(t.qs.data(),t.bs.data(),t.q2.data(),t.b2.data());t.check_dt(dt);t.rays(t.js.data(),t.j2.data(),active);
   for(int i=0;i<nq;i++)t.qs[i]=q[i]+dt*t.q2[i]/2;
   for(int i=0;i<n;i++)t.bs[i]=B[i]+dt*t.b2[i]/2;
   for(int i=0;i<nj;i++)t.js[i]=rays[i]+dt*t.j2[i]/2;
   t.rhs(t.qs.data(),t.bs.data(),t.q3.data(),t.b3.data());t.check_dt(dt);t.rays(t.js.data(),t.j3.data(),active);
   for(int i=0;i<nq;i++)t.qs[i]=q[i]+dt*t.q3[i];
   for(int i=0;i<n;i++)t.bs[i]=B[i]+dt*t.b3[i];
   for(int i=0;i<nj;i++)t.js[i]=rays[i]+dt*t.j3[i];
   t.rhs(t.qs.data(),t.bs.data(),t.q4.data(),t.b4.data());t.check_dt(dt);t.rays(t.js.data(),t.j4.data(),active);
   t.update(q,t.q1,t.q2,t.q3,t.q4,nq,0,dt);t.update(B,t.b1,t.b2,t.b3,t.b4,n,nq,dt);
   t.update(rays,t.j1,t.j2,t.j3,t.j4,nj,nq+n,dt);
   // Exited rays are terminal records, never reinjected or interpreted as crossing.
   for(int i=0;i<t.nr;i++)if(rays[2*i]<=t.e.r.front()||rays[2*i]>=t.e.r.back()) {
    rays[2*i]=std::max(t.e.r.front(),std::min(t.e.r.back(),rays[2*i]));t.comp[nq+n+2*i]=0;
   }
  }
 }catch(const std::exception& x){t.e.error=x.what();return -done-1;}
 return done;
}
}
