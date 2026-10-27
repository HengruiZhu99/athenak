// Reuse the frozen eight-field geometry kernel without changing its ABI/state.
#include "fast_integrator.cpp"
namespace {
double pchip_endpoint(double h0,double h1,double d0,double d1) {
  double m=((2*h0+h1)*d0-h0*d1)/(h0+h1);
  if(d0==0||std::signbit(m)!=std::signbit(d0))return 0.;
  if(std::signbit(d0)!=std::signbit(d1)&&std::abs(m)>3*std::abs(d0))return 3*d0;
  return m;
}
// Local Fritsch--Butland slopes; coordinate is physical r, not the grid chart.
void pchip_slopes(const Vec& r,const Vec& values,Vec& slopes) {
  int n=r.size();
  for(int f=0;f<2;f++) {
    auto secant=[&](int i){return (values[2*i+2+f]-values[2*i+f])/(r[i+1]-r[i]);};
    slopes[f]=pchip_endpoint(r[1]-r[0],r[2]-r[1],secant(0),secant(1));
    slopes[2*(n-1)+f]=pchip_endpoint(r[n-1]-r[n-2],r[n-2]-r[n-3],secant(n-2),secant(n-3));
    for(int i=1;i<n-1;i++) {
      double a=secant(i-1),b=secant(i),hp=r[i]-r[i-1],hn=r[i+1]-r[i];
      double w1=2*hn+hp,w2=hn+2*hp;
      slopes[2*i+f]=(a==0||b==0||std::signbit(a)!=std::signbit(b))?0.:(w1+w2)/(w1/a+w2/b);
    }
  }
}
void pchip_value(double x,double r0,double r1,double c0,double c1,double m0,double m1,double* out) {
  double h=r1-r0,t=(x-r0)/h,delta=c1-c0,A=h*m0-delta,B=delta-h*m1,C=(1-t)*A+t*B;
  out[0]=c0+t*delta+t*(1-t)*C;
  out[1]=(delta+(1-2*t)*C+t*(1-t)*(B-A))/h;
}
struct RayCubic {
  double a,b,c;
  double operator()(double t)const{return ((a*t+b)*t+c)*t;}
};
RayCubic ray_cubic(const double* k,int component,double dt) {
  double k1=k[component],k2=k[2+component],k3=k[4+component],k4=k[6+component];
  return {dt*(2./3)*(k1-k2-k3+k4),dt*(-1.5*k1+k2+k3-.5*k4),dt*k1};
}
// Classical RK4 third-order continuous extension. Its order conditions are
// verified independently; the first event may precede a return into the domain.
double ray_increment(const double* state,const double* k,double dt,double lo,double hi,double* inc) {
  RayCubic x=ray_cubic(k,0,dt);std::array<double,4> cuts{0,1,0,0};int count=2;
  auto add=[&](double t){if(t>0&&t<1)cuts[count++]=t;};
  if(x.a==0) {if(x.b!=0)add(-x.c/(2*x.b));}
  else {
    double disc=x.b*x.b-3*x.a*x.c;
    if(disc>=0) {
      double q=-x.b-std::copysign(std::sqrt(disc),x.b);
      if(q!=0){add(q/(3*x.a));add(x.c/q);}
      else add(-x.b/(3*x.a));
    }
  }
  std::sort(cuts.begin(),cuts.begin()+count);
  for(int j=1;j<count;j++) {
    double right=cuts[j],value=state[0]+x(right);
    if(value>lo&&value<hi)continue;
    const double wall=value<=lo?lo:hi;double left=cuts[j-1];
    for(int it=0;it<64;it++) {
      double mid=left+(right-left)/2;if(mid==left||mid==right)break;
      double v=state[0]+x(mid);
      if(wall==lo?v<=lo:v>=hi)right=mid;else left=mid;
    }
    double theta=left+(right-left)/2;
    inc[0]=wall-state[0];inc[1]=ray_cubic(k,1,dt)(theta);return theta;
  }
  for(int f=0;f<2;f++)inc[f]=dt*(k[f]+2*k[2+f]+2*k[4+f]+k[6+f])/6;
  return -1.;
}
}
struct Telegraph {
  Engine& e;double kappa,tau,start;int nr;Vec w,wp,comp;
  double lapse_floor=0.;Vec lapse_floor_log;
  std::array<double,3> floor_stats{};
  bool total_lapse_update=false;
  double lapse_attenuation=0.;
  std::array<double,2> total_floor_stats{};
  bool cubic_rays=false,dense_exits=false;Vec speed_slopes;
  Vec q1,q2,q3,q4,qs,b1,b2,b3,b4,bs,j1,j2,j3,j4,js,speeds;
  std::array<double,5> stats{};
  Telegraph(Engine& E,double K,double T,const double* W,const double* WP,int NR,double START):
      e(E),kappa(K),tau(T),start(START),nr(NR),w(W,W+E.n),wp(WP,WP+E.n),comp(9*E.n+2*NR,0) {
    if(!e.af_endpoint||e.nf!=8||kappa<0||tau<=0||!std::isfinite(kappa+tau))throw std::runtime_error("invalid telegraph setup");
    for(Vec* v:{&q1,&q2,&q3,&q4,&qs})v->resize(e.n*8);
    for(Vec* v:{&b1,&b2,&b3,&b4,&bs})v->resize(e.n);
    for(Vec* v:{&j1,&j2,&j3,&j4,&js})v->resize(2*nr);
    speeds.resize(e.n*2);
    speed_slopes.resize(e.n*2);
  }
  void set_lapse_floor(double value) {
    if(!std::isfinite(value)||value<0)throw std::runtime_error("lapse_floor must be finite and nonnegative");
    Vec limits(e.n,0.);
    if(value>0)for(int i=1;i<e.n-1;i++) {
      const double O=(1-e.r[i]*e.r[i])/2,reference=e.ref(i,2);
      const double ratio=value*O/reference;
      if(!(ratio>0)||!std::isfinite(ratio))
        throw std::runtime_error("lapse_floor is not representable in the AF lapse chart");
      double lower=std::log(ratio);
      // Correct in log space with a relative-roundoff margin, including when
      // log(ratio) is close to zero and one log-ULP would be too small.
      const double margin=4*std::numeric_limits<double>::epsilon()*std::max(1.,std::abs(lower));
      for(int round=0;round<8&&reference*std::exp(lower)/O<value;round++)lower+=margin;
      lower+=margin;
      if(reference*std::exp(lower)/O<value||!std::isfinite(std::exp(lower)))
        throw std::runtime_error("lapse_floor cannot be reconstructed at this node");
      limits[i]=lower;
    }
    lapse_floor=value;lapse_floor_log=std::move(limits);
  }
  void floor_lapse(double* q,bool accepted) {
    if(lapse_floor==0)return;
    int count=0;
    // r=0 stores an AF coefficient; scri has its own clock condition.
    for(int i=1;i<e.n-1;i++)if(std::isfinite(q[8*i+6])&&q[8*i+6]<lapse_floor_log[i]) {
      floor_stats[0]=std::max(floor_stats[0],e.r[i]);
      floor_stats[1]=std::max(floor_stats[1],lapse_floor_log[i]-q[8*i+6]);
      q[8*i+6]=lapse_floor_log[i];if(accepted)comp[8*i+6]=0.;count++;
    }
    floor_stats[2]=std::max(floor_stats[2],static_cast<double>(count));
  }
  double alpha_value(const double* q,int i)const {return e.ref(i,2)*std::exp(q[8*i+6]);}
  int store_alpha(double* q,int i,double value) {
    if(!std::isfinite(value))throw std::runtime_error("nonfinite total-lapse RK update");
    const double lower=lapse_floor>0?e.ref(i,2)*std::exp(lapse_floor_log[i]):0.;
    int clipped=0;
    if(value<lower) {
      clipped=value<=0?2:1;
      floor_stats[0]=std::max(floor_stats[0],e.r[i]);
      if(value>0)floor_stats[1]=std::max(floor_stats[1],std::log(lower)-std::log(value));
      total_floor_stats[0]=std::max(total_floor_stats[0],lower-value);
      value=lower;
    }
    if(!(value>0))throw std::runtime_error("nonpositive total-lapse RK update without a floor");
    q[8*i+6]=std::log(value)-std::log(e.ref(i,2));
    if(lapse_floor>0)q[8*i+6]=std::max(q[8*i+6],lapse_floor_log[i]);
    return clipped;
  }
  void total_rates(Vec& rate) {
    if(total_lapse_update)for(int i=1;i<e.n-1;i++)rate[8*i+6]*=e.total_alpha[i];
  }
  void make_stage(const double* q,const Vec& rate,double factor) {
    for(int j=0;j<8*e.n;j++)qs[j]=q[j]+factor*rate[j];
    if(!total_lapse_update)return;
    int clipped=0,nonpositive=0;
    for(int i=1;i<e.n-1;i++) {
      int result=store_alpha(qs.data(),i,alpha_value(q,i)+factor*rate[8*i+6]);
      clipped+=result>0;nonpositive+=result==2;
    }
    floor_stats[2]=std::max(floor_stats[2],static_cast<double>(clipped));
    total_floor_stats[1]=std::max(total_floor_stats[1],static_cast<double>(nonpositive));
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
      const double activation=lapse_attenuation>0?(1+lapse_attenuation)*alpha/(alpha+lapse_attenuation*(1+r*r)/2):1.;
      const double correction=activation*chi*(w[i]*derivative+(wp[i]+(r>0?2*w[i]/r:0))*B[i]);
      if(i>0&&i<n-1)dq[8*i+6]+=correction/alpha;
      dB[i]=e.beta[i]*adv+(kappa*w[i]*e.d[8*i+6]-B[i])/tau;
      const double c=std::sqrt(chi*(e.lapse_coefficient(r,alpha)+activation*kappa*w[i]*w[i]/tau));
      speeds[2*i]=-e.beta[i]-c;speeds[2*i+1]=-e.beta[i]+c;
      stats[0]=std::max(stats[0],std::abs(e.beta[i])+c);
      if(r>.03&&r<.3) {
        const double O=(1-r*r)/2;
        stats[3]=std::max(stats[3],std::abs((r+e.d[8*i+6])/O+alpha*r/(O*O)));
      }
    }
    // Identical negative-semidefinite puncture KO on the extra field.
    if(e.ko_difference_order==3)for(int i=1;i<n-3;i++) {
      const double midpoint=(e.r[i]+e.r[i+3])/2;
      const double taper=std::pow(std::max(1-midpoint*midpoint/(e.params[26]*e.params[26]),0.),6);
      const double third=3*(B[i+1]-B[i])-3*(B[i+2]-B[i])+(B[i+3]-B[i]);
      const double value=e.params[25]*taper*e.ko_measure(i)*third/(64*e.h);
      dB[i]+=value/e.jac(i);dB[i+1]-=3*value/e.jac(i+1);dB[i+2]+=3*value/e.jac(i+2);dB[i+3]-=value/e.jac(i+3);
    }
    if(e.ko_difference_order==4)for(int i=1;i<n-4;i++) {
      const double midpoint=(e.r[i]+e.r[i+4])/2;
      const double taper=std::pow(std::max(1-midpoint*midpoint/(e.params[26]*e.params[26]),0.),6);
      const double fourth=-4*(B[i+1]-B[i])+6*(B[i+2]-B[i])-4*(B[i+3]-B[i])+(B[i+4]-B[i]);
      const double value=e.params[25]*taper*e.ko_measure(i)*fourth/(256*e.h);
      dB[i]-=value/e.jac(i);dB[i+1]+=4*value/e.jac(i+1);dB[i+2]-=6*value/e.jac(i+2);dB[i+3]+=4*value/e.jac(i+3);dB[i+4]-=value/e.jac(i+4);
    }
    for(int i=0;i<n-1;i++)if(e.r[i]>.03&&e.r[i]<.3)
      stats[4]=std::max(stats[4],-(speeds[2*i+2]-speeds[2*i])/(e.r[i+1]-e.r[i]));
    for(int i=0;i<n;i++)if(!std::isfinite(dB[i])||!std::isfinite(dq[8*i+6]))throw std::runtime_error("nonfinite telegraph rate");
  }
  void rays(const double* state,double* rate,bool active,const double* initial=nullptr) {
    std::fill(rate,rate+2*nr,0.);
    if(!active)return;
    if(cubic_rays)pchip_slopes(e.r,speeds,speed_slopes);
    for(int j=0;j<nr;j++) {
      const double x=state[2*j],seed=(dense_exits&&initial)?initial[2*j]:x;
      if(seed<=e.r.front()||seed>=e.r.back())continue;
      // Extend only the endpoint polynomial, at most one boundary cell, to
      // evaluate stages needed for an event from an initially interior ray.
      if(dense_exits&&(x<2*e.r.front()-e.r[1]||x>2*e.r.back()-e.r[e.n-2]))throw std::runtime_error("ray exit stage exceeds one-cell continuation");
      int i=std::min(e.n-2,std::max(0,static_cast<int>((x-e.r[0])/e.h)));
      if(!e.grid_jacobian.empty())i=std::min(e.n-2,std::max(0,static_cast<int>(std::upper_bound(e.r.begin(),e.r.end(),x)-e.r.begin())-1));
      const int family=j%2;double left=speeds[2*i+family],slope=(speeds[2*(i+1)+family]-left)/(e.r[i+1]-e.r[i]);
      if(cubic_rays)pchip_value(x,e.r[i],e.r[i+1],left,speeds[2*(i+1)+family],speed_slopes[2*i+family],speed_slopes[2*(i+1)+family],rate+2*j);
      else {rate[2*j]=left+slope*(x-e.r[i]);rate[2*j+1]=slope;}
    }
  }
  void check_dt(double dt) {
    double speed=0;for(int i=0;i<e.n;i++)for(int f=0;f<2;f++)speed=std::max(speed,std::abs(speeds[2*i+f])/e.jac(i));
    stats[1]=std::max(stats[1],dt*speed/e.h);stats[2]=std::max(stats[2],dt/tau);
    if(dt*speed/e.h>.4||dt/tau>.4)throw std::runtime_error("telegraph coupled-speed/relaxation timestep guard");
    for(int i=0;i<e.n;i++)if(dt*e.params[25]/(e.h*(e.mapped_ko_local?1./(e.ko_difference_order+1):e.jac(i)))>2.)throw std::runtime_error("mapped-grid dissipation timestep guard");
  }
  void update(double* q,const Vec& a,const Vec& b,const Vec& c,const Vec& d,int size,int offset,double dt) {
    int clipped=0,nonpositive=0;
    for(int j=0;j<size;j++) {
      double inc=dt*(a[j]+2*b[j]+2*c[j]+d[j])/6;
      if(total_lapse_update&&offset==0&&j%8==6&&j/8>0&&j/8<e.n-1) {
        const double old=alpha_value(q,j/8),v=e.params[27]!=0?inc-comp[j]:inc;
        int result=store_alpha(q,j/8,old+v);
        // Include log storage/reconstruction roundoff in the physical remainder.
        comp[j]=(result||e.params[27]==0)?0.:(alpha_value(q,j/8)-old)-v;
        clipped+=result>0;nonpositive+=result==2;
        continue;
      }
      if(e.params[27]!=0){double v=inc-comp[offset+j],next=q[j]+v;comp[offset+j]=(next-q[j])-v;q[j]=next;}
      else q[j]+=inc;
    }
    floor_stats[2]=std::max(floor_stats[2],static_cast<double>(clipped));
    total_floor_stats[1]=std::max(total_floor_stats[1],static_cast<double>(nonpositive));
  }
  void update_rays(double* state,double dt) {
    for(int j=0;j<nr;j++) {
      int i=2*j,offset=9*e.n+i;
      if(state[i]<=e.r.front()||state[i]>=e.r.back())continue;
      double k[8]={j1[i],j1[i+1],j2[i],j2[i+1],j3[i],j3[i+1],j4[i],j4[i+1]},inc[2];
      double event=ray_increment(state+i,k,dt,e.r.front(),e.r.back(),inc);
      for(int f=0;f<2;f++) {
        if(f==0&&event>=0) {state[i]=inc[0]<0?e.r.front():e.r.back();comp[offset]=0.;}
        else if(e.params[27]!=0) {double v=inc[f]-comp[offset+f],next=state[i+f]+v;comp[offset+f]=(next-state[i+f])-v;state[i+f]=next;}
        else state[i+f]+=inc[f];
      }
    }
  }
};
extern "C" {
void* telegraph_create(void* handle,double kappa,double tau,const double* w,const double* wp,int nr,double start) {
 try{return new Telegraph(*static_cast<Engine*>(handle),kappa,tau,w,wp,nr,start);}catch(...){return nullptr;}
}
void telegraph_destroy(void* handle){delete static_cast<Telegraph*>(handle);}
int telegraph_set_ray_interpolation(void* handle,int mode) {
 if(mode!=0&&mode!=1)return -1;
 static_cast<Telegraph*>(handle)->cubic_rays=(mode==1);return 0;
}
int telegraph_set_ray_exit(void* handle,int mode) {
 if(mode!=0&&mode!=1)return -1;
 static_cast<Telegraph*>(handle)->dense_exits=(mode==1);return 0;
}
// Pure continuous-step/event kernel, also used by the production ray update.
void telegraph_dense_ray_step(const double* state,const double* rates,double dt,double lo,double hi,double* out) {
 if(state[0]<=lo||state[0]>=hi){out[0]=state[0];out[1]=state[1];out[2]=0.;return;}
 double inc[2];out[2]=ray_increment(state,rates,dt,lo,hi,inc);
 out[0]=out[2]>=0?(inc[0]<0?lo:hi):state[0]+inc[0];out[1]=state[1]+inc[1];
}
// Pure interpolation entry point for independent diagnostic-kernel verification.
void telegraph_ray_rates(void* handle,const double* speeds,const double* state,double* rate) {
 auto& t=*static_cast<Telegraph*>(handle);std::copy(speeds,speeds+t.speeds.size(),t.speeds.begin());t.rays(state,rate,true);
}
void telegraph_stats(void* handle,double* out){auto& t=*static_cast<Telegraph*>(handle);std::copy(t.stats.begin(),t.stats.end(),out);}
void telegraph_get_compensation(void* handle,double* out){auto& t=*static_cast<Telegraph*>(handle);std::copy(t.comp.begin(),t.comp.end(),out);}
void telegraph_set_compensation(void* handle,const double* in){auto& t=*static_cast<Telegraph*>(handle);std::copy(in,in+t.comp.size(),t.comp.begin());}
int telegraph_set_lapse_floor(void* handle,double value) {
 auto& t=*static_cast<Telegraph*>(handle);try{t.set_lapse_floor(value);return 0;}
 catch(const std::exception& x){t.e.error=x.what();return -1;}
}
int telegraph_apply_lapse_floor(void* handle,double* q) {
 auto& t=*static_cast<Telegraph*>(handle);t.floor_lapse(q,true);return 0;
}
void telegraph_lapse_floor_stats(void* handle,double* out) {
 auto& t=*static_cast<Telegraph*>(handle);std::copy(t.floor_stats.begin(),t.floor_stats.end(),out);
}
int telegraph_set_lapse_update(void* handle,int mode) {
 if(mode!=0&&mode!=1)return -1;
 static_cast<Telegraph*>(handle)->total_lapse_update=mode==1;return 0;
}
int telegraph_set_lapse_attenuation(void* handle,double scale) {
 if(!std::isfinite(scale)||scale<0||scale>1)return -1;
 static_cast<Telegraph*>(handle)->lapse_attenuation=scale;return 0;
}
void telegraph_total_lapse_stats(void* handle,double* out) {
 auto& t=*static_cast<Telegraph*>(handle);std::copy(t.total_floor_stats.begin(),t.total_floor_stats.end(),out);
}
int telegraph_rhs(void* handle,const double* q,const double* B,double* dq,double* dB,double* speeds) {
 auto& t=*static_cast<Telegraph*>(handle);try{t.rhs(q,B,dq,dB,true);std::copy(t.speeds.begin(),t.speeds.end(),speeds);return 0;}catch(const std::exception& x){t.e.error=x.what();return -1;}
}
int telegraph_advance(void* handle,double* q,double* B,double* rays,int count,double dt,double time) {
 auto& t=*static_cast<Telegraph*>(handle);int done=0,n=t.e.n,nq=8*n,nj=2*t.nr;
 try {
  for(;done<count;done++) {
   double now=time+done*dt;const bool active=now>=t.start-8*std::numeric_limits<double>::epsilon()*std::max(1.,t.start);
   t.floor_lapse(q,true);
   t.rhs(q,B,t.q1.data(),t.b1.data(),true);t.check_dt(dt);t.rays(rays,t.j1.data(),active,rays);
   t.total_rates(t.q1);t.make_stage(q,t.q1,dt/2);
   for(int i=0;i<n;i++)t.bs[i]=B[i]+dt*t.b1[i]/2;
   for(int i=0;i<nj;i++)t.js[i]=rays[i]+dt*t.j1[i]/2;
   t.floor_lapse(t.qs.data(),false);t.rhs(t.qs.data(),t.bs.data(),t.q2.data(),t.b2.data());t.check_dt(dt);t.rays(t.js.data(),t.j2.data(),active,rays);
   t.total_rates(t.q2);t.make_stage(q,t.q2,dt/2);
   for(int i=0;i<n;i++)t.bs[i]=B[i]+dt*t.b2[i]/2;
   for(int i=0;i<nj;i++)t.js[i]=rays[i]+dt*t.j2[i]/2;
   t.floor_lapse(t.qs.data(),false);t.rhs(t.qs.data(),t.bs.data(),t.q3.data(),t.b3.data());t.check_dt(dt);t.rays(t.js.data(),t.j3.data(),active,rays);
   t.total_rates(t.q3);t.make_stage(q,t.q3,dt);
   for(int i=0;i<n;i++)t.bs[i]=B[i]+dt*t.b3[i];
   for(int i=0;i<nj;i++)t.js[i]=rays[i]+dt*t.j3[i];
   t.floor_lapse(t.qs.data(),false);t.rhs(t.qs.data(),t.bs.data(),t.q4.data(),t.b4.data());t.check_dt(dt);t.rays(t.js.data(),t.j4.data(),active,rays);
   t.total_rates(t.q4);
   t.update(q,t.q1,t.q2,t.q3,t.q4,nq,0,dt);t.update(B,t.b1,t.b2,t.b3,t.b4,n,nq,dt);
   t.floor_lapse(q,true);
   if(t.dense_exits)t.update_rays(rays,dt);
   else t.update(rays,t.j1,t.j2,t.j3,t.j4,nj,nq+n,dt);
   // Exited rays are terminal records, never reinjected or interpreted as crossing.
   for(int i=0;i<t.nr;i++)if(rays[2*i]<=t.e.r.front()||rays[2*i]>=t.e.r.back()) {
    rays[2*i]=std::max(t.e.r.front(),std::min(t.e.r.back(),rays[2*i]));t.comp[nq+n+2*i]=0;
   }
  }
 }catch(const std::exception& x){t.e.error=x.what();return -done-1;}
 return done;
}
}
