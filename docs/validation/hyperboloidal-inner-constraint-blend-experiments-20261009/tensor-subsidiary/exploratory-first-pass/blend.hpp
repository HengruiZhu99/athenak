// Prescribed C(r)=1-W_gauge(r). This system is a tensor blend, not fully covariant.
std::array<C,8> Subsidiary(const hyp::LayerPoint<double>&p,double kap,
                         const ConstraintJet&q) {
 const auto one=SubsidiaryC1(p,kap,q);
 if(!g_blend)return one;
 const auto zero=SubsidiaryC0(p,kap,q);
 const auto w=hyp::SmoothCutoff(p.radius,.45,.85);
 const double c=1-w.value,o=p.omega,alpha=p.alpha,chi=p.state.chi.value;
 std::array<C,8> out{};for(int i=0;i<8;++i)out[i]=(1-c)*zero[i]+c*one[i];
 double dc[3]{}; // p points in this audit lie on the positive x axis.
 if(p.radius>0)dc[0]=-w.d;
 const auto &u=p.state;const auto gt=hyp::Geometry(u.metric);
 auto on=Omega(Lift(u),p);const double normal=on.normal.v;
 // S1-S0=-2A Theta Kij-A*k1 Theta gammaij-gammaij B0/3.
 C B=-(6*alpha*normal+3*kap)*q.q[7]/o;
 for(int i=0;i<3;++i)for(int j=0;j<3;++j)
  B+=alpha*gt.inverse[i][j]*(o*u.chi.d[i]+2*chi*p.domega[i])*q.q[4+j];
 C delta[3][3]{},trace{};
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){
  const double gamma=u.metric.g[i][j]/(chi*o*o);
  const double kij=u.a.k[i][j]/(o*chi)+gamma*u.trace.value/3;
  delta[i][j]=-2*alpha*kij*q.q[7]/o-kap*gamma*q.q[7]/o-gamma*B/3.;
  trace+=o*o*chi*gt.inverse[i][j]*delta[i][j];
 }
 // The extra spatial derivative of C occurs only in the ADM M equation.
 for(int i=0;i<3;++i){
  out[1+i]-=dc[i]*trace;
  for(int j=0;j<3;++j)for(int a=0;a<3;++a)
   out[1+i]+=o*o*chi*gt.inverse[j][a]*dc[a]*delta[i][j];
 }
 return out;
}
