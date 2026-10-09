// Separate derived physical ADM subsidiary equations for repaired C_Z4c=1.
// Linearized about the stationary Einstein reference, with coefficient gradients.
struct ConstraintJet {C q[8]{},d[3][8]{},dd[3][3][8]{};};
std::array<C,8> SubsidiaryC1(const hyp::LayerPoint<double>&p,double kap,const ConstraintJet&c){
 const auto &u=p.state;const auto gt=hyp::Geometry(u.metric);const double o=p.omega,chi=u.chi.value,alpha=p.alpha,K=u.trace.value;
 const double lapse=alpha/o;double dlapse[3]{};for(int i=0;i<3;++i)dlapse[i]=u.alpha.d[i]/o-alpha*p.domega[i]/(o*o);
 const auto bar=hyp::PenroseMetric(u.metric,u.chi);hyp::MetricJet<double> m{};
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){m.g[i][j]=bar.g[i][j]/(o*o);for(int a=0;a<3;++a){m.dg[a][i][j]=bar.dg[a][i][j]/(o*o)-2*bar.g[i][j]*p.domega[a]/(o*o*o);for(int b=0;b<3;++b)m.ddg[a][b][i][j]=bar.ddg[a][b][i][j]/(o*o)-2*(bar.dg[a][i][j]*p.domega[b]+bar.dg[b][i][j]*p.domega[a]+bar.g[i][j]*p.omega_hessian[a][b])/(o*o*o)+6*bar.g[i][j]*p.domega[a]*p.domega[b]/(o*o*o*o);}}
 const auto g=hyp::Geometry(m);if(!g.valid)throw std::runtime_error("physical geometry invalid");
 double di[3][3][3]{},digt[3][3][3]{},dgamma[3][3][3][3]{};
 for(int a=0;a<3;++a)for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int l=0;l<3;++l)for(int n=0;n<3;++n){di[a][i][j]-=g.inverse[i][l]*m.dg[a][l][n]*g.inverse[n][j];digt[a][i][j]-=gt.inverse[i][l]*u.metric.dg[a][l][n]*gt.inverse[n][j];}
 for(int a=0;a<3;++a)for(int l=0;l<3;++l)for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int n=0;n<3;++n)dgamma[a][l][i][j]+=.5*(di[a][l][n]*(m.dg[i][n][j]+m.dg[j][n][i]-m.dg[n][i][j])+g.inverse[l][n]*(m.ddg[a][i][n][j]+m.ddg[a][j][n][i]-m.ddg[a][n][i][j]));
 double kij[3][3]{},dkij[3][3][3]{};
 const double damping=kap/o;
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){
  kij[i][j]=u.a.k[i][j]/(o*chi)+m.g[i][j]*K/3;
  for(int a=0;a<3;++a)dkij[a][i][j]=u.a.dk[a][i][j]/(o*chi)
   -u.a.k[i][j]*(p.domega[a]*chi+o*u.chi.d[a])/(o*o*chi*chi)
   +(m.dg[a][i][j]*K+m.g[i][j]*u.trace.d[a])/3;
 }
 // Sij=A(DiZj+DjZi-2Theta Kij)-A*kappa1*Theta*gammaij.
 // kappa1=kap/alpha, hence A*kappa1=kap/Omega is differentiated as such.
 C s[3][3]{},ds[3][3][3]{};
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){
  C z=c.d[i][4+j]+c.d[j][4+i];
  for(int l=0;l<3;++l)z-=2*g.connection[l][i][j]*c.q[4+l];
  s[i][j]=lapse*(z-2.*kij[i][j]*c.q[7])-damping*m.g[i][j]*c.q[7];
  for(int a=0;a<3;++a){
   C dz=c.dd[a][i][4+j]+c.dd[a][j][4+i];
   for(int l=0;l<3;++l)dz-=2.*(dgamma[a][l][i][j]*c.q[4+l]+g.connection[l][i][j]*c.d[a][4+l]);
   ds[a][i][j]=dlapse[a]*(z-2.*kij[i][j]*c.q[7])
    +lapse*(dz-2.*dkij[a][i][j]*c.q[7]-2.*kij[i][j]*c.d[a][7])
    +(kap*p.domega[a]*m.g[i][j]/(o*o)-damping*m.dg[a][i][j])*c.q[7]
    -damping*m.g[i][j]*c.d[a][7];
  }
 }
 std::array<C,8> out{};out[0]=2*lapse*K*c.q[0];for(int a=0;a<3;++a)out[0]+=p.beta[a]*c.d[a][0];
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){C dm=c.d[i][1+j];for(int l=0;l<3;++l)dm-=g.connection[l][i][j]*c.q[1+l];out[0]-=2*lapse*g.inverse[i][j]*dm+4*g.inverse[i][j]*c.q[1+i]*dlapse[j];double kup=0;for(int a=0;a<3;++a)for(int b=0;b<3;++b)kup+=g.inverse[i][a]*g.inverse[j][b]*(u.a.k[a][b]/(o*chi)+m.g[a][b]*K/3);out[0]+=2*(K*g.inverse[i][j]-kup)*s[i][j];}
 for(int i=0;i<3;++i){out[1+i]=lapse*K*c.q[1+i]-.5*lapse*c.d[i][0]-c.q[0]*dlapse[i];for(int a=0;a<3;++a)out[1+i]+=p.beta[a]*c.d[a][1+i]+c.q[1+a]*u.beta.d[i][a];
  for(int j=0;j<3;++j)for(int a=0;a<3;++a){C cov=ds[a][i][j];for(int l=0;l<3;++l)cov-=g.connection[l][a][i]*s[l][j]+g.connection[l][a][j]*s[i][l];out[1+i]+=g.inverse[j][a]*cov-di[i][j][a]*s[j][a]-g.inverse[j][a]*ds[i][j][a];}}
 out[7]=lapse*c.q[0]/2.-(lapse*K+2*damping)*c.q[7];
 for(int i=0;i<3;++i){out[7]+=p.beta[i]*c.d[i][7];
  for(int j=0;j<3;++j){C cov=c.d[i][4+j];
   for(int l=0;l<3;++l)cov-=g.connection[l][i][j]*c.q[4+l];
   out[7]+=lapse*g.inverse[i][j]*cov-g.inverse[i][j]*c.q[4+i]*dlapse[j];
  }
 }
 for(int i=0;i<3;++i){
  out[4+i]=lapse*(c.q[1+i]+c.d[i][7])-damping*c.q[4+i]-c.q[7]*dlapse[i];
  for(int j=0;j<3;++j){
   out[4+i]+=p.beta[j]*c.d[j][4+i]+u.beta.d[i][j]*c.q[4+j];
   for(int l=0;l<3;++l)out[4+i]-=2*lapse*kij[i][j]*g.inverse[j][l]*c.q[4+l];
  }
 }
 return out;
}
