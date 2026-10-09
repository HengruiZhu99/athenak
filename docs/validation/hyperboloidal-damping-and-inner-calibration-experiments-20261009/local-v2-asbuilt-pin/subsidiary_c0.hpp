// Derived physical ADM subsidiary equations for the actual C_Z4c=0 kernel.
std::array<C,8> SubsidiaryC0(const hyp::LayerPoint<double>&p,double kap,const ConstraintJet&c){
 const auto &u=p.state;const auto gt=hyp::Geometry(u.metric);const double o=p.omega,chi=u.chi.value,alpha=p.alpha,K=u.trace.value;
 const double lapse=alpha/o;double dlapse[3]{};for(int i=0;i<3;++i)dlapse[i]=u.alpha.d[i]/o-alpha*p.domega[i]/(o*o);
 const auto bar=hyp::PenroseMetric(u.metric,u.chi);hyp::MetricJet<double> m{};
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){m.g[i][j]=bar.g[i][j]/(o*o);for(int a=0;a<3;++a){m.dg[a][i][j]=bar.dg[a][i][j]/(o*o)-2*bar.g[i][j]*p.domega[a]/(o*o*o);for(int b=0;b<3;++b)m.ddg[a][b][i][j]=bar.ddg[a][b][i][j]/(o*o)-2*(bar.dg[a][i][j]*p.domega[b]+bar.dg[b][i][j]*p.domega[a]+bar.g[i][j]*p.omega_hessian[a][b])/(o*o*o)+6*bar.g[i][j]*p.domega[a]*p.domega[b]/(o*o*o*o);}}
 const auto g=hyp::Geometry(m);if(!g.valid)throw std::runtime_error("physical geometry invalid");
 double di[3][3][3]{},digt[3][3][3]{},dgamma[3][3][3][3]{};
 for(int a=0;a<3;++a)for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int l=0;l<3;++l)for(int n=0;n<3;++n){di[a][i][j]-=g.inverse[i][l]*m.dg[a][l][n]*g.inverse[n][j];digt[a][i][j]-=gt.inverse[i][l]*u.metric.dg[a][l][n]*gt.inverse[n][j];}
 for(int a=0;a<3;++a)for(int l=0;l<3;++l)for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int n=0;n<3;++n)dgamma[a][l][i][j]+=.5*(di[a][l][n]*(m.dg[i][n][j]+m.dg[j][n][i]-m.dg[n][i][j])+g.inverse[l][n]*(m.ddg[a][i][n][j]+m.ddg[a][j][n][i]-m.ddg[a][n][i][j]));
 auto on=Omega(Lift(u),p);const double w=on.normal.v;double b[8]{},db[3][8]{};
 b[7]=-(6*alpha*w+3*kap)/o;
 for(int a=0;a<3;++a)db[a][7]=-6*(u.alpha.d[a]*w+alpha*on.dnormal[a].v)/o+(6*alpha*w+3*kap)*p.domega[a]/(o*o);
 for(int l=0;l<3;++l)for(int j=0;j<3;++j){const double v=o*u.chi.d[j]+2*chi*p.domega[j];b[4+l]+=alpha*gt.inverse[j][l]*v;
  for(int a=0;a<3;++a)db[a][4+l]+=(u.alpha.d[a]*gt.inverse[j][l]+alpha*digt[a][j][l])*v+alpha*gt.inverse[j][l]*(p.domega[a]*u.chi.d[j]+o*u.chi.dd[a][j]+2*u.chi.d[a]*p.domega[j]+2*chi*p.omega_hessian[a][j]);}
 C B=0,dB[3]{};for(int l=0;l<8;++l){B+=b[l]*c.q[l];for(int a=0;a<3;++a)dB[a]+=db[a][l]*c.q[l]+b[l]*c.d[a][l];}
 C s[3][3]{},ds[3][3][3]{};for(int i=0;i<3;++i)for(int j=0;j<3;++j){C z=c.d[i][4+j]+c.d[j][4+i];for(int l=0;l<3;++l)z-=2*g.connection[l][i][j]*c.q[4+l];s[i][j]=lapse*z+m.g[i][j]*B/3.;
  for(int a=0;a<3;++a){C dz=c.dd[a][i][4+j]+c.dd[a][j][4+i];for(int l=0;l<3;++l)dz-=2.*(dgamma[a][l][i][j]*c.q[4+l]+g.connection[l][i][j]*c.d[a][4+l]);ds[a][i][j]=dlapse[a]*z+lapse*dz+(m.dg[a][i][j]*B+m.g[i][j]*dB[a])/3.;}}
 std::array<C,8> out{};out[0]=2*lapse*K*c.q[0];for(int a=0;a<3;++a)out[0]+=p.beta[a]*c.d[a][0];
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){C dm=c.d[i][1+j];for(int l=0;l<3;++l)dm-=g.connection[l][i][j]*c.q[1+l];out[0]-=2*lapse*g.inverse[i][j]*dm+4*g.inverse[i][j]*c.q[1+i]*dlapse[j];double kup=0;for(int a=0;a<3;++a)for(int b=0;b<3;++b)kup+=g.inverse[i][a]*g.inverse[j][b]*(u.a.k[a][b]/(o*chi)+m.g[a][b]*K/3);out[0]+=2*(K*g.inverse[i][j]-kup)*s[i][j];}
 for(int i=0;i<3;++i){out[1+i]=lapse*K*c.q[1+i]-.5*lapse*c.d[i][0]-c.q[0]*dlapse[i];for(int a=0;a<3;++a)out[1+i]+=p.beta[a]*c.d[a][1+i]+c.q[1+a]*u.beta.d[i][a];
  for(int j=0;j<3;++j)for(int a=0;a<3;++a){C cov=ds[a][i][j];for(int l=0;l<3;++l)cov-=g.connection[l][a][i]*s[l][j]+g.connection[l][a][j]*s[i][l];out[1+i]+=g.inverse[j][a]*cov-di[i][j][a]*s[j][a]-g.inverse[j][a]*ds[i][j][a];}}
 out[7]=alpha*c.q[0]/(2*o)-(3*alpha*w+2*kap)*c.q[7]/o;
 for(int i=0;i<3;++i){out[7]+=p.beta[i]*c.d[i][7];for(int j=0;j<3;++j){C cov=c.d[i][4+j];for(int l=0;l<3;++l)cov-=gt.connection[l][i][j]*c.q[4+l];out[7]+=alpha*o*chi*gt.inverse[i][j]*cov;}}
 for(int i=0;i<3;++i){out[4+i]=alpha*(c.q[1+i]+c.d[i][7])/o-(2*alpha*K/3+kap)*c.q[4+i]/o;for(int j=0;j<3;++j){out[4+i]+=p.beta[j]*c.d[j][4+i]+u.beta.d[i][j]*c.q[4+j];for(int l=0;l<3;++l)for(int m=0;m<3;++m)out[4+i]+=(u.metric.g[i][j]*u.beta.d[l][j]-2*alpha*u.a.k[i][l]*(j==l))*gt.inverse[l][m]*c.q[4+m];}}
 return out;
}
