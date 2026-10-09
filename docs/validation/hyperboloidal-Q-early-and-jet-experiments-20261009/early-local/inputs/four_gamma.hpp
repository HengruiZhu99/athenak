// Read-only source origin: continuum/preferred/projection_audit.cpp FourGamma.
// Independent 4D Christoffel contraction using metric derivatives in all four
// directions. Geometry time derivatives come from the actual tensor RHS.
void FourGamma(const hyp::Z4cJet<double>& u,const hyp::Z4cRHS<double>& geom,
               const hyp::GaugeRHS<double>& gauge,double gamma[4]) {
  const auto b=hyp::PenroseMetric(u.metric,u.chi);
  const auto gb=hyp::Geometry(b);
  const double alpha=u.alpha.value;
  double g[4][4]{}, inv[4][4]{}, dg[4][4][4]{};
  double bd[4][3][3]{};
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){
    g[i+1][j+1]=b.g[i][j];
    inv[i+1][j+1]=gb.inverse[i][j]-u.beta.value[i]*u.beta.value[j]/(alpha*alpha);
    bd[0][i][j]=geom.metric[i][j]/u.chi.value
                -u.metric.g[i][j]*geom.chi/(u.chi.value*u.chi.value);
    for(int d=0;d<3;++d)bd[d+1][i][j]=b.dg[d][i][j];
  }
  g[0][0]=-alpha*alpha;inv[0][0]=-1/(alpha*alpha);
  for(int i=0;i<3;++i){
    inv[0][i+1]=inv[i+1][0]=u.beta.value[i]/(alpha*alpha);
    for(int j=0;j<3;++j){
      g[0][i+1]+=b.g[i][j]*u.beta.value[j];
      g[0][0]+=b.g[i][j]*u.beta.value[i]*u.beta.value[j];
    }
    g[i+1][0]=g[0][i+1];
  }
  for(int d=0;d<4;++d){
    const double da=d==0?gauge.alpha:u.alpha.d[d-1];
    double db[3];for(int i=0;i<3;++i)db[i]=d==0?gauge.beta[i]:u.beta.d[d-1][i];
    dg[d][0][0]=-2*alpha*da;
    for(int i=0;i<3;++i)for(int j=0;j<3;++j){
      dg[d][i+1][j+1]=bd[d][i][j];
      dg[d][0][i+1]+=bd[d][i][j]*u.beta.value[j]+b.g[i][j]*db[j];
      dg[d][0][0]+=bd[d][i][j]*u.beta.value[i]*u.beta.value[j]
                   +2*b.g[i][j]*u.beta.value[i]*db[j];
    }
    for(int i=0;i<3;++i)dg[d][i+1][0]=dg[d][0][i+1];
  }
  for(int a=0;a<4;++a)for(int b=0;b<4;++b)for(int c=0;c<4;++c)
    for(int l=0;l<4;++l)gamma[a]+=.5*inv[b][c]*inv[a][l]
      *(dg[b][l][c]+dg[c][l][b]-dg[l][b][c]);
}

