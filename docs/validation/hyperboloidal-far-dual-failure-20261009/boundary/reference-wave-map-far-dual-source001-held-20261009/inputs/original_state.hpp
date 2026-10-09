hyp::Z4cJet<double> State(const hyp::LayerPoint<double>&p,int family){
 auto u=p.state;auto apply=[&](int f){
 if(f==1){u.alpha.value=1.1*p.alpha;for(int j=0;j<3;++j)u.alpha.d[j]=1.1*p.dalpha[j]+.002*e[j];}
 if(f==2){u.chi.value=.8*p.state.chi.value;for(int j=0;j<3;++j)u.chi.d[j]=.8*p.state.chi.d[j]+.003*e[j];}
 if(f==3)for(int i=0;i<3;++i){u.beta.value[i]=p.beta[i]+.01*e[i];for(int j=0;j<3;++j)u.beta.d[j][i]=p.state.beta.d[j][i]+.002*e[i]*e[j];}
 if(f==4)u.trace.value=p.k_physical+.004;
 if(f==5)for(int i=0;i<3;++i)u.lambda.value[i]=p.state.lambda.value[i]+.004*e[i];
 if(f==6)for(int j=0;j<3;++j)u.chi.d[j]=p.state.chi.d[j]+.02*e[j];
 if(f==7){const double d[3]={1.25,.8,1};for(int i=0;i<3;++i)for(int j=0;j<3;++j){u.metric.g[i][j]=d[i]*d[j]*p.state.metric.g[i][j];u.a.k[i][j]=d[i]*d[j]*p.state.a.k[i][j];for(int k=0;k<3;++k){u.metric.dg[k][i][j]=d[i]*d[j]*p.state.metric.dg[k][i][j];u.a.dk[k][i][j]=d[i]*d[j]*p.state.a.dk[k][i][j];for(int l=0;l<3;++l)u.metric.ddg[k][l][i][j]=d[i]*d[j]*p.state.metric.ddg[k][l][i][j];}}}
 if(f==8){u.theta.value=.07;u.trace.value=p.k_physical+.004;}
 };
 if(family==9){for(int f=1;f<=8;++f)apply(f);}else apply(family);
 if(family>=10){const double av[4]={1e-150,1e-150,1e60,1e100},cv[4]={1e-150,1e60,1e-150,1e-150};u.alpha.value=av[family-10];u.chi.value=cv[family-10];
  const double ad[3]={.001,-.0015,.0005},cd[3]={.002,-.001,.0005};
  for(int j=0;j<3;++j){u.alpha.d[j]=u.alpha.value*ad[j];u.chi.d[j]=u.chi.value*cd[j];u.beta.value[j]=p.beta[j]+.003*e[j];u.lambda.value[j]=p.state.lambda.value[j]+.004*e[j];}u.trace.value=p.k_physical+.004;u.theta.value=.07;}
 return u;
}
