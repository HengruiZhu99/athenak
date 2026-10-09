Raw FlatFormula(const Jet&u){const auto a=VariationJets(u);Raw f{};double divbeta=0,lapalpha=0,lapchi=0,divlambda=0;double gamma[3]{};
  for(int i=0;i<3;++i){divbeta+=a[19+i].d[i];lapalpha+=a[18].dd[i][i];lapchi+=a[0].dd[i][i];divlambda+=a[14+i].d[i];}
  TJ g[3][3]{},A[3][3]{};for(int q=0;q<6;++q){g[ti[q]][tj[q]]=g[tj[q]][ti[q]]=a[1+q];A[ti[q]][tj[q]]=A[tj[q]][ti[q]]=a[8+q];}
  for(int i=0;i<3;++i)for(int j=0;j<3;++j)gamma[i]+=g[i][j].d[j];
  f[0]=(2./3.)*(a[7].value+2*a[17].value-divbeta);
  f[7]=-lapalpha+10*a[17].value;f[17]=lapchi+.5*divlambda-20*a[17].value;
  f[18]=-3*a[7].value;
  const double trace=-lapalpha+2*lapchi+divlambda;
  for(int q=0;q<6;++q){const int i=ti[q],j=tj[q];double lapg=0;for(int k=0;k<3;++k)lapg+=g[i][j].dd[k][k];
    f[1+q]=-2*A[i][j].value+a[19+j].d[i]+a[19+i].d[j]-(i==j?2.*divbeta/3:0);
    f[8+q]=-a[18].dd[i][j]-.5*lapg+.5*a[0].dd[i][j]+.5*(a[14+j].d[i]+a[14+i].d[j])+(i==j?.5*lapchi-trace/3:0);
  }
  for(int i=0;i<3;++i){double lapbeta=0,graddiv=0;for(int j=0;j<3;++j){lapbeta+=a[19+i].dd[j][j];graddiv+=a[19+j].dd[i][j];}
    f[14+i]=lapbeta+graddiv/3-(4./3.)*a[7].d[i]-(2./3.)*a[17].d[i]-10*(a[14+i].value-gamma[i]);
    f[19+i]=3.*a[14+i].value/8;
  }return f;
}
std::array<double,8> FlatConstraints(const Jet&u){const auto a=VariationJets(u);std::array<double,8>c{};TJ g[3][3]{},A[3][3]{};for(int q=0;q<6;++q){g[ti[q]][tj[q]]=g[tj[q]][ti[q]]=a[1+q];A[ti[q]][tj[q]]=A[tj[q]][ti[q]]=a[8+q];}
  for(int i=0;i<3;++i){c[0]+=2*a[0].dd[i][i];c[1+i]=-(2./3.)*(a[7].d[i]+2*a[17].d[i]);c[4+i]=.5*a[14+i].value;for(int j=0;j<3;++j){c[0]+=g[i][j].dd[i][j];c[1+i]+=A[i][j].d[j];c[4+i]-=.5*g[i][j].d[j];}}c[7]=a[17].value;return c;
}
