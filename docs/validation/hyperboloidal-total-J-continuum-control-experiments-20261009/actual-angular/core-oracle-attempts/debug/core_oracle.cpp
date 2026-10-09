// Independent flat Cauchy-core Cartesian linear formulas; no radial operator.
#define main OriginalAngularBridgeMain
#include "bridge.cpp"
#undef main
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
int main(){try{std::cout<<std::setprecision(17);double rhs_error=0,constraint_error=0,absolute=0;int cases=0;
  for(const X3&x:{X3{0,0,0},X3{.009,-.012,.020},X3{-.018,.024,.032}}){const auto p=reference.At(x[0],x[1],x[2]);if(p.omega!=1||p.alpha!=1)throw std::runtime_error("oracle is outside exact core");
    for(int J=0;J<3;++J)for(int m=0;m<=J;++m)for(int phase=0;phase<(m?2:1);++phase)for(int c=0;c<(J==0?8:(J==1?16:20));++c)for(auto w:{totalj::WJet<double>{1,0,0},totalj::WJet<double>{0,1,0},totalj::WJet<double>{0,0,1}}){
      const auto u=LiftPhysical(p,SeedPhysical(J,m,c,phase,x,w));const auto actual=Derivative(ActualDual(p,u)),expected=FlatFormula(u);const auto d=Difference(actual,expected);rhs_error=std::max(rhs_error,Norm(d)/std::max(1.,Norm(expected)));absolute=std::max(absolute,Max(d));
      const auto q=Constraints(u,p),qc=FlatConstraints(u);for(int i=0;i<8;++i)constraint_error=std::max(constraint_error,std::abs(q[i]-qc[i])/std::max(1.,std::abs(qc[i])));++cases;
    }
  }
  std::cout<<"{\"cases\":"<<cases<<",\"rhs_scaled_error\":"<<rhs_error<<",\"rhs_absolute_max\":"<<absolute<<",\"physical8_constraint_scaled_error\":"<<constraint_error<<",\"includes_origin_and_all_m\":true,\"no_radial_operator\":true}\n";
  return rhs_error<=5e-12&&constraint_error<=5e-12?0:2;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
