// Diagnostic explicit coordinate-wave lower sources; no production integration.
struct WaveSourceCheck {double theta=0,zup=0,ediv=0,negative=0;};
WaveSourceCheck WaveSources(const h::Z4cJet<AD>&u,const h::OmegaJet<AD>&o,h::Z4cJet<AD>timed,h::OmegaJet<AD>ot,double k1,double k2,const SecondTimeCheck&oracle){
 WaveSourceCheck out;h::SetStationaryOmegaNormal(timed.alpha.value,timed.beta.value,timed.alpha.d,timed.beta.d,ot);
 auto gt=h::Geometry(u.metric);auto con=h::EvolvedConstraints(u,o);AD N=u.alpha.value/o.omega,K=u.trace.value+2*u.theta.value,d=3*o.normal+k1*(2+k2),e=6*o.normal+3*k1*(1+k2),C[3]{},Z[3]{},Zc[3]{},Mup[3]{},Fi[3]{},F=con.hamiltonian/2-d*u.theta.value;
 h::ScalarJet<AD>cp{};cp.value=o.omega*o.omega*u.chi.value;for(int i=0;i<3;++i){cp.d[i]=2*o.omega*o.gradient[i]*u.chi.value+o.omega*o.omega*u.chi.d[i];for(int j=0;j<3;++j)cp.dd[i][j]=2*(o.gradient[i]*o.gradient[j]+o.omega*o.hessian[i][j])*u.chi.value+2*o.omega*(o.gradient[i]*u.chi.d[j]+o.gradient[j]*u.chi.d[i])+o.omega*o.omega*u.chi.dd[i][j];}
 auto gam=h::PenroseMetric(u.metric,cp);auto gg=h::Geometry(gam);AD E[3][3]{},S[3][3]{},L{},divcov{},CZ{};
 for(int i=0;i<3;++i){C[i]=o.gradient[i]/o.omega+u.chi.d[i]/(2*u.chi.value);Z[i]=cp.value*(u.lambda.value[i]-gt.contracted[i])/2;for(int j=0;j<3;++j)Zc[i]+=u.metric.g[i][j]*(u.lambda.value[j]-gt.contracted[j])/2;}
 for(int i=0;i<3;++i){CZ+=C[i]*Z[i];F+=partialPolynomial(Z[i],i)-2*C[i]*Z[i];divcov+=partialPolynomial(Z[i],i)-3*C[i]*Z[i];for(int j=0;j<3;++j){Mup[i]+=gg.inverse[i][j]*con.momentum[j];Fi[i]+=gg.inverse[i][j]*(con.momentum[j]+u.theta.d[j]);}Fi[i]-=k1*Z[i];}
 L=2*N*CZ-N*e*u.theta.value;
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){S[i][j]=partialPolynomial(Zc[j],i)+partialPolynomial(Zc[i],j);for(int k=0;k<3;++k)S[i][j]-=2*gg.connection[k][i][j]*Zc[k];E[i][j]=N*S[i][j]+gam.g[i][j]*L/3;}
 AD traceE{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)traceE+=gg.inverse[i][j]*E[i][j];
 double kr[3][3]{},D0g[3][3]{},D0C[3]{},D0N=timed.alpha.value.d[0]/o.omega.v,D0d=3*ot.normal.d[0];
 for(int k=0;k<3;++k){D0N-=u.beta.value[k].v*N.d[k];D0d-=u.beta.value[k].v*d.d[k];}
 for(int i=0;i<3;++i){D0C[i]=(N.d[i]*K.v+N.v*(u.trace.d[i].v+2*u.theta.d[i].v))/3;for(int j=0;j<3;++j){D0C[i]-=u.beta.dd[i][j][j].v/3;D0C[i]+=u.beta.d[i][j].v*C[j].v;}}
 double RH=2*N.v*K.v*con.hamiltonian.v;
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){kr[i][j]=K.v*gg.inverse[i][j].v/3;for(int a=0;a<3;++a)for(int b=0;b<3;++b)kr[i][j]+=o.omega.v*o.omega.v*o.omega.v*u.chi.value.v*gt.inverse[i][a].v*gt.inverse[j][b].v*u.a.k[a][b].v;D0g[i][j]=2*N.v*kr[i][j];for(int k=0;k<3;++k)D0g[i][j]-=gg.inverse[i][k].v*u.beta.d[k][j].v+gg.inverse[j][k].v*u.beta.d[k][i].v;RH+=2*(K.v*gg.inverse[i][j].v-kr[i][j])*E[i][j].v;}
 for(int i=0;i<3;++i)RH-=4*Mup[i].v*N.d[i];
 double thetaPrincipal=0,thetaLower=D0N*F.v+N.v*RH/2-N.v*D0d*u.theta.value.v-N.v*N.v*d.v*F.v,thetaGroup=0;
 for(int i=0;i<3;++i){thetaGroup+=3*N.v*N.v*C[i].v*Mup[i].v+N.v*N.d[i]*Fi[i].v-N.v*N.v*k1*Z[i].d[i]-2*N.v*D0C[i]*Z[i].v-2*N.v*N.v*C[i].v*Fi[i].v;for(int j=0;j<3;++j){thetaPrincipal+=N.v*N.v*gg.inverse[i][j].v*u.theta.dd[i][j].v;thetaGroup+=N.v*N.v*gg.inverse[i][j].d[i]*u.theta.d[j].v+N.v*u.beta.d[i][j].v*Z[i].d[j];}}
 thetaLower+=thetaGroup;out.theta=std::abs(thetaPrincipal+thetaLower-oracle.oracleTheta)/std::max(1.,std::abs(oracle.oracleTheta));out.negative=std::abs(thetaGroup)/std::max(1.,std::abs(oracle.oracleTheta));
 double covLap[3]{},metricLower[3]{},Edirect[3]{},Eexpanded[3]{},simple[3]{};
 for(int j=0;j<3;++j){Edirect[j]=-traceE.d[j];simple[j]=N.v*K.v*con.momentum[j].v-con.hamiltonian.v*N.d[j]-2*L.d[j]/3;for(int k=0;k<3;++k){simple[j]+=con.momentum[k].v*u.beta.d[j][k].v;for(int a=0;a<3;++a){simple[j]+=gg.inverse[k][a].v*N.d[a]*(S[k][j].v-2*gam.g[k][j].v*divcov.v);}}
  for(int a=0;a<3;++a)for(int b=0;b<3;++b){double lap=Zc[j].dd[a][b],tl=0;for(int m=0;m<3;++m)tl+=gam.dg[a][j][m].v*Z[m].d[b]+gam.dg[b][j][m].v*Z[m].d[a]+gam.ddg[a][b][j][m].v*Z[m].v;
   for(int k=0;k<3;++k){double db=Zc[k].d[b],dk=Zc[j].d[k];for(int h=0;h<3;++h){db-=gg.connection[h][b][k].v*Zc[h].v;dk-=gg.connection[h][k][j].v*Zc[h].v;}double correction=gg.connection[k][b][j].d[a]*Zc[k].v+gg.connection[k][b][j].v*Zc[k].d[a]+gg.connection[k][a][b].v*dk+gg.connection[k][a][j].v*db;lap-=correction;tl-=correction;}
   covLap[j]+=gg.inverse[a][b].v*lap;for(int i=0;i<3;++i)metricLower[i]+=gg.inverse[i][j].v*gg.inverse[a][b].v*tl;
   double de=E[b][j].d[a];for(int k=0;k<3;++k)de-=gg.connection[k][a][b].v*E[k][j].v+gg.connection[k][a][j].v*E[b][k].v;Edirect[j]+=gg.inverse[a][b].v*de;
  }
  Eexpanded[j]=N.v*(covLap[j]-divcov.d[j])-2*L.d[j]/3;for(int k=0;k<3;++k){Eexpanded[j]+=N.v*gg.ricci[j][k].v*Z[k].v;for(int a=0;a<3;++a)Eexpanded[j]+=gg.inverse[k][a].v*N.d[a]*(S[k][j].v-2*gam.g[k][j].v*divcov.v);}
  out.ediv=std::max(out.ediv,std::abs(Edirect[j]-Eexpanded[j])/std::max(1.,std::abs(Edirect[j])));
 }
 for(int i=0;i<3;++i){double principal=0,lower=D0N*Fi[i].v-N.v*N.v*k1*Fi[i].v,geogroup=N.v*N.v*metricLower[i];for(int a=0;a<3;++a)for(int b=0;b<3;++b)principal+=N.v*N.v*gg.inverse[a][b].v*Z[i].dd[a][b];
  for(int j=0;j<3;++j){lower+=N.v*D0g[i][j]*(con.momentum[j].v+u.theta.d[j].v)+N.v*gg.inverse[i][j].v*(simple[j]+N.d[j]*F.v+N.v*CZ.d[j]-N.v*(d*u.theta.value).d[j]);for(int k=0;k<3;++k){lower+=N.v*gg.inverse[i][j].v*u.beta.d[j][k].v*u.theta.d[k].v;geogroup+=N.v*N.v*gg.inverse[i][j].v*gg.ricci[j][k].v*Z[k].v;}}
  lower+=geogroup;out.zup=std::max(out.zup,std::abs(principal+lower-oracle.oracleZ[i])/std::max(1.,std::abs(oracle.oracleZ[i])));out.negative=std::max(out.negative,std::abs(geogroup)/std::max(1.,std::abs(oracle.oracleZ[i])));
 }
 return out;
}
