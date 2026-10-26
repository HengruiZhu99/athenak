// Included after the archived point-driver helpers; independent native checks.
struct FullExtra {Real row_error=0,residual_error=0,pivot=0,U[4][40]{};bool certified=false;};
void ShiftFullColumn(Jet&u,const Jet&ref,const bh::Hierarchy&b,int column,Real amplitude) {
 Real before[10],after[10],desired[10]{};cs::full_canonical::Momentum(u,ref,b.omega,before);
 if(column>=10){const int axis=(column-10)/10,field=(column-10)%10;Real q[10]{},m[3][3]{};q[field]=amplitude;ck::InverseMetricChart(ck::MetricFrame(u),q+1,m);
  u.chi.d[axis]+=q[0];u.alpha.d[axis]+=q[6];for(int i=0;i<3;++i){u.beta.d[axis][i]+=q[7+i];for(int j=0;j<3;++j)u.metric.dg[axis][i][j]+=m[i][j];}}
 cs::full_canonical::Momentum(u,ref,b.omega,after);for(int j=0;j<10;++j)desired[j]=before[j]-after[j]+((column==j)?amplitude:0);
 h::Z4cRHS<Real> p{};ck::Lift(u,b.omega,desired,p);u.trace.value+=p.trace;u.theta.value+=p.theta;for(int i=0;i<3;++i){u.lambda.value[i]+=p.lambda[i];for(int j=0;j<3;++j)u.a.k[i][j]+=p.a[i][j];}
}
FullExtra CheckFull(const Jet&u,const h::LayerPoint<Real>&p,const bh::Hierarchy&b) {
 FullExtra e;b.PrincipalRows(u,e.U);
 for(int col=0;col<40;++col){Jet plus=u,minus=u;ShiftFullColumn(plus,p.state,b,col,1);ShiftFullColumn(minus,p.state,b,col,-1);Real up[4],um[4];NativeU(plus,p,up);NativeU(minus,p,um);
  for(int row=0;row<4;++row)e.row_error=std::max(e.row_error,fabs((up[row]-um[row])/2-e.U[row][col])/std::max(Real(1),fabs(e.U[row][col])));}
 Real data[6]{},incoming[10],outgoing[10],target[10],forcing[4],residual[10];b.Target(u,p.state,data,incoming,outgoing,target,forcing);b.Residual(u,p.state,data,residual);
 for(int j=0;j<10;++j)e.residual_error=std::max(e.residual_error,fabs(residual[j]-2*b.c*(incoming[j]-target[j]))/std::max(Real(1),fabs(residual[j])));
 e.certified=b.PortCertificate(u,e.pivot);return e;
}
