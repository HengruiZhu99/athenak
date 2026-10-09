#define SCRI_FIRSTJET_ONLY
#include "taylor_kernel.cpp"
#undef SCRI_FIRSTJET_ONLY
Jet First(const hyp::LayerPoint<double>&p,int trial,const std::array<double,3>*d0=nullptr,const std::array<std::array<double,3>,3>*A0=nullptr){
 auto u=Lift(p.state);const int ti[5]={0,0,0,1,1},tj[5]={0,1,2,1,2};
 double h[3][3]{};for(int c=0;c<5;++c)h[ti[c]][tj[c]]=h[tj[c]][ti[c]]=.1*std::sin((trial+1)*(c+1.));h[2][2]=-h[0][0]-h[1][1];
 for(int c=0;c<5;++c)Seed(u,7+c,Linear(J(D(h[ti[c]][tj[c]]))));
 J chi0;for(int i=0;i<3;++i)for(int j=0;j<3;++j)chi0=chi0+J(D(h[i][j]))*Unit(p,i)*Unit(p,j);Seed(u,1,Linear(chi0));
 for(int c=0;c<20;++c)Seed(u,c,Linear(J(D(.1*std::cos((trial+1)*(c+2.))))*Power(p,1)));
 if(d0){for(int i=0;i<3;++i)Seed(u,17+i,Linear(J(D((*d0)[i]))));for(int c=0;c<5;++c)Seed(u,12+c,Linear(J(D((*A0)[ti[c]][tj[c]]))));}
 Consistent(u);return u;
}
std::array<double,20> R0Formula(const Jet&u,double a,int profile){
 std::array<double,20>out{};const auto gt=hyp::Geometry(u.metric),gb=hyp::Geometry(hyp::PenroseMetric(u.metric,u.chi));
 const double da=u.alpha.value.d,c=u.chi.value.d,P=u.trace.value.d,T=u.theta.value.d;
 const double diff=c-u.metric.g[0][0].d,k2=profile?2/(10*a*a)-1:0,eta=1.5/(a*a),C=1/(3*a);
 out[0]=(-P-3*da+u.beta.value[0].d)/(a*a);
 out[1]=2*(P+2*T-3*da-3*u.beta.value[0].d)/(3*a);
 out[2]=-2*(P+2*T)/(a*a)-3*diff/(a*a*a)+10*(1-k2)*T;
 out[3]=-2*P/(a*a)-(1/(a*a)+10*(2+k2))*T-3*diff/(a*a*a);
 for(int i=0;i<3;++i)out[4+i]=-eta*(u.beta.value[i].d+(i==0?C*diff:0));
 double H[3][3]{},d[3]{},tr=0;
 for(int i=0;i<3;++i){d[i]=u.lambda.value[i].d-gt.contracted[i].d;for(int j=0;j<3;++j){H[i][j]=u.metric.g[i][j].d+gb.connection[0][i][j].d;if(i==j)tr+=H[i][j];}}
 const int ti[5]={0,0,0,1,1},tj[5]={0,1,2,1,2};
 for(int k=0;k<5;++k){const int i=ti[k],j=tj[k];double h=H[i][j]-(i==j?tr/3:0);const double z=.5*((i==0?d[j]:0)+(j==0?d[i]:0)-(i==j?2*d[0]/3:0));out[12+k]=2*(h-z-u.a.k[i][j].d)/(a*a);}
 for(int i=0;i<3;++i)out[17+i]=-(10-2/(a*a))*d[i]-2*(2*u.trace.d[i].d+u.theta.d[i].d)/(3*a)+4*u.a.k[i][0].d/(a*a);
 return out;
}
int main(){double pole=0,theta_error=0,angular=0,operator_error=0;int rows=0,operator_rows=0;
 for(double a:{.5,.75,1.,2.})for(int profile:{0,1}){hyp::LayerReference<double>ref(1,a,{true,.05,.95});const auto p=ref.At(1.,0.,0.);
 for(int family=0;family<4;++family)for(int col=0;col<20;++col){auto u=Lift(p.state);J j(D(1));if(family==1)j=Power(p,1);if(family>=2)j=Unit(p,family-1);Seed(u,col,Linear(j));Consistent(u);const auto actual=Parts(p,u,a,profile),formula=R0Formula(u,a,profile);
 for(int i=0;i<20;++i)operator_error=std::max(operator_error,std::abs(actual.s[i].d-formula[i]));++operator_rows;}
 }

 for(double a:{.5,.75,1.,2.})for(int profile:{0,1})for(int trial=0;trial<20;++trial){hyp::LayerReference<double>ref(1,a,{true,.05,.95});const auto p=ref.At(1.,0.,0.);auto u=First(p,trial);
 const auto gt=hyp::Geometry(u.metric),gb=hyp::Geometry(hyp::PenroseMetric(u.metric,u.chi));
 double H[3][3]{},tr=0;for(int i=0;i<3;++i)for(int j=0;j<3;++j){H[i][j]=u.metric.g[i][j].d+gb.connection[0][i][j].d; if(i==j)tr+=H[i][j];}
 for(int i=0;i<3;++i)H[i][i]-=tr/3;
 const double P1=.1*std::cos((trial+1)*4.),T1=.1*std::cos((trial+1)*5.);
 const double d[3]={(12*H[0][0]+4*P1+2*T1)/(30*a*a+2),4*H[0][1]/(10*a*a),4*H[0][2]/(10*a*a)};
 std::array<double,3>lambda{};std::array<std::array<double,3>,3>A{};
 for(int i=0;i<3;++i){lambda[i]=gt.contracted[i].d+d[i];for(int j=0;j<3;++j)A[i][j]=H[i][j]-.5*((i==0?d[j]:0)+(j==0?d[i]:0)-(i==j?2*d[0]/3:0));}
 u=First(p,trial,&lambda,&A);const auto rs=Parts(p,u,a,profile);for(auto v:rs.s)pole=std::max(pole,std::abs(v.d));
 // chi0(n)=n^i h_ij n^j, so its angular derivative is not free.
 for(int j=1;j<3;++j)angular=std::max(angular,std::abs(u.chi.d[j].d-2*u.metric.g[0][j].d));
 const auto bb=hyp::PenroseMetric(u.metric,u.chi);const auto geom=hyp::Geometry(bb);D lap{};
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){D hess=p.omega_hessian[i][j];for(int k=0;k<3;++k)hess-=geom.connection[k][i][j]*D(p.domega[k]);lap+=geom.inverse[i][j]*hess;}
 const double alpha1=.1*std::cos((trial+1)*2.),beta1=.1*std::cos((trial+1)*6.);
 const double q=P1-3*(alpha1+beta1),k2=profile?(2/(10*a*a)-1):0;
 const double chi1=.1*std::cos((trial+1)*3.),hrr1=.1*std::cos((trial+1)*9.);
 const double N1=(chi1-hrr1)/(a*a)+2*(alpha1+beta1)/a;
 const double expected=2*lap.d/a-2*q/(a*a)-10*(2+k2)*T1-3*N1/a;
 // Fourth-order one-sided extrapolation of the assembled actual Theta RHS.
 double f[4];for(int k=0;k<4;++k){const double om=(k+1)*1e-4;const auto pp=ref.At(std::sqrt(1-2*a*om),0.,0.);const auto uu=First(pp,trial,&lambda,&A);const auto rr=Parts(pp,uu,a,profile);f[k]=rr.r[3].d+rr.s[3].d/pp.omega;}
 const double actual=4*f[0]-6*f[1]+4*f[2]-f[3];theta_error=std::max(theta_error,std::abs(actual-expected));++rows;
 }
 std::cout<<std::setprecision(17)<<"{\"operator_rows\":"<<operator_rows<<",\"full_R0_formula_error\":"<<operator_error<<",\"rows\":"<<rows<<",\"all_pole_firstjet_formula_error\":"<<pole<<",\"nonfree_angular_chi_jet_error\":"<<angular<<",\"Theta_corner_actual_limit_error\":"<<theta_error<<"}\n";
}
