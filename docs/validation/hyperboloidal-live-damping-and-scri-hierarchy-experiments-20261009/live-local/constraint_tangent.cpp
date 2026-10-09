// Exact dual20 derivative plus refined FD of coefficient fields, scratch only.
#define Point FrozenHelperPoint
#include "profile_helpers.hpp"
#undef Point
bool g_liveprofile=true;
Sample Point(const hyp::LayerReference<double>&ref,const std::array<double,3>&x,
             double k,const std::array<double,3>&n,double kappa) {
 const auto p=ref.At(x[0],x[1],x[2]);Sample s;
 for(int col=0;col<20;++col)for(int phase=0;phase<2;++phase){
  auto u=Lift(p.state);J seed(D(0,phase?0:1));
  for(int i=0;i<3;++i){seed.d[i]=D(0,phase?k*n[i]:0);
   for(int j=0;j<3;++j)seed.dd[i][j]=D(0,phase?0:-k*k*n[i]*n[j]);}
  Seed(u,col,seed);Consistent(u);
  const auto f0=Evaluate(p,u,ref.curvature_radius,kappa,g_liveprofile?1:0,true);
  const C factor=phase?I:C(1,0);
  for(int row=0;row<20;++row)s.f[row][col]+=factor*f0[row].d;
  const auto q=Constraints(u,p);for(int row=0;row<8;++row)s.q[row][col]+=factor*q[row];
 }
 return s;
}
#include "subsidiary_profile.hpp"
void Print(const M&m,int rows){std::cout<<'[';for(int i=0;i<rows;++i){if(i)std::cout<<',';std::cout<<'[';for(int j=0;j<20;++j){if(j)std::cout<<',';std::cout<<'['<<m[i][j].real()<<','<<m[i][j].imag()<<']';}std::cout<<']';}std::cout<<']';}
int main(int argc,char**argv){g_liveprofile=!(argc>1&&std::string(argv[1])=="base");std::cout<<std::setprecision(17)<<'[';bool first=true;for(double a:{.5,.75,1.,2.}){const hyp::LayerReference<double> ref(1.,a,{true,.05,.95});
 for(double r:{.1,.15,.18,.2,.225,.25,.28,.3,.5,.65,.75,.85,.9,.95,.98,.992845500317144,.9983726993838523})for(double k:{0.,1.,2.,4.,8.,16.,32.,64.,128.,256.})for(bool oblique:{false,true})for(int level=0;level<5;++level){ if(a!=.5&&(!((r==.2)||(r==.225)||(r==.25)||(r==.3)||(r==.65)||(r==.95)||(r==.9983726993838523))||!((k==0)||(k==64)||(k==256))))continue;const double h=std::min(.002,(1-r)/8)/std::pow(2.,level);
 const std::array<double,3>x={r,0,0},n=oblique?std::array<double,3>{.36,-.48,.8}:std::array<double,3>{1,0,0};const double kappa=10;const auto p=ref.At(x[0],x[1],x[2]);const auto base=Point(ref,x,k,n,kappa);M df[3]{},ddf[3][3]{},dq[3]{},ddq[3][3]{};
 const int offsets[4]={-2,-1,1,2};const double dc[4]={1.,-8.,8.,-1.};
 for(int axis=0;axis<3;++axis){Sample side[4];for(int a=0;a<4;++a){auto y=x;y[axis]+=offsets[a]*h;side[a]=Point(ref,y,k,n,kappa);}
  for(int row=0;row<20;++row)for(int col=0;col<20;++col){for(int a=0;a<4;++a){df[axis][row][col]+=dc[a]*side[a].f[row][col]/(12*h);dq[axis][row][col]+=dc[a]*side[a].q[row][col]/(12*h);}
   ddq[axis][axis][row][col]=(-side[3].q[row][col]+16.*side[2].q[row][col]-30.*base.q[row][col]+16.*side[1].q[row][col]-side[0].q[row][col])/(12*h*h);
   ddf[axis][axis][row][col]=(-side[3].f[row][col]+16.*side[2].f[row][col]-30.*base.f[row][col]+16.*side[1].f[row][col]-side[0].f[row][col])/(12*h*h);}}
 for(int i=0;i<3;++i)for(int j=i+1;j<3;++j)for(int a=0;a<4;++a)for(int b=0;b<4;++b){auto y=x;y[i]+=offsets[a]*h;y[j]+=offsets[b]*h;auto v=Point(ref,y,k,n,kappa);for(int row=0;row<20;++row)for(int col=0;col<20;++col){ddf[i][j][row][col]+=dc[a]*dc[b]*v.f[row][col]/(144*h*h);ddq[i][j][row][col]+=dc[a]*dc[b]*v.q[row][col]/(144*h*h);}}
 M exact{},frozen{},prediction{},subsidiary{},no_gradient_control{},constraint_generator{};const auto gt=hyp::Geometry(p.state.metric);hyp::OmegaJet<D> op=Omega(Lift(p.state),p);
 for(int col=0;col<20;++col){for(int phase=0;phase<2;++phase){Jet ue=Lift(p.state),uf=Lift(p.state);auto part=[phase](C z){return phase?z.imag():z.real();};
  for(int row=0;row<20;++row){J e(D(0,part(base.f[row][col]))),f=e;for(int i=0;i<3;++i){e.d[i]=D(0,part(df[i][row][col]+I*k*n[i]*base.f[row][col]));f.d[i]=D(0,part(I*k*n[i]*base.f[row][col]));for(int j=0;j<3;++j){const C dd=i<=j?ddf[i][j][row][col]:ddf[j][i][row][col];e.dd[i][j]=D(0,part(dd+I*k*(n[i]*df[j][row][col]+n[j]*df[i][row][col])-k*k*n[i]*n[j]*base.f[row][col]));f.dd[i][j]=D(0,part(-k*k*n[i]*n[j]*base.f[row][col]));}}Seed(ue,row,e);Seed(uf,row,f);}Consistent(ue);Consistent(uf);auto ce=Constraints(ue,p),cf=Constraints(uf,p);for(int row=0;row<8;++row){exact[row][col]+=(phase?I:C(1))*ce[row];frozen[row][col]+=(phase?I:C(1))*cf[row];}}
  ConstraintJet cj{};for(int row=0;row<8;++row){cj.q[row]=base.q[row][col];for(int a=0;a<3;++a){cj.d[a][row]=dq[a][row][col]+I*k*n[a]*base.q[row][col];for(int b=0;b<3;++b){const C dd=a<=b?ddq[a][b][row][col]:ddq[b][a][row][col];cj.dd[a][b][row]=dd+I*k*(n[a]*dq[b][row][col]+n[b]*dq[a][row][col])-k*k*n[a]*n[b]*base.q[row][col];}}}auto pred=Subsidiary(p,kappa,cj);
  const auto no_derivative=SubsidiaryProfile(p,kappa,cj,false);
  for(int row=0;row<8;++row){subsidiary[row][col]=pred[row];no_gradient_control[row][col]=no_derivative[row];}

 }
 for(int col=0;col<8;++col){ConstraintJet cj{};cj.q[col]=1.;for(int a=0;a<3;++a){cj.d[a][col]=I*k*n[a];for(int b=0;b<3;++b)cj.dd[a][b][col]=-k*k*n[a]*n[b];}auto v=Subsidiary(p,kappa,cj);for(int row=0;row<8;++row)constraint_generator[row][col]=v[row];}
 if(!first)std::cout<<',';first=false;std::cout<<"{\"a\":"<<a<<",\"kappa\":"<<kappa<<",\"r\":"<<r<<",\"k\":"<<k<<",\"h\":"<<h<<",\"level\":"<<level<<",\"oblique\":"<<oblique<<",\"Q\":";Print(base.q,8);std::cout<<",\"D\":";Print(exact,8);std::cout<<",\"frozen\":";Print(frozen,8);std::cout<<",\"subsidiary\":";Print(subsidiary,8);std::cout<<",\"without_dkappa2\":";Print(no_gradient_control,8);std::cout<<",\"constraint_generator\":";Print(constraint_generator,8);std::cout<<'}';
 }}std::cout<<"]\n";
}
