// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Reuse the actual pilot stage, without executing its CLI or advancing time.
#define main FrozenCoupledPilotMain
#include "coupled.cpp"
#undef main

struct ExactRow {int kind,id;Real r,chi,cp,cpp,alpha,ap,app,a,ar,k,kr,b,br,brr;};
h::Z4cJet<Real> ExactJet(const Real x[3],const ExactRow &q){
  h::Z4cJet<Real> u{};Real n[3];for(int a=0;a<3;++a)n[a]=x[a]/q.r;
  auto scalar=[&](Real f,Real d,Real dd){h::ScalarJet<Real> s{};s.value=f;
    for(int a=0;a<3;++a){s.d[a]=d*n[a];for(int b=0;b<3;++b)s.dd[a][b]=dd*n[a]*n[b]+d/q.r*((a==b?1.:0)-n[a]*n[b]);}return s;};
  u.chi=scalar(q.chi,q.cp,q.cpp);u.alpha=scalar(q.alpha,q.ap,q.app);u.trace=scalar(q.k,q.kr,0);
  for(int i=0;i<3;++i){u.metric.g[i][i]=1;u.beta.value[i]=-x[i];u.beta.d[i][i]=-1;
    for(int j=0;j<3;++j){const Real t=1.5*n[i]*n[j]-(i==j?.5:0);u.a.k[i][j]=q.a*t;
      for(int a=0;a<3;++a)u.a.dk[a][i][j]=q.ar*n[a]*t+1.5*q.a/q.r*
        (((a==i?1.:0)-n[a]*n[i])*n[j]+n[i]*((a==j?1.:0)-n[a]*n[j]));}}
  return u;
}
V Coefficients(const Real x[3],const ExactRow &q,int chart){
  V out(fields,0);const auto c=h::WormholeChart::At(x,chart);
  out[0]=std::log(q.chi/c.chi.f);out[18]=std::log(q.alpha/c.alpha.f);out[13]=3+q.k/c.weight.f;
  for(int t=0;t<6;++t){int i=tensor_i[t],j=tensor_j[t];out[7+t]=q.a*(1.5*x[i]*x[j]/(q.r*q.r)-(i==j?.5:0))/c.weight.f;}return out;
}
struct Norm {Real sum=0,volume=0,maximum=0;void Add(Real e,Real w){sum+=w*e*e;volume+=w;maximum=std::max(maximum,std::abs(e));}
  void Print()const{std::cout<<"{\"rms\":"<<std::sqrt(sum/volume)<<",\"maximum\":"<<maximum<<"}";}};
int main(int argc,char **argv){Kokkos::initialize(argc,argv);int status=0;
try{
  const int n=argc>1?std::stoi(argv[1]):8,chart=argc>4?std::stoi(argv[4]):2;
  const std::string action=argc>2?argv[2]:"coordinates",path=argc>3?argv[3]:"";
  const bool focused=argc>5&&std::string(argv[5])=="focused";
  const bool radial=argc>6&&std::string(argv[6])=="radial";
  const int surface_points=argc>7?std::stoi(argv[7]):6;
  const int bulk=argc>8?std::stoi(argv[8]):2;
  if(bulk<1||bulk>4||size_t(n)*n*n*(bulk*bulk*bulk+(focused?18:12))>50000)
    throw std::invalid_argument("bounded diagnostic cell budget");
  Pilot pilot(cubed_sphere::CompositeMesh(.25,focused?V{.62,.70,1}:V{.65,1},1,n,2,bulk),true,1e-11,4,true,true,chart,surface_points,radial);
  const auto &g=pilot.shift.grid;
  auto coordinate=[&](int point){int m=point/(g.padded*g.padded*g.padded),i=point%g.padded-g.ng,j=point/g.padded%g.padded-g.ng,k=point/(g.padded*g.padded)%g.padded-g.ng;
    const Real q[3]={(i+.5)/n,(j+.5)/n,(k+.5)/n};std::array<Real,3>x;g.mesh.Forward(m,q,x.data());return x;};
  if(action=="coordinates"){
    std::ofstream out(path);out<<std::setprecision(17)<<"kind,id,x,y,z,r\n";
    auto write=[&](int kind,int id,const Real x[3]){Real r=std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
      if(r>1+1e-13)throw std::runtime_error("exact internal ghost lies outside scri");out<<kind<<','<<id<<','<<x[0]<<','<<x[1]<<','<<x[2]<<','<<r<<'\n';};
    for(size_t p=0;p<g.active.size();++p)write(0,p,g.geometry[p].x);
    for(const auto &row:g.halos)if(row.boundary_weight==0){const auto x=coordinate(row.target);write(1,row.target,x.data());}
    for(size_t p=0;p<pilot.scri->rows.size();++p)write(2,p,pilot.scri->rows[p].geometry.x);
  }else{
    std::ifstream input(path);int count;std::string magic;input>>magic>>count;
    if(magic!="EXACT_WORMHOLE_V1")throw std::runtime_error("exact-data format");
    std::vector<ExactRow> active(g.active.size()),halo,surface(pilot.scri->rows.size());
    for(int i=0;i<count;++i){ExactRow q;input>>q.kind>>q.id>>q.r>>q.chi>>q.cp>>q.cpp>>q.alpha>>q.ap>>q.app>>q.a>>q.ar>>q.k>>q.kr>>q.b>>q.br>>q.brr;
      if(!input)throw std::runtime_error("truncated exact data");if(q.kind==0)active.at(q.id)=q;else if(q.kind==1)halo.push_back(q);else surface.at(q.id)=q;}
    std::vector<h::Z4cJet<Real>> exact(active.size());V true_delta(3*active.size());
    for(size_t p=0;p<active.size();++p){const auto &q=active[p];const auto &x=g.geometry[p].x;exact[p]=ExactJet(x,q);const auto c=Coefficients(x,q,chart);
      for(int f=0;f<fields;++f)pilot.state[fields*p+f]=c[f];for(int v=0;v<3;++v)true_delta[3*p+v]=(q.b/q.r+1)*x[v];}
    for(size_t p=0;p<surface.size();++p){const auto c=Coefficients(pilot.scri->rows[p].geometry.x,surface[p],chart);for(int f=0;f<fields;++f)pilot.scri->values[p*fields+f]=c[f];}
    const V ordinary=pilot.Padded(pilot.state);V exact_halo=ordinary;
    for(const auto &q:halo){const auto x=coordinate(q.id);const auto c=Coefficients(x.data(),q,chart);for(int f=0;f<fields;++f)exact_halo[f*g.count+q.id]=c[f];}
    V source(3*active.size());for(size_t p=0;p<active.size();++p){Real s[3];h::EllipticShiftSource(exact[p],h::Geometry(exact[p].metric),0.,s);for(int v=0;v<3;++v)source[3*p+v]=s[v];}
    // An independent discretization control differentiates the full known
    // curvature metric rate, instead of applying a pointwise product rule.
    V curvature(6*g.count,0),conservative_source(source.size());
    std::vector<std::array<Real,3>> zeros(fields);std::vector<std::array<std::array<Real,3>,3>> zz(fields);
    for(int p=0;p<g.count;++p){const auto x=coordinate(p);V values(fields);for(int f=0;f<fields;++f)values[f]=ordinary[f*g.count+p];
      const auto u=h::UnpackComposite(x.data(),values,zeros,zz,true,chart);
      for(int t=0;t<6;++t)curvature[t*g.count+p]=-2*u.alpha.value*u.a.k[tensor_i[t]][tensor_j[t]];}
    for(size_t p=0;p<active.size();++p){h::MetricRate<Real> rate{};for(int t=0;t<6;++t){Real d[3],dd[3][3];g.Derivatives(curvature,t,p,d,dd);
        const int i=tensor_i[t],j=tensor_j[t];rate.h[i][j]=rate.h[j][i]=curvature[t*g.count+g.active[p]];
        for(int a=0;a<3;++a)rate.dh[a][i][j]=rate.dh[a][j][i]=d[a];}
      Real s[3];h::MetricConnectionVariation(exact[p].metric,h::Geometry(exact[p].metric),rate,s);for(int v=0;v<3;++v)conservative_source[3*p+v]=-s[v];}
    std::cout<<std::setprecision(17)<<"{\"n\":"<<n<<",\"chart\":"<<chart<<",\"focused_shells\":"<<focused<<",\"cells\":"<<active.size()<<",\"modes\":[";
    std::ofstream points(path+".errors.csv");points<<std::setprecision(17)<<"mode,r,weight,H,M,source_error,chi_hessian_error,A_gradient_error\n";
    for(int mode=0;mode<4;++mode){const auto &padded=mode==1?exact_halo:ordinary;
      pilot.GeometryJets(pilot.state,padded);if(mode==2)pilot.jets=exact;
      Norm H,M,S,C,A,B;std::array<Norm,4> regionsH,regionsM;
      for(size_t p=0;p<active.size();++p){const auto &u=pilot.jets[p];const auto c=h::EvolvedConstraints(u,Omega(g.geometry[p].x,u));
        const Real w=g.volumes[p]/std::pow(active[p].chi,1.5),r=active[p].r;Real s[3],se=0,ce=0,ae=0;
        h::EllipticShiftSource(u,h::Geometry(u.metric),0.,s);for(int v=0;v<3;++v){if(mode==3)s[v]=conservative_source[3*p+v];se+=std::pow(s[v]-source[3*p+v],2);
          for(int a=0;a<3;++a){for(int b=0;b<3;++b)ae=std::max(ae,std::abs(u.a.dk[v][a][b]-exact[p].a.dk[v][a][b]));ce=std::max(ce,std::abs(u.chi.dd[v][a]-exact[p].chi.dd[v][a]));}}
        const Real mn=std::sqrt(c.momentum_conformal_norm2);H.Add(c.hamiltonian,w);M.Add(mn,w);S.Add(std::sqrt(se),w);C.Add(ce,w);A.Add(ae,w);
        const int region=r<.1?0:(r<.3?1:(r<.8?2:3));regionsH[region].Add(c.hamiltonian,w);regionsM[region].Add(mn,w);
        points<<mode<<','<<r<<','<<w<<','<<c.hamiltonian<<','<<mn<<','<<std::sqrt(se)<<','<<ce<<','<<ae<<'\n';}
      // Same global operator and original BCs; switch only source/geometry jets.
      if(mode==3){const auto report=pilot.shift.Solve(conservative_source,pilot.Boundary(padded),pilot.beta_delta,1e-11,0.,120);
        if(!report.converged)throw std::runtime_error("metric-rate source diagnostic failed to converge");}
      else pilot.SolveShift(padded);
      for(size_t p=0;p<active.size();++p){Real e=0;for(int v=0;v<3;++v)e+=std::pow(pilot.beta_delta[3*p+v]-true_delta[3*p+v],2);B.Add(std::sqrt(e),g.volumes[p]/std::pow(active[p].chi,1.5));}
      if(mode)std::cout<<',';std::cout<<"{\"mode\":\""<<(mode==0?"ordinary":mode==1?"exact_internal_ghosts":mode==2?"exact_jets":"metric_rate_source")<<"\",\"H\":";H.Print();std::cout<<",\"M\":";M.Print();std::cout<<",\"source_error\":";S.Print();std::cout<<",\"chi_hessian_error\":";C.Print();std::cout<<",\"A_gradient_error\":";A.Print();std::cout<<",\"shift_error\":";B.Print();std::cout<<",\"regions\":[";
      for(int r=0;r<4;++r){if(r)std::cout<<',';std::cout<<"{\"H\":";regionsH[r].Print();std::cout<<",\"M\":";regionsM[r].Print();std::cout<<'}';}std::cout<<"]}";
    }
    std::cout<<"],\"exact_shift_operator_residual\":[";
    for(int mode=0;mode<2;++mode){V beta(3*g.count,0);for(size_t p=0;p<active.size();++p)for(int v=0;v<3;++v)beta[v*g.count+g.active[p]]=true_delta[3*p+v];g.Fill(beta,3);
      if(mode)for(const auto &q:halo){const auto x=coordinate(q.id);for(int v=0;v<3;++v)beta[v*g.count+q.id]=(q.b/q.r+1)*x[v];}
      Norm norm;for(size_t p=0;p<active.size();++p){h::VectorJet<Real> b{};for(int v=0;v<3;++v){Real d[3],dd[3][3];g.Derivatives(beta,v,p,d,dd);b.value[v]=true_delta[3*p+v];for(int a=0;a<3;++a){b.d[a][v]=d[a];for(int c=0;c<3;++c)b.dd[a][c][v]=dd[a][c];}}
        Real applied[3],err=0;h::ApplyEllipticShift(exact[p].metric,h::Geometry(exact[p].metric),b,applied);for(int v=0;v<3;++v)err+=std::pow(applied[v]-source[3*p+v],2);norm.Add(std::sqrt(err),g.volumes[p]/std::pow(active[p].chi,1.5));}
      if(mode)std::cout<<',';norm.Print();}
    // Exact endpoint jets versus the actual one-sided surface reconstruction.
    Norm pole;Real exact_pole=0;
    for(size_t p=0;p<surface.size();++p){const auto &x=pilot.scri->rows[p].geometry.x;const auto sample=pilot.scri->Sample(p,ordinary,pilot.scri->values,fields);
      const auto u=h::UnpackComposite(x,sample.value,sample.d,sample.dd,true,chart);const auto parts=h::ConformalRHS(u,Omega(x,u,true),4.,-.99);
      const auto e=ExactJet(x,surface[p]);const auto ep=h::ConformalRHS(e,Omega(x,e,true),4.,-.99);Real max=0;
      for(int i=0;i<3;++i)for(int j=0;j<3;++j){max=std::max(max,std::abs(parts.pole.a[i][j]));exact_pole=std::max(exact_pole,std::abs(ep.pole.a[i][j]));}pole.Add(max,1);}
    std::cout<<"],\"scri_reconstructed_A_pole\":";pole.Print();std::cout<<",\"scri_exact_A_pole_max\":"<<exact_pole<<"}\n";
  }
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';status=2;}Kokkos::finalize();return status;}
