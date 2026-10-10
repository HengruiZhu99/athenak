// HELD observational replay; no changed gauge/Geometry/Product implementation.
#define main FrozenFarDualMain
#include "/Users/hz0693/research/hyperboloidal/build-layer-research/boundary/reference-wave-map-far-dual-source002-held-20261009/probe.cpp"
#undef main

namespace {
const int selected_bases[26]={3,7,11,15,19,27,35,42,43,47,50,51,55,
  59,63,67,71,75,83,91,98,99,103,106,107,111};
bool Selected(int base){for(int x:selected_bases)if(x==base)return true;return false;}
template<class T>void Tensor3(const T a[3][3][3]){
  std::cout<<'[';for(int i=0;i<3;++i){if(i)std::cout<<',';Matrix(a[i]);}std::cout<<']';
}
void Intermediates(const hyp::LayerPoint<D>&p,const Jet&u,const rwm::Connection<D>&c){
  const D a=u.alpha.value,h=p.alpha,x=u.chi.value,y=p.state.chi.value;
  const auto g=hyp::Geometry(u.metric),gh=hyp::Geometry(p.state.metric);
  if(!g.valid||!gh.valid)throw std::runtime_error("unchanged Geometry invalid");
  D db[3]{},dV[3][3]{},Lh[3][3]{},dL[3][3]{};
  D gc[3][3]{},gr[3][3]{},ga[3][3]{},gar[3][3]{},groups[3][3]{};
  D pa[3][3]{},pv[3][3]{},pc1[3][3][3]{},pc2[3][3][3]{},pc3[3][3][3]{};
  for(int i=0;i<3;++i)db[i]=u.beta.value[i]-p.beta[i];
  for(int i=0;i<3;++i)for(int j=0;j<3;++j){
    dV[i][j]=inner::Product<D>({a,a,x,g.inverse[i][j]})-
             inner::Product<D>({h,h,y,gh.inverse[i][j]});
    Lh[i][j]=inner::Product<D>({h,h,y,gh.inverse[i][j]})-
             inner::Product<D>({p.beta[i],p.beta[j]});
    dL[i][j]=dV[i][j]-inner::Product<D>({db[i],u.beta.value[j]})-
                       inner::Product<D>({p.beta[i],db[j]});
    gc[i][j]=inner::Product<D>({D(.5),a,a,g.inverse[i][j],u.chi.d[j]});
    gr[i][j]=inner::Product<D>({D(.5),h,h,gh.inverse[i][j],p.state.chi.d[j]});
    ga[i][j]=inner::Product<D>({a,x,g.inverse[i][j],u.alpha.d[j]});
    gar[i][j]=inner::Product<D>({h,y,gh.inverse[i][j],p.dalpha[j]});
    // Same four-term subexpression and factor order as unchanged GaugeFar.
    groups[i][j]=gc[i][j]-gr[i][j]-ga[i][j]+gar[i][j];
    pa[i][j]=inner::Product<D>({a,dL[i][j],c.scaled[0][i][j]});
    pv[i][j]=inner::Product<D>({D(2),dV[i][j],p.domega[j]});
    for(int l=0;l<3;++l){
      pc1[i][j][l]=inner::Product<D>({dL[j][l],c.scaled[i+1][j][l]});
      pc2[i][j][l]=inner::Product<D>({dL[j][l],u.beta.value[i],c.scaled[0][j][l]});
      pc3[i][j][l]=inner::Product<D>({Lh[j][l],db[i],c.scaled[0][j][l]});
    }
  }
  std::cout<<",\"geometry_live\":{\"determinant\":";Atom(g.determinant);
  std::cout<<",\"inverse\":";Matrix(g.inverse);std::cout<<'}';
  std::cout<<",\"geometry_reference\":{\"determinant\":";Atom(gh.determinant);
  std::cout<<",\"inverse\":";Matrix(gh.inverse);std::cout<<'}';
  std::cout<<",\"intermediates\":{\"db\":";Vector(db,3);
  std::cout<<",\"dV\":";Matrix(dV);std::cout<<",\"Lh\":";Matrix(Lh);
  std::cout<<",\"dL\":";Matrix(dL);
  std::cout<<",\"gradient_chi_live\":";Matrix(gc);
  std::cout<<",\"gradient_chi_reference\":";Matrix(gr);
  std::cout<<",\"gradient_alpha_live\":";Matrix(ga);
  std::cout<<",\"gradient_alpha_reference\":";Matrix(gar);
  std::cout<<",\"gradient_group\":";Matrix(groups);
  std::cout<<",\"pole_alpha_connection_unsigned\":";Matrix(pa);
  std::cout<<",\"pole_beta_dV\":";Matrix(pv);
  std::cout<<",\"pole_beta_connection1_unsigned\":";Tensor3(pc1);
  std::cout<<",\"pole_beta_connection2_unsigned\":";Tensor3(pc2);
  std::cout<<",\"pole_beta_connection3_unsigned\":";Tensor3(pc3);std::cout<<'}';
}
void RunDiagnostic(){int base_id=0,emitted=0;
  for(double a:{.5,2.})for(double r:radii)for(int direction=0;direction<2;++direction)for(int family=0;family<4;++family){
    const int current=base_id++;if(!Selected(current))continue;
    hyp::LayerReference<double>ref(1,a,{true,.05,.95});ref.Validate();
    const double xyz[3]={r*directions[direction][0],r*directions[direction][1],r*directions[direction][2]};
    const auto p=ref.At(xyz[0],xyz[1],xyz[2]);const auto pd=CastPoint(p);
    const D xd[3]={xyz[0],xyz[1],xyz[2]};const auto c=rwm::ReferenceConnection(pd,xd);
    const auto base=State(p,10+family);const auto u=SeedField(base,13,false);
    const auto actual=Call(pd,u,c,false);const auto legacy=Call(pd,u,c,true);
    std::cout<<"{\"kind\":\"metric-flux-diagnostic\",\"base_index\":"<<current
      <<",\"seed_index\":13,\"seed_id\":\"metric-STF\",\"zero_primal_gradients\":false"
      <<",\"a\":"<<a<<",\"nominal_radius\":"<<r<<",\"direction\":"<<direction
      <<",\"family\":\""<<family_names[family]<<"\",\"stored_radius\":"<<p.radius
      <<",\"W_context\":"<<hyp::SmoothCutoff(p.radius,.45,.85).value;
    Context(pd,xd,c,u);std::cout<<",\"new\":";EmitOutput(actual);
    std::cout<<",\"legacy\":";EmitOutput(legacy);Intermediates(pd,u,c);
    std::cout<<",\"helper_calls_cumulative\":"<<helper_calls<<"}\n";++emitted;
  }
  if(base_id!=112||emitted!=26||helper_calls!=52)throw std::runtime_error("fixed diagnostic registry mismatch");
}
}
int main(int argc,char**argv){try{
  if(argc!=1)throw std::runtime_error("fixed no-argument diagnostic only");
  std::cout<<std::setprecision(17);RunDiagnostic();return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
