// Actual C0/spatialnorm finite-rb point bridge; no assembled radial operator here.
#include "actual_bridge.cpp"
#include "configuration_rows.hpp"
#include "radial_normalization.hpp"
#if __has_include("constraint_rate_api.hpp")
#include "constraint_rate_api.hpp"
#define HAS_CONSTRAINT_RATE_API 1
#endif

struct RadialPoint {
  RadialRaw raw{},rhs_radial{};Raw rhs{};NormalizedRadial normalized{},normalized_rhs{};
  ReferenceRadial coefficients{};std::array<double,4> normals{};
};
RadialPoint EvaluatePoint(int J,int m,int channel,int phase,const X3&x,
                    const totalj::WJet<double>&w,bool with_source){
  const auto p=reference.At(x[0],x[1],x[2]);if(!(p.omega>0))throw std::runtime_error("noninterior point");
  const double r=p.radius;if(!(r>0))throw std::runtime_error("normal frame not evaluated at origin");
  X3 n{x[0]/r,x[1]/r,x[2]/r},t{},v{};MakeScreen(n,t,v);
  const auto state=LiftPhysical(p,SeedPhysical(J,m,channel,phase,x,w));RadialPoint out{};
  out.raw=RadialVariation(state,n);out.coefficients=MakeReferenceRadial(p,n,t,v);
  out.normalized=Normalize(out.raw,out.coefficients);
  const auto input=InputNormals(p,state);out.normals[0]=input[0];out.normals[1]=input[1];
  if(with_source){
    out.rhs=Derivative(ActualDual(p,state));const auto rows=ConfigurationSpatial(p,state,x);
    for(int f=0;f<22;++f)out.rhs_radial[f].v=out.rhs[f];
    for(const auto f:configuration_raw_indices)for(int d=0;d<3;++d)out.rhs_radial[f].d+=n[d]*rows[f].d[d].d;
    out.normalized_rhs=Normalize(out.rhs_radial,out.coefficients);
    const auto normal=OutputNormals(p,out.rhs);out.normals[2]=normal[0];out.normals[3]=normal[1];
  }
  return out;
}
std::array<double,50> PackInput(const RadialPoint&p){std::array<double,50>a{};for(int i=0;i<10;++i){a[i]=p.normalized.u[i].v;a[10+i]=p.normalized.u[i].d;a[20+i]=p.normalized.u[i].dd;a[30+i]=p.normalized.v[i].v;a[40+i]=p.normalized.v[i].d;}return a;}
std::array<double,30> PackSource(const RadialPoint&p){std::array<double,30>a{};for(int i=0;i<10;++i){a[i]=p.normalized_rhs.u[i].v;a[10+i]=p.coefficients.c.v*p.normalized_rhs.u[i].d;a[20+i]=p.normalized_rhs.v[i].v;}return a;}
void PointBatch(bool binary,bool with_source){
  int J,m,channel,phase;X3 x{};totalj::WJet<double>w;
  while(std::cin>>J>>m>>channel>>phase>>x[0]>>x[1]>>x[2]>>w.value>>w.rho_d>>w.rho_dd){
    const auto p=EvaluatePoint(J,m,channel,phase,x,w,with_source);const auto a=PackInput(p);
    if(binary){std::cout.write(reinterpret_cast<const char*>(a.data()),sizeof(a));continue;}
    for(const auto z:p.rhs)std::cout<<z<<' ';for(const auto f:configuration_raw_indices)std::cout<<p.rhs_radial[f].d<<' ';
    for(const auto z:p.raw)std::cout<<z.v<<' ';for(const auto f:configuration_raw_indices)std::cout<<p.raw[f].d<<' ';
    for(const auto z:a)std::cout<<z<<' ';for(const auto z:PackSource(p))std::cout<<z<<' ';for(const auto z:p.normals)std::cout<<z<<' ';std::cout<<'\n';
  }
}
void ReferenceBatch(){double r;while(std::cin>>r){const auto p=reference.At(r,0,0);const auto b=MakeReferenceRadial(p,{1,0,0},{0,1,0},{0,0,1});
  for(const auto&z:{b.alpha,b.chi,b.omega,b.c,b.beta_n})std::cout<<z.v<<' '<<z.d<<' '<<z.dd<<' ';
  hyp::LayerGaugeParameters g;g.physical_trace_lapse=true;g.preferred_source=false;g.scri_lapse_damping=2;
  const auto s=SpatialReference(p,{r,0,0});const auto w=hyp::LayerCoefficients(s.radius,s.alpha,g).weight;
  std::cout<<w.v.v<<' '<<w.d[0].v<<'\n';}}
double PointRadiusSquared(const X3&x){return x[0]*x[0]+x[1]*x[1]+x[2]*x[2];}
void DerivativeGate(){
  const X3 n{.36,-.48,.8};double config=0,unused=0,normals=0,fd=0,mapfd=0;std::vector<std::array<double,4>>derivatives,map_derivatives;
  for(double r:{.025,.30,.60,.85,.98})for(int J=0;J<3;++J)for(int channel=0;channel<(J==0?8:J==1?16:20);++channel){
    const X3 x{r*n[0],r*n[1],r*n[2]};const double rho=PointRadiusSquared(x);
    const totalj::WJet<double>w{1+rho/3-rho*rho/5+rho*rho*rho/7,1./3-2*rho/5+3*rho*rho/7,-2./5+6*rho/7};
    const auto p=reference.At(x[0],x[1],x[2]);const auto state=LiftPhysical(p,SeedPhysical(J,0,channel,0,x,w));
    const auto actual=Derivative(ActualDual(p,state));const auto s=ConfigurationSpatial(p,state,x);
    for(const int f:configuration_raw_indices)config=std::max(config,std::abs(s[f].v.d-actual[f])/std::max(1.,std::abs(actual[f])));
    for(double extension:{-.7,.37,3.1}){const auto z=ConfigurationSpatial(p,state,x,extension);for(const int f:configuration_raw_indices){unused=std::max(unused,std::abs(z[f].v.d-s[f].v.d));for(int d=0;d<3;++d)unused=std::max(unused,std::abs(z[f].d[d].d-s[f].d[d].d));}}
    const auto center=EvaluatePoint(J,0,channel,0,x,w,true);const auto expected=PackInput(center);const auto expected_source=PackSource(center);
    for(auto z:center.normals)normals=std::max(normals,std::abs(z)/std::max(1.,Norm(actual)));
    std::array<double,4>errors{},maperrors{};int hi=0;
    for(double h:{.001,.0005,.00025,.000125}){
      const int offset[4]={-2,-1,1,2};const double first[4]={1,-8,8,-1},second[4]={-1,16,16,-1};
      std::array<double,11>dr{};std::array<double,10>ud{},udd{},vd{},utd{};
      for(int q=0;q<4;++q){X3 y{};for(int d=0;d<3;++d)y[d]=(r+offset[q]*h)*n[d];const double z=PointRadiusSquared(y);
        const totalj::WJet<double>wy{1+z/3-z*z/5+z*z*z/7,1./3-2*z/5+3*z*z/7,-2./5+6*z/7};
        const auto point=EvaluatePoint(J,0,channel,0,y,wy,true);const auto normalized=PackInput(point);const auto source=PackSource(point);
        for(int k=0;k<11;++k)dr[k]+=first[q]*point.rhs[configuration_raw_indices[k]]/(12*h);
        for(int k=0;k<10;++k){ud[k]+=first[q]*normalized[k]/(12*h);udd[k]+=second[q]*normalized[k]/(12*h*h);vd[k]+=first[q]*normalized[30+k]/(12*h);utd[k]+=first[q]*source[k]/(12*h);}
      }
      for(int k=0;k<11;++k){const int f=configuration_raw_indices[k];errors[hi]=std::max(errors[hi],std::abs(dr[k]-center.rhs_radial[f].d)/std::max(1.,std::abs(center.rhs_radial[f].d)));}
      for(int k=0;k<10;++k){udd[k]-=30*expected[k]/(12*h*h);
        maperrors[hi]=std::max({maperrors[hi],std::abs(ud[k]-expected[10+k])/std::max(1.,std::abs(expected[10+k])),std::abs(udd[k]-expected[20+k])/std::max(1.,std::abs(expected[20+k])),std::abs(vd[k]-expected[40+k])/std::max(1.,std::abs(expected[40+k])),std::abs(center.coefficients.c.v*utd[k]-expected_source[10+k])/std::max(1.,std::abs(expected_source[10+k]))});}
      ++hi;
    }
    fd=std::max(fd,*std::min_element(errors.begin(),errors.end()));mapfd=std::max(mapfd,*std::min_element(maperrors.begin(),maperrors.end()));derivatives.push_back(errors);map_derivatives.push_back(maperrors);
  }
  std::cout<<"{\"configuration_actual_full22_scaled\":"<<config<<",\"finite_unused_extension_difference\":"<<unused<<",\"raw22_normals_scaled\":"<<normals<<",\"configuration_derivative_best_h_scaled\":"<<fd<<",\"complete_UV_map_derivative_best_h_scaled\":"<<mapfd<<",\"configuration_derivative_sequences\":[";
  for(std::size_t k=0;k<derivatives.size();++k){if(k)std::cout<<',';std::cout<<'[';for(int h=0;h<4;++h){if(h)std::cout<<',';std::cout<<derivatives[k][h];}std::cout<<']';}std::cout<<"],\"map_derivative_sequences\":[";
  for(std::size_t k=0;k<map_derivatives.size();++k){if(k)std::cout<<',';std::cout<<'[';for(int h=0;h<4;++h){if(h)std::cout<<',';std::cout<<map_derivatives[k][h];}std::cout<<']';}std::cout<<"]}\n";
}
int main(int argc,char**argv){try{std::cout<<std::setprecision(17);const std::string mode=argc>1?argv[1]:"--gate";
  if(mode=="--source-batch")PointBatch(false,true);else if(mode=="--input-binary")PointBatch(true,false);else if(mode=="--reference-batch")ReferenceBatch();
#ifdef HAS_CONSTRAINT_RATE_API
  else if(mode=="--manufactured-rate-batch")ManufacturedRateBatch();else if(mode=="--constraint-rate-batch")ConstraintRateBatch();else if(mode=="--subsidiary-batch")SubsidiaryBatch();
#endif
  else if(mode=="--gate")DerivativeGate();else throw std::runtime_error("unknown mode");
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
