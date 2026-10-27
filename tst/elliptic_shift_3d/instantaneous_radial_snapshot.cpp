// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Diagnostic only: genuine global 3D elliptic solves on frozen spherical data.
// The target beta is a warm start/comparison, NEVER an artificial RHS L beta.
#define main UnusedCoupledPilotMain
#include "coupled.cpp"
#undef main

struct SnapshotFixture {
  int n=0;std::vector<Real> radii;Real time=0;unsigned long long step=0;
  std::string hash;std::array<Real,8> boundary{};
  std::vector<Real> angular_r,angular_beta;
};
std::vector<Real> Radii(const std::string &text){std::vector<Real> radii;std::istringstream in(text);std::string word;
  while(std::getline(in,word,','))radii.push_back(std::stod(word));return radii;}
Pilot MakePilot(int n,const std::vector<Real> &radii){
  if(n!=6||radii.size()<2||radii.size()>3||radii.back()!=1)throw std::invalid_argument("initial diagnostic budget: N6, two/three layers ending at1");
  return Pilot(cubed_sphere::CompositeMesh(.25,radii,1,n,2,2),true,1e-10,4,true,true,0,6,true,2,.24);
}
void ReadScalarData(std::istream &in,V &v,std::vector<std::array<Real,3>> &d,
    std::vector<std::array<std::array<Real,3>,3>> &dd){
  for(int f=0;f<fields;++f){in>>v[f];for(int a=0;a<3;++a)in>>d[f][a];for(int a=0;a<3;++a)for(int b=0;b<3;++b)in>>dd[f][a][b];}
}
void ReadVector(std::istream &in,h::VectorJet<Real> &v){for(int i=0;i<3;++i)in>>v.value[i];
  for(int a=0;a<3;++a)for(int i=0;i<3;++i)in>>v.d[a][i];for(int a=0;a<3;++a)for(int b=0;b<3;++b)for(int i=0;i<3;++i)in>>v.dd[a][b][i];}
V Difference(const V &a,const V &b){if(a.size()!=b.size())throw std::runtime_error("difference shape");V out(a.size());for(size_t i=0;i<a.size();++i)out[i]=a[i]-b[i];return out;}
struct VectorStats{int cells=0;Real physical_rms=0,conformal_rms=0,max_component=0,max_vector=0;};
VectorStats Stats(const h::CompositeShiftStage &stage,const V &v,const std::vector<bool> &mask){
  VectorStats out;Real physical=0,conformal=0,vp=0,vc=0;
  for(size_t p=0;p<stage.grid.active.size();++p)if(mask[p]){++out.cells;Real square=0;
    for(int i=0;i<3;++i){square+=v[3*p+i]*v[3*p+i];out.max_component=std::max(out.max_component,std::abs(v[3*p+i]));}
    out.max_vector=std::max(out.max_vector,std::sqrt(square));physical+=stage.norm_volume[p]*square;vp+=stage.norm_volume[p];
    conformal+=stage.conformal_volume[p]*square;vc+=stage.conformal_volume[p];}
  if(out.cells){out.physical_rms=std::sqrt(physical/vp);out.conformal_rms=std::sqrt(conformal/vc);}return out;
}
void PrintStats(std::ostream &out,const VectorStats &s){out<<"{\"cells\":"<<s.cells<<",\"physical_rms\":"<<s.physical_rms<<",\"conformal_rms\":"<<s.conformal_rms
  <<",\"max_component\":"<<s.max_component<<",\"max_vector\":"<<s.max_vector<<"}";}
std::map<std::string,std::vector<bool>> Regions(const cubed_sphere::CompositeDifference &grid){
  const size_t count=grid.active.size();std::map<std::string,std::vector<bool>> masks;
  for(const auto &name:{"all","core_r_lt_01","bulk_cartesian","interface_neighbor","edge_corner_neighbor","scri_r_gt_09","radial_062_070"})masks[name]=std::vector<bool>(count,false);
  for(int layer=0;layer<static_cast<int>(grid.mesh.radii.size());++layer)masks["shell_"+std::to_string(layer)]=std::vector<bool>(count,false);
  for(size_t p=0;p<count;++p){const auto &x=grid.geometry[p].x;Real r=std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
    const int m=p/(grid.n*grid.n*grid.n),i=p%grid.n,j=p/grid.n%grid.n,k=p/(grid.n*grid.n)%grid.n;
    const int sides=(i==0||i==grid.n-1)+(j==0||j==grid.n-1)+(k==0||k==grid.n-1);
    masks["all"][p]=true;masks["core_r_lt_01"][p]=r<.1;masks["bulk_cartesian"][p]=grid.mesh.blocks[m].face<0;
    masks["interface_neighbor"][p]=sides>0;masks["edge_corner_neighbor"][p]=sides>1;masks["scri_r_gt_09"][p]=r>.9;masks["radial_062_070"][p]=r>=.62&&r<=.70;
    if(grid.mesh.blocks[m].layer>=0)masks["shell_"+std::to_string(grid.mesh.blocks[m].layer)][p]=true;
  }return masks;
}
V PadBeta(const h::CompositeShiftStage &stage,const V &beta,const h::CompositeShiftStage::Boundary &boundary){
  V field(3*stage.grid.count,0);for(size_t p=0;p<stage.grid.active.size();++p)for(int v=0;v<3;++v)field[v*stage.grid.count+stage.grid.active[p]]=beta[3*p+v];
  stage.grid.Fill(field,3,[&](int v,const auto &x){return boundary(x)[v];});return field;
}
V Residual(const h::CompositeShiftStage &stage,const V &source,const V &beta,const h::CompositeShiftStage::Boundary &boundary){V applied;stage.Apply(beta,applied,boundary);for(size_t i=0;i<applied.size();++i)applied[i]=-source[i]-applied[i];return applied;}

int main(int argc,char **argv){Kokkos::initialize(argc,argv);int status=0;
try{
  if(argc<2)throw std::invalid_argument("points N radiiCSV outputCSV | solve fixture outputPrefix");
  const std::string action=argv[1];
  if(action=="points"){
    if(argc!=6)throw std::invalid_argument("points N radiiCSV outputCSV");auto pilot=MakePilot(std::stoi(argv[2]),Radii(argv[3]));
    if(pilot.shift.grid.active.size()>5616)throw std::runtime_error("cell budget exceeded");
    // argv4 intentionally specifies the common radial-interface stencil flag.
    if(std::stoi(argv[4])!=1)throw std::invalid_argument("this diagnostic records radial-interface stencils=1");
    std::ofstream out(argv[5]);out<<std::setprecision(17)<<"x,y,z\n";
    for(const auto &g:pilot.shift.grid.geometry)out<<g.x[0]<<','<<g.x[1]<<','<<g.x[2]<<'\n';
    std::cout<<"{\"points\":"<<pilot.shift.grid.active.size()<<"}\n";
  }else if(action=="solve"){
    if(argc!=4)throw std::invalid_argument("solve fixture outputPrefix");std::ifstream input(argv[2]);SnapshotFixture fixture;std::string magic;size_t count;int layers,nangular;
    input>>magic>>fixture.n>>layers>>count>>fixture.time>>fixture.step>>fixture.hash;
    if(magic!="RADIAL_SNAPSHOT_CARTESIAN_V1"||fixture.hash.size()!=64)throw std::runtime_error("snapshot fixture header");
    fixture.radii.resize(layers);for(auto &r:fixture.radii)input>>r;for(auto &v:fixture.boundary)input>>v;
    input>>nangular;fixture.angular_r.resize(nangular);fixture.angular_beta.resize(nangular);
    for(int i=0;i<nangular;++i)input>>fixture.angular_r[i]>>fixture.angular_beta[i];
    auto pilot=MakePilot(fixture.n,fixture.radii);auto &stage=pilot.shift;auto &grid=stage.grid;
    if(count!=grid.active.size()||count>5616)throw std::runtime_error("snapshot fixture cell budget/layout");pilot.Initialize(0,false);
    std::array<std::vector<h::Z4cJet<Real>>,2> independent;for(auto &v:independent)v.resize(count);
    std::array<V,2> truth{V(3*count),V(3*count)};
    V value(fields);std::vector<std::array<Real,3>> d(fields);std::vector<std::array<std::array<Real,3>,3>> dd(fields);
    for(size_t p=0;p<count;++p){Real point[3];for(auto &v:point)input>>v;
      for(int i=0;i<3;++i)if(std::abs(point[i]-grid.geometry[p].x[i])>1e-13)throw std::runtime_error("fixture point order mismatch");
      for(int width=0;width<2;++width){ReadScalarData(input,value,d,dd);independent[width][p]=h::UnpackComposite(point,value,d,dd,true,0,1);ReadVector(input,independent[width][p].beta);
        for(int v=0;v<3;++v)truth[width][3*p+v]=independent[width][p].beta.value[v]+point[v];
        if(width==0){for(int f=0;f<fields;++f)pilot.state[fields*p+f]=value[f];
          pilot.state[fields*p]=pilot.EncodePositive(0,value[0],point);pilot.state[fields*p+18]=pilot.EncodePositive(18,value[18],point);}}
    }
    std::string trailing;if(!input||input>>trailing)throw std::runtime_error("truncated or trailing snapshot fixture");
    const auto &b=fixture.boundary;
    if(!(b[0]>0&&b[1]>0)||std::abs(b[1]-1)>1e-13)throw std::runtime_error("snapshot scri clock/positivity differs from prescribed data");
    for(size_t p=0;p<pilot.scri->rows.size();++p){const auto &x=pilot.scri->rows[p].geometry.x;const Real r=std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);
      Real *row=pilot.scri->values.data()+p*fields;row[0]=pilot.EncodePositive(0,b[0],x);row[18]=pilot.EncodePositive(18,b[1],x);row[13]=b[3]+3;row[14]=b[4];
      for(int t=0;t<6;++t){const int i=tensor_i[t],j=tensor_j[t];row[7+t]=1.5*b[2]*(x[i]*x[j]/(r*r)-(i==j?1./3:0));}
      for(int v=0;v<3;++v){row[15+v]=b[5]*x[v]/r;row[19+v]=b[7]*x[v]/r;}}
    const auto padded=pilot.Padded(pilot.state);pilot.GeometryJets(pilot.state,padded);const auto boundary=pilot.Boundary(padded);
    std::array<V,3> source{V(3*count),V(3*count),V(3*count)};
    std::array<V,2> continuum_residual{V(3*count),V(3*count)};
    for(size_t p=0;p<count;++p){
      h::VectorJet<Real> reference_beta{};for(int i=0;i<3;++i){reference_beta.value[i]=-grid.geometry[p].x[i];reference_beta.d[i][i]=-1;}
      Real reference_rate[3];h::ApplyEllipticShift(pilot.jets[p].metric,stage.geometry[p],reference_beta,reference_rate);
      Real rhs[3];h::EllipticShiftSource(pilot.jets[p],stage.geometry[p],0.,rhs);
      for(int i=0;i<3;++i)source[0][3*p+i]=rhs[i]-reference_rate[i];
      for(int width=0;width<2;++width){const auto &u=independent[width][p];const auto geometry=h::Geometry(u.metric);Real reference_rhs[3],applied[3];
        // This source is ONLY -DV[-2 alpha A] (plus eta V=0). The recorded
        // beta is never used to manufacture a source or hold the state fixed.
        h::EllipticShiftSource(u,geometry,0.,reference_rhs);h::ApplyEllipticShift(u.metric,geometry,u.beta,applied);
        for(int i=0;i<3;++i){source[1+width][3*p+i]=reference_rhs[i]-reference_rate[i];continuum_residual[width][3*p+i]=applied[i]-reference_rhs[i];}}
    }
    const auto started=std::chrono::steady_clock::now();
    std::array<V,3> solved{truth[0],truth[0],truth[1]};std::array<h::ShiftSolveReport,3> reports;std::array<Real,3> solve_seconds{};
    h::ShiftSolveControl controls;controls.conformal_norm_fraction=.5;controls.maximum_residual_tolerance=1e-7;
    for(int i=0;i<3;++i){const auto before=std::chrono::steady_clock::now();reports[i]=stage.Solve(source[i],boundary,solved[i],1e-10,0.,160,controls);
      solve_seconds[i]=std::chrono::duration<Real>(std::chrono::steady_clock::now()-before).count();if(!reports[i].converged)status=1;}
    std::array<V,3> residual;for(int i=0;i<3;++i)residual[i]=Residual(stage,source[i],solved[i],boundary);
    std::map<std::string,V> diagnostics;
    diagnostics["FD_source_minus_interpolated7"]=Difference(source[0],source[1]);diagnostics["source_width9_minus7"]=Difference(source[2],source[1]);
    diagnostics["reference_continuum_residual7"]=continuum_residual[0];diagnostics["reference_continuum_residual9"]=continuum_residual[1];
    diagnostics["FD_original_residual"]=residual[0];diagnostics["interpolated7_original_residual"]=residual[1];diagnostics["interpolated9_original_residual"]=residual[2];
    diagnostics["recorded_beta_discrete_residual_FD_source"]=Residual(stage,source[0],truth[0],boundary);
    diagnostics["recorded_beta_discrete_residual_interpolated7_source"]=Residual(stage,source[1],truth[0],boundary);
    diagnostics["beta_error_FD_source"]=Difference(solved[0],truth[0]);diagnostics["beta_error_interpolated7_source"]=Difference(solved[1],truth[0]);
    diagnostics["beta_error_interpolated9_source"]=Difference(solved[2],truth[1]);diagnostics["beta_solution_width9_minus7"]=Difference(solved[2],solved[1]);
    const auto masks=Regions(grid);std::ofstream points(std::string(argv[3])+".cells.csv");points<<std::setprecision(17)<<"x,y,z,r,chi,alpha";
    for(const auto &name:{"beta_target","beta_FD","beta_ref7","beta_ref9","source_FD","source_ref7","source_ref9"})for(int v=0;v<3;++v)points<<','<<name<<'_'<<v;
    points<<'\n';Real min_radius=1,min_chi=1,min_alpha=1;Real core_mean[3]{};int core_cells=0;
    for(size_t p=0;p<count;++p){const auto &x=grid.geometry[p].x;const Real r=std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);min_radius=std::min(min_radius,r);min_chi=std::min(min_chi,pilot.jets[p].chi.value);min_alpha=std::min(min_alpha,pilot.jets[p].alpha.value);
      points<<x[0]<<','<<x[1]<<','<<x[2]<<','<<r<<','<<pilot.jets[p].chi.value<<','<<pilot.jets[p].alpha.value;
      for(const V *vector:{&truth[0],&solved[0],&solved[1],&solved[2]})for(int v=0;v<3;++v)points<<','<<(*vector)[3*p+v]-x[v];
      for(const auto &vector:source)for(int v=0;v<3;++v)points<<','<<vector[3*p+v];points<<'\n';
      if(r<.1){++core_cells;for(int v=0;v<3;++v)core_mean[v]+=solved[0][3*p+v]-x[v];}}
    for(auto &v:core_mean)v/=core_cells;
    std::array<V,4> beta_padded{PadBeta(stage,solved[0],boundary),PadBeta(stage,solved[1],boundary),PadBeta(stage,solved[2],boundary),PadBeta(stage,truth[0],boundary)};
    Real boundary_normal=0,boundary_tangent=0;
    for(size_t p=0;p<pilot.scri->rows.size();++p){const auto &x=pilot.scri->rows[p].geometry.x;
      for(int choice=0;choice<3;++choice){Real beta[3],bn=0;for(int v=0;v<3;++v){beta[v]=-x[v]+pilot.scri->ReconstructFromGhost(p,beta_padded[choice],v);bn+=beta[v]*x[v];}
        boundary_normal=std::max(boundary_normal,std::abs(bn+b[1]*std::sqrt(b[0])));
        for(int v=0;v<3;++v)boundary_tangent=std::max(boundary_tangent,std::abs(beta[v]-bn*x[v]));}}
    if(boundary_normal>1e-12||boundary_tangent>1e-12)status=1;
    std::ofstream spheres(std::string(argv[3])+".spheres.csv");spheres<<std::setprecision(17)<<"r,direction,target_radial";
    for(const auto &name:{"FD","ref7","ref9","target_projection"})spheres<<','<<name<<"_radial,"<<name<<"_tangent_norm";spheres<<'\n';
    std::ostringstream angular;angular<<std::setprecision(17)<<'[';
    for(int radial=0;radial<nangular;++radial){const Real r=fixture.angular_r[radial];Real sum[4]{},squares[4]{},tangent[4]{},control[3]{};
      for(int direction=0;direction<64;++direction){const Real nz=1-2*(direction+.5)/64,phi=direction*2.39996322972865332;
        const Real normal[3]={std::sqrt(1-nz*nz)*std::cos(phi),std::sqrt(1-nz*nz)*std::sin(phi),nz};Real x[3]={r*normal[0],r*normal[1],r*normal[2]};
        const auto entries=grid.Interpolate(x,6);Real br[4]{},tan[4]{};
        for(int choice=0;choice<4;++choice){Real beta[3]={-x[0],-x[1],-x[2]};for(const auto &entry:entries)for(int v=0;v<3;++v)beta[v]+=entry.weight*beta_padded[choice][v*grid.count+entry.point];
          for(int v=0;v<3;++v)br[choice]+=beta[v]*normal[v];for(int v=0;v<3;++v)tan[choice]+=(beta[v]-br[choice]*normal[v])*(beta[v]-br[choice]*normal[v]);
          sum[choice]+=br[choice];squares[choice]+=br[choice]*br[choice];tangent[choice]+=tan[choice];}
        for(int choice=0;choice<3;++choice)control[choice]+=(br[choice]-br[3])*(br[choice]-br[3]);
        spheres<<r<<','<<direction<<','<<fixture.angular_beta[radial];for(int choice=0;choice<4;++choice)spheres<<','<<br[choice]<<','<<std::sqrt(tan[choice]);spheres<<'\n';}
      if(radial)angular<<',';angular<<"{\"r\":"<<r<<",\"target_radial\":"<<fixture.angular_beta[radial]<<",\"series\":[";
      for(int choice=0;choice<4;++choice){const Real mean=sum[choice]/64,scatter=std::sqrt(std::max(Real(0),squares[choice]/64-mean*mean));
        if(choice)angular<<',';angular<<"{\"mean\":"<<mean<<",\"raw_scatter\":"<<scatter<<",\"mean_error\":"<<mean-fixture.angular_beta[radial]<<",\"tangent_rms\":"<<std::sqrt(tangent[choice]/64);
        if(choice<3)angular<<",\"difference_from_same_target_projection_rms\":"<<std::sqrt(control[choice]/64);angular<<'}';}angular<<"]}";}
    angular<<']';
    std::cout<<std::setprecision(17)<<"{\"snapshot_time\":"<<fixture.time<<",\"snapshot_sha256\":\""<<fixture.hash<<"\",\"n\":"<<fixture.n<<",\"cells\":"<<count
      <<",\"layers\":"<<layers<<",\"positive_storage\":2,\"radial_interface_stencils\":true,\"warm_start\":\"recorded radial beta, subsequently solved\""
      <<",\"minimum_cell_radius\":"<<min_radius<<",\"minimum_chi\":"<<min_chi<<",\"minimum_alpha\":"<<min_alpha
      <<",\"core_beta_mean_not_boundary_data\":["<<core_mean[0]<<','<<core_mean[1]<<','<<core_mean[2]<<"]"
      <<",\"scri_normal_ghost_residual\":"<<boundary_normal<<",\"scri_tangential_ghost_residual\":"<<boundary_tangent
      <<",\"recorded_scri_null_residual\":"<<b[6]+b[1]*std::sqrt(b[0])<<",\"physical_log_volume\":"<<stage.log_physical_volume<<",\"solves\":[";
    for(int i=0;i<3;++i){if(i)std::cout<<',';std::cout<<"{\"source\":\""<<(i==0?"FD":(i==1?"interpolated7":"interpolated9"))<<"\",\"converged\":"<<reports[i].converged
      <<",\"iterations\":"<<reports[i].iterations<<",\"seconds\":"<<solve_seconds[i]<<",\"original_physical_residual_rms\":"<<reports[i].physical_residual_rms
      <<",\"original_conformal_residual_rms\":"<<reports[i].conformal_residual_rms<<",\"original_residual_max\":"<<reports[i].residual_max<<'}';}
    std::cout<<"],\"regional_diagnostics\":{";bool first=true;for(const auto &diagnostic:diagnostics){if(!first)std::cout<<',';first=false;std::cout<<'\"'<<diagnostic.first<<"\":{";bool region_first=true;
      for(const auto &region:masks){if(!region_first)std::cout<<',';region_first=false;std::cout<<'\"'<<region.first<<"\":";PrintStats(std::cout,Stats(stage,diagnostic.second,region.second));}std::cout<<'}';}
    std::cout<<"},\"angular_series_order\":[\"FD\",\"interpolated7\",\"interpolated9\",\"recorded_target_projection\"],\"angular_comparison\":"<<angular.str()
      <<",\"seconds_after_source_setup\":"<<std::chrono::duration<Real>(std::chrono::steady_clock::now()-started).count()
      <<",\"scope\":\"instantaneous global3D elliptic diagnostic; no evolution; no artificial L(beta_target) RHS; AF endpoint not resolved; spatial acceptance not established on N6\"}\n";
  }else throw std::invalid_argument("unknown action");
}catch(const std::exception &error){std::cerr<<error.what()<<'\n';status=2;}
Kokkos::finalize();return status;}
