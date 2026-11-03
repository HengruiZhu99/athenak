// Read-only admitted-checkpoint slices. No stepping or checkpoint writing.
#define main hyperbolic_unused_evolution_main
#include "coupled_mpi.cpp"
#undef main
int main(int argc,char**argv){
 MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);
 Kokkos::initialize(argc,argv);int code=0;
 try{
  if(argc!=5||global_variable::nranks!=1)throw std::invalid_argument("usage: checkpoint_slices n checkpoint dt fresh_prefix (one SERIAL rank)");
  const int n=std::stoi(argv[1]);const Real dt=std::stod(argv[3]);const std::string prefix=argv[4];
  if(std::filesystem::exists(prefix+".csv")||std::filesystem::exists(prefix+".json"))throw std::runtime_error("existing slice output");
  Evolution e(n,2,2,"");e.LoadCheckpoint(argv[2],2,dt);e.RefreshBoundary(dt);
  const auto halo=e.exchange()->Fill(e.state);if(halo.invalid)throw std::runtime_error("invalid halo");
  const auto native=e.kernel()->Diagnostics(e.exchange()->padded,.12,e.surface);
  if(native.invalid_cells)throw std::runtime_error("invalid native diagnostic");
  // Same original fourth-order mapped derivatives and donor-seven final rows.
  static_assert(!cubed_sphere::scri_first6,"This original-operator analysis excludes first6 variants");
  h::HyperbolicComposite model(hyperbolic_reproduction::ContinuousCoreMesh(n),false,true,2,
    h::HyperbolicComposite::NormalDerivative::sampled_numerator,h::HyperbolicComposite::RadialClosure::donor_seven);
  model.scri.values=e.Download(e.surface);const auto padded=e.Download(e.exchange()->padded);auto&g=model.grid;
  if(g.active.size()!=size_t(e.total))throw std::runtime_error("analysis grid mismatch");
  std::ofstream out(prefix+".csv");out<<std::setprecision(17);
  out<<"ray,cell,block,layer,i,j,k,x,y,z,r,chi,alpha,metric_rr,K,Theta,beta_r,beta_dev_r,D_r,Lambda_r,dr_chi,dr_alpha,dr_beta_dev_r,dr_D_r,H,Mr,Mnorm,Zr\n";
  h::HyperbolicComposite::V v(31);std::vector<std::array<Real,3>>d(31);std::vector<std::array<std::array<Real,3>,3>>dd(31);
  Real sums[2][4]{};size_t rows=0;
  for(size_t p=0;p<g.active.size();++p){
   const auto*x=g.geometry[p].x;const Real r=model.Radius(x),weight=g.DiagnosticVolume(p);const auto&cell=g.cells[p];const auto&block=g.mesh.blocks[cell.block];
   int ray=-1;if(block.face==0){if(cell.j==n/2&&cell.k==n/2)ray=0;else if(cell.j==n-1&&cell.k==n-1)ray=1;}
   if(block.face==1 && cell.j==n/2 && cell.k==n/2)ray=3;
   const Real core_dx=.36/(4*n);
   if(block.face<0 && x[1]<0 && x[1]>-core_dx && x[2]<0 && x[2]>-core_dx)ray=2;
   if(weight<=0&&ray<0)continue;
   for(int f=0;f<31;++f){v[f]=padded[f*g.count+g.active[p]];Real df[3]{},ddf[3][3]{};model.BulkDerivatives(padded,f,p,df,ddf,f<=6||f==18||(f>=22&&f<25));
    for(int a=0;a<3;++a){d[f][a]=df[a];for(int b=0;b<3;++b)dd[f][a][b]=ddf[a][b];}}
   const auto u=model.Jet(x,v,d,dd);const auto c=h::EvolvedConstraints(u,model.Omega(x,u));if(!c.valid)throw std::runtime_error("invalid sampled constraint");
   Real nr[3],mr=0,zr=0,grr=0,br=0,vr=0,Dr=0,Lr=0,dc=0,da=0,dv=0,dD=0;
   for(int a=0;a<3;++a)nr[a]=x[a]/r;
   for(int a=0;a<3;++a){mr+=nr[a]*c.momentum[a];zr+=nr[a]*c.z4.z_covector[a];br+=nr[a]*u.beta.value[a];vr+=nr[a]*v[22+a];Dr+=nr[a]*v[25+a];Lr+=nr[a]*v[15+a];dc+=nr[a]*d[0][a];da+=nr[a]*u.alpha.d[a];
    for(int b=0;b<3;++b){grr+=nr[a]*nr[b]*u.metric.g[a][b];dv+=nr[a]*nr[b]*d[22+a][b];dD+=nr[a]*nr[b]*d[25+a][b];}}
   if(weight>0)for(int k=0;k<2;++k)if(k==0||r>.12){sums[k][0]+=weight;sums[k][1]+=weight*c.hamiltonian*c.hamiltonian;sums[k][2]+=weight*c.momentum_conformal_norm2;sums[k][3]+=weight*mr*mr;}
   if(ray<0)continue;
   out<<ray<<','<<p<<','<<cell.block<<','<<block.layer<<','<<cell.i<<','<<cell.j<<','<<cell.k;
   for(auto a:{x[0],x[1],x[2],r,u.chi.value,u.alpha.value,grr,u.trace.value,u.theta.value,br,vr,Dr,Lr,dc,da,dv,dD,c.hamiltonian,mr,std::sqrt(c.momentum_conformal_norm2),zr})out<<','<<a;
   out<<'\n';++rows;
  }
  Real maximum=0;for(int k=0;k<2;++k){const auto&s=k?native.exterior:native.total;const Real expected[]={s.volume,s.h2,s.m2,s.mr2};for(int f=0;f<4;++f)maximum=std::max(maximum,std::abs(sums[k][f]-expected[f])/std::max(std::abs(expected[f]),Real(1e-30)));}
  if(maximum>1e-9)throw std::runtime_error("slice operator fails native volume-integral cross-check");
  out.close();if(!out)throw std::runtime_error("slice output failed");
  std::ofstream meta(prefix+".json");meta<<std::setprecision(17)<<"{\"schema\":"<<JsonString(checkpoint_magic)<<",\"n\":"<<n<<",\"t_over_M\":"<<3*e.time<<",\"steps\":"<<e.steps<<",\"rows\":"<<rows<<",\"native_volume_integral_relative_difference\":"<<maximum<<",\"cutoff\":0.12,\"ray_definition\":\"Native centers: ray0 face0 near center; ray1 face0 near corner; ray2 Cartesian x-line y=z=-dx/2; ray3 face1 near center. No interpolation.\",\"operator\":\"original mapped bulk derivatives and native donor-seven final3 radial rows; CPU constraint formula cross-checked against production device volume sums\"}\n";
 }catch(const std::exception&x){std::cerr<<x.what()<<'\n';code=1;}
 Kokkos::finalize();MPI_Finalize();return code;
}
