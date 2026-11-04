// Appended to a renamed copy of the production RF2 translation unit.
// Analysis only: no stepping, checkpoint writing or changed derivative operator.
int main(int argc,char**argv) {
 MPI_Init(&argc,&argv);
 MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
 MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);
 Kokkos::initialize(argc,argv);int code=0;
 try {
  if(argc!=8)throw std::invalid_argument("usage: diagnostic N checkpoint dt config smr|uniform fresh_prefix cutoff");
  const int n=std::stoi(argv[1]),rank=global_variable::my_rank;
  const Real dt=std::stod(argv[3]),cut=std::stod(argv[7]);
  const std::string prefix=argv[6],policy=argv[5];
  if(policy!="smr"&&policy!="uniform")throw std::invalid_argument("unsupported policy");
  auto& cfg=hyperbolic_reproduction::refined_configuration;
  cfg.Load(argv[4],hyperbolic_reproduction::ContinuousCoreMesh(n));
  cfg.identity+=";amr_policy="+policy;cfg.CheckCollective();
  Snapshot s;
  CollectiveLocal([&]{s=Read(argv[2],n,2,dt);
   if(!(s.dt>0&&s.dt<=dt))throw std::invalid_argument("checkpoint dt exceeds bound");
   if(std::filesystem::exists(prefix+".json")||std::filesystem::exists(prefix+".rank"+std::to_string(rank)+".csv"))
    throw std::runtime_error("refusing analysis overwrite");});
  h::CartesianAMR amr(n,s.subdivisions,std::max(32,1024/global_variable::nranks+8),
    std::max(3,cfg.smr_levels+1),cfg.shell_radii,cfg.core_half_width,cfg.shell_inner_radius);
  Restore(amr,s);auto descriptor=amr.Descriptor();
  Evolution e(n,2,2,"",&descriptor);Install(e,s);e.RefreshBoundary(s.dt);
  const auto halo=e.exchange()->Fill(e.state);
  e.SynchronizeError(halo.invalid,"analysis halo geometry rejected");
  const auto native=ReduceDiagnostic(e.kernel()->Diagnostics(e.exchange()->padded,cut,e.surface),false);
  e.SynchronizeError(native.invalid_cells,"analysis native geometry rejected");
  const auto padded=e.Download(e.exchange()->padded),surface=e.Download(e.surface);
  const auto& model=e.Model();const auto& g=model.grid;
  std::ofstream out(prefix+".rank"+std::to_string(rank)+".csv");out<<std::setprecision(17);
  out<<"ray,cell,block,layer,level,i,j,k,x,y,z,r,volume,chi,alpha,metric_rr,K,Theta,beta_r,beta_dev_r,D_r,Lambda_r,dr_chi,dr_alpha,dr_beta_dev_r,dr_D_r,H,Mr,Mnorm,Zr\n";
  V v(31);std::vector<std::array<Real,3>>d(31);std::vector<std::array<std::array<Real,3>,3>>dd(31);
  Real sums[2][4]{},bins[100][4]{};uint64_t rows=0,overlap=0,radial=0;
  CollectiveLocal([&]{for(int p=e.first;p<e.last;++p){
   const auto*x=g.geometry[p].x;const Real r=model.Radius(x),weight=g.DiagnosticVolume(p);
   const auto&cell=g.cells[p];const auto&block=g.mesh.blocks[cell.block];
   overlap+=p>=int(g.nominal_count);radial+=model.radial_plan&&model.radial_plan->Contains(p);
   int ray=-1;
   if(block.face==0){if(cell.j==n/2&&cell.k==n/2)ray=0;else if(cell.j==n-1&&cell.k==n-1)ray=1;}
   if(block.face==1&&cell.j==n/2&&cell.k==n/2)ray=3;
   if(block.face<0){const Real dx=(block.hi[0]-block.lo[0])/n;
    if(x[1]<0&&x[1]>-dx&&x[2]<0&&x[2]>-dx)ray=2;}
   if(weight<=0&&ray<0)continue;
   for(int f=0;f<31;++f){v[f]=padded[f*g.count+g.active[p]];Real df[3]{},ddf[3][3]{};
    const bool second=f<=6||f==18||(f>=22&&f<25);
    // The immutable host plan retains initial surface values. Use the actual
    // refreshed checkpoint boundary in the same donor-seven closure.
    if(model.radial_plan&&model.radial_plan->Contains(p))
      model.radial_plan->Derivatives(padded,surface,f,p,df,ddf,second);
    else g.Derivatives(padded,f,p,df,ddf,second);
    for(int a=0;a<3;++a){d[f][a]=df[a];for(int b=0;b<3;++b)dd[f][a][b]=ddf[a][b];}}
   const auto u=model.Jet(x,v,d,dd);const auto c=h::EvolvedConstraints(u,model.OmegaAt(x,u));
   if(!c.valid)throw std::runtime_error("invalid CPU constraint");
   Real nr[3],mr=0,zr=0,grr=0,br=0,vr=0,Dr=0,Lr=0,dc=0,da=0,dv=0,dD=0;
   for(int a=0;a<3;++a)nr[a]=x[a]/r;
   for(int a=0;a<3;++a){mr+=nr[a]*c.momentum[a];zr+=nr[a]*c.z4.z_covector[a];
    br+=nr[a]*u.beta.value[a];vr+=nr[a]*v[22+a];Dr+=nr[a]*v[25+a];Lr+=nr[a]*v[15+a];
    dc+=nr[a]*d[0][a];da+=nr[a]*u.alpha.d[a];
    for(int b=0;b<3;++b){grr+=nr[a]*nr[b]*u.metric.g[a][b];dv+=nr[a]*nr[b]*d[22+a][b];dD+=nr[a]*nr[b]*d[25+a][b];}}
   const Real integrals[4]={weight,weight*c.hamiltonian*c.hamiltonian,weight*c.momentum_conformal_norm2,weight*mr*mr};
   if(weight>0){for(int k=0;k<2;++k)if(k==0||r>cut)for(int f=0;f<4;++f)sums[k][f]+=integrals[f];
    const int b=std::clamp(int(r*100),0,99);for(int f=0;f<4;++f)bins[b][f]+=integrals[f];}
   if(ray<0)continue;
   out<<ray<<','<<p<<','<<cell.block<<','<<block.layer<<','<<block.refinement<<','<<cell.i<<','<<cell.j<<','<<cell.k;
   for(auto a:{x[0],x[1],x[2],r,weight,u.chi.value,u.alpha.value,grr,u.trace.value,u.theta.value,br,vr,Dr,Lr,dc,da,dv,dD,c.hamiltonian,mr,std::sqrt(c.momentum_conformal_norm2),zr})out<<','<<a;
   out<<'\n';++rows;
  }out.close();if(!out)throw std::runtime_error("slice output write failed");});
  MPI_Allreduce(MPI_IN_PLACE,sums,8,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,bins,400,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
  for(auto*p:{&rows,&overlap,&radial})MPI_Allreduce(MPI_IN_PLACE,p,1,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
  Real maximum=0;
  for(int k=0;k<2;++k){const auto&q=k?native.exterior:native.total;
   const Real expected[]={q.volume,q.h2,q.m2,q.mr2};
   for(int f=0;f<4;++f)maximum=std::max(maximum,std::abs(sums[k][f]-expected[f])/std::max(std::abs(expected[f]),Real(1e-30)));}
  if(maximum>1e-9)throw std::runtime_error("CPU/operator volume cross-check failed: "+std::to_string(maximum));
  if(!rank){std::ofstream meta(prefix+".json");meta<<std::setprecision(17);
   meta<<"{\"identity\":"<<JsonString(cfg.identity)<<",\"n\":"<<n<<",\"t_over_M\":"<<3*s.time<<",\"steps\":"<<s.steps
       <<",\"cutoff\":"<<cut<<",\"native_volume_integral_relative_difference\":"<<maximum
       <<",\"rows\":"<<rows<<",\"evolved_overlap_cells\":"<<overlap<<",\"native_radial_closure_cells\":"<<radial<<",\"integrals\":[";
   for(int k=0;k<2;++k){if(k)meta<<',';meta<<'[';for(int f=0;f<4;++f){if(f)meta<<',';meta<<sums[k][f];}meta<<']';}
   meta<<"],\"radial_bins\":[";for(int b=0;b<100;++b){if(b)meta<<',';meta<<'['<<b*.01<<','<<(b+1)*.01;
    for(int f=0;f<4;++f)meta<<','<<bins[b][f];meta<<']';}
   meta<<"],\"qualification\":\"Read-only native RF2 Restore/Install, same OmegaAt and native derivative operators; CPU integrals reproduce native kernel. Exterior admission must be checked separately. Rays use native centers without interpolation; Cartesian y,z offset follows local spacing across SMR. Radial bins include full domain; puncture-interior norms are not exterior accuracy measures.\"}\n";
   meta.close();if(!meta)throw std::runtime_error("metadata write failed");}
 }catch(const std::exception&x){std::cerr<<x.what()<<'\n';code=1;}
 MPI_Allreduce(MPI_IN_PLACE,&code,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
 Kokkos::finalize();MPI_Finalize();return code;
}
