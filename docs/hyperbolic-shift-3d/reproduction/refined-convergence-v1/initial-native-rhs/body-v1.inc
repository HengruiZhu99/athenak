// Read-only analysis: native initial RHS with production KO and with KO removed
// from the bulk kernel only. No RK step, checkpoint write or campaign mutation.
int main(int argc,char**argv){
 MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
 MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);Kokkos::initialize(argc,argv);int code=0;
 try{
  if(argc!=6)throw std::invalid_argument("usage: initial_rhs N topology_checkpoint config fresh_prefix");
  const int n=std::stoi(argv[1]),rank=global_variable::my_rank;const std::string prefix=argv[4];
  // Fifth argument makes intentional analytic t0 reconstruction explicit.
  if(std::string(argv[5])!="reconstruct-initial")throw std::invalid_argument("explicit t0 reconstruction required");
  auto&cfg=hyperbolic_reproduction::refined_configuration;cfg.Load(argv[3],hyperbolic_reproduction::ContinuousCoreMesh(n));
  cfg.identity+=";amr_policy=smr";cfg.CheckCollective();Snapshot s;
  CollectiveLocal([&]{s=Read(argv[2],n,2,.000125*16/n);if(std::filesystem::exists(prefix+".json"))throw std::runtime_error("refusing overwrite");});
  h::CartesianAMR amr(n,s.subdivisions,std::max(32,1024/global_variable::nranks+8),cfg.smr_levels+1,cfg.shell_radii,cfg.core_half_width,cfg.shell_inner_radius);
  Restore(amr,s);auto descriptor=amr.Descriptor();Evolution e(n,2,2,"",&descriptor);
  h::NativeBoostedWormhole pgen(cfg.boost_velocity,1./3.,h::boost_free_initial_gauge,cfg.initial_start_M,cfg.initial_end_M);
  const auto&g=e.Model().grid;
  for(size_t p=0;p<g.active.size();++p){auto v=pgen.Fields(g.geometry[p].x);cfg.InitialFields(g.geometry[p].x,v);for(int f=0;f<31;++f)s.data[0][31*p+f]=v[f];}
  for(auto a:{1,3,4,5,6,7})std::fill(s.data[a].begin(),s.data[a].end(),Real(0));
  const auto limit=h::NativeWormhole::ScriFields();for(int p=0;p<e.surface_total;++p)for(int f=0;f<31;++f)s.data[2][31*p+f]=limit[f];
  s.steps=0;s.time=0;s.tracker={0,0,0};s.floors={};s.diagnostics={};Install(e,s);
  const Real dt=.000125*16/n;Buffer edge;
  const auto full=e.Download(e.RHS(dt,edge));
  const auto d=ReduceDiagnostic(e.kernel()->Diagnostics(e.exchange()->padded,.3,e.surface),false);
  e.SynchronizeError(d.invalid_cells,"invalid initial diagnostic");
  const auto stats=e.kernel()->Evaluate(e.exchange()->padded,dt,false,0,e.surface,e.exchange()->positive_ko,direct_lapse);
  e.SynchronizeError(stats.bad_cells,"invalid no-KO initial RHS");const auto no_ko=e.kernel()->DownloadRHS();
  const Real boundaries[8]={.78,.825,.875,.90,.925,.95,.975,1.000001};
  Real sums[3][7][31]{},maxima[3][7][31]{},volumes[7]{};
  std::ofstream ray(prefix+".rank"+std::to_string(rank)+".csv");ray<<std::setprecision(17)<<"mode,cell,block,layer,face,i,j,k,x,y,z,r,chi_rate,metric_rr_rate,K_rate,Lambda_r_rate,alpha_rate,beta_r_rate,D_r_rate\n";
  constexpr int ti[6]={0,0,0,1,1,2},tj[6]={0,1,2,1,2,2};
  for(int p=e.first;p<e.last;++p){
   const auto*x=g.geometry[p].x;const Real r=e.Model().Radius(x),w=g.DiagnosticVolume(p);if(r<.78)continue;
   int bin=0;while(bin<6&&r>=boundaries[bin+1])++bin;
   const auto&c=g.cells[p];const auto&b=g.mesh.blocks[c.block];const int local=p-e.first;
   if(w>0)volumes[bin]+=w;
   const bool profile=p<int(g.nominal_count)&&b.face==0&&((c.j==n/2&&c.k==n/2)||(c.j==n-1&&c.k==n-1));
   for(int mode=0;mode<3;++mode){Real value[31];
    for(int f=0;f<31;++f){value[f]=mode==0?full[31*local+f]:mode==1?no_ko[31*local+f]:full[31*local+f]-no_ko[31*local+f];if(!std::isfinite(value[f]))throw std::runtime_error("nonfinite rate");if(w>0){sums[mode][bin][f]+=w*value[f]*value[f];maxima[mode][bin][f]=std::max(maxima[mode][bin][f],std::abs(value[f]));}}
    if(profile){Real grr=0,lr=0,br=0,dr=0;for(int t=0;t<6;++t)grr+=(ti[t]==tj[t]?1:2)*x[ti[t]]*x[tj[t]]/(r*r)*value[1+t];for(int a=0;a<3;++a){lr+=x[a]/r*value[15+a];br+=x[a]/r*value[22+a];dr+=x[a]/r*value[25+a];}
     ray<<mode<<','<<p<<','<<c.block<<','<<b.layer<<','<<b.face<<','<<c.i<<','<<c.j<<','<<c.k;for(Real v:{x[0],x[1],x[2],r,value[0],grr,value[13],lr,value[18],br,dr})ray<<','<<v;ray<<'\n';}
   }
  }
  ray.close();if(!ray)throw std::runtime_error("RHS profile write failed");
  MPI_Allreduce(MPI_IN_PLACE,sums,3*7*31,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);MPI_Allreduce(MPI_IN_PLACE,maxima,3*7*31,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);MPI_Allreduce(MPI_IN_PLACE,volumes,7,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
  if(rank==0){std::ofstream o(prefix+".json");o<<std::setprecision(17)<<"{\"n\":"<<n<<",\"initial_reconstructed\":true,\"steps_taken\":0,\"full_H\":"<<Norm(d.total.h2,d.total.volume)<<",\"full_M\":"<<Norm(d.total.m2,d.total.volume)<<",\"cut03_H\":"<<Norm(d.exterior.h2,d.exterior.volume)<<",\"cut03_M\":"<<Norm(d.exterior.m2,d.exterior.volume)<<",\"regions\":[";
   for(int b=0;b<7;++b){if(b)o<<',';o<<"{\"lo\":"<<boundaries[b]<<",\"hi\":"<<boundaries[b+1]<<",\"volume\":"<<volumes[b]<<",\"modes\":[";
    for(int m=0;m<3;++m){if(m)o<<',';o<<"{\"name\":\""<<(m==0?"native":m==1?"bulk_KO_zero":"bulk_KO_contribution")<<"\",\"rms\":[";for(int f=0;f<31;++f){if(f)o<<',';o<<std::sqrt(sums[m][b][f]/volumes[b]);}o<<"],\"maximum\":[";for(int f=0;f<31;++f){if(f)o<<',';o<<maxima[m][b][f];}o<<"]}";}o<<"]}";}o<<"],\"qualification\":\"Read-only analytic t0 on checkpoint topology. Full native boundary refresh and halos retained in both modes; only bulk KO is zeroed in the diagnostic override. No RK steps or state/output checkpoint changes. First region can include stencils crossing gauge blend.\"}\n";
  }
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';code=2;}
 Kokkos::finalize();MPI_Finalize();return code;
}
