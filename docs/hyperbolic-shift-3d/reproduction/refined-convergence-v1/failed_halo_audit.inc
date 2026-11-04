// Read-only forensic use of an admitted local topology fixture plus raw failed
// active fields. Never install the failed state as a checkpoint or evolve it.
// Surface-dependent rows are excluded: the failed surface was not saved.
int main(int argc,char**argv) {
 MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
 MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);Kokkos::initialize(argc,argv);int code=0;
 try {
  if(argc!=6)throw std::invalid_argument("usage: audit topology_checkpoint config failed_active fresh_prefix N");
  const int n=std::stoi(argv[5]),rank=global_variable::my_rank;const std::string prefix=argv[4];
  auto& cfg=hyperbolic_reproduction::refined_configuration;
  cfg.Load(argv[2],hyperbolic_reproduction::ContinuousCoreMesh(n));cfg.identity+=";amr_policy=smr";cfg.CheckCollective();
  Snapshot s;CollectiveLocal([&]{s=Read(argv[1],n,2,.000125);});
  h::CartesianAMR amr(n,s.subdivisions,std::max(32,1024/global_variable::nranks+8),std::max(3,cfg.smr_levels+1),cfg.shell_radii,cfg.core_half_width,cfg.shell_inner_radius);
  Restore(amr,s);auto descriptor=amr.Descriptor();Evolution e(n,2,2,"",&descriptor);
  const auto& model=e.Model();const auto& g=model.grid;
  V u(g.active.size()*31);std::ifstream input(argv[3],std::ios::binary);
  input.read(reinterpret_cast<char*>(u.data()),u.size()*sizeof(Real));
  if(!input||input.peek()!=EOF)throw std::runtime_error("failed active length/topology mismatch");
  const auto path=prefix+".rank"+std::to_string(rank)+".csv";
  if(std::filesystem::exists(path))throw std::runtime_error("refusing overwrite");
  std::ofstream out(path);out<<std::setprecision(17)<<"block,face,target,x,y,z,r,chi,lapse,minor1,minor2,det,finite,frame,donors,weight_l1\n";
  std::vector<bool> owned(g.mesh.blocks.size(),false);for(int i=e.first;i<e.last;++i)owned[g.cells[i].block]=true;
  h::ConformalHaloPlan frame(g,cfg.gauge.profile);size_t checked=0,bad=0,skipped=0;
  const int w=g.padded,bs=w*w*w,o=g.storage_halo;
  for(const auto& row:g.halos){
   const int block=row.target/bs;if(!owned[block])continue;
   if(row.boundary_weight!=0){++skipped;continue;}
   std::array<Real,31> v{};bool finite=true;Real weight_l1=0;
   const bool framed=cfg.gauge.conformal_frame_halos&&frame.enabled[row.target];
   for(const auto& d:row.entries)weight_l1+=std::abs(d.weight);
   for(int f=0;f<31;++f){
    auto donor=[&](int p){const int active=g.inverse_active[p];if(active<0)throw std::runtime_error("nonactive donor");const Real a=u[31*active+f];return framed&&h::ConformalHaloPoint::Transformed(f)?frame.points[p].ToLegacy(f,a):a;};
    const Real anchor=donor(row.entries.front().point);v[f]=anchor;
    for(const auto& d:row.entries)v[f]+=d.weight*(donor(d.point)-anchor);
    if(framed&&h::ConformalHaloPoint::Transformed(f))v[f]=frame.points[row.target].FromLegacy(f,v[f]);
    finite=finite&&std::isfinite(v[f]);
   }
   const int off=row.target%bs;const Real q[3]={(off%w-o+.5)/n,((off/w)%w-o+.5)/n,(off/(w*w)-o+.5)/n};Real x[3];g.mesh.Forward(block,q,x);
   const Real r=std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]),lapse=v[18]+(1+r*r)/2;
   const Real A=1+v[1],B=v[2],C=v[3],D=1+v[4],E=v[5],F=1+v[6],m2=A*D-B*B,det=A*(D*F-E*E)-B*(B*F-C*E)+C*(B*E-C*D);
   ++checked;if(finite&&v[0]>0&&lapse>0&&A>0&&m2>0&&det>0)continue;++bad;
   out<<block<<','<<g.mesh.blocks[block].face<<','<<row.target<<','<<x[0]<<','<<x[1]<<','<<x[2]<<','<<r<<','<<v[0]<<','<<lapse<<','<<A<<','<<m2<<','<<det<<','<<finite<<','<<framed<<','<<row.entries.size()<<','<<weight_l1<<'\n';
  }
  std::ofstream report(prefix+".rank"+std::to_string(rank)+".json");report<<"{\"checked\":"<<checked<<",\"invalid_nonboundary_rows\":"<<bad<<",\"surface_rows_excluded\":"<<skipped<<",\"active_cells\":"<<g.active.size()<<",\"admitted\":false}\n";
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';code=2;}
 Kokkos::finalize();MPI_Finalize();return code;
}
