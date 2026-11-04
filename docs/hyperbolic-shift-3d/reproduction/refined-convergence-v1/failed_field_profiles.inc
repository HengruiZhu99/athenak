// Forensic scalar/radial projections on nominal shell cells only. No derivative
// reconstruction, constraint norm, scri substitution, or endpoint admission.
int main(int argc,char**argv){
 MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
 MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);Kokkos::initialize(argc,argv);int code=0;
 try{
  if(argc!=6)throw std::invalid_argument("usage: profiles topology_checkpoint config failed_active fresh_prefix N");
  const int n=std::stoi(argv[5]),rank=global_variable::my_rank;const std::string prefix=argv[4];
  auto&cfg=hyperbolic_reproduction::refined_configuration;cfg.Load(argv[2],hyperbolic_reproduction::ContinuousCoreMesh(n));
  cfg.identity+=";amr_policy=smr";cfg.CheckCollective();Snapshot s;
  CollectiveLocal([&]{s=Read(argv[1],n,2,.000125);});
  h::CartesianAMR amr(n,s.subdivisions,std::max(32,1024/global_variable::nranks+8),std::max(3,cfg.smr_levels+1),cfg.shell_radii,cfg.core_half_width,cfg.shell_inner_radius);
  Restore(amr,s);const auto mesh=amr.Descriptor();
  V u(s.data[0].size());std::ifstream f(argv[3],std::ios::binary);f.read(reinterpret_cast<char*>(u.data()),u.size()*sizeof(Real));
  if(!f||f.peek()!=EOF)throw std::runtime_error("failed capture size mismatch");
  if(rank==0){
   const auto path=prefix+".csv";if(std::filesystem::exists(path))throw std::runtime_error("refusing overwrite");
   std::ofstream out(path);out<<std::setprecision(17)<<"state,cell,block,layer,face,i,j,k,x,y,z,r,chi,alpha,metric_rr,K,Theta,beta_dev_r,D_r,Lambda_r\n";
   const int ti[6]={0,0,0,1,1,2},tj[6]={0,1,2,1,2,2};
   for(size_t b=0;b<mesh.blocks.size();++b){const auto&block=mesh.blocks[b];if(block.face<0)continue;
    for(int k=0;k<n;++k)for(int j=0;j<n;++j)for(int i=0;i<n;++i){
     const size_t p=b*n*n*n+(k*n+j)*n+i;
     if(31*(p+1)>u.size())throw std::runtime_error("nominal indexing mismatch");
     const Real q[3]={(i+.5)/n,(j+.5)/n,(k+.5)/n};Real x[3];mesh.Forward(b,q,x);
     const Real r=std::sqrt(x[0]*x[0]+x[1]*x[1]+x[2]*x[2]);Real nr[3]={x[0]/r,x[1]/r,x[2]/r};
     for(int state=0;state<2;++state){const auto&data=state?u:s.data[0];const Real*v=data.data()+31*p;
      Real grr=1,vr=0,Dr=0,Lr=0;
      for(int a=0;a<6;++a)grr+=(ti[a]==tj[a]?1:2)*nr[ti[a]]*nr[tj[a]]*v[1+a];
      for(int a=0;a<3;++a){vr+=nr[a]*v[22+a];Dr+=nr[a]*v[25+a];Lr+=nr[a]*v[15+a];}
      out<<state<<','<<p<<','<<b<<','<<block.layer<<','<<block.face<<','<<i<<','<<j<<','<<k;
      for(Real z:{x[0],x[1],x[2],r,v[0],v[18]+(1+r*r)/2,grr,v[13]-3,v[14],vr,Dr,Lr})out<<','<<z;out<<'\n';
     }
    }
   }
   out.close();if(!out)throw std::runtime_error("profile write failed");
   std::ofstream meta(prefix+".json");meta<<std::setprecision(17)<<"{\"admitted\":false,\"reference_time\":"<<s.time<<",\"reference_steps\":"<<s.steps<<",\"active_cells\":"<<u.size()/31<<",\"nominal_cells\":"<<mesh.blocks.size()*n*n*n<<",\"qualification\":\"Nominal shell fields only; failed stage compared with separately admitted two-step SERIAL reference. No surface or overlap copies, no failed endpoint claim.\"}\n";
  }
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';code=2;}
 Kokkos::finalize();MPI_Finalize();return code;
}
