// Appended to a renamed copy of the production RF2 translation unit.
// Analysis only: no stepping or checkpoint writing; fixed derivative stencils
// with explicitly labeled optional halo-only diagnostic controls.
int main(int argc,char**argv) {
 MPI_Init(&argc,&argv);
 MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
 MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);
 Kokkos::initialize(argc,argv);int code=0;
 try {
  if(argc!=9)throw std::invalid_argument("N topology_checkpoint dt config uniform fresh_prefix cutoff native-initial|analytic-shell-halos");
  const std::string mode=argv[8];
  if(mode!="native-initial"&&mode!="analytic-shell-halos")throw std::invalid_argument("unknown initial analysis mode");
  const bool exact_halos=mode=="analytic-shell-halos",frame_control=false,legacy_frame=false,physical_frame=false,cartesian_donors=false;
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
    std::max(3,cfg.smr_levels+1),cfg.shell_radii,cfg.core_half_width,cfg.shell_inner_radius,cfg.shell_angular_subdivisions);
  Restore(amr,s);auto descriptor=amr.Descriptor();
  Evolution e(n,2,2,"",&descriptor);
  h::NativeBoostedWormhole initial(cfg.boost_velocity,1./3.,h::boost_free_initial_gauge,cfg.initial_start_M,cfg.initial_end_M);
  for(size_t p=0;p<e.Model().grid.active.size();++p){const auto*x=e.Model().grid.geometry[p].x;auto f=initial.Fields(x);cfg.InitialFields(x,f);
    for(int j=0;j<31;++j)s.data[0][31*p+j]=f[j];}
  for(int a:{1,3,4,5,6,7})std::fill(s.data[a].begin(),s.data[a].end(),Real(0));
  const auto limit=h::NativeWormhole::ScriFields();
  for(int p=0;p<e.surface_total;++p)for(int f=0;f<31;++f)s.data[2][31*p+f]=limit[f];
  s.time=0;s.steps=0;s.tracker={0,0,0};s.floors={};s.diagnostics={};
  Install(e,s);e.RefreshBoundary(s.dt);
  const auto halo=e.exchange()->Fill(e.state);
  e.SynchronizeError(halo.invalid,"analysis halo geometry rejected");
  auto native=ReduceDiagnostic(e.kernel()->Diagnostics(e.exchange()->padded,cut,e.surface),false);
  e.SynchronizeError(native.invalid_cells,"analysis native geometry rejected");
  auto padded=e.Download(e.exchange()->padded);
  const auto surface=e.Download(e.surface);
  const auto& model=e.Model();const auto& g=model.grid;
  const auto production_native=native;
  uint64_t replaced=0;Real halo_maximum=0;
  if(frame_control){
   // Analysis-only late-time control: no analytic data, active-field mutation,
   // radial mask, or identity reinterpretation. Use stored donors exclusively.
   h::ConformalHaloPlan plan(g,cfg.gauge.profile);
   std::vector<bool> owned(g.mesh.blocks.size(),false);
   for(int p=e.first;p<e.last;++p)owned[g.cells[p].block]=true;
   const int bs=g.padded*g.padded*g.padded;
   for(const auto&row:g.halos)if(plan.enabled[row.target]&&owned[row.target/bs]){
    for(int f:{0,7,8,9,10,11,12,18}){
     const int base=f*g.count,first=row.entries.front().point;
     const Real anchor=plan.points[first].ToLegacy(f,padded[base+first]);Real value=anchor;
     for(const auto&entry:row.entries)value+=entry.weight*(plan.points[entry.point].ToLegacy(f,padded[base+entry.point])-anchor);
     value=plan.points[row.target].FromLegacy(f,value);
     halo_maximum=std::max(halo_maximum,std::abs(value-padded[base+row.target]));padded[base+row.target]=value;
    }
    ++replaced;
   }
   Buffer scratch("frame_control_halos",padded.size());e.CopyFromHost(padded,scratch);
   native=ReduceDiagnostic(e.kernel()->Diagnostics(scratch,cut,e.surface),false);
   e.SynchronizeError(native.invalid_cells,"frame halo analysis invalid geometry");
  }
  if(exact_halos){
   std::vector<bool> owned(g.mesh.blocks.size(),false);
   for(int p=e.first;p<e.last;++p)owned[g.cells[p].block]=true;
   h::NativeBoostedWormhole pgen(cfg.boost_velocity,1./3.,h::boost_free_initial_gauge,cfg.initial_start_M,cfg.initial_end_M);
   std::vector<Real> donor_omega,donor_reference;
   if(physical_frame){donor_omega.resize(g.count);donor_reference.resize(g.count);
    for(size_t p=0;p<g.active.size();++p){const auto*x=g.geometry[p].x;const Real r=model.Radius(x);const Real om=cfg.gauge.profile.At(x).omega;
     donor_omega[g.active[p]]=legacy_frame?(r>=cfg.gauge.profile.outer_radius?1:om/((1-r*r)/2)):om;
     donor_reference[g.active[p]]=(1+r*r)/2;}}
   for(const auto&row:g.halos){
    int index=row.target,w=g.padded;const int i=index%w-g.storage_halo;index/=w;
    const int j=index%w-g.storage_halo;index/=w;
    const int k=index%w-g.storage_halo,block=index/w;
    if(!owned[block]||g.inverse_active[row.target]>=0)continue;
    if(cartesian_donors){
     if(g.mesh.blocks[block].face<0||row.entries.empty()||row.boundary_weight!=0)continue;
     bool all_cartesian=true;for(const auto&donor:row.entries)
      all_cartesian=all_cartesian&&g.mesh.blocks[donor.point/(w*w*w)].face<0;
     if(!all_cartesian)continue;
    }

    const Real q[3]={(i+.5)/n,(j+.5)/n,(k+.5)/n};
    const auto geometry=g.mesh.Geometry(block,q);const auto*x=geometry.x;
    const Real r=model.Radius(x);if(r<=.70||r>=.975)continue;
    std::array<double,31> fields{};
    if(!physical_frame){fields=pgen.Fields(x);cfg.InitialFields(x,fields);}
    else {
     for(int f=0;f<31;++f)fields[f]=padded[f*g.count+row.target];
     const Real om=cfg.gauge.profile.At(x).omega;
     const Real target_omega=legacy_frame?(r>=cfg.gauge.profile.outer_radius?1:om/((1-r*r)/2)):om,target_reference=(1+r*r)/2;
     for(int f:{0,7,8,9,10,11,12,18}){
      auto convert=[&](int point){const Real om=donor_omega[point],q=padded[f*g.count+point];
       return f==0?om*om*q:(f==18?(q+donor_reference[point])/om:om*q);};
      const Real anchor=convert(row.entries.front().point);Real interpolated=anchor;
      for(const auto&entry:row.entries)interpolated+=entry.weight*(convert(entry.point)-anchor);
      fields[f]=f==0?interpolated/(target_omega*target_omega):(f==18?target_omega*interpolated-target_reference:interpolated/target_omega);
     }
    }
    for(int f=0;f<31;++f){auto&value=padded[f*g.count+row.target];halo_maximum=std::max(halo_maximum,std::abs(value-fields[f]));value=fields[f];}
    ++replaced;
   }
   // Analysis scratch only: native state, checkpoint and production halos remain unchanged.
   Buffer scratch("analytic_initial_halos",padded.size());e.CopyFromHost(padded,scratch);
   native=ReduceDiagnostic(e.kernel()->Diagnostics(scratch,cut,e.surface),false);
   e.SynchronizeError(native.invalid_cells,"analytic halo control invalid geometry");
  }
  MPI_Allreduce(MPI_IN_PLACE,&replaced,1,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&halo_maximum,1,MPI_ATHENA_REAL,MPI_MAX,MPI_COMM_WORLD);

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
       <<",\"analysis_override\":"<<JsonString(frame_control?"analysis-only production legacy-frame Cartesian-to-shell halos; stored donors at checkpoint time":legacy_frame?"legacy-frame interpolation of t0 Cartesian-to-shell chi,A,lapse ghosts at .70<r<.975":(physical_frame?"physical-frame interpolation of t0 Cartesian-to-shell chi,A,lapse ghosts at .70<r<.975":(cartesian_donors?"analytic t0 Cartesian-to-shell ghosts only at .70<r<.975":(exact_halos?"analytic t0 ghosts only at .70<r<.975":"none"))))
       <<",\"replaced_ghosts\":"<<replaced<<",\"maximum_ghost_change\":"<<halo_maximum
       <<",\"production_full_integrals\":["<<production_native.total.volume<<','<<production_native.total.h2<<','<<production_native.total.m2<<','<<production_native.total.mr2<<']'
       <<",\"cutoff\":"<<cut<<",\"native_volume_integral_relative_difference\":"<<maximum
       <<",\"rows\":"<<rows<<",\"evolved_overlap_cells\":"<<overlap<<",\"native_radial_closure_cells\":"<<radial<<",\"integrals\":[";
   for(int k=0;k<2;++k){if(k)meta<<',';meta<<'[';for(int f=0;f<4;++f){if(f)meta<<',';meta<<sums[k][f];}meta<<']';}
   meta<<"],\"radial_bins\":[";for(int b=0;b<100;++b){if(b)meta<<',';meta<<'['<<b*.01<<','<<(b+1)*.01;
    for(int f=0;f<4;++f)meta<<','<<bins[b][f];meta<<']';}
   meta<<"],\"qualification\":\"Read-only native RF2 topology/config verification followed by explicit analytic initial reconstruction, zero evolution steps, same OmegaAt and derivative stencils; if analysis_override is enabled its label identifies either a t0 annulus control or a checkpoint-time production frame-halo control using stored donors; never changes active fields or scri; CPU integrals reproduce native kernel. Exterior admission must be checked separately. Rays use native centers without interpolation; Cartesian y,z offset follows local spacing across SMR. Radial bins include full domain; puncture-interior norms are not exterior accuracy measures.\"}\n";
   meta.close();if(!meta)throw std::runtime_error("metadata write failed");}
 }catch(const std::exception&x){std::cerr<<x.what()<<'\n';code=1;}
 MPI_Allreduce(MPI_IN_PLACE,&code,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
 Kokkos::finalize();MPI_Finalize();return code;
}
