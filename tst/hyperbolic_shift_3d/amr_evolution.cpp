// Standalone AMR qualification reuses the production Evolution class and all
// compiled point equations, scri closures, RK stages, floors and tracker.
#define main hyperbolic_uniform_driver_main
#include "coupled_mpi.cpp"
#undef main
#include "z4c/hyperboloidal/cartesian_amr.hpp"

namespace {
static_assert(hyperbolic_reproduction::six_shells&&h::broad_smooth_a&&h::boost_free_initial_gauge&&native_wormhole&&direct_chi&&direct_lapse);
static_assert(!h::CompositeDissipation::consistent&&!cubed_sphere::scri_first6&&h::fixed_scri_penalty_scale==.6510416666660156);
constexpr char science[]="HYP3D_AMR1;checksum-fnv1a64;native31;Cartesian-components;RK4-compensated;FD4-biased4;KO8-target-spacing;halo-Lagrange6;AthenaK-point-transfer5;six-shells:.16,.305,.45,.6,.75,.875,1;core:.18;overlap3;C6:4M-12M;restoration4:.6-.9;blend:.2-.56;free-gauge;boost:.1;M:1/3;direct-chi-lapse;chi-floor:1e-8,.03;lapse-floor:1e-10;scri-original;angular2;penalty:.6510416666660156";
struct Snapshot {
  int n=0,kind=0,subdivisions=2;uint64_t steps=0,regrids=0;Real time=0,dt=0,seed=.07;
  std::array<Real,3> tracker{};std::array<Real,8> floors{};
  std::array<Real,8> diagnostics{};
  std::vector<LogicalLocation> leaves;std::vector<int> ages;
  // Accepted bulk, RK compensation, boundary, boundary compensation, per-cell
  // floors, pre-floor signs, boundary signs, immutable historical event ledger.
  std::array<V,8> data;
};
V GatherWidth(Evolution&e,Buffer owned,int width,bool edge=false){
  const int total=edge?e.surface_total:e.total;const auto source=e.Download(owned);
  std::vector<int> counts(e.size),offsets(e.size);V out(size_t(width)*total);
  for(int r=0;r<e.size;++r){counts[r]=(edge?e.surface_counts[r]:e.counts[r])/31*width;offsets[r]=(edge?e.surface_displacements[r]:e.displacements[r])/31*width;}
  MPI_Allgatherv(source.data(),source.size(),MPI_ATHENA_REAL,out.data(),counts.data(),offsets.data(),MPI_ATHENA_REAL,MPI_COMM_WORLD);return out;
}
Snapshot Capture(Evolution&e,h::CartesianAMR&a,int kind,Real dt,uint64_t regrids,Real seed,const V&ledger){
  Snapshot s;s.n=e.n;s.kind=kind;s.subdivisions=a.mesh->nmb_rootx1;s.steps=e.steps;s.time=e.time;s.dt=dt;s.tracker=e.puncture;s.regrids=regrids;s.seed=seed;
  s.leaves.assign(a.mesh->lloc_eachmb,a.mesh->lloc_eachmb+a.mesh->nmb_total);
  for(int m=0;m<a.mesh->nmb_total;++m)s.ages.push_back(a.mesh->pmr->ncyc_since_ref(m));
  s.data[0]=e.Download(e.CompleteState(e.state));s.data[1]=GatherWidth(e,e.compensation,31);
  s.data[2]=e.Download(e.surface);s.data[3]=GatherWidth(e,e.surface_compensation,31,true);
  const auto f=e.Download(e.floor_statistics);for(int i=0;i<8;++i)s.floors[i]=f[i];
  for(int i=0;i<8;++i)MPI_Allreduce(MPI_IN_PLACE,&s.floors[i],1,MPI_ATHENA_REAL,(i==1||i==4)?MPI_MAX:MPI_SUM,MPI_COMM_WORLD);
  s.data[4]=GatherWidth(e,Kokkos::subview(e.floor_statistics,std::make_pair(size_t(8),size_t(8+e.local))),1);
  s.data[5]=GatherWidth(e,e.candidate_statistics,16);s.data[6]=GatherWidth(e,e.surface_candidate_statistics,8,true);s.data[7]=ledger;
  Real boundary_signs[3]={};
  for(auto*halo:{e.exchange(),e.normal_exchange()}){const auto signs=e.Download(halo->chi_sign_statistics);
    for(size_t i=0;i<signs.size();i+=3){boundary_signs[0]+=signs[i];boundary_signs[1]+=signs[i+1];boundary_signs[2]=std::min(boundary_signs[2],signs[i+2]);}}
  MPI_Allreduce(MPI_IN_PLACE,boundary_signs,2,MPI_ATHENA_REAL,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,boundary_signs+2,1,MPI_ATHENA_REAL,MPI_MIN,MPI_COMM_WORLD);
  if(boundary_signs[0]+boundary_signs[1]){const auto at=s.data[7].size();s.data[7].resize(at+21,0);
    s.data[7][at]=-1; // boundary aggregate; bulk events use nonnegative generation
    s.data[7][at+1]=s.time;for(int i=0;i<3;++i)s.data[7][at+2+i]=boundary_signs[i];}

  s.diagnostics={e.null_max,e.null_ghost,e.pole_max,e.clock_max,e.compatibility_max,e.cfl,e.gap,e.scri_penalty_cfl_max};
  for(int i=0;i<8;++i)MPI_Allreduce(MPI_IN_PLACE,&s.diagnostics[i],1,MPI_ATHENA_REAL,i==6?MPI_MIN:MPI_MAX,MPI_COMM_WORLD);
  return s;
}
void Validate(const cubed_sphere::CompositeDifference&g,const V&v){
  if(v.size()!=31*g.active.size())throw std::runtime_error("AMR field shape mismatch");
  for(size_t p=0;p<g.active.size();++p){const Real*q=&v[31*p];
    for(int f=0;f<31;++f)if(!std::isfinite(q[f]))throw std::runtime_error("AMR nonfinite interpolated field at cell "+std::to_string(p)+" field "+std::to_string(f));
    const Real A=1+q[1],B=q[2],C=q[3],D=1+q[4],E=q[5],F=1+q[6];
    Real r2=0;for(auto x:g.geometry[p].x)r2+=x*x;
    if(!(q[0]>0&&q[18]+(1+r2)/2>0&&A>0&&A*D-B*B>0&&A*(D*F-E*E)-B*(B*F-C*E)+C*(B*E-C*D)>0))
      {std::ostringstream why;why<<std::setprecision(17)<<"AMR invalid interpolated geometry at cell "<<p<<" xyz="<<g.geometry[p].x[0]<<","<<g.geometry[p].x[1]<<","<<g.geometry[p].x[2]<<" chi="<<q[0]<<" lapse="<<q[18]+(1+r2)/2<<" metric_minors="<<A<<","<<A*D-B*B<<","<<A*(D*F-E*E)-B*(B*F-C*E)+C*(B*E-C*D);throw std::runtime_error(why.str());}
  }
}
void Install(Evolution&e,const Snapshot&s){
  Validate(e.Model().grid,s.data[0]);
  if(s.data[1].size()!=s.data[0].size()||s.data[2].size()!=e.surface.extent(0)||s.data[3].size()!=s.data[2].size()
      ||s.data[4].size()!=size_t(e.total)||s.data[5].size()!=size_t(16*e.total)||s.data[6].size()!=size_t(8*e.surface_total))throw std::runtime_error("incomplete AMR checkpoint arrays");
  e.Upload(s.data[0]);e.CopyFromHost(s.data[2],e.surface);
  e.CopyFromHost(V(s.data[1].begin()+31*e.first,s.data[1].begin()+31*e.last),e.compensation);
  e.CopyFromHost(V(s.data[3].begin()+31*e.surface_first,s.data[3].begin()+31*(e.surface_first+e.surface_local)),e.surface_compensation);
  V floors(8+e.local);if(e.rank==0)std::copy(s.floors.begin(),s.floors.end(),floors.begin());
  std::copy(s.data[4].begin()+e.first,s.data[4].begin()+e.last,floors.begin()+8);e.CopyFromHost(floors,e.floor_statistics);
  e.CopyFromHost(V(s.data[5].begin()+16*e.first,s.data[5].begin()+16*e.last),e.candidate_statistics);
  e.CopyFromHost(V(s.data[6].begin()+8*e.surface_first,s.data[6].begin()+8*(e.surface_first+e.surface_local)),e.surface_candidate_statistics);
  e.time=s.time;e.steps=s.steps;e.puncture=s.tracker;e.track_puncture=s.kind==2;
  e.null_max=s.diagnostics[0];e.null_ghost=s.diagnostics[1];e.pole_max=s.diagnostics[2];e.clock_max=s.diagnostics[3];
  e.compatibility_max=s.diagnostics[4];e.cfl=s.diagnostics[5];e.gap=s.diagnostics[6];e.scri_penalty_cfl_max=s.diagnostics[7];
}
template<class T>void Put(std::ostream&o,const T&v){o.write(reinterpret_cast<const char*>(&v),sizeof(v));}
template<class T>void Get(std::istream&i,T&v){if(!i.read(reinterpret_cast<char*>(&v),sizeof(v)))throw std::runtime_error("truncated AMR checkpoint");}
uint64_t Checksum(std::istream&in,uint64_t size){
  uint64_t hash=14695981039346656037ull;char buffer[8192];
  while(size){const auto n=std::min<uint64_t>(sizeof(buffer),size);
    if(!in.read(buffer,n))throw std::runtime_error("AMR checksum read failed");
    for(size_t j=0;j<n;++j){hash^=static_cast<unsigned char>(buffer[j]);hash*=1099511628211ull;}size-=n;
  }return hash;
}
void Write(const std::string&path,const Snapshot&s){
  if(global_variable::my_rank)return;
  std::ofstream o(path+".tmp",std::ios::binary);o.write(science,sizeof(science));
  for(auto name:h::AMRFields::names){o.write(name.data(),name.size());o.put('\0');}
  Put(o,s.n);Put(o,s.kind);Put(o,s.subdivisions);Put(o,s.steps);Put(o,s.regrids);Put(o,s.time);Put(o,s.dt);Put(o,s.seed);Put(o,s.tracker);Put(o,s.floors);Put(o,s.diagnostics);
  Put(o,uint64_t(s.leaves.size()));for(auto l:s.leaves)Put(o,l);for(auto age:s.ages)Put(o,age);
  for(const auto&v:s.data){Put(o,uint64_t(v.size()));o.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(Real));}
  Put(o,uint64_t(0x414d5231434f4d50));o.close();if(!o)throw std::runtime_error("AMR checkpoint write failed");
  std::ifstream bytes(path+".tmp",std::ios::binary);const auto sum=Checksum(bytes,std::filesystem::file_size(path+".tmp"));
  std::ofstream append(path+".tmp",std::ios::binary|std::ios::app);Put(append,sum);append.close();if(!append)throw std::runtime_error("AMR checksum write failed");
  std::filesystem::rename(path+".tmp",path);
}
Snapshot Read(const std::string&path,int n,int kind,Real dt){
  std::ifstream i(path,std::ios::binary);char tag[sizeof(science)];i.read(tag,sizeof(tag));
  if(!i||std::memcmp(tag,science,sizeof(tag)))throw std::runtime_error("AMR scientific identity mismatch");
  for(auto name:h::AMRFields::names){std::string actual;std::getline(i,actual,'\0');if(actual!=name)throw std::runtime_error("AMR field layout mismatch");}
  Snapshot s;Get(i,s.n);Get(i,s.kind);Get(i,s.subdivisions);Get(i,s.steps);Get(i,s.regrids);Get(i,s.time);Get(i,s.dt);Get(i,s.seed);Get(i,s.tracker);Get(i,s.floors);Get(i,s.diagnostics);
  if(s.n!=n||s.kind!=kind||s.dt!=dt||(s.subdivisions!=2&&s.subdivisions!=4)||!std::isfinite(s.time)||s.time<0||!(s.seed>0&&s.seed<1))throw std::runtime_error("AMR checkpoint configuration mismatch");
  uint64_t count;Get(i,count);if(count<8||count>1024)throw std::runtime_error("AMR topology count");s.leaves.resize(count);s.ages.resize(count);
  for(auto&l:s.leaves)Get(i,l);for(auto&age:s.ages){Get(i,age);if(age<0)throw std::runtime_error("negative AMR age");}
  for(auto&v:s.data){Get(i,count);if(count>uint64_t(31)*1024*(n+8)*(n+8)*(n+8))throw std::runtime_error("AMR array size");
    v.resize(count);if(!i.read(reinterpret_cast<char*>(v.data()),count*sizeof(Real)))throw std::runtime_error("truncated AMR fields");
    for(Real x:v)if(!std::isfinite(x))throw std::runtime_error("nonfinite AMR checkpoint payload");}
  for(Real x:s.tracker)if(!std::isfinite(x)||std::abs(x)>=.16)throw std::runtime_error("invalid AMR tracker");
  if(s.data[7].size()%21)throw std::runtime_error("incomplete AMR event ledger");
  uint64_t tail;Get(i,tail);const uint64_t size=i.tellg();uint64_t stored;Get(i,stored);char extra;
  if(tail!=0x414d5231434f4d50||i.get(extra))throw std::runtime_error("incomplete or trailing AMR checkpoint");
  i.clear();i.seekg(0);if(Checksum(i,size)!=stored)throw std::runtime_error("AMR checkpoint checksum mismatch");return s;
}
void Restore(h::CartesianAMR&a,const Snapshot&s){
  auto equal=[](const auto&a,const auto&b){return a.lx1==b.lx1&&a.lx2==b.lx2&&a.lx3==b.lx3&&a.level==b.level;};
  for(const auto&l:s.leaves)if(l.level<a.mesh->root_level||l.level>a.mesh->max_level||l.lx1<0||l.lx2<0||l.lx3<0||l.lx1>=(1<<l.level)||l.lx2>=(1<<l.level)||l.lx3>=(1<<l.level))throw std::runtime_error("invalid AMR leaf");
  for(int level=a.mesh->root_level;level<a.mesh->max_level;++level){
    std::vector<int> flags(a.mesh->nmb_total);for(int m=0;m<a.mesh->nmb_total;++m){const auto&b=a.mesh->lloc_eachmb[m];
      for(const auto&l:s.leaves)if(l.level>b.level){const int shift=l.level-b.level;
        if((l.lx1>>shift)==b.lx1&&(l.lx2>>shift)==b.lx2&&(l.lx3>>shift)==b.lx3)flags[m]=1;}}
    a.Regrid(flags);
  }
  if(size_t(a.mesh->nmb_total)!=s.leaves.size())throw std::runtime_error("incomplete AMR leaf topology");
  for(size_t m=0;m<s.leaves.size();++m){if(!equal(a.mesh->lloc_eachmb[m],s.leaves[m]))throw std::runtime_error("noncanonical or unbalanced AMR leaves");a.mesh->pmr->ncyc_since_ref(m)=s.ages[m];}
}
void TransferExtra(const cubed_sphere::CompositeDifference&old,const cubed_sphere::CompositeDifference&next,Snapshot&s,const V&core,const V&core_rk){
  std::map<std::array<Real,3>,int> exact;
  for(size_t p=0;p<old.active.size();++p){const auto*x=old.geometry[p].x;exact[{x[0],x[1],x[2]}]=p;}
  V u(31*next.active.size()),rk(u.size());
  for(size_t p=0;p<next.active.size();++p){const auto*x=next.geometry[p].x;
    if(31*p<core.size()){for(int f=0;f<31;++f){u[31*p+f]=core[31*p+f];rk[31*p+f]=core_rk[31*p+f];}continue;}
    const auto cell=next.cells[p];const auto&block=next.mesh.blocks[cell.block];
    if(block.face>=0){int source=-1;
      for(int m=0;m<int(old.mesh.blocks.size());++m){const auto&b=old.mesh.blocks[m];
        if(b.face==block.face&&b.layer==block.layer){source=old.ActiveIndex(m,cell.i,cell.j,cell.k);break;}}
      if(source<0)throw std::runtime_error("missing fixed-shell cell during regrid");
      for(int f=0;f<31;++f){u[31*p+f]=s.data[0][31*source+f];rk[31*p+f]=s.data[1][31*source+f];}continue;
    }
    const auto found=exact.find({x[0],x[1],x[2]});
    if(found!=exact.end()){for(int f=0;f<31;++f){u[31*p+f]=s.data[0][31*found->second+f];rk[31*p+f]=s.data[1][31*found->second+f];}continue;}
    const auto row=old.Interpolate(x,6);if(row.empty())throw std::runtime_error("missing regrid overlap donor");const int anchor=old.inverse_active[row[0].point];
    for(int f=0;f<31;++f){Real v=s.data[0][31*anchor+f],c=s.data[1][31*anchor+f];for(const auto&e:row){const int q=old.inverse_active[e.point];if(q<0)throw std::runtime_error("invalid regrid overlap donor");v+=e.weight*(s.data[0][31*q+f]-s.data[0][31*anchor+f]);c+=e.weight*(s.data[1][31*q+f]-s.data[1][31*anchor+f]);}u[31*p+f]=v;rk[31*p+f]=c;}
  }
  // Preserve old events with their physical coordinates and mesh generation;
  // counts are never interpolated or multiplied when a leaf splits.
  for(size_t p=0;p<old.active.size();++p){bool event=s.data[4][p]!=0;
    for(int f:{0,1,2,3,8,9,10,11})event=event||s.data[5][16*p+f]!=0;
    if(event){s.data[7].push_back(s.regrids);for(auto x:old.geometry[p].x)s.data[7].push_back(x);s.data[7].push_back(s.data[4][p]);for(int f=0;f<16;++f)s.data[7].push_back(s.data[5][16*p+f]);}}
  s.data[0]=std::move(u);s.data[1]=std::move(rk);s.data[4].assign(next.active.size(),0);s.data[5].assign(16*next.active.size(),0);
}
}

int main(int argc,char**argv){
  MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);Kokkos::initialize(argc,argv);int code=0;
  try{
    if(argc<7||argc>8)throw std::invalid_argument("usage: amr_evolution n minkowski|wave|boost steps dt output [restart|-] [dynamic|fixed|uniform]");
    const int n=std::stoi(argv[1]),kind=std::string(argv[2])=="minkowski"?0:std::string(argv[2])=="wave"?1:std::string(argv[2])=="boost"?2:-1;
    const int stop=std::stoi(argv[3]);const Real dt=std::stod(argv[4]);const std::string output=argv[5],restart=argv[6],policy=argc>7?argv[7]:"dynamic";
    if(n<12||n%2||kind<0||stop<0||!std::isfinite(dt)||dt<=0||(policy!="dynamic"&&policy!="fixed"&&policy!="uniform"))throw std::invalid_argument("invalid AMR parameters");
    CollectiveLocal([&]{if(global_variable::my_rank==0){std::filesystem::create_directories(std::filesystem::path(output).parent_path());if(std::filesystem::exists(output+".checkpoint"))throw std::runtime_error("refusing AMR checkpoint overwrite");}});
    Snapshot s;if(restart!="-")CollectiveLocal([&]{s=Read(restart,n,kind,dt);});
    h::CartesianAMR amr(n,restart!="-"?s.subdivisions:policy=="uniform"?4:2,std::max(32,96/global_variable::nranks+8));
    if(restart!="-")Restore(amr,s);else if(policy!="uniform"){std::vector<int>flags(8);flags[0]=1;amr.Regrid(flags);}
    auto descriptor=amr.Descriptor();std::unique_ptr<Evolution> ev;
    CollectiveLocal([&]{ev=std::make_unique<Evolution>(n,2,kind,"",&descriptor);});
    if(restart!="-")Install(*ev,s);else{
      ev->Upload(ev->initial_state);
      s=Capture(*ev,amr,kind,dt,0,.07,{});h::NativeBoostedWormhole pgen;
      for(size_t p=0;p<ev->Model().grid.active.size();++p){const auto*x=ev->Model().grid.geometry[p].x;
        if(kind==2){const auto v=pgen.Fields(x);for(int f=0;f<31;++f)s.data[0][31*p+f]=v[f];}
        if(kind==1){const Real r2=x[0]*x[0]+x[1]*x[1]+x[2]*x[2];const Real wave=1e-5*std::exp(-r2/.025)*std::sin(17*x[0]+11*x[1]-13*x[2]);
          s.data[0][31*p+18]=wave;s.data[0][31*p+19]=.1*wave;s.data[0][31*p+25]=.2*wave;}
      }
      if(kind==2){const auto limit=h::NativeWormhole::ScriFields();for(int p=0;p<ev->surface_total;++p)for(int f=0;f<31;++f)s.data[2][31*p+f]=limit[f];}
      Install(*ev,s);
    }
    ParameterInput pin;std::unique_ptr<FastFlow> finder;
    if(kind==2){std::ostringstream input;input<<"<job>\nbasename="<<output<<".fastflow\n<fastflow>\nntheta=24\nlmax=12\nflow=standard\nflow_iterations_0=300\nflow_alpha_beta_const_0=.1\ninitial_radius_0="<<std::setprecision(17)<<s.seed<<"\nmass_tol_0=1e-10\nexpansion_rms_tol_0=.001\nhmean_tol_0=1e8\nstart_time_0=0\nstop_time_0=100\n";std::istringstream in(input.str());pin.LoadFromStream(in);
      finder=std::make_unique<FastFlow>(nullptr,&pin,0,[&](const Real x[3],z4c::HorizonGeometry<Real>&q){auto&e=*ev;const int block=e.Model().grid.mesh.Owner(x);if(block<0)return false;const int center=block*n*n*n+n*n*n/2;if(center<e.first||center>=e.last)return false;
        try{V v;std::vector<std::array<Real,3>>d;std::vector<std::array<std::array<Real,3>,3>>dd;e.Model().grid.SampleJets(e.owned_padding,31,x,v,d,dd,true);const auto u=e.Model().Jet(x,v,d,dd);q=h::PhysicalHorizonGeometry(u,e.Model().Omega(x,u));}catch(...){q={};}return true;});}
    std::ofstream history;if(ev->rank==0){history.open(output+".history.jsonl");history<<std::setprecision(17);}
    Real seed=s.seed;V ledger=s.data[7];uint64_t regrids=s.regrids;double began=MPI_Wtime();
    auto record=[&]{auto&e=*ev;e.RefreshBoundary(dt);auto halo=e.exchange()->Fill(e.state);e.SynchronizeError(halo.invalid,"AMR halo geometry rejected");
      const auto d=ReduceDiagnostic(e.kernel()->Diagnostics(e.exchange()->padded,0,e.surface));bool found=false;Real mass=std::numeric_limits<Real>::quiet_NaN();
      if(finder){e.DownloadOwnedPadded();for(int a=0;a<3;++a)finder->center[a]=e.puncture[a];finder->Find(e.steps,e.time);const auto q=finder->GetResult();found=q.found;if(found){mass=q.irreducible_mass;seed=q.mean_radius;}}
      if(e.rank==0)history<<"{\"step\":"<<e.steps<<",\"time\":"<<e.time<<",\"leaves\":"<<amr.mesh->nmb_total<<",\"H\":"<<Norm(d.total.h2,d.total.volume)<<",\"M\":"<<Norm(d.total.m2,d.total.volume)<<",\"chi\":"<<d.min_chi<<",\"lapse\":"<<d.min_lapse<<",\"det_residual\":"<<d.max_determinant_residual<<",\"null\":"<<e.null_max<<",\"ghost_null\":"<<e.null_ghost<<",\"cfl\":"<<e.cfl<<",\"penalty\":"<<h::fixed_scri_penalty_scale<<",\"penalty_cfl\":"<<e.scri_penalty_cfl_max<<",\"horizon_found\":"<<found<<",\"mass\":"<<JsonNumber(mass)<<",\"tracker\":["<<e.puncture[0]<<','<<e.puncture[1]<<','<<e.puncture[2]<<"]}\n";
      if(e.rank==0){history.flush();std::cout<<"accepted step "<<e.steps<<" leaves "<<amr.mesh->nmb_total<<" H "<<Norm(d.total.h2,d.total.volume)<<std::endl;}
    };
    record();
    try {
    while(ev->steps<uint64_t(stop)){
      if(policy=="dynamic"&&(ev->steps==2||ev->steps==4)){
        auto old=Capture(*ev,amr,kind,dt,regrids,seed,ledger);CollectiveLocal([&]{amr.Load(ev->Model().grid,old.data[0],old.data[1]);});std::vector<int>flags(amr.mesh->nmb_total);
        if(ev->steps==2){for(int m=0;m<amr.mesh->nmb_total;++m)if(amr.mesh->lloc_eachmb[m].level==amr.mesh->root_level){flags[m]=1;break;}}
        else for(int m=0;m<amr.mesh->nmb_total;++m)if(amr.mesh->lloc_eachmb[m].level>amr.mesh->root_level)flags[m]=-1;
        amr.Regrid(flags);descriptor=amr.Descriptor();std::unique_ptr<Evolution> next;
        // The replacement only needs the old active interpolation geometry.
        // Retire duplicated host halo rows before allocating another generation.
        ev->legacy_model->grid.halos.clear();ev->legacy_model->grid.halos.shrink_to_fit();
        ev->legacy_model->grid.extrapolated_halos.clear();ev->legacy_model->grid.extrapolated_halos.shrink_to_fit();
        CollectiveLocal([&]{next=std::make_unique<Evolution>(n,2,kind,"",&descriptor);});
        const auto core=amr.Gather(),rk=amr.Gather(true);TransferExtra(ev->Model().grid,next->Model().grid,old,core,rk);
        try{Install(*next,old);}catch(...){
          if(ev->rank==0){std::ofstream rejected(output+".rejected-transfer.bin",std::ios::binary);rejected.write(reinterpret_cast<const char*>(old.data[0].data()),old.data[0].size()*sizeof(Real));}
          throw;
        }
        ledger=old.data[7];ev.swap(next);++regrids;record();
      }
      // The native RHS measures the actual local characteristic CFL on every
      // new leaf. Reject an excessive requested step; do not change scri scale.
      ev->Step(dt);for(int m=0;m<amr.mesh->nmb_total;++m)++amr.mesh->pmr->ncyc_since_ref(m);record();
    }
    }catch(const std::exception&failure){
      const auto capture=ev->failed_state.extent(0)?ev->Download(ev->CompleteState(ev->failed_state)):ev->Download(ev->CompleteState(ev->state));
      CollectiveLocal([&]{if(ev->rank==0){std::ofstream f(output+".failed-stage.bin",std::ios::binary);f.write(reinterpret_cast<const char*>(capture.data()),capture.size()*sizeof(Real));
        std::ofstream report(output+".failure.json");report<<"{\"status\":\"failed\",\"accepted_steps\":"<<ev->steps<<",\"error\":"<<JsonString(failure.what())<<"}\n";}});
      throw;
    }
    auto final=Capture(*ev,amr,kind,dt,regrids,seed,ledger);CollectiveLocal([&]{Write(output+".checkpoint",final);});
    if(ev->rank==0){std::ofstream report(output+".report.json");report<<std::setprecision(17)<<"{\"status\":\"completed\",\"backend\":\""<<Kokkos::DefaultExecutionSpace::name()<<"\",\"ranks\":"<<ev->size<<",\"steps\":"<<ev->steps<<",\"regrids\":"<<regrids<<",\"seconds\":"<<MPI_Wtime()-began<<",\"floor_counts\":["<<final.floors[0]<<','<<final.floors[2]<<','<<final.floors[3]<<"],\"historical_event_records\":"<<ledger.size()/21<<"}\n";}
  }catch(const std::exception&e){std::cerr<<"AMR qualification failed: "<<e.what()<<'\n';code=2;}
  MPI_Allreduce(MPI_IN_PLACE,&code,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);Kokkos::finalize();MPI_Finalize();return code;
}
