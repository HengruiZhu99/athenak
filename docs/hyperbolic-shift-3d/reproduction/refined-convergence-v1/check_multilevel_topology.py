"""Exercise the actual production Restore on a three-level native SMR octree."""
from pathlib import Path
import subprocess,shlex,json,hashlib
HERE=Path(__file__).resolve().parent;ROOT=next(p for p in HERE.parents if (p/'CMakeLists.txt').exists())
OUT=ROOT/'output/refined-convergence';build=ROOT/'build-refined-convergence';target=build/'tst/hyperbolic_shift_3d'
src=ROOT/'tst/hyperbolic_shift_3d/amr_evolution.cpp';text=src.read_text().replace('int main(int argc, char** argv) {','int production_driver_main(int argc, char** argv) {')
text+='''
int main(int argc,char**argv){
 MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
 MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);Kokkos::initialize(argc,argv);int code=0;
 try{
  auto& cfg=hyperbolic_reproduction::refined_configuration;
  cfg.Load(argv[1],hyperbolic_reproduction::ContinuousCoreMesh(16));
  Snapshot saved;std::vector<int> counts;
  auto create=[&]{return std::make_unique<h::CartesianAMR>(8,cfg.core_subdivisions,
    512/global_variable::nranks+8,cfg.smr_levels+1,cfg.shell_radii,cfg.core_half_width,cfg.shell_inner_radius);};
  {
   auto a=create();
   for(int l=0;l<cfg.smr_levels;++l){
    for(int m=0;m<a->mesh->nmb_total;++m)a->mesh->pmr->ncyc_since_ref(m)=a->mesh->pmr->refinement_interval;
    auto& criterion=a->mesh->pmr->pmrc->rcrit.at(0);criterion.rloc_x1=criterion.rloc_x2=criterion.rloc_x3=0;criterion.rloc_rad=cfg.smr_radius;
    a->mesh->ncycle=0;a->mesh->pmr->CheckForRefinement(a->mesh->pmb_pack);
    std::vector<int> flags(a->mesh->nmb_total);int flagged=0;
    for(int m=0;m<a->mesh->nmb_total;++m){flags[m]=a->mesh->pmr->refine_flag.h_view(m);flagged+=flags[m]>0;}
    if(global_variable::my_rank==0)std::cout<<"requested "<<flagged<<" / "<<flags.size()<<std::endl;
    if(!a->Regrid(flags))throw std::runtime_error("missing SMR level");
    counts.push_back(a->mesh->nmb_total);if(global_variable::my_rank==0)std::cout<<"level "<<l+1<<" leaves "<<a->mesh->nmb_total<<std::endl;
   }
   saved.leaves.assign(a->mesh->lloc_eachmb,a->mesh->lloc_eachmb+a->mesh->nmb_total);
   saved.ages.resize(a->mesh->nmb_total);for(int m=0;m<a->mesh->nmb_total;++m)saved.ages[m]=a->mesh->pmr->ncyc_since_ref(m);
  }
  {auto b=create();Restore(*b,saved);}
  if(global_variable::my_rank==0){std::cout<<"{\\\"passed\\\":true,\\\"levels\\\":"<<cfg.smr_levels<<",\\\"leaves\\\":[";
   for(size_t i=0;i<counts.size();++i){if(i)std::cout<<',';std::cout<<counts[i];}std::cout<<"]}\\n";}
 }catch(const std::exception&e){std::cerr<<e.what()<<'\\n';code=2;}
 MPI_Allreduce(MPI_IN_PLACE,&code,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);Kokkos::finalize();MPI_Finalize();return code;
}
'''
source=OUT/'multilevel-topology.cpp';source.write_text(text)
flags={}
for line in (target/'CMakeFiles/hyperbolic_shift_3d_refined_amr.dir/flags.make').read_text().splitlines():
 if ' = ' in line:k,v=line.split(' = ',1);flags[k]=shlex.split(v)
obj=OUT/'multilevel-topology.o';binary=OUT/'multilevel-topology'
cmd=['/usr/bin/c++']+flags['CXX_DEFINES']+flags['CXX_INCLUDES']+flags['CXX_FLAGS']+['-I'+str(src.parent),'-c',str(source),'-o',str(obj)]
subprocess.run(cmd,check=True)
link=shlex.split((target/'CMakeFiles/hyperbolic_shift_3d_refined_amr.dir/link.txt').read_text());link[link.index('CMakeFiles/hyperbolic_shift_3d_refined_amr.dir/amr_evolution.cpp.o')]=str(obj);link[link.index('-o')+1]=str(binary)
subprocess.run(link,cwd=target,check=True)
run=['mpiexec','-n','2',str(binary),str(ROOT/'inputs/hyperbolic_shift_3d/refined_convergence/central20_stationary_smr025.cfg')]
result=subprocess.run(run,text=True,capture_output=True);(OUT/'multilevel-topology.log').write_text(result.stdout+result.stderr);assert result.returncode==0,result.stderr
q=json.loads(next(l for l in result.stdout.splitlines() if l.startswith('{"passed"')));q.update(command=run,source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),scope='Topology construction and actual Restore only, MPI2; no evolved fields or stability claim.')
(HERE/'multilevel-topology.json').write_text(json.dumps(q,indent=2)+'\n');print(q)
