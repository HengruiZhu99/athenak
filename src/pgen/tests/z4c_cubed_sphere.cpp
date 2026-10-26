#include <stdexcept>
#include <string>
#include "athena.hpp"
#include "mesh/mesh.hpp"
#include "mesh/cubed_sphere/extension.hpp"
#include "mesh/cubed_sphere/test_state.hpp"
#include "pgen/pgen.hpp"
#include "z4c/z4c.hpp"

namespace {
bool refine_corners=false;
void Finish(ParameterInput *pin,Mesh *mesh) {
  if (mesh->pmb_pack->pz4c->cubed_sphere)
    mesh->pmb_pack->pz4c->cubed_sphere->Report(pin);
}
// A bounded refine/derefine cycle exercises donor-plan rebuilding and native AMR
// transfer without depending on a physical solution to trigger topology changes.
void Refine(MeshBlockPack *pack) {
  auto *mesh=pack->pmesh;
  auto &flag=mesh->pmr->refine_flag;
  auto sizes=pack->pmb->mb_size;
  int first=mesh->gids_eachrank[0];
  flag.sync<HostMemSpace>();
  for (int m=0;m<pack->nmb_thispack;++m) {
    auto z=sizes.h_view(m);
    bool inner=z.x1min>=-0.5 && z.x1max<=0.5 && z.x2min>=-0.5 && z.x2max<=0.5 &&
               z.x3min>=-0.5 && z.x3max<=0.5;
    bool corners=(z.x1min>=0.5 || z.x1max<=-0.5) &&
                 (z.x2min>=0.5 || z.x2max<=-0.5) &&
                 (z.x3min>=0.5 || z.x3max<=-0.5);
    bool region=refine_corners ? corners : inner;
    int level=mesh->lloc_eachmb[first+m].level-mesh->root_level;
    flag.h_view(first+m)=region && mesh->ncycle<3 && level<1 ? 1 :
        (mesh->ncycle>=3 && level>0 ? -1 : 0);
  }
  flag.modify<HostMemSpace>(); flag.sync<DevExeSpace>();
}
}
void ProblemGenerator::Z4cCubedSphere(ParameterInput *pin,const bool restart) {
  if (restart) throw std::invalid_argument("cubed sphere test restart unsupported");
  pgen_final_func=Finish; user_ref_func=Refine;
  refine_corners=pin->GetOrAddBoolean("problem","amr_refine_corners",false);
  auto *pack=pmy_mesh_->pmb_pack;
  if (!pack->pz4c) throw std::invalid_argument("cubed sphere test requires Z4c");
  auto ind=pack->pmesh->mb_indcs; auto sizes=pack->pmb->mb_size;
  auto geometry=pack->pcoord->z4c_geometry; auto u=pack->pz4c->u0;
  std::string data=pin->GetOrAddString("problem","initial_data","minkowski");
  int kind=data=="gauge_wave" ? 1 : (data=="angular_pulse" ? 2 : 0);
  Real amp=pin->GetOrAddReal("problem","amplitude",0.001);
  Real wn=pin->GetOrAddReal("problem","wave_number",1.0);
  Real width=pin->GetOrAddReal("problem","pulse_width",0.35);
  if (!std::isfinite(amp) || fabs(amp)>=1 || !(width>0) || !std::isfinite(width) ||
      !std::isfinite(wn)) throw std::invalid_argument("invalid cubed sphere test parameters");
  if (kind==1 && (pack->pz4c->opt.lapse_oplog!=0 || pack->pz4c->opt.lapse_harmonic!=1 ||
      pack->pz4c->opt.lapse_harmonicf!=1 ||
      pack->pz4c->opt.shift_ggamma!=0 || pack->pz4c->opt.shift_alpha2ggamma!=0))
    throw std::invalid_argument("exact gauge wave requires harmonic lapse and frozen shift");
  par_for("cubed sphere test initial data",DevExeSpace(),0,pack->nmb_thispack-1,
      0,ind.nx3+2*ind.ng-1,0,ind.nx2+2*ind.ng-1,0,ind.nx1+2*ind.ng-1,
      KOKKOS_LAMBDA(int m,int k,int j,int i) {
    auto s=sizes.d_view(m);
    Real x[3]={s.x1min+(i-ind.is+0.5)*s.dx1,s.x2min+(j-ind.js+0.5)*s.dx2,
               s.x3min+(k-ind.ks+0.5)*s.dx3};
    if (geometry.extent(0)) for (int a=0;a<3;++a) x[a]=geometry(m,k,j,i).x[a];
    Real value[25]; cubed_sphere::TestState(x,0,kind,amp,wn,width,value);
    for (int n=0;n<25;++n) u(m,n,k,j,i)=value[n];
  });
  pack->pz4c->Z4cToADM(pack);
}
