// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <chrono>
#include <iomanip>
#include <iostream>
#include "z4c/hyperboloidal/puncture_refinement.hpp"
namespace h=z4c::hyperboloidal;
using Mesh=cubed_sphere::CompositeMesh;using Block=cubed_sphere::CompositeBlock;
bool SameBlock(const Block&a,const Block&b){
 if(a.face!=b.face||a.layer!=b.layer||a.refinement!=b.refinement)return false;
 for(int d=0;d<3;++d)if(a.lo[d]!=b.lo[d]||a.hi[d]!=b.hi[d])return false;return true;
}
bool SameMesh(const Mesh&a,const Mesh&b){
 if(a.half_width!=b.half_width||a.radii!=b.radii||a.face_subdivisions!=b.face_subdivisions
    ||a.cells!=b.cells||a.ghosts!=b.ghosts||a.revision!=b.revision||a.blocks.size()!=b.blocks.size())return false;
 for(size_t i=0;i<a.blocks.size();++i)if(!SameBlock(a.blocks[i],b.blocks[i]))return false;return true;
}
int Contact(const Block&a,const Block&b){
 if(a.face>=0||b.face>=0)return 0;int boundary=0;
 for(int d=0;d<3;++d){if(a.hi[d]<b.lo[d]||b.hi[d]<a.lo[d])return 0;
   boundary+=a.hi[d]==b.lo[d]||b.hi[d]==a.lo[d];}return boundary;
}
struct Checks{Real map_error=0,partition_error=0,constant_integral_error=0;size_t contacts[3]{};};
Checks Verify(const Mesh&before,const h::PunctureRefinementResult&r,const h::PuncturePosition &p,int target){
 Checks out;const auto &after=r.mesh;
 if(after.cells!=before.cells||after.ghosts!=before.ghosts||after.radii!=before.radii
    ||after.half_width!=before.half_width||after.face_subdivisions!=before.face_subdivisions)
   throw std::runtime_error("global map/shape contract changed");
 if(!p.Matches(r.position_stamp))throw std::runtime_error("puncture metadata changed");
 if(r.old_revision!=before.revision||r.new_revision!=after.revision
    ||r.new_revision-r.old_revision!=r.targeted_refinements+r.balance_refinements)
   throw std::runtime_error("revision accounting");
 std::vector<Block> old_shells,new_shells;
 long double old_cartesian_volume=0,new_cartesian_volume=0;
 for(const auto &b:before.blocks)if(b.face>=0)old_shells.push_back(b);else{
   long double volume=1;for(int a=0;a<3;++a)volume*=b.hi[a]-b.lo[a];old_cartesian_volume+=volume;}
 size_t containing=0;
 for(size_t id=0;id<after.blocks.size();++id){const auto &b=after.blocks[id];
   if(b.face>=0)new_shells.push_back(b);else{
     long double volume=1;for(int a=0;a<3;++a)volume*=b.hi[a]-b.lo[a];new_cartesian_volume+=volume;
     bool inside=true;for(int a=0;a<3;++a)inside=inside&&p.position[a]>=b.lo[a]&&p.position[a]<=b.hi[a];
     if(inside){++containing;if(b.refinement<target)throw std::runtime_error("target not met at tie leaf");}}
   // Cell centers on all returned blocks remain the same physical Cartesian
   // points when represented through the original descriptor's inverse map.
   for(int k=0;k<after.cells;++k)for(int j=0;j<after.cells;++j)for(int i=0;i<after.cells;++i){
     const Real q[3]={(i+.5)/after.cells,(j+.5)/after.cells,(k+.5)/after.cells};Real x[3],oldq[3],back[3];
     after.Forward(static_cast<int>(id),q,x);const int oldowner=before.Owner(x);
     before.Inverse(oldowner,x,oldq);before.Forward(oldowner,oldq,back);
     for(int a=0;a<3;++a)out.map_error=std::max(out.map_error,std::abs(x[a]-back[a]));
   }
 }
 if(old_shells.size()!=new_shells.size()||r.shell_leaves!=old_shells.size())throw std::runtime_error("shell count changed");
 for(size_t s=0;s<old_shells.size();++s)if(!SameBlock(old_shells[s],new_shells[s]))throw std::runtime_error("shell map changed");
 if(containing!=r.containing_leaves)throw std::runtime_error("containing tie count");
 // Integral of the constant field on the Cartesian partition, no transfer or
 // interpolation implied. The Cartesian volume is exactly independent of AMR.
 out.constant_integral_error=std::abs(static_cast<Real>(old_cartesian_volume-new_cartesian_volume));
 out.partition_error=std::abs(static_cast<Real>(new_cartesian_volume)-std::pow(2*after.half_width,3));
 for(size_t a=0;a<after.blocks.size();++a)for(size_t b=a+1;b<after.blocks.size();++b){
   const int contact=Contact(after.blocks[a],after.blocks[b]);if(!contact)continue;
   ++out.contacts[contact-1];if(std::abs(after.blocks[a].refinement-after.blocks[b].refinement)>1)
     throw std::runtime_error("Cartesian face/edge/corner balance violation");}
 Real q[3],back[3];const int owner=after.Owner(p.position.data());after.Inverse(owner,p.position.data(),q);after.Forward(owner,q,back);
 for(int a=0;a<3;++a)if(std::abs(p.position[a]-back[a])>1.e-14)throw std::runtime_error("puncture physical coordinates changed");
 if(out.map_error>3.e-14||out.partition_error>1.e-14||out.constant_integral_error>1.e-14)
   throw std::runtime_error("geometry/constant partition changed");return out;
}
int main(int argc,char **argv){Kokkos::initialize(argc,argv);int status=0;
try{const auto start=std::chrono::steady_clock::now();
 const Mesh original(.25,{.45,.55,.60,.65,.70,.75,1},1,6,2,2);const Mesh snapshot=original;
 h::PuncturePosition p;p.velocity={-.1,.2,.03};p.time=.2;p.stage=2;p.revision=7;const auto stamp=p.Capture();
 const size_t expected[]={10800,22896,34992,47088};std::cout<<std::setprecision(17)<<"{\"fixed_origin\":[";
 for(int level=0;level<=3;++level){const auto r=h::RefinePunctureNeighborhood(original,p,level);const auto c=Verify(original,r,p,level);
   if(r.active_cells!=expected[level]||r.containing_leaves!=8||r.targeted_refinements!=static_cast<size_t>(8*level)
      ||r.balance_refinements!=0||std::abs(r.minimum_spacing-1./(24*(1<<level)))>1.e-15)
     throw std::runtime_error("fixed origin count/spacing mismatch");
   const auto repeat=h::RefinePunctureNeighborhood(original,p,level);if(!SameMesh(r.mesh,repeat.mesh))throw std::runtime_error("tie result nondeterministic");
   const auto noop=h::RefinePunctureNeighborhood(r.mesh,p,level);if(!SameMesh(r.mesh,noop.mesh))throw std::runtime_error("balanced result not idempotent");
   std::cout<<(level?",":"")<<"{\"level\":"<<level<<",\"active_cells\":"<<r.active_cells<<",\"cartesian_leaves\":"<<r.cartesian_leaves
     <<",\"shell_leaves\":"<<r.shell_leaves<<",\"targeted_refinements\":"<<r.targeted_refinements<<",\"balance_refinements\":"<<r.balance_refinements
     <<",\"containing_leaves\":"<<r.containing_leaves<<",\"minimum_spacing\":"<<r.minimum_spacing<<",\"revision\":"<<r.new_revision
     <<",\"map_error\":"<<c.map_error<<",\"constant_integral_error\":"<<c.constant_integral_error
     <<",\"face_edge_corner_contacts\":["<<c.contacts[0]<<','<<c.contacts[1]<<','<<c.contacts[2]<<"]}";
 }
 p.position={.037,.013,-.021};const auto translated=h::RefinePunctureNeighborhood(original,p,3);const auto tc=Verify(original,translated,p,3);
 if(!translated.balance_refinements)throw std::runtime_error("translated test did not exercise balancing");
 const auto tr=h::RefinePunctureNeighborhood(original,p,3);if(!SameMesh(translated.mesh,tr.mesh))throw std::runtime_error("translated balance nondeterministic");
 if(!SameMesh(original,snapshot))throw std::runtime_error("successful operation changed original");
 int rejections=0;auto reject=[&](auto action){bool rejected=false;try{action();}catch(const std::exception&){rejected=true;}
   if(!rejected)throw std::runtime_error("invalid refinement accepted");++rejections;
   if(!SameMesh(original,snapshot))throw std::runtime_error("failed operation changed original");};
 // Closed-bound ties are exact; a representable offset is not snapped back.
 p.position={0,.013,.021};const auto face_tie=h::RefinePunctureNeighborhood(original,p,1);
 p.position[0]=std::nextafter(Real(0),Real(1));const auto off_tie=h::RefinePunctureNeighborhood(original,p,1);
 if(face_tie.containing_leaves!=2||off_tie.containing_leaves!=1
    ||face_tie.targeted_refinements!=2||off_tie.targeted_refinements!=1)
   throw std::runtime_error("exact tie/nextafter handling changed coordinates");
 p.position={0,0,0};reject([&]{h::RefinePunctureNeighborhood(original,p,-1);});reject([&]{h::RefinePunctureNeighborhood(original,p,31);});
 reject([&]{h::RefinePunctureNeighborhood(original,p,3,47087);});reject([&]{h::RefinePunctureNeighborhood(original,p,0,10799);});
 reject([&]{h::RefinePunctureNeighborhood(original,p,3,0);});
 p.position={.4,0,0};reject([&]{h::RefinePunctureNeighborhood(original,p,1);});
 p.position={2,0,0};reject([&]{h::RefinePunctureNeighborhood(original,p,1);});
 p.position={std::numeric_limits<Real>::quiet_NaN(),0,0};reject([&]{h::RefinePunctureNeighborhood(original,p,1);});
 p.position={0,0,0};if(!p.Matches(stamp))throw std::runtime_error("input point metadata changed");
 const auto exact_budget=h::RefinePunctureNeighborhood(original,p,3,47088);if(exact_budget.active_cells!=47088)throw std::runtime_error("exact budget rejected");
 std::cout<<"],\"translated\":{\"active_cells\":"<<translated.active_cells<<",\"targeted_refinements\":"<<translated.targeted_refinements
   <<",\"balance_refinements\":"<<translated.balance_refinements<<",\"containing_leaves\":"<<translated.containing_leaves
   <<",\"minimum_spacing\":"<<translated.minimum_spacing<<",\"map_error\":"<<tc.map_error
   <<",\"face_edge_corner_contacts\":["<<tc.contacts[0]<<','<<tc.contacts[1]<<','<<tc.contacts[2]<<"]},\"invalid_input_rejections\":"<<rejections
   <<",\"face_tie_leaves\":"<<face_tie.containing_leaves<<",\"nextafter_offset_leaves\":"<<off_tie.containing_leaves
   <<",\"original_descriptor_unchanged\":true,\"deterministic\":true,\"idempotent\":true,\"exact_budget_accepted\":true,\"elapsed_seconds\":"
   <<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<",\"qualification\":\"geometry only; no field transfer or evolution claim\",\"implementation_checks_passed\":true}\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';status=1;}
Kokkos::finalize();return status;}
