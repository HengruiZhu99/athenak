// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Independent manufactured tests of the native sparse surface remesh operator.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "globals.hpp"
#include "mesh/mesh.hpp"
#include "z4c_hyperboloidal/cartesian_algebraic_chart.hpp"
#include "z4c_hyperboloidal/mapped_halo.hpp"
#include "z4c_hyperboloidal/surface_remesh.hpp"

namespace h=z4c_hyperboloidal;
bool diagnostic_mode=false, reviewed_admission=false;
bool analytic_gates_ok=true, revised_gates_ok=true;
void Require(bool yes,const char *what) {if (!yes)throw std::runtime_error(what);}

bool Ancestor(const LogicalLocation &a,const LogicalLocation &b) {
  if (a.chart_id!=b.chart_id || a.level>b.level)return false;
  const int shift=b.level-a.level;
  return a.lx1==(b.lx1>>shift) && a.lx2==(b.lx2>>shift) && a.lx3==(b.lx3>>shift);
}

h::RemeshLayout Layout(Mesh &old_mesh,Mesh &new_mesh) {
  auto &old=*old_mesh.pmb_pack->pz4c_hyperboloidal;
  auto &next=*new_mesh.pmb_pack->pz4c_hyperboloidal;
  h::RemeshLayout l;
  l.cells=old.n[0];l.origin=old.origin;
  l.old_first=old_mesh.pmb_pack->gids;l.old_count=old.nblocks;
  l.new_first=new_mesh.pmb_pack->gids;l.new_count=next.nblocks;
  l.mesh_sequence=old_mesh.GetAMRLoadBalanceUpdateSeq();
  h::NativeDonorSelector old_maps(old_mesh,old.origin),new_maps(new_mesh,next.origin);
  for (int i=0;i<old_mesh.nmb_total;++i) {
    l.old_maps.push_back(old_maps.Map(i));l.old_owner.push_back(old_mesh.rank_eachmb[i]);
    l.old_level.push_back(old_mesh.lloc_eachmb[i].level);
    int target=-1;
    for (int j=0;j<new_mesh.nmb_total;++j)
      if (Ancestor(old_mesh.lloc_eachmb[i],new_mesh.lloc_eachmb[j]) ||
          Ancestor(new_mesh.lloc_eachmb[j],old_mesh.lloc_eachmb[i])) {target=j;break;}
    Require(target>=0,"old leaf has no related new leaf");l.old_to_new.push_back(target);
  }
  for (int j=0;j<new_mesh.nmb_total;++j) {
    l.new_maps.push_back(new_maps.Map(j));l.new_owner.push_back(new_mesh.rank_eachmb[j]);
    l.new_level.push_back(new_mesh.lloc_eachmb[j].level);
    int source=-1;
    for (int i=0;i<old_mesh.nmb_total;++i)
      if (Ancestor(old_mesh.lloc_eachmb[i],new_mesh.lloc_eachmb[j]) ||
          Ancestor(new_mesh.lloc_eachmb[j],old_mesh.lloc_eachmb[i])) {source=i;break;}
    Require(source>=0,"new leaf has no related old leaf");l.new_to_old.push_back(source);
    l.changed.push_back(l.old_level[source]!=l.new_level[j]);
  }
  return l;
}

std::array<Real,2> Angles(const chart_atlas::BlockMap &map,int n,int j,int k) {
  return {map.lo[1]+(j+.5)*(map.hi[1]-map.lo[1])/n,
          map.lo[2]+(k+.5)*(map.hi[2]-map.lo[2])/n};
}
Real Polynomial(Real u,Real v,int a,int b) {return std::pow(u,a)*std::pow(v,b);}
Real Smooth(Real u,Real v) {return std::sin(5.1*u)*std::cos(4.7*v);}

bool AnalyticGates(h::RemeshLayout l,const char *label) {
  Real worst=0;std::uint64_t checked=0;
  for (const auto &row:h::BuildSurfaceTransferRows(l)) {
    if (row.copy)continue;
    const auto target=Angles(l.new_maps[row.new_gid],l.cells,row.j,row.k);
    Real norm=0;for (const auto &term:row.fields)norm+=std::abs(term.weight);
    for (int a=0;a<=5;++a)for (int b=0;b<=5;++b) {
      Real got=0;
      for (const auto &term:row.fields) {
        const auto q=Angles(l.old_maps[term.donor.gid],l.cells,term.donor.j,term.donor.k);
        got+=term.weight*Polynomial(q[0],q[1],a,b);
      }
      const Real error=std::abs(got-Polynomial(target[0],target[1],a,b));
      const Real bound=2048*std::numeric_limits<Real>::epsilon()*std::max(Real(1),norm);
      worst=std::max(worst,error/bound);++checked;
    }
  }
  MPI_Allreduce(MPI_IN_PLACE,&worst,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&checked,1,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
  Require(checked>0 && worst<=1,"surface tensor-polynomial roundoff gate failed");
  Real errors[5]{},bounds[5]{},ratios[5]{};
  for (int resolution=0;resolution<5;++resolution) {
    l.cells=12*(1<<resolution);
    for (const auto &row:h::BuildSurfaceTransferRows(l)) {
      if (row.copy)continue;
      Real got=0;
      for (const auto &term:row.fields) {
        const auto q=Angles(l.old_maps[term.donor.gid],l.cells,term.donor.j,term.donor.k);
        got+=term.weight*Smooth(q[0],q[1]);
      }
      const auto q=Angles(l.new_maps[row.new_gid],l.cells,row.j,row.k);
      const Real error=std::abs(got-Smooth(q[0],q[1]));
      Real nodes[2][6],product[2]={1,1},norm=0,Lu=0;
      for (int i=0;i<6;++i) {
        const auto u=row.fields[i].donor,v=row.fields[6*i].donor;
        nodes[0][i]=Angles(l.old_maps[u.gid],l.cells,u.j,u.k)[0];
        nodes[1][i]=Angles(l.old_maps[v.gid],l.cells,v.j,v.k)[1];
        for (int a=0;a<2;++a)product[a]*=std::abs(q[a]-nodes[a][i]);
      }
      for (int i=0;i<6;++i) {
        Real weight=1;
        for (int j=0;j<6;++j)if (j!=i)
          weight*=(q[0]-nodes[0][j])/(nodes[0][i]-nodes[0][j]);
        Lu+=std::abs(weight);
      }
      for (const auto &term:row.fields)norm+=std::abs(term.weight);
      // Tensor remainder: (S-PuS)C + (PuS)(C-PvC). Both analytic
      // factors have amplitude <=1; their sixth derivatives are bounded by
      // omega^6. This bound is independent of any measured approximation error.
      const Real bound=(std::pow(5.1,6)*product[0]+Lu*std::pow(4.7,6)*product[1])/720;
      const Real roundoff=2048*std::numeric_limits<Real>::epsilon()*std::max(Real(1),norm);
      errors[resolution]=std::max(errors[resolution],error);
      bounds[resolution]=std::max(bounds[resolution],bound);
      ratios[resolution]=std::max(ratios[resolution],error/(bound+roundoff));
    }
  }
  MPI_Allreduce(MPI_IN_PLACE,errors,5,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,bounds,5,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,ratios,5,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
  Real orders[4];for (int i=0;i<4;++i)orders[i]=std::log2(errors[i]/errors[i+1]);
  if (!global_variable::my_rank) {
    std::cout<<std::setprecision(17)<<label<<" polynomial_checks="<<checked
        <<" max_roundoff_ratio="<<worst<<" smooth_errors=";
    for (int i=0;i<5;++i)std::cout<<(i ? "," : "")<<errors[i];
    std::cout<<" orders=";for (int i=0;i<4;++i)std::cout<<(i ? "," : "")<<orders[i];
    std::cout<<" remainder_bounds=";
    for (int i=0;i<5;++i)std::cout<<(i ? "," : "")<<bounds[i];
    std::cout<<" remainder_ratios=";
    for (int i=0;i<5;++i)std::cout<<(i ? "," : "")<<ratios[i];
    std::cout<<"\n";
  }
  for (int i=0;i<5;++i)Require(ratios[i]<=1,"surface finite-N Lagrange remainder bound failed");
  revised_gates_ok=(orders[2]>=5.5 && orders[3]>=5.5) && revised_gates_ok;
  const bool original_gate=orders[0]>=5.5 && orders[1]>=5.5;
  if (!diagnostic_mode)Require(original_gate,"surface sixth-order smooth convergence gate failed");
  return original_gate;
}

std::array<Real,31> State(const chart_atlas::BlockMap &map,int n,int j,int k) {
  const auto q=Angles(map,n,j,k);
  const Real scalar=2+.2*q[0]+.3*q[1]+.05*q[0]*q[0]*q[1]*q[1];
  h::CartesianChartJets jets{};
  for (int f=0;f<31;++f)jets[f].f=(f+1)*1e-5*scalar;
  jets[0].f+=1;
  Require(h::TryExpandCartesianChart(jets),"invalid manufactured surface metric");
  std::array<Real,31> out{};for (int f=0;f<31;++f)out[f]=jets[f].f;
  return out;
}
std::array<Real,8> Record(int gid,int n,int j,int k) {
  const Real key=gid*n*n+k*n+j+1;
  return {.5+key/1048576,.5+key/1048576+1./8192,
          key,2*key,4*key,key/128,key/8,key/1024};
}
void Seed(Mesh &mesh,const h::RemeshLayout &l) {
  auto &p=*mesh.pmb_pack->pz4c_hyperboloidal;
  auto state=Kokkos::create_mirror_view(p.s0);
  auto records=Kokkos::create_mirror_view(p.surface_chi_records);
  for (int point=0;point<p.nsurface_points;++point) {
    const auto c=p.surface_points.h_view(point);
    const int gid=l.old_first+p.face_blocks.h_view(c.face);
    const auto values=State(l.old_maps[gid],l.cells,c.j,c.k);
    for (int f=0;f<31;++f)state(c.face,f,c.k,c.j)=values[f];
    records(point)=h::UnpackChiRecord(Record(gid,l.cells,c.j,c.k).data());
  }
  Kokkos::deep_copy(p.s0,state);Kokkos::deep_copy(p.surface_chi_records,records);
}

std::uint64_t Transfer(Mesh &old_mesh,Mesh &new_mesh,const char *label,bool analytic) {
  const auto l=Layout(old_mesh,new_mesh);
  if (analytic)analytic_gates_ok=AnalyticGates(l,label) && analytic_gates_ok;
  Seed(old_mesh,l);
  h::SurfaceRemesh transfer;transfer.Prepare(*old_mesh.pmb_pack->pz4c_hyperboloidal,l);
  auto &p=*new_mesh.pmb_pack->pz4c_hyperboloidal;
  Kokkos::deep_copy(p.s0,std::numeric_limits<Real>::quiet_NaN());
  transfer.Install(p);
  const auto state=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.s0);
  const auto records=Kokkos::create_mirror_view_and_copy(HostMemSpace(),p.surface_chi_records);
  const auto rows=h::BuildSurfaceTransferRows(l);
  std::vector<int> coverage(l.old_maps.size()*l.cells*l.cells,0);
  std::uint64_t copies=0,changes=0,migrated=0,failures=0;
  Real max_ratio=0;
  for (std::size_t point=0;point<rows.size();++point) {
    const auto &row=rows[point];
    const int source=l.new_to_old[row.new_gid];
    const auto expected=State(row.copy ? l.old_maps[source] : l.new_maps[row.new_gid],
                              l.cells,row.j,row.k);
    Real norm=0;for (const auto &term:row.fields)norm+=std::abs(term.weight);
    for (int f=0;f<31;++f) {
      const Real actual=state(row.face,f,row.k,row.j);
      if (row.copy) {
        if (std::memcmp(&actual,&expected[f],sizeof(Real)))++failures;
        ++copies;
      } else {
        const Real value=f>=28 ? 0 : expected[f];
        const Real bound=4096*std::numeric_limits<Real>::epsilon()*std::max(Real(1),norm);
        const Real ratio=std::abs(actual-value)/bound;
        max_ratio=std::max(max_ratio,ratio);
        if (!std::isfinite(actual) || ratio>1 || (f>=28 && actual!=0))++failures;
        ++changes;
      }
    }
    if (row.copy && l.old_owner[source]!=l.new_owner[row.new_gid])++migrated;
    std::array<Real,8> expected_record{1e300,1e300,0,0,0,0,0,1e300};
    bool first=true;
    for (const auto &donor:row.history) {
      ++coverage[(donor.gid*l.cells+donor.k)*l.cells+donor.j];
      const auto packet=Record(donor.gid,l.cells,donor.j,donor.k);
      if (first) {expected_record=packet;first=false;}
      else for (int a=0;a<8;++a) {
        if (a==0 || a==1 || a==7)expected_record[a]=std::min(expected_record[a],packet[a]);
        else if (a==5)expected_record[a]=std::max(expected_record[a],packet[a]);
        else expected_record[a]+=packet[a];
      }
    }
    Real got[8];h::PackChiRecord(records(point),got);
    for (int a=0;a<8;++a)if (std::memcmp(&got[a],&expected_record[a],sizeof(Real)))++failures;
  }
  MPI_Allreduce(MPI_IN_PLACE,coverage.data(),coverage.size(),MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  for (std::size_t gid=0;gid<l.old_maps.size();++gid)
    for (int k=0;k<l.cells;++k)for (int j=0;j<l.cells;++j)
      Require(coverage[(gid*l.cells+k)*l.cells+j]==(l.old_maps[gid].outer_surface ? 1 : 0),
              "surface history packet duplicated or discarded");
  Real before[8]={1e300,1e300,0,0,0,0,0,1e300};
  Real after[8]={1e300,1e300,0,0,0,0,0,1e300};
  const auto account=[](Real *total,const Real *record) {
    for (int a=0;a<8;++a) {
      if (a==0 || a==1 || a==7)total[a]=std::min(total[a],record[a]);
      else if (a==5)total[a]=std::max(total[a],record[a]);
      else total[a]+=record[a];
    }
  };
  const auto &old=*old_mesh.pmb_pack->pz4c_hyperboloidal;
  for (int point=0;point<old.nsurface_points;++point) {
    const auto c=old.surface_points.h_view(point);
    const int gid=l.old_first+old.face_blocks.h_view(c.face);
    account(before,Record(gid,l.cells,c.j,c.k).data());
  }
  for (int point=0;point<p.nsurface_points;++point) {
    Real packet[8];h::PackChiRecord(records(point),packet);account(after,packet);
  }
  for (int a=0;a<8;++a) {
    const auto op=(a==0 || a==1 || a==7) ? MPI_MIN : (a==5 ? MPI_MAX : MPI_SUM);
    MPI_Allreduce(MPI_IN_PLACE,before+a,1,MPI_DOUBLE,op,MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE,after+a,1,MPI_DOUBLE,op,MPI_COMM_WORLD);
    Require(std::memcmp(before+a,after+a,sizeof(Real))==0,
            "surface global history total or extremum changed");
  }
  std::uint64_t counts[5]={copies,changes,migrated,failures,
      static_cast<std::uint64_t>(transfer.RemoteDonors())};
  MPI_Allreduce(MPI_IN_PLACE,counts,5,MPI_UINT64_T,MPI_SUM,MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE,&max_ratio,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
  int no_faces[2]={old_mesh.pmb_pack->pz4c_hyperboloidal->nfaces==0,p.nfaces==0};
  MPI_Allreduce(MPI_IN_PLACE,no_faces,2,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
  if (global_variable::nranks==8)Require(no_faces[0]>0 && no_faces[1]>0,
                                        "eight-rank test omitted zero-face ownership");
  if (!global_variable::my_rank)std::cout<<label<<" unchanged_values="<<counts[0]
      <<" changed_values="<<counts[1]<<" migrated_unchanged_points="<<counts[2]
      <<" failures="<<counts[3]<<" remote_donors="<<counts[4]
      <<" max_scaled_error="<<max_ratio<<" zero_face_ranks="<<no_faces[0]<<","<<no_faces[1]<<"\n";
  Require(counts[3]==0,"surface remesh state/history validation failed");
  return counts[2];
}

std::unique_ptr<Mesh> Load(const char *path,ParameterInput &pin) {
  IOWrapper file;file.Open(path,IOWrapper::FileMode::read);pin.LoadFromFile(file);file.Close();
  auto mesh=std::make_unique<Mesh>(&pin);mesh->BuildTreeFromScratch(&pin);
  mesh->AddCoordinatesAndPhysics(&pin);
  return mesh;
}
int main(int argc,char **argv) {
  MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
  MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);Kokkos::initialize(argc,argv);
  try {
    {
      Require(argc==4 || argc==5,"surface_remesh expects base, outer-refined and migration fixtures");
      reviewed_admission=argc==5 && std::string(argv[4])=="--reviewed-admission";
      diagnostic_mode=reviewed_admission || (argc==5 && std::string(argv[4])=="--diagnostic");
      Require(argc==4 || diagnostic_mode,"unknown surface remesh option");
      ParameterInput a,b,c;
      auto base=Load(argv[1],a),fine=Load(argv[2],b);
      Transfer(*base,*fine,"refine",true);
      Transfer(*fine,*base,"coarsen",true);
      fine.reset();
      auto migration=Load(argv[3],c);
      const auto moved=Transfer(*base,*migration,"migration",false);
      Require(moved>0,"fixture did not migrate any unchanged surface to another rank");
      if (reviewed_admission) {
        Require(revised_gates_ok,"reviewed N48/96/192 asymptotic-rate gate failed");
        if (!global_variable::my_rank)std::cout
            <<"PASS reviewed native sparse surface AMR transfer; original_gate_passed="
            <<(analytic_gates_ok ? "true" : "false")<<"\n";
      } else {
        if (!analytic_gates_ok && !global_variable::my_rank)
          std::cout<<"DIAGNOSTIC transport passed; original smooth-rate gate failed\n";
        Require(analytic_gates_ok,"original N12/24/48 smooth-rate gate remains failed");
        if (!global_variable::my_rank)std::cout<<"PASS native sparse surface AMR transfer\n";
      }
    }
  } catch (const std::exception &error) {
    std::cerr<<"rank "<<global_variable::my_rank<<": "<<error.what()<<"\n";
    MPI_Abort(MPI_COMM_WORLD,1);
  }
  Kokkos::finalize();MPI_Finalize();
}
