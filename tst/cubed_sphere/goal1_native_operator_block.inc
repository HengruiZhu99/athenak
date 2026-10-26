// Included inside the isolated native probe, after immutable serialization.
if(operator_only) {
  if(mass!=0 || !radial_sbp)throw std::invalid_argument("operator functional pilot requires Minkowski radial SBP");
  std::ofstream result("cubed_sphere_operator.csv");result<<std::setprecision(17);
  result<<"profile,region,subset,component,second_tested,cells,volume,value_l2,value_linf,first_l2,first_linf,second_l2,second_linf,mixed_l2,mixed_linf,omitted_Hessian_l2,omitted_Hessian_linf\n";
  for(int kind=0;kind<4;++kind) {
    for(auto &p:packs) {
      auto state=p.pack->pz4c->u0;auto cells=p.device_cells;
      Kokkos::deep_copy(state,p.baseline);
      Kokkos::parallel_for("native operator analytic active data",cells.extent_int(0),KOKKOS_LAMBDA(int index){
        const auto &c=cells(index);auto f=operator_oracle::Function(c.geometry.x,kind);
        for(int n=0;n<22;++n)state(c.m,n,c.k,c.j,c.i)=hyp::ReferenceComponent(n,c.reference)+operator_oracle::Coefficient(n)*f.value;
      });
    }
    fill(); // Actual active-only donor atlas, including mixed ghost corners.
    for(int region=0;region<2;++region) {
      auto &p=packs[region];auto dev=p.dev;auto fields=hyp::BindCartesianFields(dev);auto radial=p.radial;auto cells=p.device_cells;
      DvceArray3D<Real> errors("native operator component errors",cells.extent(0),22,10);
      Kokkos::parallel_for("native operator independent Cartesian oracle",cells.extent_int(0),KOKKOS_LAMBDA(int index){
        const auto &c=cells(index);const auto u=LoadProbeJet(fields,dev,c,c.reference.state,radial);const auto oracle=operator_oracle::Function(c.geometry.x,kind);
        Real badH=0;
        if(kind==1 && region==1){auto bad=c;for(int a=0;a<3;++a)for(int i=0;i<3;++i)for(int j=0;j<3;++j)bad.geometry.H[a][i][j]=0;
          const auto wrong=LoadProbeJet(fields,dev,bad,bad.reference.state,radial);
          for(int i=0;i<3;++i)for(int j=0;j<3;++j){const Real e=wrong.chi.dd[i][j]-c.reference.state.chi.dd[i][j];badH+=e*e;}}
        for(int n=0;n<22;++n){const Real coefficient=operator_oracle::Coefficient(n);auto value=operator_oracle::Component(u,n),base=operator_oracle::Component(c.reference.state,n);
          const Real e=(value.value-base.value)/coefficient-oracle.value;Real grad=0,hess=0,mixed=0;
          for(int i=0;i<3;++i){Real d=(value.d[i]-base.d[i])/coefficient-oracle.d[i];grad+=d*d;
            if(n<8 || n>13)for(int j=0;j<3;++j){Real dd=(value.dd[i][j]-base.dd[i][j])/coefficient-oracle.dd[i][j];hess+=dd*dd;if(i!=j)mixed+=dd*dd;}}
          const Real row[10]={e*e,fabs(e),grad,sqrt(grad),hess,sqrt(hess),mixed,sqrt(mixed),n==0?badH:0,n==0?sqrt(badH):0};for(int a=0;a<10;++a)errors(index,n,a)=row[a];
        }
      });
      auto data=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),errors);
      struct Stats {int cells=0;Real volume=0,sq[5]{},linf[5]{};};Stats stats[7][22];
      const char* names[7]={"all","interface","transition","outer_collar","face","edge","corner"};
      const int ng=p.pack->pmesh->mb_indcs.ng,na=p.dev.extent_int(3)-2*ng;
      for(size_t index=0;index<p.Cells().size();++index){const auto &c=p.Cells()[index];const Real r=c.reference.radius;
        const Real extent=std::max({fabs(c.geometry.x[0]),fabs(c.geometry.x[1]),fabs(c.geometry.x[2])});
        int seam=0;if(region){const Real u=-1+2*(c.j-ng+.5)/na,v=-1+2*(c.k-ng+.5)/na;seam=(fabs(u)>.8)+(fabs(v)>.8);}
        const bool mask[7]={true,region?r<.5:extent>=.18,r>=reference.layer.r0&&r<reference.layer.r1,region&&r>.9,region&&seam==0,region&&seam==1,region&&seam==2};
        for(int b=0;b<7;++b)if(mask[b])for(int n=0;n<22;++n){auto &s=stats[b][n];++s.cells;s.volume+=c.volume;for(int a=0;a<5;++a){s.sq[a]+=c.volume*data(index,n,2*a);s.linf[a]=std::max(s.linf[a],data(index,n,2*a+1));}}
      }
      for(int b=0;b<7;++b)for(int n=0;n<22;++n){const auto &s=stats[b][n];if(!s.cells)continue;
        result<<kind<<","<<(region?"shell":"bulk")<<","<<names[b]<<","<<n<<","<<((n<8||n>13)?1:0)<<","<<s.cells<<","<<s.volume;
        for(int a=0;a<5;++a)result<<","<<sqrt(s.sq[a]/s.volume)<<","<<s.linf[a];result<<"\n";
      }
    }
  }
  // Structural interface gate: perturb only bulk active fields and use the
  // exact atlas. An incoming interface cannot be repaired by smooth-field
  // consistency tests when the actual shell operator has no bulk dependency.
  for(int region=0;region<2;++region){auto &p=packs[region];auto state=p.pack->pz4c->u0;auto cells=p.device_cells;
    Kokkos::deep_copy(state,p.baseline);
    Kokkos::parallel_for("native one-sided interface perturbation",cells.extent_int(0),KOKKOS_LAMBDA(int index){const auto &c=cells(index);auto f=operator_oracle::Function(c.geometry.x,3);
      for(int n=0;n<22;++n)state(c.m,n,c.k,c.j,c.i)=hyp::ReferenceComponent(n,c.reference)+(region==0?operator_oracle::Coefficient(n)*f.value:0);});
  }
  fill();Real value_response=0,first_response=0,second_response=0;
  {auto &p=packs[1];auto dev=p.dev;auto fields=hyp::BindCartesianFields(dev);auto cells=p.device_cells;auto radial=p.radial;
    Kokkos::parallel_reduce("native shell dependency on bulk",cells.extent_int(0),KOKKOS_LAMBDA(int index,Real &value,Real &first,Real &second){const auto &c=cells(index);auto u=LoadProbeJet(fields,dev,c,c.reference.state,radial);
      for(int n=0;n<22;++n){auto s=operator_oracle::Component(u,n),b=operator_oracle::Component(c.reference.state,n);value=fmax(value,fabs(s.value-b.value));
        for(int i=0;i<3;++i){first=fmax(first,fabs(s.d[i]-b.d[i]));if(n<8||n>13)for(int j=0;j<3;++j)second=fmax(second,fabs(s.dd[i][j]-b.dd[i][j]));}}
    },Kokkos::Max<Real>(value_response),Kokkos::Max<Real>(first_response),Kokkos::Max<Real>(second_response));
  }
  auto shell_rows=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),probe_shell_plan);
  auto bulk_rows=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),bp);
  int bulk_to_shell=0,bulk_to_shell_active_radial=0,shell_to_bulk=0;
  for(size_t i=0;i<shell_rows.extent(0);++i){const auto &r=shell_rows(i);if(!r.source_shell){++bulk_to_shell;if(r.ti>=packs[1].radial.first&&r.ti<=packs[1].radial.last)++bulk_to_shell_active_radial;}}
  for(size_t i=0;i<bulk_rows.extent(0);++i)if(bulk_rows(i).source_shell)++shell_to_bulk;
  std::ofstream dependency("cubed_sphere_operator_dependency.json");dependency<<std::setprecision(17)
    <<"{\"bulk_sourced_shell_rows\":"<<bulk_to_shell<<",\"bulk_sourced_shell_rows_active_radial\":"<<bulk_to_shell_active_radial
    <<",\"shell_sourced_bulk_rows\":"<<shell_to_bulk<<",\"shell_value_response_to_bulk_only\":"<<value_response
    <<",\"shell_first_response_to_bulk_only\":"<<first_response<<",\"shell_second_response_to_bulk_only\":"<<second_response<<"}\n";
  dependency.flush();if(!dependency)throw std::runtime_error("native interface dependency output failed");
  result.flush();if(!result)throw std::runtime_error("native operator functional output failed");
  serialize("cubed_sphere_reference_after.txt");
  for(auto &p:packs)Kokkos::deep_copy(p.pack->pz4c->u0,p.original);
  mesh->time=0;mesh->ncycle=0;mesh->dt=0;scalar_probe_active_=false;mesh_sequence_=-1;Fill(0);
  std::cout<<"Native donor operator diagnostic complete; no evolution"<<std::endl;return;
}
