// Full pointwise geometric RHS check: original free vs stationary outer gauge.
// No RK evolution, gauge override in production or stability proof.
int main(int argc,char**argv){
 MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);
 MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);Kokkos::initialize(argc,argv);int code=0;
 try{
  if(argc!=3)throw std::invalid_argument("usage: initial_shear config fresh_csv");
  auto&cfg=hyperbolic_reproduction::refined_configuration;cfg.Load(argv[1],hyperbolic_reproduction::ContinuousCoreMesh(16));
  if(cfg.boost_velocity!=0)throw std::runtime_error("radial initial attribution requires stationary geometry");
  h::NativeWormhole base(1./3.,2e-13,cfg.initial_start_M,cfg.initial_end_M,h::NativeWormhole::TransitionProfile::polynomial_c6);
  if(std::filesystem::exists(argv[2]))throw std::runtime_error("refusing overwrite");
  std::ofstream out(argv[2]);out<<std::setprecision(17)<<"mode,h,r,R_over_M,alpha,chi,Ar,beta_r,beta_dev_r,metric_rr_rate,metric_rr_algebraic,shift_rate,lapse_rate,restoring_source,chi_rate,max_geometric_rate,K_rate,Theta_rate,max_A_rate,max_Lambda_rate\n";
  for(int mode=0;mode<2;++mode)for(Real eps:{.0008,.0004,.0002,.0001,.00005})for(int ir=0;ir<=11;++ir){
   const Real r=.4+ir*.05;if(r>=.999)continue;const Real x[3]={r,0,0};
   auto fields=[&](const Real*y){auto a=base.Fields(y);const auto native=a;cfg.InitialFields(y,a);
    if(mode==1){const Real rr=std::sqrt(y[0]*y[0]+y[1]*y[1]+y[2]*y[2]);const auto bg=base.Point(rr);
     const Real old_omega=(1-rr*rr)/2,S=-bg.trace/3+old_omega*bg.ar/2;
     a[18]=cfg.gauge.profile.Radial(rr).value*std::sqrt(1-2/(3*bg.areal)+bg.areal*bg.areal*S*S)-(1+rr*rr)/2;
     for(int j=0;j<3;++j)a[22+j]=(1-S)*y[j];
    }else if(mode){const Real rr=std::sqrt(y[0]*y[0]+y[1]*y[1]+y[2]*y[2]);
     const Real W=mode==2?1:h::Smoothstep9((rr-cfg.gauge.inner_begin)/(cfg.gauge.inner_end-cfg.gauge.inner_begin));
     for(int j=0;j<3;++j)a[22+j]=(1-W)*y[j]+W*native[22+j];}
    return a;};
   const auto f=fields(x);V v(f.begin(),f.end());std::vector<std::array<Real,3>>d(31);std::vector<std::array<std::array<Real,3>,3>>dd(31);
   for(int axis=0;axis<3;++axis){std::array<double,31>a[4];const int offsets[4]={-2,-1,1,2};
    for(int k=0;k<4;++k){Real y[3]={r,0,0};y[axis]+=eps*offsets[k];a[k]=fields(y);}
    for(int j=0;j<31;++j)d[j][axis]=(a[0][j]-8*a[1][j]+8*a[2][j]-a[3][j])/(12*eps);
   }
   for(int axis=0;axis<3;++axis){
    std::array<double,31> a[4];const int off[4]={-2,-1,1,2};
    for(int k=0;k<4;++k){Real y[3]={r,0,0};y[axis]+=eps*off[k];a[k]=fields(y);}
    for(int j=0;j<31;++j)dd[j][axis][axis]=(-a[0][j]+16*a[1][j]-30*f[j]+16*a[2][j]-a[3][j])/(12*eps*eps);
    for(int b=axis+1;b<3;++b){const Real weights[4]={1,-8,8,-1};
     for(int k=0;k<4;++k)for(int l=0;l<4;++l){Real y[3]={r,0,0};y[axis]+=eps*off[k];y[b]+=eps*off[l];auto z=fields(y);
      for(int j=0;j<31;++j)dd[j][axis][b]+=weights[k]*weights[l]*z[j]/(144*eps*eps);}
     for(int j=0;j<31;++j)dd[j][b][axis]=dd[j][axis][b];
    }
   }
   const auto u=h::HyperbolicComposite::Jet(x,v,d,dd);auto omega=cfg.gauge.profile.At(x);
   h::SetStationaryOmegaNormal(u.alpha.value,u.beta.value,u.alpha.d,u.beta.d,omega);
   auto parts=h::ConformalRHS(u,omega,Real(4),Real(-.99));h::Z4cRHS<Real> rate{};
   if(!h::HyperbolicDamping(x,u,omega,parts,cfg.gauge)||!h::AssembleInterior(parts,omega.omega,rate))throw std::runtime_error("initial metric RHS rejected");
   V native_rate(31);h::HyperbolicComposite::StoreGeometry(rate,native_rate.data());
   const auto B=h::HyperbolicComposite::Vector(19,v,d),aux=h::HyperbolicComposite::Vector(25,v,d);
   const auto gauge=h::HyperbolicGauge(x,u,B,aux.value,cfg.gauge);
   const auto c=h::ShiftCoefficients(r,u.alpha.value,u.chi.value,cfg.gauge);
   const Real div=u.beta.d[0][0]+u.beta.d[1][1]+u.beta.d[2][2];
   const Real algebraic=-2*u.alpha.value*v[7]+2*u.beta.d[0][0]-2*div/3;
   const Real restoring=-(1-c.inner)*cfg.gauge.restoring*c.collar*v[22]/omega.omega;
   out<<mode<<','<<eps<<','<<r<<','<<base.Point(r).areal*3<<','<<u.alpha.value<<','<<u.chi.value<<','<<v[7]<<','<<u.beta.value[0]<<','<<v[22]<<','<<native_rate[1]<<','<<algebraic<<','<<gauge.shift_regular[0]+gauge.shift_pole[0]/omega.omega<<','<<gauge.lapse_regular+gauge.lapse_pole/omega.omega<<','<<restoring<<','<<native_rate[0]; Real all=0,ar=0,lr=0;for(int j=0;j<18;++j)all=std::max(all,std::abs(native_rate[j]));for(int j=7;j<13;++j)ar=std::max(ar,std::abs(native_rate[j]));for(int j=15;j<18;++j)lr=std::max(lr,std::abs(native_rate[j]));out<<','<<all<<','<<native_rate[13]<<','<<native_rate[14]<<','<<ar<<','<<lr<<'\n';
  }
  out.close();if(!out)throw std::runtime_error("initial audit write failed");
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';code=2;}
 Kokkos::finalize();MPI_Finalize();return code;
}
