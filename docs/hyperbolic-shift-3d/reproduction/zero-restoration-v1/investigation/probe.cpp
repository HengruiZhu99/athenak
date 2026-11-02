// Read-only postmortem. Reads unadmitted arrays explicitly for analysis, never
// calls LoadCheckpoint or Step and never writes a checkpoint. Original bytes
// and admission flags remain unchanged. Finite differences are RHS directions.
#define main unused_evolution_main
#include "tst/hyperbolic_shift_3d/coupled_mpi.cpp"
#undef main
int main(int argc,char**argv){
 MPI_Init(&argc,&argv);MPI_Comm_rank(MPI_COMM_WORLD,&global_variable::my_rank);MPI_Comm_size(MPI_COMM_WORLD,&global_variable::nranks);Kokkos::initialize(argc,argv);int code=0;
 try{
  if(argc!=4||global_variable::nranks!=1)throw std::runtime_error("probe checkpoint output-prefix epsilon; one SERIAL rank only");
  const Real eps=std::stod(argv[3]);std::ifstream in(argv[1],std::ios::binary);Header h0{};in.read((char*)&h0,sizeof h0);
  if(!in||std::string(h0.magic)!="HYP3D_MPI_NZ1"||h0.n!=16||h0.nf!=31||h0.cells!=159200||h0.surface!=1536*31||h0.dt!=.0005)throw std::runtime_error("unexpected analysis input");
  V v(31*h0.cells),s(h0.surface);in.seekg(168);in.read((char*)v.data(),v.size()*8);in.seekg(168+16*v.size());in.read((char*)s.data(),s.size()*8);if(!in)throw std::runtime_error("short input");
  const std::string pre=argv[2];if(std::filesystem::exists(pre+".csv"))throw std::runtime_error("output exists");
  Evolution e(16,2,2,"");e.Upload(v);Evolution::CopyFromHost(s,e.surface);Buffer edge;
  auto vr=e.Download(e.RHS(.0005,edge));auto sr=e.Download(edge);auto projected=e.Download(e.surface);
  const auto padded=e.Download(e.exchange()->padded);auto&model=e.Model();
  std::vector<V> trace(2);for(int sign=0;sign<2;++sign){const Real de=(sign?1:-1)*eps;V vv=v,ss=projected;
   for(size_t i=0;i<vv.size();++i)vv[i]+=de*vr[i];for(size_t i=0;i<ss.size();++i)ss[i]+=de*sr[i];
   e.Upload(vv);Evolution::CopyFromHost(ss,e.surface);e.RefreshBoundary(.0005);trace[sign]=e.Download(e.surface);
  }
  std::ofstream out(pre+".csv");out<<std::setprecision(17)<<"row,x,y,z,grr,chi,chi_q_minus1,grr_dot,metric_advection,metric_curvature,metric_shift,metric_remainder,v_radial,D_radial,D_trace_dt_radial,D_pde_dt_radial,D_trace_residual_norm,D_trace_dt_norm,D_pde_dt_norm,dv_dr_radial,old_restoring_limit_radial,P_plus3,Theta,Lambda_radial,delta_D_closure_radial\n";
  for(size_t p=0;p<model.scri.rows.size();++p){const auto*x=model.scri.rows[p].geometry.x;const auto sample=model.scri.Sample(p,padded,projected,31);const auto u=model.Jet(x,sample.value,sample.d,sample.dd);const auto g=h::Geometry(u.metric);Real q=0,grr=0,rr=0,adv=0,curv=0,shift=0,div=0;
   for(int i=0;i<3;++i)div+=u.beta.d[i][i];
   for(int i=0;i<3;++i)for(int j=0;j<3;++j){q+=g.inverse[i][j]*x[i]*x[j];grr+=x[i]*u.metric.g[i][j]*x[j];curv+=x[i]*(-2*u.alpha.value*u.a.k[i][j])*x[j];shift+=x[i]*(-2./3*u.metric.g[i][j]*div)*x[j];
    for(int k=0;k<3;++k){adv+=x[i]*u.beta.value[k]*u.metric.dg[k][i][j]*x[j];shift+=x[i]*(u.metric.g[i][k]*u.beta.d[j][k]+u.metric.g[j][k]*u.beta.d[i][k])*x[j];}}
   for(int k=0;k<6;++k){const int i=model.ti[k],j=model.tj[k];rr+=(i==j?1:2)*x[i]*sr[31*p+1+k]*x[j];}
   Real vrr=0,dr=0,ddtr=0,dpder=0,res2=0,dt2=0,pde2=0,dv=0,lr=0,delta=0;
   for(int i=0;i<3;++i){const Real ddt=(trace[1][31*p+25+i]-trace[0][31*p+25+i])/(2*eps);Real dpde=sr[31*p+15+i]-projected[31*p+25+i];for(int j=0;j<3;++j){dpde+=u.beta.value[j]*(sample.d[25+i][j]-sample.d[15+i][j]);dv+=x[i]*x[j]*sample.d[22+i][j];}
    vrr+=x[i]*projected[31*p+22+i];dr+=x[i]*projected[31*p+25+i];ddtr+=x[i]*ddt;dpder+=x[i]*dpde;res2+=(ddt-dpde)*(ddt-dpde);dt2+=ddt*ddt;pde2+=dpde*dpde;lr+=x[i]*projected[31*p+15+i];delta+=x[i]*(projected[31*p+25+i]-s[31*p+25+i]);}
   out<<p<<','<<x[0]<<','<<x[1]<<','<<x[2]<<','<<grr<<','<<u.chi.value<<','<<u.chi.value*q-1<<','<<rr<<','<<adv<<','<<curv<<','<<shift<<','<<rr-adv-curv-shift<<','<<vrr<<','<<dr<<','<<ddtr<<','<<dpder<<','<<sqrt(res2)<<','<<sqrt(dt2)<<','<<sqrt(pde2)<<','<<dv<<','<<4*dv<<','<<sample.value[13]<<','<<sample.value[14]<<','<<lr<<','<<delta<<'\n';
  }
  std::ofstream meta(pre+".json");meta<<std::setprecision(17)<<"{\"read_only\":true,\"input_admitted\":"<<h0.admitted<<",\"time\":"<<h0.time<<",\"steps\":"<<h0.steps<<",\"epsilon\":"<<eps<<",\"boundary_pole\":"<<e.pole_max<<",\"boundary_clock\":"<<e.clock_max<<"}\n";
 }catch(const std::exception&x){std::cerr<<x.what()<<'\n';code=1;}Kokkos::finalize();MPI_Finalize();return code;
}
