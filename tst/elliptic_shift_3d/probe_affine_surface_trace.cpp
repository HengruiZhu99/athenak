// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
// Surface decomposition control only; immutable accepted geometry and beta.
#define main discrete_connection_audit_main
#include "audit_discrete_connection.cpp"
#undef main
struct Stats {
 Real trace_max=0,slope_max=0,tangent_max=0,normal_max=0,hessian_max=0,closure_max=0,full_tangent_max=0,full_hessian_tangent_max=0;
 void Print(){std::cout<<std::setprecision(17)<<"{\"scri_delta_beta_trace_max\":"<<trace_max<<",\"global_affine_slope_max\":"<<slope_max<<",\"tangent_gradient_difference_max\":"<<tangent_max<<",\"normal_gradient_difference_max\":"<<normal_max<<",\"hessian_difference_max\":"<<hessian_max<<",\"affine_cancellation_closure_max\":"<<closure_max<<",\"full_trace_tangent_gradient_max\":"<<full_tangent_max<<",\"full_trace_tangent_tangent_hessian_max\":"<<full_hessian_tangent_max<<'}';}
};
Stats Run(Audit&audit,bool zero,std::ofstream&out){auto&e=audit.e;auto&g=audit.g;c::HermiteBoundarySampler sampler(*e.scri);V affineActive(3*audit.N),affineTrace(3*audit.S),remainder(3*audit.N),trace=e.scri->beta;Stats s;
 if(zero)std::fill(trace.begin(),trace.end(),0);
 for(size_t p=0;p<audit.N+audit.S;++p){bool boundary=p>=audit.N;size_t k=boundary?p-audit.N:p;const Real*x=boundary?e.scri->rows[k].geometry.x:g.geometry[k].x;const auto a=e.shift.affine_derivative->At(*e.shift_affine,x);for(int v=0;v<3;++v)(boundary?affineTrace:affineActive)[3*k+v]=a.value[v];}
 for(Real x:trace)s.trace_max=std::max(s.trace_max,std::abs(x));for(int i=0;i<3;++i)for(int j=0;j<3;++j)s.slope_max=std::max(s.slope_max,std::abs(e.shift_affine->slope[i][j]));
 V rt(trace.size());for(size_t p=0;p<rt.size();++p)rt[p]=trace[p]-affineTrace[p];for(size_t p=0;p<remainder.size();++p)remainder[p]=e.beta_delta[p]-affineActive[p];
 auto fullPad=audit.Pad(e.beta_delta,trace,3),aPad=audit.Pad(affineActive,affineTrace,3),rPad=audit.Pad(remainder,rt,3);auto fp=sampler.Prepare(fullPad,trace,3,1),ap=sampler.Prepare(aPad,affineTrace,3,1),rp=sampler.Prepare(rPad,rt,3,1);
 for(size_t p=0;p<audit.S;++p){const Real*x=e.scri->rows[p].geometry.x;auto f=sampler.Sample(p,fp),a=sampler.Sample(p,ap),r=sampler.Sample(p,rp);const auto exactA=e.shift.affine_derivative->At(*e.shift_affine,x);Real tm=0,nm=0,hm=0,cm=0,ftm=0,fhm=0;
  for(int v=0;v<3;++v){Real diff[3],expected[3],normal=0,fullnormal=0;for(int d=0;d<3;++d){diff[d]=r.d[v][d]+exactA.d[d][v]-f.d[v][d];expected[d]=exactA.d[d][v]-a.d[v][d];normal+=x[d]*diff[d];fullnormal+=x[d]*f.d[v][d];cm=std::max(cm,std::abs(diff[d]-expected[d]));for(int j=0;j<3;++j){const Real hd=r.dd[v][d][j]-f.dd[v][d][j];hm=std::max(hm,std::abs(hd));cm=std::max(cm,std::abs(hd+a.dd[v][d][j]));}}
   for(int d=0;d<3;++d){tm=std::max(tm,std::abs(diff[d]-x[d]*normal));ftm=std::max(ftm,std::abs(f.d[v][d]-x[d]*fullnormal));for(int j=0;j<3;++j){Real tangent_h=0;for(int i=0;i<3;++i)for(int k=0;k<3;++k)tangent_h+=((d==i?1.:0)-x[d]*x[i])*((j==k?1.:0)-x[j]*x[k])*f.dd[v][i][k];fhm=std::max(fhm,std::abs(tangent_h));}}nm=std::max(nm,std::abs(normal));}
  s.tangent_max=std::max(s.tangent_max,tm);s.normal_max=std::max(s.normal_max,nm);s.hessian_max=std::max(s.hessian_max,hm);s.closure_max=std::max(s.closure_max,cm);s.full_tangent_max=std::max(s.full_tangent_max,ftm);s.full_hessian_tangent_max=std::max(s.full_hessian_tangent_max,fhm);out<<zero<<','<<p<<','<<x[0]<<','<<x[1]<<','<<x[2]<<','<<tm<<','<<nm<<','<<hm<<','<<cm<<','<<ftm<<','<<fhm<<'\n';
 }
 return s;
}
int main(int argc,char**argv){Kokkos::initialize(argc,argv);int code=0;try{if(argc!=4)throw std::runtime_error("accepted_snapshot tag outputcsv");auto snapshot=c::CompositeCheckpoint::Read(argv[1],argv[2]);Audit audit(snapshot);std::ofstream out(argv[3]);out<<std::setprecision(17)<<"zero_trace,point,x,y,z,tangent_difference_max,normal_difference_max,hessian_difference_max,affine_cancellation_closure_max,full_trace_tangent_max,full_trace_tangent_tangent_hessian_max\n";auto actual=Run(audit,false,out),zero=Run(audit,true,out);std::cout<<"{\"actual_trace\":";actual.Print();std::cout<<",\"zero_trace_nonzero_volume_fit\":";zero.Print();std::cout<<",\"all_surface_nodes_retained\":true,\"no_evolution_or_solve\":true,\"operator_or_sampler_adopted\":false}\n";}catch(const std::exception&e){std::cerr<<e.what()<<'\n';code=2;}Kokkos::finalize();return code;}
