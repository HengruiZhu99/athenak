#ifndef PRIVATE_INERTIAL_STABLE_IDENTITY_HPP_
#define PRIVATE_INERTIAL_STABLE_IDENTITY_HPP_
struct IdentityChecks {
 std::array<double,4>original_time{},actual_factor_relation{},stable_time{};
 std::array<double,3>original_space{},stable_space{};
 double stable_max=0,returned_jet_negative_control=0;
};
template<int N,std::size_t K>void ByOrder(const Poly<N>&a,const Poly<N>&b,std::array<double,K>&errors){static_assert(K==N+1,"Order layout");for(int i=0;i<=N;++i)for(int j=0;j<=N-i;++j)for(int k=0;k<=N-i-j;++k)errors[i+j+k]=std::max(errors[i+j+k],std::abs(a(i,j,k)-b(i,j,k))/std::max(1.,std::abs(b(i,j,k))));}
inline IdentityChecks StableIdentity(const CPoint&point,const std::array<std::array<double,4>,4>&coeff,const Coordinates&returned){
 const auto raw=CoordinatePolynomial(point,coeff);IdentityChecks e{};C3 negative_expected;double radius=0;std::array<C3,3>x;C3 rho;for(int i=0;i<3;++i){radius+=point[i]*point[i];x[i]=CartesianVariable<3>(point[i],i);rho+=x[i]*x[i];}radius=std::sqrt(radius);
 if(radius<=.05){negative_expected=raw.X[0];ByOrder(returned.T,raw.T,e.original_time);ByOrder(returned.Tdot,raw.Tdot,e.original_time);e.stable_time=e.original_time;for(int i=0;i<3;++i){ByOrder(returned.X[i],raw.X[i],e.actual_factor_relation);ByOrder(returned.Xdot[i],raw.Xdot[i],e.actual_factor_relation);ByOrder(Cut<2>(returned.X[i]),Cut<2>(raw.X[i]),e.original_space);ByOrder(Cut<2>(returned.Xdot[i]),Cut<2>(raw.Xdot[i]),e.original_space);}e.stable_space=e.original_space;}
 else {
  const auto p=CompleteReference(radius);const auto r=CartesianPower(rho,.5),o=ComposeRadial(p.omega,r,radius),L=ComposeRadial(p.L,r,radius),b=ComposeRadial(p.b,r,radius),alpha=ComposeRadial(p.alpha,r,radius);
  C3 original_time=returned.T,original_velocity=returned.Tdot,stable_time=returned.T,stable_velocity=returned.Tdot;
  std::array<C3,3>factor,factor_velocity;
  for(int i=0;i<3;++i){
   factor[i]=raw.X[i]/L;factor_velocity[i]=raw.Xdot[i]/L;
   // This relation binds the ACTUAL returned Taylor coefficients. The other
   // contractions alone would only check the prescribed input displacement.
   ByOrder(returned.X[i],o*o*factor[i],e.actual_factor_relation);ByOrder(returned.Xdot[i],o*o*factor_velocity[i],e.actual_factor_relation);
   const C3 scaled_height=b*L*x[i]/(alpha*r);
   const C3 original_height_gradient=b*L*x[i]/(alpha*o*o*r);
   original_time+=original_height_gradient*returned.X[i];original_velocity+=original_height_gradient*returned.Xdot[i];
   stable_time+=scaled_height*factor[i];stable_velocity+=scaled_height*factor_velocity[i];
  }
  negative_expected=o*o*factor[0];
  ByOrder(original_time,raw.T,e.original_time);ByOrder(original_velocity,raw.Tdot,e.original_time);ByOrder(stable_time,raw.T,e.stable_time);ByOrder(stable_velocity,raw.Tdot,e.stable_time);
  for(int i=0;i<3;++i){const C3 R=x[i]/o;C2 original,original_velocity,stable,stable_velocity;for(int j=0;j<3;++j){original+=Partial(R,j)*Cut<2>(returned.X[j]);original_velocity+=Partial(R,j)*Cut<2>(returned.Xdot[j]);const C2 scaled_jacobian=Cut<2>(o)*C2(i==j?1:0)-Cut<2>(x[i])*Partial(o,j);stable+=scaled_jacobian*Cut<2>(factor[j]);stable_velocity+=scaled_jacobian*Cut<2>(factor_velocity[j]);}ByOrder(original,Cut<2>(raw.X[i]),e.original_space);ByOrder(original_velocity,Cut<2>(raw.Xdot[i]),e.original_space);ByOrder(stable,Cut<2>(raw.X[i]),e.stable_space);ByOrder(stable_velocity,Cut<2>(raw.Xdot[i]),e.stable_space);}
 }
 for(double v:e.actual_factor_relation)e.stable_max=std::max(e.stable_max,v);for(double v:e.stable_time)e.stable_max=std::max(e.stable_max,v);for(double v:e.stable_space)e.stable_max=std::max(e.stable_max,v);
 // Exact manufactured output corruption, independent of actual source action.
 // It is a negative check of returned-coefficient binding, not a new seed.
 auto corrupted=returned.X[0];corrupted(3,0,0)+=1e-6;std::array<double,4>negative{};ByOrder(corrupted,negative_expected,negative);for(double v:negative)e.returned_jet_negative_control=std::max(e.returned_jet_negative_control,v);
 return e;
}
#endif
