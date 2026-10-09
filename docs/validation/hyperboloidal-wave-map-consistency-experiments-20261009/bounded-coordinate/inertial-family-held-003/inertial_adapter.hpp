#ifndef PRIVATE_INERTIAL_COORDINATE_ADAPTER_HPP_
#define PRIVATE_INERTIAL_COORDINATE_ADAPTER_HPP_
// New witness family only. The C0 equations, reference and gauge are unchanged.
// Xi^T=tau(rho), Xi^I=x^I zeta(rho), in physical inertial coordinates.
// The complete reference embedding has dR/dr=L/Omega^2 and dh/dR=b/alpha.
inline Coordinates InertialCoordinates(const CPoint&point,const std::array<std::array<double,4>,4>&coeff){
 const auto raw=CoordinatePolynomial(point,coeff);double radius=0;for(double x:point)radius+=x*x;radius=std::sqrt(radius);
 if(radius<=.05)return raw; // exact Minkowski identity chart, including origin.
 std::array<C3,3>x;C3 rho;for(int i=0;i<3;++i){x[i]=CartesianVariable<3>(point[i],i);rho+=x[i]*x[i];}
 const auto p=CompleteReference(radius);const auto r=CartesianPower(rho,.5),o=ComposeRadial(p.omega,r,radius),L=ComposeRadial(p.L,r,radius),b=ComposeRadial(p.b,r,radius),alpha=ComposeRadial(p.alpha,r,radius);
 const C3 spatial=o*o/L,height=b*r/alpha,zeta=RhoPolynomial(rho,coeff[1]),w=RhoPolynomial(rho,coeff[3]);Coordinates z;
 z.T=raw.T-height*zeta;z.Tdot=raw.Tdot-height*w;for(int i=0;i<3;++i){z.X[i]=spatial*raw.X[i];z.Xdot[i]=spatial*raw.Xdot[i];}return z;
}
inline double InertialAdapterIdentity(const CPoint&point,const std::array<std::array<double,4>,4>&coeff){
 const auto raw=CoordinatePolynomial(point,coeff),z=InertialCoordinates(point,coeff);double radius=0;std::array<C3,3>x;C3 rho;for(int i=0;i<3;++i){radius+=point[i]*point[i];x[i]=CartesianVariable<3>(point[i],i);rho+=x[i]*x[i];}radius=std::sqrt(radius);
 if(radius<=.05){double e=PolyError(raw.T,z.T);e=std::max(e,PolyError(raw.Tdot,z.Tdot));for(int i=0;i<3;++i)e=std::max({e,PolyError(raw.X[i],z.X[i]),PolyError(raw.Xdot[i],z.Xdot[i])});return e;}
 const auto p=CompleteReference(radius);const auto r=CartesianPower(rho,.5),o=ComposeRadial(p.omega,r,radius),L=ComposeRadial(p.L,r,radius),b=ComposeRadial(p.b,r,radius),alpha=ComposeRadial(p.alpha,r,radius);C3 time=z.T,velocity=z.Tdot;
 for(int i=0;i<3;++i){const C3 height_gradient=b*L*x[i]/(alpha*o*o*r);time+=height_gradient*z.X[i];velocity+=height_gradient*z.Xdot[i];}
 double error=std::max(PolyError(time,raw.T),PolyError(velocity,raw.Tdot));
 for(int i=0;i<3;++i){const C3 R=x[i]/o;C2 displacement,displacement_velocity;for(int j=0;j<3;++j){displacement+=Partial(R,j)*Cut<2>(z.X[j]);displacement_velocity+=Partial(R,j)*Cut<2>(z.Xdot[j]);}error=std::max({error,PolyError(displacement,Cut<2>(raw.X[i])),PolyError(displacement_velocity,Cut<2>(raw.Xdot[i]))});}
 return error;
}
#endif
