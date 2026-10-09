#ifndef PRIVATE_EINSTEIN_COORDINATE_LIFT_HPP_
#define PRIVATE_EINSTEIN_COORDINATE_LIFT_HPP_
#include "cartesian_taylor.hpp"
#include "../complete_reference.hpp"
using C3=Poly<3>;using C2=Poly<2>;using C1=Poly<1>;
using CPoint=std::array<double,3>;
struct CoordinateBackground {
 C3 omega{},alpha{},chi{},P{},a_phys{};PVector<3>beta{};PMatrix<3>g{},bar{},physical{},physical_inverse{},A{},K{};
};
inline CoordinateBackground CoordinateReference(const CPoint&point){
 CoordinateBackground b{};double radius=0;std::array<C3,3>x;C3 rho;
 for(int i=0;i<3;++i){radius+=point[i]*point[i];x[i]=CartesianVariable<3>(point[i],i);rho+=x[i]*x[i];}radius=std::sqrt(radius);
 const auto p=CompleteReference(radius);
 if(radius<=.05){b.omega=1;b.alpha=1;b.chi=1;b.a_phys=1;for(int i=0;i<3;++i)b.g[i][i]=b.bar[i][i]=b.physical[i][i]=b.physical_inverse[i][i]=1;return b;}
 const C3 r=CartesianPower(rho,.5),o=ComposeRadial(p.omega,r,radius),alpha=ComposeRadial(p.alpha,r,radius),chi=ComposeRadial(p.chi,r,radius),gr=ComposeRadial(p.g_radial,r,radius),ar=ComposeRadial(p.A_radial,r,radius),at=ComposeRadial(p.A_tangent,r,radius);
 b.omega=o;b.alpha=alpha;b.chi=chi;b.P=ComposeRadial(p.P,r,radius);b.a_phys=alpha/o;
 for(int i=0;i<3;++i){b.beta[i]=ComposeRadial(p.beta,r,radius)*x[i]/r;for(int j=0;j<3;++j){const C3 nn=x[i]*x[j]/rho,I(i==j?1.:0.);b.g[i][j]=chi*I+(gr-chi)*nn;b.bar[i][j]=b.g[i][j]/chi;b.physical[i][j]=b.bar[i][j]/(o*o);b.A[i][j]=at*I+(ar-at)*nn;b.K[i][j]=b.A[i][j]/(o*chi)+b.physical[i][j]*b.P/C3(3);}}
 b.physical_inverse=MatrixInv(b.physical);return b;
}
struct Coordinates {C3 T{},Tdot{};PVector<3>X{},Xdot{};};
inline C3 RhoPolynomial(const C3&rho,const std::array<double,4>&a){C3 out(a[3]);for(int k=2;k>=0;--k)out=out*rho+C3(a[k]);return out;}
inline Coordinates CoordinatePolynomial(const CPoint&point,const std::array<std::array<double,4>,4>&coeff){Coordinates z{};std::array<C3,3>x;C3 rho;for(int i=0;i<3;++i){x[i]=CartesianVariable<3>(point[i],i);rho+=x[i]*x[i];}z.T=RhoPolynomial(rho,coeff[0]);z.Tdot=RhoPolynomial(rho,coeff[2]);for(int i=0;i<3;++i){z.X[i]=x[i]*RhoPolynomial(rho,coeff[1]);z.Xdot[i]=x[i]*RhoPolynomial(rho,coeff[3]);}return z;}
struct CoordinateLift {
 C2 alpha{},chi{};C1 P{};PVector<2>beta{};PVector<1>lambda{};PMatrix<2>g{},hbar{},h_phys{};PMatrix<1>A{},k_phys{};
};
inline CoordinateLift EinsteinLift(const CoordinateBackground&b,const Coordinates&z){
 // Algebraically identical stationary Lie lift, with conformal trace terms
 // canceled before numerical evaluation. No equation/source change.
 CoordinateLift out{};const auto bar=CutMatrix<2>(b.bar),bi=MatrixInv(bar);
 C2 sigma=Cut<2>(z.Tdot),psi;for(int k=0;k<3;++k){sigma-=Cut<2>(b.beta[k])*Partial(z.T,k);psi+=Cut<2>(z.X[k])*Partial(b.omega,k)/Cut<2>(b.omega);}
 out.alpha=Cut<2>(b.alpha)*(sigma-psi);for(int k=0;k<3;++k)out.alpha+=Cut<2>(z.X[k])*Partial(b.alpha,k);
 PMatrix<2>h0{};C2 trh;
 for(int i=0;i<3;++i){out.beta[i]=Cut<2>(z.Xdot[i])+Cut<2>(b.beta[i])*sigma;for(int k=0;k<3;++k)out.beta[i]+=Cut<2>(z.X[k])*Partial(b.beta[i],k)-Cut<2>(b.beta[k])*Partial(z.X[i],k)-Cut<2>(b.alpha*b.alpha)*bi[i][k]*Partial(z.T,k);
  for(int j=0;j<3;++j){for(int k=0;k<3;++k)h0[i][j]+=Cut<2>(z.X[k])*Partial(b.bar[i][j],k)+bar[k][j]*Partial(z.X[k],i)+bar[i][k]*Partial(z.X[k],j)+bar[i][k]*Cut<2>(b.beta[k])*Partial(z.T,j)+bar[j][k]*Cut<2>(b.beta[k])*Partial(z.T,i);trh+=bi[i][j]*h0[i][j];out.hbar[i][j]=h0[i][j]-C2(2)*psi*bar[i][j];out.h_phys[i][j]=out.hbar[i][j]/Cut<2>(b.omega*b.omega);}
 }
 out.chi=Cut<2>(b.chi)*(C2(2)*psi-trh/C2(3));for(int i=0;i<3;++i)for(int j=0;j<3;++j)out.g[i][j]=Cut<2>(b.chi)*(h0[i][j]-bar[i][j]*trh/C2(3));
 std::array<std::array<std::array<C1,3>,3>,3>connection{};
 for(int k=0;k<3;++k)for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int l=0;l<3;++l)connection[k][i][j]+=C1(.5)*Cut<1>(bi[k][l])*(Partial(bar[l][j],i)+Partial(bar[l][i],j)-Partial(bar[i][j],l));
 C1 Xlog,XDP,OmegaDT;for(int k=0;k<3;++k){Xlog+=Cut<1>(z.X[k])*Cut<1>(Partial(b.omega*b.chi,k))/Cut<1>(b.omega*b.chi);XDP+=Cut<1>(z.X[k])*Cut<1>(Partial(b.P,k));for(int l=0;l<3;++l)OmegaDT+=Cut<1>(bi[k][l])*Cut<1>(Partial(b.omega,k))*Cut<1>(Partial(z.T,l));}
 PMatrix<1>Q{};C1 trQ,raisedAh;
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){C1 hessian=Partial(Partial(z.T,i),j);for(int k=0;k<3;++k)hessian-=connection[k][i][j]*Cut<1>(Partial(z.T,k));Q[i][j]=-Cut<1>(b.alpha*b.chi)*hessian-Cut<1>(b.chi)*(Cut<1>(Partial(b.alpha,i))*Cut<1>(Partial(z.T,j))+Cut<1>(Partial(b.alpha,j))*Cut<1>(Partial(z.T,i)))-Cut<1>(b.A[i][j])*Xlog;
  for(int k=0;k<3;++k)Q[i][j]+=Cut<1>(z.X[k])*Partial(Cut<2>(b.A[i][j]),k)+Cut<1>(b.A[k][j])*Cut<1>(Partial(z.X[k],i))+Cut<1>(b.A[i][k])*Cut<1>(Partial(z.X[k],j))+Cut<1>(b.A[k][i])*Cut<1>(b.beta[k])*Cut<1>(Partial(z.T,j))+Cut<1>(b.A[k][j])*Cut<1>(b.beta[k])*Cut<1>(Partial(z.T,i));
  trQ+=Cut<1>(bi[i][j])*Q[i][j];for(int k=0;k<3;++k)for(int l=0;l<3;++l)raisedAh+=Cut<1>(bi[i][k]*bi[j][l]*Cut<2>(b.A[k][l])*h0[i][j]);
 }
 out.P=XDP+C1(3)*Cut<1>(b.alpha)*OmegaDT+Cut<1>(b.omega/b.chi)*(trQ-raisedAh);
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){out.A[i][j]=Q[i][j]-Cut<1>(bar[i][j])*(trQ-raisedAh)/C1(3)+Cut<1>(out.chi/Cut<2>(b.chi))*Cut<1>(b.A[i][j]);out.k_phys[i][j]=(out.A[i][j]-Cut<1>(out.chi/Cut<2>(b.chi))*Cut<1>(b.A[i][j]))/Cut<1>(b.omega*b.chi)+Cut<1>(b.physical[i][j])*out.P/C1(3)+Cut<1>(b.P)*Cut<1>(out.h_phys[i][j])/C1(3);}
 const auto gi=MatrixInv(CutMatrix<2>(b.g));PMatrix<2>q{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int k=0;k<3;++k)for(int l=0;l<3;++l)q[i][j]+=gi[i][k]*out.g[k][l]*gi[l][j];for(int i=0;i<3;++i)for(int j=0;j<3;++j)out.lambda[i]+=Partial(q[i][j],j);return out;
}
#endif
