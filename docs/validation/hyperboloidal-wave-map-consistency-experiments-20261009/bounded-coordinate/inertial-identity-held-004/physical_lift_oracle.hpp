// Byte-identical original physical-variable formula with function name adapted only.
inline CoordinateLift EinsteinLiftPhysicalOriginal(const CoordinateBackground&b,const Coordinates&z){
 CoordinateLift out{};const auto physical=CutMatrix<2>(b.physical),physical_inv=CutMatrix<2>(b.physical_inverse),bar_inv=MatrixInv(CutMatrix<2>(b.bar));
 C2 sigma=Cut<2>(z.Tdot);for(int i=0;i<3;++i)sigma-=Cut<2>(b.beta[i])*Partial(z.T,i);
 out.alpha=Cut<2>(b.omega)*Cut<2>(b.a_phys)*sigma;
 for(int k=0;k<3;++k)out.alpha+=Cut<2>(b.omega)*Cut<2>(z.X[k])*Partial(b.a_phys,k);
 for(int i=0;i<3;++i){out.beta[i]=Cut<2>(z.Xdot[i])+Cut<2>(b.beta[i])*sigma;for(int k=0;k<3;++k)out.beta[i]+=Cut<2>(z.X[k])*Partial(b.beta[i],k)-Cut<2>(b.beta[k])*Partial(z.X[i],k)-Cut<2>(b.alpha*b.alpha)*bar_inv[i][k]*Partial(z.T,k);}
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){
  for(int k=0;k<3;++k)out.h_phys[i][j]+=Cut<2>(z.X[k])*Partial(b.physical[i][j],k)+physical[k][j]*Partial(z.X[k],i)+physical[i][k]*Partial(z.X[k],j)+physical[i][k]*Cut<2>(b.beta[k])*Partial(z.T,j)+physical[j][k]*Cut<2>(b.beta[k])*Partial(z.T,i);
  out.hbar[i][j]=Cut<2>(b.omega*b.omega)*out.h_phys[i][j];
 }
 std::array<std::array<std::array<C1,3>,3>,3>connection{};
 for(int k=0;k<3;++k)for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int l=0;l<3;++l)connection[k][i][j]+=C1(.5)*Cut<1>(physical_inv[k][l])*(Partial(physical[l][j],i)+Partial(physical[l][i],j)-Partial(physical[i][j],l));
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){
  C1 hessian=Partial(Partial(z.T,i),j);for(int k=0;k<3;++k)hessian-=connection[k][i][j]*Cut<1>(Partial(z.T,k));
  out.k_phys[i][j]=-Cut<1>(b.a_phys)*hessian-Cut<1>(Partial(b.a_phys,i))*Cut<1>(Partial(z.T,j))-Cut<1>(Partial(b.a_phys,j))*Cut<1>(Partial(z.T,i));
  for(int k=0;k<3;++k)out.k_phys[i][j]+=Cut<1>(z.X[k])*Partial(Cut<2>(b.K[i][j]),k)+Cut<1>(b.K[k][j])*Cut<1>(Partial(z.X[k],i))+Cut<1>(b.K[i][k])*Cut<1>(Partial(z.X[k],j))+Cut<1>(b.K[k][i])*Cut<1>(b.beta[k])*Cut<1>(Partial(z.T,j))+Cut<1>(b.K[k][j])*Cut<1>(b.beta[k])*Cut<1>(Partial(z.T,i));
 }
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){out.P+=Cut<1>(physical_inv[i][j])*out.k_phys[i][j];for(int k=0;k<3;++k)for(int l=0;l<3;++l)out.P-=Cut<1>(physical_inv[i][k]*physical_inv[j][l])*Cut<1>(b.K[k][l])*Cut<1>(out.h_phys[i][j]);out.chi-=Cut<2>(b.chi)*bar_inv[i][j]*out.hbar[i][j]/C2(3);}
 for(int i=0;i<3;++i)for(int j=0;j<3;++j){out.g[i][j]=Cut<2>(b.chi)*out.hbar[i][j]+Cut<2>(b.g[i][j]/b.chi)*out.chi;out.A[i][j]=Cut<1>(b.omega*b.chi)*(out.k_phys[i][j]-Cut<1>(b.P)*Cut<1>(out.h_phys[i][j])/C1(3)-Cut<1>(b.physical[i][j])*out.P/C1(3))+Cut<1>(out.chi/Cut<2>(b.chi))*Cut<1>(b.A[i][j]);}
 const auto gi=MatrixInv(CutMatrix<2>(b.g));PMatrix<2>q{};
 for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int k=0;k<3;++k)for(int l=0;l<3;++l)q[i][j]+=gi[i][k]*out.g[k][l]*gi[l][j];
 for(int i=0;i<3;++i)for(int j=0;j<3;++j)out.lambda[i]+=Partial(q[i][j],j);
 return out;
}
