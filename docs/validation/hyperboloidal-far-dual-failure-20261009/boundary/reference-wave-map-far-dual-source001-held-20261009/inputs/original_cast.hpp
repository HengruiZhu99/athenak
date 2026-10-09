hyp::LayerPoint<D> CastPoint(const hyp::LayerPoint<double>&p){hyp::LayerPoint<D>q{};
 q.state=Lift(p.state);q.alpha=p.alpha;q.omega=p.omega;q.radius=p.radius;q.L=p.L;q.b=p.b;q.k_bar=p.k_bar;q.k_physical=p.k_physical;
 for(int i=0;i<3;++i){q.beta[i]=p.beta[i];q.domega[i]=p.domega[i];q.dalpha[i]=p.dalpha[i];for(int j=0;j<3;++j)q.omega_hessian[i][j]=p.omega_hessian[i][j];}return q;}
