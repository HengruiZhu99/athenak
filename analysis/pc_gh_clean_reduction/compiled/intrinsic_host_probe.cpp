#include <fstream>
#include <iomanip>
#include <cstdlib>
#include "pc_gh/intrinsic_rhs.hpp"
int main(int argc,char **argv) {
  if (argc!=4) return 2;
  std::ifstream input(argv[1]);std::ofstream output(argv[2]);int count;input>>count;
  output<<std::setprecision(17);
  for (int row=0;row<count && row<std::atoi(argv[3]);++row) {
    double u[50],du[3][50],lambda,eta,kappa,rhs[50];
    for (double &x:u) input>>x;
    for (auto &direction:du) for (double &x:direction) input>>x;
    input>>lambda>>eta>>kappa;if (!input) return 3;
    pc_gh::intrinsic::PointRHS(u,du,lambda,eta,kappa,rhs);
    for (int n=0;n<50;++n) output<<(n?" ":"")<<rhs[n];output<<'\n';
  }
}
