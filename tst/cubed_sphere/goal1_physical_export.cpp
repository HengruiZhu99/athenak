// Read-only export of the tested native-compatible graph, not an evolution.
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include "goal1_physical_interface.hpp"
using namespace cubed_sphere::research_interface;
template<class T>void Write(std::ofstream&o,const std::vector<T>&v){o.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(T));}
int main(int argc,char**argv){Kokkos::initialize(argc,argv);int status=0;
try{if(argc!=4)throw std::runtime_error("need na,nr,output");int na=std::stoi(argv[1]),nr=std::stoi(argv[2]);Physical p(na,nr,cubed_sphere::Map{.25,.98,0});std::ofstream o(argv[3],std::ios::binary);std::uint64_t header[4]={0x31465342504f5347ULL,p.W.size(),static_cast<unsigned long long>(na),static_cast<unsigned long long>(nr)};o.write(reinterpret_cast<const char*>(header),sizeof(header));
 Write(o,p.W);std::vector<double>x;for(auto &point:p.points)for(double v:point.x)x.push_back(v);Write(o,x);
 for(int d=0;d<3;++d)Write(o,p.metric_defect[d]);
 for(int d=0;d<3;++d)for(const auto*s:{&p.D[d],&p.B[d]}){std::uint64_t nnz=s->values.size();o.write(reinterpret_cast<const char*>(&nnz),8);Write(o,s->offsets);Write(o,s->columns);Write(o,s->values);}
 if(!o)throw std::runtime_error("physical graph write failed");
 // Verify energy identity on the union of D,transposeD and B support.
 double error=0,scale=1;std::uint64_t boundary_only=0;
 for(int d=0;d<3;++d)for(size_t i=0;i<p.W.size();++i){std::vector<int>cols;for(int k=p.D[d].offsets[i];k<p.D[d].offsets[i+1];++k)cols.push_back(p.D[d].columns[k]);for(int k=p.B[d].offsets[i];k<p.B[d].offsets[i+1];++k){int j=p.B[d].columns[k];if(p.D[d].At(i,j)==0&&p.D[d].At(j,i)==0)++boundary_only;cols.push_back(j);}for(int j:cols){double root=sqrt(p.W[i]*p.W[j]);double lhs=p.W[i]*p.D[d].At(i,j)+p.W[j]*p.D[d].At(j,i);double rhs=p.B[d].At(i,j)-(i==size_t(j)?2*p.W[i]*p.metric_defect[d][i]:0);error=std::max(error,std::abs(lhs-rhs)/root);scale=std::max(scale,std::abs(p.W[i]*p.D[d].At(i,j))/root);}}
 std::cout<<std::setprecision(17)<<"{\"nodes\":"<<p.W.size()<<",\"complete_union_SBP_scaled_defect\":"<<error/scale<<",\"boundary_only_entries_checked\":"<<boundary_only<<",\"serialized_int_bytes\":"<<sizeof(int)<<",\"serialized_double_bytes\":"<<sizeof(double)<<"}"<<std::endl;if(error/scale>1e-10)status=2;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;status=3;}Kokkos::finalize();return status;}
