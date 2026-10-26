// Native-map true-face physical derivative kernel, zero evolution.
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include "goal1_physical_interface.hpp"
#include "goal1_operator_oracle.hpp"
using namespace cubed_sphere;
int main(int argc,char**argv){Kokkos::initialize(argc,argv);int status=0;
try{
 int na=argc>1?std::stoi(argv[1]):16,nr=argc>2?std::stoi(argv[2]):24;research_interface::Physical op(na,nr,Map{.25,.98,0});int N=op.W.size();double constant=0,metric=0,metricmin=0,identity=op.IdentityDefect(),wmin=1e100;
 for(int d=0;d<3;++d){auto f=op.D[d].Apply(std::vector<double>(N,1));for(int i=0;i<N;++i){constant=std::max(constant,std::abs(f[i]));metric=std::max(metric,std::abs(op.metric_defect[d][i]));metricmin=std::min(metricmin,op.metric_defect[d][i]);}}
 for(auto w:op.W)wmin=std::min(wmin,w);
 std::cout<<std::setprecision(17)<<"{\"nodes\":"<<N<<",\"angular\":"<<na<<",\"radial\":"<<nr<<",\"internal_face_point_pairs\":"<<op.internal_pairs<<",\"external_face_points\":"<<op.external_points<<",\"face_coordinate_defect\":"<<op.face_coordinate_defect<<",\"face_area_defect\":"<<op.face_area_defect<<",\"SBP_energy_scaled_identity_defect\":"<<identity<<",\"constant_absolute_defect\":"<<constant<<",\"raw_metric_identity_max\":"<<metric<<",\"scalar_energy_volume_growth_bound\":"<<-2*metricmin<<",\"minimum_physical_mass\":"<<wmin<<",\"profiles\":[";
 for(int kind=0;kind<4;++kind){std::vector<double>f(N);for(int i=0;i<N;++i)f[i]=operator_oracle::Function(op.points[i].x,kind).value;std::array<std::vector<double>,3>df;for(int d=0;d<3;++d)df[d]=op.D[d].Apply(f);
  std::array<std::vector<double>,9>ddf;for(int a=0;a<3;++a)for(int b=0;b<3;++b)ddf[3*a+b]=op.D[a].Apply(df[b]);
  struct Stats{double w=0,g=0,h=0,m=0,curl=0,gmax=0,hmax=0,cmax=0;};std::array<Stats,7>stats{};
  for(int i=0;i<N;++i){auto target=operator_oracle::Function(op.points[i].x,kind);double eg=0,eh=0,em=0,ec=0;
   for(int a=0;a<3;++a){eg+=std::pow(df[a][i]-target.d[a],2);for(int b=0;b<3;++b){double e=.5*(ddf[3*a+b][i]+ddf[3*b+a][i])-target.dd[a][b];eh+=e*e;if(a!=b){em+=e*e;ec+=std::pow(ddf[3*a+b][i]-ddf[3*b+a][i],2);}}}
   int block=0;while(block<6&&i>=op.blocks[block+1].start)++block;const auto &bl=op.blocks[block];int local=i-bl.start;int ix=local%bl.n[0],iy=(local/bl.n[0])%bl.n[1],iz=local/(bl.n[0]*bl.n[1]);int seam=(iy<4||iy>=bl.n[1]-4)+(iz<4||iz>=bl.n[2]-4);double radius=std::sqrt(std::pow(op.points[i].x[0],2)+std::pow(op.points[i].x[1],2)+std::pow(op.points[i].x[2],2));
   bool masks[7]={true,block==0,block!=0,block!=0&&ix<4,block!=0&&seam==1,block!=0&&seam==2,block!=0&&radius>.9};
   for(int j=0;j<7;++j)if(masks[j]){auto&s=stats[j];double w=op.W[i];s.w+=w;s.g+=w*eg;s.h+=w*eh;s.m+=w*em;s.curl+=w*ec;s.gmax=std::max(s.gmax,std::sqrt(eg));s.hmax=std::max(s.hmax,std::sqrt(eh));s.cmax=std::max(s.cmax,std::sqrt(ec));}
  }
  if(kind)std::cout<<",";std::cout<<"{\"kind\":"<<kind<<",\"regions\":[";const char* names[7]={"all","bulk","shell","inner-interface","seam-edge","seam-corner","outer-collar"};
  for(int j=0;j<7;++j){auto&s=stats[j];if(j)std::cout<<",";std::cout<<"{\"region\":\""<<names[j]<<"\",\"mass\":"<<s.w<<",\"first_rms\":"<<sqrt(s.g/s.w)<<",\"first_linf\":"<<s.gmax<<",\"second_rms\":"<<sqrt(s.h/s.w)<<",\"second_linf\":"<<s.hmax<<",\"mixed_rms\":"<<sqrt(s.m/s.w)<<",\"curl_rms\":"<<sqrt(s.curl/s.w)<<",\"curl_linf\":"<<s.cmax<<"}";}
  std::cout<<"]}";
 }
 std::cout<<"],\"dependencies\":[";
 // Both directions and every Cartesian component, plus a decoupled control.
 for(int source=0;source<2;++source){std::vector<double>f(N);for(int i=0;i<N;++i)if((i<op.blocks[1].start)==(source==0))f[i]=operator_oracle::Function(op.points[i].x,3).value;
  double response=0,negative=0;for(int d=0;d<3;++d){auto df=op.D[d].Apply(f);for(int i=0;i<N;++i)if((i<op.blocks[1].start)!=(source==0)){response=std::max(response,std::abs(df[i]));double v=0;for(int n=op.D[d].offsets[i];n<op.D[d].offsets[i+1];++n){int col=op.D[d].columns[n];if((col<op.blocks[1].start)==(i<op.blocks[1].start))v+=op.D[d].values[n]*f[col];}negative=std::max(negative,std::abs(v));}}
  if(source)std::cout<<",";std::cout<<"{\"source\":\""<<(source?"shell":"bulk")<<"\",\"target_first_response\":"<<response<<",\"decoupled_control_response\":"<<negative<<"}";
 }
 std::cout<<"],\"evolution_trajectories\":0,\"native_loaded_fields_tested\":false}"<<std::endl;
 if(identity>1e-10||constant>1e-10)status=2;
}catch(const std::exception&e){std::cerr<<e.what()<<std::endl;status=3;}
Kokkos::finalize();return status;}
