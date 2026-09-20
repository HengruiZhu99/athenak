#include <fstream>
#include <cmath>
#include <iomanip>
#include <iostream>
#include "../cartesian-design/constraint-radiation-v1.hpp"
using z4c::Z4c;
void read(const std::string &p,DvceArray5D<Real> &a){std::ifstream f(p,std::ios::binary);f.read(reinterpret_cast<char*>(a.data()),a.size()*sizeof(Real));if(!f)throw std::runtime_error(p);}
int main(int argc,char**argv){Kokkos::initialize(argc,argv);{
 const std::string path=argv[1];RegionIndcs in{};in.ng=4;in.nx1=in.nx2=in.nx3=8;in.is=in.js=in.ks=4;in.ie=in.je=in.ke=11;
 DvceArray5D<Real> bg("bg",8,25,16,16,16),u("u",8,25,16,16,16),pre("pre",8,25,16,16,16),post("post",8,25,16,16,16);
 const std::string name="z4c_snapshot_pre_boundary_rhs_rank0_cycle0_stage1";read(path+"/"+name+".background.bin",bg);read(path+"/"+name+".bin",pre);read(path+"/z4c_snapshot_post_boundary_rhs_rank0_cycle0_stage1.bin",post);Kokkos::deep_copy(u,bg);
 for(int m=0;m<8;++m)for(int k=4;k<12;++k)for(int j=4;j<12;++j)for(int i=4;i<12;++i){Real x=-2+2*(m&1)+(i-3.5)*.25,y=-2+2*((m>>1)&1)+(j-3.5)*.25,z=-2+2*((m>>2)&1)+(k-3.5)*.25;Real q=(std::pow(x-.75,2)+y*y+z*z)/.25;if(q<1)u(m,Z4c::I_Z4C_ALPHA,k,j,i)+=1e-8*std::exp(1-1/(1-q));}
 Z4c::Options opt{};opt.characteristic_radiation_areal_shift=1;Real idx[3]={4,4,4};std::cout<<std::setprecision(17)<<"[";bool comma=false;
 for(int m=0;m<8;++m)for(int k=4;k<12;++k)for(int j=4;j<12;++j)for(int i=4;i<12;++i){Real xyz[3]={-2+2*(m&1)+(i-3.5)*.25,-2+2*((m>>1)&1)+(j-3.5)*.25,-2+2*((m>>2)&1)+(k-3.5)*.25};int side[3]={};Real n[3]={};int count=0;for(int a=0;a<3;++a){if(std::abs(xyz[a])==1.875){n[a]=side[a]=xyz[a]>0?1:-1;++count;}}if(!count)continue;for(int a=0;a<3;++a)n[a]/=std::sqrt(count);Real ft,fq[3];int status=z4c::ComputeConstraintRadiationResidual(u,bg,pre,m,k,j,i,in,side,idx,n,n,xyz,opt,ft,fq);if(status)throw std::runtime_error("helperstatus");Real dk=post(m,7,k,j,i)-pre(m,7,k,j,i),dt=post(m,17,k,j,i)-pre(m,17,k,j,i),dgn=0,dan=0,tr=0,bn=0,fn=0;for(int a=0;a<3;++a){dgn+=n[a]*(post(m,14+a,k,j,i)-pre(m,14+a,k,j,i));bn+=n[a]*u(m,19+a,k,j,i);fn+=n[a]*fq[a];for(int b=0;b<3;++b){int v=8+z4c::RadiationSymmetricOffset(a,b);Real da=post(m,v,k,j,i)-pre(m,v,k,j,i);dan+=n[a]*n[b]*da;if(a==b)tr+=da;}}dan-=tr/3;Real al=u(m,18,k,j,i),ch=u(m,0,k,j,i),sq=std::sqrt(ch),cl=al*sq,lam=bn+cl;Real c1=sq*dt+ch*dgn/2,c2=4*dk/(3*sq)+2*dt/(3*sq)-2*dan/sq-dgn;
 if(comma)std::cout<<",";comma=true;std::cout<<"{\"gid\":"<<m<<",\"ijk\":["<<i<<","<<j<<","<<k<<"],\"xyz\":["<<xyz[0]<<","<<xyz[1]<<","<<xyz[2]<<"],\"incident_faces\":"<<count<<",\"FTheta\":"<<ft<<",\"FQ\":["<<fq[0]<<","<<fq[1]<<","<<fq[2]<<"],\"deltaTheta\":"<<dt<<",\"deltaKhat\":"<<dk<<",\"c1_correction_error\":"<<c1+lam*ft/al<<",\"c2_correction_error\":"<<c2-lam*fn/cl<<"}";
 }std::cout<<"]\n";}Kokkos::finalize();}
