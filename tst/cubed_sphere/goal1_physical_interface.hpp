// Isolated true-face physical SBP research operator. No production includes.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <stdexcept>
#include <vector>
#include "mesh/cubed_sphere/map.hpp"
#include "goal1_cc_coefficients.hpp"
namespace cubed_sphere { namespace research_interface {
using Pair=std::pair<int,double>;
using Rows=std::vector<std::vector<Pair>>;
struct CSR {
 std::vector<int> offsets,columns;std::vector<double> values;
 CSR()=default;
 explicit CSR(Rows &rows) {
  offsets.push_back(0);
  for(auto &r:rows){std::sort(r.begin(),r.end(),[](Pair a,Pair b){return a.first<b.first;});
   for(size_t j=0;j<r.size();){int col=r[j].first;double val=0;do{val+=r[j++].second;}while(j<r.size()&&r[j].first==col);
    if(val!=0){columns.push_back(col);values.push_back(val);}}
   offsets.push_back(columns.size());}
 }
 std::vector<double> Apply(const std::vector<double>&q)const {
  if(q.size()+1!=offsets.size())throw std::runtime_error("physical SBP size mismatch");
  std::vector<double> r(q.size());
  for(size_t i=0;i<q.size();++i)for(int a=offsets[i];a<offsets[i+1];++a)r[i]+=values[a]*q[columns[a]];
  return r;
 }
 double At(int i,int j)const {
  auto l=columns.begin()+offsets[i],r=columns.begin()+offsets[i+1];auto p=std::lower_bound(l,r,j);
  return p!=r&&*p==j?values[p-columns.begin()]:0;
 }
};
struct Line {
 int n;double h;std::vector<double> H;std::vector<std::vector<Pair>> D;
 Line(int count,double step):n(count),h(step),H(n,step),D(n) {
  if(n<16)throw std::invalid_argument("cell centered6/3 needs at least16 nodes");
  for(int i=0;i<8;++i){H[i]=h*norm[i];H[n-1-i]=h*norm[i];
   for(int j=0;j<11;++j){D[i].push_back({j,left[i][j]/h});D[n-1-i].push_back({n-1-j,-left[i][j]/h});}}
  for(int i=8;i<n-8;++i)for(auto p:std::vector<Pair>{{-3,-1./60},{-2,3./20},{-1,-3./4},{1,3./4},{2,-3./20},{3,1./60}})D[i].push_back({i+p.first,p.second/h});
  for(int i=0;i<n;++i){double sum=0;for(auto p:D[i])sum+=p.second;D[i].push_back({i,-sum});}
 }
 double Trace(int side,int i)const {int a=side?n-1-i:i;return a<8?trace[a]:0;}
};
struct Block {
 int patch,start,n[3];double lo[3],dq[3];std::array<Line,3> lines;
 Block(int p,int first,int nr,int na,const Map&m):patch(p),start(first),
  n{p<0?na:nr,na,na},lo{p<0?-m.L:0,p<0?-m.L:-1,p<0?-m.L:-1},
  dq{p<0?2*m.L/na:1./nr,p<0?2*m.L/na:2./na,p<0?2*m.L/na:2./na},
  lines{Line(n[0],dq[0]),Line(n[1],dq[1]),Line(n[2],dq[2])}{}
 int Id(const int ijk[3])const{return start+(ijk[2]*n[1]+ijk[1])*n[0]+ijk[0];}
 int Count()const{return n[0]*n[1]*n[2];}
 Point Geometry(const Map&m,const double q[3])const {
  if(patch>=0)return m.Geometry(patch,q);
  Point p{};for(int i=0;i<3;++i){p.x[i]=q[i];p.J[i][i]=1;}return p;
 }
};
inline double Mu(const Point&p){const auto &J=p.J;double det=J[0][0]*(J[1][1]*J[2][2]-J[1][2]*J[2][1])-J[0][1]*(J[1][0]*J[2][2]-J[1][2]*J[2][0])+J[0][2]*(J[1][0]*J[2][1]-J[1][1]*J[2][0]);if(!std::isfinite(det)||det==0)throw std::runtime_error("invalid physical SBP metric");return 1/std::abs(det);}
struct Face {
 int block,axis,side;std::array<double,3>x,C;double tangential;
 std::vector<Pair> trace_nodes;
};
struct Physical {
 Map map;std::vector<Block> blocks;std::vector<Point> points;
 std::vector<double> mu,W;std::array<CSR,3>D,B;
 std::array<std::vector<double>,3> metric_defect;
 double face_coordinate_defect=0,face_area_defect=0;int internal_pairs=0,external_points=0;
 Physical(int na,int nr,Map m):map(m) {
  int total=0;for(int p=-1;p<6;++p){blocks.emplace_back(p,total,nr,na,map);total+=blocks.back().Count();}
  points.resize(total);mu.resize(total);W.resize(total);std::array<Rows,3> raw,boundary;
  for(int i=0;i<3;++i){raw[i].resize(total);boundary[i].resize(total);metric_defect[i].resize(total);}
  for(auto &b:blocks)for(int k=0;k<b.n[2];++k)for(int j=0;j<b.n[1];++j)for(int i=0;i<b.n[0];++i){int ijk[3]={i,j,k};int row=b.Id(ijk);double q[3];for(int a=0;a<3;++a)q[a]=b.lo[a]+(ijk[a]+.5)*b.dq[a];points[row]=b.Geometry(map,q);mu[row]=Mu(points[row]);W[row]=mu[row]*b.lines[0].H[i]*b.lines[1].H[j]*b.lines[2].H[k];}
  for(auto &b:blocks)for(int k=0;k<b.n[2];++k)for(int j=0;j<b.n[1];++j)for(int i=0;i<b.n[0];++i){int ijk[3]={i,j,k};int row=b.Id(ijk);
   for(int a=0;a<3;++a)for(auto p:b.lines[a].D[ijk[a]]){int colijk[3]={i,j,k};colijk[a]=p.first;int col=b.Id(colijk);
    for(int d=0;d<3;++d)raw[d][row].push_back({col,.5*(points[row].J[a][d]+mu[col]/mu[row]*points[col].J[a][d])*p.second});}}
  using Key=std::array<long long,3>;std::map<Key,std::vector<Face>> faces;
  for(size_t bidx=0;bidx<blocks.size();++bidx){auto &b=blocks[bidx];
   for(int a=0;a<3;++a)for(int side=0;side<2;++side){int t1=(a+1)%3,t2=(a+2)%3;
    for(int j=0;j<b.n[t1];++j)for(int k=0;k<b.n[t2];++k){int ijk[3]{};ijk[t1]=j;ijk[t2]=k;double q[3];for(int d=0;d<3;++d)q[d]=b.lo[d]+(ijk[d]+.5)*b.dq[d];q[a]=b.lo[a]+side*b.n[a]*b.dq[a];auto p=b.Geometry(map,q);double mf=Mu(p);Face f;f.block=bidx;f.axis=a;f.side=side;f.tangential=b.lines[t1].H[j]*b.lines[t2].H[k];Key key;
     for(int d=0;d<3;++d){f.x[d]=p.x[d];f.C[d]=mf*p.J[a][d];key[d]=llround(p.x[d]*1e10);}
     for(int n=0;n<b.n[a];++n){double t=b.lines[a].Trace(side,n);if(t!=0){ijk[a]=n;f.trace_nodes.push_back({b.Id(ijk),t});}}
     if(b.patch>=0&&a==0&&side==1){AddFace(f,nullptr,raw,boundary);++external_points;}
     else faces[key].push_back(std::move(f));
    }
   }
  }
  for(auto &item:faces){auto &f=item.second;if(f.size()!=2)throw std::runtime_error("physical interface face has no unique peer");
   double area_scale=1e-15;for(int d=0;d<3;++d)area_scale=std::max({area_scale,std::abs(f[0].C[d]*f[0].tangential),std::abs(f[1].C[d]*f[1].tangential)});
   for(int d=0;d<3;++d){face_coordinate_defect=std::max(face_coordinate_defect,std::abs(f[0].x[d]-f[1].x[d]));
    double l=(f[0].side?1:-1)*f[0].C[d]*f[0].tangential,r=(f[1].side?1:-1)*f[1].C[d]*f[1].tangential;face_area_defect=std::max(face_area_defect,std::abs(l+r)/area_scale);}
   AddFace(f[0],&f[1],raw,boundary);AddFace(f[1],&f[0],raw,boundary);++internal_pairs;
  }
  if(face_coordinate_defect>1e-10||face_area_defect>1e-10)throw std::runtime_error("physical interface incompatible face geometry");
  for(int d=0;d<3;++d){for(int row=0;row<total;++row){double sum=0;for(auto p:raw[d][row])sum+=p.second;metric_defect[d][row]=sum;raw[d][row].push_back({row,-sum});}D[d]=CSR(raw[d]);B[d]=CSR(boundary[d]);}
 }
 void AddFace(const Face&f,const Face*peer,std::array<Rows,3>&raw,std::array<Rows,3>&boundary) {
  const auto&b=blocks[f.block];double sign=f.side?1:-1;
  for(auto r:f.trace_nodes){int n=(r.first-b.start);int ijk[3]={n%b.n[0],(n/b.n[0])%b.n[1],n/(b.n[0]*b.n[1])};double inv=sign*r.second/(2*mu[r.first]*b.lines[f.axis].H[ijk[f.axis]]);
   for(auto c:f.trace_nodes)for(int d=0;d<3;++d){double cc=mu[c.first]*points[c.first].J[f.axis][d];raw[d][r.first].push_back({c.first,inv*c.second*((peer?0:f.C[d])-cc)});
    if(!peer)boundary[d][r.first].push_back({c.first,sign*f.tangential*f.C[d]*r.second*c.second});}
   if(peer)for(auto c:peer->trace_nodes)for(int d=0;d<3;++d)raw[d][r.first].push_back({c.first,inv*f.C[d]*c.second});
  }
 }
 double IdentityDefect()const {
  double error=0,scale=1;
  for(int d=0;d<3;++d)for(size_t i=0;i<W.size();++i)for(int n=D[d].offsets[i];n<D[d].offsets[i+1];++n){int j=D[d].columns[n];double lhs=W[i]*D[d].values[n]+W[j]*D[d].At(j,i),rhs=B[d].At(i,j)-(i==size_t(j)?2*W[i]*metric_defect[d][i]:0);double root=std::sqrt(W[i]*W[j]);error=std::max(error,std::abs(lhs-rhs)/root);scale=std::max(scale,std::abs(W[i]*D[d].values[n])/root);}
  return error/scale;
 }
};
}}
