from pathlib import Path
p=Path(__file__).with_name('export_wave.cpp');s=p.read_text();lo=s.index('void MLS(');hi=s.index('void WriteMatrix',lo)
s=s[:lo]+r'''// Spherical integer-distance shells, all ties included: no axis preference or recursive fill.
void MLS(std::vector<hyp::SphericalGhostStencil>&plan,const hyp::SphericalGhostGrid&g,const std::vector<int>&active){
 const int nx=g.n[0],ny=g.n[1];for(auto &s:plan){const int ti=s.target%nx,tj=s.target/nx%ny,tk=s.target/(nx*ny);std::map<int,std::vector<int>> shells;for(int a:active){int dx=a%nx-ti,dy=a/nx%ny-tj,dz=a/(nx*ny)-tk;int rr=dx*dx+dy*dy+dz*dz;if(rr<=64)shells[rr].push_back(a);}std::vector<int> donors;bool done=false;
 for(const auto &shell:shells){donors.insert(donors.end(),shell.second.begin(),shell.second.end());if(donors.size()<80)continue;if(donors.size()>216)break;
 double center[3]{};for(int a:donors){center[0]+=a%nx-ti;center[1]+=a/nx%ny-tj;center[2]+=a/(nx*ny)-tk;}for(double &v:center)v/=donors.size();double scale=std::sqrt(shell.first);
 double m[10][11]{};auto target=Basis(-center[0]/scale,-center[1]/scale,-center[2]/scale);
 for(int a:donors){double dx=a%nx-ti,dy=a/nx%ny-tj,dz=a/(nx*ny)-tk;auto q=Basis((dx-center[0])/scale,(dy-center[1])/scale,(dz-center[2])/scale);double w=1/std::pow(1+dx*dx+dy*dy+dz*dz,2);for(int i=0;i<10;++i)for(int j=0;j<10;++j)m[i][j]+=w*q[i]*q[j];}for(int i=0;i<10;++i)m[i][10]=target[i];
 double maxdiag=0;for(int i=0;i<10;++i)maxdiag=std::max(maxdiag,m[i][i]);bool valid=true;for(int i=0;i<10;++i){int pivot=i;for(int j=i+1;j<10;++j)if(std::abs(m[j][i])>std::abs(m[pivot][i]))pivot=j;if(std::abs(m[pivot][i])<1e-9*maxdiag){valid=false;break;}for(int j=i;j<=10;++j)std::swap(m[i][j],m[pivot][j]);double d=m[i][i];for(int j=i;j<=10;++j)m[i][j]/=d;for(int k=0;k<10;++k)if(k!=i){double f=m[k][i];for(int j=i;j<=10;++j)m[k][j]-=f*m[i][j];}}
 if(!valid)continue;s.count=donors.size();double moments[10]{};for(int b=0;b<s.count;++b){int a=donors[b];double dx=a%nx-ti,dy=a/nx%ny-tj,dz=a/(nx*ny)-tk;auto q=Basis((dx-center[0])/scale,(dy-center[1])/scale,(dz-center[2])/scale);double w=1/std::pow(1+dx*dx+dy*dy+dz*dz,2);s.donors[b]=a;s.weights[b]=0;for(int i=0;i<10;++i)s.weights[b]+=w*q[i]*m[i][10];auto raw=Basis(dx,dy,dz);for(int i=0;i<10;++i)moments[i]+=s.weights[b]*raw[i];}double error=0;for(int i=0;i<10;++i)error=std::max(error,std::abs(moments[i]-(i==0?1:0)));if(error>2e-11)continue;done=true;break;
 }if(!done)throw std::runtime_error("local quadratic MLS donor/conditioning admission failed");}
}
''' + s[hi:];p.write_text(s)
