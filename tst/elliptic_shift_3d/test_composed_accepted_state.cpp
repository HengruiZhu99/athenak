// Copyright(C) 2026 AthenaK contributors. BSD-3-Clause; see LICENSE.
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "mesh/cubed_sphere/composite_checkpoint.hpp"
#include "z4c/hyperboloidal/composite_evolution.hpp"
#include "composed_metric_operator_probe.hpp"
namespace h=z4c::hyperboloidal;namespace c=cubed_sphere;namespace p=composed_metric_probe;
int main(int argc,char**argv){Kokkos::initialize(argc,argv);int code=0;try{
 if(argc!=5)throw std::invalid_argument("snapshot tag csv_reference output_csv");auto s=c::CompositeCheckpoint::Read(argv[1],argv[2]);if(!s.affine_shift_derivatives||s.positive_storage!=2||s.log_halo_enrichment)throw std::invalid_argument("accepted control contract");
 h::CompositeTensorEvolution e(s.mesh,s.telegraph,s.tolerance,s.spatial_order,true,s.wormhole_chart,s.isotropic_reference,s.scri_radial_points,s.radial_interface_stencils,s.positive_storage,s.telegraph_width,s.enriched_log_derivatives,s.radial_line_smoothing,s.topology_coarsening,s.log_halo_enrichment,s.log_halo_radius,s.puncture_derivative_scheme,s.boundary_derivative_scheme,s.affine_shift_derivatives);e.state=s.fields;e.scri->values=s.boundary_fields;const auto padded=e.Padded(e.state);e.GeometryJets(e.state,padded);
 c::HermiteBoundarySampler sampler(*e.scri);const auto prep=sampler.Prepare(padded,e.scri->values,22,1);std::vector<p::U>edge;for(size_t k=0;k<e.scri->rows.size();++k){const auto q=sampler.Sample(k,prep);edge.push_back(h::UnpackComposite(e.scri->rows[k].geometry.x,q.value,q.d,q.dd,true,0,2));}
 p::Operator op(e.shift,*e.scri,e.jets,edge);const auto source=op.Source();p::V applied,r(source.size());op.Apply(s.shift_warm_start,applied,e.Boundary(padded));Real maximum=0;std::ofstream csv(argv[4]);csv<<std::setprecision(17)<<"point,Vt_x,Vt_y,Vt_z\n";for(size_t k=0;k<e.jets.size();++k){csv<<k;for(int v=0;v<3;++v){const size_t i=3*k+v;r[i]=-source[i]-applied[i];maximum=std::max(maximum,std::abs(r[i]));csv<<','<<r[i];}csv<<'\n';}
 std::ifstream reference(argv[3]);if(!reference)throw std::runtime_error("missing independent allcell CSV");std::string line;std::getline(reference,line);std::vector<std::string>head;std::stringstream hline(line);while(std::getline(hline,line,','))head.push_back(line);int cols[3];for(int v=0;v<3;++v){auto it=std::find(head.begin(),head.end(),"actual_discrete_"+std::to_string(v));if(it==head.end())throw std::runtime_error("independent audit column missing");cols[v]=it-head.begin();}
 p::V error(r.size());size_t k=0;while(std::getline(reference,line)){std::stringstream row(line);std::vector<Real>x;std::string part;while(std::getline(row,part,','))x.push_back(std::stod(part));if(k>=e.jets.size())throw std::runtime_error("audit extra rows");for(int v=0;v<3;++v)error[3*k+v]=r[3*k+v]-x[cols[v]];++k;}if(k!=e.jets.size())throw std::runtime_error("audit omitted cells");Real err=0;for(Real x:error)err=std::max(err,std::abs(x));if(err>1e-10)throw std::runtime_error("composed operator does not match independent actual discrete Frechet audit");
 std::cout<<std::setprecision(17)<<"{\"cells\":"<<k<<",\"actual_discrete_rms\":"<<std::sqrt(e.shift.Dot(r,r))<<",\"actual_discrete_point\":"<<maximum<<",\"independent_agreement_rms\":"<<std::sqrt(e.shift.Dot(error,error))<<",\"independent_agreement_max\":"<<err<<",\"source_no_RHS_subtraction\":true,\"no_solve_or_evolution\":true,\"production_adopted\":false}\n";
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';code=2;}Kokkos::finalize();return code;}
