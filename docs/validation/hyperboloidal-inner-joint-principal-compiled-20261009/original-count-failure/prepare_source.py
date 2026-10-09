"""Source-only preparation. No generated source/checker is imported or run."""
from pathlib import Path
import ast
import hashlib
import difflib
import json
import shutil

P=Path(__file__).resolve().parent
ROOT=P.parents[2]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def pin(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)


source=ROOT/'tst/hyperboloidal/kernel_symbol.cpp'
old=source.read_text()
new=old.replace('#include "z4c/hyperboloidal/layer_gauge.hpp"',
                '#include "z4c/hyperboloidal/layer_gauge.hpp"\n#include "gauge_proposal.hpp"',1)
new=new.replace('bool physical_lapse, double matrix[20][20])',
                'const ijp::Parameters& par, double matrix[20][20])',1)
new=new.replace('  hyp::LayerGaugeParameters gauge;\n  gauge.preferred_source = false; // algebraic projection has no principal part\n  gauge.physical_trace_lapse = physical_lapse;\n','',1)
new=new.replace('hyp::AssembleGaugeInterior(', 'rwm::Assemble(')
for name in ('slicing','shift','base'):
    new=new.replace('hyp::InteriorLayerGauge(p, '+name+', gauge)',
                    'ijp::Gauge(p, '+name+', par)')
before=new[:new.index('int main() {')]
main='''int main() {
  std::cout << std::setprecision(17) << '[';
  bool first=true;
  int count=0;
  auto emit=[&](double alpha,double chi,bool oblique,const ijp::Parameters&par){
    if(!first)std::cout<<',';first=false;++count;
    const double A=alpha*alpha*chi;
    const double f=par.candidate?1+2*(1-par.W)/alpha:par.f;
    const double mu=par.candidate?par.W+(1-par.W)*par.G0/A:par.mu;
    const double ec=2*mu*mu/((1+mu)*(1+mu)),q=(4*mu-2*ec)/3;
    double matrix[20][20]{};Extract(alpha,chi,0.,oblique,par,matrix);
    std::cout<<"{\\"candidate\\":"<<par.candidate<<",\\"alpha\\":"<<alpha
      <<",\\"chi\\":"<<chi<<",\\"W\\":"<<par.W<<",\\"G0\\":"<<par.G0
      <<",\\"oblique\\":"<<oblique<<",\\"f\\":"<<f<<",\\"mu\\":"<<mu
      <<",\\"ec\\":"<<ec<<",\\"q\\":"<<q<<",\\"M\\":[";
    for(int i=0;i<20;++i){if(i)std::cout<<',';std::cout<<'[';
      for(int j=0;j<20;++j){if(j)std::cout<<',';std::cout<<matrix[i][j];}std::cout<<']';}
    std::cout<<"]}";
  };
  for(double alpha:{.05,1.,3.})for(double chi:{.1,1.})
    for(double W:{0.,.5,1.})for(double G0:{.375,.75})for(bool oblique:{false,true}){
      ijp::Parameters p;p.candidate=true;p.W=W;p.G0=G0;emit(alpha,chi,oblique,p);}
  for(double W:{0.,.5})for(double G0:{.375,.75})for(bool oblique:{false,true}){
    ijp::Parameters p;p.candidate=true;p.W=W;p.G0=G0;emit(1.,G0,oblique,p);}
  for(bool oblique:{false,true}){
    ijp::Parameters p;p.candidate=true;p.W=0.;p.G0=.375;
    const double alpha=54./29.;emit(alpha,p.G0/(2*alpha*alpha),oblique,p);}
  for(double mu:{.125,.375,.75,1.,2.,8.}){
    const double ec=2*mu*mu/((1+mu)*(1+mu)),q=(4*mu-2*ec)/3;
    for(double f:{1.,3.,q})for(bool oblique:{false,true}){
      ijp::Parameters p;p.f=f;p.mu=mu;emit(1.,1.,oblique,p);}}
  if(count!=262)throw std::runtime_error("declared source-only case count differs");
  std::cout<<"]\\n";
}
'''
new=before+main
(P/'probe.cpp').write_text(new)
(P/'probe-vs-actual-kernel-symbol.diff').write_text(''.join(difflib.unified_diff(
    old.splitlines(True),new.splitlines(True),fromfile=str(source),tofile=str(P/'probe.cpp'))))
reference=ROOT/'build-layer-research/continuum/reference-wave-map-principal-core-20261009/reference_wave_map.hpp'
if sha(reference)!='56d61c56bf37bf33591bec74c029c686c7d39f7abbe62dd090becda8e7176e28':
    raise RuntimeError('frozen wave-map helper differs')
shutil.copyfile(reference,P/'reference_wave_map.hpp')
for name in ('check_exact.py','prepare_source.py'):
    ast.parse((P/name).read_text(),filename=str(P/name))
note=P.parent/'inner-joint-principal-candidate-pencil-20261009'
external=[pin(source),pin(reference),pin(ROOT/'src/z4c/hyperboloidal/layer_gauge.hpp'),
          pin(note/'ASSESSMENT-v2.md'),pin(note/'ERRATUM.md'),pin(note/'index-v2.json')]
files=[pin(P/name) for name in ('PLAN.md','check_exact.py','gauge_proposal.hpp','probe.cpp',
       'reference_wave_map.hpp','prepare_source.py','probe-vs-actual-kernel-symbol.diff')]
write(P/'source-index.json',{'source_only':True,'execution_admitted':False,'files':files,
      'external_inputs':external,'planned_actual20_matrices':262,
      'planned_exact_scalar_cases':18,'no_numeric_CAS_import_compile_query':True,
      'readiness':'HELD independent root source review and future exact recipe/authorization; no execution',
      'limits':['Frozen constant-reference principal only, no nonflat lower-order source audit',
                'No pole/finite-frequency/evolution/puncture/BH admission']})
print(json.dumps({'index_sha256':sha(P/'source-index.json'),
                  'probe_sha256':sha(P/'probe.cpp'),'proposal_sha256':sha(P/'gauge_proposal.hpp'),
                  'exact_source_sha256':sha(P/'check_exact.py'),'no_scientific_execution':True},indent=2))
