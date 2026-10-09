"""Verify release/pins; prepare only fresh immutable input copies and adapters."""
from pathlib import Path
import hashlib
import json
import shutil

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
BASIS = ROOT/'build-layer-research/continuum/total-j-harmonic-basis/immutable-Cartesian-total-J-basis-20261009'
HELD = P.parent/'total-j-cartesian-control-held'
RELEASE = P.parent/'total-j-root-local-release-20261009.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy(source, destination):
    if destination.exists():
        assert sha(source) == sha(destination), str(destination)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


assert sha(RELEASE) == '0c4d415d79e6bb50acfe6f0d6262c4bcda263a537b98221282931cede87c742c'
assert sha(BASIS/'index.json') == '414f241e986e0c46b7d94820fb060b704166d973284854c53d2e8c64ee6c489e'
for row in json.loads((BASIS/'index.json').read_text())['files']:
    assert sha(BASIS/row['path']) == row['sha256'], row['path']
for row in json.loads((HELD/'source-pins.json').read_text())['files']:
    assert sha(Path(row['absolute_path'])) == row['sha256'], row['absolute_path']
for name in ('RECIPE.md', 'API-CONTRACT.md', 'admission.json', 'preparation-receipt.json', 'source-pins.json'):
    copy(HELD/name, P/'inputs/held'/name)
copy(RELEASE, P/'inputs/root-local-release.json')
copy(BASIS/'index.json', P/'inputs/basis-index.json')
for name in ('total_j_basis.hpp', 'reference_conversion.hpp', 'basis-data.json'):
    copy(BASIS/name, P/'inputs/basis'/name)
frozen = ROOT/'build-layer-research/continuum/covariant-z4-candidate/immutable-C1-stiffness-20261009/full20.cpp'
assert sha(frozen) == '7f73bbe8f59060ab59d9060ceb5e37e12c4dbb2e92acba03ec31bcea5bf6414f'
copy(frozen, P/'inputs/frozen-full20.cpp')
source = frozen.read_text()
start = source.index('hyp::LayerPoint<D> Reference(')
end = source.index('std::array<D,20> Evaluate(')
helper = ROOT/'build-layer-research/continuum/discrete-bianchi/immutable-discrete-bulk-20261009/dual_helpers.hpp'
text = '// Exact Reference/Gauge extraction; only norm=true C0 is used.\n'
text += '#include "' + str(helper) + '"\n#include "z4c/hyperboloidal/layer_gauge.hpp"\n'
text += source[start:end]
dest = P/'baseline_dual.hpp'
if dest.exists():
    assert dest.read_text() == text
else:
    dest.write_text(text)

# Independent all-m polynomial input, retaining the already-applied CG phase.
data = json.loads((BASIS/'basis-data.json').read_text())
terms, records = [], []
for record in data['records']:
    start = len(terms)
    terms.extend(record['terms'])
    records.append((record['J'], record['m'], record['spin'], record['L'], start, len(terms)-start))
header = '#include "inputs/basis/total_j_basis.hpp"\nnamespace allm {\n'
header += 'struct Term {int c,p[3];double re,im;};\nstruct Record{int J,m,s,L,start,count;};\n'
header += 'inline constexpr Term terms[]={\n'
for term in terms:
    header += '  {%d,{%d,%d,%d},%.17g,%.17g},\n' % (term['component'], *term['powers'], term['coefficient_real'], term['coefficient_imag'])
header += '};\ninline constexpr Record records[]={\n'
header += ''.join('  {'+','.join(map(str,row))+'},\n' for row in records)
header += '''};
inline totalj::Field<double> Evaluate(int J,int m,int spin,int L,int phase,
    const std::array<double,3>&x,const totalj::WJet<double>&w) {
  const Record* r=nullptr;for(const auto&q:records)if(q.J==J&&q.m==m&&q.s==spin&&q.L==L)r=&q;
  if(!r)throw std::runtime_error("missing all-m record");
  totalj::Field<double> out;out.components=spin==0?1:(spin==1?3:9);
  for(int k=r->start;k<r->start+r->count;++k) {
    const auto&t=terms[k];const totalj::Term a{t.c,t.p[0],t.p[1],t.p[2],phase?t.im:t.re};
    const double pv=totalj::MonomialDerivative(a,x,{0,0,0});
    auto&v=out.component[t.c];v.value+=pv*w.value;
    for(int i=0;i<3;++i){std::array<int,3> oi{};++oi[i];
      const double pi=totalj::MonomialDerivative(a,x,oi),wi=2*x[i]*w.rho_d;
      v.d[i]+=pi*w.value+pv*wi;
      for(int j=0;j<3;++j){auto oij=oi;++oij[j];std::array<int,3> oj{};++oj[j];
        const double pj=totalj::MonomialDerivative(a,x,oj),pij=totalj::MonomialDerivative(a,x,oij);
        v.dd[i][j]+=pij*w.value+pi*2*x[j]*w.rho_d+pj*wi
          +pv*((i==j?2:0)*w.rho_d+4*x[i]*x[j]*w.rho_dd);}
    }
  }return out;
}
struct Channel{int kind,spin,L;};
'''
kind = {'alpha':0,'metric_trace':1,'P':2,'Theta_phys':3,'beta':4,'Lambda':5,'metric_STF':6,'independent_A':7}
for J in range(3):
    header += 'inline constexpr Channel channels%d[]={\n'%J
    for c in data['channel_layouts'][str(J)]:
        header += '  {%d,%d,%d},\n'%(kind[c['name']],c['spin'],c['L'])
    header += '};\n'
header += 'inline Channel ChannelAt(int J,int c){if(J==0)return channels0[c];if(J==1)return channels1[c];return channels2[c];}\n}\n'
dest=P/'all_m_data.hpp'
if dest.exists():assert dest.read_text()==header
else:dest.write_text(header)
pins = [{'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted((P/'inputs').rglob('*')) if p.is_file()]
result={'release_and_basis_verified':True,'held_preparation_copied_unchanged':True,'input_pins':pins,'generated_baseline_dual_sha256':sha(P/'baseline_dual.hpp'),'all_m_data_sha256':sha(P/'all_m_data.hpp'),'scientific_compile_performed_by_prepare':False}
f=P/'preparation.json'
if f.exists():assert json.loads(f.read_text())==result
else:f.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'prepared':True,'inputs':len(pins),'release_verified':True},indent=2))
