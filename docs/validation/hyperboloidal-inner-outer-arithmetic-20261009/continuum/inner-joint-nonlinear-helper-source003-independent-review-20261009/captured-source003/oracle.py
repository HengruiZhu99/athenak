#!/usr/bin/env python3
"""HELD independent MP source/dual readback; admission guards precede mpmath."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def input_guard(rows):
    for row in rows:
        if hash_file(row['path']) != row['sha256']:
            raise RuntimeError('Protected input drift: ' + row['path'])


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--authorization', required=True)
    parser.add_argument('--recipe', required=True)
    parser.add_argument('--attempt', required=True)
    args = parser.parse_args()
    if sys.flags.optimize or not sys.dont_write_bytecode:
        raise RuntimeError('Unoptimized Python and bytecode-off are required')
    auth, recipe = read(args.authorization), read(args.recipe)
    if not auth.get('local_nonlinear_helper_execution_admitted'):
        raise RuntimeError('No exact root local execution authorization')
    if auth['recipe_sha256'] != hash_file(args.recipe):
        raise RuntimeError('Recipe authorization mismatch')
    if auth['source_index_sha256'] != hash_file(recipe['source_index']):
        raise RuntimeError('Source index authorization mismatch')
    input_guard(read(recipe['input_pins']))
    principal = read(recipe['principal_receipt'])
    if not (principal['completed'] and principal['passed'] and
            principal['returncode'] == 0 and principal['inputs_unchanged']):
        raise RuntimeError('Principal outer receipt did not succeed')
    for name in ('OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS'):
        if os.environ.get(name) != '1':
            raise RuntimeError('Thread guard: ' + name)
    # Scientific import is deliberately below every admission guard.
    sys.path.insert(0,recipe['mpmath_parent'])
    import mpmath as mp
    if str(Path(mp.__file__).resolve())!=recipe['mpmath_init']:
        raise RuntimeError('Unexpected mpmath import location')

    class AD:
        def __init__(self, v=0, d=0):
            self.v, self.d = mp.mpf(v), mp.mpf(d)
        def __add__(self, b):
            b = ad(b); return AD(self.v+b.v, self.d+b.d)
        __radd__ = __add__
        def __neg__(self): return AD(-self.v, -self.d)
        def __sub__(self, b): return self+-ad(b)
        def __rsub__(self, b): return ad(b)+-self
        def __mul__(self, b):
            b = ad(b); return AD(self.v*b.v, self.d*b.v+self.v*b.d)
        __rmul__ = __mul__
        def __truediv__(self, b):
            b = ad(b)
            return AD(self.v/b.v, (self.d*b.v-self.v*b.d)/(b.v*b.v))
        def __rtruediv__(self, b): return ad(b)/self
        def __pow__(self, n):
            if n == 0: return AD(1)
            if n < 0: return AD(1)/(self**(-n))
            out = AD(1)
            for unused in range(n): out = out*self
            return out

    def ad(x): return x if isinstance(x, AD) else AD(x)
    def atom(x):
        # JSON float -> mp.mpf(float) retains its exact binary64 value.
        if isinstance(x, list): return AD(x[0], x[1])
        return AD(x)
    def field(u):
        return {k: atom(u[k]) if k in ('alpha', 'chi', 'P', 'Theta')
                else [[atom(v) for v in row] for row in u[k]]
                if k in ('g', 'beta_d', 'A')
                else [[[atom(v) for v in row] for row in plane] for plane in u[k]]
                if k == 'g_d' else [atom(v) for v in u[k]] for k in u}
    def inverse(g):
        det = g[0][0]*(g[1][1]*g[2][2]-g[1][2]*g[2][1])-g[0][1]*(g[1][0]*g[2][2]-g[1][2]*g[2][0])+g[0][2]*(g[1][0]*g[2][1]-g[1][1]*g[2][0])
        inv = []
        for i in range(3):
            row = []
            for j in range(3):
                ii=[k for k in range(3) if k!=j];jj=[k for k in range(3) if k!=i]
                minor=g[ii[0]][jj[0]]*g[ii[1]][jj[1]]-g[ii[0]][jj[1]]*g[ii[1]][jj[0]]
                row.append(((-1)**(i+j))*minor/det)
            inv.append(row)
        return inv,det
    def coefficient(a,chi,W,G0):
        A=a*a*chi; X=(1-W)*G0
        return (X+W*A)/(X+(1+W)*A)
    def raw(u,conn,od):
        a,chi,b=u['alpha'],u['chi'],u['beta']; gi,unused=inverse(u['g'])
        V=[[a*a*chi*gi[i][j] for j in range(3)] for i in range(3)]
        L=[[V[i][j]-b[i]*b[j] for j in range(3)] for i in range(3)]
        R=[sum(b[j]*u['alpha_d'][j] for j in range(3))]
        S=[-a*a*u['P']-a*sum(b[j]*od[j] for j in range(3))-a*sum(L[j][k]*conn[0][j][k] for j in range(3) for k in range(3))]
        for i in range(3):
            R.append(a*a*chi*u['Lambda'][i]+sum(b[j]*u['beta_d'][j][i]+a*a*gi[i][j]*u['chi_d'][j]/2-a*chi*gi[i][j]*u['alpha_d'][j] for j in range(3)))
            S.append(sum(2*V[i][j]*od[j] for j in range(3))-sum(L[j][k]*(conn[i+1][j][k]+b[i]*conn[0][j][k]) for j in range(3) for k in range(3)))
        return R,S
    def target(row):
        u,h=field(row['input']),field(row['reference'])
        od=[atom(v) for v in row['Omega_d']]
        conn=[[[atom(v) for v in line] for line in plane] for plane in row['connection']]
        R,S=raw(u,conn,od); Rh,Sh=raw(h,conn,od)
        a,chi=u['alpha'],u['chi']; W,G0=ad(row['W']),ad(row['G0'])
        k=coefficient(a,chi,W,G0); A=a*a*chi; B=(1-W)*G0+W*A
        regular=[R[0]-a/h['alpha']*Rh[0]]+[R[i]-Rh[i] for i in range(1,4)]
        pole=[S[0]-a/h['alpha']*Sh[0]-2*(1-W)*a*(u['P']-atom(row['Phat']))]+[S[i]-Sh[i] for i in range(1,4)]
        gi,unused=inverse(u['g'])
        for i in range(3):
            regular[i+1]+=(B-A)*(u['Lambda'][i]-h['Lambda'][i])+sum(a*a*(2*k*k-ad(.5))*gi[i][j]*(u['chi_d'][j]-chi*h['chi_d'][j]/h['chi']) for j in range(3))
        parts=regular+pole; omega=atom(row['Omega'])
        rhs=[regular[i]+pole[i]/omega for i in range(4)]
        reference_literal=[Rh[i]+Sh[i]/omega for i in range(4)]
        return parts,rhs,reference_literal
    def core_target(row):
        u=field(row['input']);a,chi=u['alpha'],u['chi'];gi,unused=inverse(u['g'])
        k=coefficient(a,chi,ad(0),ad(row['G0']))
        out=[sum(u['beta'][j]*u['alpha_d'][j] for j in range(3))-a*(a+2)*u['P']]
        for i in range(3):
            out.append(ad(row['G0'])*u['Lambda'][i]+sum(u['beta'][j]*u['beta_d'][j][i]-a*chi*gi[i][j]*u['alpha_d'][j]+2*a*a*k*k*gi[i][j]*u['chi_d'][j] for j in range(3)))
        return out
    def err(x,y): return abs(x-y)/max(mp.mpf(1),abs(x),abs(y))
    maxima={};counts={};failures=[];fd=[];normals=[];literal=[];rounding=[]
    def update(name,x,y,tolerance,relative=False):
        error=abs(x-y)/abs(y) if relative and y else err(x,y)
        maxima[name]=max(maxima.get(name,mp.mpf(0)),error)
        if not (mp.isfinite(x) and mp.isfinite(y) and error<=tolerance):
            failures.append({'check':name,'error':str(error),'got':str(x),'target':str(y)})
        return error
    def require(condition,message):
        if not condition: failures.append({'check':message})
    attempt=Path(args.attempt)
    for mode in recipe['modes']:
        path=attempt/(mode+'.jsonl')
        for line_number,line in enumerate(path.read_text().splitlines(),1):
            row=json.loads(line);kind=row['kind'];counts[kind]=counts.get(kind,0)+1
            label={'mode':mode,'line':line_number,'kind':kind}
            if kind.startswith('invalid-'):
                require(row['rejected'],str(label)+' invalid accepted');continue
            if kind.startswith('coefficient'):
                with mp.workdps(110):
                    a,chi=atom(row['alpha']),atom(row['chi']);W,G=ad(row['W']),ad(row['G0'])
                    truth=coefficient(a,chi,W,G);native=atom(row['k'])
                    require(row['valid'],str(label)+' invalid coefficient')
                    update('coefficient-value',native.v,truth.v,mp.mpf('2e-14'))
                    require(0<=native.v<=1,str(label)+' coefficient bounds')
                    if row['W']==1:require(float(native.v)==.5,str(label)+' exact W1 coefficient')
                    if float(truth.v)==0:require(float(native.v)==0,str(label)+' correctly rounded coefficient zero')
                    rounding.append({**label,'native_hex':float(native.v).hex(),'oracle_rounded_hex':float(truth.v).hex(),'exact_target':str(truth.v)})
                    if kind=='coefficient-dual':
                        update('coefficient-dual',native.d,truth.d,mp.mpf('2e-10'))
                        if row['alpha'][0]==1 and row['chi'][0]==1 and row['W']==0 and row['G0']==.375:
                            require(abs(atom(row['value_only_control']).d-truth.d)>mp.mpf('1e-3'),'value-only negative control not detected')
                continue
            contrast=row.get('family','') in recipe['high_contrast_families']
            precisions=(240,280) if contrast else (80,110)
            if kind=='core-witness':
                # No O(1e297) unfactored subtraction at240/280: the exact core
                # G0*Lambda / suppressed-gradient target is independently used.
                with mp.workdps(110):
                    expected=core_target(row)
                    require(row['valid'] and row['assembled'],str(label)+' core witness invalid')
                    for i in range(4):
                        native=atom(row['rhs'][i]);relative=expected[i].v!=0
                        update('core-witness-relative' if relative else 'core-witness-zero',native.v,expected[i].v,mp.mpf('2e-10') if relative else 0,relative=relative)
                        if relative:require(abs(expected[i].v)>=mp.mpf(sys.float_info.min),'core witness target unexpectedly subnormal')
                continue
            targets=[]
            for precision in precisions:
                with mp.workdps(precision):
                    targets.append(target(row))
            parts,rhs,ref_literal=targets[-1]
            with mp.workdps(precisions[-1]):
                precision_tolerance=mp.mpf('1e-180' if contrast else '1e-65')
                for arrays0,arrays1 in zip(targets[0],targets[1]):
                    for a,b in zip(arrays0,arrays1):
                        update('MP-precision-contrast' if contrast else 'MP-precision-ordinary',a.v,b.v,precision_tolerance)
                        update('MP-precision-dual',a.d,b.d,precision_tolerance)
                if kind=='nonrepresentable':
                    nonrepresentable=any(abs(v.v)>sys.float_info.max for v in parts+rhs)
                    require(nonrepresentable or (row['valid'] and row['assembled']),str(label)+' finite target rejected')
                    if nonrepresentable:require(not row['valid'] or not row['assembled'],str(label)+' nonrepresentable accepted')
                    literal.append({**label,'nonrepresentable':nonrepresentable,'parts':[str(v.v) for v in parts],'rhs':[str(v.v) for v in rhs]})
                    continue
                require(row['valid'] and row['assembled'],str(label)+' valid source rejected')
                for native,truth in zip(row['parts'],parts):
                    update('source-parts',atom(native).v,truth.v,mp.mpf('2e-10'))
                    if kind in ('dual','principal'):update('source-dual-parts',atom(native).d,truth.d,mp.mpf('2e-10'))
                for native,truth in zip(row['rhs'],rhs):
                    update('source-rhs',atom(native).v,truth.v,mp.mpf('2e-10'))
                    if kind in ('dual','principal'):update('source-dual-rhs',atom(native).d,truth.d,mp.mpf('2e-10'))
                literal.append({**label,'reference_literal_rhs':[str(v.v) for v in ref_literal]})
                if kind=='reference':
                    require(all(v[0]==0 for v in row['parts']),str(label)+' reference parts nonzero')
                    for a in range(4):
                        for i in range(4):
                            for j in range(4):
                                cs=atom(row['connection'][a][i-1][j-1]).v if i and j else mp.mpf(0)
                                for name in ('ADM','embedding'):
                                    update('connection-'+name,cs,atom(row[name][a][i][j]).v,mp.mpf('2e-11'))
                if row['W']==1:
                    require(row['outer_value_bitwise'],str(label)+' W1 native bit mismatch')
                    require(all(float(x[component]).hex()==float(y[component]).hex() for x,y in zip(row['parts'],row['baseline_parts']) for component in range(2)),str(label)+' W1 full dual bit mismatch')
                    require(row['arithmetic']['coefficient_calls']==0,str(label)+' W1 evaluated inner k')
                if kind=='source' and row['W']==0 and row['Omega'][0]==1:
                    exact=core_target(row)
                    for native,truth in zip(row['rhs'],exact):update('independent-core',atom(native).v,truth.v,mp.mpf('2e-10'))
                if kind=='principal':
                    h=field(row['reference']);gi,unused=inverse(h['g']);a,chi=h['alpha'],h['chi'];W=ad(row['W']);k=coefficient(a,chi,W,ad(row['G0']));B=(1-W)*ad(row['G0'])+W*a*a*chi
                    expected=[AD(0) for unused in range(8)];col=row['column']
                    if col==0:expected[4]=-a*(a+2*(1-W))
                    elif col<=3:expected[col]=B
                    elif col<=6:
                        j=col-4;expected[0]=h['beta'][j]
                        for i in range(3):expected[i+1]=-a*chi*gi[i][j]
                    elif col<=9:
                        j=col-7
                        for i in range(3):expected[i+1]=2*a*a*k*k*gi[i][j]
                    else:expected[(col-10)%3+1]=h['beta'][(col-10)//3]
                    for native,truth in zip(row['parts'],expected):update('proposal-principal-coefficients',atom(native).d,truth.v,mp.mpf('2e-11'))
                if kind=='dual':
                    for index in recipe['geometry_raw22_indices']:
                        require(all(float(row['actual22'][index][c]).hex()==float(row['baseline22'][index][c]).hex() for c in range(2)),str(label)+' changed geometry row '+str(index))
                    for j,index in enumerate(recipe['gauge_raw22_indices']):update('actual22-gauge-dual',atom(row['actual22'][index]).d,rhs[j].d,mp.mpf('2e-10'))
                    errors=[]
                    for level in row['FD']:
                        levelmax=mp.mpf(0);epsilon=mp.mpf(level['epsilon'])
                        for index in recipe['gauge_raw22_indices']:
                            approximation=(atom(level['plus22'][index]).v-atom(level['minus22'][index]).v)/(2*epsilon)
                            levelmax=max(levelmax,err(approximation,atom(row['actual22'][index]).d))
                        errors.append(levelmax)
                    require(errors[-1]<=mp.mpf('5e-7'),str(label)+' final FD failed')
                    require(errors[0]>=2*errors[-1] or max(errors)<=mp.mpf('5e-9'),str(label)+' FD convergence/floor unclassified')
                    fd.append({**label,'column':row['column'],'errors':[str(v) for v in errors]})
                    inv,det=inverse(field(row['input'])['g']);aa=field(row['input'])['A'];trace=sum(inv[i][j]*aa[i][j] for i in range(3) for j in range(3))
                    normals.append({**label,'det_value_minus1':str(det.v-1),'det_tangent':str(det.d),'A_trace':str(trace.v),'A_trace_tangent':str(trace.d),'scope':'recorded input normals; no extra undeclared numerical threshold'})
    require(counts==recipe['expected_record_counts'],'exact declared query counts')
    input_guard(read(recipe['input_pins']))
    report={'passed':not failures,'counts':counts,'maxima':{k:str(v) for k,v in maxima.items()},'failures':failures,'FD_sequences':fd,'input_normals':normals,'literal_reference_or_nonrepresentable':literal,'coefficient_rounding':rounding,'source_only_inputs_unchanged':True,'scope':'finite-Omega local nonlinear helper only; no stability/native/BH/global admission'}
    out=attempt/'oracle-report.json'
    if out.exists():raise RuntimeError('Refuse overwrite oracle report')
    out.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'passed':report['passed'],'counts':counts,'failures':len(failures),'report':str(out)},sort_keys=True))
    if failures:raise SystemExit(1)


if __name__=='__main__':main()
