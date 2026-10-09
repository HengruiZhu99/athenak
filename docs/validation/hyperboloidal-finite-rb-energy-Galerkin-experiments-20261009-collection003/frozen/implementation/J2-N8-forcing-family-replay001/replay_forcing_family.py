"""Cached pointwise forcing-family gate; no kernel queries or matrix assembly."""
from pathlib import Path
import argparse, hashlib, json, shutil, time, warnings
import numpy as np
import assemble_blas as a

warnings.filterwarnings('error', category=RuntimeWarning)
P = Path(__file__).resolve().parent

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def checked(path,expected):
    actual=sha(path)
    assert actual==expected,(str(path),actual,expected)
    return {'path':str(path.resolve()),'sha256':actual,'bytes':path.stat().st_size}

def main():
    p=argparse.ArgumentParser();p.add_argument('matrix');p.add_argument('output');args=p.parse_args()
    source=P/args.matrix;dest=P/args.output;assert not dest.exists();dest.mkdir()
    started=time.monotonic();shutil.copyfile(__file__,dest/'replay_forcing_family.py')
    plan={'source_scope':'cached actual unit-W source and input maps; no kernel queries',
          'family':'every channel independently with W=1,rho,rho^2,rho^3 plus original all-channel mixed cubic',
          'forcing_construction':'at each quadrature point Phi(X)-L_actual(Phi(X)); separate Pplus incoming trace load',
          'forbidden_construction':'neither forcing nor boundary loads are defined using E,Kweak,Kstrong,SATload',
          'coefficient_error_scaled_max':2e-9,'mixed_load_readback_scaled_max':5e-11,
          'pointwise_fit_scaled_max':5e-11,'finite_required':True}
    write(dest/'plan.json',plan)
    report=json.loads((source/'report.json').read_text());J=report['J'];N=report['N'];rb=report['rb']
    assert N==8 and rb==.98 and J in (0,1,2) and report['passed_single_quadrature_algebra']
    pins=[checked(source/'operator.npz',report['operator_sha256'])]
    for f in ('report.json','reference.txt','reference-receipt.json','source/receipt.json','input/receipt.json'):
        pins.append({'path':str((source/f).resolve()),'sha256':sha(source/f),'bytes':(source/f).stat().st_size})
    for f,expected in [('assemble_blas.py','a3b140d7ef41bd5d83ffd9e03cba373f1c10f9486b07c815fd5d80c061926fd7'),
                       ('energy_coefficients.py',report['coefficient_source_sha256']),
                       ('canceled_basis_complex.py',report['canceled_source_sha256'])]:
        pins.append(checked(P/f,expected))
    data=np.load(source/'operator.npz');rho=data['radial_rho'];wr=data['radial_weights'];dirs=data['angular_directions'];wa=data['angular_weights']
    nc=len(a.layout[str(J)]);nd=nc*N;nf=4*nc+1
    refs=np.fromstring((source/'reference.txt').read_text(),sep=' ').reshape(-1,17)
    rr=json.loads((source/'reference-receipt.json').read_text());assert sha(source/'reference.txt')==rr['output_sha256']
    sr=json.loads((source/'source/receipt.json').read_text());ir=json.loads((source/'input/receipt.json').read_text())
    pins.append(checked(source/'source/output.txt',sr['output_sha256']))
    pins.append(checked(source/'input/output.bin',ir['output_sha256']))
    for receipt,mode in ((sr,'--source-batch'),(ir,'--input-binary')):
        assert receipt['exit_code']==0 and receipt['mode']==mode and receipt['executable_sha256']==report['executable_sha256']
    raw=np.fromstring((source/'source/output.txt').read_text(),sep=' ').reshape(len(rho)+1,len(a.fitdirs)+len(a.holddirs),nc,3,150)
    maps=np.memmap(source/'input/output.bin',mode='r',dtype=np.float64,shape=(len(rho)+1,len(dirs),nc,3,50))
    common,_=a.common_quadrature(N,rb);X=np.zeros((nd,nf));family=[]
    for ch,x in enumerate(a.layout[str(J)]):
        modal,_,_=a.modal(common,N,x['L'],rb*rb)
        for degree in range(4):
            col=4*ch+degree;X[ch*N:(ch+1)*N,col]=np.linalg.solve(modal,common**degree)
            family.append({'channel':ch,'name':x['name'],'L':x['L'],'polynomial_rho_degree':degree})
    X[:,-1]=data['manufactured_X'];family.append({'mixed':True,'definition':'saved original alternating per-channel cubic'})
    loads=np.zeros((nd,nf));boundary_loads=None;fits=[]
    conf=np.array([i for i,x in enumerate(a.layout[str(J)]) if x['name'] in ('alpha','metric_trace','beta','metric_STF')])
    for k,r in enumerate(np.r_[np.sqrt(rho),rb]):
        cf=a.coefficients(r,refs[k]);H=cf['H'];c=refs[k,9]
        action,fit=a.fit_source(raw[k],J,r,c,conf);fits.append(fit)
        V=np.zeros((nc,3,nd))
        for ch,x in enumerate(a.layout[str(J)]):
            jets=a.modal([r*r],N,x['L'],rb*rb)
            for derivative in range(3):V[ch,derivative,ch*N:(ch+1)*N]=jets[derivative][0]
        # Test-space field and fixed manufactured fields use actual cached point maps.
        field=np.einsum('acjf,cjd->afd',maps[k],V,optimize=False)
        U=field[:,:10];Y=np.zeros((len(dirs),20,nd));Y[:,a.qidx]=c*field[:,10:20];Y[:,a.vidx]=field[:,30:40]
        fixed_jets=np.einsum('cjd,dk->cjk',V,X,optimize=False)
        fixed=np.einsum('acjf,cjk->afk',maps[k],fixed_jets,optimize=False)
        Yfixed=np.zeros((len(dirs),20,nf));Yfixed[:,a.qidx]=c*fixed[:,10:20];Yfixed[:,a.vidx]=fixed[:,30:40]
        if k==len(rho):
            # Prescribed incoming data computed from point traces, independently of SATload.
            incoming=np.einsum('ef,afk->aek',data['Pplus'],Yfixed,optimize=False)
            boundary_loads=rb*rb*report['kin']*a.angle_work(Y,np.einsum('ef,afk->aek',H,incoming,optimize=False),wa)
            continue
        # The action is the fitted actual raw22+configuration-derivative point source.
        # It is applied to prescribed W jets before taking a pointwise source difference.
        out=np.einsum('ocj,cjk->ok',action,fixed_jets,optimize=False);base=maps[k,:,:,0,:]
        Ut=np.einsum('acf,ck->afk',base[:,:,:10],out[:nc],optimize=False)
        Qt=c*np.einsum('acf,ck->afk',base[:,:,10:20],out[:nc],optimize=False)+c*np.einsum('acf,ck->afk',np.take(maps[k],conf,axis=1)[:,:,1,10:20],out[nc:],optimize=False)
        Vt=np.einsum('acf,ck->afk',base[:,:,30:40],out[:nc],optimize=False)
        Yt=np.zeros_like(Yfixed);Yt[:,a.qidx]=Qt;Yt[:,a.vidx]=Vt
        yf=Yfixed-Yt;uf=fixed[:,:10]-Ut
        loads+=wr[k]/c*(a.angle_work(Y,np.einsum('ef,afk->aek',H,yf,optimize=False),wa)+a.angle_work(U,uf,wa))
        assert np.isfinite(loads).all()
    rhs=np.dot(data['Kweak'],X)+np.dot(data['SATload'],X)+loads+boundary_loads
    solved=np.linalg.solve(data['E'],rhs);diff=solved-X
    checks=[]
    for k,label in enumerate(family):
        physical=np.sqrt(float(np.dot(diff[:,k],np.dot(data['E'],diff[:,k]))))
        energy=np.sqrt(float(np.dot(X[:,k],np.dot(data['E'],X[:,k]))))
        checks.append({**label,'coefficient_error':a.error(solved[:,k],X[:,k]),
                       'energy_error_absolute':physical,'energy_error_scaled':physical/max(1.,energy),
                       'passed':bool(a.error(solved[:,k],X[:,k])['scaled']<=2e-9)})
    mixed={'X':a.error(X[:,-1],data['manufactured_X']),
           'pointwise_load':a.error(loads[:,-1],data['manufactured_load']),
           'incoming_load':a.error(boundary_loads[:,-1],data['manufactured_boundary_load'])}
    arrays={'family_X':X,'pointwise_manufactured_load':loads,'incoming_boundary_load':boundary_loads,'solved':solved,'difference':diff}
    for name,value in arrays.items():assert np.isfinite(value).all(),name
    np.savez_compressed(dest/'forcing-family.npz',**arrays)
    result={'J':J,'N':N,'rb':rb,'family_count':nf,'cases':checks,'mixed_readback':mixed,
            'max_coefficient_error_scaled':max(x['coefficient_error']['scaled'] for x in checks),
            'max_energy_error_scaled':max(x['energy_error_scaled'] for x in checks),
            'passed':bool(all(x['passed'] for x in checks) and max(x['scaled'] for x in mixed.values())<=5e-11),
            'fits':fits,'input_pins':pins,'source_sha256':sha(__file__),'array_sha256':sha(dest/'forcing-family.npz'),
            'seconds':time.monotonic()-started,'scope':'fixed pointwise forcing-family consistency only; no evolution, spectra or constraint preservation'}
    write(dest/'report.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','fits','input_pins')},indent=2),flush=True)
    assert result['passed'],'Preserved forcing-family failure'

if __name__=='__main__':main()
