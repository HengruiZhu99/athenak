"""First actual finite-rb energy-Galerkin matrix; no spectrum or evolution."""
from pathlib import Path
import argparse,hashlib,json,math,os,shutil,subprocess,time,warnings
import numpy as np
from scipy.special import roots_jacobi,roots_legendre,eval_jacobi
from energy_coefficients import coefficients,A1,HD,screen_rotation
warnings.filterwarnings('error',category=RuntimeWarning)
P=Path(__file__).resolve().parent
ROOT=P.parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
mm=lambda a,b:np.einsum('ik,kj->ij',a,b,optimize=False)
mv=lambda a,b:np.einsum('ij,j->i',a,b,optimize=False)
dot=lambda a,b:np.einsum('i,i->',a,b,optimize=False)
qidx=np.array([0,1,2,8,12,16,18,7,11,15]);vidx=np.array([3,4,5,9,13,17,19,6,10,14])
layout=json.loads((P/'inputs/basis/basis-data.json').read_text())['channel_layouts']
oldplan=json.loads((ROOT/'build-layer-research/boundary/total-j-local-angular-20261009/immutable-C0-spatialnorm-total-J-local-angular-20261009/angular-plan.json').read_text())
fitdirs=np.array(oldplan['fit_directions']);holddirs=np.array(oldplan['heldout_directions'])
core=np.load(ROOT/'build-layer-research/boundary/total-j-flat-core-envelope-20261009/immutable-total-J-flat-core-envelope-20261009/core-envelope-blocks.npz')

def write_json(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def error(a,b):
    r=a-b
    return {'scaled':float(np.linalg.norm(r)/max(1.,np.linalg.norm(a),np.linalg.norm(b))),
        'absolute':float(np.linalg.norm(r)),'absolute_max':float(np.max(np.abs(r)))}
def common_quadrature(Q,rb):
    x,w=roots_jacobi(Q,0,.5);B=rb*rb
    return (x+1)*B/2,w*(B/2)**1.5/2
def quadrature(Q,rb):
    # One declared refinement: identical integral, panels only at fixed cutoffs.
    edges=np.array([0.,.05,.45,.85,.90,.95,rb])**2
    nodes=[];weights=[]
    for j,(a,b) in enumerate(zip(edges[:-1],edges[1:])):
        if j==0:
            x,w=roots_jacobi(Q,0,.5);rho=(x+1)*b/2;weight=w*(b/2)**1.5/2
        else:
            x,w=roots_legendre(Q);rho=(a+b)/2+(b-a)*x/2;weight=w*(b-a)/4*np.sqrt(rho)
        nodes.extend(rho);weights.extend(weight)
    return np.array(nodes),np.array(weights)
def angular(nt,np_):
    z,w=roots_legendre(nt);phi=np.arange(np_)*2*np.pi/np_
    directions=np.array([[math.sqrt(1-a*a)*math.cos(p),math.sqrt(1-a*a)*math.sin(p),a] for a in z for p in phi])
    weights=np.repeat(w,np_)*2*np.pi/np_
    return directions,weights
def modal(rho,N,L,B):
    z=2*np.asarray(rho)/B-1;norm=np.sqrt(2*(2*np.arange(N)+L+1.5)/B**(L+1.5))
    v=np.array([norm[k]*eval_jacobi(k,0,L+.5,z) for k in range(N)]).T
    d=np.zeros_like(v);dd=np.zeros_like(v)
    for k in range(1,N):d[:,k]=norm[k]*(k+L+1.5)/B*eval_jacobi(k-1,1,L+1.5,z)
    for k in range(2,N):dd[:,k]=norm[k]*(k+L+1.5)*(k+L+2.5)/B**2*eval_jacobi(k-2,2,L+2.5,z)
    return v,d,dd
def run_queries(folder,mode,radii,dirs,J,exe,binary=False):
    folder.mkdir(parents=True,exist_ok=True);query=folder/'queries.txt';out=folder/('output.bin' if binary else 'output.txt')
    nc=len(layout[str(J)]);rows=len(radii)*len(dirs)*nc*3
    if not query.exists():
        with query.open('w') as f:
            for r in radii:
                for n in dirs:
                    for c in range(nc):
                        for w in ((1,0,0),(0,1,0),(0,0,1)):
                            f.write(' '.join(format(v,'.17g') for v in (J,0,c,0,*(r*n),*w))+'\n')
    expected={'query_sha256':sha(query),'executable_sha256':sha(exe),'mode':mode,'rows':rows,'binary':binary}
    receipt=folder/'receipt.json'
    if out.exists():
        r=json.loads(receipt.read_text());assert all(r[k]==v for k,v in expected.items());assert sha(out)==r['output_sha256']
    else:
        started=time.monotonic()
        with query.open('rb') as fi,out.open('wb') as fo,(folder/'stderr').open('wb') as fe:
            run=subprocess.run([str(exe),mode],stdin=fi,stdout=fo,stderr=fe)
        r={**expected,'exit_code':run.returncode,'seconds':time.monotonic()-started,'output_sha256':sha(out),'stderr_bytes':(folder/'stderr').stat().st_size}
        write_json(receipt,r);assert run.returncode==0 and r['stderr_bytes']==0,r
    if binary:
        assert out.stat().st_size==rows*50*8
        return np.memmap(out,dtype=np.float64,mode='r',shape=(len(radii),len(dirs),nc,3,50))
    a=np.fromstring(out.read_text(),sep=' ');assert a.size==rows*150
    return a.reshape(len(radii),len(dirs),nc,3,150)
def reference_rows(radii,exe,folder):
    payload=''.join(format(r,'.17g')+'\n' for r in radii)
    # Cached reference queries are independent of polynomial degree.
    if (folder/'reference.txt').exists():
        receipt=json.loads((folder/'reference-receipt.json').read_text())
        assert receipt['exit_code']==0 and receipt['query_sha256']==hashlib.sha256(payload.encode()).hexdigest()
        assert receipt['output_sha256']==sha(folder/'reference.txt') and receipt['executable_sha256']==sha(exe)
        rows=np.fromstring((folder/'reference.txt').read_text(),sep=' ').reshape(-1,17)
        assert len(rows)==len(radii) and np.isfinite(rows).all()
        return rows
    start=time.monotonic();run=subprocess.run([str(exe),'--reference-batch'],input=payload,text=True,capture_output=True)
    (folder/'reference.txt').write_text(run.stdout);(folder/'reference.stderr').write_text(run.stderr)
    write_json(folder/'reference-receipt.json',{'command':[str(exe),'--reference-batch'],'exit_code':run.returncode,'seconds':time.monotonic()-start,'query_sha256':hashlib.sha256(payload.encode()).hexdigest(),'output_sha256':sha(folder/'reference.txt'),'executable_sha256':sha(exe)})
    assert run.returncode==0 and not run.stderr;rows=np.fromstring(run.stdout,sep=' ').reshape(-1,17);assert len(rows)==len(radii)
    return rows
def fit_source(data,J,r,c,conf):
    nc=len(layout[str(J)]);nf=len(fitdirs)
    def extended(a):
        b=np.zeros((len(a),33,nc+len(conf)))
        b[:,:22,:nc]=a[:,:,0,33:55].transpose(0,2,1)
        b[:,22:,:nc]=a[:,:,0,55:66].transpose(0,2,1)
        b[:,22:,nc:]=a[:,conf,1,55:66].transpose(0,2,1)
        f=np.concatenate((a[:,:,:,:22],a[:,:,:,22:33]),axis=-1).transpose(0,3,1,2)
        return b.reshape(-1,nc+len(conf)),f.reshape(-1,nc*3)
    B,F=extended(data[:nf]);scale=np.linalg.norm(B,axis=0);u,s,vh=np.linalg.svd(B/scale,full_matrices=False)
    assert (scale>0).all();cond=float(s[0]/s[-1]);assert cond<1e4
    A=mm(vh.T,np.einsum('ki,kj->ij',u,F,optimize=False)/s[:,None])/scale[:,None]
    fit=error(mm(B,A),F);Bh,Fh=extended(data[nf:]);held=error(mm(Bh,A),Fh)
    if r<=.05:
        blocks=core['J%d'%J];powers=np.array([1,r*r,r**4]);dP=np.array([0,1,2*r*r])
        C=np.einsum('dpij,p->dij',blocks,powers,optimize=False)
        Cd=np.einsum('dpij,p->dij',blocks,dP,optimize=False)
        exact=np.zeros_like(A);exact[:nc]=C.transpose(1,2,0).reshape(nc,nc*3)
        for row,ch in enumerate(conf):
            exact[nc+row]=np.stack((Cd[0,ch],C[0,ch]+Cd[1,ch],C[1,ch]+Cd[2,ch]),axis=-1).reshape(nc*3)
        core_error=error(mm(Bh,exact),Fh);assert core_error['scaled']<=5e-11
        A=exact
    else:core_error=None
    a=data[nf:];inp=a[:,:,:,66:116];source_map=np.zeros((len(a),30,nc+len(conf)))
    source_map[:,:10,:nc]=inp[:,:,0,:10].transpose(0,2,1)
    source_map[:,10:20,:nc]=c*inp[:,:,0,10:20].transpose(0,2,1)
    source_map[:,10:20,nc:]=c*inp[:,conf,1,10:20].transpose(0,2,1)
    source_map[:,20:,:nc]=inp[:,:,0,30:40].transpose(0,2,1)
    predicted=np.einsum('afo,oj->afj',source_map,A,optimize=False)
    actual=a[:,:,:,116:146].transpose(0,3,1,2).reshape(len(a),30,nc*3)
    normalized=error(predicted,actual)
    assert max(fit['scaled'],held['scaled'],normalized['scaled'])<=5e-11
    return A.reshape(nc+len(conf),nc,3),{'r':float(r),'scaled_condition':cond,'fit':fit,'heldout':held,'source_normalization':normalized,'exact_core_action':core_error}
def angle_work(a,b,w):
    # Identical angular bilinear form, BLAS matrix contraction only.
    return np.dot(a.reshape(-1,a.shape[-1]).T,(w[:,None,None]*b).reshape(-1,b.shape[-1]))

def assemble(J,N,rb,Q,nt,np_,name,exe):
    dest=P/name;assert not (dest/'operator.npz').exists(),'New output prefix required';dest.mkdir(exist_ok=True)
    for f in ('assemble_degree.py','energy_coefficients.py','canceled_basis_complex.py'):shutil.copyfile(P/f,dest/f)
    rho,wr=quadrature(Q,rb);radii=np.sqrt(rho);dirs,wa=angular(nt,np_);nc=len(layout[str(J)]);nd=nc*N
    conf=np.array([i for i,x in enumerate(layout[str(J)]) if x['name'] in ('alpha','metric_trace','beta','metric_STF')]);assert len(conf)==nc//2
    source=run_queries(dest/'source','--source-batch',np.r_[radii,rb],np.r_[fitdirs,holddirs],J,exe)
    maps=run_queries(dest/'input','--input-binary',np.r_[radii,rb],dirs,J,exe,True)
    refs=reference_rows(np.r_[radii,rb],exe,dest)
    E=np.zeros((nd,nd));Ks=E.copy();Kw=E.copy();G=E.copy();forcing=np.zeros(nd);fit_reports=[];coef_reports=[]
    common,_=common_quadrature(N,rb);T=np.zeros((nd,nd));X=np.zeros(nd)
    for ch,x in enumerate(layout[str(J)]):
        V,_,_=modal(common,N,x['L'],rb*rb);T[ch*N:(ch+1)*N,ch*N:(ch+1)*N]=V
        fixed=(-1.)**ch/(ch+1)*(1+common/3-common**2/5+common**3/7)
        X[ch*N:(ch+1)*N]=np.linalg.solve(V,fixed)
    boundary=None
    for ir,r in enumerate(np.r_[radii,rb]):
        ref=refs[ir];cf=coefficients(r,ref);H=cf['H'];Kn=cf['Kn'];c=ref[9];cr=ref[10];
        eig=np.linalg.eigvalsh(H);sym=error(mm(H,Kn),mm(H,Kn).T)
        rotations=max(error(mm(screen_rotation(a).T,mm(H,screen_rotation(a))),H)['scaled'] for a in (.37,.81))
        assert eig[0]>0 and max(sym['scaled'],rotations)<=5e-11
        coef_reports.append({'r':float(r),'H_min':float(eig[0]),'H_max':float(eig[-1]),'H_condition':float(eig[-1]/eig[0]),'principal_symmetry':sym,'screen_rotation':rotations,'zeta':cf['zeta']})
        A,fr=fit_source(source[ir],J,r,c,conf);fit_reports.append(fr)
        V=np.zeros((nc,3,nd))
        for ch,x in enumerate(layout[str(J)]):
            modaljet=modal([r*r],N,x['L'],rb*rb)
            for derivative in range(3):V[ch,derivative,ch*N:(ch+1)*N]=modaljet[derivative][0]
        field=np.einsum('acjf,cjd->afd',maps[ir],V,optimize=False)
        U=field[:,:10];Ur=field[:,10:20];Urr=field[:,20:30];mom=field[:,30:40];momr=field[:,40:50]
        Y=np.zeros((len(dirs),20,nd));DsY=Y.copy();Y[:,qidx]=c*Ur;Y[:,vidx]=mom;DsY[:,qidx]=c*(cr*Ur+c*Urr);DsY[:,vidx]=c*momr
        out=np.einsum('ocj,cjd->od',A,V,optimize=False);base=maps[ir,:,:,0,:]
        Ut=np.einsum('acf,cd->afd',base[:,:,:10],out[:nc],optimize=False)
        Qt=c*np.einsum('acf,cd->afd',base[:,:,10:20],out[:nc],optimize=False)+c*np.einsum('acf,cd->afd',np.take(maps[ir],conf,axis=1)[:,:,1,10:20],out[nc:],optimize=False)
        Vt=np.einsum('acf,cd->afd',base[:,:,30:40],out[:nc],optimize=False)
        Yt=np.zeros_like(Y);Yt[:,qidx]=Qt;Yt[:,vidx]=Vt
        HY=np.einsum('ef,afd->aed',H,Y,optimize=False);HYt=np.einsum('ef,afd->aed',H,Yt,optimize=False)
        DsHY=np.einsum('ef,afd->aed',cf['DsH'],Y,optimize=False)+np.einsum('ef,afd->aed',H,DsY,optimize=False)
        if ir==len(radii):
            weak_boundary=rb*rb*angle_work(HY[:,qidx],Ut,wa)
            F=rb*rb*angle_work(Y,np.einsum('ef,afd->aed',mm(H,Kn),Y,optimize=False),wa)
            kin=ref[12]+ref[0];kout=ref[12]-ref[0];plus=(np.eye(20)+A1)/2
            Sat=-rb*rb*kin*angle_work(Y,np.einsum('ef,afd->aed',mm(H,plus),Y,optimize=False),wa)
            B=(rb*np.sqrt(wa)[:,None,None]*Y).reshape(-1,nd)
            boundary_data=np.einsum('ef,afd,d->ae',plus,Y,X,optimize=False)
            boundary_load=rb*rb*kin*np.einsum('afi,a,af->i',Y,wa,np.einsum('ef,af->ae',H,boundary_data,optimize=False),optimize=False)
            boundary={'weak':weak_boundary,'F':F,'Sat':Sat,'B':B,'Hb':H,'Knb':Kn,'Pplus':plus,'kin':kin,'kout':kout,'boundary_load':boundary_load};continue
        weight=wr[ir]/c
        E+=weight*(angle_work(Y,HY,wa)+angle_work(U,U,wa))
        Ks+=weight*(angle_work(Y,HYt,wa)+angle_work(U,Ut,wa))
        Kw+=weight*(-angle_work(DsHY[:,qidx]+cf['div_s']*HY[:,qidx],Ut,wa)+angle_work(HY[:,vidx],Vt,wa)+angle_work(U,Ut,wa))
        R=Yt-np.einsum('ef,afd->aed',Kn,DsY,optimize=False);HR=np.einsum('ef,afd->aed',H,R,optimize=False)
        G+=weight*(angle_work(Y,HR,wa)+angle_work(R,HY,wa)-angle_work(Y,np.einsum('ef,afd->aed',cf['Gamma'],Y,optimize=False),wa)+angle_work(U,Ut,wa)+angle_work(Ut,U,wa))
        yf=np.einsum('afd,d->af',Y-Yt,X,optimize=False);uf=np.einsum('afd,d->af',U-Ut,X,optimize=False)
        forcing+=weight*(np.einsum('afi,a,af->i',Y,wa,np.einsum('ef,af->ae',H,yf,optimize=False),optimize=False)+np.einsum('afi,a,af->i',U,wa,uf,optimize=False))
    Kw+=boundary['weak'];F=boundary['F'];Sat=boundary['Sat'];chol=np.linalg.cholesky(E);ev=np.linalg.eigvalsh(E);invT=np.linalg.inv(T);En=mm(invT.T,mm(E,invT))
    Jbulk=np.linalg.solve(E,Kw);Jsat=np.linalg.solve(E,Sat)
    expected_rhs=mv(Kw,X)+mv(Sat,X)+forcing+boundary['boundary_load'];solved=np.linalg.solve(E,expected_rhs)
    checks={'E_symmetry':error(E,E.T),'G_symmetry':error(G,G.T),'weak_strong':error(Kw,Ks),'bulk_identity':error(Kw+Kw.T,F+G),'bulk_solve':error(mm(E,Jbulk),Kw),'SAT_solve':error(mm(E,Jsat),Sat),'modal_congruence':error(mm(T.T,mm(En,T)),E),'manufactured_forced_rhs':error(solved,X)}
    algebra_pass=all(checks[k]['scaled']<=2e-9 for k in ('E_symmetry','G_symmetry','bulk_solve','SAT_solve','modal_congruence','manufactured_forced_rhs'))
    projected_trace=np.einsum('ef,afd->aed',boundary['Pplus'],boundary['B'].reshape(len(dirs),20,nd),optimize=False).reshape(-1,nd)
    rank_s=np.linalg.svd(projected_trace,compute_uv=False)
    rank=int(np.sum(rank_s>rank_s[0]*1e-10))
    arrays={'E':E,'Kweak':Kw,'Kstrong':Ks,'Gvolume':G,'Fboundary':F,'SATload':Sat,'Jbulk':Jbulk,'Jsat':Jsat,'B':boundary['B'],'Hb':boundary['Hb'],'Knb':boundary['Knb'],'Pplus':boundary['Pplus'],'nodal_from_modal':T,'E_nodal':En,'manufactured_X':X,'manufactured_load':forcing,'manufactured_boundary_load':boundary['boundary_load'],'energy_cholesky':chol,'radial_rho':rho,'radial_weights':wr,'angular_directions':dirs,'angular_weights':wa,'incoming_singular_values':rank_s}
    np.savez_compressed(dest/'operator.npz',**arrays)
    report={'J':J,'N':N,'rb':rb,'Q_per_panel':Q,'Q_total':len(radii),'quadrature_panels_r':[0.,.05,.45,.85,.90,.95,rb],'angular_rule':[nt,np_],'dofs':nd,'dof_order':'channel-major modal degree; nodal_from_modal maps modal to common-rho nodal coefficients','kin':float(boundary['kin']),'kout':float(boundary['kout']),'expected_incoming_rank':nc//2,'observed_incoming_rank':rank,'energy_modal_min':float(ev[0]),'energy_modal_condition':float(ev[-1]/ev[0]),'energy_nodal_condition':float(np.linalg.cond(En)),'energy_diagonally_scaled_nodal_condition':float(np.linalg.cond(En/np.sqrt(np.outer(np.diag(En),np.diag(En))))),'checks':checks,'fit_reports':fit_reports,'coefficient_reports':coef_reports,'operator_sha256':sha(dest/'operator.npz'),'executable':str(exe),'executable_sha256':sha(exe),'source_sha256':sha(__file__),'coefficient_source_sha256':sha(P/'energy_coefficients.py'),'canceled_source_sha256':sha(P/'canceled_basis_complex.py'),'scope':'Actual finite-rb normal-energy Galerkin operator gate only; no generator spectra/propagation/CPBC/scri result','thresholds':{'algebra':2e-9,'bulk_weak_strong_and_quadrature':2e-8,'energy_condition_max':1e12},'passed_single_quadrature_algebra':bool(algebra_pass and ev[-1]/ev[0]<=1e12 and rank==nc//2 and checks['weak_strong']['scaled']<=2e-8 and checks['bulk_identity']['scaled']<=2e-8)}
    write_json(dest/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('fit_reports','coefficient_reports')},indent=2),flush=True)
    assert report['passed_single_quadrature_algebra'],'Failed saved single-quadrature gate; preserve and inspect before refinement'
    return arrays,report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--J',type=int,default=0);parser.add_argument('--N',type=int,default=8);parser.add_argument('--rb',type=float,default=.98);parser.add_argument('--Q',type=int,default=64);parser.add_argument('--theta',type=int,default=12);parser.add_argument('--phi',type=int,default=24);parser.add_argument('--name',required=True);args=parser.parse_args()
    assert args.J in (0,1,2) and args.N in (12,16) and args.rb==.98,'Released degree-extension controls only'
    assemble(args.J,args.N,args.rb,args.Q,args.theta,args.phi,args.name,P/'radial-bridge-release')
