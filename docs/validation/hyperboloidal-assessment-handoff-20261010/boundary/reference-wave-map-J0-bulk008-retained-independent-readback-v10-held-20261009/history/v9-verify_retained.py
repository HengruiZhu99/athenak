"""HELD saved-data reconstruction. No scientific import occurs before exact admission."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import time
import traceback
import warnings

HERE = Path(__file__).resolve().parent
NORMALIZATION_AUDIT = None
WEIGHTING_AUDIT = None
DERIVATIVE_AUDIT = None
WEAK_AUDIT = None
BASIS_DIVISION_AUDIT = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda s:
                      (_ for _ in ()).throw(ValueError(s)))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verify(pins):
    for path, digest in pins.items():
        if sha(path) != digest:
            raise RuntimeError('changed pinned input ' + path)


def weighting_summary():
    if WEIGHTING_AUDIT is None:
        return None
    result = WEIGHTING_AUDIT.summary()
    result['scope'] = ('Only the two broadcast products sqrt(weights)*a and '
                       'sqrt(weights)*b in bilinear(); no normalization audit mixing')
    result['contraction_scope'] = ('BLAS dgemm and its operand/reduction order are unchanged; '
                                   'the fallback here performs no contraction')
    result['weighting_sqrt_scope'] = 'The original NumPy sqrt and rounded root operands are unchanged'
    return result


def derivative_summary():
    if DERIVATIVE_AUDIT is None:
        return None
    result = DERIVATIVE_AUDIT.summary()
    result['scope'] = ('Only the second multiply (rounded c*cr)*ur in the '
                       'left-associated line353 derivative expression; separate '
                       'from normalization and bilinear weighting audits')
    result['operation_order'] = ('c*cr remains the original strict NumPy product; '
                                 'the corrected second product is added with the '
                                 'unchanged strict NumPy addition to c*c*urr')
    result['unchanged_scope'] = ('No other derivative product, division, addition, '
                                'einsum, BLAS operation, reduction, gate or threshold changed')
    return result



def weak_summary():
    if WEAK_AUDIT is None:
        return None
    result = WEAK_AUDIT.summary()
    result['scope'] = ('Only div*hy[:,qidx] in the weak derivative bilinear; '
                       'separate from normalization/derivative/weighting audits')
    result['operation_order'] = ('div=2*c/r and H action retain their original '
                                 'rounding; strict addition, all three bilinears, '
                                 'weak-sum addition and final measure multiplication unchanged')
    result['measured_scope'] = ('Primary diagnostic window289..320 has one flag at313, '
                              '422 inexact tiny subnormal products/141 unique pairs; '
                              'other grids require their unchanged readback gates')
    return result



def basis_division_summary():
    if BASIS_DIVISION_AUDIT is None:
        return None
    result = BASIS_DIVISION_AUDIT.summary()
    result['scope'] = ('Only retained-basis flat/col immediately before SVD; '
                       'separate from normalization and all three product audits')
    result['contraction_scope'] = ('This adapter performs no contraction; the original '
                                   'NumPy SVD, input layout and conditioning gate are unchanged')
    result['operand_scope'] = ('The original flat array and finite_positive_column2_norm '
                              'rounded positive col operands are unchanged')
    result['measured_scope'] = ('Completed radial diagnostic covers609..640: '
                              '84 inexact nonzero subnormal quotients only at627; '
                              'all152064components classified, no other window flags')
    return result


def scientific_readback(case, context, out):
    # Only called after the standard-library admission/byte checks in main.
    import numpy as np
    from scipy.linalg.blas import dgemm
    np.seterr(all='raise')
    warnings.filterwarnings('error', category=RuntimeWarning)
    from tiny_normalization import NormalizationArithmetic
    from fast_weighting import FastWeightingArithmetic
    from column_norm import finite_positive_column2_norm
    global NORMALIZATION_AUDIT, WEIGHTING_AUDIT, DERIVATIVE_AUDIT, WEAK_AUDIT, BASIS_DIVISION_AUDIT
    norm = NormalizationArithmetic(np)
    NORMALIZATION_AUDIT = norm
    weighting = FastWeightingArithmetic(np)
    WEIGHTING_AUDIT = weighting
    derivative_arithmetic = FastWeightingArithmetic(np)
    DERIVATIVE_AUDIT = derivative_arithmetic
    weak_arithmetic = FastWeightingArithmetic(np)
    WEAK_AUDIT = weak_arithmetic
    basis_division_arithmetic = NormalizationArithmetic(np)
    BASIS_DIVISION_AUDIT = basis_division_arithmetic
    folder = Path(case['directory'])
    report = load(folder / 'report.json')
    z = np.load(folder / 'operator.npz', allow_pickle=False)
    for key in z.files:
        if not np.isfinite(z[key]).all():
            raise ValueError('nonfinite retained array '+key)
    if report.get('passed_single_quadrature_algebra') is not True:
        raise ValueError('owner single-rule admission missing')
    checks = []
    qidx = [0, 1, 2, 8, 12, 16, 18, 7, 11, 15]
    vidx = [3, 4, 5, 9, 13, 17, 19, 6, 10, 14]
    cfg = [0, 1, 4, 6]
    rawcfg = [0, 1, 2, 3, 4, 5, 6, 18, 19, 20, 21]
    pairs = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
    nc, N, nd, rb = 8, 8, 64, .98
    if [report['J'], report['N'], report['dofs'], report['rb']] != [0, N, nd, rb]:
        raise ValueError('only declared J0/N8/rb.98 admitted')

    def mm(a, b):
        return np.einsum('ik,kj->ij', a, b, optimize=False)

    def err(a, b):
        a, b = np.asarray(a), np.asarray(b)
        d = a-b
        return {'scaled': float(np.linalg.norm(d)/max(1., np.linalg.norm(a), np.linalg.norm(b))),
                'absolute': float(np.linalg.norm(d)),
                'absolute_max': float(np.max(np.abs(d))) if d.size else 0.}

    def check(name, a, b, tolerance):
        e = err(a, b)
        checks.append(dict(name=name, tolerance=tolerance, passed=e['scaled'] <= tolerance, **e))
        return e

    def bilinear(a, b, weights):
        root = np.sqrt(weights)[:, None, None]
        aa = weighting.mul(root, a, "bilinear:left_root_times_a").reshape(-1, a.shape[-1])
        bb = weighting.mul(root, b, "bilinear:right_root_times_b").reshape(-1, b.shape[-1])
        result = dgemm(1., aa.T, bb)
        if not np.isfinite(result).all():
            raise ValueError('nonfinite independent contraction')
        return result

    def action(m, a):
        return np.einsum('ij,ajk->aik', m, a, optimize=False)

    M = np.zeros((20, 20))
    M[:8, :8] = [[0,0,0,-1,0,0,0,0], [0,0,0,2/3,4/3,0,0,-2/3],
        [0,0,0,0,0,-2,0,4/3], [-1,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,1/2,0], [-2/3,1/3,-1/2,0,0,0,2/3,0],
        [0,0,0,-4/3,-2/3,0,0,4/3], [-1,1/2,0,0,0,0,1,0]]
    for first in (8, 12):
        M[first:first+4, first:first+4] = [[0,-2,0,1],[-.5,0,.5,0],[0,0,0,1],[0,0,1,0]]
    for first in (16, 18):
        M[first:first+2, first:first+2] = [[0,-2],[-.5,0]]
    H = np.eye(20)+mm(M.T, M)
    P = (np.eye(20)+M)/2
    check('principal_involution', mm(M, M), np.eye(20), 5e-11)
    check('principal_H_symmetry', mm(H, M), mm(H, M).T, 5e-11)

    # Independent Jacobi recurrence and its derivative chain in z=2rho/B-1.
    def jacobi_jet(k, beta, x):
        p0 = np.ones_like(x); d0 = np.zeros_like(x); dd0 = d0.copy()
        if k == 0:
            return p0, d0, dd0
        p1 = (-beta+(beta+2)*x)/2
        d1 = np.full_like(x, (beta+2)/2); dd1 = np.zeros_like(x)
        for n in range(2, k+1):
            q = 2*n+beta
            den = 2*n*(n+beta)*(q-2)
            linear = (q-1)*q*(q-2)
            constant = -(q-1)*beta*beta
            back = 2*(n-1)*(n+beta-1)*q
            mult = linear*x+constant
            p = (mult*p1-back*p0)/den
            d = (linear*p1+mult*d1-back*d0)/den
            dd = (2*linear*d1+mult*dd1-back*dd0)/den
            p0,d0,dd0,p1,d1,dd1 = p1,d1,dd1,p,d,dd
        return p1, d1, dd1

    def modal_jet(rho, ell):
        rho = np.atleast_1d(rho)
        B = rb*rb; x = 2*rho/B-1
        result = np.empty((3, len(rho), N))
        for k in range(N):
            norm = math.sqrt(2*(2*k+ell+1.5)/B**(ell+1.5))
            p,d,dd = jacobi_jet(k, ell+.5, x)
            result[:, :, k] = np.stack((norm*p, norm*2*d/B, norm*4*dd/B**2))
        return result

    # Golub-Welsch for the common weight (1+z)^.5, independent of scipy.special.
    beta = .5
    diagonal = np.array([beta*beta/((2*k+beta)*(2*k+beta+2)) for k in range(N)])
    off = np.array([2/(2*k+beta)*math.sqrt(k*k*(k+beta)*(k+beta)/
                   ((2*k+beta-1)*(2*k+beta+1))) for k in range(1, N)])
    tri = np.diag(diagonal)+np.diag(off, 1)+np.diag(off, -1)
    nodes, _ = np.linalg.eigh(tri)
    common = (nodes+1)*rb*rb/2
    layouts = load(context['basis_data'])['channel_layouts']['0']
    ell = [item['L'] for item in layouts]
    T = np.zeros((nd, nd)); family = np.zeros((nd, 33))
    for channel in range(nc):
        v = modal_jet(common, ell[channel])[0]
        block = slice(channel*N, (channel+1)*N)
        T[block, block] = v
        values = np.column_stack([common**degree for degree in range(4)])
        family[block, 4*channel:4*channel+4] = np.linalg.solve(v, values)
        family[block, -1] = np.linalg.solve(v, (-1.)**channel/(channel+1)*
                                      (1+common/3-common**2/5+common**3/7))
    check('independent_nodal_from_modal', T, z['nodal_from_modal'], 2e-9)
    check('independent_fixed_family', family, z['family_X'], 2e-9)
    check('mixed_definition', family[:, -1], z['manufactured_X'], 2e-9)
    radii = z['source_coefficient_radii']
    weights = z['radial_weights']; angles = z['angular_weights']
    refs = z['source_reference_rows']; coefs = z['source_coefficients']
    directions = z['angular_directions']
    nr, na = len(radii), len(angles)
    if (coefs.shape != (nr,12,8,3) or refs.shape != (nr,17)
            or len(weights) != nr-1 or radii[-1] != rb
            or not np.array_equal(z['source_configuration_channels'], cfg)
            or not np.array_equal(z['source_coefficient_is_exact_core'], (radii <= .05).astype(np.int8))):
        raise ValueError('retained axes/configuration/core flags differ')
    if not(np.isfinite(weights).all() and (weights > 0).all()
           and np.isfinite(angles).all() and (angles > 0).all()):
        raise ValueError('positive finite physical quadrature required')
    if not(np.all(refs[:,[0,3,6,9]]>0) and np.all(radii>0)
           and np.all(np.diff(radii)>0) and np.all(refs[:,15]==1)
           and np.all(refs[:,16]==0)):
        raise ValueError('positive reference and harmonic-branch domain required')
    external_refs=np.loadtxt(folder/'reference.txt').reshape(nr,17)
    if not np.array_equal(refs,external_refs):
        raise ValueError('retained reference rows differ from pinned source export')
    check('retained_rho_radii', radii[:-1]**2, z['radial_rho'], 5e-11)
    check('angular_measure', angles.sum(), 4*math.pi, 5e-11)
    check('unit_angular_directions', np.sum(directions**2, axis=1), np.ones(na), 5e-11)
    maps = np.memmap(folder/'input/output.bin', mode='r', dtype='<f8', shape=(nr,na,nc,3,50))
    if (folder/'input/output.bin').stat().st_size != nr*na*nc*3*50*8:
        raise ValueError('input map byte count differs')
    E=np.zeros((nd,nd)); Ks=E.copy(); Kw=E.copy(); G=E.copy(); loads=np.zeros((nd,33))
    source_max = dict(raw_action=0., normalized_action=0., normalized_values=0.,
                      normalized_configuration_derivative=0., manual_normal=0.,
                      source_condition=0., normal_input_scaled=0., normal_output_scaled=0.)
    source_absolute_max={key:0. for key in source_max if key!='source_condition'}

    def source_error(key,a,b):
        e=err(a,b)
        source_max[key]=max(source_max[key],e['scaled'])
        source_absolute_max[key]=max(source_absolute_max[key],e['absolute_max'])

    def tensor(raw, first):
        a = np.zeros((3,3))
        for entry,(i,j) in enumerate(pairs):
            a[i,j]=a[j,i]=raw[first+entry]
        return a

    def screen(n):
        axis = np.array([0.,0.,1.]) if abs(n[2]) < .9 else np.array([0.,1.,0.])
        t = np.cross(axis,n); t /= np.linalg.norm(t)
        return np.stack((n,t,np.cross(n,t)))

    def normalized(raw, dr, ref, r, n):
        norm.counts['normalizations_attempted'] += 1
        alpha,ad,_,chi,cd,_,omega,_,_,c,cr,_,beta,bd,_ = ref[:15]
        m, d, mat, vec = norm.mul, norm.div, norm.mm, norm.mv
        cc=m(c,c,'c_squared'); aa=m(alpha,alpha,'alpha_squared')
        chichi=m(chi,chi,'chi_squared')
        basis=screen(n); frame=basis.copy()
        frame[0]=m(frame[0],c,'frame_normal_scale')
        framed=np.zeros((3,3)); framed[0]=m(cr,basis[0],'frame_radial_derivative')
        coframe=basis.copy(); coframe[0]=d(coframe[0],c,'coframe_normal_scale')
        coframed=np.zeros((3,3))
        coframed[0]=d(m(-cr,basis[0],'coframe_radial_derivative_product'),cc,
                       'coframe_radial_derivative_quotient')
        outer=m(n[:,None],n[None,:],'unit_radial_outer')
        g=m(chi,np.eye(3)+m(d(1,cc,'inverse_c_squared')-1,outer,'metric_anisotropy'),
            'metric_chi_scale')
        gi=d(np.eye(3)+m(cc-1,outer,'inverse_metric_anisotropy'),chi,'inverse_metric_chi_scale')
        lapse_over_c=d(alpha,c,'lapse_over_c')
        radial_k=d(bd,lapse_over_c,'radial_extrinsic')
        tangent_k=d(beta,m(r,lapse_over_c,'tangent_extrinsic_denominator'),'tangent_extrinsic')
        trace=radial_k+m(2,tangent_k,'trace_tangent_factor')
        tf_tangent=tangent_k-d(trace,3,'trace_third')
        tf_radial=d(radial_k-d(trace,3,'trace_third'),cc,'TF_radial_scale')
        ar=m(chi,m(tf_tangent,np.eye(3),'A_tangent')+
             m(tf_radial-tf_tangent,outer,'A_anisotropy'),'A_chi_scale')
        raised=mat(gi,mat(ar,gi,'A_raise_right'),'A_raise_left')
        dg=tensor(raw,1); da=tensor(raw,8); gdr=tensor(dr,1)
        h=d(mat(frame,mat(dg,frame.T,'metric_frame_right'),'metric_frame_left'),chi,'h_chi')
        hd=d(mat(framed,mat(dg,frame.T,'hd_dg_frame'),'hd_frame_derivative_left')+
             mat(frame,mat(gdr,frame.T,'hd_gdr_frame'),'hd_metric_derivative')+
             mat(frame,mat(dg,framed.T,'hd_dg_frame_derivative'),'hd_frame_derivative_right'),
             chi,'hd_chi')-d(m(h,cd,'h_times_chi_r'),chi,'h_chi_r_over_chi')
        required=float(np.sum(m(raised,dg,'reference_A_trace_tangent')))
        atrace=d(m(g,required,'A_trace_metric_product'),3,'A_trace_third')
        a=d(mat(frame,mat(da-atrace,frame.T,'A_frame_right'),'A_frame_left'),chi,'A_frame_chi')
        def chart(v):
            return np.array([v[0,0],v[0,1],v[0,2],d(v[1,1]-v[2,2],2,'STF_chart_half'),v[1,2]])
        beta_chart=vec(coframe,raw[19:22],'shift_chart')
        u=np.r_[d(raw[18],alpha,'lapse_value'),d(raw[0],chi,'chi_value'),chart(h),
                d(beta_chart,alpha,'shift_lapse_scale')]
        ud=np.r_[d(dr[18],alpha,'lapse_radial')-
                    d(m(raw[18],ad,'lapse_times_alpha_r'),aa,'lapse_alpha_r_over_alpha_squared'),
                 d(dr[0],chi,'chi_radial')-
                    d(m(raw[0],cd,'raw_chi_times_chi_r'),chichi,'chi_r_over_chi_squared'),chart(hd),
                 d(vec(coframed,raw[19:22],'shift_chart_frame_derivative')+
                   vec(coframe,dr[19:22],'shift_chart_radial'),alpha,'shift_radial_lapse_scale')-
                    d(m(beta_chart,ad,'shift_chart_times_alpha_r'),aa,'shift_alpha_r_over_alpha_squared')]
        v=np.r_[d(raw[7],omega,'physical_P_omega'),d(raw[17],omega,'Theta_omega'),chart(a),
                m(chi,vec(coframe,raw[14:17],'Lambda_chart'),'Lambda_chi')]
        normals=np.array([float(np.sum(m(gi,dg,'metric_normal'))),
                          float(np.sum(m(gi,da,'A_normal'))-required)])
        return u,ud,v,normals

    sourceplan=load(context['queries'])
    fitdirs=np.array(next(g['directions'] for g in sourceplan['groups'] if g['family']=='fit'))
    holddirs=np.array(next(g['directions'] for g in sourceplan['groups'] if g['family']=='heldout'))
    source_angles=np.concatenate((fitdirs,holddirs))
    source_stream=(folder/'source/output.txt').open()
    source_query=(folder/'source/queries.txt').open()
    input_query=(folder/'input/queries.txt').open()
    for ir,r in enumerate(radii):
        ref=refs[ir]; alpha,ad=ref[:2]; c,cr=ref[9:11]; beta,bd=ref[12:14]
        Kn=beta*np.eye(20)+alpha*M; ds_kn=c*(bd*np.eye(20)+ad*M); div=2*c/r
        gamma=mm(H,ds_kn)+div*mm(H,Kn)
        table={'A':M,'H':H,'Kn':Kn,'Hr':np.zeros_like(H),'Knr':bd*np.eye(20)+ad*M,
               'DsH':np.zeros_like(H),'DsKn':ds_kn,'Gamma':gamma}
        for key,value in table.items():
            check('coefficient_'+key+'_'+str(ir),value,z['coefficient_'+key][ir],5e-11)
        check('coefficient_scalars_'+str(ir),[r,c,cr,div,1.],z['coefficient_scalars'][ir],5e-11)
        raw=np.empty((len(source_angles),nc,3,150))
        for ia,n in enumerate(source_angles):
            for channel in range(nc):
                for derivative in range(3):
                    query=np.fromstring(source_query.readline(),sep=' ')
                    expected=np.r_[0,0,channel,0,r*n,np.eye(3)[derivative]]
                    if query.shape!=(10,) or not np.array_equal(query,expected):
                        raise ValueError('source query order/value differs')
                    row=np.fromstring(source_stream.readline(),sep=' ')
                    if row.shape!=(150,) or not np.isfinite(row).all():
                        raise ValueError('source row shape/finite')
                    raw[ia,channel,derivative]=row
                    input_dr=np.zeros(22);input_dr[rawcfg]=row[55:66]
                    source_dr=np.zeros(22);source_dr[rawcfg]=row[22:33]
                    ntrue=query[4:7]/np.linalg.norm(query[4:7])
                    u,ud,v,ni=normalized(row[33:55],input_dr,ref,r,ntrue)
                    ut,udt,vt,no=normalized(row[:22],source_dr,ref,r,ntrue)
                    source_error('normalized_values',np.r_[u,v],np.r_[row[66:76],row[96:106]])
                    source_error('normalized_configuration_derivative',ud,row[76:86])
                    source_error('normalized_action',np.r_[ut,c*udt,vt],row[116:146])
                    source_error('manual_normal',np.r_[ni,no],row[146:150])
                    source_max['normal_input_scaled']=max(source_max['normal_input_scaled'],
                        float(np.max(np.abs(ni)))/max(1.,float(np.linalg.norm(row[33:55]))))
                    source_max['normal_output_scaled']=max(source_max['normal_output_scaled'],
                        float(np.max(np.abs(no)))/max(1.,float(np.linalg.norm(row[:22]))))
                    source_absolute_max['normal_input_scaled']=max(source_absolute_max['normal_input_scaled'],float(np.max(np.abs(ni))))
                    source_absolute_max['normal_output_scaled']=max(source_absolute_max['normal_output_scaled'],float(np.max(np.abs(no))))
        basis=np.zeros((len(source_angles),33,12))
        basis[:,:22,:8]=raw[:,:,0,33:55].transpose(0,2,1)
        basis[:,22:,:8]=raw[:,:,0,55:66].transpose(0,2,1)
        basis[:,22:,8:]=raw[:,cfg,1,55:66].transpose(0,2,1)
        target=np.concatenate((raw[:,:,:,:22],raw[:,:,:,22:33]),axis=-1).transpose(0,3,1,2)
        prediction=np.einsum('afo,ocj->afcj',basis,coefs[ir],optimize=False)
        source_error('raw_action',prediction,target)
        source_normalizer=np.zeros((len(source_angles),30,12))
        source_normalizer[:,:10,:8]=raw[:,:,0,66:76].transpose(0,2,1)
        source_normalizer[:,10:20,:8]=c*raw[:,:,0,76:86].transpose(0,2,1)
        source_normalizer[:,10:20,8:]=c*raw[:,cfg,1,76:86].transpose(0,2,1)
        source_normalizer[:,20:,:8]=raw[:,:,0,96:106].transpose(0,2,1)
        normalized_prediction=np.einsum('afo,ocj->afcj',source_normalizer,coefs[ir],optimize=False)
        source_error('normalized_action',normalized_prediction,raw[:,:,:,116:146].transpose(0,3,1,2))
        flat=basis[:len(fitdirs)].reshape(-1,12); col=finite_positive_column2_norm(np,flat)
        singular=np.linalg.svd(basis_division_arithmetic.div(flat,col,"retained_basis_flat_over_col"),compute_uv=False)
        source_max['source_condition']=max(source_max['source_condition'],float(singular[0]/singular[-1]))
        for n in directions:
            for channel in range(nc):
                for derivative in range(3):
                    query=np.fromstring(input_query.readline(),sep=' ')
                    expected=np.r_[0,0,channel,0,r*n,np.eye(3)[derivative]]
                    if query.shape!=(10,) or not np.array_equal(query,expected):
                        raise ValueError('input-map query order/value differs')
        if not np.isfinite(maps[ir]).all():
            raise ValueError('nonfinite retained input map')
        fieldbasis=np.zeros((8,3,nd))
        for channel in range(nc):
            fieldbasis[channel,:,channel*N:(channel+1)*N]=modal_jet([r*r],ell[channel])[:,0]
        point=np.einsum('acjf,cjd->afd',maps[ir],fieldbasis,optimize=False)
        u,ur,urr,v,vr=[point[:,start:start+10] for start in (0,10,20,30,40)]
        y=np.zeros((na,20,nd));dy=y.copy();y[:,qidx]=c*ur;y[:,vidx]=v
        dy[:,qidx]=derivative_arithmetic.mul(c*cr,ur,"line353:rounded_c_cr_times_ur")+c*c*urr;dy[:,vidx]=c*vr
        amplitude=np.einsum('ocj,cjd->od',coefs[ir],fieldbasis,optimize=False)
        base=maps[ir,:,:,0,:]
        ut=np.einsum('acf,cd->afd',base[:,:,:10],amplitude[:8],optimize=False)
        qt=c*np.einsum('acf,cd->afd',base[:,:,10:20],amplitude[:8],optimize=False)
        qt+=c*np.einsum('acf,cd->afd',maps[ir][:,cfg,1,10:20],amplitude[8:],optimize=False)
        vt=np.einsum('acf,cd->afd',base[:,:,30:40],amplitude[:8],optimize=False)
        yt=np.zeros_like(y);yt[:,qidx]=qt;yt[:,vidx]=vt
        hy=action(H,y);hyt=action(H,yt)
        if ir==nr-1:
            kin=beta+alpha;kout=beta-alpha
            if not(kin>0 and kout<0):raise ValueError('outer characteristic signs')
            F=rb*rb*bilinear(y,action(mm(H,Kn),y),angles)
            sat=-rb*rb*kin*bilinear(y,action(mm(H,P),y),angles)
            B=(rb*np.sqrt(angles)[:,None,None]*y).reshape(-1,nd)
            weak=rb*rb*bilinear(hy[:,qidx],ut,angles)
            boundary_load=rb*rb*kin*bilinear(y,action(mm(H,P),
                np.einsum('afd,dk->afk',y,family,optimize=False)),angles)
            Kw+=weak
            break
        measure=weights[ir]/c
        E+=measure*(bilinear(y,hy,angles)+bilinear(u,u,angles))
        Ks+=measure*(bilinear(y,hyt,angles)+bilinear(u,ut,angles))
        Kw+=measure*(-bilinear(action(H,dy)[:,qidx]+weak_arithmetic.mul(div,hy[:,qidx],"line394:div_times_hy_q"),ut,angles)+
                     bilinear(hy[:,vidx],vt,angles)+bilinear(u,ut,angles))
        remainder=yt-action(Kn,dy)
        sourcepart=bilinear(y,action(H,remainder),angles)
        umass=bilinear(u,ut,angles)
        G+=measure*(sourcepart+sourcepart.T-bilinear(y,action(gamma,y),angles)+umass+umass.T)
        fy=np.einsum('afd,dk->afk',y-yt,family,optimize=False)
        fu=np.einsum('afd,dk->afk',u-ut,family,optimize=False)
        loads+=measure*(bilinear(y,action(H,fy),angles)+bilinear(u,fu,angles))
        if ir%32==0:write(out/'progress.json',{'radius_index':ir,'total_radii':nr,'source_max':source_max})
    if source_stream.read().strip() or source_query.read().strip() or input_query.read().strip():
        raise ValueError('unexpected extra source/query rows')
    source_stream.close();source_query.close();input_query.close()
    check('boundary_H_block',H,z['Hb'],5e-11)
    check('boundary_Kn_block',Kn,z['Knb'],5e-11)
    check('boundary_incoming_projector_block',P,z['Pplus'],5e-11)
    old=np.load(context['old_boundary_blocks_ONLY'],allow_pickle=False)
    for key in ('Hb','Knb','Pplus','constraint_left','gauge_left','TT_left'):
        if not np.array_equal(z[key],old[key]):
            raise ValueError('changed exact historical boundary-only block '+key)
    calculated={'E':E,'Kstrong':Ks,'Kweak':Kw,'Gvolume':G,'Fboundary':F,'SATload':sat,'B':B,
                'family_pointwise_load':loads,'family_incoming_load':boundary_load}
    for key,value in calculated.items():
        check('independently_recomputed_'+key,value,z[key],2e-8 if key in ('Kweak','Kstrong','Gvolume','Fboundary') else 2e-9)
    check('mixed_point_load',loads[:,-1],z['manufactured_load'],2e-9)
    check('mixed_incoming_load',boundary_load[:,-1],z['manufactured_boundary_load'],2e-9)
    check('weak_strong',Kw,Ks,2e-8);check('volume_trace_identity',Kw+Kw.T,F+G,2e-8)
    check('energy_symmetry',E,E.T,2e-9)
    eigen, vectors=np.linalg.eigh(E)
    if not(np.isfinite(eigen).all() and eigen[0]>0 and eigen[-1]/eigen[0]<=1e12):
        raise ValueError('energy positivity/condition gate')
    check('energy_eigensolver_backward_residual',mm(E,vectors),vectors*eigen,2e-9)
    chol=np.linalg.cholesky(E);check('energy_cholesky',mm(chol,chol.T),E,2e-9)
    check('retained_bulk_solve',mm(E,z['Jbulk']),Kw,2e-9)
    check('retained_SAT_solve',mm(E,z['Jsat']),sat,2e-9)
    check('nodal_congruence',mm(T.T,mm(z['E_nodal'],T)),E,2e-9)
    solved=np.linalg.solve(E,mm(Kw+sat,family)+loads+boundary_load)
    for column in range(33):
        label='mixed' if column==32 else layouts[column//4]['name']+'_rho_power_'+str(column%4)
        check('direct_forced_family_'+label,solved[:,column],family[:,column],2e-9)
    check('retained_family_solved',solved,z['family_solved'],2e-9)
    np.savez_compressed(out/'reconstructed.npz',**calculated,family_X=family,
                        family_solved=solved,nodal_from_modal=T)
    for key,expected in [('constraint_left',2),('gauge_left',2),('TT_left',0),('Pplus',4)]:
        rows=P if key=='Pplus' else z[key]
        t=np.einsum('ef,afd->aed',rows,B.reshape(na,20,nd),optimize=False).reshape(-1,nd)
        s=np.linalg.svd(t,compute_uv=False)
        threshold=max(1e-11,1e-10*float(s[0])) if len(s) else 1e-11
        rank=int(np.sum(s>threshold))
        checks.append({'name':'incoming_rank_'+key,'rank':rank,'expected':expected,'passed':rank==expected})
    symwork=F+2*sat;workeigen=np.linalg.eigvalsh((symwork+symwork.T)/2)
    checks.append({'name':'dissipative_boundary_work','max_eigenvalue':float(workeigen[-1]),
                   'passed':float(workeigen[-1])<=2e-9*max(1.,float(np.linalg.norm(F)),float(np.linalg.norm(sat)))})
    source_pass=all(v<=5e-11 for k,v in source_max.items() if k!='source_condition') and source_max['source_condition']<=1e4
    result={'scope':'Saved finite64 energy/operator reconstruction only; no generator spectrum, new PDE query, propagation or continuum/native acceptance',
        'case':case['name'],'checks':checks,'source_checks':source_max,
        'tiny_normalization_arithmetic':norm.summary(),
        'tiny_bilinear_weighting_arithmetic':weighting_summary(),
        'tiny_derivative_product_arithmetic':derivative_summary(),
        'tiny_weak_product_arithmetic':weak_summary(),
        'tiny_retained_basis_division_arithmetic':basis_division_summary(),
        'source_absolute_maxima':source_absolute_max,'source_passed':source_pass,
        'energy_condition':float(eigen[-1]/eigen[0]),'energy_min':float(eigen[0]),
        'root_pinned_outer_receipt':case['outer_receipt_sha256'],
        'generator_eigenvalues_computed':False,'quadrature_and_energy_symmetric_eigensolves_only':True,
        'raw_second_configuration_and_first_momentum_jets_not_exported':True,
        'passed':bool(source_pass and all(item['passed'] for item in checks))}
    write(out/'result.json',result)
    if not result['passed']:raise RuntimeError('preserved independent retained-data gate failure')
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--authorization',type=Path,required=True)
    ap.add_argument('--authorization-sha256',required=True);ap.add_argument('--case',required=True)
    ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    out=a.output.resolve()
    if out.parent!=(HERE/'attempts').resolve():raise ValueError('fresh direct attempt child required')
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();record={'completed':False,'returncode':None,'scope':'independent saved-data only'}
    protected={str(Path(__file__).resolve()):sha(__file__)}
    try:
        if os.environ.get('PYTHONDONTWRITEBYTECODE')!='1':raise RuntimeError('bytecode must be disabled')
        if sha(a.authorization)!=a.authorization_sha256:raise RuntimeError('root authorization hash differs')
        auth=load(a.authorization);recipe=load(HERE/'recipe.json');index=load(HERE/'source-index.json')
        if not(auth.get('independent_saved_matrix_readback_authorized') is True
               and auth.get('recipe_sha256')==sha(HERE/'recipe.json')
               and auth.get('source_index_sha256')==sha(HERE/'source-index.json')
               and auth.get('readback_source_sha256')==sha(__file__)
               and a.case in auth.get('cases',[]) and a.case in recipe['cases']):
            raise RuntimeError('exact separate source/recipe/case release missing')
        for key,value in recipe['environment'].items():
            if os.environ.get(key)!=value:raise RuntimeError('reviewed runtime environment differs: '+key)
        protected.update(recipe['pins']);protected[str(a.authorization.resolve())]=sha(a.authorization)
        owner_index=load(recipe['context']['owner_source_index'])
        for row in owner_index['files']+owner_index['external_inputs']:
            previous=protected.setdefault(row['path'],row['sha256'])
            if previous!=row['sha256']:raise RuntimeError('conflicting exact input pins')
        protected[str(HERE/'recipe.json')]=sha(HERE/'recipe.json')
        protected[str(HERE/'source-index.json')]=sha(HERE/'source-index.json')
        for row in index['files']:protected[row['path']]=row['sha256']
        verify(protected);write(out/'pins-before.json',protected)
        case=recipe['cases'][a.case]
        outer=load(Path(case['directory'])/'receipt.json')
        if not(outer.get('completed') is True and outer.get('returncode')==0
               and outer.get('inputs_unchanged') is True and outer.get('stage')=='bulk'):
            raise RuntimeError('successful exact-stage bulk receipt required')
        pair=load(recipe['context']['paired_receipt'])
        pair_report=load(recipe['context']['paired_report'])
        if not(pair.get('completed') is True and pair.get('returncode')==0
               and pair.get('inputs_unchanged') is True and pair.get('stage')=='paired_readback'
               and pair_report.get('passed') is True and len(pair_report['results'])==32):
            raise RuntimeError('all completed paired008 prerequisites required')
        unit = load(recipe['weighting_rounding_prerequisite']['receipt'])
        unit_recipe = load(recipe['weighting_rounding_prerequisite']['recipe'])
        if not (unit.get('passed_independent_exact_bit_units') is True
                and unit.get('unit_cases') == unit.get('expected_unit_cases') == 96
                and unit.get('inputs_unchanged') is True
                and unit_recipe.get('helper_sha256') == sha(HERE/'tiny_normalization.py')):
            raise RuntimeError('unchanged helper exact96-bit-unit prerequisite missing')
        wrapper_units = auth.get('fast_weighting_wrapper_units', {})
        wrapper_unit_path = Path(wrapper_units.get('path', 'MISSING')).resolve()
        if not (wrapper_units.get('sha256') == sha(wrapper_unit_path)):
            raise RuntimeError('exact fast weighting wrapper unit receipt missing')
        unit_fast = load(wrapper_unit_path)
        protected[str(wrapper_unit_path)] = wrapper_units['sha256']
        if not (unit_fast.get('passed_fast_weighting_wrapper_units') is True
                and unit_fast.get('inputs_unchanged') is True
                and unit_fast.get('checks') == 20
                and unit_fast.get('wrapper_source_sha256') == sha(HERE/'fast_weighting.py')
                and unit_fast.get('helper_source_sha256') == sha(HERE/'tiny_normalization.py')):
            raise RuntimeError('exact20 fast weighting wrapper units did not pass')
        column_units = auth.get('column_norm_units', {})
        column_unit_path = Path(column_units.get('path', 'MISSING')).resolve()
        if column_units.get('sha256') != sha(column_unit_path):
            raise RuntimeError('exact column2-norm unit receipt missing')
        column_unit = load(column_unit_path)
        protected[str(column_unit_path)] = column_units['sha256']
        if not (column_unit.get('passed_column_norm_units') is True
                and column_unit.get('inputs_unchanged') is True
                and column_unit.get('checks') == column_unit.get('expected_checks') == 24
                and column_unit.get('helper_source_sha256') == sha(HERE/'column_norm.py')):
            raise RuntimeError('exact24 column2-norm unit prerequisite did not pass')
        write(out/'pins-before.json',protected)
        result=scientific_readback(case,recipe['context'],out)
        record.update(completed=True,returncode=0,passed=result['passed'])
    except BaseException as exc:
        record.update(returncode=1,failure=type(exc).__name__+': '+str(exc))
        (out/'failure.txt').write_text(traceback.format_exc())
    finally:
        if NORMALIZATION_AUDIT is not None:
            write(out/'tiny-normalization-rounding.json',NORMALIZATION_AUDIT.summary())
        if WEIGHTING_AUDIT is not None:
            write(out/'tiny-bilinear-weighting-rounding.json',weighting_summary())
        if DERIVATIVE_AUDIT is not None:
            write(out/'tiny-derivative-product-rounding.json',derivative_summary())
        if WEAK_AUDIT is not None:
            write(out/'tiny-weak-product-rounding.json',weak_summary())
        if BASIS_DIVISION_AUDIT is not None:
            write(out/'tiny-retained-basis-division-rounding.json',basis_division_summary())
        try:verify(protected);record['inputs_unchanged']=True
        except BaseException as exc:record['inputs_unchanged']=False;record['post_pin_failure']=str(exc)
        record['seconds']=time.monotonic()-started;write(out/'receipt.json',record)
    print(json.dumps(record))
    if not(record['completed'] and record['inputs_unchanged']):raise SystemExit(1)


if __name__=='__main__':main()
