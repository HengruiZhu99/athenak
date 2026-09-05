"""Exact off-surface 3+1 residual against coordinate four-dimensional Ricci.

The constraint covector has arbitrary derivatives: no reduced Einstein equation
is imposed. Also check the implicit curvature response to reduction damping.
This is a continuum contraction oracle, not a production-grid stability test.
"""
import copy
from pathlib import Path
import sys

import sympy as s

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pc_gh_symbolic'))
from verify_4d_component_oracle import Jet2,make_jet,assert_zero

R=s.Rational


def four_metric(alpha,beta,gamma):
    g=[[Jet2.constant(0) for _ in range(4)] for _ in range(4)]
    g[0][0]=-alpha*alpha+sum(gamma[i][j]*beta[i]*beta[j] for i in range(3) for j in range(3))
    for i in range(3):
        g[0][i+1]=g[i+1][0]=sum(gamma[i][j]*beta[j] for j in range(3))
        for j in range(3): g[i+1][j+1]=gamma[i][j]
    return g


def coordinate_ricci(jets,axes):
    n=len(jets);g=s.Matrix(n,n,lambda i,j:jets[i][j].value);inv=g.inv()
    dg=[s.Matrix(n,n,lambda i,j:jets[i][j].grad[a]) for a in axes]
    ddg=[[s.Matrix(n,n,lambda i,j:jets[i][j].hess[a][b]) for b in axes] for a in axes]
    di=[-inv*d*inv for d in dg]
    conn=s.MutableDenseNDimArray.zeros(n,n,n);dconn=s.MutableDenseNDimArray.zeros(n,n,n,n)
    for r in range(n):
        for a in range(n):
            for b in range(n):
                first=s.Matrix([(dg[a][k,b]+dg[b][k,a]-dg[k][a,b])/2 for k in range(n)])
                conn[r,a,b]=(inv*first)[r]
                for d in range(n):
                    df=s.Matrix([(ddg[d][a][k,b]+ddg[d][b][k,a]-ddg[d][k][a,b])/2 for k in range(n)])
                    dconn[d,r,a,b]=(di[d]*first+inv*df)[r]
    ricci=s.Matrix(n,n,lambda a,b:
        sum(dconn[r,r,a,b]-dconn[b,r,a,r] for r in range(n))+
        sum(conn[r,a,b]*conn[k,r,k]-conn[k,a,r]*conn[r,b,k] for r in range(n) for k in range(n)))
    return ricci,conn,g,inv


def main():
    alpha=make_jet(R(7,5),lambda a:2*a-3,lambda a,b:(a+1)*(b+2)-4)
    beta=[make_jet(R(i-1,7),lambda a,i=i:(i+1)*(a+2)-3,
                   lambda a,b,i=i:2*i+a-b) for i in range(3)]
    lower=s.Matrix([[1,0,0],[R(1,5),1,0],[R(-1,7),R(2,9),1]])
    G=lower*lower.T;Gu=G.inv()
    gamma=[[None]*3 for _ in range(3)]
    for i in range(3):
        for j in range(i,3):
            gamma[i][j]=gamma[j][i]=make_jet(G[i,j],lambda a,i=i,j=j:(a+1)*(i+2)-j-1,
                lambda a,b,i=i,j=j:(a+i+2)*(b+j+1)-3)
    ricci4,conn4,g4,gu4=coordinate_ricci(four_metric(alpha,beta,gamma),range(4))
    ricci3,conn3,_,_=coordinate_ricci(gamma,range(1,4))
    lapse=alpha.value;shift=s.Matrix([b.value for b in beta])
    n=s.Matrix([1/lapse,*[-b/lapse for b in shift]])
    nc=s.Matrix([-lapse,0,0,0])
    c=s.Matrix([R(2,17),R(-1,19),R(3,23),R(1,29)])
    dc=s.Matrix(4,4,lambda a,b:R((a+2)*(b+1)-5,37))
    cp=(n.T*c)[0];ci=c[1:4,0];kappa=R(2,5)
    direct=s.Matrix(4,4,lambda a,b:ricci4[a,b]-(dc[a,b]+dc[b,a])/2+
        sum(conn4[r,a,b]*c[r] for r in range(4))+
        kappa*((nc[a]*c[b]+nc[b]*c[a])/2-g4[a,b]*cp/2))
    assert any(x!=0 for x in direct), 'The fixture must remain off the reduced Einstein surface'
    dG=[s.Matrix(3,3,lambda i,j:gamma[i][j].grad[a]) for a in range(4)]
    ddG=[[s.Matrix(3,3,lambda i,j:gamma[i][j].hess[a][b]) for b in range(4)] for a in range(4)]
    k=s.zeros(3);dk=[s.zeros(3) for _ in range(4)]
    for i in range(3):
        for j in range(3):
            numerator=dG[0][i,j]-sum(shift[r]*dG[r+1][i,j]+
                G[i,r]*beta[r].grad[j+1]+G[j,r]*beta[r].grad[i+1] for r in range(3))
            k[i,j]=-numerator/(2*lapse)
            for a in range(4):
                derivative=ddG[a][0][i,j]-sum(beta[r].grad[a]*dG[r+1][i,j]+
                    shift[r]*ddG[a][r+1][i,j]+dG[a][i,r]*beta[r].grad[j+1]+
                    G[i,r]*beta[r].hess[a][j+1]+dG[a][j,r]*beta[r].grad[i+1]+
                    G[j,r]*beta[r].hess[a][i+1] for r in range(3))
                dk[a][i,j]=-derivative/(2*lapse)+numerator*alpha.grad[a]/(2*lapse**2)
    trace=s.trace(Gu*k)
    dtrace=[s.trace(-Gu*dG[a]*Gu*k+Gu*dk[a]) for a in range(4)]
    ntrace=(dtrace[0]-sum(shift[r]*dtrace[r+1] for r in range(3)))/lapse
    hess=s.Matrix(3,3,lambda i,j:alpha.hess[i+1][j+1]-
                  sum(conn3[r,i,j]*alpha.grad[r+1] for r in range(3)))
    dcp=[]
    for a in range(4):
        dn=s.Matrix([-alpha.grad[a]/lapse**2,*[-beta[r].grad[a]/lapse+
            shift[r]*alpha.grad[a]/lapse**2 for r in range(3)]])
        dcp.append((dn.T*c)[0]+sum(n[r]*dc[a,r] for r in range(4)))
    ncp=(dcp[0]-sum(shift[r]*dcp[r+1] for r in range(3)))/lapse
    acceleration=s.Matrix([alpha.grad[i+1]/lapse for i in range(3)])
    predicted_nn=ntrace+s.trace(Gu*hess)/lapse-s.trace(Gu*k*Gu*k)-ncp+(ci.T*Gu*acceleration)[0]-kappa*cp/2
    assert_zero('off-surface normal-normal residual',(n.T*direct*n)[0]-predicted_nn)
    for i in range(3):
        momentum=sum(Gu[j,r]*(dk[j+1][r,i]-sum(conn3[l,j,r]*k[l,i]+
            conn3[l,j,i]*k[r,l] for l in range(3))) for j in range(3) for r in range(3))-dtrace[i+1]
        lie_c=dc[0,i+1]-sum(shift[r]*dc[r+1,i+1]+ci[r]*beta[r].grad[i+1] for r in range(3))
        predicted_ni=-momentum-(lie_c/lapse+dcp[i+1]-acceleration[i]*cp)/2-(k*Gu*ci)[i]-kappa*ci[i]/2
        assert_zero(f'off-surface mixed residual {i}',sum(n[a]*direct[a,i+1] for a in range(4))-predicted_ni)
        for j in range(i,3):
            lie_k=dk[0][i,j]-sum(shift[r]*dk[r+1][i,j]+k[i,r]*beta[r].grad[j+1]+
                                k[j,r]*beta[r].grad[i+1] for r in range(3))
            sym_dc=(dc[i+1,j+1]+dc[j+1,i+1])/2-sum(conn3[r,i,j]*ci[r] for r in range(3))
            predicted_ij=ricci3[i,j]+trace*k[i,j]-2*(k*Gu*k)[i,j]-lie_k/lapse-hess[i,j]/lapse-sym_dc-k[i,j]*cp-kappa*G[i,j]*cp/2
            assert_zero(f'off-surface spatial residual {i},{j}',direct[i+1,j+1]-predicted_ij)
    print('PASS: all 10 arbitrary off-surface residual projections equal coordinate 4D Ricci exactly')

    # At fixed state/first jets, changing reduction rate changes gamma_tt only
    # through B_t: delta gamma_tt = -2 delta_lambda gamma_{k(i} b_{j)}^k.
    # Arbitrary lapse/shift accelerations cancel out of Ricci independently.
    rate=R(3,7);b=s.Matrix(3,3,lambda i,j:R(2*i-j+1,31));strain=(b*G+G*b.T)/2
    changed=copy.deepcopy(gamma)
    for i in range(3):
        for j in range(i,3): changed[i][j].hess[0][0]-=2*rate*strain[i,j]
    changed_alpha=copy.deepcopy(alpha);changed_alpha.hess[0][0]+=R(3,11)
    changed_beta=copy.deepcopy(beta)
    for i in range(3): changed_beta[i].hess[0][0]+=R(i+1,13)
    changed_ricci,*_=coordinate_ricci(four_metric(changed_alpha,changed_beta,changed),range(4))
    difference=changed_ricci-ricci4
    assert_zero('implicit rate response nn',(n.T*difference*n)[0]-rate*s.trace(b)/lapse**2)
    for i in range(3):
        assert_zero(f'implicit rate response ni {i}',sum(n[a]*difference[a,i+1] for a in range(4)))
        for j in range(i,3):
            assert_zero(f'implicit rate response ij {i},{j}',difference[i+1,j+1]+rate*strain[i,j]/lapse**2)
    print('PASS: exact homogeneous B-reduction curvature response to a changed finite damping rate')


if __name__=='__main__': main()
