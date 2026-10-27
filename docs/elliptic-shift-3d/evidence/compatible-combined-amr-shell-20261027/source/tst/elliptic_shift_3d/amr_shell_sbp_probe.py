"""Combined Cartesian 2:1 AMR core and collocated cubed-sphere shell candidate.

This joins the previously independent component grids. No Cartesian periodic
wrap survives, and no synthetic coarse core values enter the resulting action.
The implementation reuses the two tested block actions and replaces their
temporary boundary fluxes with one physical norm-adjoint interface coupling.
It remains a Python research operator, not a native evolution adapter.
"""
import numpy as np
from compatible_sbp import face,mortar,basis,mm
from cartesian_sbp_probe import Grid
from mapped_sbp_probe import ShellGrid,lobatto


class AMRShellGrid:
    trace=ShellGrid.trace
    normal=ShellGrid.normal

    def __init__(self,n,radial=(0.,.45,1.),half_width=.25):
        self.cart=Grid(n,2,True);self.shell=ShellGrid(n,radial,half_width)
        self.n=n;self.half_width=half_width;self.radial=radial
        self.core_count=len(self.cart.blocks);self.cells=(self.core_count+len(self.shell.blocks)-1)*n**3
        if self.cells>50000:raise ValueError('combined AMR/shell cell budget')
        scale=2*half_width;self.scale=scale
        self.coordinates=np.concatenate((scale*(self.cart.coordinates-.5),self.shell.coordinates[1:]))
        self.weights=np.concatenate((scale**3*self.cart.weights,self.shell.weights[1:]))
        self.locals=[self.cart.local]*self.core_count+self.shell.locals[1:]
        self.blocks=[(-1,b) for b in range(self.core_count)]+self.shell.blocks[1:]
        self.jacobians=[];self.cofactors=[]
        for _,width in self.cart.blocks:
            J=np.prod(scale*width);self.jacobians.append(np.full((n,n,n),J))
            self.cofactors.append(np.broadcast_to(np.diag(J/(scale*width)),(n,n,n,3,3)).copy())
        self.jacobians+=self.shell.jacobians[1:];self.cofactors+=self.shell.cofactors[1:]
        self.outer=[(b-1+self.core_count,a,s) for b,a,s in self.shell.outer]
        self.metric_identity_max=self.shell.metric_identity_max
        self.normal_mismatch=self.shell.normal_mismatch
        self.core_shell=[]
        H=self.cart.local['H'];child=face([(.5,n),(.5,n)],2)
        parent=face([(1.,n)],2);restriction,_=mortar(parent,child)
        child_to_parent=np.kron(restriction,restriction)
        sf=lobatto(n);shellface=dict(x=sf['x'],H=sf['H'],V=basis(sf['x'],2)[0],degree=2)
        P,Q=mortar(child,shellface);base_to_shell=np.kron(Q,Q);shell_to_base=np.kron(P,P)
        Hbase=np.kron(child['H'],child['H']);Hs=np.kron(sf['H'],sf['H'])
        self.transfer_adjoint_max=0.;self.transfer_composition_max=0.;self.transfer_polynomial_max=0.;self.amplification=0.
        for patch in range(6):
            axis=patch//2;side=1-patch%2;tangents=[a for a in range(3) if a!=axis]
            parts=[];weights=[];points=[];groups={}
            for b,(lo,width) in enumerate(self.cart.blocks):
                if abs(lo[axis]+side*width[axis]-side)>1e-12:continue
                start=len(parts)*n*n
                parts.append((b,axis,side,slice(start,start+n*n)))
                weights.extend(width[tangents[0]]*width[tangents[1]]*np.kron(H,H))
                points.extend(np.stack(np.meshgrid(*[lo[a]+width[a]*self.cart.local['x'] for a in tangents],indexing='ij'),axis=-1).reshape(-1,2))
                quadrant=tuple(np.floor((lo[tangents]+1e-12)/.5).astype(int))
                groups.setdefault(quadrant,[]).append((b,len(parts)-1,lo,width))
            Ha=np.array(weights);points=np.array(points);B=np.zeros((4*n*n,len(Ha)))
            for quadrant,leaves in groups.items():
                base_indices=(np.arange(n)[:,None]+quadrant[0]*n)*(2*n)+(np.arange(n)[None,:]+quadrant[1]*n)
                if len(leaves)==1:
                    b,k,lo,width=leaves[0]
                    if not np.allclose(width[tangents],.5):raise ValueError('incomplete coarse core face')
                    B[np.ix_(base_indices.ravel(),np.arange(k*n*n,(k+1)*n*n))]=np.eye(n*n)
                elif len(leaves)==4:
                    for b,k,lo,width in leaves:
                        if not np.allclose(width[tangents],.25):raise ValueError('unsupported core face refinement')
                        offset=np.rint((lo[tangents]-.5*np.array(quadrant))/.25).astype(int)
                        child_indices=(np.arange(n)[:,None]+offset[0]*n)*(2*n)+(np.arange(n)[None,:]+offset[1]*n)
                        B[np.ix_(base_indices.ravel(),np.arange(k*n*n,(k+1)*n*n))]=child_to_parent[:,child_indices.ravel()]
                else:raise ValueError('core face does not cover a complete quadrant')
            if len(groups)!=4:raise ValueError('core face does not cover the shell interface')
            prolong=B.T*Hbase[None,:]/Ha[:,None]
            M=mm(prolong,shell_to_base);N=M.T*Ha[None,:]/Hs[:,None]
            self.transfer_adjoint_max=max(self.transfer_adjoint_max,float(np.max(abs(Ha[:,None]*M-N.T*Hs[None,:]))))
            self.transfer_composition_max=max(self.transfer_composition_max,float(np.max(abs(N-mm(base_to_shell,B)))))
            sx,sy=np.meshgrid(sf['x'],sf['x'],indexing='ij')
            for i in range(3):
                for j in range(3):
                    a=points[:,0]**i*points[:,1]**j;s=(sx**i*sy**j).ravel()
                    self.transfer_polynomial_max=max(self.transfer_polynomial_max,float(np.max(abs(mm(M,s)-a))),float(np.max(abs(mm(N,a)-s))))
            self.amplification=max(self.amplification,float(np.max(np.sum(abs(M),axis=1))),float(np.max(np.sum(abs(N),axis=1))))
            swap=tangents!=[(axis+1)%3,(axis+2)%3]
            self.core_shell.append((parts,1+patch,M,N,swap,axis,side))

    def gather_core(self,values,parts):
        return np.concatenate([self.trace(values,(b,a,s)).reshape((self.n*self.n,)+values.shape[4:]) for b,a,s,_ in parts])

    def derivative(self,values,axis,sign):
        n=self.n;tail=values.shape[4:];core=values[:self.core_count]
        out=self.cart.derivative(core,axis,sign)/self.scale
        # Remove only the Cartesian periodic wraps, identified geometrically.
        for L,R,P,Q in self.cart.links[axis]:
            if not all(abs(self.cart.blocks[b][0][axis]+self.cart.blocks[b][1][axis]-1)<1e-12 for b,_,_,_ in L[1]):continue
            if not all(abs(self.cart.blocks[b][0][axis])<1e-12 for b,_,_,_ in R[1]):raise ValueError('misidentified core wrap')
            def gather(plan):
                result=np.zeros((plan[0]*n,plan[0]*n)+tail)
                for b,s,i,j in plan[1]:result[i,j]=self.cart.trace(core[b],axis,s)
                return result
            l=gather(L);r=gather(R);jl=l-self.cart.transfer(r,P);jr=r-self.cart.transfer(l,Q)
            fl=-.5*jl;fr=.5*jr
            if sign:fl+=sign*.25*(jl-self.cart.transfer(jr,P));fr+=sign*.25*(jr-self.cart.transfer(jl,Q))
            for plan,flux in [(L,fl),(R,fr)]:
                for b,s,i,j in plan[1]:out[b]-=self.cart.lift(flux[i,j],axis,s,self.scale*self.cart.blocks[b][1][axis])
        placeholder=np.concatenate((np.zeros((1,n,n,n)+tail),values[self.core_count:]))
        shell_out=self.shell.derivative(placeholder,axis,sign)
        for link in self.shell.links:
            L,R=link[:2]
            if L[0]!=0:continue
            r=self.shell.trace(placeholder,R);jl=-self.shell.transfer(r,link,True);jr=r
            normal=self.shell.normal(R)[...,axis].reshape((n,n)+(1,)*len(tail))
            flux=-.5*normal*jr
            if sign:flux+=sign*.25*abs(normal)*(jr-self.shell.transfer(jl,link,False))
            shell_out[R[0]]-=self.shell.lift(flux,R)/self.shell.jacobians[R[0]].reshape((n,n,n)+(1,)*len(tail))
        for parts,sb,M,N,swap,normal_axis,side in self.core_shell:
            if axis!=normal_axis:continue
            l=self.gather_core(values,parts);r=self.shell.trace(placeholder,(sb,0,0))
            if swap:r=np.swapaxes(r,0,1)
            r=r.reshape((n*n,)+tail)
            transfer=lambda T,v:mm(T,v.reshape(v.shape[0],-1)).reshape((T.shape[0],)+tail)
            jl=l-transfer(M,r);jr=r-transfer(N,l);normal=2*side-1
            fl=-.5*normal*jl;fr=.5*normal*jr
            if sign:fl+=sign*.25*(jl-transfer(M,jr));fr+=sign*.25*(jr-transfer(N,jl))
            for b,a,s,sl in parts:out[b]+=self.cart.lift(fl[sl].reshape((n,n)+tail),a,s,self.scale*self.cart.blocks[b][1][a])
            fr=fr.reshape((n,n)+tail)
            if swap:fr=np.swapaxes(fr,0,1)
            shell_out[sb]+=self.shell.lift(self.scale**2*fr,(sb,0,0))/self.shell.jacobians[sb].reshape((n,n,n)+(1,)*len(tail))
        return np.concatenate((out,shell_out[1:]))

    def boundary_integral(self,u,v,axis):
        return self.shell.boundary_integral(np.concatenate((np.zeros_like(u[:1]),u[self.core_count:])),np.concatenate((np.zeros_like(v[:1]),v[self.core_count:])),axis)
