"""Matrix-free SBP pair on a Cartesian 2:1 leaf mesh, standalone qualification.

Periodic outer faces are an engineering control, not hyperboloidal scri.
All leaf cells remain. Trace transfers enter derivative fluxes; no state reset.
"""
import numpy as np
from compatible_sbp import local,face,mortar,mm

class Grid:
    def __init__(self,n,p=2,refine=True):
        self.n=n;self.p=p;self.local=local(n,p);self.blocks=[]
        for k in range(2):
            for j in range(2):
                for i in range(2):
                    lo=np.array([i,j,k],dtype=float)/2
                    if refine and i==j==k==0:
                        for z in range(2):
                            for y in range(2):
                                for x in range(2):self.blocks.append((lo+np.array([x,y,z])/4,np.full(3,.25)))
                    else:self.blocks.append((lo,np.full(3,.5)))
        self.cells=len(self.blocks)*n**3
        if self.cells>50000:raise ValueError('Cartesian probe exceeds 50000 cells')
        q=np.stack(np.meshgrid(self.local['x'],self.local['x'],self.local['x'],indexing='ij'),axis=-1)
        self.coordinates=np.stack([lo+q*width for lo,width in self.blocks])
        H=self.local['H'];W=H[:,None,None]*H[None,:,None]*H[None,None,:]
        self.weights=np.stack([np.prod(width)*W for _,width in self.blocks])
        self.links=[[] for _ in range(3)]
        for axis in range(3):
            tangents=[a for a in range(3) if a!=axis];groups={}
            for b,(lo,width) in enumerate(self.blocks):
                for side in [0,1]:
                    plane=round(float((lo[axis]+side*width[axis])%1),12)
                    groups.setdefault(plane,{0:[],1:[]})[side].append((b,side,lo[tangents],(lo+width)[tangents]))
            for sides in groups.values():
                unused_left=set(range(len(sides[1])));unused_right=set(range(len(sides[0])))
                def overlaps(l,r):return np.all(np.minimum(l[3],r[3])-np.maximum(l[2],r[2])>1e-12)
                while unused_left:
                    li=min(unused_left);left=[li];right=[j for j in unused_right if overlaps(sides[1][li],sides[0][j])]
                    if len(right)==1:left=[i for i in unused_left if overlaps(sides[1][i],sides[0][right[0]])]
                    if not left or not right or (len(left),len(right)) not in [(1,1),(1,4),(4,1)]:raise ValueError('unsupported Cartesian interface covering')
                    L=[sides[1][i] for i in left];R=[sides[0][i] for i in right]
                    unused_left.difference_update(left);unused_right.difference_update(right)
                    low=np.min([f[2] for f in L],axis=0);high=np.max([f[3] for f in L],axis=0)
                    def plan(fs):
                        resolution=1 if len(fs)==1 else 2
                        parts=[]
                        for b,side,lo,hi in fs:
                            offset=np.rint((lo-low)/(high-low)*resolution).astype(int)
                            parts.append((b,side,slice(offset[0]*n,(offset[0]+1)*n),slice(offset[1]*n,(offset[1]+1)*n)))
                        return resolution,parts
                    lplan=plan(L);rplan=plan(R)
                    if lplan[0]==rplan[0]:P=Q=None
                    else:
                        lf=face([(1/lplan[0],n)]*lplan[0],p);rf=face([(1/rplan[0],n)]*rplan[0],p);P,Q=mortar(lf,rf)
                    self.links[axis].append((lplan,rplan,P,Q))
                if unused_right:raise ValueError('unpaired periodic face')
        self.coarse_fine_links=sum(P is not None for links in self.links for _,_,P,_ in links)

    def axis_apply(self,values,axis,operator):
        moved=np.moveaxis(values,axis,0)
        return np.moveaxis(mm(operator,moved.reshape(self.n,-1)).reshape(moved.shape),0,axis)

    def trace(self,values,axis,side):
        moved=np.moveaxis(values,axis,0)
        return np.einsum('i,i...->...',self.local['trace'][side],moved,optimize=False)

    def lift(self,values,axis,side,width):
        shape=(self.n,)+(1,)*values.ndim
        normal=(self.local['trace'][side]/(self.local['H']*width)).reshape(shape)
        return np.moveaxis(normal*values[None,...],0,axis)

    @staticmethod
    def transfer(values,P):
        if P is None:return values.copy()
        temp=np.einsum('ij,jk...->ik...',P,values,optimize=False)
        return np.einsum('jk,ik...->ij...',P,temp,optimize=False)

    def derivative(self,values,axis,sign):
        if sign not in [-1,0,1]:raise ValueError('derivative pair selector')
        key='Dplus' if sign==1 else 'Dminus' if sign==-1 else 'D'
        out=np.stack([self.axis_apply(values[b],axis,self.local[key])/width[axis] for b,(_,width) in enumerate(self.blocks)])
        def gather(plan):
            resolution,parts=plan;result=np.zeros((resolution*self.n,resolution*self.n)+values.shape[4:])
            for b,side,s,t in parts:result[s,t,...]=self.trace(values[b],axis,side)
            return result
        def scatter(plan,flux):
            for b,side,s,t in plan[1]:out[b]+=self.lift(flux[s,t,...],axis,side,self.blocks[b][1][axis])
        for L,R,P,Q in self.links[axis]:
            l=gather(L);r=gather(R);jl=l-self.transfer(r,P);jr=r-self.transfer(l,Q)
            fl=-.5*jl;fr=.5*jr
            if sign:
                fl+=sign*.25*(jl-self.transfer(jr,P));fr+=sign*.25*(jr-self.transfer(jl,Q))
            scatter(L,fl);scatter(R,fr)
        return out
