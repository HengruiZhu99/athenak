#!/usr/bin/env python3
"""Plot the actual cube-to-sphere map and a computational mesh slice."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def mapped(patch,s,u,v,L,R,stretch=0):
    s,u,v=np.broadcast_arrays(s,u,v)
    if stretch:
        s=-np.expm1(-stretch*s)/(-np.expm1(-stretch))
    a=patch//2;b=(a+1)%3;c=(a+2)%3
    d=(1-s)*L+s*R/np.sqrt(1+u*u+v*v)
    x=np.zeros(s.shape+(3,));x[...,a]=d*(1 if patch%2==0 else -1)
    x[...,b]=d*u;x[...,c]=d*v
    return x


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--bulk',type=int,default=16)
    parser.add_argument('--nr',type=int,default=272)
    parser.add_argument('--na',type=int,default=12)
    parser.add_argument('--core-half-width',type=float,default=.25)
    parser.add_argument('--radial-stretch',type=float,default=0)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    L=args.core_half_width;R=1.;assert 0<L<R/np.sqrt(3)
    assert args.radial_stretch==0 or .01<=args.radial_stretch<=6
    def mapping(patch,s,u,v,L,R):
        return mapped(patch,s,u,v,L,R,args.radial_stretch)
    colors=['#d55e00','#ee9a77','#0072b2','#64acd5','#009e73','#78c5aa']
    labels=['+X','−X','+Y','−Y','+Z','−Z']
    plt.rcParams.update({'font.size':10,'axes.titlesize':12,'savefig.dpi':180})
    fig=plt.figure(figsize=(13,6.3),layout='constrained')
    ax=fig.add_subplot(121,projection='3d');ax.set_proj_type('ortho')
    t=np.linspace(-1,1,31);U,V=np.meshgrid(t,t)
    for patch,color in enumerate(colors):
        x=mapping(patch,1,U,V,L,R)
        ax.plot_surface(x[...,0],x[...,1],x[...,2],color=color,alpha=.075,linewidth=0,shade=False)
        for fixed in np.linspace(-1,1,7):
            for u,v in [(fixed,t),(t,fixed)]:
                x=mapping(patch,1,u,v,L,R)
                ax.plot(x[...,0],x[...,1],x[...,2],color=color,lw=.55,alpha=.65)
        for u,v in [(-1,-1),(-1,1),(1,-1),(1,1),(0,0)]:
            x=mapping(patch,np.linspace(0,1,41),u,v,L,R)
            ax.plot(x[...,0],x[...,1],x[...,2],color=color,lw=.8,alpha=.7)
        x=mapping(patch,1.06,0,0,L,R)
        ax.text(*x,labels[patch],color=color,ha='center',va='center',fontsize=11)
    for axis in range(3):
        other=[a for a in range(3) if a!=axis]
        for p in [-L,L]:
            for q in [-L,L]:
                pts=np.zeros((2,3));pts[:,axis]=[-L,L];pts[:,other[0]]=p;pts[:,other[1]]=q
                ax.plot(pts[:,0],pts[:,1],pts[:,2],color='#333333',lw=1.4)
    ax.set(xlabel='x',ylabel='y',zlabel='z',xlim=(-1.1,1.1),ylim=(-1.1,1.1),zlim=(-1.1,1.1))
    ax.set_box_aspect((1,1,1));ax.view_init(elev=22,azim=35)
    ax.set_title('Six shell patches around a Cartesian core\nOuter faces tile one smooth sphere')
    bx=fig.add_subplot(122)
    for patch in range(4):
        tt=np.linspace(-1,1,301)
        x=mapping(patch,1,tt,0,L,R) if patch<2 else mapping(patch,1,0,tt,L,R)
        inner=mapping(patch,0,tt,0,L,R) if patch<2 else mapping(patch,0,0,tt,L,R)
        poly=np.concatenate([x[:,:2],inner[::-1,:2]])
        bx.fill(poly[:,0],poly[:,1],color=colors[patch],alpha=.1)
        # All angular lines; radial lines are subsampled for legibility.
        for a in np.linspace(-1,1,args.na+1):
            x=mapping(patch,np.linspace(0,1,101),a,0,L,R) if patch<2 else mapping(patch,np.linspace(0,1,101),0,a,L,R)
            bx.plot(x[:,0],x[:,1],color=colors[patch],lw=.6,alpha=.65)
        stride=max(1,args.nr//24)
        radial=sorted(set([0,args.nr,*range(0,args.nr+1,stride)]))
        for r in radial:
            x=mapping(patch,r/args.nr,tt,0,L,R) if patch<2 else mapping(patch,r/args.nr,0,tt,L,R)
            bx.plot(x[:,0],x[:,1],color=colors[patch],lw=.45,alpha=.6)
        x=mapping(patch,.45,0,0,L,R)
        bx.text(x[0],x[1],labels[patch],ha='center',va='center',color=colors[patch],fontsize=12)
    for x in np.linspace(-L,L,args.bulk+1):
        bx.plot([x,x],[-L,L],color='#555555',lw=.4,alpha=.55)
        bx.plot([-L,L],[x,x],color='#555555',lw=.4,alpha=.55)
    theta=np.linspace(0,2*np.pi,601)
    bx.plot(np.cos(theta),np.sin(theta),color='#151515',lw=1.8)
    bx.plot([-L,L,L,-L,-L],[-L,-L,L,L,-L],color='#333333',lw=1.5)
    bx.text(0,0,'Cartesian\ncore',ha='center',va='center',fontsize=10,
            bbox={'facecolor':'white','alpha':.85,'edgecolor':'none','pad':2})
    bx.annotate('Spherical outer face: r = 1\nFuture null infinity (Ω = 0)',xy=(.7,.714),xytext=(-.1,1.16),
                ha='center',arrowprops={'arrowstyle':'-','color':'#333333'},fontsize=10)
    bx.set(xlabel='x',ylabel='y',xlim=(-1.12,1.12),ylim=(-1.12,1.34))
    bx.set_aspect('equal');bx.set_title(f'z = 0 mesh slice · bulk {args.bulk}³\nShell: {args.nr} radial × {args.na} × {args.na} per patch · stretch {args.radial_stretch:.5g}')
    bx.set_xlabel(f'x\nAll angular lines; approximately every {stride}th radial line shown.',fontsize=9)
    fig.suptitle('Cubed-sphere hyperboloidal computational domain',fontsize=16)
    fig.supxlabel(f'Compactified Cartesian coordinates · core half-width {L:.8g} · no excision\nActive cell centers have Ω > 0; ghost cells beyond the sphere are not shown.',fontsize=10)
    for suffix in ['png','pdf']:
        fig.savefig(args.output/f'computational-domain.{suffix}',bbox_inches='tight')
    source=Path(__file__).resolve().parents[2]/'src/mesh/cubed_sphere/map.hpp'
    data={'map_source':str(source),'map_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'L':L,'R':R,'bulk':args.bulk,'shell_nr':args.nr,'shell_nangular':args.na,'radial_stretch':args.radial_stretch,
          'scope':'Exact spatial map; radial gridlines subsampled as labeled, outer faces at scri.'}
    (args.output/'domain-plot.json').write_text(json.dumps(data,indent=2)+'\n')


if __name__=='__main__':main()
