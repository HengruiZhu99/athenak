"""Audit and plot native leaf layouts. Optionally overlay an archived radiation shell."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from layout import leaves, make, verify

# name, exterior distance, radial root blocks, cells in rho and z
LAYOUTS = [
    ("small36-b36", 36, 2, 36, 36),
    ("wide288-b36", 288, 2, 36, 36),
    ("wide288-72x36", 288, 1, 72, 36),
    ("wide288-b72", 288, 2, 72, 72),
    ("wide288-constant-exterior", 288, 16, 36, 36),
]


def audit(leaves_with_dx, extent):
    area = 0
    for i, (r0,r1,z0,z1,dx) in enumerate(leaves_with_dx):
        assert 0 <= r0 < r1 <= extent and 0 <= z0 < z1 <= extent
        area += (r1-r0)*(z1-z0)
        for a,b,c,d,h in leaves_with_dx[:i]:
            overlap_r, overlap_z = min(b,r1)-max(a,r0), min(d,z1)-max(c,z0)
            assert overlap_r <= 0 or overlap_z <= 0
            if min(overlap_r,overlap_z)==0 and max(overlap_r,overlap_z)>0:
                assert max(dx,h) <= 2*min(dx,h)
    assert area == extent**2


def shell_spacings(frame, layout):
    """Sample the proposed initial spacing at each energy-weighted archive cell."""
    from diagnostics import read_binary
    field = read_binary(frame)
    adm = read_binary(frame.with_name(frame.name.replace(".tmunu.",".adm.")))
    np.testing.assert_array_equal(field["mb_geometry"], adm["mb_geometry"])
    assert field["cycle"] == adm["cycle"]
    records=[]
    for b, (r0,r1,z0,z1,*_) in enumerate(field["mb_geometry"]):
        e = field["mb_data"]["tmunu_E"][b][0].astype(float)
        nz,nr = e.shape
        dr,dz = (r1-r0)/nr,(z1-z0)/nz
        rho,z = np.meshgrid(r0+(np.arange(nr)+.5)*dr, abs(z0+(np.arange(nz)+.5)*dz))
        g=[]
        for suffix in ("xx","xy","xz","yy","yz","zz"):
            v=adm["mb_data"]["adm_g"+suffix][b][0].astype(float)
            g.append((v[:-1,:-1]+v[1:,:-1]+v[:-1,1:]+v[1:,1:])/4)
        xx,xy,xz,yy,yz,zz=g
        w = rho*dr*dz*e*np.sqrt(xx*yy*zz+2*xy*xz*yz-xx*yz**2-yy*xz**2-zz*xy**2)
        dx=np.full(e.shape, np.nan)
        for a,b1,c,d,h in layout:
            dx[(a<=rho)&(rho<b1)&(c<=z)&(z<d)]=h
        assert np.isfinite(dx).all() and np.isfinite(w).all()
        records.append(np.stack((w, rho*np.sqrt(zz), dx, np.full_like(dx,dr))).reshape(4,-1))
    w,r,dx,archive_dx=np.concatenate(records,axis=1)
    order=np.argsort(r)
    p90=r[order[np.searchsorted(np.cumsum(w[order]), .9*w.sum())]]
    mask=r>=p90
    return dict(cycle=int(field["cycle"]), code_time=float(field["time"]),
                projected_outer_decile_dx=float(np.average(dx[mask], weights=w[mask])),
                archived_outer_decile_dx=float(np.average(archive_dx[mask], weights=w[mask])),
                energy_fraction_dx_gt_half=float(w[dx>.5].sum()/w.sum()))


def draw(meshes, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Rectangle
    fig, axes=plt.subplots(1,3,figsize=(10.4,3.75),layout="constrained")
    cmap=ListedColormap(["#123954","#226181","#44879a","#72adae","#a4c9bf","#cddfd2","#e9efdf"])
    steps=2.**np.arange(-4,3)
    norm=BoundaryNorm(np.arange(-4.5,3.5),cmap.N)
    for ax,(tag,limit,title) in zip(axes,[
        ("wide288-b36",288,"36 × 36 cells per block"),
        ("wide288-72x36",288,"72 × 36 cells per block"),
        ("wide288-72x36",18,"Inner 18 code units")]):
        parts=[(Rectangle((a,c),b-a,d-c),np.log2(h)) for a,b,c,d,h in meshes[tag]]
        pc=PatchCollection([p for p,_ in parts],cmap=cmap,norm=norm,edgecolor="#48565b",linewidth=.55)
        pc.set_array(np.array([v for _,v in parts]))
        ax.add_collection(pc)
        if limit==288:
            ax.add_patch(Rectangle((0,0),36,36,fill=False,ec="#ab4028",lw=1.35,linestyle="--"))
        else:
            ax.add_patch(Rectangle((3,0),6,3,fill=False,ec="white",lw=1.2,linestyle="--"))
        ax.set(aspect="equal",xlim=(0,limit),ylim=(0,limit),title=title,xlabel=r"$\rho$ (code units)",
               xticks=np.linspace(0,limit,5),yticks=np.linspace(0,limit,5))
    axes[0].set_ylabel(r"$z$ (code units)")
    cb=fig.colorbar(pc,ax=axes,location="bottom",ticks=np.log2(steps),fraction=.08,aspect=65)
    cb.ax.set_xticklabels([f"{h:g}" for h in steps]);cb.set_label("Cell spacing in both directions (code units)")
    fig.savefig(output,dpi=175)
    plt.close(fig)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("exe",type=Path);cli.add_argument("output",type=Path)
    cli.add_argument("--shell",type=Path)
    args=cli.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    meshes,rows={},[]
    with tempfile.TemporaryDirectory() as folder:
        p=Path(folder)/"input"
        for tag,extent,nroot,bx,bz in LAYOUTS:
            p.write_text(make(True,True,True,extent=extent,root_blocks=nroot,
                              block_cells=bx,axial_block_cells=bz))
            subprocess.run([args.exe.resolve(),"-i",p,"-m","mesh_refinement/max_nmb_per_rank=512"],
                           cwd=folder,check=True,stdout=subprocess.DEVNULL)
            dump=p.parent/"mesh_structure.dat"
            count,_=verify(dump,True,bx,bz)
            grid=[(a,b,c,d,(b-a)/bx) for a,b,c,d in leaves(dump)]
            audit(grid,extent)
            for a,b,c,d,h in grid: assert (d-c)/bz==h
            meshes[tag]=grid
            row=dict(layout=tag,extent=extent,roots=[nroot,nroot*bx//bz],block_cells=[bx,bz],
                     initial_blocks=count,active_cells=count*bx*bz,
                     equatorial_shells=[[a,b,h] for a,b,c,d,h in sorted(grid) if c==0])
            if args.shell and extent==288: row["archived_shell"]=shell_spacings(args.shell,grid)
            rows.append(row)
    # An identical initial cell layout gives a direct block-size comparison.
    for a,b,c,d,h in meshes["wide288-b36"]:
        for x,y,z,w,k in meshes["wide288-72x36"]:
            if min(b,y)>max(a,x) and min(d,w)>max(c,z): assert h==k
    (args.output/"initial-meshes.json").write_text(json.dumps(rows,indent=2)+"\n")
    draw(meshes,args.output/"nested-mesh.png")


if __name__ == "__main__":
    main()
