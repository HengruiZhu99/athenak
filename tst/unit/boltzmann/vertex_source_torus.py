"""Measure the actual gravity source against the analytic launch E and covariant S_y."""
import argparse
import csv
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

root = Path(__file__).resolve().parents[3]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("athena", type=Path)
parser.add_argument("spectral_input", type=Path)
args = parser.parse_args()
exe = args.athena.resolve()
spectral = args.spectral_input.resolve()
with spectral.open() as stream:
    center, width, norm, impact = map(float, stream.readline().split()[5:9])
deck = (root/"inputs/boltzmann/photon_torus.athinput").read_text()
deck += "\n<fastflow>\nnum_horizons = 0\n"
deck = deck.replace("<boltzmann>", "<boltzmann>\nsource_order = 2")
writer = csv.writer(sys.stdout, lineterminator="\n")
writer.writerow("order n dx E_L1 E_Linf Sphi_L1 Sphi_Linf H_L2 E_order Sphi_order vertex_E_minus_FV_fraction vertex_J_minus_FV_fraction min_E min_cov_eigenvalue trace_rel_max".split())
for order in (2, 4):
    prev = None
    for n in (48, 96, 192, 384):
        with tempfile.TemporaryDirectory(prefix="gravity-source-") as tmp:
            p = Path(tmp)
            (p/"input").write_text(deck)
            # Check shared vertices before selecting one copy for quadrature.
            command = [str(exe), "-i", "input", f"problem/torus_file={spectral}", "time/nlim=0",
                       "mesh/x1max=12", "mesh/x2min=-6", "mesh/x2max=6",
                       f"mesh/nx1={n}",f"mesh/nx2={n}","boltzmann/nmu_half=8","boltzmann/npsi=12",
                       "boltzmann/transport_order=4",f"boltzmann/source_order={order}",
                       "time/integrator=ssprk104","fastflow/num_horizons=0"]
            run = subprocess.run(command, cwd=p, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,text=True)
            if run.returncode:
                raise RuntimeError(run.stdout[-2000:])
            data=np.loadtxt(p/"initial.vertices.0")
            xy, first, inverse=np.unique(data[:,:2],axis=0,return_index=True,return_inverse=True)
            np.testing.assert_allclose(data[:,[19,22]],data[first[inverse]][:,[19,22]],rtol=0,atol=1e-13)
            data=data[first]
            r,z=xy.T
            u=((r-center)**2+z*z)/width**2
            exact=np.zeros(len(r))
            support=u<1
            exact[support]=norm*np.exp(1-1/(1-u[support]))
            exact_j=np.divide(exact*impact,r,out=np.zeros(len(r)),where=r>0)
            dx=12/n
            # Axisymmetric proper-volume errors normalized to the analytic source.
            volume=r*data[:,2]**1.5
            volume[r==12]*=.5
            volume[abs(z)==6]*=.5
            err=np.abs(data[:,19]-exact)
            jr=np.abs(data[:,22]-exact_j)
            l1=np.sum(volume*err)/np.sum(volume*exact)
            j1=np.sum(volume*jr)/np.sum(volume*np.abs(exact_j))
            orders=np.log2(np.array(prev)/[l1,j1]) if prev else [0,0]
            energy = data[:,19]
            mask = energy > 1e-12*energy.max()
            psi4 = data[mask,2]
            comps = data[mask,13:19]/(energy[mask,None]*psi4[:,None])
            pressure = comps[:,np.array([[0,1,2],[1,3,4],[2,4,5]])]
            current = data[mask,20:23]/(energy[mask,None]*np.sqrt(psi4)[:,None])
            covariance = pressure-current[:,:,None]*current[:,None,:]
            low = float(np.linalg.eigvalsh(covariance).min())
            trace = float(np.max(np.abs(pressure.trace(axis1=1,axis2=2)-1)))
            history=np.loadtxt(p/"torus.user.hst",ndmin=2)[0]
            factor=2*np.pi*dx**2
            charge_E=factor*np.sum(volume*energy)/history[2]-1
            charge_J=factor*np.sum(volume*r*data[:,22])/history[3]-1
            assert energy.min()>=0
            if order==4:
                assert low>=-1e-10 and trace<1e-9, (low,trace)
            writer.writerow([order,n,dx,l1,err.max()/norm,j1,jr.max()/np.max(np.abs(exact_j)),
                             np.sqrt(2*np.pi*dx**2*np.sum(volume*data[:,9]**2)),*orders,
                             charge_E,charge_J,energy.min(),low,trace])
            sys.stdout.flush()
            prev=l1,j1
