from pathlib import Path
import json,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).parent;r=json.load(open(p/'coupled-rk3-pilot.json'))['runs'][-1]['records'];t=np.array([a['time']for a in r])/1000
fig,axs=plt.subplots(1,3,figsize=(12.2,3.65),layout='constrained');fig.set_facecolor('white')
a=axs[0];a.semilogy(t,[q['original_zero_rate_weighted_norm']for q in r],'-o',color='#a03030',label='Original zero_rate');a.semilogy(t,[q['exterior_memory_full_weighted_norm']for q in r],'-o',color='#236192',label='Coupled exterior');a.set_ylabel('Weighted state norm / initial');a.set_title('Same growing eigenmode seed');a.legend(fontsize=8)
a=axs[1];a.semilogy(t,[q['exterior_memory_full_weighted_norm']for q in r],'-o',label='Full domain');a.semilogy(t,[q['retained_interior_weighted_norm']for q in r],'-s',label='Retained interior');a.semilogy(t,[q['sampled_induced_norm']for q in r],'--',color='#8256a0',label='Sampled induced norm');a.set_ylabel('Weighted state norm / initial');a.set_title('Transient amplification remains');a.legend(fontsize=8)
a=axs[2];a.semilogy(t,np.array([q['theta_max_full']for q in r])/r[0]['theta_max_full'],'-o',color='#21765a');a.set_ylabel(r'$\max|\Theta| / \max|\Theta(0)|$');a.set_title('Constraint mode after closure change')
for a in axs:a.set_xlabel(r'$t/(1000M)$');a.grid(alpha=.2);a.set_xlim(0,50)
fig.savefig(p/'coupled-exterior-comparison.png',dpi=190);fig.savefig(p/'coupled-exterior-comparison.pdf')
