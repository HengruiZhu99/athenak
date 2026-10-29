#!/usr/bin/env python3
"""Plot independently decoded V6 snapshot localization; no physical energy claim."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).resolve().parent
s=json.loads((root/'localization/localization.json').read_text())['localization']
regions=s['regions'];names=['Cartesian','transition_shell','middle_shell','outer_shell']
fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
values=[regions[n]['norm_squared_fraction']*100 for n in names]+[s['surface_norm_squared_fraction']*100]
axes[0].bar(['Core','Inner shell','Middle shell','Outer shell','Scri'],values,color=['#777777','#c4492d','#4575b4','#4575b4','#4575b4'])
axes[0].set_yscale('log');axes[0].set_ylabel('Fraction of squared amplitude (%)');axes[0].set_title('V6 Minkowski at coordinate t = 1')
for i,x in enumerate(values):axes[0].text(i,x*1.6,f'{x:.3g}%',ha='center',fontsize=8)
axes[0].set_ylim(1e-8,1000)
axes[1].bar(np.arange(16),100*np.array(regions['transition_shell']['i_row_fractions']),color='#c4492d')
axes[1].set_xlabel('Inner-shell radial cell index (0 = core side)');axes[1].set_ylabel('Fraction of total squared amplitude (%)')
axes[1].set_title('Concentrated near the Cartesian core')
fig.supxlabel('Unweighted tangent-coordinate norm; this is localization, not physical energy.',fontsize=9)
fig.savefig(root/'localization.png',dpi=180);fig.savefig(root/'localization.pdf')
