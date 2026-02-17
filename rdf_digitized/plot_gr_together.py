import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def get(name):
    data=pd.read_csv(f'{PROJECT_ROOT}/rdf_digitized/{name}.csv',names=['r','g(r)'])
    df=pd.DataFrame(data)
    r=np.array(df['r'])
    gr=np.array(df['g(r)'])

    first=r[0]

    idx = np.argsort(r)
    r =r[idx]
    gr= gr[idx]

    r1=np.linspace(first,10,100)
    pchip=PchipInterpolator(r,gr)
    gr1=pchip(r1)
    return [r1,gr1]



r1=get("rdf_data_from_paper")[0]
gr1=get("rdf_data_from_paper")[1]
r2=get("ovito_rdf")[0]
gr2=get("ovito_rdf")[1]
r3=get("liquid_rdf")[0]
gr3=get("liquid_rdf")[1]

plt.figure(figsize=(6, 4), dpi=150)
ax = plt.gca()
ax.set_xlim([2, 10])
ax.set_ylim([0, 4])
plt.plot(r1,gr1,linestyle=':', alpha=0.7, label='a-Bi experimental')
plt.plot(r2,gr2, label='a-Bi LAMMPS')
plt.plot(r3,gr3,linestyle='--', alpha=0.7, label='liquid-Bi experimental')
plt.plot()
plt.legend()
plt.xlabel('r / \u212B')
plt.ylabel('g(r)')
plt.show()
plt.savefig(f'{PROJECT_ROOT}/rdf_digitized/dual_plot.png')
