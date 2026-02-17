import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

data=pd.read_csv(f'{PROJECT_ROOT}/rdf_digitized/rdf.csv',names=['r','RDF'])
df=pd.DataFrame(data)
r=np.array(df['r'])
rdf=np.array(df['RDF'])

idx = np.argsort(r)
r =r[idx]
rdf= rdf[idx]

r1=np.linspace(2,10,100)
pchip=PchipInterpolator(r,rdf)
rdf1=pchip(r1)
rho0=0.0283 #from paper

rhor=rdf1/(4*np.pi*r1**2*rho0)
print(rhor[-1])
gr=[rhor[i] for i in range(len(rhor))]

#plt.figure(figsize=(6, 4), dpi=150)
ax = plt.gca()
ax.set_xlim([0, 10])
ax.set_ylim([0, 3])
plt.plot(r1,gr)
plt.xlabel('r / \u212B')
plt.ylabel('g(r)')
plt.show()
plt.savefig(f'{PROJECT_ROOT}/rdf_digitized/gr.png')