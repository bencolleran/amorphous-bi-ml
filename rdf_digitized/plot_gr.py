import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
name="liquid_rdf"
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

plt.figure(figsize=(6, 4), dpi=150)
ax = plt.gca()
ax.set_xlim([0, 10])
ax.set_ylim([0, 5])
plt.plot(r1,gr1)
plt.xlabel('r / \u212B')
plt.ylabel('g(r)')
plt.show()
plt.savefig(f'{PROJECT_ROOT}/rdf_digitized/{name}.png')

# plt.figure(figsize=(6, 4), dpi=150)
# ax = plt.gca()
# ax.set_xlim([0, 10])
# ax.set_ylim([0, 3])
# plt.plot(r,gr)
# plt.xlabel('r / \u212B')
# plt.ylabel('g(r)')
# plt.show()
# plt.savefig(f'{PROJECT_ROOT}/rdf_digitized/gr_newest.png')