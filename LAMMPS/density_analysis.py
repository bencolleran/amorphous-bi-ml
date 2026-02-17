from analysis_tools import get_mean_bond_lengths,coordination_analysis
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[0]

a=np.load("soaps_20260212_152158.npy")#0.8
b=np.load("soaps_20260212_152203.npy")#1.0
c=np.load("soaps_20260212_152204.npy")#0.9
d=np.load("soaps_20260212_152218.npy")#1.1
e=np.load("soaps_20260212_152219.npy")#1.2

exp_names=['20260212_152158','20260212_152203','20260212_152204','20260212_152218','20260212_152219']

def get_xy_var(name,ps=False):
    var=np.load(f"{PROJECT_ROOT}/soaps_{name}.npy")
    length=var.shape[0]
    timestep=np.array(range(length))*100
    timestep_ps=np.array(range(length))*0.1
    var_new=get_mean_bond_lengths(name,bond_cutoff=3.4)
    if ps:
        return timestep_ps,var_new
    return timestep,var_new

dict_exp_names={'20260212_152158':0.8,
                '20260212_152204':0.9,
                '20260212_152203':1.0,
                '20260212_152218':1.1,
                '20260212_152219':1.2}

from scipy.signal import savgol_filter

fig, ax = plt.subplots()
i=10
s=10#sample
linestyles = [
    "-",    # solid
    "--",   # dashed
    "-.",   # dash-dot
    ":",    # dotted
    (0, (3, 1, 1, 1))
]
for i,key in enumerate(dict_exp_names):
    x,y=get_xy_var(key,ps=True)
    window_length = 100   # tune: larger → smoother
    polyorder = 3
    y_sg = savgol_filter(y, window_length=window_length, polyorder=polyorder)
    ax.plot(x[i:][::s],y_sg[i:][::s],label=f"{dict_exp_names[key]} scale",linestyle=linestyles[i])
ax.set_xlabel("timestep")
ax.set_ylabel("mean bond length")
ax.legend()
fig.savefig(f"{PROJECT_ROOT}/density_bonds.png")




# from scipy.interpolate import PchipInterpolator
# r1=np.linspace(first,10,100)
# pchip=PchipInterpolator(r,gr)
# gr1=pchip(r1)




dict_different_T={'20260212_152203':800,
                  '20260212_212048':1000,
                  '20260212_212133':1200}



# counts_3 = (arr == 3).sum(axis=1)#to get CN over the whole trajectory
# counts_4 = (arr == 4).sum(axis=1)
# counts_5 = (arr == 5).sum(axis=1)
# counts_6 = (arr == 6).sum(axis=1)


categories=[2,3,4,5,6]
all_counts=[]
labels=[]
idx=300
for key in dict_different_T:
    arr=coordination_analysis(key,bond_cutoff=3.4)[idx]
    counts_2 = (arr == 2).sum(axis=0)
    counts_3 = (arr == 3).sum(axis=0)
    counts_4 = (arr == 4).sum(axis=0)
    counts_5 = (arr == 5).sum(axis=0)
    counts_6 = (arr == 6).sum(axis=0)
    counts=np.array([
            counts_2,
            counts_3, 
            counts_4,
            counts_5,
            counts_6])/192
    all_counts.append(counts)
    labels.append(dict_different_T[key])

x = np.arange(len(categories))
width = 0.25

fig, ax = plt.subplots(figsize=(6,4))

ax.bar(x - width, all_counts[0], width, label=f"{labels[0]} K")
ax.bar(x,         all_counts[1], width, label=f"{labels[1]} K")
ax.bar(x + width, all_counts[2], width, label=f"{labels[2]} K")

ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_xlabel("Coordination Number")
ax.set_ylabel("Fraction")
ax.legend()

fig.tight_layout()
fig.savefig(f"{PROJECT_ROOT}/coordination.png")