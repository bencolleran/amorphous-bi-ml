import numpy as np

#arr=np.load("20260203_163603/forces.npy")

arr=np.load("20260205_122512/forces.npy")
n_atoms=192
arr=arr[3:200]

x=np.array([0.1*i for i in range(len(arr))])
y=arr
avg=np.array([np.mean(arr) for _ in x])
name="force magnitudes"
import matplotlib.pyplot as plt

def moving_average(x, window):
    return np.convolve(x, np.ones(window)/window, mode='valid')
w=20
ma = moving_average(arr, window=w)

fig, ax = plt.subplots()
ax.set_title(name)
ax.scatter(x,y,marker='x',s=10)
#ax.plot(x,avg)
ax.plot(x[(w-1):],ma)
ax.set_xlim(0, 20)
ax.set_ylabel("mean force magnitude per atom (eV/\u212B)")
ax.set_xlabel("simulation time (ps)")
fig.savefig(f"images/{name}_plot", dpi=600, bbox_inches="tight")


#procedure is heat for 20 ps, cool for 30 ps, nvt for 10 ps
