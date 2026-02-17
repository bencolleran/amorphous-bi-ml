import numpy as np

#arr=np.load("20260203_163603/forces.npy")

arr=np.load("20260205_122512/total_energies.npy")
n_atoms=192
arr=arr[9:200]/n_atoms
x=np.array([0.1*i for i in range(len(arr))])
y=arr
avg=np.array([np.mean(arr) for _ in x])
name="energies"

import numpy as np

def moving_average(x, window):
    pad = window // 2
    x_pad = np.pad(x, pad_width=pad, mode="reflect")
    return np.convolve(x_pad, np.ones(window)/window, mode='same')
w=10
ma = moving_average(arr, window=w)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.set_title(name)
ax.scatter(x,y,marker='x',s=10)
#ax.plot(x[(w-1):],ma)
ax.plot(x,ma[5:-5])
#ax.plot(x,avg)
ax.set_xlim(0, 20)
ax.set_ylabel("energy per atom (eV)")
ax.set_xlabel("simulation time (ps)")
#fig.savefig(f"{name}_plot", dpi=600, bbox_inches="tight")


#procedure is heat for 20 ps, cool for 30 ps, nvt for 10 ps
def plot_var(name):
    if name=='forces':
        filename='forces'
        ylabel="mean force magnitude per atom (meV/\u212B)"
        name="forces"
        f=0
        z=3
    else:
        filename="total_energies"
        ylabel="energy per atom (meV)"
        name="energies"
        f=697.4
        z=9
    import numpy as np

    #arr=np.load("20260203_163603/forces.npy")

    arr=np.load(f"20260205_122512/{filename}.npy")
    n_atoms=192
    arr=(arr[z:200]/n_atoms+f)*1000
    x=np.array([0.1*i for i in range(len(arr))])
    y=arr
    avg=np.array([np.mean(arr) for _ in x])
    print(max(arr))

    def moving_average(x, window):
        pad = window // 2
        x_pad = np.pad(x, pad_width=pad, mode="reflect")
        return np.convolve(x_pad, np.ones(window)/window, mode='same')
    w=40
    ma = moving_average(arr, window=w)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    #ax.set_title(name)
    ax.scatter(x,y,marker='x',s=10)
    #ax.plot(x[(w-1):],ma)
    ax.plot(x,ma[int(w/2):-int(w/2)],color="red")
    #ax.plot(x,avg)
    ax.set_xlim(0, 20)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("simulation time (ps)")
    fig.savefig(f"{name}_plot", dpi=600, bbox_inches="tight")
    return

plot_var("energies")
plot_var("forces")