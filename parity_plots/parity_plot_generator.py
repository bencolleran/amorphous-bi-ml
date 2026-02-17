import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def func(name,str1,str2,var):
    filepath=f"/u/vld/sedm7085/test/{name}.csv"
    df=pd.read_csv(filepath)
    x=df[str1]
    y=df[str2]
    import numpy as np

    y_true = np.array(x)
    y_pred = np.array(y)

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    print(rmse)
    plt.figure(figsize=(6, 4), dpi=150)
    if var=='force':
        plt.title(name)
        plt.scatter(x,y)
        plt.xlabel("force per atom / eV/A")
        plt.ylabel("force per atom / eV/A")
        plt.savefig(f"{name}_plot", dpi=600, bbox_inches="tight")
    elif var=='energy':
        plt.title(name)
        plt.scatter(x,y)
        plt.xlabel("energy per atom / eV")
        plt.ylabel("energy per atom / eV")
        plt.savefig(f"{name}_plot", dpi=600, bbox_inches="tight")


# func('energies','energy','REF_energy','energy')
# func("per_atom_energies","energy_per_atom","REF_energy_per_atom","energy")
# func("force_magnitudes","F_mag","REF_F_mag",'force')
# func("forces_x","fx","REF_fx","force")
# func("forces_y","fy","REF_fy","force")
# func("forces_z","fz","REF_fz","force")

def parity_plot(name,var,str1=None,str2=None):
    filepath=f"/u/vld/sedm7085/project/parity_plots/{name}.csv"
    df=pd.read_csv(filepath)
    
    if var=='force':
        x=df[['fx', 'fy', 'fz']].to_numpy().ravel()
        y=df[['REF_fx', 'REF_fy', 'REF_fz']].to_numpy().ravel()

        y_true = np.array(x)
        y_pred = np.array(y)
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        print(rmse)
        plt.figure(figsize=(6, 4), dpi=150)
        #plt.title(f"{name} RMSE={round(rmse*1000,2)} meV/A")
        plt.scatter(x,y,marker='x', s=10)
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        plt.plot(lims, lims)
        plt.xlabel("predicted force per atom / eV/A")
        plt.ylabel("DFT force per atom / eV/A")
        plt.savefig(f"{name}_plot", dpi=600, bbox_inches="tight")
    elif var=='energy':
        x=df[str1]
        y=df[str2]

        y_true = np.array(x)
        y_pred = np.array(y)
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        print(rmse)
        plt.figure(figsize=(6, 4), dpi=150)
        #plt.title(f"{name} RMSE={round(rmse*1000,2)} meV")
        plt.scatter(x,y,marker='x', s=10)
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        plt.plot(lims, lims)
        plt.xlabel("predicted energy per atom / eV")
        plt.ylabel("DFT energy per atom / eV")
        plt.savefig(f"{name}_plot", dpi=600, bbox_inches="tight")

parity_plot("energies","energy","energy","REF_energy")
parity_plot("forces","force")