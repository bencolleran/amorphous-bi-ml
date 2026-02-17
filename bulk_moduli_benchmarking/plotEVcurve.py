import matplotlib.pyplot as plt
from ase.eos import EquationOfState
import numpy as np
import json
from ase.io import read
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

atoms_exp=read(f'{PROJECT_ROOT}/structures/Bi_1.cell')

filelist=['/u/vld/sedm7085/autoplex_out/d7/cd/3d/d7cd3d12-ae31-4b2c-b82e-498b660b36a9_1/remote_job_data.json',#LDA
          '/u/vld/sedm7085/autoplex_out/9c/f2/b5/9cf2b5e6-a3ad-4fa4-8039-6362f0079310_1/remote_job_data.json',#PBE
          '/u/vld/sedm7085/autoplex_out/94/6d/c0/946dc0d4-502e-476c-a38a-96e8bf3a8cff_1/remote_job_data.json',#PBESOL
          '/u/vld/sedm7085/autoplex_out/26/6d/03/266d0343-6230-43a8-9261-5c5867aeabde_1/remote_job_data.json',#RSCAN
          '/u/vld/sedm7085/autoplex_out/56/66/cc/5666ccee-40cb-4f81-8890-e5ee82406b51_1/remote_job_data.json',#LDA_SOC
          '/u/vld/sedm7085/autoplex_out/0e/f4/b5/0ef4b5ff-7c43-4cfc-b169-1c06f19f10df_1/remote_job_data.json',#PBE_SOC
          '/u/vld/sedm7085/autoplex_out/2d/43/01/2d43018e-2bcf-49c6-9e37-cf2a3be29502_1/remote_job_data.json',#PBESOL_SOC
          '/u/vld/sedm7085/autoplex_out/13/58/18/135818e1-b768-4e41-a8df-5fe662c21194_1/remote_job_data.json'#RSCAN_SOC
]

test=['/u/vld/sedm7085/autoplex_out/e4/9e/c9/e49ec916-8eae-425a-a510-285218948bfa_1/remote_job_data.json',#LDA
      '/u/vld/sedm7085/autoplex_out/21/19/e0/2119e0ac-7ec6-48e4-98bf-31b9c174fa83_1/remote_job_data.json',#PBE
      '/u/vld/sedm7085/autoplex_out/78/8f/79/788f791c-93be-4ca4-aaf2-9f6573d1071f_1/remote_job_data.json',#PBESOL
      '/u/vld/sedm7085/autoplex_out/9d/40/39/9d4039f3-f102-4a53-b1be-c3affb0b7f13_1/remote_job_data.json'#RSCAN
      ]

converged_test=[]

def plot_ev(file):
    with open(file,'r') as f:
        data=json.load(f)
    volume=data[0]['output']['relax']['volume']
    energy=data[0]['output']['relax']['energy']
    x=data[0]['output']['relax']['EOS']['birch_murnaghan']
    name=data[0]['name'][4:-15]
    print(name)

    eos = EquationOfState(volume, energy, eos='birchmurnaghan')

    #v0, e0, B0 = eos.fit()

    v_fit = np.linspace(min(volume), max(volume), 200)
    e_fit = eos.func(v_fit, *eos.eos_parameters)
    plt.figure(figsize=(6, 4), dpi=150)
    plt.scatter(volume, energy, color='red', label='DFT data')
    plt.plot(v_fit, e_fit, color='blue', label='Birch-Murnaghan fit')
    plt.legend()
    plt.savefig(f"evcurve_{name}.png", dpi=600, bbox_inches="tight")
    print(x["b0 GPa"])

def plot_all_ev(files,SOC=False):
    plt.figure(figsize=(6, 4), dpi=150)
    for file in files:
        with open(file,'r') as f:
            data=json.load(f)
        volume=data[0]['output']['relax']['volume']
        energy=data[0]['output']['relax']['energy']
        x=data[0]['output']['relax']['EOS']['birch_murnaghan']
        name=data[0]['name'][4:-15]

        eos = EquationOfState(volume, energy, eos='birchmurnaghan')

        v0, e0, B0 = eos.fit()
        print()
        print(f'{name} {x["b0 GPa"]} GPa')
        energy_0=[e-e0 for e in energy]
        v_fit = np.linspace(min(volume), max(volume), 200)
        e_fit = eos.func(v_fit, *eos.eos_parameters)
        e_fit_0=e_fit-e0
        plt.scatter(np.array(volume)/6,energy_0,s=10, marker='x')
        plt.plot(v_fit/6, e_fit_0, label=f'{name}')
    plt.scatter(atoms_exp.get_volume()/6,0,label='experimental',marker='x', color='k')
    plt.xlabel("Volume per atom (Å³)", fontsize=12)
    plt.ylabel("Energy per atom (eV)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    print()
    if SOC:
        plt.savefig(f"{PROJECT_ROOT}/bulk_moduli_benchmarking/evcurve_all_SOC.png", dpi=600, bbox_inches="tight")
    else:
        plt.savefig(f"{PROJECT_ROOT}/bulk_moduli_benchmarking/evcurve_all.png", dpi=600, bbox_inches="tight")

def plot_both():
    plt.figure(figsize=(10, 5), dpi=150)
    colours=['r','orange','g','b']
    for i in range(4):
        with open(filelist[i],'r') as f:
            data=json.load(f)
        volume=data[0]['output']['relax']['volume']
        energy=data[0]['output']['relax']['energy']
        x=data[0]['output']['relax']['EOS']['birch_murnaghan']
        name=data[0]['name'][4:-15]

        eos = EquationOfState(volume, energy, eos='birchmurnaghan')

        v0, e0, B0 = eos.fit()
        print(name,x["b0 GPa"])
        energy_0=[e-e0 for e in energy]
        v_fit = np.linspace(min(volume), max(volume), 200)
        e_fit = eos.func(v_fit, *eos.eos_parameters)
        e_fit_0=e_fit-e0
        plt.scatter(np.array(volume)/6,energy_0,s=10, marker='x',color=colours[i])
        plt.plot(v_fit/6, e_fit_0, label=f'{name}',color=colours[i])
        with open(filelist[i+4],'r') as f:
            data=json.load(f)
        volume=data[0]['output']['relax']['volume']
        energy=data[0]['output']['relax']['energy']
        x=data[0]['output']['relax']['EOS']['birch_murnaghan']
        name=data[0]['name'][4:-15]

        eos = EquationOfState(volume, energy, eos='birchmurnaghan')

        v0, e0, B0 = eos.fit()
        print(name,x["b0 GPa"],)
        energy_0=[e-e0 for e in energy]
        v_fit = np.linspace(min(volume), max(volume), 200)
        e_fit = eos.func(v_fit, *eos.eos_parameters)
        e_fit_0=e_fit-e0
        plt.scatter(np.array(volume)/6,energy_0,s=10, marker='x',color=colours[i])
        plt.plot(v_fit/6, e_fit_0,'--', label=f'{name}',color=colours[i])
    plt.scatter(atoms_exp.get_volume()/6,0,label='experimental',marker='x', color='k')
    plt.xlabel("Volume per atom (Å³)", fontsize=12)
    plt.ylabel("Energy per atom (eV)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{PROJECT_ROOT}/bulk_moduli_benchmarking/evcurve_comparison.png", dpi=600, bbox_inches="tight")

#plot_all_ev(filelist[:4])
#plot_all_ev(filelist[4:],SOC=True)
#plot_both()
plot_all_ev(test,SOC=True)
#for file in filelist:
plot_ev()