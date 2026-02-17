from ase import Atoms
from ase.io import read
from quippy.potential import Potential
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore", module="ase.calculators.castep")
warnings.filterwarnings("ignore", module="ase.io.castep")
import subprocess
commands = [
    "export OPENBLAS_NUM_THREADS=1",
    "export OMP_NUM_THREADS=1"
]

for cmd in commands:
    subprocess.run(cmd, shell=True, check=True)

#script must be run in mlip_tests dir and run the below commands first
#run in bash: 
# export OPENBLAS_NUM_THREADS=1
# export OMP_NUM_THREADS=1

filepath="/u/vld/sedm7085/project/structures"

structures=os.listdir(filepath)
d = {(i+1): v for i, v in enumerate(structures)}

atoms=read(f'{filepath}/{d[2]}',format='castep-cell')

def structure(i):
    atoms=read(f'{filepath}/{d[i]}',format='castep-cell')
    return atoms
num_iterations=25
num_atoms=len(atoms)
paths=[f'{i+1}.xml' for i in range(num_iterations) ]
machine_learning_iteration=[i+1 for i in range(num_iterations)]
#print(machine_learning_iteration)

for path in paths:
    xml_path = path
    xml_text = open(xml_path, "r").read()

    gap = Potential(args_str="IP GAP", param_str=xml_text)
    atoms.calc = gap

    #print(atoms.get_potential_energy())
    forces=atoms.get_forces()
    mean_force_mag=np.sum([np.linalg.norm(np.array(forces[i])) for i in range(num_atoms)])/num_atoms
    #print(mean_force_mag)


def generate(atoms,x):
    num_atoms=len(atoms)
    paths=[f'{i+1}.xml' for i in range(num_iterations) ]
    energy=[]
    force=[]
    for path in paths:
        xml_path = path
        xml_text = open(xml_path, "r").read()

        gap = Potential(args_str="IP GAP", param_str=xml_text)
        atoms.calc = gap
        energy.append(atoms.get_potential_energy())
        forces=atoms.get_forces()
        force.append(np.sum([np.linalg.norm(np.array(forces[i])) for i in range(num_atoms)])/num_atoms)
    if x=='energies':
        return energy
    if x=='forces':
        return force
    if x=='norm_energies':
        energy=[energy[i]-energy[-1] for i in range(len(energy)-1)]
        return energy

#print(generate(atoms,'forces'))

numerals=['\u2160','\u2161','\u2162','\u2163','\u2164']
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# plt.figure()
# for i in range(5):
#     plt.plot(machine_learning_iteration,generate(structure(i+1),'energies'), label=f'Bi_{i+1}')
# plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
# plt.xlabel('machine learning iteration')
# plt.ylabel('energy / eV')
# plt.legend()
# plt.savefig(f"energy_graph.png", dpi=600, bbox_inches="tight")

plt.figure()
for i in range(5):
    plt.plot(machine_learning_iteration[:-1],generate(structure(i+1),'norm_energies'), label=f'Bi_{i+1}')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('machine learning iteration')
plt.ylabel('energy / eV')
plt.legend()
plt.savefig(f"normalized_energy_graph.png", dpi=600, bbox_inches="tight")

plt.figure()
for i in range(5):
    plt.plot(machine_learning_iteration[:-1],(np.array(generate(structure(i+1),'norm_energies'))+i), label=f'Bi_{i+1}')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('machine learning iteration')
plt.ylabel('energy / eV/atom')
#plt.legend()
plt.savefig(f"norm_energy_graph.png", dpi=600, bbox_inches="tight")

plt.figure()
for i in range(5):
    plt.plot(machine_learning_iteration,generate(structure(i+1),'forces'), label=f'Bi {numerals[i]}')
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.xlabel('machine learning iteration')
plt.ylabel('forces / eV/Å atom')
plt.legend()
plt.savefig(f"forces_graph.png", dpi=600, bbox_inches="tight")

