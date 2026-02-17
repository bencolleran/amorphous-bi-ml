from ase.io import read
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

pathname="/u/vld/sedm7085/project/dft_data"
names=os.listdir(pathname)

filtered = [x for x in names if "CASTEP_" in x and x.split("CASTEP_")[-1].isdigit() and len(x.split("CASTEP_")[-1])==1]
print(filtered)
filtered=sorted(filtered)
print(filtered)
lst1=filtered[:5]
lst2=filtered[5:10]
lst3=filtered[10:15]
lst4=filtered[15:]
filtered=lst1+lst3+lst2+lst4
energies=[read(f'{pathname}/{i}/castep.castep').get_potential_energy() for i in filtered]
print(energies)
functionals=['LDA','PBE','PBEsol','rSCAN']
n=5
Bi_1=[energies[n*i]/6 for i in range(int(len(energies)/n))]
Bi_2=[energies[n*i+1]/4 for i in range(int(len(energies)/n))]
Bi_3=[energies[n*i+2]/8 for i in range(int(len(energies)/n))]
Bi_4=[energies[n*i+3]/16 for i in range(int(len(energies)/n))]
Bi_5=[energies[n*i+4]/2 for i in range(int(len(energies)/n))]

data={
    'functionals':functionals,
    'Bi_1': Bi_1,
    'Bi_2': Bi_2,
    'Bi_3': Bi_3,
    'Bi_4': Bi_4,
    'Bi_5': Bi_5,
}
one='\u2160'
two='\u2161'
three='\u2162'
four='\u2163'
five='\u2164'
df=pd.DataFrame(data)
df.set_index('functionals',inplace=True)
print(df)
df_sub=df.subtract(df['Bi_1'],axis=0)
print(df_sub)

print(list(df_sub['Bi_2']))
categories=[f'Bi {two}',f'Bi {three}',f'Bi {four}',f'Bi {five}']
data1={
    f'Bi {two}':list(df_sub['Bi_2']),
    f'Bi {three}':list(df_sub['Bi_3']),
    f'Bi {four}':list(df_sub['Bi_4']),
    f'Bi {five}':list(df_sub['Bi_5'])
    }

functionals=['LDA','PBE','PBEsol','rSCAN']
fig, ax = plt.subplots(figsize=(6, 4))
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
x_positions = np.arange(len(categories))
line_width = 0.3  # how wide each horizontal line is

for i, cat in enumerate(categories):
    y_values = data1[cat]
    for j, (y, c) in enumerate(zip(y_values, colors)):
        # small vertical offset so lines don't overlap
        y_offset = 0.005 * j
        ax.hlines(y=y + y_offset, xmin=i - line_width/2, xmax=i + line_width/2,
                  color=c, lw=3)

ax.set_xticks(x_positions)
ax.set_xticklabels(categories)
ax.set_ylabel(f'energies relative to Bi {one} / eV/atom')
ax.set_xlim(-0.5, len(categories) - 0.5)
plt.tight_layout()
plt.show()
plt.legend(functionals,bbox_to_anchor=(1.05, 1), loc="upper left")
fig.savefig(f"{PROJECT_ROOT}/geometry_optimisation_benchmarking/colored_horizontal_lines_structs.png", dpi=600, bbox_inches="tight")
'''
atoms_LDA=read('/u/vld/sedm7085/relaxation/LDA/CASTEP/castep.castep')
atoms_PBE=read('/u/vld/sedm7085/relaxation/PBE/CASTEP/castep.castep')
atoms_PBESOL=read('/u/vld/sedm7085/relaxation/PBESOL/CASTEP/castep.castep')
atoms_RSCAN=read('/u/vld/sedm7085/relaxation/RSCAN/CASTEP/castep.castep')


atoms=[atoms_LDA,atoms_PBE,atoms_PBESOL,atoms_RSCAN]
functionals=['LDA','PBE','PBEsol','rSCAN']

energies=[atom.get_potential_energy() for atom in atoms]

#print(type(atoms_LDA.get_cell()[0]))
#print(type(atoms_LDA.get_volume()))
volumes=[atom.get_volume() for atom in atoms]
a=[np.linalg.norm(atom.get_cell()[0]) for atom in atoms]
b=[np.linalg.norm(atom.get_cell()[1]) for atom in atoms]
c=[np.linalg.norm(atom.get_cell()[2]) for atom in atoms]

a_exp=a[0]
b_exp=b[0]
c_exp=c[0]
volumes_exp=volumes[0]

angstrom = "\u212B"   # Unicode code for Å
angstrom_cubed = "\u212B\u00B3"  # Å + superscript 3
percentage= '\u0025'

length=range(len(a))
a_rel_error=[100*(a[i]-a_exp)/a_exp for i in length]
b_rel_error=[100*(b[i]-b_exp)/b_exp for i in length]
c_rel_error=[100*(c[i]-c_exp)/c_exp for i in length]
volume_rel_error=[100*(volumes[i]-volumes_exp)/volumes_exp for i in length]

data ={ 
    'functional':functionals,
    f'a ({angstrom})':a,
    f'b ({angstrom})':b,
    f'c ({angstrom})':c,
    f'volume ({angstrom_cubed})':volumes,
    f'a error ({percentage})':a_rel_error,
    f'b error ({percentage})':b_rel_error,
    f'c error ({percentage})':c_rel_error,
    f'volume error ({percentage})':volume_rel_error,
    'potential energy': energies
}
df=pd.DataFrame(data)
df.set_index('functional',inplace=True)
print(df)
'''