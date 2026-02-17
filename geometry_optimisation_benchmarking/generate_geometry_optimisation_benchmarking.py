from ase.io import read
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

path=f'{PROJECT_ROOT}/geometry_optimisation_benchmarking/geometry_optimisation_benchmarking_data/'

atoms_exp=read('/u/vld/sedm7085/project/structures/Bi_1.cell')
atoms_LDA=read(f'{path}LDA_CASTEP_geom_opt_800/castep.castep')
atoms_PBE=read(f'{path}PBE_CASTEP_geom_opt_800/castep.castep')
atoms_PBESOL=read(f'{path}PBESOL_CASTEP_geom_opt_800/castep.castep')
atoms_RSCAN=read(f'{path}RSCAN_CASTEP_geom_opt_800/castep.castep')


atoms=[atoms_exp,atoms_LDA,atoms_PBE,atoms_PBESOL,atoms_RSCAN]
functionals=['exp','LDA','PBE','PBEsol','rSCAN']

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
}
df=pd.DataFrame(data)
df.set_index('functional',inplace=True)
print(df)

categories=[f'a',f'b',f'c',f'volume']
data1={
    f'a':a_rel_error[1:],
    f'b':b_rel_error[1:],
    f'c':c_rel_error[1:],
    f'volume':volume_rel_error[1:]
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
        y_offset = 0.02 * j
        ax.hlines(y=y + y_offset, xmin=i - line_width/2, xmax=i + line_width/2,
                  color=c, lw=3)

ax.set_xticks(x_positions)
ax.set_xticklabels(categories)
ax.set_ylabel(f"percentage relative errors / {percentage}")
ax.set_xlim(-0.5, len(categories) - 0.5)
plt.tight_layout()
plt.show()
plt.legend(functionals,bbox_to_anchor=(1.05, 1), loc="upper left")
fig.savefig(f"{PROJECT_ROOT}/geometry_optimisation_bencmarking/colored_horizontal_lines.png", dpi=600, bbox_inches="tight")