from ase.io import read, write, lammpsdata
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0]

n = 2
x = n
y = n
z = int(n/2)
n_atoms = x * y * z

#scale = 0.9297582964171913   # set to e.g. 1.05 to scale cell & positions by 5%
#old was scale=n
#dont just trust the name of a structure, use ase to calculate the volume etc
scaled_volume = 0.8
scale=np.cbrt(scaled_volume)

atoms=lammpsdata.read_lammps_data(f"{PROJECT_ROOT}/Bi.data",style="atomic")

supercell = atoms.repeat((x, y, z))

# helper to compute density in g/cm^3
# mass in amu, volume in Å^3 -> density (g/cm^3) = mass_amu * (1.66053906660) / volume_A3
def density_g_cm3(atoms_obj):
    atoms_obj.set_atomic_numbers([83] * len(atoms_obj))
    total_mass_amu = np.sum(atoms_obj.get_masses())
    vol_A3 = atoms_obj.get_volume()
    if vol_A3 <= 0:
        return float('nan')
    return total_mass_amu * 1.66053906660 / vol_A3

# density before scaling
dens_before = density_g_cm3(supercell)
print(f"Atoms in supercell: {len(supercell)}")
print(f"Density (g/cm^3) before scaling: {dens_before:.6f}")

# apply scaling to cell and positions (preserves fractional coordinates)
if scaled_volume != 1.0:
    new_cell = supercell.get_cell() * float(scale)
    supercell.set_cell(new_cell, scale_atoms=True)
    dens_after = density_g_cm3(supercell)
    print(f"Applied scale = {scale}")
    print(f"Volume (Å^3) after scaling: {supercell.get_volume():.6f}")
    print(f"Density (g/cm^3) after scaling: {dens_after:.6f}")
else:
    print("No scaling applied (scale = 1.0)")

# write out (same pattern as you had)
name=f"{PROJECT_ROOT}/structures/Bi_{x}_{y}_{z}_{48*x*y*z}_{scaled_volume}_scale.data"
print(name)
write(
    name,
    supercell,
    format="lammps-data",
    atom_style="atomic",
)
