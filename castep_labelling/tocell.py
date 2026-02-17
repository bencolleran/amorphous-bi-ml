from ase.build import bulk
from ase.io import write,read

atoms = read('1.dat', format="lammps-dump-text",specorder=["Bi"])#how to read lammps dump files
# Write CASTEP .cell file
write('1.cell', atoms)
