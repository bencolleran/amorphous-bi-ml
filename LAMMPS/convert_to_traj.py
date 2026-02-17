import glob, re
from ase.io import read, Trajectory
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0]
#name='20260203_163603'
name="20260205_122512"
data_folder = f"{PROJECT_ROOT}/{name}/NVT"      # folder containing *.data
output_traj = f"{PROJECT_ROOT}/{name}/trajectory.traj"   # output ASE trajectory
atom_type_map = {1: "Bi"}          # LAMMPS atom type -> element
pbc = (True, True, True)

# get sorted list of data files
data_files = sorted(glob.glob(f"{data_folder}/dump_custom.Bi.*.dat"),
               key=lambda f: int(re.search(r"Bi\.(\d+)\.dat$", f).group(1)))

traj = Trajectory(output_traj, mode="w")

for fname in data_files:
    # determine how many frames are in this dump file; read returns a list-like when index=slice
    frames = read(
        fname,
        format="lammps-dump-text",
        index=slice(None))
    # frames is an Atoms object if single frame, or a list if many; normalize:
    if not isinstance(frames, (list, tuple)):
        frames = [frames]
    for atoms in frames:
        atoms.set_pbc(pbc)
        atoms.set_atomic_numbers([83] * len(atoms))
        traj.write(atoms)

traj.close()
print("Done.")

