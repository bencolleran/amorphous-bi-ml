from pathlib import Path
import numpy as np
from ase.io import Trajectory, read
from quippy.potential import Potential
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

PROJECT_ROOT = Path(__file__).resolve().parents[0]
#name = '20260203_163603'
name="20260205_122512"
#name="20260210_130003"

traj_file = f"{PROJECT_ROOT}/{name}/trajectory.traj"
out_forces_file = f"{PROJECT_ROOT}/{name}/forces_all.npy"     # will store shape (n_frames, n_atoms, 3)
#potential_spec = "QUIP param_filename=gap.xml"

traj = Trajectory(traj_file, mode="r")
n_frames = len(traj)   # number of frames (cheap for .traj)
if n_frames == 0:
    raise RuntimeError("Empty trajectory")

# Get number of atoms from first frame without loading entire traj
first = traj[0]
n_atoms = len(first)

# Pre-allocate storage (if memory allows)
forces_out = np.empty((n_frames, n_atoms, 3), dtype=float)
forces_out1 = np.empty((n_frames), dtype=float)

# Read GAP XML and initialise Potential using the form that works in your environment
xml_path = Path(PROJECT_ROOT) / "gap_file.xml"

xml_text = xml_path.read_text()


# Use the constructor that you confirmed works:
gap = Potential(args_str="IP GAP", param_str=xml_text)

# Attach calculator to an ASE Atoms instance (as in your working example)
atoms = first.copy()          # ASE Atoms
atoms.calc = gap              # attach calculator
# Loop frames (streaming)
for i, frame in enumerate(traj):
    # Update positions and cell in-place to avoid creating new Atoms each frame:
    atoms.set_positions(frame.get_positions(wrap=True))
    atoms.set_cell(frame.get_cell(), scale_atoms=False)
    atoms.set_pbc(frame.get_pbc())

    # Compute forces (calls quippy/GAP)
    forces = atoms.get_forces()   # returns (n_atoms, 3) in eV/Å
    forces_out[i] = forces
    #forces_out1[i]= np.sum(np.linalg.norm(forces,axis=1))
    #forces_out1[i]=[1 for _ in range(n_atoms)]
    # optionally save per-frame results incrementally to avoid large memory use
    # e.g., np.save(f"forces_frame_{i:06d}.npy", forces)
# Close trajectory and save results
traj.close()
np.save(out_forces_file, forces_out)
print("Saved forces shape:", forces_out.shape)
