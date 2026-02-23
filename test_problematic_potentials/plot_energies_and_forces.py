import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from jobflow_remote.jobs.jobcontroller import JobController
import re,json
from datetime import datetime
from datetime import datetime
import numpy as np
from pathlib import Path
import gzip
from ase.io import read
from quippy.potential import Potential
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", module="ase.calculators.castep")
warnings.filterwarnings("ignore", module="ase.io.castep")

PROJECT_ROOT = Path(__file__).resolve().parents[0]

unit = "\u00C5"
quip_energy=[]
quip_forces=[]
quip_energy_new=[]
quip_forces_new=[]


for structure in [1,2,3]:
    atoms=read(f"{PROJECT_ROOT}/{structure}.dat",specorder=["Bi"])
    xml_path = Path(PROJECT_ROOT) / "initial_autoplex_mlip/gap_file.xml"
    gap = Potential(args_str="IP GAP", param_filename=str(xml_path))#correct quip code
    atoms.calc = gap
    quip_energy.append(atoms.get_total_energy()/192.0)
    quip_forces.append(atoms.get_forces())
    atoms.calc=None

    xml_path_new = Path(PROJECT_ROOT) / "defective_extended_data_mlip/GAP.xml"
    gap_new = Potential(args_str="IP GAP", param_filename=str(xml_path_new))#correct quip code
    atoms.calc = gap_new
    quip_energy_new.append(atoms.get_total_energy()/192.0)
    quip_forces_new.append(atoms.get_forces())
    atoms.calc=None

quip_forces=np.array(quip_forces).ravel()
quip_energy=np.array(quip_energy)
quip_forces_new=np.array(quip_forces_new).ravel()
quip_energy_new=np.array(quip_energy_new)

fig, ax = plt.subplots()
ax.scatter(quip_energy_new, quip_energy, s=10, marker='x')
# ax.plot([min(quip_energy_new), max(quip_energy_new)],
#         [min(quip_energy_new), max(quip_energy_new)], "k--")
ax.set_xlabel("new energy / eV")
ax.set_ylabel("old energy / eV")
fig.tight_layout()
fig.savefig(f"{PROJECT_ROOT}/images/energy_plot_new.png")


fig, ax = plt.subplots()
ax.scatter(quip_forces_new, quip_forces, s=10, marker='x')
# ax.plot([min(quip_forces_new), max(quip_forces_new)],
#         [min(quip_forces_new), max(quip_forces_new)], "k--")
ax.set_xlabel(f"new forces / eV/{unit}")
ax.set_ylabel(f"old forces / eV/{unit}")
fig.tight_layout()
fig.savefig(f"{PROJECT_ROOT}/images/forces_plot_new.png")