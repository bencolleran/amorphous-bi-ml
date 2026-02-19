from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from autoplex.misc.castep.jobs import BaseCastepMaker, CastepStaticMaker
from autoplex.misc.castep.utils import CastepInputGenerator, CastepStaticSetGenerator
from jobflow import Flow
from jobflow_remote import submit_flow
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0]

files = list(Path(f"{PROJECT_ROOT}/LAMMPS_structures").glob("*.dat"))

jobs=[]

for i,file in enumerate(files):
    atoms = read(file, format="lammps-dump-text",specorder=["Bi"])#how to read lammps dump files
    pmg_structure = AseAtomsAdaptor.get_structure(atoms)
    calc=BaseCastepMaker(
    input_set_generator=CastepStaticSetGenerator(
        user_param_settings={
        "task": 'SINGLEPOINT',
        'cut_off_energy': 600,
        'xc_functional': "PBE",
        'max_scf_cycles': 100,
        },
        user_cell_settings={
        "kpoints_mp_grid": "3 3 3",
        "species_pot": [("Bi", 'SOC19')],
        }
    )
)
    job=calc.make(structure=pmg_structure)
    job.name=f"{Path(file).name}"
    jobs.append(job)

flow=Flow(jobs, name="label lammps data")
resources = {"qverbatim": "#$ -cwd\n#$ -pe smp 128\n#$ -N Autoplex_jf_test\n#$ -o $JOB_ID.log\n#$ -e $JOB_ID.err\n#$ -P cpu\n#$ -l s_rt=05:00:00"}
print(submit_flow(flow , worker="autoplex_project_worker", resources=resources, project="autoplex_project"))