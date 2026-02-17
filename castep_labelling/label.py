from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from autoplex.misc.castep.jobs import BaseCastepMaker, CastepStaticMaker
from autoplex.misc.castep.utils import CastepInputGenerator, CastepStaticSetGenerator
from jobflow import Flow
from jobflow_remote import submit_flow
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0]

parser=argparse.ArgumentParser()
parser.add_argument('--kpoints', type=str, default='6 6 6', help='argument is specified in the form "a b c"')
parser.add_argument('--structure', type=str,)
args=parser.parse_args()


#atoms = read(f'{PROJECT_ROOT}/{args.structure}.dat', format="lammps-dump-text",specorder=["Bi"])#how to read lammps dump files
atoms = read(f'{PROJECT_ROOT}/{args.structure}.cell')
pmg_structure = AseAtomsAdaptor.get_structure(atoms)

static_job = BaseCastepMaker(
    input_set_generator=CastepStaticSetGenerator(
        user_param_settings={
        "task": 'SINGLEPOINT',
        'cut_off_energy': 600,
        'xc_functional': "PBE",
        'max_scf_cycles': 100,
        },
        user_cell_settings={
        "kpoints_mp_grid": args.kpoints,
        "species_pot": [("Bi", 'SOC19')],
        }
    )
).make(structure=pmg_structure)

static_flow = Flow(static_job, name=f'{args.kpoints}',  output=static_job.output)

resources = {"qverbatim": "#$ -cwd\n#$ -pe smp 128\n#$ -N Autoplex_jf_test\n#$ -o $JOB_ID.log\n#$ -e $JOB_ID.err\n#$ -P cpu\n#$ -l s_rt=05:00:00"}
#resources = {"qverbatim": "#$ -cwd\n#$ -pe smp 64\n#$ -N Autoplex_jf_test\n#$ -o $JOB_ID.log\n#$ -e $JOB_ID.err\n#$ -P highmem\n#$ -l s_rt=05:00:00"}

print(submit_flow(static_flow , worker="autoplex_project_worker", resources=resources, project="autoplex_project"))

#make sure that mongodb is running
#use jobflow to make a way to do lots of castep calculations