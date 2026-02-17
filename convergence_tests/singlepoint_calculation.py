from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from autoplex.misc.castep.jobs import BaseCastepMaker, CastepStaticMaker
from autoplex.misc.castep.utils import CastepInputGenerator, CastepStaticSetGenerator
from jobflow import Flow
from jobflow_remote import submit_flow
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

parser=argparse.ArgumentParser()
parser.add_argument('--xc', type=str, default='PBE', help='XC functional')
parser.add_argument('--structure', type=str, default='Bi_1', help='structure file name in structure folder')
parser.add_argument('--kpoints', type=str, default='6 6 6', help='argument is specified in the form "a b c"')
parser.add_argument('--ENCUT', type=float, default= 800.0)
args=parser.parse_args()


atoms = read(f'{PROJECT_ROOT}/structures/{args.structure}.cell')
pmg_structure = AseAtomsAdaptor.get_structure(atoms)

static_job = BaseCastepMaker(
    input_set_generator=CastepStaticSetGenerator(
        user_param_settings={
        "task": 'SINGLEPOINT',
        #"relativistic_treatment":'DIRAC',
        #"spin_treatment":'vector',
        #"spin_orbit_coupling": True,
        'cut_off_energy': args.ENCUT,
        'xc_functional': args.xc,
        'max_scf_cycles': 100,
        },
        user_cell_settings={
        "kpoints_mp_grid": args.kpoints,
        "species_pot": [("Bi", 'SOC19')],
        # 'kpoint_mp_offset': '0.0 0.0 0.0',
        }
    )
).make(structure=pmg_structure)

static_flow = Flow(static_job, name=f'{args.xc}_{args.kpoints}_{args.ENCUT}',  output=static_job.output)

#resources = {"qverbatim": "#$ -cwd\n#$ -pe smp 64\n#$ -N Autoplex_jf_test\n#$ -o $JOB_ID.log\n#$ -e $JOB_ID.err\n#$ -P cpu\n#$ -l s_rt=05:00:00"}
resources = {"qverbatim": "#$ -cwd\n#$ -pe smp 64\n#$ -N Autoplex_jf_test\n#$ -o $JOB_ID.log\n#$ -e $JOB_ID.err\n#$ -P highmem\n#$ -l s_rt=05:00:00"}

print(submit_flow(static_flow , worker="autoplex_project_worker", resources=resources, project="autoplex_project"))
