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
parser.add_argument('--name', type=str, default=None, help='job name override')
args=parser.parse_args()

job_name=args.name or f'geometry_optimisation_{args.xc}'#the data was renamed after copying and pasting


atoms = read(f'{PROJECT_ROOT}/structures/Bi_1.cell')
pmg_structure = AseAtomsAdaptor.get_structure(atoms)

static_job = CastepStaticMaker(
    name=job_name,
    input_set_generator=CastepStaticSetGenerator(
        user_param_settings={
        "task": "GEOMETRYOPTIMIZATION",
        'cut_off_energy': 800.0,
        'xc_functional': args.xc,
        'max_scf_cycles': 100,
        },
        user_cell_settings={
        "kpoints_mp_grid": '3 3 3'
        # 'kpoint_mp_grid': '1 1 1',
        # 'kpoint_mp_offset': '0.0 0.0 0.0',
        }
    )
).make(structure=pmg_structure)

#Functionals: PBE, PBESOL, RSCAN, LDA
#path to castep info: /u/vld/sedm7085/autoplex/src/autoplex/misc/castep/castep_keywords.json
#Allowed values: LDA, PW91, PBE, B86BPBE, PBESOL, RPBE, WC, BLYP, LDA-C, LDA-X, ZERO, HF, PBE0, B3LYP, HSE03, HSE06, EXX-X, HF-LDA, EXX, EXX-LDA, SHF, SX, SHF-LDA, SX-LDA, WDA, SEX, SEX-LDA, RSCAN.
static_flow = Flow(static_job, output=static_job.output)

resources = {"qverbatim": "#$ -cwd\n#$ -pe smp 64\n#$ -N Autoplex_jf_test\n#$ -o $JOB_ID.log\n#$ -e $JOB_ID.err\n#$ -P cpu\n#$ -l s_rt=05:00:00"}

print(submit_flow(static_flow , worker="autoplex_project_worker", resources=resources, project="autoplex_project"))
