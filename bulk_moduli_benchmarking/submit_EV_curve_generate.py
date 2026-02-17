from eos import CommonEosMaker
from atomate2.common.jobs.eos import PostProcessEosEnergy
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
parser.add_argument('--kpoints', type=str, default='8 8 8', help='argument is specified in the form "a b c"')
parser.add_argument('--ENCUT', type=float, default=1100)
args=parser.parse_args()


atoms = read(f'{PROJECT_ROOT}/structures/{args.structure}.cell')
pmg_structure = AseAtomsAdaptor.get_structure(atoms)

intial_geom_opt = BaseCastepMaker(
    input_set_generator=CastepStaticSetGenerator(
        user_param_settings={
        "task": 'GEOMETRYOPTIMIZATION',
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
)

final_geom_opt = CastepStaticMaker(
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
        # 'kpoint_mp_grid': '1 1 1',
        # 'kpoint_mp_offset': '0.0 0.0 0.0',
        }
    )
)

eos_maker = CommonEosMaker(
    name=f'EOS_{args.xc}_SOcoupled',
    initial_relax_maker=intial_geom_opt,
    eos_relax_maker=final_geom_opt,
    static_maker=None,
    linear_strain=(-0.05,0.05),
    number_of_frames=10,
    postprocessor=PostProcessEosEnergy()
)
resources = {"qverbatim": "#$ -cwd\n#$ -pe smp 64\n#$ -N Autoplex_jf_test\n#$ -o $JOB_ID.log\n#$ -e $JOB_ID.err\n#$ -P cpu\n#$ -l s_rt=05:00:00"}
#resources = {"qverbatim": "#$ -cwd\n#$ -pe smp 64\n#$ -N Autoplex_jf_test\n#$ -o $JOB_ID.log\n#$ -e $JOB_ID.err\n#$ -P highmem\n#$ -l s_rt=05:00:00"}

print(submit_flow(eos_maker.make(pmg_structure), worker="autoplex_project_worker", resources=resources, project="autoplex_project"))
