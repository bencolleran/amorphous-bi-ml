from ase.build import bulk
from pymatgen.io.ase import AseAtomsAdaptor
from autoplex.misc.castep.jobs import BaseCastepMaker, CastepStaticMaker
from autoplex.misc.castep.utils import CastepInputGenerator, CastepStaticSetGenerator
from jobflow import Flow
from jobflow_remote import submit_flow

atoms = bulk("Si", "diamond", a=5.1)
pmg_structure = AseAtomsAdaptor.get_structure(atoms)

static_job = CastepStaticMaker(
    name="test_static",
    input_set_generator=CastepStaticSetGenerator(
        user_param_settings={
        'cut_off_energy': 300.0,
        'xc_functional': 'PBE',
        'max_scf_cycles': 100,
        },
        user_cell_settings={
        'kpoint_mp_spacing': 0.07
        # 'kpoint_mp_grid': '1 1 1',
        # 'kpoint_mp_offset': '0.0 0.0 0.0',
        }
    )
).make(structure=pmg_structure)

static_flow = Flow(static_job, output=static_job.output)

resources = {"qverbatim": "#$ -cwd\n#$ -pe smp 64\n#$ -N Autoplex_jf_test\n#$ -o $JOB_ID.log\n#$ -e $JOB_ID.err\n#$ -P cpu\n#$ -l s_rt=05:00:00"}
#resources = {"qverbatim": "#$ -cwd\n#$ -pe smp 64\n#$ -N Autoplex_jf_test\n#$ -o $JOB_ID.log\n#$ -e $JOB_ID.err\n#$ -P highmem\n#$ -l s_rt=05:00:00"}
# resources = {"qverbatim": "#SBATCH --reservation=castep_workshop", 
#              "nodes": 1, 
#              "ntasks_per_node": 16, 
#              "partition": "short", 
#              "time": "00:20:00"}

print(submit_flow(static_flow , worker="autoplex_project_worker", resources=resources, project="autoplex_project"))
