from ase.build import bulk
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from autoplex.settings import RssConfig
from autoplex.auto.rss.flows import RssMaker
from autoplex.misc.castep.jobs import BaseCastepMaker, CastepStaticMaker
from autoplex.misc.castep.utils import CastepInputGenerator, CastepStaticSetGenerator
from jobflow import Flow
from jobflow_remote import submit_flow


castep_maker = CastepStaticMaker(
    name="static_castep",
    input_set_generator=CastepStaticSetGenerator(
        user_param_settings={
            "cut_off_energy": 600.0,
            "xc_functional": "PBE",
            "max_scf_cycles": 1000,
            "elec_energy_tol": 1e-7,
            "smearing_scheme": "Gaussian",
            "smearing_width": 0.05,   
            "spin_polarized": False,
            "finite_basis_corr": "automatic",
            "mixing_scheme": "Pulay",
            "mix_charge_amp": 0.6
        },
        user_cell_settings={
            "kpoint_mp_spacing": 0.03,
        },
    ),
)

rss_config=RssConfig.from_file('/u/vld/sedm7085/project/autoplex/aSi_rss.yaml')

rss_job=RssMaker(
    name='Si_rss',
    rss_config=rss_config,
    static_energy_maker=castep_maker,
    static_energy_maker_isolated_atoms=castep_maker
    ).make()#constructs a Flow object, so is equivalent to task1,2


resources = {"qverbatim": "#$ -cwd\n#$ -pe smp 64\n#$ -N Autoplex_jf_test\n#$ -o $JOB_ID.log\n#$ -e $JOB_ID.err\n#$ -P cpu\n#$ -l s_rt=05:00:00"}
# resources = {"qverbatim": "#SBATCH --reservation=castep_workshop", 
#              "nodes": 1, 
#              "ntasks_per_node": 16, 
#              "partition": "short", 
#              "time": "00:20:00"}

print(submit_flow(rss_job , worker="autoplex_project_worker", resources=resources, project="autoplex_project"))
