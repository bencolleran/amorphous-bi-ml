from jobflow_remote.jobs.jobcontroller import JobController
import re,json
from datetime import datetime
from datetime import datetime
import numpy as np
from pathlib import Path
import gzip
from ase.io import read,write
from quippy.potential import Potential
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", module="ase.calculators.castep")
warnings.filterwarnings("ignore", module="ase.io.castep")

PROJECT_ROOT = Path(__file__).resolve().parents[0]


jc = JobController.from_project_name("autoplex_project")
database_ids = [f.db_ids for f in jc.get_flows_info()]

num_completed=0
run_time=[]
completed=[]
jc.get_jobs_info(db_ids=database_ids[-1])
for id in database_ids[-1]:
    if str(jc.get_job_info(db_id=id).state)=="JobState.COMPLETED":
        num_completed+=1
        completed.append(id)

print(num_completed)
#print(np.mean(run_time))

energy=[]
forces=[]
structures=[]
for id in completed:
    run_dir=jc.get_job_info(db_id=id).run_dir
    file=f'{run_dir}/remote_job_data.json'
    with open(file,'r') as f:
        data=json.load(f)
        energy.append(data[0]['output']['output']["energy_per_atom"])
        forces.append(data[0]['output']['output']["forces"])
    structure=f'{run_dir}/CASTEP/castep.cell.gz'
    with gzip.open(structure, "rt") as f: 
        atoms=read(structure)

forces=np.array(forces).ravel()
energy=np.array(energy)

#make a function that gets all the run directories for the dataset

jc = JobController.from_project_name("autoplex_project")
database_ids = [f.db_ids for f in jc.get_flows_info()]

atoms_list=[]
jc.get_jobs_info(db_ids=database_ids[-1])
for id in database_ids[-1]:
    if str(jc.get_job_info(db_id=id).state)=="JobState.COMPLETED":
        run_dir=jc.get_job_info(db_id=id).run_dir
        structure=f'{run_dir}/CASTEP/castep.cell.gz'
        info=f'{run_dir}/remote_job_data.json'
        atoms = read(structure, format="castep-cell")
        with open(info,'r') as f:
            data=json.load(f)
        #config_type=initial rss_group=traj energy_sigma=0.001 force_sigma=0.0316 virial_sigma=0.0632 pbc="T T T"
        atoms.arrays['REF_forces']=np.asarray(data[0]['output']['output']["forces"])
        atoms.arrays['charge']=np.asarray([site["properties"]["charge"] for site in data[0]['output']['structure']["sites"]])
        stress=np.array(data[0]['output']['output']["stress"])
        volume=data[0]['output']['structure']['lattice']['volume']
        GPA_to_eV_per_A3 = 0.006241509
        atoms.info['REF_virial']=(-volume*stress*GPA_to_eV_per_A3/len(atoms)).flatten()#per atom virial tensor
        atoms.info['REF_energy']=data[0]['output']['output']["energy_per_atom"]*len(atoms) #total energy of cell
        # atoms.info['config_type']='lammps'
        # atoms.info['rss_group']='lammps'
        # atoms.info['energy_sigma']=0
        # atoms.info['force_sigma']=0
        # atoms.info['virial_sigma']=0
        atoms.info['pbc']="T T T"
        del atoms.arrays["initial_magmoms"]
        del atoms.arrays["castep_labels"]
        write(f"{PROJECT_ROOT}/lammps.extxyz", atoms, format="extxyz",append=True)

