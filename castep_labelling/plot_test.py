from jobflow_remote.jobs.jobcontroller import JobController
import re
import json
import numpy as np

def get_job_output_dirs(db_id):
    jc = JobController.from_project_name("autoplex_project")
    doc=jc.get_jobs_info(db_ids=[str(db_id)])
    match = re.search(r"run_dir='([^']+)'", str(doc[0]))
    if match:
        run_dir = match.group(1)
        return f'{run_dir}/remote_job_data.json'

num_jobs=9
flow_id=5257
tests=[get_job_output_dirs(i) for i in range(flow_id,flow_id+num_jobs)]

energy={}
energies=[]

for test in tests:
        with open(test,'r') as f:
            data=json.load(f)
            energy[data[0]["name"]]=data[0]['output']['output']["energy_per_atom"]
            energies.append(data[0]['output']['output']["energy_per_atom"])
            #print(data[0]["name"])

one=energies[:3]
two=energies[3:6]
three=energies[6:]
x=[1,16,27]
print(one,two,three)

# One=[one[i]-energies[2] for i in range(3)]
# Two=[two[i]-energies[5] for i in range(3)]
# Three=[three[i]-energies[8] for i in range(3)]

n=1000

One=[(one[i]-energies[0])*n for i in range(3)]
Two=[(two[i]-energies[3])*n for i in range(3)]
Three=[(three[i]-energies[6])*n for i in range(3)]
print(One,Two,Three)


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
# # Plot all three
# plt.plot(x, One, label="Dataset 1")
# plt.plot(x, Two, label="Dataset 2")
# plt.plot(x, Three, label="Dataset 3")
# #plt.yscale("log")
# plt.xlabel("number of kpoints per dimension")
# plt.ylabel("energy / meV")
# plt.legend()
# plt.show()
# plt.savefig("test")

names=['one','two','three']
for j,energy in enumerate([one,two,three]):
    energy_dif=np.array([np.abs((energy[i+1]-energy[i]))*1000 for i in range(len(energy)-1)])
    plt.scatter(x[1:],energy_dif,label=f"structure {names[j]}",marker="D")
plt.xticks(x[1:])
plt.xlabel("number of kpoints")
plt.ylabel("energy change relative to the previous energy / meV")
plt.legend()
plt.show()
plt.savefig("test_dif")

#average time for 3x3x3 is average of 2 20, 2 21, 3 15 average is 2 28