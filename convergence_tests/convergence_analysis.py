
import matplotlib.pyplot as plt
from ase.eos import EquationOfState
import numpy as np
import json
from ase.io import read
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from jobflow_remote.jobs.jobcontroller import JobController
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

g_PBE_kpoints_800_small=['/u/vld/sedm7085/autoplex_out/86/aa/25/86aa256e-7c9c-4188-ba37-1df2f95ced33_1/remote_job_data.json',
       '/u/vld/sedm7085/autoplex_out/12/d7/42/12d742e7-3247-435f-b642-653131cd7db0_1/remote_job_data.json',
       '/u/vld/sedm7085/autoplex_out/4c/a3/5d/4ca35d9b-27fa-48ba-ba86-5dd55e650292_1/remote_job_data.json',
       '/u/vld/sedm7085/autoplex_out/17/55/3f/17553fbf-974a-4bcf-b1cf-d27536d7eefc_1/remote_job_data.json',
       '/u/vld/sedm7085/autoplex_out/3b/2d/69/3b2d6988-d7a1-4de2-a0a3-a48c33141a26_1/remote_job_data.json',
       '/u/vld/sedm7085/autoplex_out/9f/d0/32/9fd03253-0c48-4848-a4b7-20dc5c731bf2_1/remote_job_data.json',
       '/u/vld/sedm7085/autoplex_out/f9/24/6c/f9246c75-34b2-4d93-9026-062827f28df7_1/remote_job_data.json',
       '/u/vld/sedm7085/autoplex_out/6c/c4/bc/6cc4bc4f-c60d-4000-8b13-f6ac24e3107c_1/remote_job_data.json',
       '/u/vld/sedm7085/autoplex_out/26/19/11/261911ba-1f48-4d4d-b221-b8613bab3a88_1/remote_job_data.json'
]

g_PBE_encut_666_small=['/u/vld/sedm7085/autoplex_out/c8/6d/67/c86d671f-5b47-4be2-8eb8-f62e54583cf1_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/37/58/b6/3758b6c5-c8f2-4de3-a9d3-69015abe2a2a_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/b3/38/2d/b3382dff-7dca-4be7-aabc-b5803bd0b65a_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/8d/d6/4f/8dd64fbd-b742-4a17-89ce-8899225dbc0d_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/a3/58/d2/a358d2e5-2a76-4b55-8535-c3f64309264b_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/1d/b4/2a/1db42a18-b022-40eb-9777-e34aecfac158_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/e9/52/77/e95277ea-07f6-4b7b-bd15-9e189534dfb1_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/70/9d/2c/709d2cbf-0573-4fef-8ba5-b3369a8c0931_1/remote_job_data.json',
        ]

g_PBE_encut_666=['/u/vld/sedm7085/autoplex_out/2f/6d/97/2f6d979c-a19b-44fe-ab12-e2f377d7549e_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/e9/e6/36/e9e636f8-8a4b-43a0-8a01-f2d10214e887_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/12/f5/a4/12f5a408-8177-4d15-b21f-558f708faac7_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/3c/f8/d2/3cf8d295-b3af-40f5-b9dd-dd2e16511cb2_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/97/49/77/97497705-5ba4-4cc3-88c0-b956d37fda3f_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/fa/2f/b2/fa2fb24a-12af-43f4-8144-0474d357e653_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/f5/68/f8/f568f819-370c-43d0-83af-c508cf3b415e_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/cf/c3/70/cfc3702b-193e-46d9-b432-696ef8fb10b0_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/5d/62/60/5d626030-e29c-4623-8b84-f559a524acf9_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/70/e7/b3/70e7b30b-6434-4886-b779-770cefb241e2_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/29/e6/15/29e6155a-95e8-410e-97b5-02db492de127_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/79/8b/80/798b8002-8aeb-48ae-806d-b5540535e3e0_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/7f/94/56/7f94562b-5820-4e71-a2e8-61ec725d3267_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/7c/4d/31/7c4d31f3-53fe-4756-b407-54779381c104_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/68/24/51/6824518b-f2e7-4c48-97df-28e73a03e3bb_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/1d/b4/2a/1db42a18-b022-40eb-9777-e34aecfac158_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/e9/52/77/e95277ea-07f6-4b7b-bd15-9e189534dfb1_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/70/9d/2c/709d2cbf-0573-4fef-8ba5-b3369a8c0931_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/c3/a8/d1/c3a8d166-d73d-4ac1-a296-91af78a0428c_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/9d/6c/a3/9d6ca39f-cc29-4ab6-894a-b97ee633da00_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/ef/5e/51/ef5e5135-2837-4a20-9268-0aa6b8a985ec_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/18/22/57/18225755-7977-4bc5-bd73-1f30fdab125c_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/cd/c2/cb/cdc2cb85-f5d3-4b03-9765-37d7d2822b5d_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/5b/e3/31/5be33124-99d8-43d5-8989-72fd67f5c21f_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/d7/f4/c3/d7f4c3bc-7cec-4191-abd7-5e50d7c1d714_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/af/61/00/af610094-08ca-4160-aba1-a96123aab930_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/0d/1b/80/0d1b80e5-24c1-4325-a262-4fa2414466d2_1/remote_job_data.json',
        ]

g_RSCAN_encut_666=['/u/vld/sedm7085/autoplex_out/30/a8/1c/30a81c5d-bd04-46b1-af18-1e9887202b2f_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/13/b2/5a/13b25a6c-609c-4191-b609-93d7bd90c2c6_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/0a/cf/c1/0acfc116-bbf6-4b87-b181-6d78a93db85d_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/cf/0e/81/cf0e81c5-1422-4696-8d30-94a1c8715f9f_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/25/d1/8b/25d18bd3-b66f-47ae-8de8-8d0505627c8f_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/8f/fe/28/8ffe285e-8345-465e-b719-342b716d498e_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/08/ae/1c/08ae1c7f-9367-4a4b-bdb4-4e714973f068_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/6e/23/bb/6e23bbfc-d1c5-402b-b589-a9aa1bc5114e_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/b4/21/28/b4212850-b3ae-42cc-b963-f7707891b018_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/b4/e2/f0/b4e2f044-1298-4717-bef9-914cd07ab077_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/c3/31/b0/c331b019-ff47-4392-9807-5562cb75ce14_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/f4/b9/e8/f4b9e86a-e2d4-485c-b5b0-4408ea46e111_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/08/71/61/0871616c-8747-4aa3-88bd-d7b478621b6c_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/1f/2e/77/1f2e7757-2615-48bd-b49b-0a6145bdc118_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/24/a8/ea/24a8ea69-8ab5-4ae1-9cc2-75d71f0052e5_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/92/21/d0/9221d04f-7322-49f2-a20d-23cba3a1607b_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/80/ff/1a/80ff1a72-ddb5-43fe-b1e4-248afdb08d8a_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/00/26/c4/0026c44e-4bf5-473a-88f5-7eb53feb3a61_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/af/9e/29/af9e299e-3815-41e5-bf0a-10158ecf69ee_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/96/c9/f3/96c9f355-23e8-4f72-b77a-bd1b717169bd_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/a8/ed/f7/a8edf73e-5ef1-4478-a401-7afd03df57be_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/cb/88/7a/cb887af6-1518-4512-810c-319aed4805a9_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/a2/9e/92/a29e922a-0054-48b5-9b6e-8f0d7b369efb_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/d0/ce/15/d0ce152b-15e2-4438-85a7-9284e233161f_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/fb/ef/87/fbef87f6-1c34-43c6-892b-7243cc3da996_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/8f/48/b7/8f48b72f-ea29-4931-a55e-fae09fb9999a_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/13/62/40/136240ff-6780-4822-8fb7-23bfb797630a_1/remote_job_data.json',
]

g_PBE_1200_kpoints=['/u/vld/sedm7085/autoplex_out/3b/38/12/3b3812eb-7977-4759-a7ae-3046069ba8bc_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/d0/25/22/d02522ca-db66-45b9-a00b-1839ff4b6722_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/9c/a3/dc/9ca3dce3-2918-461a-8149-d5d6a5f4beaf_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/4f/7d/d3/4f7dd3a6-eec6-4939-8a48-eec569529826_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/ad/05/f8/ad05f883-d33a-47ab-ba60-1c221660fef2_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/4c/0d/ff/4c0dffdc-b5a3-479e-9a0e-aa0439806fef_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/03/89/e1/0389e1b7-5aff-4594-a07a-243af143c84b_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/96/a7/8b/96a78be2-9e3f-449b-a4bf-1bab12b423ca_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/75/7a/59/757a5963-1e98-477f-88bf-4c624af43498_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/1b/d2/70/1bd2700a-362a-4d6f-bbb7-3e30c49cda0c_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/b5/dc/cb/b5dccb96-c68d-45f0-b2ab-b30a3fc0cc51_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/6f/97/59/6f975955-7de3-4199-adb9-c4f444e82a98_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/45/41/06/454106f2-b22e-4af8-af68-4c73fa908923_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/85/2d/cb/852dcbc6-1953-44e2-ab29-39de72954e02_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/c3/d7/ff/c3d7ff56-8a4e-4349-b305-9367315daa69_1/remote_job_data.json',
]

g_RSCAN_1200_kpoints=['/u/vld/sedm7085/autoplex_out/20/92/4d/20924de5-ab50-41e1-bb51-571a14b8044c_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/05/1b/2c/051b2c96-edb8-43b8-b361-d1ff66f5f72d_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/32/f4/76/32f47657-67dd-4132-a554-c7ac353cca15_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/f8/10/f6/f810f6f3-89fa-4298-bd41-935d33221b70_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/8c/cd/d6/8ccdd630-6384-48ad-a942-71047cfbf16c_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/1b/9e/f8/1b9ef8ce-ac59-4b4a-a667-ddd81bf228d7_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/50/99/e6/5099e684-5253-4cbd-9ed7-863c9bdf7d87_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/47/d4/41/47d44192-1290-4a96-b1f5-2e6312b4db55_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/f6/83/e3/f683e390-d1de-480d-91b2-eec56e721723_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/6f/15/97/6f15970d-c307-4104-aae5-639c16f3b66e_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/39/6b/e9/396be977-ede4-44ec-b6e3-0d9487c99cb0_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/e5/d7/1f/e5d71ff4-0a3d-4fde-bd22-258776b0a085_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/f2/4f/a4/f24fa4fa-c6fb-4a89-9c69-06f2496209cb_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/9b/61/dc/9b61dcdd-24c5-4441-b07a-7c5389f09e70_1/remote_job_data.json',
        '/u/vld/sedm7085/autoplex_out/26/b9/5b/26b95b7a-aae5-4461-82e7-af76ed620cb2_1/remote_job_data.json',
]


def get_job_output_dirs(db_id):
    jc = JobController.from_project_name("autoplex_project")
    doc=jc.get_jobs_info(db_ids=[str(db_id)])
    match = re.search(r"run_dir='([^']+)'", str(doc[0]))
    if match:
        run_dir = match.group(1)
        return f'{run_dir}/remote_job_data.json'

PBE_encut_666=[get_job_output_dirs(i) for i in range(1968,1995)]
RSCAN_encut_666=[get_job_output_dirs(i) for i in range(1995,2022)]
PBE_1100_kpoints=[get_job_output_dirs(i) for i in range(2022,2037)]
RSCAN_1100_kpoints=[get_job_output_dirs(i) for i in range(2037,2052)]

np.set_printoptions(precision=4, suppress=True)
lst=[g_PBE_encut_666,g_RSCAN_encut_666,g_PBE_1200_kpoints,g_RSCAN_1200_kpoints]#old
lst1=[PBE_encut_666,RSCAN_encut_666,PBE_1100_kpoints,RSCAN_1100_kpoints]

filenames=lst1[0]

energy, encut, xc, kpoints, forces =([] for i in range(5))
for file in filenames:
    with open(file,'r') as f:
        data=json.load(f)
    energy.append(data[0]['output']['output']["energy_per_atom"])
    encut.append(data[0]['output']['input']['input_set']['param']['cut_off_energy'])
    xc.append(data[0]['output']['input']['input_set']['param']['xc_functional'])
    kpoints.append(data[0]['output']['input']['input_set']['cell']['kpoints_mp_grid'])
    forces.append(data[0]['output']['output']["forces"])

def name(file):
    with open(file[0],'r') as f:
        data=json.load(f)
    encut_1=(data[0]['output']['input']['input_set']['param']['cut_off_energy'])
    xc_1=(data[0]['output']['input']['input_set']['param']['xc_functional'])
    kpoints_1=(data[0]['output']['input']['input_set']['cell']['kpoints_mp_grid'])[0]
    if encut_1==1100:
        return f'{xc_1}_{encut_1}_kpoints'
    else:
        return f'{xc_1}_encut_{kpoints_1}{kpoints_1}{kpoints_1}'

energy_dif=np.array([np.abs((energy[i+1]-energy[i]))*1000 for i in range(len(energy)-1)])#plot against Encut[1:]
#energy_perc=np.array([np.abs((energy_dif[i+1]-energy_dif[i]))*100 for i in range(len(energy)-2)])#plot against Encut[2:]#BS not percentage
energy_ref=np.array([np.abs(energy[i]-energy[-1])*1000 for i in range(len(energy))])
forces_rmse_ref=np.array([(np.sum([(np.linalg.norm(np.array(forces[j][i])-np.array(forces[-1][i])))/np.sqrt(6) for i in range(6)])/6) for j in range(len(forces))])


def plot_graph_encut(y_var):
    filepath=f'{PROJECT_ROOT}/convergence_tests/graphs/'
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(100))
    plt.yscale('log')
    if y_var[0]==energy_dif[0]:
        plt.grid(True)
        plt.xlabel('cut off energy / eV')
        plt.ylabel('log of energy change /meV')
        plt.scatter(encut[1:],energy_dif)
        plt.plot(np.linspace(300,1600,(len(energy)-2)),np.ones(len(energy)-2))
        plt.savefig(f"{filepath}{name(filenames)}_energy_dif.png", dpi=600, bbox_inches="tight")
    #elif y_var[0]==energy_perc[0]:
    #    plt.grid(True)
    #    plt.xlabel('cut off energy / eV')
    #    plt.ylabel('log of percentage energy change /%')
    #    plt.scatter(encut[2:],energy_perc)
    #    plt.plot(np.linspace(300,1600,(len(energy)-2)),np.ones(len(energy)-2)/10)
    #    plt.savefig(f"{filepath}{name(filenames)}_energy_perc.png", dpi=600, bbox_inches="tight")
    elif y_var[0]==energy_ref[0]:
        plt.grid(True)
        plt.xlabel('cut off energy / eV')
        plt.ylabel('log of energy relative to final energy /meV')
        plt.scatter(encut,energy_ref)
        plt.plot(np.linspace(300,1600,(len(energy)-2)),np.ones(len(energy)-2))
        plt.plot(np.linspace(300,1600,(len(energy)-2)),np.ones(len(energy)-2)/10)
        plt.savefig(f"{filepath}{name(filenames)}_energy_ref.png", dpi=600, bbox_inches="tight")
    elif y_var[0]==forces_rmse_ref[0]:
        plt.grid(True)
        plt.xlabel('cut off energy / eV')
        plt.ylabel('log of rmse forces relative to final forces per atom / eV/A')
        plt.scatter(encut,forces_rmse_ref)
        #plt.plot(np.linspace(300,1600,(len(energy)-2)),np.ones(len(energy)-2))
        plt.savefig(f"{filepath}{name(filenames)}_forces_rmse_ref.png", dpi=600, bbox_inches="tight")

def plot_graph_kpoints(y_var):
    kpoints=list(range(6,21))
    filepath=f'{PROJECT_ROOT}/convergence_tests/graphs/'
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator())
    plt.yscale('log')
    if y_var[0]==energy_dif[0]:
        plt.grid(True)
        plt.xlabel('kpoints')
        plt.ylabel('log of energy change /meV')
        plt.scatter(kpoints[1:],energy_dif)
        plt.plot(np.linspace(6,20,(len(energy)-2)),np.ones(len(energy)-2))
        plt.savefig(f"{filepath}{name(filenames)}_energy_dif.png", dpi=600, bbox_inches="tight")
    #elif y_var[0]==energy_perc[0]:
    #    plt.grid(True)
    #    plt.xlabel('kpoints')
    #    plt.ylabel('log of percentage energy change')
    #    plt.scatter(kpoints[2:],energy_perc)
    #    plt.plot(np.linspace(6,20,(len(energy)-2)),np.ones(len(energy)-2))
    #    plt.savefig(f"{filepath}{name(filenames)}_energy_perc.png", dpi=600, bbox_inches="tight")
    elif y_var[0]==energy_ref[0]:
        plt.grid(True)
        plt.xlabel('kpoints')
        plt.ylabel('log of energy relative to final energy /meV')
        plt.scatter(kpoints,energy_ref)
        plt.plot(np.linspace(6,20,(len(energy)-2)),np.ones(len(energy)-2))
        plt.savefig(f"{filepath}{name(filenames)}_energy_ref.png", dpi=600, bbox_inches="tight")
    elif y_var[0]==forces_rmse_ref[0]:
        plt.grid(True)
        plt.xlabel('kpoints')
        plt.ylabel('log of rmse forces relative to final forces per atom / eV/A')
        plt.scatter(kpoints,forces_rmse_ref)
        plt.plot(np.linspace(6,20,(len(energy)-2)),np.ones(len(energy)-2))
        plt.savefig(f"{filepath}{name(filenames)}_forces_rmse_ref.png", dpi=600, bbox_inches="tight")


encut_change=True
if encut_change==True:
    plot_graph_encut(energy_ref)
    #plot_graph_encut(energy_perc)
    plot_graph_encut(energy_dif)
    plot_graph_encut(forces_rmse_ref)
else:
    plot_graph_kpoints(energy_ref)
    #plot_graph_kpoints(energy_perc)
    plot_graph_kpoints(energy_dif)
    plot_graph_kpoints(forces_rmse_ref)
#had to rerun because all jobs were geometry optimizations