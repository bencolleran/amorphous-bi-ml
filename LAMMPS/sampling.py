from pathlib import Path
from quippy import descriptors
from ase.io import read, write, lammpsdata, lammpsrun
import numpy as np
import glob,re,shutil

PROJECT_ROOT = Path(__file__).resolve().parents[0]

filepath_train=f"{PROJECT_ROOT}/../parity_plots/train.extxyz"
filepath_check=f"{PROJECT_ROOT}/Bi.data"
filepath_test=f"{PROJECT_ROOT}/../parity_plots/test.extxyz"
name="20260205_122512"
files = glob.glob(f"{PROJECT_ROOT}/{name}/NVT/dump_custom.Bi.*.dat")
files = sorted(files,key=lambda f: int(re.search(r'\.(\d+)\.dat$', f).group(1)))
# atoms_lammps = read(files[0], format="lammps-dump-text",specorder=["Bi"])#how to read lammps dump files
# atoms=lammpsdata.read_lammps_data(filepath_check,style="atomic")#how to read .data files
# training_set=read(filepath_train,index=":")#how to read .extxyz files

def fps(X, n_samples):#X is the list of soaps, n_samples is the number of structures wanted
    N = X.shape[0]

    selected = [0] 
    distances = np.linalg.norm(X - X[0], axis=1)

    for _ in range(1, n_samples):
        idx = np.argmax(distances)
        selected.append(idx)

        new_dist = np.linalg.norm(X - X[idx], axis=1)
        distances = np.minimum(distances, new_dist)

    return np.array(selected)

def fps_add_minimal(X, n_add, initial=None):
    X = np.asarray(X)
    N = X.shape[0]
    selected = list(initial).any() if initial else [0]
    if not initial:
        n_add-=1

    # distances to nearest selected
    d = np.full(N, np.inf)
    for s in selected:
        d = np.minimum(d, np.linalg.norm(X - X[s], axis=1))
    d[selected] = -np.inf#selected is a list of indices, see fancy indexing

    new = []
    for _ in range(n_add):
        i = int(np.argmax(d))
        new.append(i)
        d = np.minimum(d, np.linalg.norm(X - X[i], axis=1))
        d[i] = -np.inf

    return np.array(new), np.array(selected + new)#the second list is the same as the old fps function

def fps_correct(X, n_add, initial=None):
    X = np.asarray(X)
    N = X.shape[0]

    # If no initial vectors provided, start from X[0]
    if initial is None or len(initial) == 0:
        selected_vecs = X[[0]]
        n_add -= 1
    else:
        selected_vecs = np.asarray(initial)

    # compute distance to nearest selected vector
    d = np.full(N, np.inf)

    for v in selected_vecs:
        d = np.minimum(d, np.linalg.norm(X - v, axis=1))

    new_indices = []
    for _ in range(n_add):
        i = int(np.argmax(d))
        new_indices.append(i)

        new_vec = X[i]
        d = np.minimum(d, np.linalg.norm(X - new_vec, axis=1))
        d[i] = -np.inf
        print("added another")

    new_indices = np.array(new_indices)
    new_vecs = X[new_indices]

    all_vecs = np.vstack([selected_vecs, new_vecs])

    return new_vecs, all_vecs, new_indices

def label_set(filepath,file_out):
    soap=descriptors.Descriptor(
    "soap " \
    "l_max=10 " \
    "n_max=12 " \
    "atom_sigma=0.5 " \
    "cutoff=5.0 " \
    "cutoff_transition_width=1.0 " \
    "central_weight=1.0 " \
    "n_species=1 " \
    "species_Z={83} " \
    "average=False"
)
    dataset=read(filepath,index=":")#works for .extxyz filetypes
    num_structs=len(dataset)
    soaps=np.empty((num_structs,859))

    for i,struc in enumerate(dataset):
        per_atom_soap=soap.calc(dataset[i])["data"]
        avg=per_atom_soap.mean(axis=0)
        soaps[i]=avg
        if i%100==0:
           print("calculated another hundred")

    np.save(f"{PROJECT_ROOT}/soaps_{file_out}.npy",soaps)
    return

def label_set_idx(filepath,idx,file_out):
    soap=descriptors.Descriptor(
    "soap " \
    "l_max=10 " \
    "n_max=12 " \
    "atom_sigma=0.5 " \
    "cutoff=5.0 " \
    "cutoff_transition_width=1.0 " \
    "central_weight=1.0 " \
    "n_species=1 " \
    "species_Z={83} " \
    "average=False"
)
    dataset=read(filepath,index=":")#works for .extxyz filetypes
    num_structs=len(dataset)
    soaps=np.empty((num_structs,859))

    for i,struc in enumerate(dataset[:idx]):
        per_atom_soap=soap.calc(dataset[i])["data"]
        avg=per_atom_soap.mean(axis=0)
        soaps[i]=avg
        if i%100==0:
           print("calculated another hundred")
    np.save(f"{PROJECT_ROOT}/soaps_{file_out}.npy",soaps)
    return

def label_lammps(name,file_out=None):
    soap=descriptors.Descriptor(
    "soap " \
    "l_max=10 " \
    "n_max=12 " \
    "atom_sigma=0.5 " \
    "cutoff=5.0 " \
    "cutoff_transition_width=1.0 " \
    "central_weight=1.0 " \
    "n_species=1 " \
    "species_Z={83} " \
    "average=False"
)
    files = glob.glob(f"{PROJECT_ROOT}/{name}/NVT/dump_custom.Bi.*.dat")
    files = sorted(files,key=lambda f: int(re.search(r'\.(\d+)\.dat$', f).group(1)))
    soaps_lammps=np.empty((len(files),859))
    for i,file in enumerate(files):
        atoms_lammps = read(file, format="lammps-dump-text",specorder=["Bi"])
        per_atom_soap=soap.calc(atoms_lammps)["data"]
        avg=per_atom_soap.mean(axis=0)
        soaps_lammps[i]=avg
        if i%100==0:
            print("calculated another hundred")
    if not file_out:
        file_out=name
    np.save(f"{PROJECT_ROOT}/soaps_{file_out}.npy",soaps_lammps)
    return

# label_lammps("20260210_102745")
# label_lammps("20260210_130003")
# label_lammps("20260210_143203")
# label_lammps("20260210_154820")

#print(fps_correct(X=np.load(f"{PROJECT_ROOT}/soaps_20260210_102745.npy"),n_add=200,initial=np.load(f"{PROJECT_ROOT}/soaps_train.npy"))[2])

# label_lammps("20260212_152158")#0.8
# label_lammps("20260212_152203")#1.0
# label_lammps("20260212_152204")#0.9
# label_lammps("20260212_152218")#1.1
# label_lammps("20260212_152219")#1.2

a=np.load("soaps_20260212_152158.npy")#0.8
b=np.load("soaps_20260212_152203.npy")#1.0
c=np.load("soaps_20260212_152204.npy")#0.9
d=np.load("soaps_20260212_152218.npy")#1.1
e=np.load("soaps_20260212_152219.npy")#1.2
new_structures=np.vstack([a,b,c,d,e])

# print(fps_correct(X=new_structures,n_add=10,initial=np.load(f"{PROJECT_ROOT}/soaps_train.npy"))[2])

# def export_new_structures(indices):
#     filenames=['20260212_152158','20260212_152203','20260212_152204','20260212_152218','20260212_152219']
#     files=[f"/NVT/dump_custom.Bi.{i*100}.dat" for i in range(801)]

#     all_files=[]
#     for file_prefix in filenames:
#         for file_suffix in files:
#             all_files.append(f"{PROJECT_ROOT}/{file_prefix}{file_suffix}")
#     all_files=np.array(all_files)


#     # destination = Path(f"{PROJECT_ROOT}/new_directory")
#     # destination.mkdir(exist_ok=True)  # create folder if it doesn't exist
#     destination = "/u/vld/sedm7085/project/castep_labelling/LAMMPS_structures"

#     for file in all_files[indices]:
#         shutil.copy(file, destination)
#         print("moved_file")
#     return

def export_new_structures(indices):
    filenames=['20260212_152158','20260212_152203','20260212_152204','20260212_152218','20260212_152219']
    files=[f"/NVT/dump_custom.Bi.{i*100}.dat" for i in range(801)]

    all_files=[]
    for file_prefix in filenames:
        for file_suffix in files:
            all_files.append(f"{PROJECT_ROOT}/{file_prefix}{file_suffix}")
    all_files=np.array(all_files)

    # destination = Path(f"{PROJECT_ROOT}/new_directory")
    # destination.mkdir(exist_ok=True)  # create folder if it doesn't exist
    destination = "/u/vld/sedm7085/project/castep_labelling/LAMMPS_structures"

    from pathlib import Path
    import shutil

    for file in all_files[indices]:
        src = Path(file)
        dst = Path(destination) / src.name

        counter = 1
        while dst.exists():
            dst = Path(destination) / f"{src.stem}_{counter}{src.suffix}"
            counter += 1

        shutil.copy(src, dst)
        print("moved_file")

    return



#export_new_structures([0,2])

# To use for time cost for DFT, randomly selected structures
# import random
# random.seed(42)
# print(random.randint(0, 4004),random.randint(0, 4004),random.randint(0, 4004))
# export_new_structures([random.randint(0, 4004),random.randint(0, 4004),random.randint(0, 4004)])

#print(fps_correct(X=new_structures,n_add=10,initial=np.load(f"{PROJECT_ROOT}/soaps_train.npy"))[2])

export_new_structures(fps_correct(X=new_structures,n_add=100,initial=np.load(f"{PROJECT_ROOT}/soaps_train.npy"))[2])