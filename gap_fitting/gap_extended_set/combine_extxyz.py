from ase.io import read, write
from pathlib import Path

PROJECT_ROOT=Path(__file__).parents[0]

def combine(file1,file2,file3):
    file1 = f"{PROJECT_ROOT}/{file1}.extxyz"
    file2 = f"{PROJECT_ROOT}/{file2}.extxyz"

    frames1 = read(file1, ":")
    frames2 = read(file2, ":")

    combined = frames1 + frames2

    for frame in combined:
        if "initial_charges" in frame.arrays:
            frame.calc = None
            frame.arrays["charge"]=frame.arrays["initial_charges"]
            del frame.arrays["initial_charges"]
    write(f"{PROJECT_ROOT}/{file3}.extxyz", combined, format="extxyz")
    return

combine("test","lammps_test","new_test")
combine("train","lammps_train","new_train")