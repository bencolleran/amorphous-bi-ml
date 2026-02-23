import random
import sys
from pathlib import Path

PROJECT_ROOT=Path(__file__).parents[0]

def read_extxyz(filename):
    frames = []
    with open(filename, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break  # EOF

            natoms = int(line.strip())
            comment = f.readline()

            atoms = [f.readline() for _ in range(natoms)]
            frames.append([line, comment] + atoms)

    return frames


def write_extxyz(filename, frames):
    with open(filename, "w") as f:
        for frame in frames:
            f.writelines(frame)


if __name__ == "__main__":
    input_file = f"{PROJECT_ROOT}/lammps.extxyz"

    frames = read_extxyz(input_file)
    random.seed(42)
    random.shuffle(frames)

    split_index = int(0.9 * len(frames))

    train_frames = frames[:split_index]
    test_frames = frames[split_index:]

    write_extxyz(f"{PROJECT_ROOT}/lammps_train.extxyz", train_frames)
    write_extxyz(f"{PROJECT_ROOT}/lammps_test.extxyz", test_frames)

    print(f"Total frames: {len(frames)}")
    print(f"Train: {len(train_frames)}")
    print(f"Test: {len(test_frames)}")