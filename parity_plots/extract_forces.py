#run from terminal like: python extract_energies.py quip_test.extxyz 

import sys, csv

f = open(sys.argv[1])
out = csv.writer(open("forces.csv", "w"))
out.writerow((
    "frame", "atom",
    "fx", "fy", "fz",
    "REF_fx", "REF_fy", "REF_fz"
))

frame = 0

while True:
    nat = f.readline()
    if not nat:
        break

    n = int(nat.split()[0])     # number of atoms
    header = f.readline()       # skip header line

    for i in range(n):
        line = f.readline().split()

        # extxyz layout:
        # 0   1  2  3    4  5  6        7  8  9
        # sym x  y  z  REF_fx REF_fy REF_fz   fx fy fz

        REF_fx = float(line[4])
        REF_fy = float(line[5])
        REF_fz = float(line[6])

        fx = float(line[7])
        fy = float(line[8])
        fz = float(line[9])

        out.writerow((
            frame, i,
            fx, fy, fz,
            REF_fx, REF_fy, REF_fz
        ))

    frame += 1

f.close()
print("Wrote forces.csv")
