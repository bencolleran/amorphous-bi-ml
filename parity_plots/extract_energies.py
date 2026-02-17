#run from terminal like: python extract_energies.py quip_test.extxyz 

import sys, re, csv

f = open(sys.argv[1])
out = csv.writer(open("energies.csv","w"))
out.writerow(("frame","energy","REF_energy"))

frame = 0
num_re = re.compile(
    r'REF_energy\s*=\s*([-\d.eE+]+)|\benergy\s*=\s*([-\d.eE+]+)'
)

while True:
    nat = f.readline()
    if not nat:
        break

    n = int(nat.split()[0])          # number of atoms
    header = f.readline() or ""

    ref = en = None
    for m in re.finditer(num_re, header):
        if m.group(1):
            ref = float(m.group(1))
        if m.group(2):
            en = float(m.group(2))

    # divide by number of atoms
    en_pa  = en  / n if en  is not None else None
    ref_pa = ref / n if ref is not None else None

    out.writerow((frame, en_pa, ref_pa))

    # skip atom lines
    for _ in range(n):
        f.readline()

    frame += 1

f.close()
print("Wrote energies.csv")