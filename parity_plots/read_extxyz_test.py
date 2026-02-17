# run from terminal like: python extract_energies.py quip_test.extxyz

import sys, re, csv
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python extract_energies.py <file.extxyz>")
    sys.exit(1)

infile = sys.argv[1]

f = open(infile)
out = csv.writer(open("lattices.csv", "w"))
# add lattice and volume columns
out.writerow(("frame", "energy_per_atom", "REF_energy_per_atom", "volume", "lattice"))

frame = 0
num_re = re.compile(
    r'REF_energy\s*=\s*([-\d.eE+]+)|\benergy\s*=\s*([-\d.eE+]+)'
)
# capture Lattice= either "..." or unquoted
lattice_re = re.compile(r'Lattice\s*=\s*"(.*?)"|Lattice\s*=\s*([-\d.eE+\s]+)', re.IGNORECASE)

while True:
    nat = f.readline()
    if not nat:
        break

    try:
        n = int(nat.split()[0])          # number of atoms
    except Exception:
        # malformed line: skip / break
        break

    header = f.readline() or ""

    ref = en = None
    for m in re.finditer(num_re, header):
        if m.group(1):
            ref = float(m.group(1))
        if m.group(2):
            en = float(m.group(2))

    # attempt to find lattice
    lattice_str = None
    mlat = lattice_re.search(header)
    if mlat:
        lattice_str = mlat.group(1) if mlat.group(1) is not None else mlat.group(2)

    lattice_values = None
    volume = None
    if lattice_str:
        # split on whitespace or commas
        parts = re.split(r'[\s,]+', lattice_str.strip())
        # filter out any empty parts
        parts = [p for p in parts if p != ""]
        try:
            vals = [float(x) for x in parts]
            if len(vals) == 9:
                # assume row-major: a1 a2 a3  b1 b2 b3  c1 c2 c3
                lattice_values = np.array([[vals[0], vals[1], vals[2]],
                                           [vals[3], vals[4], vals[5]],
                                           [vals[6], vals[7], vals[8]]], dtype=float)
                # compute absolute value of determinant for volume
                volume = float(abs(np.linalg.det(lattice_values)))
            else:
                # not 9 numbers: leave lattice_values None
                lattice_values = None
                volume = None
        except Exception:
            lattice_values = None
            volume = None

    # divide by number of atoms
    en_pa = en / n if en is not None else None
    ref_pa = ref / n if ref is not None else None

    # prepare lattice string for CSV (write as single-space list) or empty
    lattice_out = ""
    if lattice_values is not None:
        lattice_out = " ".join(f"{val:.8g}" for val in lattice_values.flatten())

    out.writerow((frame,
                  "" if en_pa is None else f"{en_pa:.12g}",
                  "" if ref_pa is None else f"{ref_pa:.12g}",
                  "" if volume is None else f"{volume:.12g}",
                  lattice_out))

    # skip atom lines
    for _ in range(n):
        f.readline()

    frame += 1

f.close()
print("Wrote volumes.csv")
