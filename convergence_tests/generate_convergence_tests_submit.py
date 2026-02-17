ecut=300
n=1600

kpoints=6
m=20

for i in range(ecut,n+50,50):
    print(f'python singlepoint.py --kpoints "6 6 6" --ENCUT {i} --xc RSCAN')


for i in range(ecut,n+50,50):
    print(f'python singlepoint.py --kpoints "6 6 6" --ENCUT {i} --xc PBE')


for i in range(kpoints,m+1):
    print(f'python singlepoint.py --kpoints "{i} {i} {i}" --ENCUT 1100 --xc PBE')


for i in range(kpoints,m+1):
    print(f'python singlepoint.py --kpoints "{i} {i} {i}" --ENCUT 1100 --xc RSCAN')
