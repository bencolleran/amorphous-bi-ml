import pandas as pd
from pathlib import Path
import glob
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[0]
def read_lammps_log(name):
    data=[]
    headers=None
    file=glob.glob(f"{PROJECT_ROOT}/{name}/*.log")[0]
    print(file)
    with open(file) as f:
        for line in f:
            line=line.strip()
            if not line:
                continue

            if line.startswith("Step"):
                headers=line.split()
                continue
            
            if headers:
                parts=line.split()
                if len(parts)==len(headers):
                    try:
                        row=[float(x) for x in parts]
                        data.append(row)
                    except ValueError:
                        pass
    df = pd.DataFrame(data, columns=headers)

    # if "Step" in df.columns:
    #     df["Step"] = df["Step"].astype(int)
    #     df = df.set_index("Step")

    return df

pot1 = "20260212_152203"
pot2 = "20260223_154848"

name=pot1
df =read_lammps_log(name)
#print(df.columns)
Temp=df["Temp"].to_numpy()
Step=df["Step"].to_numpy()
import matplotlib.pyplot as plt
x=Step
y=Temp
plt.ylabel("T / K")
plt.ylabel("time / fs")
plt.scatter(x,y)
plt.savefig(f"{PROJECT_ROOT}/images/{name}_temperature_variance", dpi=600, bbox_inches="tight")

print(f"Temperature variation in Liquid is {np.var(Temp[:300])}")
print(f"Temperature variation in cooling is {np.var(Temp[300:500])}")
print(f"Temperature variation in solid is {np.var(Temp[500:800])}")

#temperature variation with new potential is smaller