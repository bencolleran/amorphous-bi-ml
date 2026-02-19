import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0]
def read_lammps_log(filename):
    data=[]
    headers=None
    with open(f"{PROJECT_ROOT}/{filename}") as f:
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

    if "Step" in df.columns:
        df["Step"] = df["Step"].astype(int)
        df = df.set_index("Step")

    return df

df =read_lammps_log("20260127_123259/549164.log")
#print(df.columns)
print(df[["Temp",'c_MSD[4]']].head())
Temp=df["Temp"].to_numpy()
MeanSquare=df["c_MSD[4]"].to_numpy()
import matplotlib.pyplot as plt
x=Temp
y=MeanSquare
plt.xlabel("T / K")
plt.ylabel("MSD / A")
plt.scatter(x,y)
plt.savefig(f"{PROJECT_ROOT}/images/plot", dpi=600, bbox_inches="tight")