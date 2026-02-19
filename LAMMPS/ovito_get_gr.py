#!/usr/bin/env python3
from pathlib import Path
import glob
import numpy as np
import matplotlib.pyplot as plt
from ovito.io import import_file
from ovito.modifiers import CoordinationAnalysisModifier, TimeAveragingModifier

# -------- user config ----------
PROJECT_ROOT = Path(__file__).resolve().parents[0]
name = "20260205_162918"        # your folder name
name = "20260205_122512"
files = sorted(glob.glob(f"{PROJECT_ROOT}/{name}/NVT/*.dat"))
cutoff = 3.4
cutoff=10
n_bins = 200
out_prefix = PROJECT_ROOT / f"{name}_rdf_timeavg"
# -------------------------------

# keep only files [i_start : i_end)  (Python slicing rules)
# i_start = 1 #inclusive
# i_end   = 201 #inclusive 201
# files = files[i_start:(i_end+1)]

i_start = 500 #inclusive
i_end   = 600 #inclusive 201
files = files[i_start:(i_end+1)]
out_prefix=f"{out_prefix}_amorphous"


pipeline = import_file(files, multiple_frames=True)

# 1) make RDF per frame
pipeline.modifiers.append(
    CoordinationAnalysisModifier(
        cutoff=cutoff,
        number_of_bins=n_bins,
    )
)
pipeline.modifiers.append(
    TimeAveragingModifier(
        operate_on = "table:coordination-rdf",)
)

data = pipeline.compute()
rdf_table = data.tables['coordination-rdf[average]'].xy()
arr = np.asarray(rdf_table)   # shape (n_bins, 2)
r  = arr[:, 0]
gr = arr[:, 1]

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(r, gr, label="time-averaged g(r)")
ax.set_xlim(2,10)
ax.set_xlabel("r")
ax.set_ylabel("g(r)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{PROJECT_ROOT}/images/{out_prefix}_avg.png", dpi=300, bbox_inches="tight")
plt.show()
