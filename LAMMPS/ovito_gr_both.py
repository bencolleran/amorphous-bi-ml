# #!/usr/bin/env python3
# from pathlib import Path
# import glob
# import numpy as np
# import matplotlib.pyplot as plt
# from ovito.io import import_file
# from ovito.modifiers import CoordinationAnalysisModifier, TimeAveragingModifier

# name = "20260205_122512"
# # -------- user config ----------
# PROJECT_ROOT = Path(__file__).resolve().parents[0]
# #name = "20260205_162918"        # your folder name
# files = sorted(glob.glob(f"{PROJECT_ROOT}/{name}/NVT/*.dat"))
# cutoff = 3.4
# cutoff=10
# n_bins = 200

# # -------------------------------

# #keep only files [i_start : i_end)  (Python slicing rules)
# i_start = 1 #inclusive
# i_end   = 201 #inclusive 201
# files = files[i_start:(i_end+1)]

# pipeline = import_file(files, multiple_frames=True)

# # 1) make RDF per frame
# pipeline.modifiers.append(CoordinationAnalysisModifier(cutoff=cutoff,number_of_bins=n_bins,))
# pipeline.modifiers.append(TimeAveragingModifier(operate_on = "table:coordination-rdf",))
# data = pipeline.compute()
# rdf_table = data.tables['coordination-rdf[average]'].xy()
# arr = np.asarray(rdf_table)   # shape (n_bins, 2)
# r1  = arr[:, 0]
# gr1 = arr[:, 1]


# files = sorted(glob.glob(f"{PROJECT_ROOT}/{name}/NVT/*.dat"))
# cutoff = 3.4
# cutoff=10
# n_bins = 200
# # -------------------------------

# #keep only files [i_start : i_end)  (Python slicing rules)
# i_start = 500 #inclusive
# i_end   = 600 #inclusive 201
# files = files[i_start:(i_end+1)]

# pipeline = import_file(files, multiple_frames=True)

# # 1) make RDF per frame
# pipeline.modifiers.append(CoordinationAnalysisModifier(cutoff=cutoff,number_of_bins=n_bins,))
# pipeline.modifiers.append(TimeAveragingModifier(operate_on = "table:coordination-rdf",))
# data = pipeline.compute()
# rdf_table = data.tables['coordination-rdf[average]'].xy()
# arr = np.asarray(rdf_table)   # shape (n_bins, 2)
# r2  = arr[:, 0]
# gr2 = arr[:, 1]


from ovito_tools import gr_timeaveraged
import matplotlib.pyplot as plt
from pathlib import Path
name = "20260205_122512"
liquid=gr_timeaveraged(name,1,201,n_bins=200)
solid=gr_timeaveraged(name,500,600,n_bins=200)
PROJECT_ROOT = Path(__file__).resolve().parents[0]
out_prefix = PROJECT_ROOT / f"{name}_rdf_timeavg1"
out_prefix=f"{out_prefix}_both"



fig, ax = plt.subplots(figsize=(6,4))
ax.plot(liquid[0], liquid[1], label="simulated liquid",color='red')
ax.plot(solid[0], solid[1], label="simulated amorphous",color="blue",alpha=0.5)
ax.set_xlim(2,10)
ax.set_ylim(0,3.5)
ax.set_xlabel("r (\u212B)")
ax.set_ylabel("g(r)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{out_prefix}_avg.png", dpi=300, bbox_inches="tight")
plt.show()
