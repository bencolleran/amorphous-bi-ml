from ovito_tools import gr_timeaveraged,gr_single_frame,gr_from_csv
import matplotlib.pyplot as plt
from pathlib import Path
name = "20260210_102745"
name="20260210_130003"
experimental=gr_from_csv("rdf_data_from_paper")
#solid=gr_timeaveraged(name,i_start=700,i_end=-5,n_bins=100)
solid=gr_timeaveraged(name,i_start=790,i_end=792,n_bins=100)
PROJECT_ROOT = Path(__file__).resolve().parents[0]
out_prefix = PROJECT_ROOT / f"{name}_rdf_timeavg4"
out_prefix=f"{out_prefix}_and_single"




fig, ax = plt.subplots(figsize=(6,4))
# ax.plot(r1, gr1, label="simulated liquid",color='red')
# ax.plot(r2, gr2, label="simulated amorphous",color="blue",alpha=0.5)
ax.plot(experimental[0], experimental[1], label="experimental")
ax.plot(solid[0], solid[1], label="simulated amorphous",alpha=0.5)
ax.set_xlim(2,10)
ax.set_ylim(0,5)
ax.set_xlabel("r (\u212B)")
ax.set_ylabel("g(r)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{PROJECT_ROOT}/images/{out_prefix}_avg.png", dpi=300, bbox_inches="tight")
plt.show()
