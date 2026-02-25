from ovito_tools import gr_timeaveraged,gr_single_frame,gr_from_csv
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
pot1 = "20260212_152203"
pot2 = "20260223_154848"
old=gr_timeaveraged(pot1,750,800,n_bins=20)
new=gr_timeaveraged(pot2,750,800,n_bins=20)
paper=gr_from_csv("correct_literature_rdf")
PROJECT_ROOT = Path(__file__).resolve().parents[0]
out_prefix = f"comparison_rdf_timeavg_new"




fig, ax = plt.subplots(figsize=(6,4))
# ax.plot(r1, gr1, label="simulated liquid",color='red')
# ax.plot(r2, gr2, label="simulated amorphous",color="blue",alpha=0.5)

def peak_picker(x,y):
    peak_pick=1
    peaks, _ = find_peaks(y)
    top_peaks = peaks[np.argsort(y[peaks])[-int(peak_pick):]]
    peak_pick_sign = -1
    for p in top_peaks:
        xp = float(x[p])
        yp = float(y[p])
        ax.annotate(
            f"{xp:.2f}", (xp, yp),
            xytext=(10 * peak_pick_sign, 10),
            textcoords="offset points",
            fontsize=5,
            ha='center', va='top',
            arrowprops=dict(arrowstyle="-", lw=0.3, shrinkA=0, shrinkB=1)
        )
        peak_pick_sign *= -1
    return

ax.plot(old[0], old[1], label="old potential",)
peak_picker(old[0],old[1])
ax.plot(new[0], new[1], label="new potential",)
ax.plot(paper[0],paper[1],label="literature")
ax.set_xlim(2,10)
ax.set_ylim(0,5)
ax.set_xlabel("r (\u212B)")
ax.set_ylabel("g(r)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{PROJECT_ROOT}/images/{out_prefix}_avg.png", dpi=300, bbox_inches="tight")
plt.show()

#I need to redigitize my RDF as it is incorrect