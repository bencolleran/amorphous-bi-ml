from ovito_tools import gr_timeaveraged,gr_from_csv
import matplotlib.pyplot as plt
from pathlib import Path
name = "20260205_122512"
liquid=gr_timeaveraged(name,1,201,n_bins=1000)
exp=gr_from_csv("liquid_rdf")
PROJECT_ROOT = Path(__file__).resolve().parents[0]
out_prefix = PROJECT_ROOT / f"{name}_rdf_timeavg1"
out_prefix=f"{out_prefix}_both"


plt.figure(figsize=(6, 4), dpi=150)
ax = plt.gca()
ax.set_xlim([2, 10])
ax.set_ylim([0, 5])
ax.plot(liquid[0], liquid[1], label="simulated liquid",color='red')
ax.plot(exp[0], exp[1], label="experimental liquid",color="blue",alpha=0.5)
plt.legend()
plt.xlabel('r / \u212B')
plt.ylabel('g(r)')
plt.show()
plt.savefig(f'{PROJECT_ROOT}/dual_plot.png')
