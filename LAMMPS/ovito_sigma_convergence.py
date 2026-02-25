from ovito_tools import gr_timeaveraged,gr_kde_from_ovito_new
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
import numpy as np

name = "20260223_154848"
PROJECT_ROOT = Path(__file__).resolve().parents[0]

import numpy as np
from scipy.integrate import cumulative_trapezoid

def integrate(var, rho=0.0273, r_max=None):
    r, g = var
    y = 4*np.pi*r**2*rho*g
    I = cumulative_trapezoid(y, r, initial=0)
    return np.interp(r_max, r, I) if r_max else I


def stationary_points(var, n=1, kind="max", y_min=None, y_max=None,x_min=None,x_max=None,depth_min=None):
    x, y = var
    x = np.asarray(x); y = np.asarray(y)

    # find candidate indices (maxima or minima)
    if kind == "max":
        idx, _ = find_peaks(y,prominence=depth_min)
    elif kind == "min":
        idx, _ = find_peaks(-y,prominence=-depth_min)
    else:
        raise ValueError("kind must be 'max' or 'min'")

    # apply threshold filters to candidates first
    if y_min is not None:
        idx = idx[y[idx] >= y_min]
    if y_max is not None:
        idx = idx[y[idx] <= y_max]
    if x_min is not None:
        idx = idx[x[idx] >= x_min]
    if x_max is not None:
        idx = idx[x[idx] <= x_max]

    if idx.size == 0:
        return []

    # sort by prominence in value (for minima use -y to get deepest)
    sort_vals = y[idx] if kind == "max" else -y[idx]
    top_idx = idx[np.argsort(sort_vals)[-n:]]  # pick n largest by sort_vals
    top_idx = np.sort(top_idx)                 # ensure increasing x order

    return [(float(x[i]), float(y[i])) for i in top_idx]

sigmas = [0.005,0.01,0.02,0.03,0.05,0.07,0.1,0.15,0.18,0.2,0.3,0.4,0.5]

for i,sigma in enumerate(sigmas):
    test=gr_kde_from_ovito_new(name,frame=-1,cutoff=11,n_bins=200,sigma=sigma)
    maxima=stationary_points(test,n=1,kind="max")
    minima=stationary_points(test,n=1,kind="min",y_max=1,x_min=3,x_max=4,depth_min=0.1)
    cn=integrate(test,r_max=minima[0][0])
    print(f"CN={cn}, maxima={maxima[0][0]}, minima={minima[0][0]},sigma={sigma}")
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(test[0], test[1],)
    ax.set_xlim(2,10)
    ax.set_ylim(0,5)
    ax.set_xlabel("r (\u212B)")
    ax.set_ylabel("g(r)")
    ax.set_title(f"sigma={sigma}")
    fig.tight_layout()
    fig.savefig(f"{PROJECT_ROOT}/sigma_convergence_images/{i}_{sigma}_plot.png", dpi=300, bbox_inches="tight")
