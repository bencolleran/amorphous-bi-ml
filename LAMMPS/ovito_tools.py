
def gr_timeaveraged(name,i_start,i_end,n_bins=200,cutoff=10):
    import numpy as np
    from pathlib import Path
    import glob
    import re
    from ovito.io import import_file
    from ovito.modifiers import CoordinationAnalysisModifier, TimeAveragingModifier
    
    PROJECT_ROOT = Path(__file__).resolve().parents[0]

    files = glob.glob(f"{PROJECT_ROOT}/{name}/NVT/dump_custom.Bi.*.dat")
    files = sorted(files,key=lambda f: int(re.search(r'\.(\d+)\.dat$', f).group(1)))

    # i_start = 1 #inclusive
    # i_end   = 201 #inclusive 201
    files = files[i_start:(i_end+1)]

    pipeline = import_file(files, multiple_frames=True)

    # 1) make RDF per frame
    pipeline.modifiers.append(CoordinationAnalysisModifier(cutoff=cutoff,number_of_bins=n_bins,))
    pipeline.modifiers.append(TimeAveragingModifier(operate_on = "table:coordination-rdf",))
    data = pipeline.compute()
    rdf_table = data.tables['coordination-rdf[average]'].xy()
    arr = np.asarray(rdf_table)   # shape (n_bins, 2)
    r  = arr[:, 0]
    gr = arr[:, 1]
    return [r,gr]

def gr_single_frame(name,frame,num_points=100,n_bins=200,cutoff=10):
    import numpy as np
    from pathlib import Path
    import glob
    import re
    from ovito.io import import_file
    from scipy.interpolate import PchipInterpolator
    from ovito.modifiers import CoordinationAnalysisModifier, TimeAveragingModifier
    
    PROJECT_ROOT = Path(__file__).resolve().parents[0]

    files = glob.glob(f"{PROJECT_ROOT}/{name}/NVT/dump_custom.Bi.*.dat")
    files = sorted(files,key=lambda f: int(re.search(r'\.(\d+)\.dat$', f).group(1)))

    files = files[frame]

    pipeline = import_file(files, multiple_frames=False)

    # 1) make RDF per frame
    pipeline.modifiers.append(CoordinationAnalysisModifier(cutoff=cutoff,number_of_bins=n_bins,))
    data = pipeline.compute()
    rdf_table = data.tables['coordination-rdf'].xy()
    arr = np.asarray(rdf_table)   # shape (n_bins, 2)
    r  = arr[:, 0]
    gr = arr[:, 1]
    first=r[0]
    r1=np.linspace(first,cutoff,num_points)
    pchip=PchipInterpolator(r,gr)
    gr1=pchip(r1)
    return [r1,gr1]

def gr_from_csv(file,num_points=100):
    from scipy.interpolate import PchipInterpolator
    from pathlib import Path
    import pandas as pd
    import numpy as np
    PROJECT_ROOT = Path(__file__).resolve().parents[0]
    filename=f"{PROJECT_ROOT}/{file}.csv"
    data=pd.read_csv(filename,names=['r','g(r)'])
    df=pd.DataFrame(data)
    r=np.array(df['r'])
    gr=np.array(df['g(r)'])

    first=r[0]

    idx = np.argsort(r)
    r =r[idx]
    gr= gr[idx]

    r1=np.linspace(first,10,num_points)
    pchip=PchipInterpolator(r,gr)
    gr1=pchip(r1)
    return [r1,gr1]

def gr_kde_from_ovito_new(name,frame=None,i_start=None,i_end=None,cutoff=11.0,n_bins=200,sigma=0.18):
    import numpy as np
    from pathlib import Path
    import glob, re
    from ovito.io import import_file
    from ovito.modifiers import CoordinationAnalysisModifier, TimeAveragingModifier
    from scipy.signal import fftconvolve
    PROJECT_ROOT = Path(__file__).resolve().parents[0]
    files = glob.glob(f"{PROJECT_ROOT}/{name}/NVT/dump_custom.Bi.*.dat")
    files = sorted(files, key=lambda f: int(re.search(r'\.(\d+)\.dat$', f).group(1)))

    if (i_start is not None) and (i_end is not None):
        sel_files = files[i_start:(i_end + 1)]
        pipeline = import_file(sel_files, multiple_frames=True)
        use_avg_table = True
    else:
        if frame is None:
            raise ValueError("Either provide `frame` (single-frame) or `i_start` and `i_end` (time-averaged).")
        sel_files = files[frame]
        pipeline = import_file(sel_files, multiple_frames=False)
        use_avg_table = False

    pipeline.modifiers.append(CoordinationAnalysisModifier(cutoff=cutoff, number_of_bins=n_bins))
    if use_avg_table:
        pipeline.modifiers.append(TimeAveragingModifier(operate_on="table:coordination-rdf"))

    data = pipeline.compute()
    table_name = 'coordination-rdf[average]' if use_avg_table else 'coordination-rdf'
    tbl = data.tables[table_name].xy()
    tbl = np.asarray(tbl)
    r = tbl[:, 0]               # bin centers
    g = tbl[:, 1]               # OVITO g(r) or averaged g(r)
    dr = r[1] - r[0]

    # invert g(r) -> pair counts C(r)
    N = int(data.particles.count)
    try:
        volume = float(data.cell.volume)
    except Exception:
        Lx, Ly, Lz = data.cell.lengths
        volume = float(Lx * Ly * Lz)
    rho = N / volume
    # C(r) = g(r) * (N * 4π r^2 dr * rho) / 2
    C = g * (N * 4.0 * np.pi * (r**2) * dr * rho) / 2.0

    # gaussian kernel (sampled at dr)
    kernel_r = np.arange(-int(np.ceil(4.0 * sigma / dr)) * dr,
                         int(np.ceil(4.0 * sigma / dr)) * dr + dr,
                         dr)
    kernel = np.exp(-0.5 * (kernel_r / sigma)**2)
    kernel /= (kernel.sum() * 1.0)   # normalized in discrete sum (dr factor cancels later in inversion)
    C_smooth = fftconvolve(C, kernel, mode='same')
    denom = (N * 4.0 * np.pi * (r**2) * dr * rho) / 2.0
    # avoid division by zero at r=0
    denom[0] = np.inf
    g_kde = C_smooth / denom
    g_kde[0] = 0.0

    return r, g_kde


if __name__=="__main__":
    from pathlib import Path
    import matplotlib.pyplot as plt
    PROJECT_ROOT = Path(__file__).resolve().parents[0]

    pot1 = "20260212_152203"
    pot2 = "20260223_154848"
    paper_sol=gr_from_csv("correct_literature_rdf")
    paper_liq=gr_from_csv("liquid_rdf")
    sigma=0.15
    new_sol=gr_kde_from_ovito_new(pot2,frame=-1,cutoff=11,n_bins=200,sigma=sigma)
    #old=gr_kde_from_ovito_new(pot1,i_start=790,i_end=800,cutoff=11,n_bins=200,sigma=0.18)
    new_liq=gr_kde_from_ovito_new(pot2,i_start=10,i_end=300,cutoff=11,n_bins=200,sigma=sigma)


    fig, ax = plt.subplots(figsize=(6,4))
    # ax.plot(r1, gr1, label="simulated liquid",color='red')
    # ax.plot(r2, gr2, label="simulated amorphous",color="blue",alpha=0.5)

    #ax.plot(old[0], old[1], label="old potential",)
    # ax.plot(new_liq[0], new_liq[1], label="MD liquid",)
    # ax.plot(paper_liq[0],paper_liq[1],label="literature liquid",linestyle=":")
    # out_prefix = f"liquid"
    ax.plot(new_sol[0], new_sol[1], label="MD solid",)
    ax.plot(paper_sol[0],paper_sol[1],label="literature solid",linestyle=":")
    out_prefix = f"solid"
    ax.set_xlim(2,10)
    ax.set_ylim(0,3)
    ax.set_xlabel("r (\u212B)")
    ax.set_ylabel("g(r)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{PROJECT_ROOT}/images/KDE_{sigma}{out_prefix}.png", dpi=300, bbox_inches="tight")
    plt.show()