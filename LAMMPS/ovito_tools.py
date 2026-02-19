
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


if __name__=="__main__":
    from pathlib import Path
    import matplotlib.pyplot as plt

    name="20260205_122512"
    data=gr_single_frame(name,-1,n_bins=50)
    PROJECT_ROOT = Path(__file__).resolve().parents[0]

    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(data[0],data[1], label="simulated amorphous",color="blue",alpha=0.5)
    ax.set_xlim(2,10)
    ax.set_ylim(0,5)
    ax.set_xlabel("r (\u212B)")
    ax.set_ylabel("g(r)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{PROJECT_ROOT}/images/{name}_test.png", dpi=300, bbox_inches="tight")
    plt.show()