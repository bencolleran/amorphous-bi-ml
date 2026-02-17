#specify the name of the LAMMPS run
def load_dataset_to_ovito(name,start=0,end=None,frame=None):
    from ovito.io import import_file
    from pathlib import Path
    import glob,re
    PROJECT_ROOT = Path(__file__).resolve().parents[0]
    files = glob.glob(f"{PROJECT_ROOT}/{name}/NVT/dump_custom.Bi.*.dat")
    if len(files)==0:
        raise ValueError("directory does not exist")
    files = sorted(files,key=lambda f: int(re.search(r'\.(\d+)\.dat$', f).group(1)))
    if frame:
        files = files[frame]
        multiple=False
    elif end:
        files=files[start:(end+1)]
        multiple=True
    else:
        files=files
        multiple=True
    return import_file(files, multiple_frames=multiple)
    

# Finds mean bond lengths in a dataset using ovito (in Å)
def get_mean_bond_lengths(path_to_dataset, bond_cutoff=1.85):
    import numpy as np
    from pathlib import Path
    from ovito.modifiers import BondAnalysisModifier, CreateBondsModifier

    # Set up ovito pipeline
    pipeline = load_dataset_to_ovito(path_to_dataset,frame=0)
    pipeline.modifiers.append(CreateBondsModifier(cutoff=bond_cutoff))
    pipeline.modifiers.append(BondAnalysisModifier(bins = 100000, length_cutoff=bond_cutoff))

    mean_bond_lengths = []
    for i, frame in enumerate(pipeline.frames):

        # Compute bond lengths
        data = pipeline.compute(i)
        bond_lengths_xy = data.tables["bond-length-distr"].xy()
        l_values, l_counts = bond_lengths_xy[:,0], bond_lengths_xy[:,1]
        
        # Calculate mean bond length
        if l_counts.sum() == 0:
            print(f"WARNING: No bonds found in {Path(path_to_dataset).name} (Frame: {i}) with cutoff {bond_cutoff} Å. Returning 0")
            return 0
        
        else:
            mean_bond_length = np.average(l_values, weights=l_counts)
            mean_bond_lengths.append(mean_bond_length)
    
    return np.array(mean_bond_lengths)

# Finds bond angles in a structure using ovito (in degrees)
def get_mean_bond_angles(path_to_dataset, bond_cutoff=1.85):
    import numpy as np
    from pathlib import Path
    from ovito.modifiers import BondAnalysisModifier, CreateBondsModifier

    # Set up ovito pipeline
    pipeline = load_dataset_to_ovito(path_to_dataset)
    pipeline.modifiers.append(CreateBondsModifier(cutoff=bond_cutoff))
    pipeline.modifiers.append(BondAnalysisModifier(bins = 100000, length_cutoff=bond_cutoff))
    
    mean_bond_angles = []
    for i, frame in enumerate(pipeline.frames):

        # Compute bond lengths
        data = pipeline.compute(i)
        bond_angles_xy = data.tables["bond-angle-distr"].xy()
        a_values, a_counts = bond_angles_xy[:,0], bond_angles_xy[:,1]
        
        # Calculate mean bond length
        if a_counts.sum() == 0:
            print(f"WARNING: No bonds found in {Path(path_to_dataset).name} (Frame: {i}) with cutoff {bond_cutoff} Å. Returning 0")
            return 0
        
        else:
            mean_bond_angle = np.average(a_values, weights=a_counts)
            mean_bond_angles.append(mean_bond_angle)
    
    return np.array(mean_bond_angles)

# 1D array of coordination number for each particle in the structure, given a bond length cutoff
def coordination_analysis(path_to_dataset, bond_cutoff=1.85):
    from pathlib import Path
    import numpy as np
    from ovito.modifiers import CoordinationAnalysisModifier

    # Set up ovito pipeline
    pipeline = load_dataset_to_ovito(path_to_dataset)
    pipeline.modifiers.append(CoordinationAnalysisModifier(cutoff=bond_cutoff))

    coord_arrs = []
    for i, frame in enumerate(pipeline.frames):

        # Compute bond lengths
        data = pipeline.compute(i)
        coord_arr = data.particles["Coordination"].array # type: ignore

        # Guard against zero atoms
        num_atoms = len(coord_arr) 
        if num_atoms == 0:
            print(f"WARNING: No atoms found in {Path(path_to_dataset).name} (Frame: {i}) with cutoff {bond_cutoff} Å")
            return None

        # Guard against all zero coordination
        if coord_arr.max() == 0:
            print(f"WARNING: All particles have zero coordination in {Path(path_to_dataset).name} (Frame: {i}) with cutoff {bond_cutoff} Å")
        
        coord_arrs.append(coord_arr)
    
    return np.array(coord_arrs)

# List of 2D array of (r, g(r)) given a cutoff and number of bins for each structure in a dataset
def calculate_rdfs(path_to_dataset, rdf_cutoff=6, rdf_bins=200):
    from ovito.modifiers import CoordinationAnalysisModifier

    # Set up ovito pipeline
    pipeline = load_dataset_to_ovito(path_to_dataset)
    pipeline.modifiers.append(
        CoordinationAnalysisModifier(
            cutoff=rdf_cutoff,
            number_of_bins=int(rdf_bins)
        )
    )

    rdf_arrs = []
    for i, frame in enumerate(pipeline.frames):
        data = pipeline.compute(i)
        rdf_arrs.append(data.tables["coordination-rdf"].xy())
    
    return rdf_arrs

# Returns r, mean(g(r)), std(g(r)) for analysing the average rdf of a dataset
def calculate_rdf_mean_std(path_to_dataset, rdf_cutoff=6, rdf_bins=200):
    import numpy as np

    rdfs = calculate_rdfs(path_to_dataset, rdf_cutoff, rdf_bins)

    r = np.asarray(rdfs[0], float)[:, 0]
    g = np.stack([np.asarray(rdf, float)[:, 1] for rdf in rdfs])

    return r, g.mean(0), g.std(0)


# Histogram of (r, count) given a bond length cutoff and number of bins
def bond_length_analysis(path_to_dataset, bond_cutoff=1.85, bond_bins=200):
    from pathlib import Path
    from ovito.modifiers import BondAnalysisModifier, CreateBondsModifier

    # Set up ovito pipeline
    pipeline = load_dataset_to_ovito(path_to_dataset)
    pipeline.modifiers.append(CreateBondsModifier(cutoff=bond_cutoff))
    pipeline.modifiers.append(
        BondAnalysisModifier(
            bins=int(bond_bins),
            length_cutoff=bond_cutoff
        )
    )

    bond_length_arrs = []
    for i, frame in enumerate(pipeline.frames):
        data = pipeline.compute(i)
        bond_lengths = data.tables["bond-length-distr"].xy()

        if bond_lengths[:, 1].sum() == 0:
            print(f"WARNING: No bonds found in {Path(path_to_dataset).name} (Frame: {i})")

        bond_length_arrs.append(bond_lengths)
    
    return bond_length_arrs

# Histogram of (angle, count) given a bond length cutoff and number of bins
def bond_angle_analysis(path_to_dataset, bond_cutoff=1.85, bond_bins=200):
    from pathlib import Path
    from ovito.modifiers import BondAnalysisModifier, CreateBondsModifier

    # Set up ovito pipeline
    pipeline = load_dataset_to_ovito(path_to_dataset)
    pipeline.modifiers.append(CreateBondsModifier(cutoff=bond_cutoff))
    pipeline.modifiers.append(
        BondAnalysisModifier(
            bins=int(bond_bins),
            length_cutoff=bond_cutoff
        )
    )

    bond_angle_arrs = []
    for i, frame in enumerate(pipeline.frames):
        data = pipeline.compute(i)
        bond_angles = data.tables["bond-angle-distr"].xy()

        if bond_angles[:, 1].sum() == 0:
            print(f"WARNING: No bond angles found in {Path(path_to_dataset).name} (Frame: {i})")

        bond_angle_arrs.append(bond_angles)
    
    return bond_angle_arrs

#My new funciton
def get_ring_statistics(path_to_dataset, bond_cutoff=1.85, min_ring=3,max_ring=10):
    from ovito.modifiers import FindRingsModifier,CreateBondsModifier
    import numpy as np
    pipeline = load_dataset_to_ovito(path_to_dataset)
    pipeline.modifiers.append(CreateBondsModifier(cutoff=bond_cutoff))
    pipeline.modifiers.append(FindRingsModifier(
        minimum_ring_size=min_ring,
        maximum_ring_size=max_ring,))
    
    ring_arrs = []
    n_frames = pipeline.source.num_frames

    for i in range(n_frames):
        data = pipeline.compute(i)
        table = data.tables['ring-size-histogram']
        arr = np.zeros(max_ring - min_ring + 1)

        sizes = table['Ring Size']
        counts = table['Count']

        for size, count in zip(sizes, counts):
            if min_ring <= size <= max_ring:
                arr[size - min_ring] = count

        ring_arrs.append(arr)

    return np.array(ring_arrs)

# Histogram of (angle, probability density) given a bond length cutoff, number of bins, minimum_angle truncation
def calculate_adfs(path_to_dataset, bond_cutoff=1.85, adf_bins=40, angle_min=0):
    from pathlib import Path
    from ovito.modifiers import BondAnalysisModifier, CreateBondsModifier

    # Set up ovito pipeline
    pipeline = load_dataset_to_ovito(path_to_dataset)
    pipeline.modifiers.append(CreateBondsModifier(cutoff=bond_cutoff))
    pipeline.modifiers.append(
        BondAnalysisModifier(
            bins=int(adf_bins),
            length_cutoff=bond_cutoff
        )
    )

    bin_width = 180 / adf_bins

    adf_arrs = []
    for i, frame in enumerate(pipeline.frames):
        data = pipeline.compute(i)
        adf = data.tables["bond-angle-distr"].xy()
        adf = adf[adf[:, 0] >= angle_min]

        total = adf[:, 1].sum()
        if total > 0:
            adf[:, 1] /= (total * bin_width)
        else:
            print(f"WARNING: No bond angles for ADF in {Path(path_to_dataset).name} (Frame: {i})")

        adf_arrs.append(adf)
    
    return adf_arrs


# Returns angle, mean(probability), std(probability) for analysing the average adf of a dataset
def calculate_adf_mean_std(path_to_dataset, bond_cutoff=1.85, adf_bins=40, angle_min=0):
    import numpy as np

    rdfs = calculate_adfs(path_to_dataset, bond_cutoff, adf_bins, angle_min)

    angles = np.asarray(rdfs[0], float)[:, 0]
    probabilities = np.stack([np.asarray(rdf, float)[:, 1] for rdf in rdfs])
    
    return angles, probabilities.mean(0), probabilities.std(0)

# Converts file format using ase read and write
def ase_convert(path_to_file, suffix, out_dir=None):
    from pathlib import Path
    from ase.io import read, write

    path = Path(path_to_file)

    out_path = (Path(out_dir) / path.name if out_dir else path).with_suffix(suffix)

    write(out_path, read(path, ":"))

# View structure using ase
def view_structure(path_to_file):
    from pathlib import Path
    from ase.io import read
    from ase.visualize import view
    from IPython.display import display

    if not Path(path_to_file).exists():
        raise ValueError(f"Specified filepath does not exist")
    
    atoms = read(str(path_to_file))
    print(Path(path_to_file).name)
    
    display(view(atoms, viewer='x3d'))


# Plot and Save function:
#   if marker = "": uses shaded regions for error bars (continuous data)
#   if peak_pick = int(n): labels the top n peaks
#   style_file: allows specification of a mplstyle file
#   tight: enables plt.tightlayout()
#   pdf: creates a pdf file of the graph
#   std_limit: prevents std error bars or shading going above or below a certain limit
#   data_label: adds a label to the data for showing on a legend
#   multiplot_datasets: enables plotting of multiple datasets on one graph 
#        datasets = [{
#            'x': x, 
#            'y': y, 
#            'std': std (optional), 
#            'label': plot_name (optional),  
#            'marker': marker (optional), 
#            'linestyle': linestyle (optional)
#        }]
def scatter_plot(x=None, y=None, std=None, data_label=None, xlabel=None, ylabel=None, title=None, linestyle="-", 
                 marker="o", plot_name="graph", out_dir=None, style_file=None, pdf=False, 
                 peak_pick=None, tight=False, lower_std_limit=None, upper_std_limit=None,
                 legend=False, multiplot_datasets=None, color=None, shading_alpha=0.3, **kwargs):
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    # If datasets provided, use multi-plot mode; otherwise single-plot mode
    if multiplot_datasets is None:
        multiplot_datasets = [{
            'x': x, 'y': y, 'std': std, 
            'label': plot_name, 
            'marker': marker, 
            'linestyle': linestyle,
            'color': color
        }]
    
    # Load style file if specified
    if style_file:
        try:
            plt.style.use(str(style_file))
        except Exception as e:
            raise ValueError(f"Failed to load style file '{style_file}': {e}") from e
    
    # Create the plot
    fig, ax = plt.subplots()
    
    # Plot each dataset
    for data in multiplot_datasets:
        x = np.asarray(data['x'])
        y = np.asarray(data['y'])
        std = data.get('std')
        label = data.get('label', 'Data')
        marker = data.get('marker', 'o')
        linestyle = data.get('linestyle', '-')
        color = data.get('color', None)
        
        if x is None or y is None:
            print(f"WARNING: No data in x or y for {label}")
            continue
        
        # Plot data with or without error bars/shading
        if std is not None:
            std = np.asarray(std)
            if upper_std_limit is not None:
                y_high = np.minimum(y + std, upper_std_limit)
            else:
                y_high = y + std
            if lower_std_limit is not None:
                y_low = np.maximum(y - std, lower_std_limit)
            else:
                y_low = y - std
            
            if marker != "":
                yerr = np.vstack((y - y_low, y_high - y))
                ax.errorbar(x, y, yerr=yerr, fmt=linestyle, capsize=5, capthick=2, label=label, color=color)
            else:
                ax.plot(x, y, linestyle, label=label, color=color)
                ax.fill_between(x, y_low, y_high, alpha=shading_alpha, color=color)
        else:
            ax.plot(x, y, linestyle, marker=marker, label=label, color=color)
        
        # Peak picking
        if peak_pick: 
            from scipy.signal import find_peaks
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
    
    # Set labels and title
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    
    # Add legend
    if legend:
        ax.legend()
    
    # Tight layout
    if tight:
        plt.tight_layout()
    
    # Save paths
    if not out_dir:
        out_dir = Path.cwd()
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{plot_name}.png"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    
    if pdf:
        pdf_path = out_dir / f"{plot_name}.pdf"
        plt.savefig(pdf_path, bbox_inches='tight')
    
    plt.close('all')
    print(f"Created {plot_name}")


if __name__=="__main__":
    # print(get_mean_bond_lengths("20260205_122512",bond_cutoff=3.3))
    # print(len(get_mean_bond_lengths("20260205_122512",bond_cutoff=3.3)))
    # load_dataset_to_ovito("20260205_122512")
    # scatter_plot(x=[i for i in range(604)],y=get_mean_bond_lengths("20260205_122512",bond_cutoff=3.3),out_dir="fig",plot_name="bonds")
    # print(get_mean_bond_angles("20260205_122512",bond_cutoff=3.3)[0])
    # print(get_mean_bond_lengths("20260205_122512",bond_cutoff=3.3)[0])
    # print(coordination_analysis("20260205_122512",bond_cutoff=3.3)[0])
    # print(get_ring_statistics("20260205_122512",bond_cutoff=3.3,min_ring=3,max_ring=7)[0:4])
    print(coordination_analysis("20260205_122512",bond_cutoff=3.3).shape)

    