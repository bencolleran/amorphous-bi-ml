# Wrap model in a graph-pes calculator, allowing for loading of both .pt and .xml files
def create_calculator(path_to_model, device="cpu", skin=0, eval=False):
    import torch
    from pathlib import Path
    from graph_pes.models import load_model
    from graph_pes.utils.calculator import GraphPESCalculator

    if device not in {"cpu", "cuda"}:
        raise ValueError(f"Unknown device: {device}")

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    path_to_model = str(path_to_model)

    # Load MACE models using graph_pes
    if Path(path_to_model).suffix == ".pt":
        if eval:
            model = load_model(path_to_model).to(device).eval()
        else:
            model = load_model(path_to_model)

        calc = GraphPESCalculator(model, skin=skin, device=device)

    # Load GAP models using quippy
    elif Path(path_to_model).suffix == ".xml":
        from quippy.potential import Potential
        
        class GAPWrapper:
            def __init__(self, potential):
                self.potential = potential
                self.name = "GAP"

            def get_potential_energy(self, atoms, **kwargs):
                return self.potential.get_potential_energy(atoms)

            def get_forces(self, atoms, **kwargs):
                return self.potential.get_forces(atoms)

            def get_stress(self, atoms=None):
                return self.potential.get_stress(atoms=atoms)

            def __getattr__(self, attr):
                return getattr(self.potential, attr)

        raw_gap = Potential(param_filename=path_to_model)
        calc = GAPWrapper(raw_gap)
    
    else:
        from graph_pes.interfaces import mace_mp
        model = mace_mp(path_to_model)
        calc = GraphPESCalculator(model, skin=skin, device=device)
    
    return calc


# Helper function for loading a dataset into an ase list of atoms
def load_dataset(path_to_dataset):
    from ase.io import read

    # Load test set
    dataset = read(str(path_to_dataset), index=":")

    # Ensure testset is a list
    if not isinstance(dataset, list):
        dataset = [dataset]
    
    return dataset


# Helper function for loading a dataset into an ovito pipeline
def load_dataset_to_ovito(path_to_dataset):
    import warnings
    import ovito
    from ovito.io import import_file
    
    warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')

    # Set up ovito pipeline
    for p in list(ovito.scene.pipelines):
        p.remove_from_scene()

    pipeline = import_file(path_to_dataset)

    return pipeline


# Helper function to convert a list to a binned histogram: returning (bin_centres, probability)
def get_histogram(values, bins=200):
    import numpy as np
    
    counts, edges = np.histogram(values, bins=bins)
    prob = counts / counts.sum()
    centers = 0.5 * (edges[:-1] + edges[1:])

    return centers, prob


# Use an ase calculator to predict forces (in eV/Å)
def predict_forces(calculator, path_to_dataset, histogram=False, bins=200):
    import numpy as np

    dataset = load_dataset(path_to_dataset)

    # predict forces with the model
    predicted_forces=[]
    for structure in dataset:
        structure.calc=calculator
        forces=structure.get_forces()
        predicted_forces.append(forces)
    predicted_forces=np.concatenate(predicted_forces).flatten()

    if histogram:
        return get_histogram(predicted_forces, bins)
    
    return predicted_forces


# Use an ase calculator to predict energies (in eV/atom)
def predict_energies(calculator, path_to_dataset, histogram=False, bins=200):
    import numpy as np

    dataset = load_dataset(path_to_dataset)

    # Predict energies with the model
    predicted_energies_per_atom=[]
    for structure in dataset:
        structure.calc=calculator
        energy=structure.get_potential_energy()
        energy_per_atom=energy / len(structure)
        predicted_energies_per_atom.append(energy_per_atom)
    predicted_energies_per_atom = np.array(predicted_energies_per_atom)

    if histogram:
        return get_histogram(predicted_energies_per_atom, bins)
    
    return predicted_energies_per_atom


# Extract true energies from a labelled dataset (in eV/atom)
def parse_energies(path_to_dataset, histogram=False, bins=200):
    import numpy as np

    dataset = load_dataset(path_to_dataset)

    # extract true DFT energies
    true_energies_per_atom = []
    for structure in dataset:
        total_energy = structure.get_potential_energy()
        energy_per_atom = total_energy / len(structure)  
        true_energies_per_atom.append(energy_per_atom)
    true_energies_per_atom = np.array(true_energies_per_atom)

    if histogram:
        return get_histogram(true_energies_per_atom, bins)
    
    return true_energies_per_atom


# Extract true forces from a labelled dataset
def parse_forces(path_to_dataset, histogram=False, bins=200):
    import numpy as np

    dataset = load_dataset(path_to_dataset)

    # extract true DFT forces
    true_forces = []
    for structure in dataset:  
        true_forces.append(structure.get_forces())
    true_forces = np.concatenate(true_forces).flatten()

    if histogram:
        return get_histogram(true_forces, bins)
    
    return true_forces


# Finds mean bond lengths in a dataset using ovito (in Å)
def get_mean_bond_lengths(path_to_dataset, bond_cutoff=1.85):
    import numpy as np
    from pathlib import Path
    from ovito.modifiers import BondAnalysisModifier, CreateBondsModifier

    # Set up ovito pipeline
    pipeline = load_dataset_to_ovito(path_to_dataset)
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


# Find cell parameters using ASE: returns list of tuples (a, b, c, alpha, beta, gamma) for each structure in the test set
def get_cell_parameters(path_to_dataset):
    import numpy as np

    # Load test set
    dataset = load_dataset(path_to_dataset)

    cell_parameters = []
    for structure in dataset:
        cell = structure.get_cell()
        a, b, c = cell.lengths()
        alpha, beta, gamma = cell.angles()
        cell_parameters.append((a, b, c, alpha, beta, gamma))

    return np.array(cell_parameters)


# 1D array of coordination number for each particle in the structure, given a bond length cutoff
def coordination_analysis(path_to_dataset, bond_cutoff=1.85):
    from pathlib import Path
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
    
    return coord_arrs


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


# Ring statistics as array of counts for rings of size 3 to max_rings (inclusive)
def ring_analysis(path_to_dataset, bond_cutoff=1.85, min_ring=3, max_ring=24):
    import numpy as np
    from pathlib import Path
    from matscipy.rings import ring_statistics

    # Load dataset into ase atoms
    dataset = load_dataset(path_to_dataset)

    ring_arrs = []
    for i, structure in enumerate(dataset):

        ring_counts = np.asarray(
            ring_statistics(
                structure,
                cutoff=bond_cutoff,
                maxlength=max_ring
            )
        )

        ring_counts = np.pad(
            ring_counts,
            (0, max(0, max_ring + 1 - len(ring_counts)))
        )[min_ring:max_ring + 1]

        total = ring_counts.sum()
        if total == 0:
            print(f"WARNING: No rings detected in {Path(path_to_dataset).name} (Frame: {i})")
            ring_arrs.append(ring_counts.astype(float))
        else:
            ring_arrs.append(ring_counts)
    
    return ring_arrs


# Voronoi face distribution, edge distribution and volume distribution given a relative face threshold, face and edge limits, volume histogram range and number of atoms (for volume scaling)
#  Relative face threshold: faces with surface area below (threshold*total polyhedron surface area) are discounted when counting faces and edges
def voronoi_faces_distribution(path_to_dataset, rel_face_threshold=0.01, faces_min=0, faces_max=24):
    import numpy as np
    from pathlib import Path
    from ovito.modifiers import VoronoiAnalysisModifier

    # Set up ovito pipeline
    pipeline = load_dataset_to_ovito(path_to_dataset)
    pipeline.modifiers.append(
        VoronoiAnalysisModifier(
            compute_indices=True,
            relative_face_threshold=rel_face_threshold
        )
    )

    face_dist_arrs = []
    for i, frame in enumerate(pipeline.frames):
        data = pipeline.compute(i)
        indices = np.array(data.particles["Voronoi Index"])

        if len(indices) == 0:
            print(f"WARNING: No Voronoi polyhedra in {Path(path_to_dataset).name} (Frame: {i})")
            face_dist_arrs.append(None)
            continue

        faces_per_poly = indices.sum(axis=1)
        face_counts = np.bincount(faces_per_poly, minlength=faces_max + 1)

        face_dist_arrs.append(np.column_stack((
            np.arange(faces_min, faces_max + 1),
            face_counts[faces_min:faces_max + 1] / len(indices)
        )))
    
    return face_dist_arrs


# Voronoi edge distribution given a relative face threshold and edge limits
#  Relative face threshold: faces with surface area below (threshold*total polyhedron surface area) are discounted when counting faces and edges
def voronoi_edges_distribution(path_to_dataset, rel_face_threshold=0.01, edges_min=0, edges_max=24):
    import numpy as np
    from pathlib import Path
    from ovito.modifiers import VoronoiAnalysisModifier

    # Set up ovito pipeline
    pipeline = load_dataset_to_ovito(path_to_dataset)
    pipeline.modifiers.append(
        VoronoiAnalysisModifier(
            compute_indices=True,
            relative_face_threshold=rel_face_threshold
        )
    )

    edge_dist_arrs = []
    for i, frame in enumerate(pipeline.frames):
        data = pipeline.compute(i)
        indices = np.array(data.particles["Voronoi Index"])

        if len(indices) == 0:
            print(f"WARNING: No Voronoi polyhedra in {Path(path_to_dataset).name} (Frame: {i})")
            edge_dist_arrs.append(None)
            continue

        faces_with_k_edges = indices.sum(axis=0)
        faces_with_k_edges = np.pad(
            faces_with_k_edges,
            (0, max(0, edges_max + 1 - len(faces_with_k_edges)))
        )

        edge_dist_arrs.append(np.column_stack((
            np.arange(edges_min, edges_max + 1),
            faces_with_k_edges[edges_min:edges_max + 1] /
            faces_with_k_edges.sum()
        )))
    
    return edge_dist_arrs


# Voronoi volume histogram:
#   Returns:
#        x=scaled volume fraction such that x=1 corresponds to the mean cell voronoi volume
#        y = probability
#  vol_min, vol_max: range of x values to include
def voronoi_volume_distribution(path_to_dataset, bins = None, vol_min = 0, vol_max = 2):
    import numpy as np
    from pathlib import Path
    from ovito.modifiers import VoronoiAnalysisModifier

    # Set up ovito pipeline
    pipeline = load_dataset_to_ovito(path_to_dataset)
    pipeline.modifiers.append(VoronoiAnalysisModifier())

    vol_dist_arrs = []
    for i, frame in enumerate(pipeline.frames):
        data = pipeline.compute(i)
        num_atoms = data.particles.count
        volumes = data.particles["Atomic Volume"].array # type: ignore

        if len(volumes) == 0:
            print(f"WARNING: No Voronoi volumes in {Path(path_to_dataset).name} (Frame: {i})")
            vol_dist_arrs.append(None)
            continue

        # Set default value of bins
        if bins is None:
            bins = num_atoms // 5

        scaled_vol = volumes / volumes.sum() * num_atoms
        counts, edges = np.histogram(
            scaled_vol,
            bins=bins,
            range=(vol_min, vol_max)
        )

        vol_dist_arrs.append(np.column_stack((
            (edges[:-1] + edges[1:]) / 2,
            counts / counts.sum()
        )))
    
    return vol_dist_arrs


# Returns volume, mean(probability), std(probability) for analysing the average voronoi volume distribution of a dataset
def voronoi_volume_mean_std(path_to_dataset, bins=40, vol_min=0, vol_max=2):
    import numpy as np
    
    dists = [d for d in voronoi_volume_distribution(path_to_dataset, bins, vol_min, vol_max) if d is not None]
    
    v = np.asarray(dists[0], float)[:, 0]
    p = np.stack([np.asarray(d, float)[:, 1] for d in dists])
    
    return v, p.mean(0), p.std(0)


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