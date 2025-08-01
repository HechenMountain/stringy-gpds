###################################
## Generate interpolation tables ##
###################################
import numpy as np
import csv 

from joblib import Parallel, delayed
from tqdm import tqdm

from . import core
from . import helpers as hp
from . import config as cfg
from . import special as sp
from . import adim
# mpmath precision set in config
from .config import mp

def generate_moment_table(eta,t,mu,solution,particle,moment_type,moment_label, evolution_order, error_type,
                          im_j_max=100,step=.1):
    """
    Generate tables for interpolation of input and evolved moments. The real part of j is defined by the value in get_j_base.

    Parameters
    ----------
    eta : float
        Skewness parameter.
    t : float
        Mandelstam t.
    mu : float
        Resolution scale.
    solution : str
        "+" or "-" solution for input singlet moment.
    particle : str
        "quark" or "gluon".
    moment_type : str
        "non_singlet_isovector", "non_singlet_isoscalar",  "singlet".
    moment_label : str
        A(tilde), B(tilde) depending on H(tilde) or E(tilde) GPD etc.
    evolution_order : str
        "lo", "nlo",... .
    error_type : str
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"
    im_j_max : float, optional
        Maximum value of the imaginary part of j. Default is 100.
    step : float, optional
        Step size for imaginary j-grid. Default is 0.1.

    Returns
    -------
    None

    Notes
    -----
    Writes the generated table to a CSV file specified by INTERPOLATION_TABLES_PATH
    """

    def compute_moment(j):
        # For mu != 1 we interpolate the evolved moments
        if mu != 1 or (moment_type == "singlet" and solution not in ["+","-"]):
            return core.evolve_conformal_moment(j, eta, t,mu,A0=1,particle=particle,moment_type=moment_type,
                                           moment_label=moment_label,evolution_order=evolution_order,
                                           error_type=error_type)
        # For mu = 1 we use the corresponding input moments
        elif moment_type == "non_singlet_isovector":
            return core.non_singlet_isovector_moment(j,eta,t,
                                                moment_label=moment_label,
                                                evolution_order=evolution_order, error_type=error_type)
        elif moment_type == "non_singlet_isoscalar":
            return core.non_singlet_isoscalar_moment(j,eta,t,
                                                moment_label=moment_label,
                                                evolution_order=evolution_order, error_type=error_type)
        elif moment_type == "singlet":
            val = core.singlet_moment(j, eta, t, 
                                  moment_label=moment_label,
                                  solution=solution, evolution_order=evolution_order, error_type=error_type)
            return val[0] if error_type == "central" else val[1]
        else:
            raise ValueError(f"Unknown moment_type {moment_type}")

    # Define output filename
    if mu != 1 or (moment_type == "singlet" and solution not in ["+","-"]) or moment_type != "singlet":
        prefix = cfg.INTERPOLATION_TABLE_PATH / f"{moment_type}_{particle}_moments_{moment_label}_{evolution_order}"
    elif moment_type == "singlet":
        prefix = cfg.INTERPOLATION_TABLE_PATH/ f"{moment_type}_{solution}_moments_{moment_label}_{evolution_order}"
    filename = hp.generate_filename(eta, t, mu, prefix, error_type)

    # Grids
    rj, _ = core.get_j_base(particle=particle,moment_type=moment_type,moment_label=moment_label)
    im_j = np.arange(0.0, im_j_max + step, step)  # Im ≥ 0 only
    # Generate grid points
    grid_points = im_j
    def compute_wrapper(ij):
        return (ij, compute_moment(complex(rj,ij)))

    with hp.tqdm_joblib(tqdm(total=len(grid_points))) as progress_bar:
        results = Parallel(n_jobs=-1)(
            delayed(compute_wrapper)(ij) for ij in grid_points
        )

    # Add conjugated points: f(j*) = f(j)* for Im(j) < 0
    mirrored = []
    for ij, val in results[1:]: 
        if ij > 0:
            conjugate = tuple(np.conj(x) for x in val)
            mirrored.append((-ij, conjugate))

    all_data = results + mirrored
    all_data.sort(key=lambda x: x[0])  # sort by Im(j)

    # Write CSV
    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Im(j)", "value", "nd_value"])
        for ij, val in all_data:
            writer.writerow([ij, val[0], val[1]])

    print(f"Successfully wrote table to {filename}")

def generate_harmonic_table(indices,j_re_min=0, j_re_max=10, j_im_min=0, j_im_max=110,step=0.25, n_jobs=-1):
    """
    Generate tables for interpolation of (nested) harmonic numbers.

    Parameters
    ----------
    indices : list or tuple
        A list of integers specifying the nested harmonic sum indices (e.g., 1, [2,1], etc.).
    j_re_min : float, optional
        Minimum real part of conformal spin j. Default is 0.
    j_re_max : float, optional
        Maximum real part of conformal spin j. Default is 10.
    j_im_min : float, optional
        Minimum imaginary part of conformal spin j. Default is 0.
    j_im_max : float, optional
        Maximum imaginary part of conformal spin j. Default is 110.
    step : float, optional
        Step size in the j-plane for generating the grid. Default is 0.25.
    n_jobs : int, optional
        Number of parallel jobs to run. Default is -1 (use all available cores).

    Returns
    -------
    None

    Notes
    -----
    Writes the generated table to a CSV file specified by INTERPOLATION_TABLES_PATH.
    Currently supports up to 3 indices. Alternating harmonic sums are not properly handled.
    Use supplied tables generated by NestedHarmonics.nb Mathematica notebook.
    """

    def compute_values(j_re, j_im):
        j = complex(j_re, j_im)
        if isinstance(indices,int):
            val = sp.harmonic_number(indices, j)
        else:
            val = sp.nested_harmonic_number(indices, j, interpolation=False)
        row = [[j_re, j_im, val]]

        # Add conjugated values
        if j_im > 0:
            row.append([j_re, -j_im, np.conj(val)])
        return row

    re_vals = np.arange(j_re_min, j_re_max + step, step)
    im_vals = np.arange(j_im_min, j_im_max + step, step)

    if isinstance(indices,int):
        m1 = indices
        filename = cfg.INTERPOLATION_TABLE_PATH / f"harmonic_m1_{m1}.csv"
    elif len(indices) == 2:
        m1, m2 = indices
        filename = cfg.INTERPOLATION_TABLE_PATH / f"nested_harmonic_m1_{m1}_m2_{m2}.csv"
    elif len(indices) == 3:
        m1, m2, m3 = indices
        filename = cfg.INTERPOLATION_TABLE_PATH / f"nested_harmonic_m1_{m1}_m2_{m2}_m3_{m3}.csv"
    else:
        raise ValueError("Table generation currently only supports 3 nested harmonics")

    # Generate grid points
    grid = [(j_re, j_im) for j_re in re_vals for j_im in im_vals]
    with hp.tqdm_joblib(tqdm(total=len(grid))) as progress_bar:
        # Parallel computation over (j_re, j_im) pairs
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_values)(j_re, j_im)
            for j_re, j_im in grid
        )

    # Flatten the list of lists
    data = [row for sublist in results for row in sublist]
    data.sort(key=lambda x: (x[0], x[1]))  # Sort by Re(j), Im(j)

    # Write to CSV
    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Re(j)", "Im(j)", "harmonic_number"])
        writer.writerows(data)

    print(f"Successfully wrote table to {filename}")

def generate_anomalous_dimension_table(suffix,moment_type,evolve_type,evolution_order="nlo",
                                            j_re_min=1e-4, j_re_max=6, j_im_min=0, j_im_max=110,
                                            step=0.25, n_jobs=-1):
    """
    Generate tables for interpolation of anomalous dimensions

    Parameters
    ----------
    suffix : str
        "qq", "qg", "gq", or "gg" anomalous dimensions
    moment_type : str
        "non_singlet_isovector" or "singlet"
    evolve_type : str
        "vector" or "axial"
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    j_re_min : float, optional
        Minimum real part of conformal spin j. Default is 1e-4.
    j_re_max : float, optional
        Maximum real part of conformal spin j. Default is 6.
    j_im_min : float, optional
        Minimum imaginary part of conformal spin j. Default is 0.
    j_im_max : float, optional
        Maximum imaginary part of conformal spin j. Default is 110.
    step : float, optional
        Step size in the j-plane for generating the grid. Default is 0.25.
    n_jobs : int, optional
        Number of parallel jobs to run. Default is -1 (use all available cores).

    Returns
    -------
    None

    Notes
    -----
    Writes the generated table to a CSV file specified by INTERPOLATION_TABLES_PATH.
    """
    def compute_values(j_re, j_im):
        j = mp.mpc(j_re,j_im)
        if suffix == "qq":
            val = adim.gamma_qq(j,moment_type,evolve_type,evolution_order,interpolation=False)
        elif suffix == "qg":
            val = adim.gamma_qg(j,evolve_type,evolution_order,interpolation=False)
        elif suffix == "gq":
            val = adim.gamma_gq(j,evolve_type,evolution_order,interpolation=False)
        elif suffix == "gg":
            val = adim.gamma_gg(j,evolve_type,evolution_order,interpolation=False)
        else:
            raise ValueError(f"Wrong suffix {suffix}")
        
        val = complex(val)
        row = [[j_re, j_im, val]]
        # Add conjugated values
        if j_im > 0:
            row.append([j_re, -j_im, np.conj(val)])
        return row
    hp.check_evolution_order(evolution_order)
    re_vals = np.arange(j_re_min, j_re_max + step, step)
    im_vals = np.arange(j_im_min, j_im_max + step, step)
    
    if moment_type != "singlet" and suffix == "qq":
        filename = cfg.INTERPOLATION_TABLE_PATH / f"gamma_{suffix}_non_singlet_{evolution_order}.csv"
    else:
        filename = cfg.INTERPOLATION_TABLE_PATH / f"gamma_{suffix}_{evolve_type}_{evolution_order}.csv"

    # Generate grid points
    grid = [(j_re, j_im) for j_re in re_vals for j_im in im_vals]
    with hp.tqdm_joblib(tqdm(total=len(grid))) as progress_bar:
        # Parallel computation over (j_re, j_im) pairs
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_values)(j_re, j_im)
            for j_re, j_im in grid
        )

    # Flatten the list of lists
    data = [row for sublist in results for row in sublist]
    data.sort(key=lambda x: (x[0], x[1]))  # Sort by Re(j), Im(j)

    # Write to CSV
    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Re(j)", "Im(j)", f"gamma_{suffix}"])
        writer.writerows(data)

    print(f"Successfully wrote table to {filename}")
