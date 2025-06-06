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
    Generate tables for interpolation of input and evolved moments. The real part of j is defined by the value in get_j_base
    """
    if moment_label in ["A","B"]:
        evolve_type = "vector"
    elif moment_label in ["Atilde","Btilde"]:
        evolve_type = "axial"

    def compute_moment(j):
        # For mu != 1 we interpolate the evolved moments
        if mu != 1 or (moment_type == "singlet" and solution not in ["+","-"]):
            return core.evolve_conformal_moment(j, eta, t,mu,A0=1,particle=particle,moment_type=moment_type,
                                           moment_label=moment_label,evolution_order=evolution_order,
                                           error_type=error_type)
        # For mu = 1 we use the corresponding input moments
        elif moment_type == "non_singlet_isovector":
            return core.non_singlet_isovector_moment(j,eta,t,
                                                moment_label=moment_label, evolve_type=evolve_type,
                                                evolution_order=evolution_order, error_type=error_type)
        elif moment_type == "non_singlet_isoscalar":
            return core.non_singlet_isoscalar_moment(j,eta,t,
                                                moment_label=moment_label, evolve_type=evolve_type,
                                                evolution_order=evolution_order, error_type=error_type)
        elif moment_type == "singlet":
            val = core.singlet_moment(j, eta, t, 
                                  moment_label=moment_label, evolve_type=evolve_type,
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
    re_j_b, _ = core.get_j_base(particle=particle,moment_type=moment_type,moment_label=moment_label)
    # First value is lower value of nd evolution when resummed
    # Second value is real part of Mellin-Barnes integration variable
    # Third part is intermediate value since pchip needs >= 4 data points.
    # Fourth part is from shift in fractional finite sum:
    # k + j - k0 + 1 where k0 = 1 for non-singlet and 2 for singlet
    if moment_type != "singlet":
        re_j = [0.8, re_j_b,re_j_b + 0.4, re_j_b + 0.8]
    else:
        re_j = [1.8, re_j_b,re_j_b + 0.4, re_j_b + 0.8]
    im_j = np.arange(0.0, im_j_max + step, step)  # Im ≥ 0 only
    # Generate grid points
    grid_points = [
        (rj, ij)
        for rj in re_j
        for ij in im_j
    ]
    def compute_wrapper(rj, ij):
        return (rj, ij, compute_moment(complex(rj, ij)))

    with hp.tqdm_joblib(tqdm(total=len(grid_points))) as progress_bar:
        results = Parallel(n_jobs=-1)(
            delayed(compute_wrapper)(rj, ij) for rj, ij in grid_points
        )

    # Add mirrored points: f(j*) = f(j)* for Im(j) < 0
    mirrored = []
    for rj, ij, val in results[1:]:  # skip ij = 0 to avoid duplication
        if ij > 0:
            mirrored.append((rj, -ij, np.conj(val)))

    all_data = results + mirrored
    all_data.sort(key=lambda x: (x[0], x[1]))  # sort by Re(j), Im(j)

    # Write CSV
    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Re(j)", "Im(j)", "value"])
        for rj, ij, val in all_data:
            writer.writerow([rj, ij, val])

    print(f"Successfully wrote table to {filename}")

def generate_harmonic_table(indices,j_re_min=0, j_re_max=10, j_im_min=0, j_im_max=110,step=0.25, n_jobs=-1):
    def compute_values(j_re, j_im):
        j = complex(j_re, j_im)
        if isinstance(indices,int):
            val = sp.harmonic_number(indices, j)
        else:
            val = sp.nested_harmonic_number(indices, j, interpolation=False)
        row = [[j_re, j_im, val]]
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
    
def generate_nested_harmonic_table(m1, m2,
                                            j_re_min=1e-4, j_re_max=3, j_im_min=0, j_im_max=110,
                                            step=0.25, n_jobs=-1):
    def compute_values(j_re, j_im, m1, m2):
        j = complex(j_re, j_im)
        val = sp.nested_harmonic_number([m1, m2], j, interpolation=False)
        row = [[j_re, j_im, val]]
        if j_im > 0:
            row.append([j_re, -j_im, np.conj(val)])
        return row

    re_vals = np.arange(j_re_min, j_re_max + step, step)
    im_vals = np.arange(j_im_min, j_im_max + step, step)

    filename = cfg.INTERPOLATION_TABLE_PATH / f"nested_harmonic_m1_{m1}_m2_{m2}.csv"

    # Generate grid points
    grid = [(j_re, j_im) for j_re in re_vals for j_im in im_vals]
    with hp.tqdm_joblib(tqdm(total=len(grid))) as progress_bar:
        # Parallel computation over (j_re, j_im) pairs
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_values)(j_re, j_im, m1, m2)
            for j_re, j_im in grid
        )

    # Flatten the list of lists
    data = [row for sublist in results for row in sublist]
    data.sort(key=lambda x: (x[0], x[1]))  # Sort by Re(j), Im(j)

    # Write to CSV
    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Re(j)", "Im(j)", "nested_harmonic_number"])
        writer.writerows(data)

    print(f"Successfully wrote table to {filename}")

def generate_anomalous_dimension_table(suffix,moment_type,evolve_type,evolution_order="nlo",
                                            j_re_min=1e-4, j_re_max=6, j_im_min=0, j_im_max=110,
                                            step=0.25, n_jobs=-1):
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
