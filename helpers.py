import numpy as np
import csv
import re
import os

from scipy.interpolate import RegularGridInterpolator
from joblib.parallel import BatchCompletionCallBack

import config as cfg

##########################
#### Helper functions ####
##########################

####################
####   Checks   ####
####################

def check_evolution_order(evolution_order):
    if evolution_order not in ["LO","NLO","NNLO"]:
        raise ValueError(f"Wrong evolution_order {evolution_order} for evolution equation")

def check_error_type(error_type):
    if error_type not in ["central","plus","minus"]:
        raise ValueError(f"error_type must be central, plus or minus and not {error_type}")

def check_particle_type(particle):
    if particle not in ["quark", "gluon"]:
        raise ValueError(f"particle must be quark or gluon and not {particle}")
    
def check_moment_type_label(moment_type, moment_label):
    valid_combinations = {
        "non_singlet_isovector": {"A", "B", "Atilde"},
        "non_singlet_isoscalar": {"A", "B", "Atilde"},
        "singlet": {"A", "B", "Atilde"},
    }

    if moment_type not in valid_combinations:
        raise ValueError(f"Unsupported moment_type: {moment_type}")

    if moment_label not in valid_combinations[moment_type]:
        raise ValueError(f"Unsupported moment_label '{moment_label}' for moment_type '{moment_type}'")

def check_evolve_type(evolve_type):
    if evolve_type not in ["vector","axial"]:
        raise ValueError(f"evolve_type must be vector or axial and not {evolve_type}.")
    
def check_parity(parity):
    if parity not in ["even", "odd","none"]:
        raise ValueError(f"Parity must be even, odd or none and not {parity}")

def error_sign(error,error_type):
    check_error_type(error_type)
    if error_type == "minus":
        return -error
    else:
        return error
    
def get_regge_slope(moment_type, moment_label, evolve_type, evolution_order="LO"):
    check_moment_type_label(moment_type, moment_label)
    check_evolve_type(evolve_type)

    try:
        return cfg.REGGE_SLOPES[evolve_type][moment_type][moment_label][evolution_order]
    except KeyError:
        raise ValueError(f"Missing Regge slope for: evolve_type={evolve_type}, moment_type={moment_type}, moment_label={moment_label}, evolution_order={evolution_order}")
    
def get_moment_normalizations(moment_type, moment_label, evolve_type, evolution_order="LO"):
    check_moment_type_label(moment_type, moment_label)
    check_evolve_type(evolve_type)

    try:
        return cfg.MOMENT_NORMALIZATIONS[evolve_type][moment_type][moment_label][evolution_order]
    except KeyError:
        raise ValueError(f"Missing moment normalization for: evolve_type={evolve_type}, moment_type={moment_type}, moment_label={moment_label}, evolution_order={evolution_order}")

##########################
####   File Handling  ####
##########################

def load_lattice_moment_data(particle,moment_type, moment_label, pub_id):
    """
    Load data from a .csv file, extracting 'n' values from the header and associating them with rows. 
    Modify FILE_NAME and FILE_PATH as needed

    Args:
    - particle (str): quark or gluon
    - moment_type (str): The type of moment (e.g., "Isovector").
    - moment_label (str): The label of the moment (e.g., "A").
    - pub_id (str): The publication ID.

    Returns:
        tuple: A tuple containing the data and a dictionary mapping 'n' values to row indices.
    """
    moment_path = cfg.MOMENTUM_SPACE_MOMENTS_PATH
    FILE_NAME = f"{moment_type}_{particle}_moments_{moment_label}_{pub_id}.csv"
    FILE_PATH = f"{moment_path}{FILE_NAME}"

    # Check if the file exists
    if not os.path.exists(FILE_PATH):
        # print(f"No data available for {moment_type}{moment_label} in {pub_id}")
        return None, None

    with open(FILE_PATH, 'r') as f:
        # Read and split the header by commas
        header = f.readline().strip().split(',')

        # Load the rest of the file as data
        data = np.loadtxt(f, delimiter=',')

    # Extract 'n' values from the header
    n_values = []
    for col_name in header:
        match = re.search(rf'{moment_label}(\d+)0_', col_name)
        if match:
            n_values.append(int(match.group(1)))

    # Create a dictionary mapping 'n' values to row indices
    n_to_row_map = {n: i for i, n in enumerate(n_values)}

    return data, n_to_row_map

def Fn0_values(n,particle, moment_type, moment_label, pub_id):
    """
    Return central values for An0 for a given n, moment type, label, and publication ID.
    """
    data, n_to_row_map = load_lattice_moment_data(particle,moment_type, moment_label, pub_id)

    if data is None and n_to_row_map is None:
        #print(f"No data found for {moment_type} {moment_label} {pub_id}. Skipping.")
        return None
    
    # Check if the requested n is available in the data
    if n not in n_to_row_map:
        return None  # Requested n does not exist
    
    # Get the corresponding column index for An0_val
    col_idx = n_to_row_map[n]
    
    # Return the column data
    return data[:, col_idx]

def Fn0_errors(n,particle, moment_type, moment_label, pub_id):
    """
    Return errors for An0 for a given n, moment type, label, and publication ID.
    """
    data, n_to_row_map = load_lattice_moment_data(particle,moment_type, moment_label, pub_id)

    if data is None and n_to_row_map is None:
        #print(f"No data found for {moment_type} {moment_label} {pub_id}. Skipping.")
        return None 
    
    # Check if the requested n is available in the data
    if n not in n_to_row_map:
        return None  # Requested n does not exist
    
    # Get the corresponding column index for An0_err
    col_idx = n_to_row_map[n]+1
    
    # Return the column data
    return data[:, col_idx]

def load_Cz_data(particle,moment_type,pub_id_A,pub_id_Atilde):
    def t_values(moment_type, moment_label, pub_id):
        """Return the -t values for a given moment type, label, and publication ID."""
        data, n_to_row_map = load_lattice_moment_data(particle,moment_type, moment_label, pub_id)
        
        if data is not None:
            # Safely access data[:, 0] since data is not None
            return data[:, 0]
        return None  

    A10_val = Fn0_values(1,particle,moment_type,"A",pub_id_A)
    A10_err = Fn0_errors(1,particle,moment_type,"A",pub_id_A)
    A_t_vals = t_values(moment_type,"A",pub_id_A)
    Atilde20_val = Fn0_values(2,particle,moment_type,"Atilde",pub_id_Atilde)
    Atilde20_err = Fn0_errors(2,particle,moment_type,"Atilde",pub_id_Atilde)
    A_tilde_t_vals = t_values(moment_type,"Atilde",pub_id_Atilde)

    if (A10_val is None or (isinstance(A10_val, np.ndarray) and A10_val.size == 0)) or \
    (Atilde20_val is None or (isinstance(Atilde20_val, np.ndarray) and Atilde20_val.size == 0)):
        #print("No data found")
        return  None, None, None

    if np.any(A_tilde_t_vals != A_t_vals):
        print("Warning: different t values encountered.")
        print(A_t_vals)
        print(A_tilde_t_vals)
    Cz = (Atilde20_val - A10_val)/2
    Cz_err = np.sqrt(A10_err**2+Atilde20_err**2)/2

    return A_t_vals, Cz, Cz_err

def generate_filename(eta, t, mu, prefix="FILE_NAME",error_type="central"):
    """
    Generate a filename based on eta, t, and mu formatted as three-digit values.
    """
    error_mapping = {
        "central": "",  # The column with the central value
        "plus": "_plus",     # The column with the + error value
        "minus": "_minus"     # The column with the - error value
    }
    eta_str = f"{abs(eta):.2f}".replace(".", "").zfill(3)
    t_str = f"{abs(t):.2f}".replace(".", "").zfill(3)
    mu_str = f"{abs(mu):.2f}".replace(".", "").zfill(3)
    err_str = error_mapping.get(error_type)
    filename = f"{prefix}_{eta_str}_{t_str}_{mu_str}{err_str}.csv"
    return filename

def parse_filename(filename, prefix="FILE_NAME"):
    """
    Extract eta, t, mu, and error_type from a filename if it matches the expected format.
    Returns a tuple (eta, t, mu, error_type) or None if not a match.
    """
    pattern = re.compile(rf"{prefix}_(\d{{3}})_(\d{{3}})_(\d{{3}})(_plus|_minus)?\.csv")
    match = pattern.match(filename)
    if match:
        eta = float(match.group(1)) / 100
        t = float(match.group(2)) / 100
        mu = float(match.group(3)) / 100
        error_suffix = match.group(4)
        error_type = "central"
        if error_suffix == "_plus":
            error_type = "plus"
        elif error_suffix == "_minus":
            error_type = "minus"
        return eta, t, mu, error_type
    return None

def save_gpd_data(x_values, eta, t, mu,y_values,particle="quark",gpd_type="non_singlet_isovector",gpd_label="H",error_type="central"):
    """
    Save the function f(x, eta, t, mu) evaluated at x_values to a CSV file.
    """
    if len(x_values) != len(y_values):
        raise ValueError(f"x_values ({len(x_values)}) and y_values({len(y_values)}) are of unequal length")

    gpd_name = gpd_type + "_" + particle + "_GPD_" + gpd_label

    filename = cfg.GPD_PATH / generate_filename(eta, t, mu,gpd_name,error_type)
    data = np.column_stack((x_values, y_values))
    np.savetxt(filename, data, delimiter=",")
    print(f"Saved data to {filename}")

def load_gpd_data(eta, t, mu,particle="quark",gpd_type="non_singlet_isovector",gpd_label="H",error_type ="central"):
    """
    Load data from CSV if it exists, otherwise return None.
    """

    gpd_name = gpd_type + "_" + particle + "_GPD_" + gpd_label
    filename = cfg.GPD_PATH / generate_filename(eta, t, mu,gpd_name,error_type)
    if os.path.exists(filename):
        data = np.loadtxt(filename, delimiter=",")
        x_values, y_values = data[:, 0], data[:, 1]
        return x_values, y_values
    else:
        return None, None


def load_lattice_gpd_data(eta_in,t_in,mu_in,particle,gpd_type,gpd_label, pub_id,error_type="central"):
    """
    Load data from a .csv file, extracting 
    Modify FILE_NAME and FILE_PATH as needed

    Args:
        eta_in (float): skewness parameter
        t_in (float): Mandelstam t
        mu_in (float): Resolution scale
        gpd_type (str): The type of moment (e.g., "Isovector").
        moment_label (str): The label of the moment (e.g., "A").
        pub_id (str): The publication ID.
        erorry_type(str): "central", "plus" or "minus"
    Returns:
        tuple: A tuple containing the data and a dictionary mapping 'n' values to row indices.
    """
    if (pub_id,gpd_type,gpd_label,eta_in,t_in,mu_in) in cfg.GPD_cfg.PUBLICATION_MAPPING:
        (color,parameter_set) = cfg.GPD_cfg.PUBLICATION_MAPPING[(pub_id,gpd_type,gpd_label,eta_in,t_in,mu_in)]
    else:
        #print("No data found on filesystem")
        return None, None, None
    check_error_type(error_type)
    error_mapping = {
        "central": "",  # The column with the central value
        "plus": "_plus",     # The column with the + error value
        "minus": "_minus"     # The column with the - error value
    }
    error = error_mapping.get(error_type)
    gpd_name = gpd_type + "_" + particle + "_GPD_" + gpd_label
    
    FILE_NAME = f"{gpd_name}_{pub_id}_{parameter_set}{error}.csv"
    FILE_PATH = cfg.GPD_PATH / FILE_NAME

    # Check if the file exists
    if not os.path.exists(FILE_PATH):
        # print(f"No data available for {moment_type}{moment_label} in {pub_id}")
        return None, None
    data = []

    with open(FILE_PATH, 'r') as f:
        # Read and split the header by commas
        comment = f.readline()
        header = f.readline().strip().split(',')

        # Load the rest of the file as data
        data = np.loadtxt(f, delimiter=',')
        x_values = data[:,0]
        gpd_values = data[:,1]
    return x_values, gpd_values

def save_ft_to_csv(b_x_fm, b_y_fm, data, filename):
    header = "b_x_fm,b_y_fm,FT[b_x_fm,b_y_fm]"
    
    # Ensure correct shape: Convert 2D grid into column format
    b_x_flat = np.repeat(b_x_fm, len(b_y_fm))  # Repeat each b_x value for all b_y
    b_y_flat = np.tile(b_y_fm, len(b_x_fm))    # Tile b_y values for each b_x
    data_flat = data.ravel()                   # Flatten the data
    
    # Stack columns and save
    data_with_b_x_b_y = np.column_stack((b_x_flat, b_y_flat, data_flat))
    np.savetxt(filename, data_with_b_x_b_y, delimiter=",", header=header, comments='')
    print(f"Saved data to {filename}")
def read_ft_from_csv(filename):
    # Load the data (skip the header row)
    loaded_data = np.loadtxt(filename, delimiter=",", skiprows=1)

    # Extract columns
    b_x_flat, b_y_flat, data_flat = loaded_data[:, 0], loaded_data[:, 1], loaded_data[:, 2]

    # Reconstruct unique x and y values
    b_x_fm = np.unique(b_x_flat)
    b_y_fm = np.unique(b_y_flat)

    # Reshape data back into 2D grid
    data = data_flat.reshape(len(b_y_fm), len(b_x_fm))  # Must match original shape

    return b_x_fm, b_y_fm, data

##########################
####   Interpolation  ####
##########################
class HarmonicInterpolator:
    def __init__(self, re_interp, im_interp):
        self.re_interp = re_interp
        self.im_interp = im_interp

    def __call__(self, j_complex: complex) -> complex:
        pt = [j_complex.real, j_complex.imag]
        return complex(self.re_interp(pt).item(), self.im_interp(pt).item())

# Cache interpolation
@cfg.memory.cache
def harmonic_interpolator(indices):
    if isinstance(indices,int):
        m1 = indices
        filename = cfg.ANOMALOUS_DIMENSIONS_PATH / f"harmonic_m1_{m1}.csv"
    elif len(indices) == 2:
        m1, m2 = indices
        filename = cfg.ANOMALOUS_DIMENSIONS_PATH / f"nested_harmonic_m1_{m1}_m2_{m2}.csv"
    elif len(indices) == 3:
        m1, m2, m3 = indices
        filename = cfg.ANOMALOUS_DIMENSIONS_PATH / f"nested_harmonic_m1_{m1}_m2_{m2}_m3_{m3}.csv"
    else:
        raise ValueError("harmonic_interpolator currently only supports 3 nested harmonics")

    re_j_set = set()
    im_j_set = set()
    values = {}

    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header
        for row in reader:
            re_j = float(row[0])
            im_j = float(row[1])
            val = complex(row[2].replace(" ", ""))

            re_j_set.add(re_j)
            im_j_set.add(im_j)
            values[(re_j, im_j)] = val

    # Sort axes
    re_j_vals = sorted(re_j_set)
    im_j_vals = sorted(im_j_set)

    # Create grid of complex values
    real_grid = np.empty((len(re_j_vals), len(im_j_vals)))
    imag_grid = np.empty((len(re_j_vals), len(im_j_vals)))

    for i, re in enumerate(re_j_vals):
        for j, im in enumerate(im_j_vals):
            val = values[(re, im)]
            real_grid[i, j] = val.real
            imag_grid[i, j] = val.imag

    # Create interpolators for real and imaginary parts
    re_interp = RegularGridInterpolator((re_j_vals, im_j_vals), real_grid, bounds_error=False, fill_value=None)
    im_interp = RegularGridInterpolator((re_j_vals, im_j_vals), imag_grid, bounds_error=False, fill_value=None)

    # def interpolator(j_complex):
    #     # print(m1,m2,j_complex)
    #     pt = [j_complex.real, j_complex.imag]
    #     return complex(re_interp(pt).item(), im_interp(pt).item())

    # return interpolator
    return HarmonicInterpolator(re_interp, im_interp)

##########################
####   Progress Bar   ####
##########################

class TqdmBatchCompletionCallback(BatchCompletionCallBack):
    def __init__(self, *args, tqdm_object=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tqdm_object = tqdm_object

    def __call__(self, *args, **kwargs):
        if self.tqdm_object:
            self.tqdm_object.update(n=self.batch_size)
        return super().__call__(*args, **kwargs)

def tqdm_joblib(tqdm_object):
    import joblib
    joblib.parallel.BatchCompletionCallBack = lambda *args, **kwargs: TqdmBatchCompletionCallback(*args, tqdm_object=tqdm_object, **kwargs)
    return tqdm_object