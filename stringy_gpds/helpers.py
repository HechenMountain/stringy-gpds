import numpy as np
import csv
import re
import os

from scipy.interpolate import RegularGridInterpolator, PchipInterpolator
from joblib.parallel import BatchCompletionCallBack

from . import config as cfg

##########################
#### Helper functions ####
##########################
def mpmath_vectorize(fn):
    """
    Decorator to enable mpmath functions to accept NumPy arrays.
    Also works with scalar inputs.

    Parameters
    ----------
    fn : callable
        The mpmath function to be vectorized.

    Returns
    -------
    callable
        A function that supports both scalar and NumPy array inputs.
    """
    vectorized_fn = np.vectorize(fn, otypes=[object])
    def wrapper(*args, **kwargs):
        if any(isinstance(arg, np.ndarray) for arg in args):
            return vectorized_fn(*args, **kwargs)
        else:
            return fn(*args, **kwargs)
    return wrapper

ERROR_MAP = {
"central": 0,  # central value
"plus": 1,     # upper bound
"minus": 2     # lower bound
}

####################
####   Checks   ####
####################

def check_evolution_order(evolution_order):
    """
    Check the evolution order.

    Parameters
    ----------
    evolution_order : str
        Must be either "lo", "nlo" or "nnlo"

    Raises
    ------
    ValueError
        If an unsupported evolution order is provided.
    """
    if evolution_order not in ["lo","nlo","nnlo"]:
        raise ValueError(f"Wrong evolution_order {evolution_order} for evolution equation")

def check_error_type(error_type):
    """
    Check the error type.

    Parameters
    ----------
    error_type : str
        Must be either "central", "plus", or "minus".

    Raises
    ------
    ValueError
        If an unsupported error type is provided.
    """
    if error_type not in ["central","plus","minus"]:
        raise ValueError(f"error_type must be central, plus or minus and not {error_type}")

def check_particle_type(particle):
    """
    Check the particle type.

    Parameters
    ----------
    particle : str
        Must be either "quark" or "gluon".

    Raises
    ------
    ValueError
        If an unsupported particle type is provided.
    """
    if particle not in ["quark", "gluon"]:
        raise ValueError(f"particle must be quark or gluon and not {particle}")
    
def check_moment_type_label(moment_type, moment_label):
    """
    Check the combination of moment type and moment label.

    Parameters
    ----------
    moment_type : str
        Must be either "non_singlet_isovector", "non_singlet_isoscalar", or "singlet".
    moment_label : str
        Must be either "A", "B", or "Atilde".

    Raises
    ------
    ValueError
        If the moment type or label is not supported or mismatched.
    """
    valid_combinations = {
        "non_singlet_isovector": {"A", "B", "Atilde"},
        "non_singlet_isoscalar": {"A", "B", "Atilde"},
        "singlet": {"A", "B", "Atilde"},
    }

    if moment_type not in valid_combinations:
        raise ValueError(f"Unsupported moment_type: {moment_type}")

    if moment_label not in valid_combinations[moment_type]:
        raise ValueError(f"Unsupported moment_label '{moment_label}' for moment_type '{moment_type}'")
    
def check_parity(parity):
    """
    Check the parity label.

    Parameters
    ----------
    parity : str
        Must be either "even", "odd", or "none".

    Raises
    ------
    ValueError
        If an unsupported parity is provided.
    """
    if parity not in ["even", "odd","none"]:
        raise ValueError(f"Parity must be even, odd or none and not {parity}")

def error_sign(error,error_type):
    """
    Apply the correct sign based on the error type.

    Parameters
    ----------
    error : array_like
        The input error values.
    error_type : str
        "central", "plus", or "minus".

    Returns
    -------
    ndarray
        Error values with corresponding sign
    """
    check_error_type(error_type)
    sign = -1 if error_type == "minus" else 1
    return sign * np.asarray(error)
    
def get_evolve_type(moment_label):
    """
    Determine the evolution type based on the moment label.

    Parameters
    ----------
    moment_label : str
        A(tilde), B(tilde) depending on H(tilde) or E(tilde) GPD etc.

    Returns
    -------
    str
        The evolution type: "vector" if moment_label is "A" or "B",
        "axial" if moment_label is "Atilde" or "Btilde".

    Raises
    ------
    ValueError
        If moment_label is not one of the expected values.
    """
    if moment_label in ["A", "B"]:
        evolve_type = "vector"
    elif moment_label in ["Atilde", "Btilde"]:
        evolve_type = "axial"
    else:
        raise ValueError(f"Invalid moment_label '{moment_label}'. Expected 'A', 'B', 'Atilde', or 'Btilde'.")
    return evolve_type

def get_regge_slope(moment_type, moment_label, evolution_order="nlo"):
    """
    Retrieve the Regge slope for a given moment configuration as defined in config.py

    Parameters
    ----------
    moment_type : str
        The type of moment (e.g., "Isovector").
    moment_label : str
        A(tilde), B(tilde) depending on H(tilde) or E(tilde) GPD etc.
    evolution_order : str, optional
        "lo", "nlo",... Default is "nlo"

    Returns
    -------
    float
        The Regge slope.

    Raises
    ------
    ValueError
        If the configuration is not found in REGGE_SLOPES defined in config.py .
    """
    check_moment_type_label(moment_type, moment_label)

    evolve_type = get_evolve_type(moment_label)

    try:
        return cfg.REGGE_SLOPES[evolve_type][moment_type][moment_label][evolution_order]
    except KeyError:
        raise ValueError(f"Missing Regge slope for: evolve_type={evolve_type}, moment_type={moment_type}, moment_label={moment_label}, evolution_order={evolution_order}")
    
def get_moment_normalizations(moment_type, moment_label, evolution_order="nlo"):
    """
    Retrieve the normalization factor for a given moment configuration
    as defined in MOMENT_NORMALIZATIONS in config.py

    Parameters
    ----------
    moment_type : str
        The type of moment (e.g., "Isovector").
    moment_label : str
        A(tilde), B(tilde) depending on H(tilde) or E(tilde) GPD etc.
    evolution_order : str, optional
        "lo", "nlo",... Default is "nlo"

    Returns
    -------
    float
        The normalization value.

    Raises
    ------
    ValueError
        If the configuration is not found in MOMENT_NORMALIZATIONS in config.py
    """
    check_moment_type_label(moment_type, moment_label)
    evolve_type = get_evolve_type(moment_label)

    try:
        return cfg.MOMENT_NORMALIZATIONS[evolve_type][moment_type][moment_label][evolution_order]
    except KeyError:
        raise ValueError(f"Missing moment normalization for: evolve_type={evolve_type}, moment_type={moment_type}, moment_label={moment_label}, evolution_order={evolution_order}")

##########################
####   File Handling  ####
##########################

def read_lattice_moment_data(particle,moment_type, moment_label, pub_id):
    """
    Read lattice data from a csv file, extracting 'n' values from the header and associating them with rows.
    Modify FILE_NAME and FILE_PATH as needed.

    Parameters
    ----------
    particle : str
        Either "quark" or "gluon".
    moment_type : str
        The type of moment (e.g., "Isovector").
    moment_label : str
        A(tilde), B(tilde) depending on H(tilde) or E(tilde) GPD etc.
    pub_id : str
        ArXiv publication ID

    Returns
    -------
    tuple
        A tuple (data, n_map), where `data` is the moment data, and `n_to_row_map` is a dictionary
        mapping conformal spin `n` values to moment values.
    
    Note
    ----
    File name structure is moment_type_particle_moments_moment_label_pub_id.csv. First column contains
    t values, and then A_n,0, A_n,0_error etc.
    """
    FILE_NAME = f"{moment_type}_{particle}_moments_{moment_label}_{pub_id}.csv"
    FILE_PATH = cfg.MOMENTUM_SPACE_MOMENTS_PATH / FILE_NAME

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
    Return central values forFn0 moment for a given conformal spin `n`, moment type, label, and publication ID.

    Parameters
    ----------
    n : int or list of int
        Conformal spin(s) for which to retrieve the central values.
    particle : str
        Either "quark" or "gluon".
    moment_type : str
        The type of moment (e.g., "Isovector").
    moment_label : str
        A(tilde), B(tilde) depending on H(tilde) or E(tilde) GPD etc.
    pub_id : str
        ArXiv publication ID

    Returns
    -------
    list
        Central values for Fn0 corresponding to the given input.
    """
    data, n_to_row_map = read_lattice_moment_data(particle,moment_type, moment_label, pub_id)

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
    Return associated errors for central values of Fn0 moment for a given conformal spin `n`, moment type, label, and publication ID.

    Parameters
    ----------
    n : int or list of int
        Conformal spin(s) for which to retrieve the central values.
    particle : str
        Either "quark" or "gluon".
    moment_type : str
        The type of moment (e.g., "Isovector").
    moment_label : str
        A(tilde), B(tilde) depending on H(tilde) or E(tilde) GPD etc.
    pub_id : str
        ArXiv publication ID

    Returns
    -------
    list
        Error values for Fn0 corresponding to the given input.
    """
    data, n_to_row_map = read_lattice_moment_data(particle,moment_type, moment_label, pub_id)

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

def read_Cz_data(particle,moment_type,pub_id_A,pub_id_Atilde):
    """
    Read in lattice data for spin-orbit corellation.

    Parameters
    ----------
    particle : str
        "quark" or "gluon
    moment_type : str
        "non_singlet_isovector", "non_singlet_isoscalar", "singlet"
    pub_id_A : str
        ArXiv publication ID containing A_1,0 moment
    pub_id_Atilde : str
        ArXiv publication ID containing Atilde_2,0 moment
        
    Returns
    -------
    tuple of list or None
        Tuple containing (t_values, C_z_values, C_z_errors), each a list.
        Returns None if no or insufficient data is available.
    
    Note
    ----
    Neglects m_q supressed term from transversity moments.
    """
    def t_values(moment_type, moment_label, pub_id):
        """Return the -t values for a given moment type, label, and publication ID."""
        data, n_to_row_map = read_lattice_moment_data(particle,moment_type, moment_label, pub_id)
        
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

    # Filter for NaN
    all_arrays = [
        A10_val, A10_err, A_t_vals,
        Atilde20_val, Atilde20_err, A_tilde_t_vals
    ]
    mask = np.ones_like(A10_val, dtype=bool)
    for arr in all_arrays:
        mask &= ~np.isnan(arr)
        
    A10_val       = A10_val[mask]
    A10_err       = A10_err[mask]
    A_t_vals      = A_t_vals[mask]

    Atilde20_val  = Atilde20_val[mask]
    Atilde20_err  = Atilde20_err[mask]
    A_tilde_t_vals = A_tilde_t_vals[mask]

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

def read_Lz_data(particle,moment_type,pub_id_A,pub_id_B,pub_id_Atilde):
    """
    Read in lattice data for orbital angular momentum.

    Parameters
    ----------
    particle : str
        "quark" or "gluon
    moment_type : str
        "non_singlet_isovector", "non_singlet_isoscalar", "singlet"
    pub_id_A : str
        ArXiv publication ID containing A_2,0 moment
    pub_id_B : str
        ArXiv publication ID containing B_2,0 moment
    pub_id_Atilde : str
        ArXiv publication ID containing Atilde_1,0 moment
        
    Returns
    -------
    tuple of list or None
        Tuple containing (t_values, L_z_values, L_z_errors), each a list.
        Returns None if no or insufficient data is available.
    """
    def t_values(moment_type, moment_label, pub_id):
        """Return the -t values for a given moment type, label, and publication ID."""
        data, n_to_row_map = read_lattice_moment_data(particle,moment_type, moment_label, pub_id)
        
        if data is not None:
            # Safely access data[:, 0] since data is not None
            return data[:, 0]
        return None  

    A20_val = Fn0_values(2, particle,moment_type, moment_label="A", pub_id=pub_id_A)
    A20_err = Fn0_errors(2, particle,moment_type, moment_label="A", pub_id=pub_id_A)
    A_t_vals = t_values(moment_type,"A",pub_id_A)

    B20_val = Fn0_values(2, particle,moment_type, moment_label="B", pub_id=pub_id_A)
    B20_err = Fn0_errors(2, particle,moment_type, moment_label="B", pub_id=pub_id_A)
    B_t_vals = t_values(moment_type,"B",pub_id_B)

    Atilde10_val = Fn0_values(1,particle,moment_type,"Atilde",pub_id_Atilde)
    Atilde10_err = Fn0_errors(1,particle,moment_type,"Atilde",pub_id_Atilde)
    A_tilde_t_vals = t_values(moment_type,"Atilde",pub_id_Atilde)

    # Filter for NaN
    all_arrays = [
        A20_val, A20_err, A_t_vals,
        B20_val, B20_err, B_t_vals,
        Atilde10_val, Atilde10_err, A_tilde_t_vals
    ]
    mask = np.ones_like(A20_val, dtype=bool)
    for arr in all_arrays:
        mask &= ~np.isnan(arr)

    A20_val       = A20_val[mask]
    A20_err       = A20_err[mask]
    A_t_vals      = A_t_vals[mask]

    B20_val       = B20_val[mask]
    B20_err       = B20_err[mask]
    B_t_vals      = B_t_vals[mask]

    Atilde10_val  = Atilde10_val[mask]
    Atilde10_err  = Atilde10_err[mask]
    A_tilde_t_vals = A_tilde_t_vals[mask]
    
    if (A20_val is None or (isinstance(A20_val, np.ndarray) and A20_val.size == 0)) or \
    (Atilde10_val is None or (isinstance(Atilde10_val, np.ndarray) and Atilde10_val.size == 0)) or \
    (B20_val is None or (isinstance(B20_val, np.ndarray) and B20_val.size == 0)):
        #print("No data found")
        return  None, None, None

    if np.any((A_tilde_t_vals != A_t_vals) | (A_t_vals != B_t_vals)):
        print("Warning: different t values encountered.")
        print(A_t_vals)
        print(A_tilde_t_vals)
    Lz = .5 * (A20_val + B20_val - Atilde10_val)
    Lz_err = .5 * np.sqrt((A20_err**2 + B20_err**2 + Atilde10_err**2))

    return A_t_vals, Lz, Lz_err

def generate_filename(eta, t, mu, prefix="FILE_NAME",error_type="central"):
    """
    Generate a filename based on eta, t, and mu formatted as three-digit values.

    Parameters
    ----------
    eta : float
        skewness parameter
    t : float 
        Mandelstam t
    mu : float
        Resolution scale
    
    Returns
    -------
    tuple or None
        A tuple (eta, t, mu, error_type) if the filename matches the expected pattern;
        otherwise, None.
    
    Note
    ----
    File name format is prefix_eee_ttt_mmm_error_type.csv where e = eta, t = t, m = mu
    and with error_type = "central" being omitted.
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

    Parameters
    ----------
    filename : str
        The full filename to parse.
    prefix : str, optional
        The prefix expected at the start of the filename. Default is "FILE_NAME".

    Returns
    -------
    tuple or None
        A tuple (eta, t, mu, error_type) if the filename matches the expected pattern;
        otherwise, None.
    
    Note
    ----
    File name format is prefix_eee_ttt_mmm_error_type.csv where e = eta, t = t, m = mu
    and with error_type = "central" being omitted.
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

def save_gpd_data(x_values, eta, t, mu,gpd_values,particle="quark",gpd_type="non_singlet_isovector",gpd_label="H",evolution_order="nlo",error_type="central"):
    """
    Save generated GPD data to csv.

    Parameters
    ----------
    x_values : array_like
        parton x values
    eta_in : float
        skewness parameter
    t_in : float 
        Mandelstam t
    mu_in : float
        Resolution scale
    gpd_values : array_like
        Associated GPD values
    particle : str
        "quark" or "gluon
    gpd_type : str
        "non_singlet_isovector", "non_singlet_isoscalar", "singlet"
    moment_label : str
        A(tilde), B(tilde) depending on H(tilde) or E(tilde) GPD etc.
    erorry_type : str
        "central", "plus" or "minus"

    Returns
    -------
    None
    """
    if len(x_values) != len(gpd_values):
        raise ValueError(f"x_values ({len(x_values)}) and gpd_values({len(gpd_values)}) are of unequal length")

    gpd_name = f"{gpd_type}_{particle}_GPD_{gpd_label}_{evolution_order}"
    filename = cfg.GPD_PATH / generate_filename(eta, t, mu,gpd_name,error_type)
    data = np.column_stack((x_values, gpd_values))
    np.savetxt(filename, data, delimiter=",")
    print(f"Saved data to {filename}")

def read_gpd_data(eta, t, mu,particle="quark",gpd_type="non_singlet_isovector",gpd_label="H",evolution_order="nlo",error_type ="central"):
    """
    Load generated GPD data from csv, extracting an array of parton x values and corresponding GPD values.

    Parameters
    ----------
    eta_in : float
        skewness parameter
    t_in : float 
        Mandelstam t
    mu_in : float
        Resolution scale
    particle : str
        "quark" or "gluon
    gpd_type : str
        "non_singlet_isovector", "non_singlet_isoscalar", "singlet"
    moment_label : str
        A(tilde), B(tilde) depending on H(tilde) or E(tilde) GPD etc.
    erorry_type : str
        "central", "plus" or "minus"

    Returns
    -------
    x_values, gpd_values : tuple of lists
    """

    gpd_name = f"{gpd_type}_{particle}_GPD_{gpd_label}_{evolution_order}"
    filename = cfg.GPD_PATH / generate_filename(eta, t, mu,gpd_name,error_type)

    if os.path.exists(filename):
        data = np.loadtxt(filename, delimiter=",")
        x_values, gpd_values = data[:, 0], data[:, 1]
        return x_values, gpd_values
    else:
        return None, None


def read_lattice_gpd_data(eta_in,t_in,mu_in,particle,gpd_type,gpd_label, pub_id,error_type="central"):
    """
    Load lattice gpd data from csv, extracting an array of parton x values and corresponding GPD values

    Parameters
    ----------
    eta_in : float
        skewness parameter
    t_in : float 
        Mandelstam t
    mu_in : float
        Resolution scale
    particle : str
        "quark" or "gluon
    gpd_type : str
        "non_singlet_isovector", "non_singlet_isoscalar", "singlet"
    moment_label : str
        A(tilde), B(tilde) depending on H(tilde) or E(tilde) GPD etc.
    pub_id : str
        ArXiv publication ID.
    erorry_type : str
        "central", "plus" or "minus"

    Returns
    -------
    x_values, gpd_values : tuple of lists
    """
    if (pub_id,gpd_type,gpd_label,eta_in,t_in,mu_in) in cfg.GPD_PUBLICATION_MAPPING:
        (color,parameter_set) = cfg.GPD_PUBLICATION_MAPPING[(pub_id,gpd_type,gpd_label,eta_in,t_in,mu_in)]
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
    """
    Save 2D Fourier transform data to csv.

    Flattens input 2D grid and writes data in a column format
    with structure: b_x_fm, b_y_fm, FT[b_x_fm, b_y_fm].

    Parameters
    ----------
    b_x_fm : array_like
        x component of impact parameter in fm
    b_y_fm : array_like
        y component of impact parameter in fm
    data : array_like
        2D array of Fourier-transformed data corresponding to the (b_x_fm, b_y_fm) grid.
    filename : str
        Path to the output csv.

    Returns
    -------
    None
    """
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
    """
    Read 2D Fourier transform data from csv.

    csv file contains three columns: b_x_fm, b_y_fm, and the
    corresponding Fourier-transformed values. Reconstructs the 2D grid.

    Parameters
    ----------
    filename : str
        Path to the csv containing the Fourier transformed data.

    Returns
    -------
    b_x_fm : array_like
        x component of impact parameter in fm
    b_y_fm : array_like
        y component of impact parameter in fm
    data : array_like
        2D array of Fourier-transformed data corresponding to the (b_x_fm, b_y_fm) grid.
    """
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

def update_dipole_csv(file_path, n, particle, moment_type, moment_label, evolution_order, A_D, m_D2,lattice=False):
    """
    Update or append the dipole parameters obtained by the fit in the csv.

    Parameters
    ----------
    file_path : str
        File path of csv table containing dipole parameters.
    n : int
        Conformal spin
    particle : str
        "quark" or "gluon".
    moment_type : str
        "non_singlet_isovector", "non_singlet_isoscalar", or "singlet".
    moment_label : str
        A(tilde), B(tilde) depending on H(tilde) or E(tilde) GPD etc.
    evolution_order : str
        lo, nlo.
    error_type : str
        Choose central, upper or lower value for input PDF parameters.
    A_D : float
        Dipole amplitude parameter.
    m_D2 : float
        Dipole mass parameter squared.
    lattice : bool, optional
        If True, updates the corresponding csv containing parameters for lattice dipole moments.
    
    Returns
    -------
    None

    Notes
    -----
    Either updates an existing row in the csv or appends a new row if no match is found.
    """
    file_path = cfg.Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # For laticce moments we use the pub_id instead of evolution_order
    if lattice :
        header = ["particle", "moment_type", "moment_label", "n", "pub_id", "A_D", "m_D2"]
    else:
        header = ["particle", "moment_type", "moment_label", "n", "evolution_order", "A_D", "m_D2"]
    key = (particle, moment_type, moment_label, str(n), evolution_order)

    rows = []
    found = False

    if file_path.exists():
        with open(file_path, newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Ensure header exists
        if rows and rows[0] != header:
            raise ValueError("CSV file header does not match expected format.")

        # Check and update existing row
        for i, row in enumerate(rows[1:], start=1):
            if tuple(row[:5]) == key:
                rows[i] = [particle, moment_type, moment_label, str(n), evolution_order, f"{A_D:.5f}", f"{m_D2:.5f}"]
                found = True
                break

    if not found:
        if not rows:
            rows.append(header)
        rows.append([particle, moment_type, moment_label, str(n), evolution_order, f"{A_D:.5f}", f"{m_D2:.5f}"])

    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

##########################
####   Interpolation  ####
##########################
class ComplexInterpolator:
    """
    Wrapper to make a complex-valued interpolator pickleable for use with joblib.

    Handles multi-dimensional interpolation where the first two input dimensions
    correspond to the real and imaginary parts of a complex number.

    Parameters
    ----------
    re_interp : callable
        Interpolator object returning the real part of the output.
    im_interp : callable
        Interpolator object returning the imaginary part of the output.

    Methods
    -------
    __call__(j_complex, *args)
        Evaluates the interpolator at the given complex conformal spin and additional arguments.
    """
    def __init__(self, re_interp, im_interp):
        self.re_interp = re_interp
        self.im_interp = im_interp

    def __call__(self, j_complex: complex, *args) -> complex:
        pt = [j_complex.real, j_complex.imag, *args]
        return complex(self.re_interp(pt).item(), self.im_interp(pt).item())

class ImagOnlyInterpolator:
    """
    Extracts and evaluates only the imaginary part for input of a base interpolator.

    Parameters
    ----------
    base_interp : callable
        Interpolator defined along the imaginary axis.

    Methods
    -------
    __call__(pt)
        Evaluates the interpolator using pt[1], assuming pt = (Re j, Im j, ...).
    """
    def __init__(self, base_interp):
        self.base_interp = base_interp

    def __call__(self, pt):
        return self.base_interp(pt[1]) 

class JointInterpolator:
    """
    Combines two separate interpolators to return a tuple of results. We handle using 
    only the imaginary part in ImagOnlyInterpolator for reusability.

    Parameters
    ----------
    interp0 : callable
        First interpolator.
    interp1 : callable
        Second interpolator.

    Methods
    -------
    __call__(x)
        Evaluates both interpolators at the given point and returns their outputs as a tuple.
    """
    def __init__(self, interp0, interp1):
        self.interp0 = interp0
        self.interp1 = interp1

    def __call__(self, x):
        return self.interp0(x), self.interp1(x)

# Cache interpolation
@cfg.memory.cache
def build_harmonic_interpolator(indices):
    """
    Constructs a complex-valued interpolator for the (nested) harmonic numbers
    appearing in the evolution equations using RegularGridInterpolator interpolation
    based on the given index set.

    Parameters
    ----------
    indices : list or tuple
        A list of integers specifying the nested harmonic sum indices (e.g., 1, [2,1], etc.).

    Returns
    -------
    Pickleable scipy.interpolate.RegularGridInterpolator interpolator object for evaluating the anomalous dimension

    Notes
    -----
    This function is cached using joblib to avoid redundant computation.
    """
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
    re_interp = RegularGridInterpolator((re_j_vals, im_j_vals), real_grid, bounds_error=False, fill_value=None,method='pchip')
    im_interp = RegularGridInterpolator((re_j_vals, im_j_vals), imag_grid, bounds_error=False, fill_value=None,method='pchip')

    return ComplexInterpolator(re_interp, im_interp)

# Cache interpolation
@cfg.memory.cache
def build_gamma_interpolator(suffix,moment_type,evolve_type,evolution_order):
    """
    Constructs a complex-valued interpolator for the anomalous dimensions
    appearing in the evolution equations using RegularGridInterpolator interpolation
    based on the given suffix and evolution settings.

    Parameters
    ----------
    suffix : str
        "qq", "qg", "gq", or "gg" anomalous dimensions
    moment_type : str
        "non_singlet_isovector" or "singlet"
    evolve_type : str
        "vector" or "axial"
    evolution_order : str, optional
        "lo", "nlo".... Default is "nlo".

    Returns
    -------
    Pickleable scipy.interpolate.RegularGridInterpolator interpolator object for evaluating the anomalous dimension

    Notes
    -----
    This function is cached using joblib to avoid redundant computation.
    """
    check_evolution_order(evolution_order)
    if moment_type != "singlet" and suffix == "qq":
        filename = cfg.INTERPOLATION_TABLE_PATH / f"gamma_{suffix}_non_singlet_{evolution_order}.csv"
    else:
        filename = cfg.INTERPOLATION_TABLE_PATH / f"gamma_{suffix}_{evolve_type}_{evolution_order}.csv"

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
    re_interp = RegularGridInterpolator((re_j_vals, im_j_vals), real_grid, bounds_error=False, fill_value=None,method='pchip')
    im_interp = RegularGridInterpolator((re_j_vals, im_j_vals), imag_grid, bounds_error=False, fill_value=None,method='pchip')

    # return interpolator
    return ComplexInterpolator(re_interp, im_interp)

# Cache interpolation
@cfg.memory.cache
def build_moment_interpolator(eta,t,mu,solution,particle,moment_type,moment_label, evolution_order, error_type):
    """
    Constructs a complex-valued interpolator for the evolved conformal moment 
    using pchip interpolation based on the given kinematic parameters and evolution settings.

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
        "non_singlet_isovector", "non_singlet_isoscalar", or "singlet".
    moment_label : str
        A(tilde), B(tilde) depending on H(tilde) or E(tilde) GPD etc.
    evolution_order : str
        lo, nlo.
    error_type : str
        Choose central, upper or lower value for input PDF parameters.

    Returns
    -------
    Pickleable scipy.interpolate.PchipInterpolator interpolator object for evaluating the evolved conformal moment.

    Notes
    -----
    This function is cached using joblib to avoid redundant computation.
    """
    check_particle_type(particle)
    check_error_type(error_type)
    check_moment_type_label(moment_type,moment_label)
    check_evolution_order(evolution_order)
    # Build filename
    if mu != 1 or (moment_type == "singlet" and solution not in ["+","-"]) or moment_type != "singlet":
        prefix = cfg.INTERPOLATION_TABLE_PATH / f"{moment_type}_{particle}_moments_{moment_label}_{evolution_order}"
    elif moment_type == "singlet":
        prefix = cfg.INTERPOLATION_TABLE_PATH/ f"{moment_type}_{solution}_moments_{moment_label}_{evolution_order}"
    filename = generate_filename(eta, t, mu, prefix, error_type)

    im_j_vals = []
    val0_list = []
    val1_list = []

    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header
        for row in reader:
            im_j = float(row[0])
            val0 = complex(row[1].replace(" ", ""))
            val1 = complex(row[2].replace(" ", ""))

            im_j_vals.append(im_j)
            val0_list.append(val0)
            val1_list.append(val1)

    # Sort axes
    sorted_data = sorted(zip(im_j_vals, val0_list, val1_list), key=lambda x: x[0])
    im_j_vals, val0_list, val1_list = zip(*sorted_data)
    im_j_vals = np.array(im_j_vals)

    # Separate real and imaginary parts
    val0_real = np.array([v.real for v in val0_list])
    val0_imag = np.array([v.imag for v in val0_list])
    val1_real = np.array([v.real for v in val1_list])
    val1_imag = np.array([v.imag for v in val1_list])

    # Create interpolators for real and imaginary parts
    base_re0 = PchipInterpolator(im_j_vals, val0_real, extrapolate=True)
    base_im0 = PchipInterpolator(im_j_vals, val0_imag, extrapolate=True)
    base_re1 = PchipInterpolator(im_j_vals, val1_real, extrapolate=True)
    base_im1 = PchipInterpolator(im_j_vals, val1_imag, extrapolate=True)

    # Discard real parts in call as it is fixed by j_base
    re0_interp = ImagOnlyInterpolator(base_re0)
    im0_interp = ImagOnlyInterpolator(base_im0)
    re1_interp = ImagOnlyInterpolator(base_re1)
    im1_interp = ImagOnlyInterpolator(base_im1)

    interp0 = ComplexInterpolator(re0_interp, im0_interp)
    interp1 = ComplexInterpolator(re1_interp, im1_interp)

    return JointInterpolator(interp0, interp1)

##########################
####   Progress Bar   ####
##########################

class TqdmBatchCompletionCallback(BatchCompletionCallBack):
    """
    Custom Joblib BatchCompletionCallback to updates tqdm progress bar.

    Used to track progress of parallel jobs.

    Parameters
    ----------
    *args : tuple
        Positional arguments passed to the base BatchCompletionCallBack.
    tqdm_object : tqdm.tqdm, optional
        A tqdm progress bar object to be updated after each completed batch.
    **kwargs : dict
        Keyword arguments passed to the base BatchCompletionCallBack.

    Methods
    -------
    __call__(*args, **kwargs)
        Called when a batch is completed. Updates the tqdm progress bar.
    """
    def __init__(self, *args, tqdm_object=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.tqdm_object = tqdm_object

    def __call__(self, *args, **kwargs):
        if self.tqdm_object:
            self.tqdm_object.update(n=self.batch_size)
        return super().__call__(*args, **kwargs)

def tqdm_joblib(tqdm_object):
    """
    Patches Joblib to report progress via a tqdm progress bar.

    Replaces Joblib's default BatchCompletionCallBack with
    a custom version that updates the provided tqdm progress bar.

    Parameters
    ----------
    tqdm_object : tqdm.tqdm
        The tqdm progress bar to be updated during parallel processing.

    Returns
    -------
    tqdm.tqdm
        The same tqdm object passed in, for use in a with-statement or tracking externally.

    Examples
    --------
    >>> with tqdm_joblib(tqdm(total=10)) as progress_bar:
    >>>     Parallel(n_jobs=2)(delayed(some_func)(i) for i in range(10))
    """
    import joblib
    joblib.parallel.BatchCompletionCallBack = lambda *args, **kwargs: TqdmBatchCompletionCallback(*args, tqdm_object=tqdm_object, **kwargs)
    return tqdm_object