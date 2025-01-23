# # Dependencies
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt
from scipy.integrate import quad, trapezoid
from joblib import Parallel, delayed
from scipy.special import gamma, digamma

import time
import re
import os

from mstw_pdf import MSTW_PDF,MSTW_PDF_LO


########################################
#### Dictionaries and data handling ####
########################################
BASE_PATH = "/mnt/c/Users/flori/Documents/PostDoc/Data/GPD/"
# Add some colors
saturated_pink = (1.0, 0.1, 0.6)  


PUBLICATION_MAPPING = {
    "2305.11117": ("cyan",2),
    "0705.4295": ("orange",2),
    "1908.10706": (saturated_pink,2),
    "2410.03539": ("green",2)
# Add more publication IDs and corresponding colors here
}

def initialize_moment_to_function():
    # Define dictionary that maps conformal moments names and types to expressions
    global MOMENT_TO_FUNCTION
    MOMENT_TO_FUNCTION = {
    # Contains a Pair of moment_type and moment_label to match input PDF and evolution type
    ("NonSingletIsovector", "A"): (non_singlet_isovector_moment,"vector"),
    ("NonSingletIsovector", "Atilde"): (non_singlet_isovector_moment,"axial"),
    ("NonSingletIsoscalar", "A"): (non_singlet_isoscalar_moment,"vector"),
    ("NonSingletIsoscalar", "Atilde"): (non_singlet_isoscalar_moment,"axial"),
    ("Singlet","A"): (singlet_moment, "vector"),
    ("Singlet","Atilde"): (singlet_moment, "axial"),
    }

def get_regge_slope(moment_type,moment_label,evolve_type):
    """Set Regge slopes, modify manually
    
    Parameters:
    - moment_type (str.): NonSingletIsovector, NonSingletIsoscalar, Singlet
    - moment_label (str.): A, A_tide...
    - evolve_type (str.): Type of evolution equation
    """
    check_moment_type_label(moment_type,moment_label)
    check_evolve_type(evolve_type)

    if evolve_type == "vector":
        if moment_type == "NonSingletIsovector":
            if moment_label == "A":
                alpha_prime = 1.069
                return alpha_prime
        if moment_type == "NonSingletIsoscalar":
            if moment_label == "A":
                alpha_prime = 0.891
                return alpha_prime
        if moment_type == "Singlet":
            if moment_label == "A":
                alpha_prime_s = 1.828
                alpha_prime_T = 0.627
                alpha_prime_S = 4.277
                return alpha_prime_s, alpha_prime_T, alpha_prime_S
    elif evolve_type == "axial":
        if moment_type == "NonSingletIsovector":
            if moment_label == "Atilde":
                alpha_prime = 0.399939
                return alpha_prime
        if moment_type == "NonSingletIsoscalar":
            if moment_label == "Atilde":
                alpha_prime = 0.247658
                return alpha_prime
        if moment_type == "Singlet":
            if moment_label == "A":
                print("Axial Singlet is to do")
                alpha_prime_s = 1
                alpha_prime_T = 1
                alpha_prime_S = 1
                return alpha_prime_s, alpha_prime_T, alpha_prime_S
    else:
        raise ValueError(f"Evolve type {evolve_type} for moment {moment_type} with label {moment_label} unavailable.")
            


def load_lattice_data(moment_type, moment_label, pub_id):
    """
    Load data from a .csv file, extracting 'n' values from the header and associating them with rows. 
    Modify FILE_NAME and FILE_PATH as needed

    Args:
        moment_type (str): The type of moment (e.g., "Isovector").
        moment_label (str): The label of the moment (e.g., "A").
        pub_id (str): The publication ID.

    Returns:
        tuple: A tuple containing the data and a dictionary mapping 'n' values to row indices.
    """
    FILE_NAME = f"{moment_type}Moments{moment_label}{pub_id}.csv"
    FILE_PATH = f"{BASE_PATH}{FILE_NAME}"

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

def Fn0_values(n, moment_type, moment_label, pub_id):
    """
    Return central values for An0 for a given n, moment type, label, and publication ID.
    """
    data, n_to_row_map = load_lattice_data(moment_type, moment_label, pub_id)

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

def Fn0_errors(n, moment_type, moment_label, pub_id):
    """
    Return errors for An0 for a given n, moment type, label, and publication ID.
    """
    data, n_to_row_map = load_lattice_data(moment_type, moment_label, pub_id)

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


##########################
#### Helper functions ####
##########################

def get_alpha_s():
    """
    Returns alpha_s at the input scale of 1 GeV from the MSTW PDF best fit.
    Note that the MSTW best fit obtains alpha_S(mu=1 GeV**2)=0.68183, different from the world average
    """
    index_alpha_s=MSTW_PDF[MSTW_PDF["Parameter"] == "alpha_S(Q0^2)"].index[0]
    alpha_s_in = MSTW_PDF_LO.iloc[index_alpha_s,0][0]
    return alpha_s_in

def check_error_type(error_type):
    if error_type not in ["central","plus","minus"]:
        raise ValueError("error_type must be central, plus or minus")

def check_particle_type(particle):
    if particle not in ["quark", "gluon"]:
        raise ValueError("particle must be quark or gluon")
    
def check_moment_type_label(moment_type, moment_label):
    if (moment_type, moment_label) not in MOMENT_TO_FUNCTION:
        raise ValueError(f"Unsupported moment_type and or label\n (moment_type, moment_label): {moment_type, moment_label} not in {MOMENT_TO_FUNCTION}")

def check_evolve_type(evolve_type):
    if evolve_type not in ["vector","axial"]:
        raise ValueError("evolve_type must be vector or axial.")
    
def check_parity(parity):
    if parity not in ["even", "odd","none"]:
        raise ValueError("Parity must be even, odd or none")

#####################################
### Input for Evolution Equations ###
#####################################
def evolve_alpha_s(mu, Nf = 3):
    """
    Evolve alpha_S=g**/(4pi) from some input scale mu_in to some other scale mu.
    Note that the MSTW best fit obtains alpha_S(mu=1 GeV**2)=0.68183, different from the world average
    
    Arguments:
    mu -- The momentum scale of the process
    
    Returns:
    The evolved value of alpha_s at mu**2
    """
    # Set parameters
    Nc = 3
    mu_R = 1 # 1 GeV
    # Extract value of alpha_S at the renormalization point of mu_R**2 = 1 GeV**2
    alpha_s_in = get_alpha_s()
    beta_0 = 2/3* Nf - 11/3 * Nc

     # Evolve using LO RGE
    log_term = np.log(mu**2 / mu_R**2)
    denominator = 1 - (alpha_s_in / (4 * np.pi)) * beta_0 * log_term
    
    # Debug:
    # print(index_alpha_s)
    # print(alpha_s_in)

    result = alpha_s_in / denominator

    return result

def integral_uv_pdf_regge(j,eta,alpha_p,t, error_type="central"):
    """
    Result of the integral of the Reggeized uv(x) PDF based on the given LO parameters and selected errors.
    
    Arguments:
    j -- conformal spin,
    eta -- skewness (scalar or array)(placeholder for now),
    alpha_p -- Regge slope,
    t -- Mandelstam t (scalar or array),
    error_type -- A string indicating whether to use 'central', 'plus', or 'minus' errors. Default is 'central'.
    
    Returns:
    The value of the Reggeized integral of uv(x) based on the selected parameters and error type.
    """
    # Check type
    check_error_type(error_type)

     # Define a dictionary that maps the error_type to column indices
    error_mapping = {
        "central": 0,  # The column with the central value
        "plus": 1,     # The column with the + error value
        "minus": 2     # The column with the - error value
    }
    
    # Get the column index corresponding to the error_type
    error_col_index = error_mapping.get(error_type, 0)  # Default to 'central' if error_type is invalid

    # Get row index of entry
    index_A_u=MSTW_PDF[MSTW_PDF["Parameter"] == "A_u"].index[0]
    index_eta_1=MSTW_PDF[MSTW_PDF["Parameter"] == "eta_1"].index[0]
    index_eta_2=MSTW_PDF[MSTW_PDF["Parameter"] == "eta_2"].index[0]
    index_epsilon_u=MSTW_PDF[MSTW_PDF["Parameter"] == "epsilon_u"].index[0]
    index_gamma_u=MSTW_PDF[MSTW_PDF["Parameter"] == "gamma_u"].index[0]

    # Extracting parameter values based on the error_type argument
    A_u = MSTW_PDF_LO.iloc[index_A_u,0][0] + int(error_col_index>0)*MSTW_PDF_LO.iloc[index_A_u,0][error_col_index]
    eta_1 = MSTW_PDF_LO.iloc[index_eta_1,0][0] + int(error_col_index>0)*MSTW_PDF_LO.iloc[index_eta_1,0][error_col_index]
    eta_2 = MSTW_PDF_LO.iloc[index_eta_2,0][0] + int(error_col_index>0)*MSTW_PDF_LO.iloc[index_eta_2,0][error_col_index]
    epsilon_u = MSTW_PDF_LO.iloc[index_epsilon_u,0][0] + int(error_col_index>0)*MSTW_PDF_LO.iloc[index_epsilon_u,0][error_col_index]
    gamma_u = MSTW_PDF_LO.iloc[index_gamma_u,0][0] + int(error_col_index>0)*MSTW_PDF_LO.iloc[index_gamma_u,0][error_col_index]
    
    # Convert eta and t to numpy arrays for vectorized operations
    eta = np.atleast_1d(eta)
    t = np.atleast_1d(t)
    
    # If eta and t are 1D, ensure compatibility by broadcasting
    if eta.ndim == 1:
        eta = eta[:, np.newaxis]  # Make eta a column vector
    if t.ndim == 1:
        t = t[np.newaxis, :]      # Make t a row vector

    # Analytical result of the integral
    frac_1 = epsilon_u*gamma(eta_1+j-alpha_p*t -.5)/(gamma(eta_1+eta_2+j-alpha_p*t+.5))
    frac_2 = (eta_1+eta_2-gamma_u+eta_1*gamma_u+j*(1+gamma_u)-(1+gamma_u)*alpha_p*t)*gamma(eta_1+j-alpha_p*t-1)/gamma(1+eta_1+eta_2+j-alpha_p*t)
    result = A_u*gamma(1+eta_2)*(frac_1+frac_2)

    # Return the result while preserving the original dimensions
    if result.size == 1:
        return result.item()  # Return a scalar if the result is a single value
    return result

def integral_dv_pdf_regge(j,eta,alpha_p,t, error_type="central"):
    """
    Result of the integral of the Reggeized dv(x) PDF based on the given LO parameters and selected errors.
    
    Arguments:
    j -- conformal spin,
    eta -- skewness (scalar or array)(placeholder for now),
    alpha_p -- Regge slope,
    t -- Mandelstam t (scalar or array),
    error_type -- A string indicating whether to use 'central', 'plus', or 'minus' errors. Default is 'central'.
    
    Returns:
    The value of the Reggeized integral of dv(x) based on the selected parameters and error type.
    """
    # Check type
    check_error_type(error_type)

    # Define a dictionary that maps the error_type to column indices
    error_mapping = {
        "central": 0,  # The column with the central value
        "plus": 1,     # The column with the + error value
        "minus": 2     # The column with the - error value
    }
    
    # Get the column index corresponding to the error_type
    error_col_index = error_mapping.get(error_type, 0)

    # Get row index of entry
    index_A_d = MSTW_PDF[MSTW_PDF["Parameter"] == "A_d"].index[0]
    index_eta_3 = MSTW_PDF[MSTW_PDF["Parameter"] == "eta_3"].index[0]
    index_eta_2=MSTW_PDF[MSTW_PDF["Parameter"] == "eta_2"].index[0]
    # Only eta_4-eta_2 given
    index_eta_42 = MSTW_PDF[MSTW_PDF["Parameter"] == "eta_4-eta_2"].index[0]
    index_epsilon_d = MSTW_PDF[MSTW_PDF["Parameter"] == "epsilon_d"].index[0]
    index_gamma_d = MSTW_PDF[MSTW_PDF["Parameter"] == "gamma_d"].index[0]

    # Extracting parameter values based on the error_type argument
    A_d = MSTW_PDF_LO.iloc[index_A_d, 0][0] + int(error_col_index>0) * MSTW_PDF_LO.iloc[index_A_d, 0][error_col_index]
    eta_3 = MSTW_PDF_LO.iloc[index_eta_3, 0][0] + int(error_col_index>0) * MSTW_PDF_LO.iloc[index_eta_3, 0][error_col_index]
    # eta_4=(eta_4-eta_2) + eta_2, Add errors in quadrature
    eta_4 = (MSTW_PDF_LO.iloc[index_eta_42, 0][0] + MSTW_PDF_LO.iloc[index_eta_2, 0][0]) + int(error_col_index>0) *np.sign(MSTW_PDF_LO.iloc[index_eta_42, 0][error_col_index]) * np.sqrt(MSTW_PDF_LO.iloc[index_eta_42, 0][error_col_index]**2+MSTW_PDF_LO.iloc[index_eta_2, 0][error_col_index]**2)
    epsilon_d = MSTW_PDF_LO.iloc[index_epsilon_d, 0][0] + int(error_col_index>0) * MSTW_PDF_LO.iloc[index_epsilon_d, 0][error_col_index]
    gamma_d = MSTW_PDF_LO.iloc[index_gamma_d, 0][0] + int(error_col_index>0) * MSTW_PDF_LO.iloc[index_gamma_d, 0][error_col_index]
    
    # Convert eta and t to numpy arrays for vectorized operations
    eta = np.atleast_1d(eta)
    t = np.atleast_1d(t)
    
    # If eta and t are 1D, ensure compatibility by broadcasting
    if eta.ndim == 1:
        eta = eta[:, np.newaxis]  # Make eta a column vector
    if t.ndim == 1:
        t = t[np.newaxis, :]      # Make t a row vector
    # Analytical result of the integral
    frac_1 = epsilon_d*gamma(eta_3+j-alpha_p*t -.5)/(gamma(eta_3+eta_4+j-alpha_p*t+.5))
    frac_2 = (eta_3+eta_4-gamma_d+eta_3*gamma_d+j*(1+gamma_d)-(1+gamma_d)*alpha_p*t)*gamma(eta_3+j-alpha_p*t-1)/gamma(1+eta_3+eta_4+j-alpha_p*t)
    result = A_d*gamma(1+eta_4)*(frac_1+frac_2)
    # Return the result while preserving the original dimensions
    if result.size == 1:
        return result.item()  # Return a scalar if the result is a single value
    return result

def integral_sv_pdf_regge(j,eta,alpha_p,t, error_type="central"):
    """
    Result of the integral of the Reggeized sv(x) PDF based on the given LO parameters and selected errors.
    
    Arguments:
    j -- conformal spin,
    eta -- skewness (scalar or array)(placeholder for now),
    alpha_p -- Regge slope,
    t -- Mandelstam t (scalar or array),
    error_type -- A string indicating whether to use 'central', 'plus', or 'minus' errors. Default is 'central'.
    
    Returns:
    The value of the Reggeized integral of sv(x) based on the selected parameters and error type.
    """
    # Check type
    check_error_type(error_type)

    error_mapping = {
        "central": 0,
        "plus": 1,
        "minus": 2
    }
    
    error_col_index = error_mapping.get(error_type, 0)

    # delta_- fixed to 0.2
    index_A_m = MSTW_PDF[MSTW_PDF["Parameter"] == "A_-"].index[0]
    index_eta_m = MSTW_PDF[MSTW_PDF["Parameter"] == "eta_-"].index[0]
    index_x_0 = MSTW_PDF[MSTW_PDF["Parameter"] == "x_0"].index[0]

    A_m = MSTW_PDF_LO.iloc[index_A_m, 0][0] + int(error_col_index>0) * MSTW_PDF_LO.iloc[index_A_m, 0][error_col_index]
    delta_m = .2
    eta_m = MSTW_PDF_LO.iloc[index_eta_m, 0][0] + int(error_col_index>0) * MSTW_PDF_LO.iloc[index_eta_m, 0][error_col_index]
    x_0 = MSTW_PDF_LO.iloc[index_x_0, 0][0] + int(error_col_index>0) * MSTW_PDF_LO.iloc[index_x_0, 0][error_col_index]
    
    # Convert eta and t to numpy arrays for vectorized operations
    eta = np.atleast_1d(eta)
    t = np.atleast_1d(t)
    
    # If eta and t are 1D, ensure compatibility by broadcasting
    if eta.ndim == 1:
        eta = eta[:, np.newaxis]  # Make eta a column vector
    if t.ndim == 1:
        t = t[np.newaxis, :]      # Make t a row vector

    # Analytical result of the integral
    frac = gamma(1+eta_m)*gamma(j+delta_m-1-alpha_p*t)/(x_0*gamma(1+delta_m+eta_m+j-alpha_p*t))
    result = -A_m*(j-1-delta_m*(x_0-1)-x_0*(eta_m+j-alpha_p*t)-alpha_p*t)*frac
     # Return the result while preserving the original dimensions
    if result.size == 1:
        return result.item()  # Return a scalar if the result is a single value
    return result

def integral_S_pdf_regge(j,eta,alpha_p,t, error_type="central"):
    """
    Result of the integral of the Reggeized Sv(x) PDF based on the given LO parameters and selected errors.
    
    Arguments:
    j -- conformal spin,
    eta -- skewness (scalar or array)(placeholder for now),
    alpha_p -- Regge slope,
    t -- Mandelstam t (scalar or array),
    error_type -- A string indicating whether to use 'central', 'plus', or 'minus' errors. Default is 'central'.
    
    Returns:
    The value of the Reggeized integral of Sv(x) based on the selected parameters and error type.
    """
    # Check type
    check_error_type(error_type)
    
    error_mapping = {
        "central": 0,
        "plus": 1,
        "minus": 2
    }
    
    error_col_index = error_mapping.get(error_type, 0)

    index_A_S = MSTW_PDF[MSTW_PDF["Parameter"] == "A_S"].index[0]
    index_delta_S = MSTW_PDF[MSTW_PDF["Parameter"] == "delta_S"].index[0]
    index_eta_S = MSTW_PDF[MSTW_PDF["Parameter"] == "eta_S"].index[0]
    index_epsilon_S = MSTW_PDF[MSTW_PDF["Parameter"] == "epsilon_S"].index[0]
    index_gamma_S = MSTW_PDF[MSTW_PDF["Parameter"] == "gamma_S"].index[0]

    A_S = MSTW_PDF_LO.iloc[index_A_S, 0][0] + int(error_col_index>0) * MSTW_PDF_LO.iloc[index_A_S, 0][error_col_index]
    delta_S = MSTW_PDF_LO.iloc[index_delta_S, 0][0] + int(error_col_index>0) * MSTW_PDF_LO.iloc[index_delta_S, 0][error_col_index]
    eta_S = MSTW_PDF_LO.iloc[index_eta_S, 0][0] + int(error_col_index>0) * MSTW_PDF_LO.iloc[index_eta_S, 0][error_col_index]
    epsilon_S = MSTW_PDF_LO.iloc[index_epsilon_S, 0][0] + int(error_col_index>0) * MSTW_PDF_LO.iloc[index_epsilon_S, 0][error_col_index]
    gamma_S = MSTW_PDF_LO.iloc[index_gamma_S, 0][0] + int(error_col_index>0) * MSTW_PDF_LO.iloc[index_gamma_S, 0][error_col_index]
    
    # Convert eta and t to numpy arrays for vectorized operations
    eta = np.atleast_1d(eta)
    t = np.atleast_1d(t)
    
    # If eta and t are 1D, ensure compatibility by broadcasting
    if eta.ndim == 1:
        eta = eta[:, np.newaxis]  # Make eta a column vector
    if t.ndim == 1:
        t = t[np.newaxis, :]      # Make t a row vector
    # Analytical result of the integral
    frac_1 = epsilon_S*gamma(delta_S+j-alpha_p*t -.5)/(gamma(delta_S+eta_S+j-alpha_p*t+.5))
    frac_2 = (delta_S+eta_S-gamma_S+delta_S*gamma_S+j*(1+gamma_S)-(1+gamma_S)*alpha_p*t)*gamma(delta_S+j-alpha_p*t-1)/gamma(1+delta_S+eta_S+j-alpha_p*t)
    result = A_S*gamma(1+eta_S)*(frac_1+frac_2)
    # Return the result while preserving the original dimensions
    if result.size == 1:
        return result.item()  # Return a scalar if the result is a single value
    return result

def integral_s_plus_pdf_regge(j,eta,alpha_p,t, error_type="central"):
    """
    Result of the integral of the Reggeized s_+(x) PDF based on the given LO parameters and selected errors.
    
    Arguments:
    j -- conformal spin,
    eta -- skewness (scalar or array)(placeholder for now),
    alpha_p -- Regge slope,
    t -- Mandelstam t (scalar or array),
    error_type -- A string indicating whether to use 'central', 'plus', or 'minus' errors. Default is 'central'.
    
    Returns:
    The value of the Reggeized integral of s_+(x) based on the selected parameters and error type.
    """
    # Check type
    check_error_type(error_type)
    
    error_mapping = {
        "central": 0,
        "plus": 1,
        "minus": 2
    }
    
    error_col_index = error_mapping.get(error_type, 0)

    index_A_p = MSTW_PDF[MSTW_PDF["Parameter"] == "A_+"].index[0]
    index_delta_S = MSTW_PDF[MSTW_PDF["Parameter"] == "delta_S"].index[0]
    index_eta_p = MSTW_PDF[MSTW_PDF["Parameter"] == "eta_+"].index[0]
    index_epsilon_S = MSTW_PDF[MSTW_PDF["Parameter"] == "epsilon_S"].index[0]
    index_gamma_S = MSTW_PDF[MSTW_PDF["Parameter"] == "gamma_S"].index[0]

    A_p = MSTW_PDF_LO.iloc[index_A_p, 0][0] + int(error_col_index>0) * MSTW_PDF_LO.iloc[index_A_p, 0][error_col_index]
    delta_S = MSTW_PDF_LO.iloc[index_delta_S, 0][0] + int(error_col_index>0) * MSTW_PDF_LO.iloc[index_delta_S, 0][error_col_index]
    eta_p = MSTW_PDF_LO.iloc[index_eta_p, 0][0] + int(error_col_index>0) * MSTW_PDF_LO.iloc[index_eta_p, 0][error_col_index]
    epsilon_S = MSTW_PDF_LO.iloc[index_epsilon_S, 0][0] + int(error_col_index>0) * MSTW_PDF_LO.iloc[index_epsilon_S, 0][error_col_index]
    gamma_S = MSTW_PDF_LO.iloc[index_gamma_S, 0][0] + int(error_col_index>0) * MSTW_PDF_LO.iloc[index_gamma_S, 0][error_col_index]

    # Convert eta and t to numpy arrays for vectorized operations
    eta = np.atleast_1d(eta)
    t = np.atleast_1d(t)
    
    # If eta and t are 1D, ensure compatibility by broadcasting
    if eta.ndim == 1:
        eta = eta[:, np.newaxis]  # Make eta a column vector
    if t.ndim == 1:
        t = t[np.newaxis, :]      # Make t a row vector

    # Analytical result of the integral
    frac_1 = epsilon_S*gamma(delta_S+j-alpha_p*t -.5)/(gamma(delta_S+eta_p+j-alpha_p*t+.5))
    frac_2 = (delta_S+eta_p-gamma_S+delta_S*gamma_S+j*(1+gamma_S)-(1+gamma_S)*alpha_p*t)*gamma(delta_S+j-alpha_p*t-1)/gamma(1+delta_S+eta_p+j-alpha_p*t)
    result = A_p*gamma(1+eta_p)*(frac_1+frac_2)
    # Return the result while preserving the original dimensions
    if result.size == 1:
        return result.item()  # Return a scalar if the result is a single value
    return result

def integral_Delta_pdf_regge(j,eta,alpha_p,t, error_type="central"):
    """
    Result of the integral of the Reggeized Delta(x)=ubar(x)-dbar(x) PDF based on the given LO parameters and selected errors.
    
    Arguments:
    j -- conformal spin,
    eta -- skewness (scalar or array)(placeholder for now),
    alpha_p -- Regge slope,
    t -- Mandelstam t (scalar or array),
    error_type -- A string indicating whether to use 'central', 'plus', or 'minus' errors. Default is 'central'.
    
    Returns:
    The value of the Reggeized integral of Delta(x) based on the selected parameters and error type.
    """
    # Check type
    check_error_type(error_type)
    
     # Define a dictionary that maps the error_type to column indices
    error_mapping = {
        "central": 0,  # The column with the central value
        "plus": 1,     # The column with the + error value
        "minus": 2     # The column with the - error value
    }
    
    # Get the column index corresponding to the error_type
    error_col_index = error_mapping.get(error_type, 0)  # Default to 'central' if error_type is invalid

    # Get row index of entry
    index_A_Delta=MSTW_PDF[MSTW_PDF["Parameter"] == "A_Delta"].index[0]
    index_eta_Delta=MSTW_PDF[MSTW_PDF["Parameter"] == "eta_Delta"].index[0]
    index_eta_S=MSTW_PDF[MSTW_PDF["Parameter"] == "eta_S"].index[0]
    index_gamma_Delta=MSTW_PDF[MSTW_PDF["Parameter"] == "gamma_Delta"].index[0]
    index_delta_Delta=MSTW_PDF[MSTW_PDF["Parameter"] == "delta_Delta"].index[0]

    # Extracting parameter values based on the error_type argument
    A_Delta = MSTW_PDF_LO.iloc[index_A_Delta,0][0] + int(error_col_index>0)*MSTW_PDF_LO.iloc[index_A_Delta,0][error_col_index]
    eta_Delta = MSTW_PDF_LO.iloc[index_eta_Delta,0][0] + int(error_col_index>0)*MSTW_PDF_LO.iloc[index_eta_Delta,0][error_col_index]
    eta_S = MSTW_PDF_LO.iloc[index_eta_S,0][0] + int(error_col_index>0)*MSTW_PDF_LO.iloc[index_eta_S,0][error_col_index]
    gamma_Delta = MSTW_PDF_LO.iloc[index_gamma_Delta,0][0] + int(error_col_index>0)*MSTW_PDF_LO.iloc[index_gamma_Delta,0][error_col_index]
    delta_Delta = MSTW_PDF_LO.iloc[index_delta_Delta,0][0] + int(error_col_index>0)*MSTW_PDF_LO.iloc[index_delta_Delta,0][error_col_index]

    # Convert eta and t to numpy arrays for vectorized operations
    eta = np.atleast_1d(eta)
    t = np.atleast_1d(t)
    
    # If eta and t are 1D, ensure compatibility by broadcasting
    if eta.ndim == 1:
        eta = eta[:, np.newaxis]  # Make eta a column vector
    if t.ndim == 1:
        t = t[np.newaxis, :]      # Make t a row vector

    # Analytical result of the integral
    frac_1 = (2+eta_Delta+eta_S+j-alpha_p*t)*(3+eta_Delta+eta_S+j-alpha_p*t)
    frac_2 = gamma(3+eta_S)*gamma(j+eta_Delta-1-alpha_p*t)/(gamma(2+eta_Delta+eta_S+j-alpha_p*t))
    result = A_Delta*(1+((delta_Delta*(eta_Delta+j-alpha_p*t)+gamma_Delta*(3+eta_Delta+eta_S+j-alpha_p*t))*(eta_Delta+j-1+alpha_p*t))/frac_1)*frac_2
    # Return the result while preserving the original dimensions
    if result.size == 1:
        return result.item()  # Return a scalar if the result is a single value
    return result

def integral_gluon_pdf_regge(j,eta,alpha_p,t, error_type="central"):
    """
    Result of the integral of the Reggeized g(x) PDF based on the given LO parameters and selected errors.
    
    Arguments:
    j -- conformal spin,
    eta -- skewness (scalar or array)(placeholder for now),
    alpha_p -- Regge slope,
    t -- Mandelstam t (scalar or array),
    error_type -- A string indicating whether to use 'central', 'plus', or 'minus' errors. Default is 'central'.
    
    Returns:
    The value of the Reggeized integral of g(x) based on the selected parameters and error type.
    """
    # Check type
    check_error_type(error_type)
    
     # Define a dictionary that maps the error_type to column indices
    error_mapping = {
        "central": 0,  # The column with the central value
        "plus": 1,     # The column with the + error value
        "minus": 2     # The column with the - error value
    }
    
    # Get the column index corresponding to the error_type
    error_col_index = error_mapping.get(error_type, 0)  # Default to 'central' if error_type is invalid

    # Get row index of entry
    index_A_g=MSTW_PDF[MSTW_PDF["Parameter"] == "A_g"].index[0]
    index_delta_g=MSTW_PDF[MSTW_PDF["Parameter"] == "delta_g"].index[0]
    index_eta_g=MSTW_PDF[MSTW_PDF["Parameter"] == "eta_g"].index[0]
    index_epsilon_g=MSTW_PDF[MSTW_PDF["Parameter"] == "epsilon_g"].index[0]
    index_gamma_g=MSTW_PDF[MSTW_PDF["Parameter"] == "gamma_g"].index[0]

    # Extracting parameter values based on the error_type argument
    A_g = MSTW_PDF_LO.iloc[index_A_g,0][0] + int(error_col_index>0)*MSTW_PDF_LO.iloc[index_A_g,0][error_col_index]
    delta_g = MSTW_PDF_LO.iloc[index_delta_g,0][0] + int(error_col_index>0)*MSTW_PDF_LO.iloc[index_delta_g,0][error_col_index]
    eta_g = MSTW_PDF_LO.iloc[index_eta_g,0][0] + int(error_col_index>0)*MSTW_PDF_LO.iloc[index_eta_g,0][error_col_index]
    epsilon_g = MSTW_PDF_LO.iloc[index_epsilon_g,0][0] + int(error_col_index>0)*MSTW_PDF_LO.iloc[index_epsilon_g,0][error_col_index]
    gamma_g = MSTW_PDF_LO.iloc[index_gamma_g,0][0] + int(error_col_index>0)*MSTW_PDF_LO.iloc[index_gamma_g,0][error_col_index]

    # Convert eta and t to numpy arrays for vectorized operations
    eta = np.atleast_1d(eta)
    t = np.atleast_1d(t)
    
    # If eta and t are 1D, ensure compatibility by broadcasting
    if eta.ndim == 1:
        eta = eta[:, np.newaxis]  # Make eta a column vector
    if t.ndim == 1:
        t = t[np.newaxis, :]      # Make t a row vector

    # Analytical result of the integral
    frac_1 = epsilon_g*gamma(delta_g+j-alpha_p*t -.5)/(gamma(delta_g+eta_g+j-alpha_p*t+.5))
    frac_2 = (delta_g+eta_g-gamma_g+delta_g*gamma_g+j*(1+gamma_g)-(1+gamma_g)*alpha_p*t)*gamma(delta_g+j-alpha_p*t-1)/gamma(delta_g+eta_g+j-alpha_p*t+1)
    result = A_g*gamma(1+eta_g)*(frac_1+frac_2)
    #2025:
    # frac_1 = epsilon_g*gamma(delta_g+j-alpha_p*t -1.5)/(gamma(delta_g+eta_g+j-alpha_p*t-.5))
    # frac_2 = (-1+delta_g+eta_g-2*gamma_g+delta_g*gamma_g+j*(1+gamma_g)-(1+gamma_g)*alpha_p*t)*gamma(delta_g+j-alpha_p*t-2)/gamma(delta_g+eta_g+j-alpha_p*t)
    # result = A_g*gamma(1+eta_g)*(frac_1+frac_2)
     # Return the result while preserving the original dimensions
    if result.size == 1:
        return result.item()  # Return a scalar if the result is a single value
    return result

# Define Reggeized conformal moments
def non_singlet_isovector_moment(j,eta,t, moment_label="A",evolve_type="vector", error_type="central"):
   # Check type
    check_error_type(error_type)
    check_moment_type_label("NonSingletIsovector",moment_label)
    check_evolve_type(evolve_type)

    alpha_prime = get_regge_slope("NonSingletIsovector",moment_label,evolve_type)

    if moment_label == "A":
       norm, gu, gd = 1,1,1
    elif moment_label =="Atilde":
       norm, gu, gd = 0.603429, 0.843, -0.427

    return norm * (gu * integral_uv_pdf_regge(j,eta,alpha_prime,t,error_type)
           - gd * integral_dv_pdf_regge(j,eta,alpha_prime,t,error_type))

def u_minus_d_pdf_regge(j,eta,t, error_type="central"):
    """ Currently only experimental function that does not set ubar=dbar"""
    # Check type
    check_error_type(error_type)
    # Value optmized for range -t < 5 GeV
    alpha_prime = 0.675606
    # Normalize to 1 at t = 0
    return 1.107*(integral_uv_pdf_regge(j,eta,alpha_prime,t,error_type)
                    -integral_dv_pdf_regge(j,eta,alpha_prime,t,error_type)
                    -integral_Delta_pdf_regge(j,alpha_prime,t,error_type))

def non_singlet_isoscalar_moment(j,eta,t, moment_label="A",evolve_type="vector", error_type="central"):
    # Check type
    check_error_type(error_type)
    check_moment_type_label("NonSingletIsoscalar",moment_label)
    check_evolve_type(evolve_type)

    alpha_prime = get_regge_slope("NonSingletIsoscalar",moment_label,evolve_type)

    if moment_label == "A":
       norm, gu, gd = 1,1,1
    elif moment_label =="Atilde":
       norm, gu, gd = 0.331774, 0.843, -0.427

    return norm * (gu * integral_uv_pdf_regge(j,eta,alpha_prime,t,error_type)
            + gd * integral_dv_pdf_regge(j,eta,alpha_prime,t,error_type))

def u_plus_d_pdf_regge(j,eta,t, error_type="central"):
    """ Currently only experimental function that does not set ubar=dbar"""
    # Check type
    check_error_type(error_type)
    # Value optmized for range -t < 5 GeV
    alpha_prime = 0.949256
    # Normalize to 1 at t = 0
    return 0.973*(integral_uv_pdf_regge(j,eta,alpha_prime,t,error_type)
                    +integral_dv_pdf_regge(j,eta,alpha_prime,t,error_type)
                    +integral_Delta_pdf_regge(j,alpha_prime,t,error_type))

def d_hat(j,eta,t):
    """
    Compute the skewness dependent kinematical factor for Reggeized
    spin-j exchanges.
    Parameters:
    - j (complex): conformal spin 
    - eta (float): skewness 
    - t (float < 0): Mandelstam t 

    Returns:
    The value of d_hat (= 1 for eta == 0)
    """
    m_N = 0.93827 # Nucleon mass in GeV
    if eta == 0:
        result = 1
    else :
        result = mp.hyp2f1(-j/2, -(j-1)/2, 1/2 - j, - 4 * m_N**2/t * eta**2)
    return result
    
def quark_singlet_regge_A(j,eta,t, Nf=3, alpha_prime_ud=0.891, error_type="central"):
    # Check type
    check_error_type(error_type)
    uv = integral_uv_pdf_regge(j,eta,alpha_prime_ud,t,error_type) 
    dv = integral_dv_pdf_regge(j,eta,alpha_prime_ud,t,error_type)
    Delta = integral_Delta_pdf_regge(j,eta,alpha_prime_ud,t,error_type)
    Sv = integral_S_pdf_regge(j,eta,alpha_prime_ud,t,error_type)
    s_plus = integral_s_plus_pdf_regge(j,eta,alpha_prime_ud,t,error_type)

    if Nf == 3 or Nf == 4:
        result = uv + dv + Sv 
    elif Nf == 2:
        result = uv + dv + Sv - s_plus
    elif Nf == 1:
        result = .5*(Sv-s_plus+2*uv-2*Delta)
    else :
        raise ValueError("Currently only (integer) 1 <= Nf <= 3 supported")
    return result
    
def quark_singlet_regge_D(j,eta,t, Nf=3, alpha_prime_ud=0.891,alpha_prime_s=1.828, error_type="central"):
    # Check type
    check_error_type(error_type)
    uv = integral_uv_pdf_regge(j,eta,alpha_prime_ud,t,error_type) 
    dv = integral_dv_pdf_regge(j,eta,alpha_prime_ud,t,error_type)
    Delta = integral_Delta_pdf_regge(j,eta,alpha_prime_ud,t,error_type)
    Sv = integral_S_pdf_regge(j,eta,alpha_prime_ud,t,error_type)
    s_plus = integral_s_plus_pdf_regge(j,eta,alpha_prime_ud,t,error_type)

    uv_s = integral_uv_pdf_regge(j,eta,alpha_prime_s,t,error_type) 
    dv_s = integral_dv_pdf_regge(j,eta,alpha_prime_s,t,error_type)
    Sv_s = integral_S_pdf_regge(j,eta,alpha_prime_s,t,error_type)
    s_plus_s = integral_s_plus_pdf_regge(j,eta,alpha_prime_s,t,error_type)
    Delta_s = integral_Delta_pdf_regge(j,eta,alpha_prime_s,t,error_type)

    if eta == 0:
        return 0

    if Nf == 3 or Nf == 4:
        term_1 = uv + dv + Sv 
        term_2 = uv_s + dv_s + Sv_s 
    elif Nf == 2:
        term_1 = uv + dv + Sv - s_plus
        term_2 = uv_s + dv_s + Sv_s - s_plus_s
    elif Nf == 1:
        term_1 = .5*(Sv-s_plus+2*uv-2*Delta)
        term_2 = .5*(Sv_s-s_plus_s+2*uv_s-2*Delta_s)
    else :
        raise ValueError("Currently only (integer) 1 <= Nf <= 3 supported")
    
    result = (d_hat(j,eta,t)-1)*(term_1-term_2)
    return result

def quark_singlet_regge(j,eta,t,Nf=3,moment_label="A",evolve_type="vector",error_type="central"):
    # Check type
    check_error_type(error_type)
    check_evolve_type(evolve_type)
    check_moment_type_label("Singlet",moment_label)
    # alpha_prime_ud = 0.891
    # alpha_prime_s = 1.828
    alpha_prime_ud = get_regge_slope("NonSingletIsoscalar",moment_label,evolve_type)
    alpha_prime_s, _, _ = get_regge_slope("Singlet",moment_label,evolve_type)

    term_1 = quark_singlet_regge_A(j,eta,t,Nf,alpha_prime_ud,error_type)
    term_2 = quark_singlet_regge_D(j,eta,t,Nf,alpha_prime_ud,alpha_prime_s,error_type)
    result = term_1 + term_2
    return result

def gluon_regge_A(j,eta,t, alpha_prime_T = 0.627, error_type="central"):
    # Check type
    check_error_type(error_type)
    return integral_gluon_pdf_regge(j,eta,alpha_prime_T,t,error_type)

def gluon_regge_D(j,eta,t, alpha_prime_T = 0.627, alpha_prime_S = 4.277, error_type="central"):
    # Check type
    check_error_type(error_type)
    if eta == 0:
        return 0
    else :
        term_1 = (d_hat(j,eta,t)-1)
        term_2 = gluon_regge_A(j,eta,t,alpha_prime_T,error_type)
        term_3 = integral_gluon_pdf_regge(j,eta,t,alpha_prime_S,error_type)
        result =term_1 * (term_2-term_3)
        return result

def gluon_singlet_regge(j,eta,t,moment_label="A",evolve_type="vector", error_type="central"):
    # Check type
    check_error_type(error_type)
    check_evolve_type(evolve_type)
    check_moment_type_label("Singlet",moment_label)
    # alpha_prime_T = 0.627
    # alpha_prime_S = 4.277
    _, alpha_prime_T, alpha_prime_S = get_regge_slope("Singlet",moment_label,evolve_type)
    term_1= gluon_regge_A(j,eta,t,alpha_prime_T,error_type)
    if eta == 0:
        result = term_1
    else :
        term_2 = gluon_regge_D(j,eta,t,alpha_prime_T,alpha_prime_S,error_type)
        result = term_1 + term_2
    return result

def singlet_moment(j,eta,t,Nf=3,moment_label="A",evolve_type="vector",solution="+",error_type="central"):
    # Check type
    check_error_type(error_type)
    check_evolve_type(evolve_type)
    # Switch sign
    if solution == "+":
        solution = "-"
    elif solution == "-":
        solution = "+"
    else:
        raise ValueError("Invalid solution type. Use '+' or '-'.")
    
    quark_in = quark_singlet_regge(j,eta,t,Nf,moment_label,evolve_type,error_type)
    # Note: j/6 already included in gamma_qg and gamma_gg definitions
    gluon_prf = (gamma_qg(j-1,Nf,evolve_type)/
                    (gamma_qq(j-1)-gamma_pm(j-1,Nf,evolve_type,solution)))
    gluon_in = gluon_singlet_regge(j,eta,t,moment_label,evolve_type,error_type)
    result = quark_in + gluon_prf * gluon_in
    return result

# Initialize the MOMENT_TO_FUNCTION dictionary
# after all functions are defined
initialize_moment_to_function()

################################
##### Evolution Equations ######
################################

def gamma_qq(j):
   """
   Return conformal anomalous dimension for conformal spin-j

   Arguments:
   j -- conformal spin
   """

   Nc = 3
   Cf = (Nc**2-1)/(2*Nc)
   result = - Cf * (-4*digamma(j+2)+4*digamma(1)+2/((j+1)*(j+2))+3)

   return result

def gamma_qg(j, Nf=3, evolve_type = "vector"):
    """
    Compute off-diagonal qg anomalous dimension
    Parameters:
    j -- conformal spin
    Nf -- Number of active flavors (default Nf = 3 )
    evolve_type -- "vector" or "axial"
    Returns:
    Value of anomalous dimension
    """
    # Note addition factor of j/6 (see (K.1) in 0504030)
    Tf = 1/2
    if evolve_type == "vector":
        result = -24*Nf*Tf*(j**2+3*j+4)/(j*(j+1)*(j+2)*(j+3))*j/6
    elif evolve_type == "axial":
        result = -24*Nf*Tf/((j+1)*(j+2))*j/6
    else:
        raise ValueError("Type must be axial or vector")
    return result

def gamma_gq(j, evolve_type = "vector"):
    """
    Compute off-diagonal gq anomalous dimension
    Parameters:
    j -- conformal spin
    evolve_type -- "vector" or "axial"
    Returns:
    Value of anomalous dimension
    """
    # Check evolve_type
    check_evolve_type(evolve_type)

    Nc = 3
    Cf = (Nc**2-1)/(2*Nc)
    if evolve_type == "vector":
        result = -Cf*(j**2+3*j+4)/(3*(j+1)*(j+2))*6/j
    elif evolve_type == "axial":
        result = -Cf*j*(j+3)/(3*(j+1)*(j+2))*6/j
    else:
        raise ValueError("Type must be axial or vector")
    return result

def gamma_gg(j, Nf = 3, evolve_type = "vector"):
    """
    Compute diagonal gg anomalous dimension
    Parameters:
    j -- conformal spin
    evolve_type -- "vector" or "axial"
    Nf -- Number of active flavors (default Nf = 3 )
    Returns:
    Value of anomalous dimension
    """
    # Check evolve_type
    check_evolve_type(evolve_type)

    Nc = 3
    Ca = Nc
    beta_0 = 2/3* Nf - 11/3 * Nc
    if evolve_type == "vector":
        result = -Ca*(-4*digamma(j+2)+4*digamma(1)+8*(j**2+3*j+3)/(j*(j+1)*(j+2)*(j+3))-beta_0/Ca)
    elif evolve_type == "axial":
        result = -Ca*(-4*digamma(j+2)+4*digamma(1)+8/((j+1)*(j+2))-beta_0/Ca)
    else:
        raise ValueError("Type must be axial or vector")
    return result

def gamma_pm(j, Nf = 3, evolve_type = "vector",solution="+"):
    """ Compute the (+) and (-) eigenvalues of the LO evolution equation of the coupled singlet quark and gluon GPD
    Arguments:
    j -- conformal spin,
    evolve_type -- "vector" or "axial"
    Nf -- Number of active flavors (default Nf = 3 )
    Returns:
    The eigenvalues (+) and (-) in terms of an array
    """
    # Check evolve_type
    check_evolve_type(evolve_type)
    
    base = gamma_qq(j)+gamma_gg(j,Nf,evolve_type)
    root = np.sqrt((gamma_qq(j)-gamma_gg(j,Nf,evolve_type))**2+4*gamma_gq(j,evolve_type)*gamma_qg(j,Nf,evolve_type))
    if solution == "+":
        return (base + root)/2
    elif solution == "-":
        return (base - root)/2
    else:
        raise ValueError("Invalid solution evolve_type. Use '+' or '-'.")

def Gamma_pm(j,Nf=3,evolve_type="vector",solution="+"):
    """ Returns the fraction 1-gamma_pm/gamma_q

    Parameters:
    - j (float): conformal spin
    - Nf (int, optional): number ofactive flavors. Default is 3.
    - evolve_type (str. optional): Either vector or axial evolution
    - solution (str. optiona): Positive (+) or negative (-) eigenvalue of gamma_pm
    """
    # Check evolve_type
    check_evolve_type(evolve_type)
    
    if solution not in ["+","-"]:
        raise ValueError("Solution must be '+' or '-' ")

    result = 1-gamma_pm(j,Nf,evolve_type,solution)/gamma_qq(j)
    return result

def evolve_conformal_moment(j,eta,t,mu,Nf = 3,particle="quark",moment_type="NonSingletIsovector",moment_label ="A", error_type = "central"):
    """
    Evolve the conformal moment F_{j}^{+-} from some input scale mu_in to some other scale mu.
    Note that the MSTW best fit obtains alpha_S(mu=1 GeV**2)=0.68183, different from the world average

    Arguments:
    j -- conformal spin
    eta -- skewness
    t -- Mandelstam t
    mu -- Resolution scale
    mu -- The momentum scale of the process
    Nf -- Number of active flavors (default Nf = 3)
    moment_type -- NonSingletIsovector, NonSingletIsoscalar, or Singlet
    moment_label -- A(Tilde) B(Tilde) depending on H(Tilde) or E(Tilde) GPD
    error_type -- Choose central, upper or lower value for input PDF parameters
    Returns:
    The value of the evolved conformal moment at scale mu
    """
    check_particle_type(particle)
    check_moment_type_label(moment_type,moment_label)
    check_error_type(error_type)
    if particle == "gluon" and moment_type != "Singlet":
        raise ValueError("Gluon is only Singlet")
    
    # Set parameters
    Nc = 3
    beta_0 = 2/3* Nf - 11/3 * Nc

    # Extract value of alpha_S at the renormalization point of mu_R**2 = 1 GeV**2
    alpha_s_in = get_alpha_s()

    # Precompute alpha_s fraction:
    alpha_frac  = (alpha_s_in/evolve_alpha_s(mu,Nf))    
    gpd_in, evolve_type = MOMENT_TO_FUNCTION.get((moment_type, moment_label))

    if moment_type == "Singlet":
        anomalous_dim_p = gamma_pm(j-1,Nf,evolve_type,"+")
        anomalous_dim_m = gamma_pm(j-1,Nf,evolve_type,"-")
        gpd_in_p = gpd_in(j,eta,t, Nf, moment_label, evolve_type,"+",error_type)
        gpd_in_m = gpd_in(j,eta,t, Nf, moment_label,evolve_type,"-",error_type)
        evolve_moment_p = gpd_in_p * alpha_frac**(anomalous_dim_p/beta_0)
        evolve_moment_m = gpd_in_m *alpha_frac**(anomalous_dim_m/beta_0)
        if particle == "quark":
            # Manually fix the scale to 0.51 @ mu = 2 GeV from 2310.08484
            A0 = 0.51/0.5618
            term_1 = gamma_qq(j-1)/(gamma_pm(j-1,Nf,evolve_type,"+")-gamma_pm(j-1,Nf,evolve_type,"-"))
            term_2 = Gamma_pm(j-1,Nf,evolve_type,"-") * evolve_moment_p
            term_3 = Gamma_pm(j-1,Nf,evolve_type,"+") * evolve_moment_m
        if particle == "gluon":
            # Manually fix the scale to 0.501 @ mu = 2 GeV from 2310.08484
            A0 = 0.501/0.43807
            term_1 = gamma_gq(j-1,evolve_type)/(gamma_pm(j-1,Nf,evolve_type,"+")-gamma_pm(j-1,Nf,evolve_type,"-"))
            term_2 = evolve_moment_p
            term_3 = evolve_moment_m
        result = A0*term_1*(term_2-term_3)

    elif moment_type in ["NonSingletIsovector","NonSingletIsoscalar"]:
        anomalous_dim = gamma_qq(j-1)
        result = gpd_in(j,eta,t,moment_label,evolve_type,error_type) * alpha_frac**(anomalous_dim/beta_0)   
    else : 
        raise ValueError("Moment Type must be Singlet, NonSingletIsovector or NonSingletIsoscalar")
    
    return result

def evolve_singlet_D(j,eta,t,mu,Nf=3,particle="quark",moment_label="A",error_type="central"):
    check_particle_type(particle)
    check_moment_type_label("Singlet",moment_label)
    if particle == "quark":
        # Manually fix the scale to 1.3 @ mu = 2 GeV from 2310.08484
        D0 = 1.3/1.0979
    else :
        # Manually fix the scale from holography (II.9) in 2204.08857
        D0 = 2.57/3.0439
        #2025
        #D0 = 1

    eta = 1 # Result is eta independent 
    term_1 = evolve_conformal_moment(j,eta,t,mu,Nf,particle,"Singlet",moment_label,error_type)
    term_2 = evolve_conformal_moment(j,0,t,mu,Nf,particle,"Singlet",moment_label,error_type)
    result = D0 * (term_1-term_2)/eta**2
    return result

def evolve_quark_non_singlet(j,eta,t,mu,Nf=3,moment_type="NonSingletIsovector",moment_label = "A",error_type="central"):
    result = evolve_conformal_moment(j,eta,t,mu,Nf,"quark",moment_type,moment_label,error_type)
    return result

def evolve_quark_singlet(j,eta,t,mu,Nf=3,moment_label = "A",error_type="central"):
    result = evolve_conformal_moment(j,eta,t,mu,Nf,"quark","Singlet",moment_label,error_type)
    return result

def evolve_gluon_singlet(j,eta,t,mu,Nf=3,moment_label = "A",error_type="central"):
    result = evolve_conformal_moment(j,eta,t,mu,Nf,"gluon","Singlet",moment_label,error_type)
    return result

def evolve_quark_singlet_D(eta,t,mu,Nf=3,moment_label = "A",error_type="central"):
    result = evolve_singlet_D(eta,t,mu,Nf,"quark",moment_label,error_type)
    return result

def evolve_gluon_singlet_D(j,eta,t,mu,Nf=3,moment_label = "A",error_type="central"):
    result = evolve_singlet_D(eta,t,mu,Nf,"gluon",moment_label,error_type)
    return result

def fourier_transform_moment(j,eta,mu,b_vec,Nf=3,particle="quark",moment_type="NonSingletIsovector", moment_label="A", Delta_max = 5,num_points=100, error_type="central"):
    """
    Optimized calculation of Fourier transformed moments using adaptive integration.

    Parameters:
    - j (float): Conformal spin
    - eta (float): Skewness parameter
    - mu (float): RGE scale
    - b_vec: (b_x, b_y), the vector for which to compute the result
    - Nf (int, optional): Number of active flavors. Default is 3.
    - particle (str. optional): "quark" or "gluon". Default is quark.
    - moment_type (str. optional): Singlet, NonSingletIsovector or NonSingletIsoscalar. Default is NonSingletIsovector.
    - moment_label (str. optiona): Label of conformal moment, e.g. A
    - Delta_max (float, optional): maximum radius for the integration domain (limits the integration bounds)
    - num_points: number of points for discretizing the domain (adapt as needed)
    - error_type (str. optional): Whether to use central, plus or minus value of input PDF. Default is central.

    Returns:
    - The value of the Fourier transformed moment at (b_vec)
    """
    check_error_type(error_type)
    check_particle_type(particle)
    check_moment_type_label(moment_type,moment_label)
    b_x, b_y = b_vec
    # Limits of integration for Delta_x, Delta_y on a square grid
    x_min, x_max = -Delta_max, Delta_max
    y_min, y_max = -Delta_max, Delta_max
    # Discretize the grid (vectorized)
    Delta_x_vals = np.linspace(x_min, x_max, num_points)
    Delta_y_vals = np.linspace(y_min, y_max, num_points)

    # Create a meshgrid for delta_x, delta_y
    Delta_x_grid, Delta_y_grid = np.meshgrid(Delta_x_vals, Delta_y_vals)

    def integrand(Delta_x,Delta_y,b_x,b_y):
        t = -(Delta_x**2+Delta_y**2)
        exponent = -1j * (b_x * Delta_x + b_y * Delta_y)
        result = evolve_conformal_moment(j,eta,t,mu,Nf,particle,moment_type,moment_label,error_type)*np.exp(exponent)/(2*np.pi**2)
        return result
    
    # Compute the integrand for each pair of (Delta_x, Delta_y) values
    integrand_values = integrand(Delta_x_grid, Delta_y_grid, b_x, b_y)
    # Perform the numerical integration using the trapezoidal rule for efficiency
    integral_result = np.real(trapezoid(trapezoid(integrand_values, Delta_x_vals, axis=0), Delta_y_vals))

    return integral_result
    


################################
#### Mellin-Barnes Integral ####
################################


# Define conformal partial waves
def conformal_partial_wave(j, x, eta, particle = "quark", parity="none"):
    """
    Calculate the conformal partial waves for quark and gluon GPDs and generate their
    respective "even" or "odd" combinations.

    Parameters:
    j (complex): Conformal spin
    x (float): Value of parton x
    eta (float): Value of skewness
    particle (str, optional): The particle species 'quark' or 'gluon' default is 'quark'.
    parity (str, optional): The parity of the function. Either 'even', 'odd' or 'none'. Default is 'none'.

    Returns:
    mpc: Value of even or odd combination of conformal quark partial waves

    Raises:
    ValueError: If the `parity` argument is not "even", "odd" or "none".

    Notes:
    - The result is vectorized later on using np.vectorize for handling array inputs.
    """
    check_particle_type(particle)
    check_parity(parity)
    if parity not in ["even", "odd","none"]:
        raise ValueError("Parity must be even, odd or none")
    
    # Precompute factors that do not change
    if particle == "quark":
        gamma_term = lambda j: 2.0**j * gamma(1.5 + j) / (gamma(0.5) * gamma(j))
        sin_term = lambda j: mp.sin(np.pi * j) / np.pi
        eta_prf = 1 / eta**j 
        def cal_P(x):
            arg = (1 + x / eta)
            hyp = mp.hyp2f1(-j, j + 1, 2, 0.5 * arg)
            result = eta_prf * arg * hyp * gamma_term(j)
            return result
        def cal_Q(x): 
            hyp = mp.hyp2f1(0.5 * j, 0.5 * (j + 1), 1.5 + j, (eta / x)**2) 
            result = 1 / x**j * hyp * sin_term(j)
            return result
    else:   
        gamma_term = lambda j: 2.0**(j-1) * gamma(1.5 + j) / (gamma(0.5) * gamma(j-1))
        sin_term =lambda j: mp.sin(np.pi * (j+1))  / np.pi 
        eta_prf = 1 / eta**(j-1)
        def cal_P(x):
            arg = (1. + x / eta)
            hyp = mp.hyp2f1(-j, j + 1, 3, 0.5 * arg)
            result = eta_prf * arg**2 * hyp * gamma_term(j)
            return result
        def cal_Q(x): 
            hyp = mp.hyp2f1(0.5 * (j-1), 0.5 * j, 1.5 + j, (eta / x)**2) 
            result = 1 / x**(j-1) * hyp * sin_term(j)
            return result

    def p_j(x):
        # Initialize P_term and Q_term with zero
        P_term = 0
        Q_term = 0        
        if eta - np.abs(x) >= 0 :  # If condition for cal_P is satisfied
            # Note continuity at x = eta gives cal_P = cal_Q
            P_term =  cal_P(x)
        elif x - eta > 0 :
            Q_term = cal_Q(x)
        return P_term + Q_term
    
    if parity == "even":    # Even parity
        result = p_j(x) + p_j(-x)
    elif parity == "odd":   # Odd parity
        result = p_j(x) - p_j(-x)
    else :
        result = p_j(x)     # No parity for non_singlet

    #result = complex(result.real, result.imag)
    return result


# Define get_j_base which contains real part of integration variable
def get_j_base(particle="quark",parity="none"):
    check_particle_type(particle)
    check_parity(parity)

    if particle == "quark":
        if parity == "even":
            print(f"j_base_q of parity {parity} is tbd")
            j_base = 1.1
        elif parity == "odd":
            j_base = 2.7
        elif parity == "none":
            j_base = .95
        return j_base 
    elif particle == "gluon":
        if parity == "even":
            j_base = 3
        elif parity == "odd":
            print(f"j_base_g of parity {parity} is tbd")
            j_base = 2
        elif parity == "none":
            print(f"j_base_g of parity {parity} is tbd")
            j_base = 2
        else :
            raise ValueError("Parity must be even, odd or none")
        return j_base



def mellin_barnes_gpd(x, eta, t, mu, Nf=3, particle = "quark", moment_type="Singlet",moment_label="A", error_type="central",real_imag ="real",j_max = 15, n_jobs=-1):
    """
    Numerically evaluate the Mellin-Barnes integral parallel to the imaginary axis to obtain the corresponding GPD
    
    Parameters:
    - x (float): Parton x
    - eta (float): Skewness.
    - t (float): Mandelstam t
    - mu (float): Resolution scale
    - Nf (int 1<= Nf <=3 ): Number of flavors
    - particle (str): particle species (quark or gluon)
    - moment_type (str): Singlet, NonSingletIsovector, NonSingletIsoscalar
    - moment_label (str): A, Atilde, B
    - error_type (str): value of input PDFs (central, plus, minus)
    - real_imag (str): Choose to compute real part, imaginary part or both
    - j_max (float): Integration range parallel to the imaginary axis
    - n_jobs (int): Number of subregions, and thus processes, the integral is split into
    - n_k (int): Number of sampling points within the interval [-j_max,j_max]
    
    Returns: 
    - The value of the Mellin-Barnes integral with real and imaginary part.
    Note:
    - For low x and/or eta it is recommended to divide the integration region
    """
    check_particle_type(particle)
    check_error_type(error_type)

    check_moment_type_label(moment_type,moment_label)
    
    if moment_type == "Singlet":
        if particle == "quark":
            # Scale fixed by it's value at the input scale:
            # print(uv_PDF(1e-3)+dv_PDF(1e-3)+Sv_PDF(1e-3)) 
            norm = 1.78160932e+03/ 1.61636674e+03
            parity = "odd"
        elif particle == "gluon":
            # Scale fixed by it's value at the input scale:
            # print(.1 gluon_PDF(.1))
            norm = 0.86852857/9.93131764e-01
            parity = "even"
    elif moment_type == "NonSingletIsovector":
        # Scale fixed by it's value at the input scale:
        # print(uv_minus_dv_PDF(1e-4))
        norm = 152.92491544/153.88744991730528
        parity = "none"
    elif moment_type == "NonSingletIsoscalar":
        # To Do
        print("NonSingletIsoscalar norm is To Do")
        norm = 1
        parity = "none"

    check_parity(parity)
    j_base = get_j_base(particle,parity)

    if eta == 0: # Avoid division by zero in the Q partial wave
        eta = 1e-6
    # Integrand function which returns both real and imaginary parts
    def integrand(k, real_imag):
        """
        Calculates the integrand for the given k value and returns either 
        the real, imaginary part, or both.

        Args:
            k: The independent variable for integration.
            real_imag: A string specifying whether to return 'real', 'imag', or 'both'.

        Returns:
            If real_imag is 'real':
                The real part of the integrand.
            If real_imag is 'imag':
                The imaginary part of the integrand.
            If real_imag is 'both':
                A tuple containing the real and imaginary parts of the integrand.
        """
        z = j_base + 1j * k 
        dz = 1j
        sin_term = mp.sin(np.pi * z)
        pw_val = conformal_partial_wave(z, x, eta, particle, parity)

        if particle == "quark":
            if moment_type == "Singlet":
                mom_val = evolve_quark_singlet(z,eta,t,mu,Nf,moment_label,error_type)
            else:
                mom_val = evolve_quark_non_singlet(z,eta,t,mu,Nf,moment_type,moment_label,error_type)
        else:
            # (-1) from shift in Sommerfeld-Watson transform
            mom_val = (-1) * evolve_gluon_singlet(z,eta,t,mu,Nf, moment_label,error_type)
        result = -.5j * dz * pw_val * mom_val / sin_term

        if real_imag == 'real':
            return result.real
        elif real_imag == 'imag':
            return result.imag
        elif real_imag == 'both':
            return result.real, result.imag
        else:
            raise ValueError("real_imag must be either 'real', 'imag', or 'both'")

    def find_integration_bound(integrand, j_max, tolerance=1e-2, step_size=10, max_iterations=50):
        """
        Finds an appropriate upper integration bound for an oscillating integrand.

        Parameters:
        - integrand (function): The function to be integrated.
        - tolerance (str. optional): The desired tolerance for the integrand's absolute value. Standard is 1e-2
        - step_size (int. optional): The increment to increase the integration bound in each step. Standard is 10
        - max_iterations (int. optional): The maximum number of iterations to perform. Standard is 50

        Returns:
            The determined upper integration bound.

        Raises:
            ValueError: If the maximum number of iterations is reached without finding a suitable bound.
        """
        iterations = 0

        while abs(integrand(j_max, "real")) > tolerance and iterations < max_iterations:
            j_max += step_size
            iterations += 1

        if iterations == max_iterations:
            raise ValueError("Maximum number of iterations reached without finding a suitable bound. Increase initial value of j_max")

        # Check for rapid oscillations
        if abs(integrand(j_max,  "real") - integrand(j_max + 2,  "real")) > tolerance:
            while abs(integrand(j_max,  "real")) > tolerance and iterations < max_iterations:
                j_max += step_size
                iterations += 1

            if iterations == max_iterations:
                raise ValueError("Maximum number of iterations reached without finding a suitable bound. Increase initial value of j_max")
        #print(iterations,integrand(j_max,real_imag))
        return j_max

    # Function to integrate over a subinterval of k 
    def integrate_subinterval(k_values, real_imag):
        """
        Integrates the integrand over the specified subinterval and 
        returns either the real, imaginary part, or both.

        Parameters:
        - k_values (arr.): A list or array containing the minimum and maximum k values.
        - real_imag (str.): A string specifying whether to return 'real', 'imag', or 'both'.

        Returns:
            If real_imag is 'real':
                A tuple containing the real part of the integral and its error.
            If real_imag is 'imag':
                A tuple containing the imaginary part of the integral and its error.
            If real_imag is 'both':
                A tuple containing the real part, its error, the imaginary part, 
                and its error.
        """
        k_min = k_values[0]
        k_max = k_values[-1]

        if real_imag == 'real':
            integral, error = quad(lambda k: integrand(k, 'real'), k_min, k_max, limit = 200)
            # Use symmetry of the real part of the integrand
            integral *= 2
            error *= 2
            return integral, error
        elif real_imag == 'imag':
            integral, error = quad(lambda k: integrand(k, 'imag'), k_min, k_max, limit = 200)
            return integral, error
        elif real_imag == 'both':
            real_integral, real_error = quad(lambda k: integrand(k, 'real'), k_min, k_max, limit = 200)
            imag_integral, imag_error = quad(lambda k: integrand(k, 'imag'), k_min, k_max, limit = 200)
            return real_integral, real_error, imag_integral, imag_error
        else:
            raise ValueError("real_imag must be either 'real', 'imag', or 'both'") 

    # Dynamically determine integration bound
    j_max = find_integration_bound(integrand, j_max) 

    # Define the number of subintervals (equal to n_jobs)
    n_subintervals = n_jobs if n_jobs > 0 else os.cpu_count()  # Default to all cores if n_jobs isn't specified
    # Generate an array that when split into n_subintervals contains k_min and k_max
    if real_imag == "real":
        k_range = np.linspace(0, j_max, n_subintervals + 1) # real part is symmetric
    else:
        k_range = np.linspace(-j_max, j_max, n_subintervals + 1) 

    k_values_split = []
    # Split the range into equal subintervals
    for i in range(n_subintervals):
        k_values_split.append(k_range[i:i+2])
    # Debug
    # print(j_max)
    # print(j_base)
    # # Print the subintervals to verify
    # print(k_range)
    # for i in range(len(k_values_split)):
    #     print(k_values_split[i])
    # plot_integrand(x,eta,t,mu,Nf,particle,moment_type,moment_label,error_type,j_max)    
    #print(particle,parity,moment_type,norm)


    # Parallelize the integration over the subintervals of k
    results = Parallel(n_jobs=n_subintervals)(
        delayed(integrate_subinterval)(k_values_split[i],real_imag) for i in range(n_subintervals)
    )

    if real_imag == "both":
        real_integral = sum(result[0] for result in results)
        real_error = np.sqrt(sum(result[1]**2 for result in results))
        imag_integral = sum(result[2] for result in results)
        imag_error = np.sqrt(sum(result[3]**2 for result in results))
        integral = real_integral +1j * imag_integral
        error = real_error + 1j * imag_error
    else :
    # Sum the results from all subintervals for real and imaginary parts, and accumulate the errors
        integral = sum(result[0] for result in results)
        error = np.sqrt(sum(result[1]**2 for result in results))
        
    # Check for the estimated error
    if np.abs(error) > 1e-3:
        print(f"Warning: Large error estimate: {error}")
    return norm * integral

# Check normalizations:
# vectorized_mellin_barnes_gpd=np.vectorize(mellin_barnes_gpd)
# x_Array = np.array([1e-2, 0.1, 0.3, 0.5, 0.7, 1]) 

# print(uv_minus_dv_PDF(x_Array)) 
# print(vectorized_mellin_barnes_gpd(x_Array,1e-5, -1e-4, 1, moment_type="NonSingletIsovector", moment_label="A", j_max=100, n_jobs =-1 ))

# print(uv_plus_dv_plus_S_PDF(x_Array))
# print(vectorized_mellin_barnes_gpd(x_Array, 1e-5, -1e-4, 1, particle="quark", moment_type="Singlet", moment_label="A", j_max=100, n_jobs = -1 ))

# print(x_Array*gluon_PDF(x_Array))
# print(vectorized_mellin_barnes_gpd(x_Array, 1e-3, -1e-4, 1, particle="gluon", moment_type="Singlet", moment_label="A", j_max=100, n_jobs = -1 ))
# del x_Array




################################
####### Plot functions #########
################################


def plot_moment(n,eta,y_label,mu_in=2,t_max=3,Nf=3,particle="quark",moment_type="NonSingletIsovector", moment_label="A", n_t=50):
    """
    Generates plots of lattice data and RGE-evolved functions for a given moment type and label. Unless there is a different scale
    defined in PUBLICATION_MAPPING, the default is mu = 2 GeV.
    
    Parameters:
    - n (int): conformal spin
    - eta (float): skewness parameter
    - y_label (str.): the label of the y axis
    - mu_in (float, optional): The resolution scale. (default is 2 GeV).
    - t_max (float, optional): Maximum t value for the x-axis (default is 3).
    - Nf (int, optional): Number of active flavors (default is 3)
    - particle (str., optional): Either quark or gluon
    - moment_type (str): The type of moment (e.g., "NonSingletIsovector").
    - moment_label (str): The label of the moment (e.g., "A").
    - n_t (int, optional): Number of points for t_fine (default is 50).
    - num_columns (int, optional): Number of columns for the grid layout (default is 3).
    """
    check_particle_type(particle)
    check_moment_type_label(moment_type,moment_label)
    # Accessor functions for -t, values, and errors
    def t_values(moment_type, moment_label, pub_id):
        """Return the -t values for a given moment type, label, and publication ID."""
        data, n_to_row_map = load_lattice_data(moment_type, moment_label, pub_id)

        if data is None and n_to_row_map is None:
            print(f"No data found for {moment_type} {moment_label} {pub_id}. Skipping.")
            return None 
        
        if data is not None:
            # Safely access data[:, 0] since data is not None
            return data[:, 0]
        else:
            print(f"Data is None for {moment_type} {moment_label} {pub_id}. Skipping.")
        return None  # Or handle accordingly
    
    # Compute results for the evolution functions
    def compute_results(j, eta, t_vals, mu, Nf=3, particle="quark", moment_type="NonSingletIsovector", moment_label="A"):
        """Compute central, plus, and minus results for a given evolution function."""
        results = Parallel(n_jobs=-1)(
            delayed(lambda t: float(evolve_conformal_moment(j, eta, t, mu, Nf, particle, moment_type, moment_label, "central")))(t)
            for t in t_vals
        )
        results_plus = Parallel(n_jobs=-1)(
            delayed(lambda t: float(evolve_conformal_moment(j, eta, t, mu, Nf, particle, moment_type, moment_label, "plus")))(t)
            for t in t_vals
        )
        results_minus = Parallel(n_jobs=-1)(
            delayed(lambda t: float(evolve_conformal_moment(j, eta, t, mu, Nf, particle, moment_type, moment_label, "minus")))(t)
            for t in t_vals
        )
        return results, results_plus, results_minus

    # Define the finer grid for t-values
    t_fine = np.linspace(-t_max, 0, n_t)
    
    # Initialize the figure and axes for subplots
    fig, ax = plt.subplots(figsize=(7, 7)) 

    evolve_moment_central, evolve_moment_plus, evolve_moment_minus = compute_results(n,eta,t_fine,mu_in,Nf,particle,moment_type,moment_label)
    
    # Plot the RGE functions
    ax.plot(-t_fine, evolve_moment_central, color="blue", linewidth=2, label="This work")
    ax.fill_between(-t_fine, evolve_moment_minus, evolve_moment_plus, color="blue", alpha=0.2)
    
    # Plot data from publications

    for pub_id, (color,mu) in PUBLICATION_MAPPING.items():
        if mu != mu_in:
            continue
        data, n_to_row_map = load_lattice_data(moment_type, moment_label, pub_id)
        if data is None or n not in n_to_row_map:
            continue
        t_vals = t_values(moment_type, moment_label, pub_id)
        Fn0_vals = Fn0_values(n, moment_type, moment_label, pub_id)
        Fn0_errs = Fn0_errors(n, moment_type, moment_label, pub_id)
        ax.errorbar(t_vals, Fn0_vals, yerr=Fn0_errs, fmt='o', color=color, label=f"{pub_id}")

    # Add labels and formatting
    ax.set_xlabel("$-t\,[\mathrm{GeV}^2]$", fontsize=14)
    if particle == "gluon" and n == 2:
        ax.set_ylabel(f"$A_g(t,\mu = {mu_in}\,[\mathrm{{GeV}}])$", fontsize=14)
    elif particle == "quark" and moment_type == "Singlet" and n == 2:
        ax.set_ylabel(f"$A_{{u+d+s}}(t,\mu = {mu_in}\,[\mathrm{{GeV}}])$", fontsize=14)
    else:
        ax.set_ylabel(f"{y_label}$(j={n}, \\eta=0, t, \\mu={mu_in}\, \\mathrm{{GeV}})$", fontsize=14)
    ax.legend()
    ax.grid(True, which="both")
    ax.set_xlim([0, t_max])
    
    plt.show()

def plot_moments_D_on_grid(t_max, mu, Nf=3, n_t=50,display="both"):
    """
    Plot evolution for gluon or quark singlet D term conformal moments.

    Parameters:
        t_max (float): Maximum value of Mandelstam t.
        mu (float): Scale parameter (e.g., 2 GeV).
        Nf (int, optional): Number of flavors (default is 3).
        n_t (int, optional): Number of data points
        display (str, optional): "quark", "gluon", or "both" to specify which plots to display (default is "both").
    """
    if display not in ["quark", "gluon", "both"]:
        raise ValueError("Invalid value for display. Use 'quark', 'gluon', or 'both'.")
    
    # Compute results for the evolution functions
    def compute_results_D(j, eta, t_vals, mu, Nf=3, particle="quark", moment_type="NonSingletIsovector", moment_label="A"):
        """Compute central, plus, and minus results for a given evolution function."""
        results = Parallel(n_jobs=-1)(
            delayed(lambda t: float(evolve_singlet_D(j, eta, t, mu, Nf, particle, moment_label, "central")))(t)
            for t in t_vals
        )
        results_plus = Parallel(n_jobs=-1)(
            delayed(lambda t: float(evolve_singlet_D(j, eta, t, mu, Nf, particle, moment_label, "plus")))(t)
            for t in t_vals
        )
        results_minus = Parallel(n_jobs=-1)(
            delayed(lambda t: float(evolve_singlet_D(j, eta, t, mu, Nf, particle, moment_label, "minus")))(t)
            for t in t_vals
        )
        return results, results_plus, results_minus

    def plot_results(t_values, results, results_plus, results_minus, xlabel, ylabel, ax=None):
        """Plot results on the given axis, or create a new figure if ax is None."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(-t_values, results, label="This work", color="blue")
        ax.fill_between(-t_values, results_minus, results_plus, color="blue", alpha=0.2)
        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.grid(True)
        ax.legend()
        if ax is None:
            plt.tight_layout()
            plt.show()

    def plot_evolve_gluon_D(t_values, mu, Nf=3, ax=None):
        """Compute and plot gluon D evolution."""
        results, results_plus, results_minus = compute_results_D(2, 0, t_values, mu, Nf, "gluon","A")
        plot_results(t_values, results, results_plus, results_minus,
                    xlabel="$-t\,[\mathrm{GeV}^2]$", ylabel="$D_g(t,\mu = 2\,[\mathrm{GeV}])$", ax=ax)

    def plot_evolve_quark_singlet_D(t_values, mu, Nf=3, ax=None):
        """Compute and plot quark singlet D evolution."""
        results, results_plus, results_minus = compute_results_D(2, 0, t_values, mu, Nf, "quark","A")
        plot_results(t_values, results, results_plus, results_minus,
                    xlabel="$-t\,[\mathrm{GeV}^2]$", ylabel="$D_{u+d+s}(t,\mu = 2\,[\mathrm{GeV}])$", ax=ax)

    # Set up the figure and axes
    if display == "both":
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
        ax_gluon, ax_quark = axes
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax_gluon = ax_quark = ax

    t_values = np.linspace(-t_max,-1e-2,n_t)

    # Plot gluon evolution if requested
    if display in ["gluon", "both"]:
        plot_evolve_gluon_D(t_values, mu, Nf, ax=ax_gluon)

    # Plot quark singlet evolution if requested
    if display in ["quark", "both"]:
        plot_evolve_quark_singlet_D(t_values, mu, Nf, ax=ax_quark)

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


def plot_moments_on_grid(eta,y_label,t_max=3, Nf=3,particle="quark",moment_type="NonSingletIsovector", moment_label="A", n_t=50, num_columns=3):
    """
    Generates plots of lattice data and RGE-evolved functions for a given moment type and label on a grid. Only supports data with equal mu.
    If no data is available it sets mu = 2 GeV.
    
    Parameters:
    - eta (float): skewness parameter
    - y_label (str.): the label of the y axis
    - t_max (float, optional): Maximum t value for the x-axis (default is 3).
    - Nf (int, optional): Number of active flavors (default is 3)
    - particle (str., optional): Either quark or gluon
    - moment_type (str): The type of moment (e.g., "NonSingletIsovector").
    - moment_label (str): The label of the moment (e.g., "A").
    - n_t (int, optional): Number of points for t_fine (default is 50).
    - num_columns (int, optional): Number of columns for the grid layout (default is 3).
    """
    check_particle_type(particle)
    check_moment_type_label(moment_type,moment_label)
    # Accessor functions for -t, values, and errors
    def t_values(moment_type, moment_label, pub_id):
        """Return the -t values for a given moment type, label, and publication ID."""
        data, n_to_row_map = load_lattice_data(moment_type, moment_label, pub_id)

        if data is None and n_to_row_map is None:
            print(f"No data found for {moment_type} {moment_label} {pub_id}. Skipping.")
            return None 
        
        if data is not None:
            # Safely access data[:, 0] since data is not None
            return data[:, 0]
        else:
            print(f"Data is None for {moment_type} {moment_label} {pub_id}. Skipping.")
        return None  # Or handle accordingly
    
    # Compute results for the evolution functions
    def compute_results(j, eta, t_vals, mu, Nf=3, particle="quark", moment_type="NonSingletIsovector", moment_label="A"):
        """Compute central, plus, and minus results for a given evolution function."""
        results = Parallel(n_jobs=-1)(
            delayed(lambda t: float(evolve_conformal_moment(j, eta, t, mu, Nf, particle, moment_type, moment_label, "central")))(t)
            for t in t_vals
        )
        results_plus = Parallel(n_jobs=-1)(
            delayed(lambda t: float(evolve_conformal_moment(j, eta, t, mu, Nf, particle, moment_type, moment_label, "plus")))(t)
            for t in t_vals
        )
        results_minus = Parallel(n_jobs=-1)(
            delayed(lambda t: float(evolve_conformal_moment(j, eta, t, mu, Nf, particle, moment_type, moment_label, "minus")))(t)
            for t in t_vals
        )
        return results, results_plus, results_minus

    # Define the finer grid for t-values
    t_fine = np.linspace(-t_max, 0, n_t)
    
    # Initialize a list to store the number of n values per publication
    publication_data = {}
    
    # Loop through each publication ID to calculate the total number of plots
    for pub_id in PUBLICATION_MAPPING:
        data, n_to_row_map = load_lattice_data(moment_type, moment_label, pub_id)
        if data is None and n_to_row_map is None:
            #print(f"No data found for {pub_id}. Skipping.")
            continue
        num_n_values = (data.shape[1] - 1) // 2
        publication_data[pub_id] = num_n_values
    
    # Find the highest n value across all publications
    if publication_data:
        max_n_value = max(publication_data.values())
    else:
        max_n_value = 5
    
    # Calculate the number of rows needed for the grid layout
    num_rows = (max_n_value + num_columns - 1) // num_columns
    
    # Initialize the figure and axes for subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_columns, figsize=(15, num_rows * 5))
    axes = axes.flatten()
    
    # Loop through each n value up to the maximum n value
    if moment_type == "Singlet":
        n_0 = 2
    else:
        n_0 = 1
    for n in range(n_0, max_n_value + 1):
        ax = axes[n - 1]  # Select the appropriate axis
        # All lattice data currently at mu = 2 GeV and Nf = 3
        evolve_moment_central, evolve_moment_plus, evolve_moment_minus = compute_results(n,eta,t_fine,2,3,particle,moment_type,moment_label)
        
        # Plot the RGE functions
        ax.plot(-t_fine, evolve_moment_central, color="blue", linewidth=2, label="This work")
        ax.fill_between(-t_fine, evolve_moment_minus, evolve_moment_plus, color="blue", alpha=0.2)
        
        # Plot data from publications
        mu = None
        if publication_data:
            for pub_id, (color,mu) in PUBLICATION_MAPPING.items():
                data, n_to_row_map = load_lattice_data(moment_type, moment_label, pub_id)
                if data is None or n not in n_to_row_map:
                    continue
                t_vals = t_values(moment_type, moment_label, pub_id)
                Fn0_vals = Fn0_values(n, moment_type, moment_label, pub_id)
                Fn0_errs = Fn0_errors(n, moment_type, moment_label, pub_id)
                ax.errorbar(t_vals, Fn0_vals, yerr=Fn0_errs, fmt='o', color=color, label=f"{pub_id}")
        if mu == None:
            mu = 2 
        # Add labels and formatting
        ax.set_xlabel("$-t\,[\mathrm{GeV}^2]$", fontsize=14)
        if particle == "gluon" and n == 2:
            ax.set_ylabel(f"$A_g(t,\mu = {mu}\,[\mathrm{{GeV}}])$", fontsize=14)
        elif particle == "quark" and moment_type == "Singlet" and n == 2:
            ax.set_ylabel(f"$A_{{u+d+s}}(t,\mu = {mu}\,[\mathrm{{GeV}}])$", fontsize=14)
        else:
            ax.set_ylabel(f"{y_label}$(j={n}, \\eta=0, t, \\mu={mu}\, \\mathrm{{GeV}})$", fontsize=14)
        ax.legend()
        ax.grid(True, which="both")
        ax.set_xlim([0, t_max])
        #ax.set_ylim([0, np.max(x_plus_RGE_evolved_moment)])
    
    # Remove unused axes
    for i in range(max_n_value, len(axes)):
        fig.delaxes(axes[i])
        
    if moment_type == "Singlet":
        fig.delaxes(axes[0])
    
    # Adjust GridSpec to remove whitespace
    gs = plt.GridSpec(num_rows, num_columns, figure=fig)
    remaining_axes = fig.get_axes()  # Get all remaining axes

    for i, ax in enumerate(remaining_axes):
        ax.set_subplotspec(gs[i])  # Assign axes to new GridSpec slots
    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

def plot_fourier_transform_moments(j,eta,mu,plot_title,Nf=3,particle="quark",moment_type="NonSingletIsovector", moment_label="A", b_max = 2,Delta_max = 5,num_points=100,error_type="central"):
    """
    Generates a density plot of the 2D Fourier transfrom of RGE-evolved 
    conformal moments for a given moment type and label.
    
    Parameters:
    - j (float): Conformal spin
    - eta (float): Skewness parameter
    - mu (float): RGE scale
    - plot_title (str.): Title of the plot
    - particle (str. optional): "quark" or "gluon". Default is quark.
    - moment_type (str. optional): Singlet, NonSingletIsovector or NonSingletIsoscalar. Default is NonSingletIsovector.
    - moment_label (str. optiona): Label of conformal moment, e.g. A
    - b_max (float, optional): Maximum b value for the vector b_vec=[b_x,b_y] (default is 2).
    - Delta_max (float, optional): Maximum value for Delta integration (default is 11).
    - num_points (float, optional): Number of intervals to split [-Delta_max, Delta_max] interval (default is 100).
    - error_type (str. optional): Whether to use central, plus or minus value of input PDF. Default is central.
    """
    check_particle_type(particle)
    check_moment_type_label(moment_type,moment_label)
    # Define the grid for b_vec
    b_x = np.linspace(-b_max, b_max, 50)  # Range of x-component of b_vec
    b_y = np.linspace(-b_max, b_max, 50)  # Range of y-component of b_vec
    b_x, b_y = np.meshgrid(b_x, b_y)  # Create a grid of (b_x, b_y)
    # Flatten the grid for parallel processing
    b_vecs = np.array([b_x.ravel(), b_y.ravel()]).T

    # Parallel computation using joblib
    fourier_transform_moment_values_flat = Parallel(n_jobs=-1)(delayed(fourier_transform_moment)(j,eta,mu,b_vec,Nf,particle,moment_type, moment_label,Delta_max,num_points,error_type) for b_vec in b_vecs)

    # Reshape the result back to the grid shape
    fourier_transform_moment_values = np.array(fourier_transform_moment_values_flat).reshape(b_x.shape)

    # Convert Gev^-1 to fm
    hbarc = 0.1975

    # Create the 2D density plot
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(b_x*hbarc, b_y*hbarc, fourier_transform_moment_values, shading='auto', cmap='viridis')
    plt.colorbar()
    plt.xlabel(r'$b_x\,[\mathrm{fm}]$', fontsize=14)
    plt.ylabel(r'$b_y\,[\mathrm{fm}]$', fontsize=14)
    plt.title(f"{plot_title}$(j={j}, \\eta=0, t, \\mu=2\, \\mathrm{{GeV}})$", fontsize=14)
    plt.show()


def plot_conformal_partial_wave(j,eta,particle="quark",parity="none"):
    """Plots the conformal partial wave over conformal spin-j for given eta, particle and parity.

    Parameters:
    - j (float): conformal spin
    - eta (float): skewness
    - particle (str., optiona): quark or gluon. Default is quark
    - parity (str., optional): even, odd, or none. Default is none
    """
    check_particle_type(particle)
    check_parity(parity)

    x_values = np.linspace(-1, 1, 200)
    y_values = Parallel(n_jobs=-1)(delayed(conformal_partial_wave)(j, x, eta , particle, parity) for x in x_values)

    # Separate real and imaginary parts
    y_values_real = [float(y.real) for y in y_values]
    y_values_imag = [float(y.imag) for y in y_values]

    # Create subplots for real and imaginary parts
    plt.figure(figsize=(10, 6))  # Adjust figure size for better visualization

    #plt.subplot(2, 1, 1)
    plt.plot(x_values, y_values_real)
    plt.xlabel("x")
    plt.ylabel("Real Part")
    plt.title(f"Real Part of Conformal Partial Wave for {particle} with Parity {parity}")

    #plt.subplot(2, 1, 2)
    plt.plot(x_values, y_values_imag)
    plt.xlabel("x")
    plt.ylabel("Imaginary Part")
    plt.title(f"Imaginary Part of Conformal Partial Wave for {particle} with Parity {parity}")

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()

def plot_mellin_barnes_gpd_integrand(x, eta, t, mu, Nf=3, particle="quark", moment_type="Singlet", moment_label="A", parity = "none", error_type="central", j_max=7.5):
    """
    Plot the real and imaginary parts of the integrand of the Mellin-Barnes integral over k with j = j_base + i*k.

    Parameters:
    - x, eta, t, mu: Physical parameters.
    - Nf (int): Number of flavors.
    - particle (str): Particle species ("quark" or "gluon").
    - moment_type (str. optional): Moment type ("Singlet", "NonSingletIsovector", "NonSingletIsoscalar").
    - moment_label (str. optional): Moment label ("A", "Atilde", "B").
    - parity (str., optional)
    - error_type (str. optional): PDF value type ("central", "plus", "minus").
    - j_max (float. optional): Maximum value of imaginary part k for plotting.
    """
    check_parity(parity)
    check_error_type(error_type)
    check_particle_type(particle)
    check_moment_type_label(moment_type,moment_label)

    if ((moment_type == "Singlet" and particle == "quark" and parity != "odd")
        or (moment_type == "Singlet" and particle == "gluon" and parity != "even")
        or (moment_type == "NonSingletIsovector" and parity != "none")
        or (moment_type == "NonSingletIsoscalar" and parity != "none")
        or (particle == "gluon" and moment_type != "Singlet")
        ): 
        print(f"Warning: Wrong parity of {parity} for moment_type of {moment_type} for particle {particle}")

    if moment_type == "Singlet":
        if particle == "quark":
            parity = "odd"
        elif particle == "gluon":
            parity = "even"
        else:
            raise ValueError("Particle must be 'quark' or 'gluon'")
    elif moment_type == "NonSingletIsovector":
        parity = "none"
    elif moment_type == "NonSingletIsoscalar":
        parity = "none"

    j_base = get_j_base(particle, parity)

    if eta == 0:
        eta = 1e-6

    def integrand_real(k):
        z = j_base + 1j * k
        dz = 1j
        sin_term = mp.sin(np.pi * z)
        pw_val = conformal_partial_wave(z, x, eta, particle, parity)
        if particle == "quark":
            if moment_type == "Singlet":
                mom_val = evolve_quark_singlet(z, eta, t, mu, Nf, moment_label, error_type)
            else:
                mom_val = evolve_quark_non_singlet(z, eta, t, mu, Nf, moment_type, moment_label, error_type)
        else:
            mom_val = evolve_gluon_singlet(z, eta, t, mu, Nf=Nf, error_type=error_type)
        result = -0.5j * dz * pw_val * mom_val / sin_term
        return result.real

    def integrand_imag(k):
        z = j_base + 1j * k
        dz = 1j
        sin_term = mp.sin(np.pi * z)
        pw_val = conformal_partial_wave(z, x, eta, particle, parity)
        if particle == "quark":
            if moment_type == "Singlet":
                mom_val = evolve_quark_singlet(z, eta, t, mu, Nf, moment_label, error_type)
            else:
                mom_val = evolve_quark_non_singlet(z, eta, t, mu, Nf, moment_type, moment_label, error_type)
        else:
            mom_val = (-1) * evolve_gluon_singlet(z, eta, t, mu, Nf=Nf, error_type=error_type)
        result = -0.5j * dz * pw_val * mom_val / sin_term
        return result.imag

    print(f"Integrand at j_max={j_max}")
    print(integrand_real(j_max))
    print(integrand_imag(j_max))

    # Define k range for plotting
    k_values = np.linspace(-j_max, j_max, 300)
    # Parallel computation of real and imaginary parts
    real_values = Parallel(n_jobs=-1)(delayed(integrand_real)(k) for k in k_values)
    imag_values = Parallel(n_jobs=-1)(delayed(integrand_imag)(k) for k in k_values)

    # Plotting
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(k_values, real_values, label="Real Part", color="blue")
    ax[0].set_ylabel("Real Part")
    ax[0].legend()
    ax[0].grid()

    ax[1].plot(k_values, imag_values, label="Imaginary Part", color="red")
    ax[1].set_xlabel("k")
    ax[1].set_ylabel("Imaginary Part")
    ax[1].legend()
    ax[1].grid()

    plt.suptitle("Integrand Real and Imaginary Parts")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_singlet_quark_gpd(eta, t, mu, Nf=3, real_imag="real", sampling=True, n_init=os.cpu_count(), n_points=20, x_0=1e-2, x_1=1, error_bars=True):
    """
    Plot the real or imaginary part of the singlet quark GPD
    with dynamically adjusted x intervals, including error bars.
    The function supports both positive and negative values of parton x.

    Parameters:
    - eta (float): Skewness.
    - t (float): Mandelstam t
    - mu (float): Resolution scale
    - Nf (int 1<= Nf <= 4): Number of flavors
    - real_imag (str): Choose to plot real part, imaginary part or both. Imaginary part should be zero, so this is just for checks.
    - sampling (True or False): Choose whether to plot using importance sampling  
    - n_init (int): Sampling size, default is available number of CPUs
    - n_points (int): Number of plot points
    - x_0 (float): lower bound on parton x
    - x_1 (float): upper bound on parton x
    - error_bars (True or False): Compute error bars as well
    """
    # Ensure x_0 < x_1 for a valid range
    if x_0 >= x_1:
        raise ValueError("x_0 must be less than x_1.")

    if x_0 <= 0:
        raise ValueError("x_0 must be greater than zero.")

    def compute_result(x, error_type="central"):
        return mellin_barnes_gpd(x, eta, t, mu, Nf,particle="quark",moment_type="Singlet", real_imag=real_imag, error_type=error_type,n_jobs=1)

    if sampling:
        x_values = np.linspace(x_0, x_1, n_init)

        # Measure time for sampling initial points
        start_time_sampling = time.time()
        results = Parallel(n_jobs=-1)(delayed(compute_result)(x) for x in x_values)
        end_time_sampling = time.time()

        # Compute differences and scale intervals
        diffs = np.abs(np.diff(results))
        scaled_intervals = diffs / np.sum(diffs)
        cumulative_intervals = np.cumsum(np.insert(scaled_intervals, 0, 0))

        # Output sampling time
        print(f"Time for initial sampling: {end_time_sampling - start_time_sampling:.6f} seconds")

    # Measure time for adaptive grid computation
    start_time_adaptive = time.time()
    if sampling:
        x_values = np.interp(np.linspace(0, 1, n_points), cumulative_intervals, x_values)
    else:
        x_values = np.linspace(x_0, x_1, n_points)

    results = Parallel(n_jobs=-1)(delayed(compute_result)(x) for x in x_values)

    # Error bar computations
    if error_bars:
        results_plus = Parallel(n_jobs=-1)(delayed(compute_result)(x, error_type="plus") for x in x_values)
        results_minus = Parallel(n_jobs=-1)(delayed(compute_result)(x, error_type="minus") for x in x_values)
    else:
        results_plus = results
        results_minus = results

    end_time_adaptive = time.time()

    # Extract real and imaginary parts of results
    real_parts = np.real(results)
    imag_parts = np.imag(results)

    # Compute real and imaginary error bars
    real_errors_plus = abs(np.real(results_plus) - real_parts)
    real_errors_minus = abs(real_parts - np.real(results_minus))
    imag_errors_plus = abs(np.imag(results_plus) - imag_parts)
    imag_errors_minus = abs(imag_parts - np.imag(results_minus))

    # Output plot generation time
    print(f"Time for plot computation: {end_time_adaptive - start_time_adaptive:.6f} seconds")

    # Define the data and labels for real and imaginary parts
    plot_parts = [
        ("real", real_parts, real_errors_minus, real_errors_plus, 'b', 'Singlet Sea Quark GPD'),
        ("imag", imag_parts, imag_errors_minus, imag_errors_plus, 'r', 'Imaginary Part of Singlet Sea Quark GPD')
    ]

    # Plot real and/or imaginary parts
    for part, data, errors_minus, errors_plus, color, title in plot_parts:
        if real_imag in (part, "both"):
            if error_bars:
                plt.errorbar(
                    x_values, data,
                    yerr=(errors_minus, errors_plus),
                    fmt='o', label=(f"$\\eta={eta:.2f}$"
                                    "\n"
                                    f"$t={t:.2f} \\text{{ GeV}}^2$"
                                    "\n"
                                    f"$\\mu = {mu} \\text{{ GeV}}$"),
                    color=color, capsize=3
                )
            else:
                plt.scatter(
                    x_values, data,
                    label=(f"$\\eta={eta:.2f}$"
                                "\n"
                                f"$t={t:.2f} \\text{{ GeV}}^2$"
                                "\n"
                                f"$\\mu = {mu} \\text{{ GeV}}$"),
                    color=color
                )

    # Set the title based on real_imag
    if real_imag == "real":
        plt.title('Singlet Sea Quark GPD')
    elif real_imag == "imag":
        plt.title('Imaginary Part of Singlet Sea Quark GPD')
    elif real_imag == "both":
        plt.title('Real and Imaginary Part of Singlet Sea Quark GPD')

    # Add vertical lines to separate DGLAP from ERBL region
    plt.axvline(x=eta, linestyle='--')   
    plt.axvline(x=-eta, linestyle='--')

    plt.xlim(x_0, x_1)
    plt.xlabel('x')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_non_singlet_quark_gpd(eta, t, mu, Nf=3, real_imag="real", sampling=True, n_init=os.cpu_count(), n_points=30, x_0=-1, x_1=1, error_bars=True):
    """
    Plot the real or imaginary part of the non-singlet quark GPD
    with dynamically adjusted x intervals, including error bars.
    The function supports both positive and negative values of parton x.

    Parameters:
    - eta (float): Skewness.
    - t (float): Mandelstam t
    - mu (float): Resolution scale
    - Nf (int 1<= Nf <= 4): Number of flavors
    - real_imag (str): Choose to plot real part, imaginary part or both. Imaginary part should be zero, so this is just for checks.
    - sampling (True or False): Choose whether to plot using importance sampling 
    - n_init (int): Sampling size, default is available number of CPUs
    - n_points (int): Number of plot points
    - x_0 (float): lower bound on parton x
    - x_1 (float): upper bound on parton x
    - error_bars (True or False): Compute error bars as well
    """
    # Ensure x_0 < x_1 for a valid range
    if x_0 >= x_1:
        raise ValueError("x_0 must be less than x_1.")

    # Validate real_imag input
    if real_imag not in ("real", "imag", "both"):
        raise ValueError("Invalid option for real_imag. Choose from 'real', 'imag', or 'both'.")

    def compute_result(x, error_type="central"):
        return mellin_barnes_gpd(x, eta, t, mu, Nf, particle="quark",moment_type="NonSingletIsovector",real_imag=real_imag, error_type=error_type, n_jobs=1)

    if sampling:
        x_values = np.linspace(x_0, x_1, n_init)

        # Measure time for sampling initial points
        start_time_sampling = time.time()
        results = Parallel(n_jobs=-1)(delayed(compute_result)(x) for x in x_values)
        end_time_sampling = time.time()

        # Compute differences and scale intervals
        diffs = np.abs(np.diff(results))
        scaled_intervals = diffs / np.sum(diffs)
        cumulative_intervals = np.cumsum(np.insert(scaled_intervals, 0, 0))

        # Output sampling time
        print(f"Time for initial sampling: {end_time_sampling - start_time_sampling:.6f} seconds")

    # Measure time for adaptive grid computation
    start_time_adaptive = time.time()
    if sampling:
        x_values = np.interp(np.linspace(0, 1, n_points), cumulative_intervals, x_values)
    else:
        x_values = np.linspace(x_0, x_1, n_points)

    results = Parallel(n_jobs=-1)(delayed(compute_result)(x) for x in x_values)

    # Error bar computations
    if error_bars:
        results_plus = Parallel(n_jobs=-1)(delayed(compute_result)(x, error_type="plus") for x in x_values)
        results_minus = Parallel(n_jobs=-1)(delayed(compute_result)(x, error_type="minus") for x in x_values)
    else:
        results_plus = results
        results_minus = results

    end_time_adaptive = time.time()

    # Extract real and imaginary parts of results
    real_parts = np.real(results)
    imag_parts = np.imag(results)

    # Compute real and imaginary error bars
    real_errors_plus = abs(np.real(results_plus) - real_parts)
    real_errors_minus = abs(real_parts - np.real(results_minus))
    imag_errors_plus = abs(np.imag(results_plus) - imag_parts)
    imag_errors_minus = abs(imag_parts - np.imag(results_minus))

    # Output plot generation time
    print(f"Time for plot computation: {end_time_adaptive - start_time_adaptive:.6f} seconds")

    # Define the data and labels for real and imaginary parts
    plot_parts = [
        ("real", real_parts, real_errors_minus, real_errors_plus, 'b', 'Non-Singlet Quark GPD'),
        ("imag", imag_parts, imag_errors_minus, imag_errors_plus, 'r', 'Imaginary Part of Non-Singlet Quark GPD')
    ]


    # Plot real and/or imaginary parts
    for part, data, errors_minus, errors_plus, color, title in plot_parts:
        if real_imag in (part, "both"):
            if error_bars:
                plt.errorbar(
                    x_values, data,
                    yerr=(errors_minus, errors_plus),
                    fmt='o', label=(f"$\\eta={eta:.2f}$"
                                    "\n"
                                    f"$t={t:.2f} \\text{{ GeV}}^2$"
                                    "\n"
                                    f"$\\mu = {mu} \\text{{ GeV}}$"),
                    color=color, capsize=3
                )
            else:
                plt.scatter(
                    x_values, data,
                    label=(f"$\\eta={eta:.2f}$"
                                "\n"
                                f"$t={t:.2f} \\text{{ GeV}}^2$"
                                "\n"
                                f"$\\mu = {mu} \\text{{ GeV}}$"),
                    color=color
                )

    # Set the title based on real_imag
    if real_imag == "real":
        plt.title('Non-Singlet Isovector Quark GPD')
    elif real_imag == "imag":
        plt.title('Imaginary Part of Non-Singlet Isovector Quark GPD')
    elif real_imag == "both":
        plt.title('Real and Imaginary Part of Non-Singlet Isovector Quark GPD')

    # Add vertical lines to separate DGLAP from ERBL region
    plt.axvline(x=eta, linestyle='--')   
    plt.axvline(x=-eta, linestyle='--')

    plt.xlim(x_0, x_1)
    plt.xlabel('x')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def plot_gluon_gpd(eta, t, mu, Nf=3, real_imag="real", sampling=True, n_init=os.cpu_count(), n_points=20, x_0=1e-2, x_1=1, error_bars=True):
    """
    Plot the real or imaginary part of the non-singlet quark GPD
    with dynamically adjusted x intervals, including error bars.
    The function supports both positive and negative values of parton x.

    Parameters:
    - eta (float): Skewness.
    - t (float): Mandelstam t
    - mu (float): Resolution scale
    - Nf (int 1<= Nf <= 4): Number of flavors
    - real_imag (str): Choose to plot real part, imaginary part or both. Imaginary part should be zero, so this is just for checks.
    - sampling (True or False): Choose whether to plot using importance sampling 
    - n_init (int): Sampling size, default is available number of CPUs
    - n_points (int): Number of plot points
    - x_0 (float): lower bound on parton x
    - x_1 (float): upper bound on parton x
    - error_bars (True or False): Computer error bars as well
    """
    # Ensure x_0 < x_1 for a valid range
    if x_0 >= x_1:
        raise ValueError("x_0 must be less than x_1.")

    if x_0 <= 0:
        raise ValueError("x_0 must be greater than zero.")

    def compute_result(x, error_type="central"):
        return mellin_barnes_gpd(x, eta, t, mu, Nf, particle="gluon",moment_type="Singlet",real_imag=real_imag, error_type=error_type,n_jobs=1)

    if sampling:
        x_values = np.linspace(x_0, x_1, n_init)

        # Measure time for sampling initial points
        start_time_sampling = time.time()
        results = Parallel(n_jobs=-1)(delayed(compute_result)(x) for x in x_values)
        end_time_sampling = time.time()

        # Compute differences and scale intervals
        diffs = np.abs(np.diff(results))
        scaled_intervals = diffs / np.sum(diffs)
        cumulative_intervals = np.cumsum(np.insert(scaled_intervals, 0, 0))

        # Output sampling time
        print(f"Time for initial sampling: {end_time_sampling - start_time_sampling:.6f} seconds")

    # Measure time for adaptive grid computation
    start_time_adaptive = time.time()
    if sampling:
        x_values = np.interp(np.linspace(0, 1, n_points), cumulative_intervals, x_values)
    else:
        x_values = np.linspace(x_0, x_1, n_points)

    results = Parallel(n_jobs=-1)(delayed(compute_result)(x) for x in x_values)

    # Error bar computations
    if error_bars:
        results_plus = Parallel(n_jobs=-1)(delayed(compute_result)(x, error_type="plus") for x in x_values)
        results_minus = Parallel(n_jobs=-1)(delayed(compute_result)(x, error_type="minus") for x in x_values)
    else:
        results_plus = results
        results_minus = results

    end_time_adaptive = time.time()

    # Extract real and imaginary parts of results
    real_parts = np.real(results)
    imag_parts = np.imag(results)

    # Compute real and imaginary error bars
    real_errors_plus = abs(np.real(results_plus) - real_parts)
    real_errors_minus = abs(real_parts - np.real(results_minus))
    imag_errors_plus = abs(np.imag(results_plus) - imag_parts)
    imag_errors_minus = abs(imag_parts - np.imag(results_minus))

    # Output plot generation time
    print(f"Time for plot computation: {end_time_adaptive - start_time_adaptive:.6f} seconds")

    # Define the data and labels for real and imaginary parts
    plot_parts = [
        ("real", real_parts, real_errors_minus, real_errors_plus, 'b', 'Singlet Gluon GPD'),
        ("imag", imag_parts, imag_errors_minus, imag_errors_plus, 'r', 'Imaginary Part of Singlet Gluon GPD')
    ]

    # Plot real and/or imaginary parts
    for part, data, errors_minus, errors_plus, color, title in plot_parts:
        if real_imag in (part, "both"):
            if error_bars:
                plt.errorbar(
                    x_values, data,
                    yerr=(errors_minus, errors_plus),
                    fmt='o', label=(f"$\\eta={eta:.2f}$"
                                    "\n"
                                    f"$t={t:.2f} \\text{{ GeV}}^2$"
                                    "\n"
                                    f"$\\mu = {mu} \\text{{ GeV}}$"),
                    color=color, capsize=3
                )
            else:
                plt.scatter(
                    x_values, data,
                    label=(f"$\\eta={eta:.2f}$"
                                "\n"
                                f"$t={t:.2f} \\text{{ GeV}}^2$"
                                "\n"
                                f"$\\mu = {mu} \\text{{ GeV}}$"),
                    color=color
                )

    # Set the title based on real_imag
    if real_imag == "real":
        plt.title('Singlet Gluon GPD')
    elif real_imag == "imag":
        plt.title('Imaginary Part of Singlet Gluon GPD')
    elif real_imag == "both":
        plt.title('Real and Imaginary Part of Singlet Gluon GPD')

    # Add vertical lines to separate DGLAP from ERBL region
    plt.axvline(x=eta, linestyle='--')   
    plt.axvline(x=-eta, linestyle='--')

    plt.xlim(x_0, x_1)
    plt.xlabel('x')
    plt.legend()
    plt.grid(True)
    plt.show()