# # Dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from joblib import Parallel, delayed
from scipy.special import gamma, digamma
import mpmath as mp
import re
import os
############################################
############################################
# Set up dictionaries
# Add some colors
saturated_pink = (1.0, 0.1, 0.6)  # Higher red, some blue, minimal green
# Define a dictionary that maps publication IDs to a color
def initialize_dictionaries():
    global publication_mapping 
    publication_mapping = {
        "2305.11117": "cyan",
        "0705.4295": "orange",
        "1908.10706": saturated_pink
    # Add more publication IDs and corresponding colors here
    }
    # Define dictionary that maps conformal moments names and types to expressions
    global moment_to_function
    moment_to_function = {
    ("NonSingletIsovector", "A"): uv_minus_dv_Regge,
    #("NonSingletIsovector", "A"): u_minus_d_Regge,
    ("NonSingletIsoscalar", "A"): uv_plus_dv_Regge,
    #("NonSingletIsoscalar", "A"): u_plus_d_Regge,
    }
############################################
############################################
# Import MSTW PDF data
# Base path to main data directory
base_path = "/mnt/c/Users/flori/Documents/PostDoc/Jupyter/Data/GPD/"
# Define the file path to the .dat file and extract its content
MSTW_path = f"{base_path}MSTW_Table_4.dat"
# Read the .dat file into a DataFrame
columns = ["Parameter", "LO", "NLO", "NNLO"]
data = []

# Parsing the .dat file
with open(MSTW_path, "r") as file:
    for line in file:
        # Skip header lines
        if line.startswith("#") or line.strip() == "":
            continue
        
        # Split the line into columns
        parts = line.split("\t")
        parameter = parts[0]
        lo_values = eval(parts[1].strip())  # Convert string "[...]" to a Python list
        nlo_values = eval(parts[2].strip())
        nnlo_values = eval(parts[3].strip())
        
        # Append to the data list
        data.append([parameter, lo_values, nlo_values, nnlo_values])

# Create a DataFrame from the parsed data
MSTWpdf = pd.DataFrame(data, columns=columns)

def load_lattice_data(moment_type, moment_label, pub_id):
    """
    Load data from a .dat file, extracting 'n' values from the header and associating them with rows.

    Args:
        moment_type (str): The type of moment (e.g., "Isovector").
        moment_label (str): The label of the moment (e.g., "A").
        pub_id (str): The publication ID.

    Returns:
        tuple: A tuple containing the data, and a dictionary mapping 'n' values to row indices.
    """

    filename = f"{moment_type}Moments{moment_label}{pub_id}.dat"
    file_path = f"{base_path}{filename}"

    # Check if the file exists
    if not os.path.exists(file_path):
        #print(f"No data available for {moment_type}{moment_label} in {pub_id}")
        return None, None

    with open(file_path, 'r') as f:
        header = f.readline().strip()
        data = np.loadtxt(f)

    # Extract 'n' values from the header
    n_values = []
    for col_name in header.split():
        match = re.search(rf'{moment_label}(\d+)0_', col_name)
        if match:
            n_values.append(int(match.group(1)))

    # Create a dictionary mapping 'n' values to row indices
    n_to_row_map = {n: i for i, n in enumerate(n_values)}

    return data, n_to_row_map

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

# Extract LO, NLO and NLO columns
MSTWpdf_LO=MSTWpdf[["LO"]]
MSTWpdf_NLO=MSTWpdf[["NLO"]]
MSTWpdf_NNLO=MSTWpdf[["NNLO"]]

# Define plot funtion
# Define the PDFs using Eqs. (6-12) in  0901.0002 
def uv(x, error_type="central"):
    """
    Compute the uv(x) PDF based on the given LO parameters and selected errors.
    
    Arguments:
    x -- The value of parton x.
    error_type -- A string indicating whether to use 'central', 'plus', or 'minus' errors. Default is 'central'.
    
    Returns:
    The value of uv(x) based on the selected parameters and error type.
    """
     # Define a dictionary that maps the error_type to column indices
    error_mapping = {
        "central": 0,  # The column with the central value
        "plus": 1,     # The column with the + error value
        "minus": 2     # The column with the - error value
    }
    
    # Get the column index corresponding to the error_type
    error_col_index = error_mapping.get(error_type, 0)  # Default to 'central' if error_type is invalid

    # Get row index of entry
    index_A_u=MSTWpdf[MSTWpdf["Parameter"] == "A_u"].index[0]
    index_eta_1=MSTWpdf[MSTWpdf["Parameter"] == "eta_1"].index[0]
    index_eta_2=MSTWpdf[MSTWpdf["Parameter"] == "eta_2"].index[0]
    index_epsilon_u=MSTWpdf[MSTWpdf["Parameter"] == "epsilon_u"].index[0]
    index_gamma_u=MSTWpdf[MSTWpdf["Parameter"] == "gamma_u"].index[0]

    # Extracting parameter values based on the error_type argument
    A_u = MSTWpdf_LO.iloc[index_A_u,0][0] + int(error_col_index>0)*MSTWpdf_LO.iloc[index_A_u,0][error_col_index]
    eta_1 = MSTWpdf_LO.iloc[index_eta_1,0][0] + int(error_col_index>0)*MSTWpdf_LO.iloc[index_eta_1,0][error_col_index]
    eta_2 = MSTWpdf_LO.iloc[index_eta_2,0][0] + int(error_col_index>0)*MSTWpdf_LO.iloc[index_eta_2,0][error_col_index]
    epsilon_u = MSTWpdf_LO.iloc[index_epsilon_u,0][0] + int(error_col_index>0)*MSTWpdf_LO.iloc[index_epsilon_u,0][error_col_index]
    gamma_u = MSTWpdf_LO.iloc[index_gamma_u,0][0] + int(error_col_index>0)*MSTWpdf_LO.iloc[index_gamma_u,0][error_col_index]
    
    # Compute the uv(x) equation using numpy operations
    result = A_u * (x ** (eta_1 - 1)) * ((1 - x) ** eta_2) * (1 + epsilon_u * np.sqrt(x) + gamma_u * x)
    
    return result

def dv(x, error_type="central"):
    # Define a dictionary that maps the error_type to column indices
    error_mapping = {
        "central": 0,  # The column with the central value
        "plus": 1,     # The column with the + error value
        "minus": 2     # The column with the - error value
    }
    
    # Get the column index corresponding to the error_type
    error_col_index = error_mapping.get(error_type, 0)

    # Get row index of entry
    index_A_d = MSTWpdf[MSTWpdf["Parameter"] == "A_d"].index[0]
    index_eta_3 = MSTWpdf[MSTWpdf["Parameter"] == "eta_3"].index[0]
    index_eta_2=MSTWpdf[MSTWpdf["Parameter"] == "eta_2"].index[0]
    # Only eta_4-eta_2 given
    index_eta_42 = MSTWpdf[MSTWpdf["Parameter"] == "eta_4-eta_2"].index[0]
    index_epsilon_d = MSTWpdf[MSTWpdf["Parameter"] == "epsilon_d"].index[0]
    index_gamma_d = MSTWpdf[MSTWpdf["Parameter"] == "gamma_d"].index[0]

    # Extracting parameter values based on the error_type argument
    A_d = MSTWpdf_LO.iloc[index_A_d, 0][0] + int(error_col_index>0) * MSTWpdf_LO.iloc[index_A_d, 0][error_col_index]
    eta_3 = MSTWpdf_LO.iloc[index_eta_3, 0][0] + int(error_col_index>0) * MSTWpdf_LO.iloc[index_eta_3, 0][error_col_index]
    # eta_4=(eta_4-eta_2) + eta_2, Add errors in quadrature
    eta_4 = (MSTWpdf_LO.iloc[index_eta_42, 0][0] + MSTWpdf_LO.iloc[index_eta_2, 0][0]) + int(error_col_index>0) *np.sign(MSTWpdf_LO.iloc[index_eta_42, 0][error_col_index]) * np.sqrt(MSTWpdf_LO.iloc[index_eta_42, 0][error_col_index]**2+MSTWpdf_LO.iloc[index_eta_2, 0][error_col_index]**2)
    epsilon_d = MSTWpdf_LO.iloc[index_epsilon_d, 0][0] + int(error_col_index>0) * MSTWpdf_LO.iloc[index_epsilon_d, 0][error_col_index]
    gamma_d = MSTWpdf_LO.iloc[index_gamma_d, 0][0] + int(error_col_index>0) * MSTWpdf_LO.iloc[index_gamma_d, 0][error_col_index]
    
    # Compute the uv(x) equation using numpy operations
    result = A_d * (x ** (eta_3 - 1)) * ((1 - x) ** eta_4) * (1 + epsilon_d * np.sqrt(x) + gamma_d * x)
    
    return result

def sv(x, error_type="central"):
    error_mapping = {
        "central": 0,
        "plus": 1,
        "minus": 2
    }
    
    error_col_index = error_mapping.get(error_type, 0)

    # delta_- fixed to 0.2
    index_A_m = MSTWpdf[MSTWpdf["Parameter"] == "A_-"].index[0]
    index_eta_m = MSTWpdf[MSTWpdf["Parameter"] == "eta_-"].index[0]
    index_x_0 = MSTWpdf[MSTWpdf["Parameter"] == "x_0"].index[0]

    A_m = MSTWpdf_LO.iloc[index_A_m, 0][0] + int(error_col_index>0) * MSTWpdf_LO.iloc[index_A_m, 0][error_col_index]
    delta_m = .2
    eta_m = MSTWpdf_LO.iloc[index_eta_m, 0][0] + int(error_col_index>0) * MSTWpdf_LO.iloc[index_eta_m, 0][error_col_index]
    x_0 = MSTWpdf_LO.iloc[index_x_0, 0][0] + int(error_col_index>0) * MSTWpdf_LO.iloc[index_x_0, 0][error_col_index]

    result = A_m * (x ** (delta_m - 1)) * ((1 - x) ** eta_m) * (1 -x/x_0)
    
    return result

def Sv(x, error_type="central"):
    error_mapping = {
        "central": 0,
        "plus": 1,
        "minus": 2
    }
    
    error_col_index = error_mapping.get(error_type, 0)

    index_A_S = MSTWpdf[MSTWpdf["Parameter"] == "A_S"].index[0]
    index_delta_S = MSTWpdf[MSTWpdf["Parameter"] == "delta_S"].index[0]
    index_eta_S = MSTWpdf[MSTWpdf["Parameter"] == "eta_S"].index[0]
    index_epsilon_S = MSTWpdf[MSTWpdf["Parameter"] == "epsilon_S"].index[0]
    index_gamma_S = MSTWpdf[MSTWpdf["Parameter"] == "gamma_S"].index[0]

    A_S = MSTWpdf_LO.iloc[index_A_S, 0][0] + int(error_col_index>0) * MSTWpdf_LO.iloc[index_A_S, 0][error_col_index]
    delta_S = MSTWpdf_LO.iloc[index_delta_S, 0][0] + int(error_col_index>0) * MSTWpdf_LO.iloc[index_delta_S, 0][error_col_index]
    eta_S = MSTWpdf_LO.iloc[index_eta_S, 0][0] + int(error_col_index>0) * MSTWpdf_LO.iloc[index_eta_S, 0][error_col_index]
    epsilon_S = MSTWpdf_LO.iloc[index_epsilon_S, 0][0] + int(error_col_index>0) * MSTWpdf_LO.iloc[index_epsilon_S, 0][error_col_index]
    gamma_S = MSTWpdf_LO.iloc[index_gamma_S, 0][0] + int(error_col_index>0) * MSTWpdf_LO.iloc[index_gamma_S, 0][error_col_index]

    result = A_S * (x ** (delta_S - 1)) * ((1 - x) ** eta_S) * (1 + epsilon_S * np.sqrt(x) + gamma_S * x)
    
    return result

def s_plus(x, error_type="central"):
    error_mapping = {
        "central": 0,
        "plus": 1,
        "minus": 2
    }
    
    error_col_index = error_mapping.get(error_type, 0)

    index_A_p = MSTWpdf[MSTWpdf["Parameter"] == "A_+"].index[0]
    index_delta_S = MSTWpdf[MSTWpdf["Parameter"] == "delta_S"].index[0]
    index_eta_p = MSTWpdf[MSTWpdf["Parameter"] == "eta_+"].index[0]
    index_epsilon_S = MSTWpdf[MSTWpdf["Parameter"] == "epsilon_S"].index[0]
    index_gamma_S = MSTWpdf[MSTWpdf["Parameter"] == "gamma_S"].index[0]

    A_p = MSTWpdf_LO.iloc[index_A_p, 0][0] + int(error_col_index>0) * MSTWpdf_LO.iloc[index_A_p, 0][error_col_index]
    delta_S = MSTWpdf_LO.iloc[index_delta_S, 0][0] + int(error_col_index>0) * MSTWpdf_LO.iloc[index_delta_S, 0][error_col_index]
    eta_p = MSTWpdf_LO.iloc[index_eta_p, 0][0] + int(error_col_index>0) * MSTWpdf_LO.iloc[index_eta_p, 0][error_col_index]
    epsilon_S = MSTWpdf_LO.iloc[index_epsilon_S, 0][0] + int(error_col_index>0) * MSTWpdf_LO.iloc[index_epsilon_S, 0][error_col_index]
    gamma_S = MSTWpdf_LO.iloc[index_gamma_S, 0][0] + int(error_col_index>0) * MSTWpdf_LO.iloc[index_gamma_S, 0][error_col_index]

    result = A_p * (x ** (delta_S - 1)) * ((1 - x) ** eta_p) * (1 + epsilon_S * np.sqrt(x) + gamma_S * x)
    
    return result

def Delta(x, error_type="central"):
    """
    Compute the Delta(x)=dbar-ubar PDF based on the given LO parameters and selected errors.
    """
     # Define a dictionary that maps the error_type to column indices
    error_mapping = {
        "central": 0,  # The column with the central value
        "plus": 1,     # The column with the + error value
        "minus": 2     # The column with the - error value
    }
    
    # Get the column index corresponding to the error_type
    error_col_index = error_mapping.get(error_type, 0)  # Default to 'central' if error_type is invalid

    # Get row index of entry
    index_A_Delta=MSTWpdf[MSTWpdf["Parameter"] == "A_Delta"].index[0]
    index_eta_Delta=MSTWpdf[MSTWpdf["Parameter"] == "eta_Delta"].index[0]
    index_eta_S=MSTWpdf[MSTWpdf["Parameter"] == "eta_S"].index[0]
    index_gamma_Delta=MSTWpdf[MSTWpdf["Parameter"] == "gamma_Delta"].index[0]
    index_delta_Delta=MSTWpdf[MSTWpdf["Parameter"] == "delta_Delta"].index[0]

    # Extracting parameter values based on the error_type argument
    A_Delta = MSTWpdf_LO.iloc[index_A_Delta,0][0] + int(error_col_index>0)*MSTWpdf_LO.iloc[index_A_Delta,0][error_col_index]
    eta_Delta = MSTWpdf_LO.iloc[index_eta_Delta,0][0] + int(error_col_index>0)*MSTWpdf_LO.iloc[index_eta_Delta,0][error_col_index]
    eta_S = MSTWpdf_LO.iloc[index_eta_S,0][0] + int(error_col_index>0)*MSTWpdf_LO.iloc[index_eta_S,0][error_col_index]
    gamma_Delta = MSTWpdf_LO.iloc[index_gamma_Delta,0][0] + int(error_col_index>0)*MSTWpdf_LO.iloc[index_gamma_Delta,0][error_col_index]
    delta_Delta = MSTWpdf_LO.iloc[index_delta_Delta,0][0] + int(error_col_index>0)*MSTWpdf_LO.iloc[index_delta_Delta,0][error_col_index]


    # Compute the uv(x) equation using numpy operations
    result = A_Delta * (x ** (eta_Delta - 1)) * (1 - x) ** (eta_S+2) * (1 + gamma_Delta*x + delta_Delta*x**2)
    
    return result

def gv(x, error_type="central"):
     # Define a dictionary that maps the error_type to column indices
    error_mapping = {
        "central": 0,  # The column with the central value
        "plus": 1,     # The column with the + error value
        "minus": 2     # The column with the - error value
    }
    
    # Get the column index corresponding to the error_type
    error_col_index = error_mapping.get(error_type, 0)  # Default to 'central' if error_type is invalid

    # Get row index of entry
    index_A_g=MSTWpdf[MSTWpdf["Parameter"] == "A_g"].index[0]
    index_delta_g=MSTWpdf[MSTWpdf["Parameter"] == "delta_g"].index[0]
    index_eta_g=MSTWpdf[MSTWpdf["Parameter"] == "eta_g"].index[0]
    index_epsilon_g=MSTWpdf[MSTWpdf["Parameter"] == "epsilon_g"].index[0]
    index_gamma_g=MSTWpdf[MSTWpdf["Parameter"] == "gamma_g"].index[0]

    # Extracting parameter values based on the error_type argument
    A_g = MSTWpdf_LO.iloc[index_A_g,0][0] + int(error_col_index>0)*MSTWpdf_LO.iloc[index_A_g,0][error_col_index]
    delta_g = MSTWpdf_LO.iloc[index_delta_g,0][0] + int(error_col_index>0)*MSTWpdf_LO.iloc[index_delta_g,0][error_col_index]
    eta_g = MSTWpdf_LO.iloc[index_eta_g,0][0] + int(error_col_index>0)*MSTWpdf_LO.iloc[index_eta_g,0][error_col_index]
    epsilon_g = MSTWpdf_LO.iloc[index_epsilon_g,0][0] + int(error_col_index>0)*MSTWpdf_LO.iloc[index_epsilon_g,0][error_col_index]
    gamma_g = MSTWpdf_LO.iloc[index_gamma_g,0][0] + int(error_col_index>0)*MSTWpdf_LO.iloc[index_gamma_g,0][error_col_index]

    # Compute the uv(x) equation using numpy operations
    # At LO this is the full expression
    result = A_g * (x ** (delta_g - 1)) * ((1 - x) ** eta_g) * (1 + epsilon_g * np.sqrt(x) + gamma_g * x)
    
    return result 

def evolve_alpha_s(mu):
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
    Nf = 3
    mu_R2 = 1 # 1 GeV**2
    # Extract value of alpha_S at the renormalization point of mu_R**2 = 1 GeV**2
    index_alpha_s=MSTWpdf[MSTWpdf["Parameter"] == "alpha_S(Q0^2)"].index[0]
    alpha_s_in = MSTWpdf_LO.iloc[index_alpha_s,0][0]
    beta_0 = 11/3 * Nc - 2/3* Nf

     # Evolve using LO RGE
    log_term = np.log(mu**2 / mu_R2)
    denominator = 1 + (alpha_s_in / (4 * np.pi)) * beta_0 * log_term
    
    # Debug:
    # print(index_alpha_s)
    # print(alpha_s_in)

    result = alpha_s_in / denominator

    return result

def int_uv_Regge(j,eta,alpha_p,t, error_type="central"):
    """
    Compute the integral of the Reggeized uv(x) PDF based on the given LO parameters and selected errors.
    
    Arguments:
    j -- conformal spin,
    eta -- skewness (scalar or array)(placeholder for now),
    alpha_p -- Regge slope,
    t -- Mandelstam t (scalar or array),
    error_type -- A string indicating whether to use 'central', 'plus', or 'minus' errors. Default is 'central'.
    
    Returns:
    The value of the Reggeized integral of uv(x) based on the selected parameters and error type.
    """
     # Define a dictionary that maps the error_type to column indices
    error_mapping = {
        "central": 0,  # The column with the central value
        "plus": 1,     # The column with the + error value
        "minus": 2     # The column with the - error value
    }
    
    # Get the column index corresponding to the error_type
    error_col_index = error_mapping.get(error_type, 0)  # Default to 'central' if error_type is invalid

    # Get row index of entry
    index_A_u=MSTWpdf[MSTWpdf["Parameter"] == "A_u"].index[0]
    index_eta_1=MSTWpdf[MSTWpdf["Parameter"] == "eta_1"].index[0]
    index_eta_2=MSTWpdf[MSTWpdf["Parameter"] == "eta_2"].index[0]
    index_epsilon_u=MSTWpdf[MSTWpdf["Parameter"] == "epsilon_u"].index[0]
    index_gamma_u=MSTWpdf[MSTWpdf["Parameter"] == "gamma_u"].index[0]

    # Extracting parameter values based on the error_type argument
    A_u = MSTWpdf_LO.iloc[index_A_u,0][0] + int(error_col_index>0)*MSTWpdf_LO.iloc[index_A_u,0][error_col_index]
    eta_1 = MSTWpdf_LO.iloc[index_eta_1,0][0] + int(error_col_index>0)*MSTWpdf_LO.iloc[index_eta_1,0][error_col_index]
    eta_2 = MSTWpdf_LO.iloc[index_eta_2,0][0] + int(error_col_index>0)*MSTWpdf_LO.iloc[index_eta_2,0][error_col_index]
    epsilon_u = MSTWpdf_LO.iloc[index_epsilon_u,0][0] + int(error_col_index>0)*MSTWpdf_LO.iloc[index_epsilon_u,0][error_col_index]
    gamma_u = MSTWpdf_LO.iloc[index_gamma_u,0][0] + int(error_col_index>0)*MSTWpdf_LO.iloc[index_gamma_u,0][error_col_index]
    
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

def int_dv_Regge(j,eta,alpha_p,t, error_type="central"):
    """
    Compute the integral of the Reggeized dv(x) PDF based on the given LO parameters and selected errors.
    
    Arguments:
    j -- conformal spin,
    eta -- skewness (scalar or array)(placeholder for now),
    alpha_p -- Regge slope,
    t -- Mandelstam t (scalar or array),
    error_type -- A string indicating whether to use 'central', 'plus', or 'minus' errors. Default is 'central'.
    
    Returns:
    The value of the Reggeized integral of dv(x) based on the selected parameters and error type.
    """
    # Define a dictionary that maps the error_type to column indices
    error_mapping = {
        "central": 0,  # The column with the central value
        "plus": 1,     # The column with the + error value
        "minus": 2     # The column with the - error value
    }
    
    # Get the column index corresponding to the error_type
    error_col_index = error_mapping.get(error_type, 0)

    # Get row index of entry
    index_A_d = MSTWpdf[MSTWpdf["Parameter"] == "A_d"].index[0]
    index_eta_3 = MSTWpdf[MSTWpdf["Parameter"] == "eta_3"].index[0]
    index_eta_2=MSTWpdf[MSTWpdf["Parameter"] == "eta_2"].index[0]
    # Only eta_4-eta_2 given
    index_eta_42 = MSTWpdf[MSTWpdf["Parameter"] == "eta_4-eta_2"].index[0]
    index_epsilon_d = MSTWpdf[MSTWpdf["Parameter"] == "epsilon_d"].index[0]
    index_gamma_d = MSTWpdf[MSTWpdf["Parameter"] == "gamma_d"].index[0]

    # Extracting parameter values based on the error_type argument
    A_d = MSTWpdf_LO.iloc[index_A_d, 0][0] + int(error_col_index>0) * MSTWpdf_LO.iloc[index_A_d, 0][error_col_index]
    eta_3 = MSTWpdf_LO.iloc[index_eta_3, 0][0] + int(error_col_index>0) * MSTWpdf_LO.iloc[index_eta_3, 0][error_col_index]
    # eta_4=(eta_4-eta_2) + eta_2, Add errors in quadrature
    eta_4 = (MSTWpdf_LO.iloc[index_eta_42, 0][0] + MSTWpdf_LO.iloc[index_eta_2, 0][0]) + int(error_col_index>0) *np.sign(MSTWpdf_LO.iloc[index_eta_42, 0][error_col_index]) * np.sqrt(MSTWpdf_LO.iloc[index_eta_42, 0][error_col_index]**2+MSTWpdf_LO.iloc[index_eta_2, 0][error_col_index]**2)
    epsilon_d = MSTWpdf_LO.iloc[index_epsilon_d, 0][0] + int(error_col_index>0) * MSTWpdf_LO.iloc[index_epsilon_d, 0][error_col_index]
    gamma_d = MSTWpdf_LO.iloc[index_gamma_d, 0][0] + int(error_col_index>0) * MSTWpdf_LO.iloc[index_gamma_d, 0][error_col_index]
    
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

def int_sv_Regge(j,eta,alpha_p,t, error_type="central"):
    """
    Compute the integral of the Reggeized sv(x) PDF based on the given LO parameters and selected errors.
    
    Arguments:
    j -- conformal spin,
    eta -- skewness (scalar or array)(placeholder for now),
    alpha_p -- Regge slope,
    t -- Mandelstam t (scalar or array),
    error_type -- A string indicating whether to use 'central', 'plus', or 'minus' errors. Default is 'central'.
    
    Returns:
    The value of the Reggeized integral of sv(x) based on the selected parameters and error type.
    """
    error_mapping = {
        "central": 0,
        "plus": 1,
        "minus": 2
    }
    
    error_col_index = error_mapping.get(error_type, 0)

    # delta_- fixed to 0.2
    index_A_m = MSTWpdf[MSTWpdf["Parameter"] == "A_-"].index[0]
    index_eta_m = MSTWpdf[MSTWpdf["Parameter"] == "eta_-"].index[0]
    index_x_0 = MSTWpdf[MSTWpdf["Parameter"] == "x_0"].index[0]

    A_m = MSTWpdf_LO.iloc[index_A_m, 0][0] + int(error_col_index>0) * MSTWpdf_LO.iloc[index_A_m, 0][error_col_index]
    delta_m = .2
    eta_m = MSTWpdf_LO.iloc[index_eta_m, 0][0] + int(error_col_index>0) * MSTWpdf_LO.iloc[index_eta_m, 0][error_col_index]
    x_0 = MSTWpdf_LO.iloc[index_x_0, 0][0] + int(error_col_index>0) * MSTWpdf_LO.iloc[index_x_0, 0][error_col_index]
    
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

def int_Sv_Regge(j,eta,alpha_p,t, error_type="central"):
    """
    Compute the integral of the Reggeized Sv(x) PDF based on the given LO parameters and selected errors.
    
    Arguments:
    j -- conformal spin,
    eta -- skewness (scalar or array)(placeholder for now),
    alpha_p -- Regge slope,
    t -- Mandelstam t (scalar or array),
    error_type -- A string indicating whether to use 'central', 'plus', or 'minus' errors. Default is 'central'.
    
    Returns:
    The value of the Reggeized integral of Sv(x) based on the selected parameters and error type.
    """
    error_mapping = {
        "central": 0,
        "plus": 1,
        "minus": 2
    }
    
    error_col_index = error_mapping.get(error_type, 0)

    index_A_S = MSTWpdf[MSTWpdf["Parameter"] == "A_S"].index[0]
    index_delta_S = MSTWpdf[MSTWpdf["Parameter"] == "delta_S"].index[0]
    index_eta_S = MSTWpdf[MSTWpdf["Parameter"] == "eta_S"].index[0]
    index_epsilon_S = MSTWpdf[MSTWpdf["Parameter"] == "epsilon_S"].index[0]
    index_gamma_S = MSTWpdf[MSTWpdf["Parameter"] == "gamma_S"].index[0]

    A_S = MSTWpdf_LO.iloc[index_A_S, 0][0] + int(error_col_index>0) * MSTWpdf_LO.iloc[index_A_S, 0][error_col_index]
    delta_S = MSTWpdf_LO.iloc[index_delta_S, 0][0] + int(error_col_index>0) * MSTWpdf_LO.iloc[index_delta_S, 0][error_col_index]
    eta_S = MSTWpdf_LO.iloc[index_eta_S, 0][0] + int(error_col_index>0) * MSTWpdf_LO.iloc[index_eta_S, 0][error_col_index]
    epsilon_S = MSTWpdf_LO.iloc[index_epsilon_S, 0][0] + int(error_col_index>0) * MSTWpdf_LO.iloc[index_epsilon_S, 0][error_col_index]
    gamma_S = MSTWpdf_LO.iloc[index_gamma_S, 0][0] + int(error_col_index>0) * MSTWpdf_LO.iloc[index_gamma_S, 0][error_col_index]
    
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
def int_s_plus_Regge(j,eta,alpha_p,t, error_type="central"):
    """
    Compute the integral of the Reggeized s_+(x) PDF based on the given LO parameters and selected errors.
    
    Arguments:
    j -- conformal spin,
    eta -- skewness (scalar or array)(placeholder for now),
    alpha_p -- Regge slope,
    t -- Mandelstam t (scalar or array),
    error_type -- A string indicating whether to use 'central', 'plus', or 'minus' errors. Default is 'central'.
    
    Returns:
    The value of the Reggeized integral of s_+(x) based on the selected parameters and error type.
    """
    error_mapping = {
        "central": 0,
        "plus": 1,
        "minus": 2
    }
    
    error_col_index = error_mapping.get(error_type, 0)

    index_A_p = MSTWpdf[MSTWpdf["Parameter"] == "A_+"].index[0]
    index_delta_S = MSTWpdf[MSTWpdf["Parameter"] == "delta_S"].index[0]
    index_eta_p = MSTWpdf[MSTWpdf["Parameter"] == "eta_+"].index[0]
    index_epsilon_S = MSTWpdf[MSTWpdf["Parameter"] == "epsilon_S"].index[0]
    index_gamma_S = MSTWpdf[MSTWpdf["Parameter"] == "gamma_S"].index[0]

    A_p = MSTWpdf_LO.iloc[index_A_p, 0][0] + int(error_col_index>0) * MSTWpdf_LO.iloc[index_A_p, 0][error_col_index]
    delta_S = MSTWpdf_LO.iloc[index_delta_S, 0][0] + int(error_col_index>0) * MSTWpdf_LO.iloc[index_delta_S, 0][error_col_index]
    eta_p = MSTWpdf_LO.iloc[index_eta_p, 0][0] + int(error_col_index>0) * MSTWpdf_LO.iloc[index_eta_p, 0][error_col_index]
    epsilon_S = MSTWpdf_LO.iloc[index_epsilon_S, 0][0] + int(error_col_index>0) * MSTWpdf_LO.iloc[index_epsilon_S, 0][error_col_index]
    gamma_S = MSTWpdf_LO.iloc[index_gamma_S, 0][0] + int(error_col_index>0) * MSTWpdf_LO.iloc[index_gamma_S, 0][error_col_index]

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

def int_Delta_Regge(j,eta,alpha_p,t, error_type="central"):
    """
    Compute the integral of the Reggeized Delta(x)=ubar(x)-dbar(x) PDF based on the given LO parameters and selected errors.
    
    Arguments:
    j -- conformal spin,
    eta -- skewness (scalar or array)(placeholder for now),
    alpha_p -- Regge slope,
    t -- Mandelstam t (scalar or array),
    error_type -- A string indicating whether to use 'central', 'plus', or 'minus' errors. Default is 'central'.
    
    Returns:
    The value of the Reggeized integral of Delta(x) based on the selected parameters and error type.
    """
     # Define a dictionary that maps the error_type to column indices
    error_mapping = {
        "central": 0,  # The column with the central value
        "plus": 1,     # The column with the + error value
        "minus": 2     # The column with the - error value
    }
    
    # Get the column index corresponding to the error_type
    error_col_index = error_mapping.get(error_type, 0)  # Default to 'central' if error_type is invalid

    # Get row index of entry
    index_A_Delta=MSTWpdf[MSTWpdf["Parameter"] == "A_Delta"].index[0]
    index_eta_Delta=MSTWpdf[MSTWpdf["Parameter"] == "eta_Delta"].index[0]
    index_eta_S=MSTWpdf[MSTWpdf["Parameter"] == "eta_S"].index[0]
    index_gamma_Delta=MSTWpdf[MSTWpdf["Parameter"] == "gamma_Delta"].index[0]
    index_delta_Delta=MSTWpdf[MSTWpdf["Parameter"] == "delta_Delta"].index[0]

    # Extracting parameter values based on the error_type argument
    A_Delta = MSTWpdf_LO.iloc[index_A_Delta,0][0] + int(error_col_index>0)*MSTWpdf_LO.iloc[index_A_Delta,0][error_col_index]
    eta_Delta = MSTWpdf_LO.iloc[index_eta_Delta,0][0] + int(error_col_index>0)*MSTWpdf_LO.iloc[index_eta_Delta,0][error_col_index]
    eta_S = MSTWpdf_LO.iloc[index_eta_S,0][0] + int(error_col_index>0)*MSTWpdf_LO.iloc[index_eta_S,0][error_col_index]
    gamma_Delta = MSTWpdf_LO.iloc[index_gamma_Delta,0][0] + int(error_col_index>0)*MSTWpdf_LO.iloc[index_gamma_Delta,0][error_col_index]
    delta_Delta = MSTWpdf_LO.iloc[index_delta_Delta,0][0] + int(error_col_index>0)*MSTWpdf_LO.iloc[index_delta_Delta,0][error_col_index]

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

def int_gv_Regge(j,eta,alpha_p,t, error_type="central"):
    """
    Compute the integral of the Reggeized g(x) PDF based on the given LO parameters and selected errors.
    
    Arguments:
    j -- conformal spin,
    eta -- skewness (scalar or array)(placeholder for now),
    alpha_p -- Regge slope,
    t -- Mandelstam t (scalar or array),
    error_type -- A string indicating whether to use 'central', 'plus', or 'minus' errors. Default is 'central'.
    
    Returns:
    The value of the Reggeized integral of g(x) based on the selected parameters and error type.
    """
     # Define a dictionary that maps the error_type to column indices
    error_mapping = {
        "central": 0,  # The column with the central value
        "plus": 1,     # The column with the + error value
        "minus": 2     # The column with the - error value
    }
    
    # Get the column index corresponding to the error_type
    error_col_index = error_mapping.get(error_type, 0)  # Default to 'central' if error_type is invalid

    # Get row index of entry
    index_A_g=MSTWpdf[MSTWpdf["Parameter"] == "A_g"].index[0]
    index_delta_g=MSTWpdf[MSTWpdf["Parameter"] == "delta_g"].index[0]
    index_eta_g=MSTWpdf[MSTWpdf["Parameter"] == "eta_g"].index[0]
    index_epsilon_g=MSTWpdf[MSTWpdf["Parameter"] == "epsilon_g"].index[0]
    index_gamma_g=MSTWpdf[MSTWpdf["Parameter"] == "gamma_g"].index[0]

    # Extracting parameter values based on the error_type argument
    A_g = MSTWpdf_LO.iloc[index_A_g,0][0] + int(error_col_index>0)*MSTWpdf_LO.iloc[index_A_g,0][error_col_index]
    delta_g = MSTWpdf_LO.iloc[index_delta_g,0][0] + int(error_col_index>0)*MSTWpdf_LO.iloc[index_delta_g,0][error_col_index]
    eta_g = MSTWpdf_LO.iloc[index_eta_g,0][0] + int(error_col_index>0)*MSTWpdf_LO.iloc[index_eta_g,0][error_col_index]
    epsilon_g = MSTWpdf_LO.iloc[index_epsilon_g,0][0] + int(error_col_index>0)*MSTWpdf_LO.iloc[index_epsilon_g,0][error_col_index]
    gamma_g = MSTWpdf_LO.iloc[index_gamma_g,0][0] + int(error_col_index>0)*MSTWpdf_LO.iloc[index_gamma_g,0][error_col_index]

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
     # Return the result while preserving the original dimensions
    if result.size == 1:
        return result.item()  # Return a scalar if the result is a single value
    return result

# Define Reggeized conformal moments
def uv_minus_dv_Regge(j,eta,t, error_type="central"):
   # Value from the paper
   alpha_prime = 1.069
   # Value optmized for range -t < 5 GeV
   # alpha_prime = 0.650439
   # Normalize to 1 at t = 0
   #return 1.006*(uv(x,error_type)-dv(x,error_type))*x**(j-1-alpha_prime*t)
   return 1.006*(int_uv_Regge(j,eta,alpha_prime,t,error_type)-int_dv_Regge(j,eta,alpha_prime,t,error_type))

def u_minus_d_Regge(j,eta,t, error_type="central"):
   # Value optmized for range -t < 5 GeV
   alpha_prime = 0.675606
   # Normalize to 1 at t = 0
   #return 1.107*(uv(x,error_type)-dv(x,error_type)-Delta(x,error_type))*x**(j-1-alpha_prime*t)
   return 1.107*(int_uv_Regge(j,eta,alpha_prime,t,error_type)-int_dv_Regge(j,eta,alpha_prime,t,error_type)-int_Delta_Regge(j,alpha_prime,t,error_type))

def uv_plus_dv_Regge(j,eta,t, error_type="central"):
   # Value from the paper
   alpha_prime = 0.891
   # Value optmized for range -t < 5 GeV
   #alpha_prime = 0.953598
   # Normalize to 1 at t = 0
   #return 1.002*(uv(x,error_type)+dv(x,error_type))*x**(j-1-alpha_prime*t)
   return 1.002*(int_uv_Regge(j,eta,alpha_prime,t,error_type)+int_dv_Regge(j,eta,alpha_prime,t,error_type))

def u_plus_d_Regge(j,eta,t, error_type="central"):
   # Value optmized for range -t < 5 GeV
   alpha_prime = 0.949256
   # Normalize to 1 at t = 0
   #return 0.973*(uv(x,error_type)+dv(x,error_type)+Delta(x,error_type))*x**(j-1-alpha_prime*t)
   return 0.973*(int_uv_Regge(j,eta,alpha_prime,t,error_type)+int_dv_Regge(j,eta,alpha_prime,t,error_type)+int_Delta_Regge(j,alpha_prime,t,error_type))



def d_hat(j,eta,t):
    m_N = 0.93827
    return mp.hyp2f1(-j/2, -(j-1)/2, 1/2 - j, - 4 * m_N**2/t * eta**2)

def quark_singlet_Regge(j,eta,t, Nf=3, error_type="central"):
    alpha_prime_ud = 0.891
    alpha_prime_s = 1.828
    uv = int_uv_Regge(j,eta,alpha_prime_ud,t,error_type) 
    dv = int_dv_Regge(j,eta,alpha_prime_ud,t,error_type)
    Delta = int_Delta_Regge(j,eta,alpha_prime_ud,t,error_type)
    # Sv = int_Sv_Regge(j,eta,alpha_prime_s,t,error_type)
    Sv = int_Sv_Regge(j,eta,alpha_prime_ud,t,error_type)
    s_plus = int_s_plus_Regge(j,eta,alpha_prime_ud,t,error_type)

    uv_s = int_uv_Regge(j,eta,alpha_prime_s,t,error_type) 
    dv_s = int_dv_Regge(j,eta,alpha_prime_s,t,error_type)
    Sv_s = int_Sv_Regge(j,eta,alpha_prime_s,t,error_type)
    s_plus_s = int_s_plus_Regge(j,eta,alpha_prime_s,t,error_type)
    Delta_s = int_Delta_Regge(j,eta,alpha_prime_s,t,error_type)

    if Nf == 3:
        #term_1 = uv + dv + Sv 
        term_1 = uv + dv + Sv 
        term_2 = uv_s + dv_s + Sv_s 
        term_3 = (d_hat(j,eta,t)-1)*(term_1-term_2)
        return term_1 + eta**2*term_3
    elif Nf == 2:
        term_1 = uv + dv + Sv - s_plus
        term_2 = uv_s + dv_s + Sv_s - s_plus_s
        term_3 = (d_hat(j,eta,t)-1)*(term_1-term_2)
        return term_1 + eta**2*term_3
    elif Nf == 1:
        term_1 = .5*(Sv-s_plus+2*uv-2*Delta)
        term_2 = .5*(Sv_s-s_plus_s+2*uv_s-2*Delta_s)
        term_3 = (d_hat(j,eta,t)-1)*(term_1-term_2)
        return term_1 + eta**2*term_3
    else :
        raise ValueError("Currently only inte 1 <= Nf <= 3 supported")
    
def quark_singlet_Regge_A(j,eta,t, Nf=3, error_type="central"):
    alpha_prime_ud = 0.891
    uv = int_uv_Regge(j,eta,alpha_prime_ud,t,error_type) 
    dv = int_dv_Regge(j,eta,alpha_prime_ud,t,error_type)
    Delta = int_Delta_Regge(j,eta,alpha_prime_ud,t,error_type)
    Sv = int_Sv_Regge(j,eta,alpha_prime_ud,t,error_type)
    s_plus = int_s_plus_Regge(j,eta,alpha_prime_ud,t,error_type)
    if Nf == 3:
        #term_1 = uv + dv + Sv 
        term_1 = uv + dv + Sv 
        return term_1
    elif Nf == 2:
        term_1 = uv + dv + Sv - s_plus
        return term_1
    elif Nf == 1:
        term_1 = .5*(Sv-s_plus+2*uv-2*Delta)
        return term_1
    else :
        raise ValueError("Currently only integer 1 <= Nf <= 3 supported")
    
def quark_singlet_Regge_D(j,eta,t, Nf=3, error_type="central"):
    return (d_hat(j,eta,t)-1)/eta**2*(quark_singlet_Regge(j,eta,t,Nf,error_type)-quark_singlet_Regge_A(j,eta,t,Nf,error_type))
    
def g_Regge_A(j,eta,t, error_type="central"):
    alpha_prime_T = 0.627
    return int_gv_Regge(j,eta,alpha_prime_T,t,error_type)

def g_Regge_D(j,eta,t, error_type="central"):
    alpha_prime_S = 4.277
    return (d_hat(j,eta,t)-1)/eta**2*(int_gv_Regge(j,eta,alpha_prime_S,t,error_type)-g_Regge_A(j,eta,t,error_type))

def g_Regge(j,eta,t, error_type="central"):
    term_1= g_Regge_A(j,eta,t,error_type)
    term_2 = g_Regge_D(j,eta,t,error_type)
    return term_1+eta**2*term_2



# RGEs of moments
def gamma_qq(j):
   """"
   Return conformal spin-j anomalous dimension

   Arguments:
   j -- conformal spin
   """
   if j.real < 0:
    raise ValueError("j must be positive.")

   Nc = 3
   Cf = (Nc**2-1)/(2*Nc)
   result = - Cf * (-4*digamma(j+2)+4*digamma(1)+2/((j+1)*(j+2))+3)

   return result


def gamma_qg(j, Nf=3, evolve_type = "vector"):
    """
    Compute off-diagonal gq anomalous dimension
    Parameters:
    j -- conformal spin
    Nf -- Number of active flavors (default Nf = 3 )
    evolve_type -- "vector" or "axial"
    Returns:
    Value of anomalous dimension
    """
    Tf = 1/2
    if evolve_type == "vector":
        result = -24*Nf*Tf*(j**2+3*j+4)/(j*(j+1)*(j+2)*(j+3))
    elif evolve_type == "axial":
        result = -24*Nf*Tf/((j+1)*(j+2))
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
    Nc = 3
    Cf = (Nc**2-1)/(2*Nc)
    if evolve_type == "vector":
        result = -Cf*(j**2+3*j+4)/(3*(j+1)*(j+2))
    elif evolve_type == "axial":
        result = -Cf*j*(j+3)/(3*(j+1)*(j+2))
    else:
        raise ValueError("Type must be axial or vector")
    return result

def gamma_gg(j, Nf = 3, evolve_type = "vector"):
    """
    Compute off-diagonal gq anomalous dimension
    Parameters:
    j -- conformal spin
    evolve_type -- "vector" or "axial"
    Nf -- Number of active flavors (default Nf = 3 )
    Returns:
    Value of anomalous dimension
    """
    Nc = 3
    Ca = Nc
    beta_0 = 11/3 * Nc - 2/3* Nf
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
    base = gamma_qq(j)+gamma_gg(j,Nf,evolve_type)
    root = np.sqrt((gamma_qq(j)-gamma_gg(j,Nf,evolve_type))**2+4*gamma_gq(j,evolve_type)*gamma_qg(j,Nf,evolve_type))
    if solution == "+":
        return (base + root)/2
    elif solution == "-":
        return (base - root)/2
    else:
        raise ValueError("Invalid solution evolve_type. Use '+' or '-'.")

def Gamma_pm(j,Nf=3,evolve_type="vector",solution="+"):
    result = 1-gamma_pm(j,Nf,evolve_type,solution)/gamma_qq(j)
    return result

def evolve_conformal_moment(GPD_in,j,mu,Nf = 3,evolve_type="NonSinglet",solution="+"):
    """
    Evolve the conformal moment F_{j}^{+-} from some input scale mu_in to some other scale mu.
    Note that the MSTW best fit obtains alpha_S(mu=1 GeV**2)=0.68183, different from the world average

    Arguments:
    j -- conformal spin
    mu -- The momentum scale of the process
    Nf -- Number of active flavors (default Nf = 3)
    evolve_type -- Vector or Axial singlet conformal moment or any non-singlet conformal moment

    Returns:
    The evolved value of the conformal moment at mu
    """
    # Set parameters
    Nc = 3
    beta_0 = 11/3 * Nc - 2/3* Nf

    # Extract value of alpha_S at the renormalization point of mu_R**2 = 1 GeV**2
    index_alpha_S=MSTWpdf[MSTWpdf["Parameter"] == "alpha_S(Q0^2)"].index[0]
    alpha_s_in = MSTWpdf_LO.iloc[index_alpha_S,0][0]
        
    if evolve_type in ["vector","axial"]:
        anomalous_dim = gamma_pm(j,Nf,evolve_type,solution)
    elif evolve_type == "NonSinglet":
        anomalous_dim = gamma_qq(j)   
    else : 
        raise ValueError("Evolve type must be vector, axial or NonSinglet")
    result = GPD_in * (alpha_s_in/evolve_alpha_s(mu))**(anomalous_dim/beta_0)

    return result

initialize_dictionaries()

def evolve_quark_non_singlet(j,eta,t,mu,Nf=3,evolve_type="NonSingletIsovector",error_type="central"):
    GPD_in = moment_to_function.get((evolve_type, "A"), None)
    result = evolve_conformal_moment(GPD_in(j,eta,t, error_type),j-1,mu,Nf,"NonSinglet")
    return result

def plot_moments(moment_type, moment_label, y_label, t_max=3, n_t=50, num_columns=3):
    """
    Generates plots of lattice data and RGE-evolved functions for a given moment type and label.
    
    Parameters:
        moment_type (str): The type of moment (e.g., "NonSingletIsovector").
        moment_label (str): The label of the moment (e.g., "A").
        t_max (float, optional): Maximum t value for the x-axis (default is 3).
        n_t (int, optional): Number of points for t_fine (default is 50).
        num_columns (int, optional): Number of columns for the grid layout (default is 3).
    """

    # Define the finer grid for t-values
    t_fine = np.linspace(-t_max, 0, n_t)
    
    # Initialize a list to store the number of n values per publication
    publication_data = {}
    
    # Loop through each publication ID to calculate the total number of plots
    for pub_id in publication_mapping:
        data, n_to_row_map = load_lattice_data(moment_type, moment_label, pub_id)
        if data is None and n_to_row_map is None:
            #print(f"No data found for {pub_id}. Skipping.")
            continue
        num_n_values = (data.shape[1] - 1) // 2
        publication_data[pub_id] = num_n_values
    
    # Find the highest n value across all publications
    max_n_value = max(publication_data.values())
    
    # Calculate the number of rows needed for the grid layout
    num_rows = (max_n_value + num_columns - 1) // num_columns
    
    # Initialize the figure and axes for subplots
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_columns, figsize=(15, num_rows * 5))
    axes = axes.flatten()
    
    # Loop through each n value up to the maximum n value
    for n in range(1, max_n_value + 1):
        ax = axes[n - 1]  # Select the appropriate axis
        function = moment_to_function.get((moment_type, moment_label), None)
        
    #def evolve_quark_non_singlet(j,eta,t,mu,Nf=3,evolve_type="NonSingletIsovector",error_type="central"):
    # GPD_in = moment_to_function.get((evolve_type, "A"), None)
    # result = evolve_conformal_moment(GPD_in(j,eta,t, error_type),j-1,mu,Nf,"NonSinglet")
    # return result

        if function:
            x_RGE_evolved_moment = evolve_conformal_moment(
               function(n, 0, t_fine, error_type="central"), n-1, 2,evolve_type="NonSinglet"
            )[0, :]
            x_plus_RGE_evolved_moment = evolve_conformal_moment(
               function(n, 0, t_fine, error_type="plus"), n-1, 2,evolve_type="NonSinglet"
            )[ 0, :]
            x_minus_RGE_evolved_moment = evolve_conformal_moment(
               function(n, 0, t_fine, error_type="minus"), n-1, 2,evolve_type="NonSinglet"
            )[0, :]
        else:
            print(f"Function for {moment_type} and {moment_label} not found.")
            continue
        
        # Plot the RGE functions
        ax.plot(-t_fine, x_RGE_evolved_moment, color="blue", linewidth=2, label="This work")
        ax.fill_between(-t_fine, x_minus_RGE_evolved_moment, x_plus_RGE_evolved_moment, color="blue", alpha=0.2)
        
        # Plot data from publications
        for pub_id, color in publication_mapping.items():
            data, n_to_row_map = load_lattice_data(moment_type, moment_label, pub_id)
            if data is None or n not in n_to_row_map:
                continue
            t_vals = t_values(moment_type, moment_label, pub_id)
            Fn0_vals = Fn0_values(n, moment_type, moment_label, pub_id)
            Fn0_errs = Fn0_errors(n, moment_type, moment_label, pub_id)
            ax.errorbar(t_vals, Fn0_vals, yerr=Fn0_errs, fmt='o', color=color, label=f"{pub_id}")
        
        # Add labels and formatting
        ax.set_xlabel("$-t\,[\mathrm{GeV}^2]$", fontsize=14)
        ax.set_ylabel(f"{y_label}$(j={n}, \\eta=0, t, \\mu=2\, \\mathrm{{GeV}})$", fontsize=14)
        ax.legend()
        ax.grid(True, which="both")
        ax.set_xlim([0, t_max])
        #ax.set_ylim([0, np.max(x_plus_RGE_evolved_moment)])
    
    # Remove unused axes
    for i in range(max_n_value, len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    plt.show()


def ft_moment(moment_type, moment_label, n, b_vec, Delta_max = 11,num_points=100, error_type="central"):
    """
    Optimized calculation of Fourier transformed moments using adaptive integration.

    Parameters:
    - b_vec: (b_x, b_y), the vector for which to compute the result
    - Delta_max: maximum radius for the integration domain (limits the integration bounds)
    - num_points: number of points for discretizing the domain (adapt as needed)

    Returns:
    - The value of the Fourier transformed moment at (b_vec)
    """
    b_x, b_y = b_vec
    # Limits of integration for Delta_x, Delta_y on a square grid
    x_min, x_max = -Delta_max, Delta_max
    y_min, y_max = -Delta_max, Delta_max
    # Discretize the grid (vectorized)
    Delta_x_vals = np.linspace(x_min, x_max, num_points)
    Delta_y_vals = np.linspace(y_min, y_max, num_points)

    # Create a meshgrid for delta_x, delta_y
    Delta_x_grid, Delta_y_grid = np.meshgrid(Delta_x_vals, Delta_y_vals)

    #Get the function from the dictionary
    function = moment_to_function.get((moment_type, moment_label), None)
    if function:
        def integrand(Delta_x,Delta_y,b_x,b_y):
            t = -(Delta_x**2+Delta_y**2)
            exponent = -1j * (b_x * Delta_x + b_y * Delta_y)
            #print(f"t shape: {t.shape}")
            x_int_moment = function(n, 0, t, error_type)
            # Debug shapes
            #print(f"x_integrated_moment shape: {x_int_moment.shape}")
            result = evolve_conformal_moment(x_int_moment, n-1, 2,evolve_type="NonSinglet")*np.exp(exponent)/(2*np.pi**2)
            return result
        
        # Compute the integrand for each pair of (Delta_x, Delta_y) values
        integrand_values = integrand(Delta_x_grid, Delta_y_grid, b_x, b_y)
        # Perform the numerical integration using the trapezoidal rule for efficiency
        integral_result = np.real(trapezoid(trapezoid(integrand_values, Delta_x_vals, axis=0), Delta_y_vals))
        #integral_result = 1

        return integral_result
    else:
        print(f"Function for {moment_type} and {moment_label} not found.")
        return

def plot_ft_moments(moment_type, moment_label, plot_title,n=1, b_max = 2,Delta_max = 11,num_points=100):
    """
    Generates a density plot of the 2D Fourier transfrom of RGE-evolved 
    conformal moments for a given moment type and label.
    
    Parameters:
        moment_type (str): The type of moment (e.g., "NonSingletIsovector").
        moment_label (str): The label of the moment (e.g., "A").
        plot_title (str): The title of the plot.
        n_t (int, optional): Number of points for t_fine (default is 50).
        b_max (float, optional): Maximum b value for the vector b_vec=[b_x,b_y] (default is 2).
        Delta_max (float, optional): Maximum value for Delta integration (default is 11).
        num_points (float, optional): Number of intervals to split [-Delta_max, Delta_max] interval (default is 100).
    """
    # Define the grid for b_vec
    b_x = np.linspace(-b_max, b_max, 50)  # Range of x-component of b_vec
    b_y = np.linspace(-b_max, b_max, 50)  # Range of y-component of b_vec
    b_x, b_y = np.meshgrid(b_x, b_y)  # Create a grid of (b_x, b_y)
    # Flatten the grid for parallel processing
    b_vecs = np.array([b_x.ravel(), b_y.ravel()]).T

    # Parallel computation using joblib
    ft_moment_values_flat = Parallel(n_jobs=-1)(delayed(ft_moment)(moment_type, moment_label,n,b_vec,Delta_max,num_points) for b_vec in b_vecs)

    # Reshape the result back to the grid shape
    ft_moment_values = np.array(ft_moment_values_flat).reshape(b_x.shape)

    # Convert Gev^-1 to fm
    hbarc = 0.1975

    # Create the 2D density plot
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(b_x*hbarc, b_y*hbarc, ft_moment_values, shading='auto', cmap='viridis')
    plt.colorbar()
    plt.xlabel(r'$b_x\,[\mathrm{fm}]$', fontsize=14)
    plt.ylabel(r'$b_y\,[\mathrm{fm}]$', fontsize=14)
    plt.title(f"{plot_title}$(j={n}, \\eta=0, t, \\mu=2\, \\mathrm{{GeV}})$", fontsize=14)
    plt.show()
