# Dependencies
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt

# Set display options
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000)        # Set width to avoid wrapping
pd.set_option('display.max_colwidth', None) # Show full content of each column

############################################
############################################
# Import MSTW PDF data
# Base path to main data directory
BASE_PATH = "/mnt/c/Users/flori/Documents/PostDoc/Data/PDFs/"
# Define the file path to the .csv file and extract its content
MSTW_PATH = f"{BASE_PATH}MSTW.csv"

# Columns for the DataFrame
columns = ["Parameter", "LO", "NLO", "NNLO"]

# Read the CSV file and parse it
data = []
with open(MSTW_PATH, 'r',newline='') as file:
    next(file) # Skip header
    reader = csv.reader(file)  # Standard CSV reader
    
    # Read the first row (header) as a string, then parse the rest as lists
    for row in reader:
        parameter =row[0]  # The first column is the parameter (string)
        # Convert the string representation of arrays into actual lists
        lo_values = np.array([row[1],row[2],row[3]], dtype=float)
        nlo_values = np.array([row[4],row[5],row[6]], dtype=float)
        nnlo_values = np.array([row[7],row[8],row[9]], dtype=float)
        
        # Append the row as a list of data
        data.append([parameter, lo_values, nlo_values, nnlo_values])

# Create the pandas DataFrame
MSTW_PDF = pd.DataFrame(data, columns=columns)

# Create the pandas DataFrame
MSTW_PDF = pd.DataFrame(data, columns=columns)

# Extract LO, NLO, and NNLO columns
MSTW_PDF_LO = MSTW_PDF[["LO"]]
MSTW_PDF_NLO = MSTW_PDF[["NLO"]]
MSTW_PDF_NNLO = MSTW_PDF[["NNLO"]]

# Create a DataFrame from the parsed data
MSTW_PDF = pd.DataFrame(data, columns=columns)

# Extract LO, NLO, and NNLO columns
MSTW_PDF_LO = MSTW_PDF[["LO"]]
MSTW_PDF_NLO = MSTW_PDF[["NLO"]]
MSTW_PDF_NNLO = MSTW_PDF[["NNLO"]]

# Helpers:
def check_error_type(error_type):
    if error_type not in ["central","plus","minus"]:
        raise ValueError("error_type must be central, plus or minus")

############################################
############################################
def pdf(x,A_pdf,eta_1,eta_2,epsilon,gamma_pdf):
    """
    PDF parametrization for uv, dv, S, g. Note that at NLO the gluon parametrization gets aditional
    terms. This is handled separately in the gluon PDF:
    """
    result = A_pdf * (1-x)**eta_2*x**(eta_1-1)*(1+epsilon*np.sqrt(x)+gamma_pdf*x)
    # print(A_pdf,eta_1,eta_2,epsilon,gamma_pdf)
    return result

def pdf_error(x,A_pdf,delta_A_pdf,eta_1,delta_eta_1,eta_2,delta_eta_2,
              epsilon,delta_epsilon,gamma_pdf,delta_gamma_pdf,error_type):
    dpdf_dA_pdf = (1-x)**eta_2*x**(eta_1-1)*(1+epsilon*np.sqrt(x)+gamma_pdf * x)
    dpdf_deta_1 = A_pdf*(1-x)**eta_2*x**(eta_1-1)*(1+epsilon*np.sqrt(x)+gamma_pdf * x)*np.log(x)
    dpdf_deta_2 = A_pdf*(1-x)**eta_2*x**(eta_1-1)*(1+epsilon*np.sqrt(x)+gamma_pdf * x)*np.log(1-x)
    dpdf_depsilon = A_pdf*(1-x)**eta_2*x**(eta_1-.5)
    dpdf_dgamma = A_pdf*(1-x)**eta_2*x**(eta_1)
    
    # print(dpdf_dA_pdf,dpdf_deta_1,dpdf_deta_2,dpdf_depsilon,dpdf_dgamma)

    check_error_type(error_type)
    if error_type == "central":
            return 0

    Delta_A_pdf = dpdf_dA_pdf * delta_A_pdf
    Delta_eta_1 = dpdf_deta_1 * delta_eta_1
    Delta_eta_2 = dpdf_deta_2 * delta_eta_2
    Delta_epsilon = dpdf_depsilon * delta_epsilon
    Delta_gamma_pdf = dpdf_dgamma * delta_gamma_pdf

    result = np.sqrt(Delta_A_pdf**2+Delta_eta_1**2+Delta_eta_2**2+Delta_epsilon**2 + Delta_gamma_pdf**2)
    return result

# Define the PDFs using Eqs. (6-12) in  0901.0002 
def uv_pdf(x, evolution_order="LO",error_type="central"):
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
    check_error_type(error_type)
    # Get the column index corresponding to the error_type
    error_col_index = error_mapping.get(error_type, 0)  # Default to 'central' if error_type is invalid

    # Get row index of entry
    index_A_u=MSTW_PDF[MSTW_PDF["Parameter"] == "A_u"].index[0]
    index_eta_1=MSTW_PDF[MSTW_PDF["Parameter"] == "eta_1"].index[0]
    index_eta_2=MSTW_PDF[MSTW_PDF["Parameter"] == "eta_2"].index[0]
    index_epsilon_u=MSTW_PDF[MSTW_PDF["Parameter"] == "epsilon_u"].index[0]
    index_gamma_u=MSTW_PDF[MSTW_PDF["Parameter"] == "gamma_u"].index[0]

    # Extracting parameter values
    A_pdf = MSTW_PDF[[evolution_order]].iloc[index_A_u,0][0]
    eta_1 = MSTW_PDF[[evolution_order]].iloc[index_eta_1,0][0]
    eta_2 = MSTW_PDF[[evolution_order]].iloc[index_eta_2,0][0]
    epsilon = MSTW_PDF[[evolution_order]].iloc[index_epsilon_u,0][0]
    gamma_pdf = MSTW_PDF[[evolution_order]].iloc[index_gamma_u,0][0]

    if error_type == "central":
        result = pdf(x,A_pdf,eta_1,eta_2,epsilon,gamma_pdf)
    else:
    # Extracting errors
        delta_A_pdf  = MSTW_PDF[[evolution_order]].iloc[index_A_u,0][error_col_index]
        delta_eta_1 = MSTW_PDF[[evolution_order]].iloc[index_eta_1,0][error_col_index]
        delta_eta_2 = MSTW_PDF[[evolution_order]].iloc[index_eta_2,0][error_col_index]
        delta_epsilon = MSTW_PDF[[evolution_order]].iloc[index_epsilon_u,0][error_col_index]
        delta_gamma_pdf = MSTW_PDF[[evolution_order]].iloc[index_gamma_u,0][error_col_index]

        
        result = pdf_error(x,A_pdf,delta_A_pdf,eta_1,delta_eta_1,eta_2,delta_eta_2,epsilon,delta_epsilon,gamma_pdf,delta_gamma_pdf,error_type)
    return result

def dv_pdf(x, evolution_order="LO",error_type="central"):
    # Define a dictionary that maps the error_type to column indices
    error_mapping = {
        "central": 0,  # The column with the central value
        "plus": 1,     # The column with the + error value
        "minus": 2     # The column with the - error value
    }
    check_error_type(error_type)
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

    # Extracting parameter values
    A_pdf = MSTW_PDF[[evolution_order]].iloc[index_A_d,0][0]
    eta_1 = MSTW_PDF[[evolution_order]].iloc[index_eta_3,0][0]
    eta_2 = MSTW_PDF[[evolution_order]].iloc[index_eta_42, 0][0] + MSTW_PDF[[evolution_order]].iloc[index_eta_2, 0][0]
    epsilon = MSTW_PDF[[evolution_order]].iloc[index_epsilon_d,0][0]
    gamma_pdf = MSTW_PDF[[evolution_order]].iloc[index_gamma_d,0][0]

    if error_type == "central":
        result = pdf(x,A_pdf,eta_1,eta_2,epsilon,gamma_pdf)
    else:
        # Extracting errors
        delta_A_pdf  = MSTW_PDF[[evolution_order]].iloc[index_A_d,0][error_col_index]
        delta_eta_1 = MSTW_PDF[[evolution_order]].iloc[index_eta_3,0][error_col_index]
        delta_eta_2 = np.sign(MSTW_PDF[[evolution_order]].iloc[index_eta_42, 0][error_col_index]) * np.sqrt(MSTW_PDF[[evolution_order]].iloc[index_eta_42, 0][error_col_index]**2+MSTW_PDF[[evolution_order]].iloc[index_eta_2, 0][error_col_index]**2)
        delta_epsilon = MSTW_PDF[[evolution_order]].iloc[index_epsilon_d,0][error_col_index]
        delta_gamma_pdf = MSTW_PDF[[evolution_order]].iloc[index_gamma_d,0][error_col_index]

        result = pdf_error(x,A_pdf,delta_A_pdf,eta_1,delta_eta_1,eta_2,delta_eta_2,epsilon,delta_epsilon,gamma_pdf,delta_gamma_pdf,error_type)
    return result

def sv_pdf(x, evolution_order="LO",error_type="central"):
    error_mapping = {
        "central": 0,
        "plus": 1,
        "minus": 2
    }
    check_error_type(error_type)
    error_col_index = error_mapping.get(error_type, 0)

    index_A_m = MSTW_PDF[MSTW_PDF["Parameter"] == "A_-"].index[0]
    index_delta_m = MSTW_PDF[MSTW_PDF["Parameter"] == "delta_-"].index[0]
    index_eta_m = MSTW_PDF[MSTW_PDF["Parameter"] == "eta_-"].index[0]
    index_x_0 = MSTW_PDF[MSTW_PDF["Parameter"] == "x_0"].index[0]

    A_m = MSTW_PDF[[evolution_order]].iloc[index_A_m, 0][0]
    delta_m = MSTW_PDF[[evolution_order]].iloc[index_delta_m, 0][0] 
    eta_m = MSTW_PDF[[evolution_order]].iloc[index_eta_m, 0][0] 
    x_0 = MSTW_PDF[[evolution_order]].iloc[index_x_0, 0][0]

    if error_type == "central":
        result = result = A_m * (x ** (delta_m - 1)) * ((1 - x) ** eta_m) * (1 -x/x_0)
    else:
        # Extracting errors
        delta_A_m  = MSTW_PDF[[evolution_order]].iloc[index_A_m,0][error_col_index]
        delta_delta_m = MSTW_PDF[[evolution_order]].iloc[index_delta_m, 0][error_col_index]
        delta_eta_m = MSTW_PDF[[evolution_order]].iloc[index_eta_m,0][error_col_index]
        delta_x_0 = MSTW_PDF[[evolution_order]].iloc[index_x_0,0][error_col_index]

        dpdf_dA_m = (x ** (delta_m - 1)) * ((1 - x) ** eta_m) * (1 -x/x_0)
        dpdf_ddelta_m =  A_m * (x ** (delta_m - 1)) * ((1 - x) ** eta_m) * (1 -x/x_0) * np.log(x)
        dpdf_deta_m =  A_m * (x ** (delta_m - 1)) * ((1 - x) ** eta_m) * (1 -x/x_0)*np.log(1-x)
        dpdf_dx_0 =  A_m * (x ** (delta_m)) * ((1 - x) ** eta_m) / x_0**2

        Delta_A_m = dpdf_dA_m * delta_A_m
        Delta_delta_m = dpdf_ddelta_m * delta_delta_m
        delta_eta_m = dpdf_deta_m * delta_eta_m
        Delta_x_0 = dpdf_dx_0 * delta_x_0

        result =  np.sqrt(Delta_A_m**2+Delta_delta_m**2+delta_eta_m**2+Delta_x_0**2)

    return result

def S_pdf(x, evolution_order="LO",error_type="central"):
    error_mapping = {
        "central": 0,
        "plus": 1,
        "minus": 2
    }
    check_error_type(error_type)
    error_col_index = error_mapping.get(error_type, 0)

    index_A_S = MSTW_PDF[MSTW_PDF["Parameter"] == "A_S"].index[0]
    index_delta_S = MSTW_PDF[MSTW_PDF["Parameter"] == "delta_S"].index[0]
    index_eta_S = MSTW_PDF[MSTW_PDF["Parameter"] == "eta_S"].index[0]
    index_epsilon_S = MSTW_PDF[MSTW_PDF["Parameter"] == "epsilon_S"].index[0]
    index_gamma_S = MSTW_PDF[MSTW_PDF["Parameter"] == "gamma_S"].index[0]

    # Extracting parameter values
    A_pdf = MSTW_PDF[[evolution_order]].iloc[index_A_S,0][0]
    eta_1 = MSTW_PDF[[evolution_order]].iloc[index_delta_S,0][0]
    eta_2 = MSTW_PDF[[evolution_order]].iloc[index_eta_S,0][0]
    epsilon = MSTW_PDF[[evolution_order]].iloc[index_epsilon_S,0][0]
    gamma_pdf = MSTW_PDF[[evolution_order]].iloc[index_gamma_S,0][0]

    if error_type == "central":
        result = pdf(x,A_pdf,eta_1,eta_2,epsilon,gamma_pdf)
    else:
    # Extracting errors
        delta_A_pdf  = MSTW_PDF[[evolution_order]].iloc[index_A_S,0][error_col_index]
        delta_eta_1 = MSTW_PDF[[evolution_order]].iloc[index_delta_S,0][error_col_index]
        delta_eta_2 = MSTW_PDF[[evolution_order]].iloc[index_eta_S,0][error_col_index]
        delta_epsilon = MSTW_PDF[[evolution_order]].iloc[index_epsilon_S,0][error_col_index]
        delta_gamma_pdf = MSTW_PDF[[evolution_order]].iloc[index_gamma_S,0][error_col_index]

        result = pdf_error(x,A_pdf,delta_A_pdf,eta_1,delta_eta_1,eta_2,delta_eta_2,epsilon,delta_epsilon,gamma_pdf,delta_gamma_pdf,error_type)

    return result

def s_plus_pdf(x, evolution_order="LO",error_type="central"):
    error_mapping = {
        "central": 0,
        "plus": 1,
        "minus": 2
    }
    check_error_type(error_type)
    error_col_index = error_mapping.get(error_type, 0)

    index_A_p = MSTW_PDF[MSTW_PDF["Parameter"] == "A_+"].index[0]
    index_delta_S = MSTW_PDF[MSTW_PDF["Parameter"] == "delta_S"].index[0]
    index_eta_p = MSTW_PDF[MSTW_PDF["Parameter"] == "eta_+"].index[0]
    index_epsilon_S = MSTW_PDF[MSTW_PDF["Parameter"] == "epsilon_S"].index[0]
    index_gamma_S = MSTW_PDF[MSTW_PDF["Parameter"] == "gamma_S"].index[0]

    # Extracting parameter values
    A_pdf = MSTW_PDF[[evolution_order]].iloc[index_A_p,0][0]
    eta_1 = MSTW_PDF[[evolution_order]].iloc[index_delta_S,0][0]
    eta_2 = MSTW_PDF[[evolution_order]].iloc[index_eta_p,0][0]
    epsilon = MSTW_PDF[[evolution_order]].iloc[index_epsilon_S,0][0]
    gamma_pdf = MSTW_PDF[[evolution_order]].iloc[index_gamma_S,0][0]

    if error_type == "central":
        result = pdf(x,A_pdf,eta_1,eta_2,epsilon,gamma_pdf)
    else:
        # Extracting errors
        delta_A_pdf  = MSTW_PDF[[evolution_order]].iloc[index_A_p,0][error_col_index]
        delta_eta_1 = MSTW_PDF[[evolution_order]].iloc[index_delta_S,0][error_col_index]
        delta_eta_2 = MSTW_PDF[[evolution_order]].iloc[index_eta_p,0][error_col_index]
        delta_epsilon = MSTW_PDF[[evolution_order]].iloc[index_epsilon_S,0][error_col_index]
        delta_gamma_pdf = MSTW_PDF[[evolution_order]].iloc[index_gamma_S,0][error_col_index]


        result = pdf_error(x,A_pdf,delta_A_pdf,eta_1,delta_eta_1,eta_2,delta_eta_2,epsilon,delta_epsilon,gamma_pdf,delta_gamma_pdf,error_type)

    return result

def Delta_pdf(x, evolution_order="LO",error_type="central"):
    """
    Compute the Delta(x)=dbar-ubar PDF based on the given LO parameters and selected errors.
    """
     # Define a dictionary that maps the error_type to column indices
    error_mapping = {
        "central": 0,  # The column with the central value
        "plus": 1,     # The column with the + error value
        "minus": 2     # The column with the - error value
    }
    check_error_type(error_type)
    # Get the column index corresponding to the error_type
    error_col_index = error_mapping.get(error_type, 0)  # Default to 'central' if error_type is invalid

    # Get row index of entry
    index_A_Delta=MSTW_PDF[MSTW_PDF["Parameter"] == "A_Delta"].index[0]
    index_eta_Delta=MSTW_PDF[MSTW_PDF["Parameter"] == "eta_Delta"].index[0]
    index_eta_S=MSTW_PDF[MSTW_PDF["Parameter"] == "eta_S"].index[0]
    index_gamma_Delta=MSTW_PDF[MSTW_PDF["Parameter"] == "gamma_Delta"].index[0]
    index_delta_Delta=MSTW_PDF[MSTW_PDF["Parameter"] == "delta_Delta"].index[0]

    # Extracting parameter values based on the error_type argument
    A_Delta = MSTW_PDF[[evolution_order]].iloc[index_A_Delta,0][0] 
    eta_Delta = MSTW_PDF[[evolution_order]].iloc[index_eta_Delta,0][0] 
    eta_S = MSTW_PDF[[evolution_order]].iloc[index_eta_S,0][0] 
    gamma_Delta = MSTW_PDF[[evolution_order]].iloc[index_gamma_Delta,0][0] 
    delta_Delta = MSTW_PDF[[evolution_order]].iloc[index_delta_Delta,0][0]
    # Compute the Delta(x) PDF
    if error_type == "central":
        result = A_Delta * (x ** (eta_Delta - 1)) * (1 - x) ** (eta_S+2) * (1 + gamma_Delta*x + delta_Delta*x**2)
    else:
        dpdf_dA = (x ** (eta_Delta - 1)) * (1 - x) ** (eta_S+2) * (1 + gamma_Delta*x + delta_Delta*x**2)
        dpdf_deta_Delta = A_Delta * (x ** (eta_Delta - 1)) * (1 - x) ** (eta_S+2) * (1 + gamma_Delta*x + delta_Delta*x**2)*np.log(x)
        dpdf_deta_S = A_Delta * (x ** (eta_Delta - 1)) * (1 - x) ** (eta_S+2) * (1 + gamma_Delta*x + delta_Delta*x**2)*np.log(1-x)
        dpdf_dgamma_Delta = A_Delta * (x ** (eta_Delta)) * (1 - x) ** (eta_S+2)
        dpdf_ddeltaDelta = A_Delta * (x ** (eta_Delta + 1)) * (1 - x) ** (eta_S+2) 
        Delta_A = dpdf_dA * MSTW_PDF[[evolution_order]].iloc[index_A_Delta,0][error_col_index]
        Delta_eta_Delta =  dpdf_deta_Delta *  MSTW_PDF[[evolution_order]].iloc[index_eta_Delta,0][error_col_index]
        Delta_eta_S = dpdf_deta_S * MSTW_PDF[[evolution_order]].iloc[index_eta_S,0][error_col_index]
        Delta_gamma_Delta = dpdf_dgamma_Delta * MSTW_PDF[[evolution_order]].iloc[index_gamma_Delta,0][error_col_index]
        Delta_delta_Delta = dpdf_ddeltaDelta * MSTW_PDF[[evolution_order]].iloc[index_delta_Delta,0][error_col_index]
        result = np.sqrt(Delta_A**2 + Delta_delta_Delta**2 + Delta_eta_Delta**2 +Delta_eta_S**2+Delta_gamma_Delta**2 +Delta_delta_Delta**2)
    
    return result

def gluon_pdf(x, evolution_order="LO",error_type="central"):
     # Define a dictionary that maps the error_type to column indices
    error_mapping = {
        "central": 0,  # The column with the central value
        "plus": 1,     # The column with the + error value
        "minus": 2     # The column with the - error value
    }
    check_error_type(error_type)
    # Get the column index corresponding to the error_type
    error_col_index = error_mapping.get(error_type, 0)  # Default to 'central' if error_type is invalid

    # Get row index of entry
    index_A_g=MSTW_PDF[MSTW_PDF["Parameter"] == "A_g"].index[0]
    index_delta_g=MSTW_PDF[MSTW_PDF["Parameter"] == "delta_g"].index[0]
    index_eta_g=MSTW_PDF[MSTW_PDF["Parameter"] == "eta_g"].index[0]
    index_epsilon_g=MSTW_PDF[MSTW_PDF["Parameter"] == "epsilon_g"].index[0]
    index_gamma_g=MSTW_PDF[MSTW_PDF["Parameter"] == "gamma_g"].index[0]

    # Get row index of entry
    index_A_g_prime=MSTW_PDF[MSTW_PDF["Parameter"] == "A_g'"].index[0]
    index_delta_g_prime=MSTW_PDF[MSTW_PDF["Parameter"] == "delta_g'"].index[0]
    index_eta_g_prime=MSTW_PDF[MSTW_PDF["Parameter"] == "eta_g'"].index[0]

    # Extracting parameter values
    A_pdf = MSTW_PDF[[evolution_order]].iloc[index_A_g,0][0]
    eta_1 = MSTW_PDF[[evolution_order]].iloc[index_delta_g,0][0]
    eta_2 = MSTW_PDF[[evolution_order]].iloc[index_eta_g,0][0]
    epsilon = MSTW_PDF[[evolution_order]].iloc[index_epsilon_g,0][0]
    gamma_pdf = MSTW_PDF[[evolution_order]].iloc[index_gamma_g,0][0]

    A_pdf_prime = MSTW_PDF[[evolution_order]].iloc[index_A_g_prime,0][0]
    eta_1_prime = MSTW_PDF[[evolution_order]].iloc[index_delta_g_prime,0][0]
    eta_2_prime = MSTW_PDF[[evolution_order]].iloc[index_eta_g_prime,0][0]

    if error_type == "central":
        result = pdf(x,A_pdf,eta_1,eta_2,epsilon,gamma_pdf)
        if evolution_order != "LO":
            result += pdf(x,A_pdf_prime,eta_1_prime,eta_2_prime,0,0)
    else:
        # Extracting errors
        delta_A_pdf  = MSTW_PDF[[evolution_order]].iloc[index_A_g,0][error_col_index]
        delta_eta_1 = MSTW_PDF[[evolution_order]].iloc[index_delta_g,0][error_col_index]
        delta_eta_2 = MSTW_PDF[[evolution_order]].iloc[index_eta_g,0][error_col_index]
        delta_epsilon = MSTW_PDF[[evolution_order]].iloc[index_epsilon_g,0][error_col_index]
        delta_gamma_pdf = MSTW_PDF[[evolution_order]].iloc[index_gamma_g,0][error_col_index]
        result = pdf_error(x,A_pdf,delta_A_pdf,eta_1,delta_eta_1,eta_2,delta_eta_2,epsilon,delta_epsilon,gamma_pdf,delta_gamma_pdf,error_type)
        if evolution_order != "LO":
            # Extracting errors
            delta_A_pdf_prime  = MSTW_PDF[[evolution_order]].iloc[index_A_g_prime,0][error_col_index]
            delta_eta_1_prime = MSTW_PDF[[evolution_order]].iloc[index_delta_g_prime,0][error_col_index]
            delta_eta_2_prime = MSTW_PDF[[evolution_order]].iloc[index_eta_g_prime,0][error_col_index]

            dpdf_dA_pdf = (1-x)**eta_2_prime*x**(eta_1_prime-1)
            dpdf_deta_1 = A_pdf_prime*(1-x)**eta_2_prime*x**(eta_1_prime-1)*np.log(x)
            dpdf_deta_2 = A_pdf_prime*(1-x)**eta_2_prime*x**(eta_1_prime-1)*np.log(1-x)

            #print(dpdf_dA_pdf,dpdf_deta_1,dpdf_deta_2)
            Delta_A_prime = dpdf_dA_pdf * delta_A_pdf_prime
            Delta_eta_1_prime = dpdf_deta_1 * delta_eta_1_prime
            Delta_eta_2_prime = dpdf_deta_2 * delta_eta_2_prime

            result += np.sqrt(Delta_A_prime**2+Delta_eta_1_prime**2+Delta_eta_2_prime**2)
    return result

def uv_minus_dv_pdf(x, evolution_order="LO",error_type="central"):
    uv = uv_pdf(x,evolution_order,error_type)
    dv= dv_pdf(x,evolution_order,error_type)
    if error_type == "central":
        result = uv-dv
    else:
        result = np.sqrt(uv**2+dv**2)
    return result

def uv_plus_dv_plus_S_pdf(x, evolution_order="LO",error_type="central"):
    uv = uv_pdf(x,evolution_order,error_type)
    dv = dv_pdf(x,evolution_order,error_type)
    Spdf = S_pdf(x,evolution_order,error_type)
    if error_type == "central":
        result = uv+dv+Spdf
    else:
        result = np.sqrt(uv**2+dv**2+Spdf**2)
    return result

######################
### Plot Functions ###
######################

def plot_uv_pdf(x_0=1e-2,evolution_order="LO",logplot=False,error_bars=True):
    vectorized_uv_pdf = np.vectorize(uv_pdf)
    if logplot:
        x_vals = np.logspace(np.log10(x_0), np.log10(1 - 1e-4), 100)
    else:
        x_vals = np.linspace(x_0,1-1e-4,100)
    y_vals = x_vals* vectorized_uv_pdf(x_vals,evolution_order)
    if error_bars:
        plus_error = x_vals * vectorized_uv_pdf(x_vals,evolution_order,"plus")
        minus_error = x_vals * vectorized_uv_pdf(x_vals,evolution_order,"minus")
    else:
        plus_error = 0 * y_vals
        minus_error = 0 * y_vals

    plt.errorbar(
            x_vals, y_vals,
            yerr=(minus_error, plus_error),
            fmt='o')
    if logplot:
        plt.xscale('log')
    plt.grid(True)
    plt.show()

def plot_dv_pdf(x_0=1e-2,evolution_order="LO",logplot=False,error_bars=True):
    vectorized_dv_pdf = np.vectorize(dv_pdf)
    if logplot:
        x_vals = np.logspace(np.log10(x_0), np.log10(1 - 1e-4), 100)
    else:
        x_vals = np.linspace(x_0,1-1e-4,100)
    y_vals = x_vals* vectorized_dv_pdf(x_vals,evolution_order)
    if error_bars:
        plus_error = x_vals * vectorized_dv_pdf(x_vals,evolution_order,"plus")
        minus_error = x_vals * vectorized_dv_pdf(x_vals,evolution_order,"minus")
    else:
        plus_error = 0 * y_vals
        minus_error = 0 * y_vals

    plt.errorbar(
            x_vals, y_vals,
            yerr=(minus_error, plus_error),
            fmt='o')
    if logplot:
        plt.xscale('log')
    plt.grid(True)
    plt.show()

def plot_uv_minus_dv_pdf(x_0=1e-2,evolution_order="LO",logplot=False,error_bars=True):
    vectorized_uv_minus_dv_pdf = np.vectorize(uv_minus_dv_pdf)
    if logplot:
        x_vals = np.logspace(np.log10(x_0), np.log10(1 - 1e-4), 100)
    else:
        x_vals = np.linspace(x_0,1-1e-4,100)
    y_vals = vectorized_uv_minus_dv_pdf(x_vals,evolution_order)
    if error_bars:
        plus_error = vectorized_uv_minus_dv_pdf(x_vals,evolution_order,"plus")
        minus_error = vectorized_uv_minus_dv_pdf(x_vals,evolution_order,"minus")
    else:
        plus_error = 0 * y_vals
        minus_error = 0 * y_vals

    plt.errorbar(
            x_vals, y_vals,
            yerr=(minus_error, plus_error),
            fmt='o')
    if logplot:
        plt.xscale('log')
    plt.grid(True)
    plt.show()

def plot_uv_plus_dv_plus_S_pdf(x_0=1e-2,evolution_order="LO",logplot=False,error_bars=True):
    vectorized_uv_plus_dv_plus_S_pdf = np.vectorize(uv_plus_dv_plus_S_pdf)
    if logplot:
        x_vals = np.logspace(np.log10(x_0), np.log10(1 - 1e-4), 100)
    else:
        x_vals = np.linspace(x_0,1-1e-4,100)
    y_vals = vectorized_uv_plus_dv_plus_S_pdf(x_vals,evolution_order)
    if error_bars:
        plus_error = vectorized_uv_plus_dv_plus_S_pdf(x_vals,evolution_order,"plus")
        minus_error = vectorized_uv_plus_dv_plus_S_pdf(x_vals,evolution_order,"minus")
    else:
        plus_error = 0 * y_vals
        minus_error = 0 * y_vals

    plt.errorbar(
            x_vals, y_vals,
            yerr=(minus_error, plus_error),
            fmt='o')
    if logplot:
        plt.xscale('log')
    plt.grid(True)
    plt.show()

def plot_gluon_pdf(x_0=1e-2,evolution_order="LO",logplot=False,error_bars=True):
    vectorized_gluon_pdf = np.vectorize(gluon_pdf)
    if logplot:
        x_vals = np.logspace(np.log10(x_0), np.log10(1 - 1e-4), 100)
    else:
        x_vals = np.linspace(x_0,1-1e-4,100)
    y_vals = x_vals * vectorized_gluon_pdf(x_vals,evolution_order)
    if error_bars:
        plus_error = x_vals * vectorized_gluon_pdf(x_vals,evolution_order,"plus")
        minus_error = x_vals * vectorized_gluon_pdf(x_vals,evolution_order,"minus")
    else:
        plus_error = 0 * y_vals
        minus_error = 0 * y_vals

    plt.errorbar(
            x_vals, y_vals,
            yerr=(minus_error, plus_error),
            fmt='o')
    if logplot:
        plt.xscale('log')
    plt.grid(True)
    plt.show()
