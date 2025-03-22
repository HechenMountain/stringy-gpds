# ############################################### #
# Polarized PDFs currently under assumption of    #
# isospin and qbar=ubar=dbar=s=sbar for polarized #
# PDFs. Modify AAC_Table_2.csv to not use this    #
# assumption.                                     #
# ############################################### #
#
# Dependencies
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt

# Unpolarized PDFs needed for parametrization
from mstw_pdf import (
    uv_pdf, dv_pdf, gluon_pdf,
    sv_pdf,s_plus_pdf, S_pdf,
    Delta_pdf
)

# Set display options
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000)        # Set width to avoid wrapping
pd.set_option('display.max_colwidth', None) # Show full content of each column

BASE_PATH = "/mnt/c/Users/flori/Documents/PostDoc/Data/PDFs/"
# Define the file path to the .csv file and extract its content
AAC_PATH = f"{BASE_PATH}AAC_Table_2.csv"

# Columns for the DataFrame
columns = ["Parameter", "LO", "NLO", "NNLO"]

# Read the CSV file and parse it
data = []
with open(AAC_PATH, 'r',newline='') as file:
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
AAC_PDF = pd.DataFrame(data, columns=columns)

# Create the pandas DataFrame
AAC_PDF = pd.DataFrame(data, columns=columns)

# Extract LO, NLO, and NNLO columns
AAC_PDF_LO = AAC_PDF[["LO"]]
AAC_PDF_NLO = AAC_PDF[["NLO"]]
AAC_PDF_NNLO = AAC_PDF[["NNLO"]]

# Create a DataFrame from the parsed data
AAC_PDF = pd.DataFrame(data, columns=columns)

# Extract LO, NLO, and NNLO columns
AAC_PDF_LO = AAC_PDF[["LO"]]
AAC_PDF_NLO = AAC_PDF[["NLO"]]
AAC_PDF_NNLO = AAC_PDF[["NNLO"]]

def polarized_uv_pdf(x, error_type="central"):
    """
    Compute the polarized uv(x) PDF based on the given LO parameters and selected errors.
    
    Arguments:
    x -- The value of parton x.
    error_type -- A string indicating whether to use 'central', 'plus', or 'minus' errors. Default is 'central'.
    
    Returns:
    The value of the polarized uv(x) based on the selected parameters and error type.
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
    index_delta_A_u=AAC_PDF[AAC_PDF["Parameter"] == "Delta_A_u"].index[0]
    index_alpha_u=AAC_PDF[AAC_PDF["Parameter"] == "alpha_u"].index[0]
    index_delta_lambda_u=AAC_PDF[AAC_PDF["Parameter"] == "Delta_lambda_u"].index[0]
    index_delta_gamma_u=AAC_PDF[AAC_PDF["Parameter"] == "Delta_gamma_u"].index[0]

    delta_A_u = AAC_PDF_LO.iloc[index_delta_A_u,0][0] + int(error_col_index>0)*AAC_PDF_LO.iloc[index_delta_A_u,0][error_col_index]
    alpha_u = AAC_PDF_LO.iloc[index_alpha_u,0][0] + int(error_col_index>0)*AAC_PDF_LO.iloc[index_alpha_u,0][error_col_index]
    delta_gamma_u = AAC_PDF_LO.iloc[index_delta_gamma_u,0][0] + int(error_col_index>0)*AAC_PDF_LO.iloc[index_delta_gamma_u,0][error_col_index]
    delta_lambda_u = AAC_PDF_LO.iloc[index_delta_lambda_u,0][0] + int(error_col_index>0)*AAC_PDF_LO.iloc[index_delta_lambda_u,0][error_col_index]
    
    result = delta_A_u * x**(alpha_u)*(1+delta_gamma_u* x**(delta_lambda_u)) * uv_pdf(x,"central")
    return result

def polarized_dv_pdf(x, error_type="central"):
    """
    Compute the polarized dv(x) PDF based on the given LO parameters and selected errors.
    
    Arguments:
    x -- The value of parton x.
    error_type -- A string indicating whether to use 'central', 'plus', or 'minus' errors. Default is 'central'.
    
    Returns:
    The value of the polarized dv(x) based on the selected parameters and error type.
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
    index_delta_A_d=AAC_PDF[AAC_PDF["Parameter"] == "Delta_A_d"].index[0]
    index_alpha_d=AAC_PDF[AAC_PDF["Parameter"] == "alpha_d"].index[0]
    index_delta_lambda_d=AAC_PDF[AAC_PDF["Parameter"] == "Delta_lambda_d"].index[0]
    index_delta_gamma_d=AAC_PDF[AAC_PDF["Parameter"] == "Delta_gamma_d"].index[0]

    delta_A_d = AAC_PDF_LO.iloc[index_delta_A_d,0][0] + int(error_col_index>0)*AAC_PDF_LO.iloc[index_delta_A_d,0][error_col_index]
    alpha_d = AAC_PDF_LO.iloc[index_alpha_d,0][0] + int(error_col_index>0)*AAC_PDF_LO.iloc[index_alpha_d,0][error_col_index]
    delta_gamma_d = AAC_PDF_LO.iloc[index_delta_gamma_d,0][0] + int(error_col_index>0)*AAC_PDF_LO.iloc[index_delta_gamma_d,0][error_col_index]
    delta_lambda_d = AAC_PDF_LO.iloc[index_delta_lambda_d,0][0] + int(error_col_index>0)*AAC_PDF_LO.iloc[index_delta_lambda_d,0][error_col_index]
    
    result = delta_A_d * x**(alpha_d)*(1+delta_gamma_d* x**(delta_lambda_d)) * dv_pdf(x,"central")
    return result

def polarized_gluon_pdf(x, error_type="central"):
    """
    Compute the polarized gluon(x) PDF based on the given LO parameters and selected errors.
    
    Arguments:
    x -- The value of parton x.
    error_type -- A string indicating whether to use 'central', 'plus', or 'minus' errors. Default is 'central'.
    
    Returns:
    The value of the polarized gluon(x) based on the selected parameters and error type.
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
    index_delta_A_g=AAC_PDF[AAC_PDF["Parameter"] == "Delta_A_g"].index[0]
    index_alpha_g=AAC_PDF[AAC_PDF["Parameter"] == "alpha_g"].index[0]
    index_delta_lambda_g=AAC_PDF[AAC_PDF["Parameter"] == "Delta_lambda_g"].index[0]
    index_delta_gamma_g=AAC_PDF[AAC_PDF["Parameter"] == "Delta_gamma_g"].index[0]

    delta_A_g = AAC_PDF_LO.iloc[index_delta_A_g,0][0] + int(error_col_index>0)*AAC_PDF_LO.iloc[index_delta_A_g,0][error_col_index]
    alpha_g = AAC_PDF_LO.iloc[index_alpha_g,0][0] + int(error_col_index>0)*AAC_PDF_LO.iloc[index_alpha_g,0][error_col_index]
    delta_gamma_g = AAC_PDF_LO.iloc[index_delta_gamma_g,0][0] + int(error_col_index>0)*AAC_PDF_LO.iloc[index_delta_gamma_g,0][error_col_index]
    delta_lambda_g = AAC_PDF_LO.iloc[index_delta_lambda_g,0][0] + int(error_col_index>0)*AAC_PDF_LO.iloc[index_delta_lambda_g,0][error_col_index]
    
    result = delta_A_g * x**(alpha_g)*(1+delta_gamma_g* x**(delta_lambda_g))*gluon_pdf(x,"central")
    return result

def polarized_s_pdf(x, error_type="central"):
    """
    Compute the polarized sv(x) PDF based on the given LO parameters and selected errors.
    
    Arguments:
    x -- The value of parton x.
    error_type -- A string indicating whether to use 'central', 'plus', or 'minus' errors. Default is 'central'.
    
    Returns:
    The value of the polarized sv(x) based on the selected parameters and error type.
    """
    print("Warning: Wrong output when Delta s = Delta sbar is assumed")
    print("Verify that AAC_Table_2.csv is correctly modified")
     # Define a dictionary that maps the error_type to column indices
    error_mapping = {
        "central": 0,  # The column with the central value
        "plus": 1,     # The column with the + error value
        "minus": 2     # The column with the - error value
    }
    
    # Get the column index corresponding to the error_type
    error_col_index = error_mapping.get(error_type, 0)  # Default to 'central' if error_type is invalid

    # Get row index of entry
    index_delta_A_s=AAC_PDF[AAC_PDF["Parameter"] == "Delta_A_s"].index[0]
    index_alpha_s=AAC_PDF[AAC_PDF["Parameter"] == "alpha_s"].index[0]
    index_delta_lambda_s=AAC_PDF[AAC_PDF["Parameter"] == "Delta_lambda_s"].index[0]
    index_delta_gamma_s=AAC_PDF[AAC_PDF["Parameter"] == "Delta_gamma_s"].index[0]

    delta_A_s = AAC_PDF_LO.iloc[index_delta_A_s,0][0] + int(error_col_index>0)*AAC_PDF_LO.iloc[index_delta_A_s,0][error_col_index]
    alpha_s = AAC_PDF_LO.iloc[index_alpha_s,0][0] + int(error_col_index>0)*AAC_PDF_LO.iloc[index_alpha_s,0][error_col_index]
    delta_gamma_s = AAC_PDF_LO.iloc[index_delta_gamma_s,0][0] + int(error_col_index>0)*AAC_PDF_LO.iloc[index_delta_gamma_s,0][error_col_index]
    delta_lambda_s = AAC_PDF_LO.iloc[index_delta_lambda_s,0][0] + int(error_col_index>0)*AAC_PDF_LO.iloc[index_delta_lambda_s,0][error_col_index]
    
    result = delta_A_s * x**(alpha_s)*(1+delta_gamma_s* x**(delta_lambda_s))*(s_plus_pdf(x,"central")+sv_pdf(x,"central"))/2
    return result

def polarized_sbar_pdf(x, error_type="central"):
    """
    Compute the polarized sv(x) PDF based on the given LO parameters and selected errors.
    
    Arguments:
    x -- The value of parton x.
    error_type -- A string indicating whether to use 'central', 'plus', or 'minus' errors. Default is 'central'.
    
    Returns:
    The value of the polarized sv(x) based on the selected parameters and error type.
    """
    print("Warning: Wrong output when Delta s = Delta sbar is assumed")
    print("Verify that AAC_Table_2.csv is correctly modified")
     # Define a dictionary that maps the error_type to column indices
    error_mapping = {
        "central": 0,  # The column with the central value
        "plus": 1,     # The column with the + error value
        "minus": 2     # The column with the - error value
    }
    
    # Get the column index corresponding to the error_type
    error_col_index = error_mapping.get(error_type, 0)  # Default to 'central' if error_type is invalid

    # Get row index of entry
    index_delta_A_sbar=AAC_PDF[AAC_PDF["Parameter"] == "Delta_A_sbar"].index[0]
    index_alpha_sbar=AAC_PDF[AAC_PDF["Parameter"] == "alpha_sbar"].index[0]
    index_delta_lambda_sbar=AAC_PDF[AAC_PDF["Parameter"] == "Delta_lambda_sbar"].index[0]
    index_delta_gamma_sbar=AAC_PDF[AAC_PDF["Parameter"] == "Delta_gamma_sbar"].index[0]

    delta_A_sbar = AAC_PDF_LO.iloc[index_delta_A_sbar,0][0] + int(error_col_index>0)*AAC_PDF_LO.iloc[index_delta_A_sbar,0][error_col_index]
    alpha_sbar = AAC_PDF_LO.iloc[index_alpha_sbar,0][0] + int(error_col_index>0)*AAC_PDF_LO.iloc[index_alpha_sbar,0][error_col_index]
    delta_gamma_sbar = AAC_PDF_LO.iloc[index_delta_gamma_sbar,0][0] + int(error_col_index>0)*AAC_PDF_LO.iloc[index_delta_gamma_sbar,0][error_col_index]
    delta_lambda_sbar = AAC_PDF_LO.iloc[index_delta_lambda_sbar,0][0] + int(error_col_index>0)*AAC_PDF_LO.iloc[index_delta_lambda_sbar,0][error_col_index]
    
    result = delta_A_sbar * x**(alpha_sbar)*(1+delta_gamma_sbar* x**(delta_lambda_sbar))*(s_plus_pdf(x,"central")-sv_pdf(x,"central"))/2
    return result

def polarized_s_plus_pdf(x, error_type="central"):
    """
    Compute the polarized sv(x) PDF based on the given LO parameters and selected errors.
    
    Arguments:
    x -- The value of parton x.
    error_type -- A string indicating whether to use 'central', 'plus', or 'minus' errors. Default is 'central'.
    
    Returns:
    The value of the polarized sv(x) based on the selected parameters and error type.
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
    index_delta_A_s_plus=AAC_PDF[AAC_PDF["Parameter"] == "Delta_A_s_plus"].index[0]
    index_alpha_s_plus=AAC_PDF[AAC_PDF["Parameter"] == "alpha_s_plus"].index[0]
    index_delta_lambda_s_plus=AAC_PDF[AAC_PDF["Parameter"] == "Delta_lambda_s_plus"].index[0]
    index_delta_gamma_s_plus=AAC_PDF[AAC_PDF["Parameter"] == "Delta_gamma_s_plus"].index[0]

    delta_A_s_plus = AAC_PDF_LO.iloc[index_delta_A_s_plus,0][0] + int(error_col_index>0)*AAC_PDF_LO.iloc[index_delta_A_s_plus,0][error_col_index]
    alpha_s_plus = AAC_PDF_LO.iloc[index_alpha_s_plus,0][0] + int(error_col_index>0)*AAC_PDF_LO.iloc[index_alpha_s_plus,0][error_col_index]
    delta_gamma_s_plus = AAC_PDF_LO.iloc[index_delta_gamma_s_plus,0][0] + int(error_col_index>0)*AAC_PDF_LO.iloc[index_delta_gamma_s_plus,0][error_col_index]
    delta_lambda_s_plus = AAC_PDF_LO.iloc[index_delta_lambda_s_plus,0][0] + int(error_col_index>0)*AAC_PDF_LO.iloc[index_delta_lambda_s_plus,0][error_col_index]
    
    result = delta_A_s_plus * x**(alpha_s_plus)*(1+delta_gamma_s_plus* x**(delta_lambda_s_plus))*s_plus_pdf(x,"central")
    return result

def polarized_ubar_pdf(x, error_type="central"):
    """
    Compute the polarized ubar(x) PDF based on the given LO parameters and selected errors.
    
    Arguments:
    x -- The value of parton x.
    error_type -- A string indicating whether to use 'central', 'plus', or 'minus' errors. Default is 'central'.
    
    Returns:
    The value of the polarized ubar(x) based on the selected parameters and error type.
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
    index_delta_A_ubar=AAC_PDF[AAC_PDF["Parameter"] == "Delta_A_ubar"].index[0]
    index_alpha_ubar=AAC_PDF[AAC_PDF["Parameter"] == "alpha_ubar"].index[0]
    index_delta_lambda_ubar=AAC_PDF[AAC_PDF["Parameter"] == "Delta_lambda_ubar"].index[0]
    index_delta_gamma_ubar=AAC_PDF[AAC_PDF["Parameter"] == "Delta_gamma_ubar"].index[0]

    delta_A_ubar = AAC_PDF_LO.iloc[index_delta_A_ubar,0][0] + int(error_col_index>0)*AAC_PDF_LO.iloc[index_delta_A_ubar,0][error_col_index]
    alpha_ubar = AAC_PDF_LO.iloc[index_alpha_ubar,0][0] + int(error_col_index>0)*AAC_PDF_LO.iloc[index_alpha_ubar,0][error_col_index]
    delta_gamma_ubar = AAC_PDF_LO.iloc[index_delta_gamma_ubar,0][0] + int(error_col_index>0)*AAC_PDF_LO.iloc[index_delta_gamma_ubar,0][error_col_index]
    delta_lambda_ubar = AAC_PDF_LO.iloc[index_delta_lambda_ubar,0][0] + int(error_col_index>0)*AAC_PDF_LO.iloc[index_delta_lambda_ubar,0][error_col_index]
    
    result = delta_A_ubar * x**(alpha_ubar)*(1+delta_gamma_ubar* x**(delta_lambda_ubar))*(-2*Delta_pdf(x,"central") + S_pdf(x,"central")-s_plus_pdf(x,"central"))/4
    return result

def polarized_dbar_pdf(x, error_type="central"):
    """
    Compute the polarized dbar(x) PDF based on the given LO parameters and selected errors.
    
    Arguments:
    x -- The value of parton x.
    error_type -- A string indicating whether to use 'central', 'plus', or 'minus' errors. Default is 'central'.
    
    Returns:
    The value of the polarized dbar(x) based on the selected parameters and error type.
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
    index_delta_A_dbar=AAC_PDF[AAC_PDF["Parameter"] == "Delta_A_dbar"].index[0]
    index_alpha_dbar=AAC_PDF[AAC_PDF["Parameter"] == "alpha_dbar"].index[0]
    index_delta_lambda_dbar=AAC_PDF[AAC_PDF["Parameter"] == "Delta_lambda_dbar"].index[0]
    index_delta_gamma_dbar=AAC_PDF[AAC_PDF["Parameter"] == "Delta_gamma_dbar"].index[0]

    delta_A_dbar = AAC_PDF_LO.iloc[index_delta_A_dbar,0][0] + int(error_col_index>0)*AAC_PDF_LO.iloc[index_delta_A_dbar,0][error_col_index]
    alpha_dbar = AAC_PDF_LO.iloc[index_alpha_dbar,0][0] + int(error_col_index>0)*AAC_PDF_LO.iloc[index_alpha_dbar,0][error_col_index]
    delta_gamma_dbar = AAC_PDF_LO.iloc[index_delta_gamma_dbar,0][0] + int(error_col_index>0)*AAC_PDF_LO.iloc[index_delta_gamma_dbar,0][error_col_index]
    delta_lambda_dbar = AAC_PDF_LO.iloc[index_delta_lambda_dbar,0][0] + int(error_col_index>0)*AAC_PDF_LO.iloc[index_delta_lambda_dbar,0][error_col_index]
    
    result = delta_A_dbar * x**(alpha_dbar)*(1+delta_gamma_dbar* x**(delta_lambda_dbar))*(2*Delta_pdf(x,"central") + S_pdf(x,"central")-s_plus_pdf(x,"central"))/4
    return result

def polarized_S_pdf(x, error_type="central"):
    """
    Compute the polarized S(x) PDF based on the given LO parameters and selected errors.
    
    Arguments:
    x -- The value of parton x.
    error_type -- A string indicating whether to use 'central', 'plus', or 'minus' errors. Default is 'central'.
    
    Returns:
    The value of the polarized S(x) based on the selected parameters and error type.
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
    index_delta_A_S=AAC_PDF[AAC_PDF["Parameter"] == "Delta_A_S"].index[0]
    index_alpha_S=AAC_PDF[AAC_PDF["Parameter"] == "alpha_S"].index[0]
    index_delta_lambda_S=AAC_PDF[AAC_PDF["Parameter"] == "Delta_lambda_S"].index[0]
    index_delta_gamma_S=AAC_PDF[AAC_PDF["Parameter"] == "Delta_gamma_S"].index[0]

    delta_A_S = AAC_PDF_LO.iloc[index_delta_A_S,0][0] + int(error_col_index>0)*AAC_PDF_LO.iloc[index_delta_A_S,0][error_col_index]
    alpha_S = AAC_PDF_LO.iloc[index_alpha_S,0][0] + int(error_col_index>0)*AAC_PDF_LO.iloc[index_alpha_S,0][error_col_index]
    delta_gamma_S = AAC_PDF_LO.iloc[index_delta_gamma_S,0][0] + int(error_col_index>0)*AAC_PDF_LO.iloc[index_delta_gamma_S,0][error_col_index]
    delta_lambda_S = AAC_PDF_LO.iloc[index_delta_lambda_S,0][0] + int(error_col_index>0)*AAC_PDF_LO.iloc[index_delta_lambda_S,0][error_col_index]
    
    result = delta_A_S * x**(alpha_S)*(1+delta_gamma_S* x**(delta_lambda_S)) * S_pdf(x,"central")
    return result


def polarized_uv_minus_dv_pdf(x, error_type="central"):
    """
    Compute the polarized uv-dv(x) PDF based on the given LO parameters and selected errors.
    
    Arguments:
    x -- The value of parton x.
    error_type -- A string indicating whether to use 'central', 'plus', or 'minus' errors. Default is 'central'.
    
    Returns:
    The value of the polarized uv-dv(x) based on the selected parameters and error type.
    """
    
    result = polarized_uv_pdf(x,error_type) - polarized_dv_pdf(x,error_type)
    return result

def polarized_uv_plus_dv_plus_S_pdf(x, error_type="central"):
    """
    Compute the polarized uv+dv+S(x) PDF based on the given LO parameters and selected errors.
    
    Arguments:
    x -- The value of parton x.
    error_type -- A string indicating whether to use 'central', 'plus', or 'minus' errors. Default is 'central'.
    
    Returns:
    The value of the polarized uv+dv+S(x) based on the selected parameters and error type.
    """

    result = polarized_uv_pdf(x,error_type) + polarized_dv_pdf(x,error_type) + polarized_S_pdf(x,error_type)
    return result

######################
### Plot Functions ###
######################

def plot_polarized_uv_minus_dv_pdf(x_0=1e-2,logplot = False):
    vectorized_polarized_uv_minus_dv_pdf = np.vectorize(polarized_uv_minus_dv_pdf)
    x_vals = np.linspace(x_0,1,100)
    y_vals = vectorized_polarized_uv_minus_dv_pdf(x_vals)
    y_vals_plus = abs(vectorized_polarized_uv_minus_dv_pdf(x_vals,"plus") - y_vals)
    y_vals_minus = abs(y_vals - vectorized_polarized_uv_minus_dv_pdf(x_vals,"minus"))

    plt.errorbar(
            x_vals, y_vals,
            yerr=(y_vals_minus, y_vals_plus),
            fmt='o')
    plt.grid(True)
    if logplot:
        plt.xscale('log')
    plt.show()

def plot_polarized_uv_plus_dv_plus_S_pdf(x_0=1e-2,logplot = False,error_bars=True):
    vectorized_polarized_uv_plus_dv_plus_S_pdf = np.vectorize(polarized_uv_plus_dv_plus_S_pdf)
    x_vals = np.linspace(x_0,1,100)
    y_vals = vectorized_polarized_uv_plus_dv_plus_S_pdf(x_vals)
    y_vals_plus = abs(vectorized_polarized_uv_plus_dv_plus_S_pdf(x_vals,"plus") - y_vals)
    y_vals_minus = abs(y_vals - vectorized_polarized_uv_plus_dv_plus_S_pdf(x_vals,"minus"))
    if error_bars:
        plt.errorbar(
                x_vals, y_vals,
                yerr=(y_vals_minus, y_vals_plus),
                fmt='o')
    else:
        plt.scatter(x_vals,y_vals)
    plt.grid(True)
    if logplot:
        plt.xscale('log')
    plt.show()

def plot_polarized_gluon_pdf(x_0=1e-2,logplot = False,error_bars=True):
    vectorized_polarized_gluon_pdf = np.vectorize(polarized_gluon_pdf)
    x_vals = np.linspace(x_0,1,100)
    y_vals = x_vals * vectorized_polarized_gluon_pdf(x_vals)
    if error_bars:
        y_vals_plus = abs(x_vals *vectorized_polarized_gluon_pdf(x_vals,"plus") - y_vals)
        y_vals_minus = abs(y_vals - x_vals *vectorized_polarized_gluon_pdf(x_vals,"minus"))
    else:
        y_vals_plus = 0 * y_vals
        y_vals_minus = 0 * y_vals

    plt.errorbar(
            x_vals, y_vals,
            yerr=(y_vals_minus, y_vals_plus),
            fmt='o')
    plt.grid(True)
    if logplot:
        plt.xscale('log')
    plt.show()