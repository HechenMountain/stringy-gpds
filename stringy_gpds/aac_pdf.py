# ############################################### #
# Polarized PDFs currently under assumption of    #
# isospin and qbar=ubar=dbar=s=sbar for polarized #
# PDFs. Modify AAC_Table_2.csv to not use this    #
# assumption.                                     #
# ############################################### #
#
# Dependencies
import csv
import numpy as np
import matplotlib.pyplot as plt

# Unpolarized PDFs needed for parametrization
from mstw_pdf import (
    uv_pdf, dv_pdf, gluon_pdf,
    sv_pdf,s_plus_pdf, S_pdf,
    Delta_pdf
)

import config as cfg

from helpers import check_error_type

# Columns for the DataFrame
columns = ["Parameter", "LO", "NLO", "NNLO"]

# Read the CSV file and parse it
data = []
with open(cfg.AAC_PATH, 'r',newline='') as file:
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
AAC_PDF =  {row[0]: {"LO": row[1], "NLO": row[2], "NNLO": row[3]} for row in data}
    
# PDFs
def polarized_pdf(x,delta_A_pdf,alpha_pdf,delta_lambda_pdf,delta_gamma_pdf,evolution_order):
    """
    Returns the bare value (without the unpolarized input PDF) of the polarized PDF
    """
    if evolution_order != "LO":
        result = delta_A_pdf * x**(alpha_pdf)*(1+delta_gamma_pdf * (x**(delta_lambda_pdf)-1))
    else:
        result = delta_A_pdf * x**(alpha_pdf)*(1+delta_gamma_pdf* x**(delta_lambda_pdf))
    return result

def polarized_pdf_error(x,delta_A_pdf,delta_delta_A_pdf,alpha_pdf,delta_alpha_pdf,delta_lambda_pdf,delta_delta_lambda_pdf,delta_gamma_pdf,delta_delta_gamma_pdf,evolution_order,error_type):
    check_error_type(error_type)
    if error_type == "central":
            return 0
    
    dpdf_dA_pdf =  x**(alpha_pdf)*(1+delta_gamma_pdf* x**(delta_lambda_pdf))
    dpdf_dalpha = delta_A_pdf* x**(alpha_pdf)*(1+delta_gamma_pdf* x**(delta_lambda_pdf))*np.log(x)
    dpdf_dlambda = delta_A_pdf* x**(alpha_pdf+delta_lambda_pdf)*(delta_gamma_pdf)*np.log(x)
    dpdf_dgamma = delta_A_pdf* x**(alpha_pdf+delta_lambda_pdf)

    if evolution_order != "LO":
        dpdf_dA_pdf += -x**alpha_pdf*delta_gamma_pdf
        dpdf_dalpha += -delta_A_pdf*x**alpha_pdf*delta_gamma_pdf
        dpdf_dgamma += -delta_A_pdf*x**alpha_pdf

    Delta_A = dpdf_dA_pdf * delta_delta_A_pdf
    Delta_alpha= dpdf_dalpha * delta_alpha_pdf
    Delta_lambda = dpdf_dlambda * delta_delta_lambda_pdf
    Delta_gamma = dpdf_dgamma  * delta_delta_gamma_pdf

    result = np.sqrt(Delta_A**2+Delta_alpha**2+Delta_lambda**2+Delta_gamma**2)
    return result

def polarized_uv_pdf(x, evolution_order="LO",error_type="central"):
    """
    Compute the polarized uv(x) PDF and returns either it's central value or the corresponding error
    
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
    check_error_type(error_type)
    # Get the column index corresponding to the error_type
    error_col_index = error_mapping.get(error_type, 0)  # Default to 'central' if error_type is invalid

    delta_A_u = AAC_PDF["Delta_A_u"][evolution_order][0]
    alpha_u = AAC_PDF["alpha_u"][evolution_order][0]
    delta_gamma_u = AAC_PDF["Delta_gamma_u"][evolution_order][0]
    delta_lambda_u = AAC_PDF["Delta_lambda_u"][evolution_order][0]
    if error_type == "central":
        result = polarized_pdf(x,delta_A_u,alpha_u,delta_lambda_u,delta_gamma_u,evolution_order)* uv_pdf(x,evolution_order,"central")
    else:
        # Extracting errors
        delta_delta_A_u =  AAC_PDF["Delta_A_u"][evolution_order][error_col_index]
        delta_alpha_u = AAC_PDF["alpha_u"][evolution_order][error_col_index]
        delta_delta_gamma_u = AAC_PDF["Delta_gamma_u"][evolution_order][error_col_index]
        delta_delta_lambda_u = AAC_PDF["Delta_lambda_u"][evolution_order][error_col_index]

        
        result = polarized_pdf_error(x,delta_A_u,delta_delta_A_u,alpha_u,delta_alpha_u,delta_lambda_u,delta_delta_lambda_u,delta_gamma_u,delta_delta_gamma_u,evolution_order,error_type)*uv_pdf(x,evolution_order,"central")

    return result

def polarized_dv_pdf(x, evolution_order="LO",error_type="central"):
    """
    Compute the polarized dv(x) PDF and returns either it's central value or the corresponding error
    
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
    check_error_type(error_type)
    # Get the column index corresponding to the error_type
    error_col_index = error_mapping.get(error_type, 0)  # Default to 'central' if error_type is invalid

    delta_A_d = AAC_PDF["Delta_A_d"][evolution_order][0]
    alpha_d =  AAC_PDF["alpha_d"][evolution_order][0]
    delta_gamma_d = AAC_PDF["Delta_gamma_d"][evolution_order][0]
    delta_lambda_d = AAC_PDF["Delta_lambda_d"][evolution_order][0]
    if error_type == "central":
        result = polarized_pdf(x,delta_A_d,alpha_d,delta_lambda_d,delta_gamma_d,evolution_order)*dv_pdf(x,evolution_order,"central")
    else:
        # Extracting errors
        delta_delta_A_d = AAC_PDF["Delta_A_d"][evolution_order][error_col_index]
        delta_alpha_d = AAC_PDF["alpha_d"][evolution_order][error_col_index]
        delta_delta_gamma_d = AAC_PDF["Delta_gamma_d"][evolution_order][error_col_index]
        delta_delta_lambda_d = AAC_PDF["Delta_lambda_d"][evolution_order][error_col_index]

        
        result = polarized_pdf_error(x,delta_A_d,delta_delta_A_d,alpha_d,delta_alpha_d,delta_lambda_d,delta_delta_lambda_d,delta_gamma_d,delta_delta_gamma_d,evolution_order,error_type)*dv_pdf(x,evolution_order,"central")

    return result

def polarized_gluon_pdf(x, evolution_order="LO",error_type="central"):
    """
    Compute the polarized gluon(x) PDF and returns either it's central value or the corresponding error
    
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
    check_error_type(error_type)
    # Get the column index corresponding to the error_type
    error_col_index = error_mapping.get(error_type, 0)  # Default to 'central' if error_type is invalid

    delta_A_g = AAC_PDF["Delta_A_g"][evolution_order][0]
    alpha_g = AAC_PDF["alpha_g"][evolution_order][0]
    delta_gamma_g = AAC_PDF["Delta_gamma_g"][evolution_order][0]
    delta_lambda_g = AAC_PDF["Delta_lambda_g"][evolution_order][0]
    if error_type == "central":
        result = polarized_pdf(x,delta_A_g,alpha_g,delta_lambda_g,delta_gamma_g,evolution_order)*gluon_pdf(x,evolution_order,"central")
    else:
        # Extracting errors
        delta_delta_A_g = AAC_PDF["Delta_A_g"][evolution_order][error_col_index]
        delta_alpha_g = AAC_PDF["alpha_g"][evolution_order][error_col_index]
        delta_delta_gamma_g = AAC_PDF["Delta_gamma_g"][evolution_order][error_col_index]
        delta_delta_lambda_g = AAC_PDF["Delta_lambda_g"][evolution_order][error_col_index]

        
        result = polarized_pdf_error(x,delta_A_g,delta_delta_A_g,alpha_g,delta_alpha_g,delta_lambda_g,delta_delta_lambda_g,delta_gamma_g,delta_delta_gamma_g,evolution_order,error_type)*gluon_pdf(x,evolution_order,"central")

    return result

def polarized_s_pdf(x, evolution_order="LO",error_type="central"):
    """
    Compute the polarized sv(x) PDF and returns either it's central value or the corresponding error
    
    Arguments:
    x -- The value of parton x.
    error_type -- A string indicating whether to use 'central', 'plus', or 'minus' errors. Default is 'central'.
    
    Returns:
    The value of the polarized sv(x) based on the selected parameters and error type.
    """
    print("Warning: Wrong output when Delta s = Delta sbar is assumed")
    print("Verify that AAC.csv is correctly modified")
     # Define a dictionary that maps the error_type to column indices
    error_mapping = {
        "central": 0,  # The column with the central value
        "plus": 1,     # The column with the + error value
        "minus": 2     # The column with the - error value
    }
    
    # Get the column index corresponding to the error_type
    error_col_index = error_mapping.get(error_type, 0)  # Default to 'central' if error_type is invalid
    check_error_type(error_type)
    # Get row index of entry

    delta_A_s = AAC_PDF["Delta_A_s"][evolution_order][0]
    alpha_s = AAC_PDF["alpha_s"][evolution_order][0]
    delta_gamma_s = AAC_PDF["Delta_gamma_s"][evolution_order][0]
    delta_lambda_s =  AAC_PDF["Delta_lambda_s"][evolution_order][0]
    if error_type == "central":
        result = polarized_pdf(x,delta_A_s,alpha_s,delta_lambda_s,delta_gamma_s,evolution_order)*(s_plus_pdf(x,evolution_order,"central")+sv_pdf(x,evolution_order,"central"))/2

    else:
        # Extracting errors
        delta_delta_A_s = AAC_PDF["Delta_A_s"][evolution_order][error_col_index]
        delta_alpha_s = AAC_PDF["alpha_s"][evolution_order][error_col_index]
        delta_delta_gamma_s = AAC_PDF["Delta_gamma_s"][evolution_order][error_col_index]
        delta_delta_lambda_s = AAC_PDF["Delta_lambda_s"][evolution_order][error_col_index]
        
        result = polarized_pdf_error(x,delta_A_s,delta_delta_A_s,alpha_s,delta_alpha_s,delta_lambda_s,delta_delta_lambda_s,delta_gamma_s,delta_delta_gamma_s,evolution_order,error_type)*(s_plus_pdf(x,evolution_order,"central")+sv_pdf(x,evolution_order,"central"))/2

    return result

def polarized_sbar_pdf(x, evolution_order="LO",error_type="central"):
    """
    Compute the polarized sv(x) PDF and returns either it's central value or the corresponding error
    
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
    check_error_type(error_type)
    # Get the column index corresponding to the error_type
    error_col_index = error_mapping.get(error_type, 0)  # Default to 'central' if error_type is invalid

    delta_A_sbar = AAC_PDF["Delta_A_sbar"][evolution_order][0]
    alpha_sbar = AAC_PDF["alpha_sbar"][evolution_order][0] 
    delta_gamma_sbar = AAC_PDF["Delta_gamma_sbar"][evolution_order][0]
    delta_lambda_sbar = AAC_PDF["Delta_lambda_sbar"][evolution_order][0]
    if error_type == "central":
        result = polarized_pdf(x,delta_A_sbar,alpha_sbar,delta_lambda_sbar,delta_gamma_sbar,evolution_order)*(s_plus_pdf(x,evolution_order,"central")-sv_pdf(x,evolution_order,"central"))/2
    if error_type != "central":
        # Extracting errors
        delta_delta_A_sbar = AAC_PDF["Delta_A_sbar"][evolution_order][error_col_index]
        delta_alpha_sbar = AAC_PDF["alpha_sbar"][evolution_order][error_col_index]
        delta_delta_gamma_sbar = AAC_PDF["Delta_gamma_sbar"][evolution_order][error_col_index]
        delta_delta_lambda_sbar = AAC_PDF["Delta_lambda_sbar"][evolution_order][error_col_index]

        result = polarized_pdf_error(x,delta_A_sbar,delta_delta_A_sbar,alpha_sbar,delta_alpha_sbar,delta_lambda_sbar,delta_delta_lambda_sbar,delta_gamma_sbar,delta_delta_gamma_sbar,evolution_order,error_type)*(s_plus_pdf(x,evolution_order,"central")-sv_pdf(x,evolution_order,"central"))/2

    return result

def polarized_s_plus_pdf(x, evolution_order="LO",error_type="central"):
    """
    Compute the polarized sv(x) PDF and returns either it's central value or the corresponding error
    
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
    check_error_type(error_type)
    # Get the column index corresponding to the error_type
    error_col_index = error_mapping.get(error_type, 0)  # Default to 'central' if error_type is invalid

    delta_A_s_plus = AAC_PDF["Delta_A_s_plus"][evolution_order][0]
    alpha_s_plus = AAC_PDF["alpha_s_plus"][evolution_order][0] 
    delta_gamma_s_plus = AAC_PDF["Delta_gamma_s_plus"][evolution_order][0]
    delta_lambda_s_plus = AAC_PDF["Delta_lambda_splus"][evolution_order][0]
    if error_type == "central":
        result = polarized_pdf(x,delta_A_s_plus,alpha_s_plus,delta_lambda_s_plus,delta_gamma_s_plus,evolution_order)*s_plus_pdf(x,evolution_order,"central")
    else :
        # Extracting errors
        delta_delta_A_s_plus = AAC_PDF["Delta_A_s_plus"][evolution_order][error_col_index]
        delta_alpha_s_plus = AAC_PDF["alpha_s_plus"][evolution_order][error_col_index]
        delta_delta_gamma_s_plus = AAC_PDF["Delta_gamma_s_plus"][evolution_order][error_col_index]
        delta_delta_lambda_s_plus = AAC_PDF["Delta_lambda_splus"][evolution_order][error_col_index]
    

        result = polarized_pdf_error(x,delta_A_s_plus,delta_delta_A_s_plus,alpha_s_plus,delta_alpha_s_plus,delta_lambda_s_plus,delta_delta_lambda_s_plus,delta_gamma_s_plus,delta_delta_gamma_s_plus,evolution_order,error_type)*s_plus_pdf(x,evolution_order,"central")
    
    return result

def polarized_ubar_pdf(x, evolution_order="LO",error_type="central"):
    """
    Compute the polarized ubar(x) PDF and returns either it's central value or the corresponding error
    
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
    check_error_type(error_type)
    # Get the column index corresponding to the error_type
    error_col_index = error_mapping.get(error_type, 0)  # Default to 'central' if error_type is invalid

    delta_A_ubar = AAC_PDF["Delta_A_ubar"][evolution_order][0]
    alpha_ubar = AAC_PDF["alpha_ubar"][evolution_order][0]
    delta_gamma_ubar = AAC_PDF["Delta_gamma_ubar"][evolution_order][0]
    delta_lambda_ubar = AAC_PDF["Delta_lambda_ubar"][evolution_order][0]

    # print(delta_A_ubar,alpha_ubar,delta_gamma_ubar,delta_lambda_ubar)

    if error_type == "central":
        result = polarized_pdf(x,delta_A_ubar,alpha_ubar,delta_lambda_ubar,delta_gamma_ubar,evolution_order)*(-2*Delta_pdf(x,evolution_order,"central") + S_pdf(x,evolution_order,"central")-s_plus_pdf(x,evolution_order,"central"))/4
    else:
        # Extracting errors
        delta_delta_A_ubar = AAC_PDF["Delta_A_ubar"][evolution_order][error_col_index]
        delta_alpha_ubar = AAC_PDF["alpha_ubar"][evolution_order][error_col_index]
        delta_delta_gamma_ubar = AAC_PDF["Delta_gamma_ubar"][evolution_order][error_col_index]
        delta_delta_lambda_ubar = AAC_PDF["Delta_lambda_ubar"][evolution_order][error_col_index]
    

        result = polarized_pdf_error(x,delta_A_ubar,delta_delta_A_ubar,alpha_ubar,delta_alpha_ubar,delta_lambda_ubar,delta_delta_lambda_ubar,delta_gamma_ubar,delta_delta_gamma_ubar,evolution_order,error_type)* \
                    (-2*Delta_pdf(x,evolution_order,"central") + S_pdf(x,evolution_order,"central")-s_plus_pdf(x,evolution_order,"central"))/4

    return result

def polarized_dbar_pdf(x, evolution_order="LO",error_type="central"):
    """
    Compute the polarized dbar(x) PDF and returns either it's central value or the corresponding error
    
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
    check_error_type(error_type)
    # Get the column index corresponding to the error_type
    error_col_index = error_mapping.get(error_type, 0)  # Default to 'central' if error_type is invalid

    delta_A_dbar = AAC_PDF["Delta_A_dbar"][evolution_order][0]
    alpha_dbar = AAC_PDF["alpha_dbar"][evolution_order][0]
    delta_gamma_dbar = AAC_PDF["Delta_gamma_dbar"][evolution_order][0]
    delta_lambda_dbar = AAC_PDF["Delta_lambda_dbar"][evolution_order][0]
    if error_type == "central":
        result = polarized_pdf(x,delta_A_dbar,alpha_dbar,delta_lambda_dbar,delta_gamma_dbar,evolution_order)*(2*Delta_pdf(x,evolution_order,"central") + S_pdf(x,evolution_order,"central")-s_plus_pdf(x,evolution_order,"central"))/4
    else:
        # Extracting errors
        delta_delta_A_dbar = AAC_PDF["Delta_A_dbar"][evolution_order][error_col_index]
        delta_alpha_dbar = AAC_PDF["alpha_dbar"][evolution_order][error_col_index]
        delta_delta_gamma_dbar = AAC_PDF["Delta_gamma_dbar"][evolution_order][error_col_index]
        delta_delta_lambda_dbar = AAC_PDF["Delta_lambda_dbar"][evolution_order][error_col_index]
    

        result = polarized_pdf_error(x,delta_A_dbar,delta_delta_A_dbar,alpha_dbar,delta_alpha_dbar,delta_lambda_dbar,delta_delta_lambda_dbar,delta_gamma_dbar,delta_delta_gamma_dbar,evolution_order,error_type)* \
                   (2*Delta_pdf(x,evolution_order,"central") + S_pdf(x,evolution_order,"central")-s_plus_pdf(x,evolution_order,"central"))/4
        
    return result

def polarized_S_pdf(x, evolution_order="LO",error_type="central"):
    """
    Compute the polarized S(x) PDF and returns either it's central value or the corresponding error
    
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
    check_error_type(error_type)
    # Get the column index corresponding to the error_type
    error_col_index = error_mapping.get(error_type, 0)  # Default to 'central' if error_type is invalid

    delta_A_S = AAC_PDF["Delta_A_S"][evolution_order][0]
    alpha_S = AAC_PDF["alpha_S"][evolution_order][0]
    delta_gamma_S = AAC_PDF["Delta_gamma_S"][evolution_order][0]
    delta_lambda_S = AAC_PDF["Delta_lambda_S"][evolution_order][0]
    
    if error_type == "central":
        result = polarized_pdf(x,delta_A_S,alpha_S,delta_lambda_S,delta_gamma_S,evolution_order) * S_pdf(x,evolution_order,"central")
    else:
        # Extracting errors
        delta_delta_A_S = AAC_PDF["Delta_A_S"][evolution_order][error_col_index]
        delta_alpha_S = AAC_PDF["alpha_S"][evolution_order][error_col_index]
        delta_delta_gamma_S = AAC_PDF["Delta_gamma_S"][evolution_order][error_col_index]
        delta_delta_lambda_S = AAC_PDF["Delta_lambda_S"][evolution_order][error_col_index]
    

        result = polarized_pdf_error(x,delta_A_S,delta_delta_A_S,alpha_S,delta_alpha_S,delta_lambda_S,delta_delta_lambda_S,delta_gamma_S,delta_delta_gamma_S,evolution_order,error_type)* \
                    S_pdf(x,evolution_order,"central")
    return result


def polarized_uv_minus_dv_pdf(x, evolution_order="LO",error_type="central"):
    """
    Compute the polarized uv-dv(x) PDF and returns either it's central value or the corresponding error
    
    Arguments:
    x -- The value of parton x.
    error_type -- A string indicating whether to use 'central', 'plus', or 'minus' errors. Default is 'central'.
    
    Returns:
    The value of the polarized uv-dv(x) based on the selected parameters and error type.
    """
    
    polarized_uv = polarized_uv_pdf(x,evolution_order,error_type) 
    polarized_dv = polarized_dv_pdf(x,evolution_order,error_type)
    if error_type == "central":
        result = polarized_uv-polarized_dv
    else:
        result = np.sqrt(polarized_uv**2+polarized_dv**2)
    return result

def polarized_uv_plus_dv_plus_S_pdf(x, evolution_order="LO",error_type="central"):
    """
    Compute the polarized uv+dv+S(x) PDF and returns either it's central value or the corresponding error
    
    Arguments:
    x -- The value of parton x.
    error_type -- A string indicating whether to use 'central', 'plus', or 'minus' errors. Default is 'central'.
    
    Returns:
    The value of the polarized uv+dv+S(x) based on the selected parameters and error type.
    """
    polarized_uv = polarized_uv_pdf(x,evolution_order,error_type) 
    polarized_dv = polarized_dv_pdf(x,evolution_order,error_type)
    polarized_S = polarized_S_pdf(x,evolution_order,error_type)
    if error_type == "central":
        result = polarized_uv+polarized_dv+polarized_S
    else:
        result = np.sqrt(polarized_uv**2+polarized_dv**2+polarized_S**2)
    return result

######################
### Plot Functions ###
######################

def plot_polarized_uv_pdf(x_0=1e-2,evolution_order="LO",logplot = False,error_bars=True):
    vectorized_polarized_uv_pdf = np.vectorize(polarized_uv_pdf)
    if logplot:
        x_vals = np.logspace(np.log10(x_0), np.log10(1 - 1e-4), 100)
    else:
        x_vals = np.linspace(x_0,1-1e-4,100)
    y_vals = x_vals * vectorized_polarized_uv_pdf(x_vals,evolution_order)
    if error_bars:
        plus_error = x_vals * vectorized_polarized_uv_pdf(x_vals,evolution_order,"plus")
        minus_error = x_vals * vectorized_polarized_uv_pdf(x_vals,evolution_order,"minus")
    else:
        plus_error = 0 * y_vals
        minus_error = 0 * y_vals

    plt.errorbar(
            x_vals, y_vals,
            yerr=(minus_error, plus_error),
            fmt='o')
    plt.grid(True)
    if logplot:
        plt.xscale('log')
    plt.show()

def plot_polarized_dv_pdf(x_0=1e-2,evolution_order="LO",logplot = False,error_bars=True):
    vectorized_polarized_dv_pdf = np.vectorize(polarized_dv_pdf)
    if logplot:
        x_vals = np.logspace(np.log10(x_0), np.log10(1 - 1e-4), 100)
    else:
        x_vals = np.linspace(x_0,1-1e-4,100)
    y_vals = x_vals * vectorized_polarized_dv_pdf(x_vals,evolution_order)
    if error_bars:
        plus_error = x_vals * vectorized_polarized_dv_pdf(x_vals,evolution_order,"plus")
        minus_error = x_vals *  vectorized_polarized_dv_pdf(x_vals,evolution_order,"minus")
    else:
        plus_error = 0 * y_vals
        minus_error = 0 * y_vals

    plt.errorbar(
            x_vals, y_vals,
            yerr=(minus_error, plus_error),
            fmt='o')
    plt.grid(True)
    if logplot:
        plt.xscale('log')
    plt.show()


def plot_polarized_ubar_pdf(x_0=1e-2,evolution_order="LO",logplot = False,error_bars=True):
    vectorized_polarized_ubar_pdf = np.vectorize(polarized_ubar_pdf)
    if logplot:
        x_vals = np.logspace(np.log10(x_0), np.log10(1 - 1e-4), 100)
    else:
        x_vals = np.linspace(x_0,1-1e-4,100)
    y_vals = x_vals * vectorized_polarized_ubar_pdf(x_vals,evolution_order)
    if error_bars:
        plus_error = x_vals * vectorized_polarized_ubar_pdf(x_vals,evolution_order,"plus")
        minus_error = x_vals *  vectorized_polarized_ubar_pdf(x_vals,evolution_order,"minus")
    else:
        plus_error = 0 * y_vals
        minus_error = 0 * y_vals

    plt.errorbar(
            x_vals, y_vals,
            yerr=(minus_error, plus_error),
            fmt='o')
    plt.grid(True)
    if logplot:
        plt.xscale('log')
    plt.show()

def plot_polarized_uv_minus_dv_pdf(x_0=1e-2,evolution_order="LO",logplot = False,error_bars=True):
    vectorized_polarized_uv_minus_dv_pdf = np.vectorize(polarized_uv_minus_dv_pdf)
    if logplot:
        x_vals = np.logspace(np.log10(x_0), np.log10(1 - 1e-4), 100)
    else:
        x_vals = np.linspace(x_0,1-1e-4,100)
    y_vals = vectorized_polarized_uv_minus_dv_pdf(x_vals,evolution_order)
    if error_bars:
        plus_error = vectorized_polarized_uv_minus_dv_pdf(x_vals,evolution_order,"plus")
        minus_error = vectorized_polarized_uv_minus_dv_pdf(x_vals,evolution_order,"minus")
    else:
        plus_error = 0 * y_vals
        minus_error = 0 * y_vals

    plt.errorbar(
            x_vals, y_vals,
            yerr=(minus_error, plus_error),
            fmt='o')
    plt.grid(True)
    if logplot:
        plt.xscale('log')
    plt.show()

def plot_polarized_uv_plus_dv_plus_S_pdf(x_0=1e-2,evolution_order="LO",logplot = False,error_bars=True):
    vectorized_polarized_uv_plus_dv_plus_S_pdf = np.vectorize(polarized_uv_plus_dv_plus_S_pdf)
    if logplot:
        x_vals = np.logspace(np.log10(x_0), np.log10(1 - 1e-4), 100)
    else:
        x_vals = np.linspace(x_0,1-1e-4,100)
    y_vals = vectorized_polarized_uv_plus_dv_plus_S_pdf(x_vals,evolution_order)
    if error_bars:
        plus_error = vectorized_polarized_uv_plus_dv_plus_S_pdf(x_vals,evolution_order,"plus")
        minus_error = vectorized_polarized_uv_plus_dv_plus_S_pdf(x_vals,evolution_order,"minus")
    else:
        plus_error = 0 * y_vals
        minus_error = 0 * y_vals

    plt.errorbar(
            x_vals, y_vals,
            yerr=(minus_error, plus_error),
            fmt='o')
    plt.grid(True)
    if logplot:
        plt.xscale('log')
    plt.show()

def plot_polarized_gluon_pdf(x_0=1e-2,y_0=-1,y_1=1,evolution_order="LO",logplot = False,error_bars=True):
    vectorized_polarized_gluon_pdf = np.vectorize(polarized_gluon_pdf)
    if logplot:
        x_vals = np.logspace(np.log10(x_0), np.log10(1 - 1e-4), 100)
    else:
        x_vals = np.linspace(x_0,1-1e-4,100)
    y_vals = x_vals * vectorized_polarized_gluon_pdf(x_vals,evolution_order)
    if error_bars:
        plus_error = x_vals *vectorized_polarized_gluon_pdf(x_vals,evolution_order,"plus")
        minus_error = x_vals *vectorized_polarized_gluon_pdf(x_vals,evolution_order,"minus")
    else:
        plus_error = 0 * y_vals
        minus_error = 0 * y_vals

    plt.errorbar(
            x_vals, y_vals,
            yerr=(minus_error, plus_error),
            fmt='o')
    plt.grid(True)
    plt.ylim([y_0,y_1])
    if logplot:
        plt.xscale('log')
    plt.show()