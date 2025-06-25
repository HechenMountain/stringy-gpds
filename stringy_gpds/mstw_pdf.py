# Dependencies
import csv
import numpy as np
import matplotlib.pyplot as plt

from . import config as cfg
from .helpers import check_evolution_order, check_error_type

############################################
############################################

# Read the CSV file and parse it
data = []
with open(cfg.MSTW_PATH, 'r',newline='') as file:
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

# Create dictionary
MSTW_PDF =  {row[0]: {"lo": row[1], "nlo": row[2], "nnlo": row[3]} for row in data}

####################   
#### Define PDFs ###
#################### 
def get_alpha_s(evolution_order="nlo"):
    """
    Returns alpha_s at the input scale of 1 GeV from the MSTW PDF best fit.
    Note that the MSTW best fit obtains alpha_S(mu=1 GeV**2)=0.68183, different from the world average
    Parameters:
    - evolution_order (str. optional): lo, nlo or nnlo
    """
    check_evolution_order(evolution_order)
    alpha_s_in = MSTW_PDF["alpha_S(Q0^2)"][evolution_order][0]
    return alpha_s_in

def pdf(x,A_pdf,eta_1,eta_2,epsilon,gamma_pdf):
    """
    PDF parametrization for uv, dv, S, g
    """
    """
    Compute the uv, dv, S and g PDF

    Parameters
    ----------
    x : float
        parton x
    A_pdf : float
        Normalization constant of the unpolarized PDF
    eta_1 : float
        Small-x parameter.
    eta_2 : float
        Large-x parameter
    epsilon : float
        sqrt(x) prefactor
    gamma_pdf : float
        Additional linear piece
    evolution_order : str
        "lo", "nlo",...

    Returns
    -------
    float
        The value of the PDF.
    
    Note
    ----
    At nlo the gluon parametrization gets aditional terms. 
    This is handled separately in the gluon PDF:
    """
    result = A_pdf * (1-x)**eta_2*x**(eta_1-1)*(1+epsilon*np.sqrt(x)+gamma_pdf*x)
    # print(A_pdf,eta_1,eta_2,epsilon,gamma_pdf)
    return result

def pdf_error(x,A_pdf,delta_A_pdf,eta_1,delta_eta_1,eta_2,delta_eta_2,
              epsilon,delta_epsilon,gamma_pdf,delta_gamma_pdf,error_type):
    """
    PDF error parametrization for uv, dv, S, g
    """
    """
    Compute the uv, dv, S and g PDF

    Parameters
    ----------
    x : float
        parton x
    A_pdf : float
        Normalization constant of the polarized PDF
    delta_A_pdf : float
        Error of normalization
    eta_1 : float
        Small-x parameter.
    delta_eta_1 : float
        Error of eta_1
    eta_2 : float
        Large-x parameter
    delta_eta_2 : float
        error of eta_2
    epsilon : float
        sqrt(x) prefactor
    delta_epsilon : float
        Error of epsilon
    gamma_pdf : float
        Additional linear piece
    delta_gamma_pdf : float
        Error of gamma_pdf
    evolution_order : str
        "lo", "nlo",...
    error_type : str
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters.

    Returns
    -------
    float
        The value of the PDF.
    
    Note
    ----
    At nlo the gluon parametrization gets aditional terms. 
    This is handled separately in the gluon PDF:
    """
    dpdf_dA_pdf = (1-x)**eta_2*x**(eta_1-1)*(1+epsilon*np.sqrt(x)+gamma_pdf * x)
    dpdf_deta_1 = A_pdf*(1-x)**eta_2*x**(eta_1-1)*(1+epsilon*np.sqrt(x)+gamma_pdf * x)*np.log(x)
    dpdf_deta_2 = A_pdf*(1-x)**eta_2*x**(eta_1-1)*(1+epsilon*np.sqrt(x)+gamma_pdf * x)*np.log(1-x)
    dpdf_depsilon = A_pdf*(1-x)**eta_2*x**(eta_1-.5)
    dpdf_dgamma = A_pdf*(1-x)**eta_2*x**(eta_1)

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
def uv_pdf(x, evolution_order="nlo",error_type="central"):
    """
    Compute the uv PDF and return either its central value or the corresponding value with error.

    Parameters
    ----------
    x : float
        The value of parton x.
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"

    Returns
    -------
    float
        The value of the uv based on the selected parameters and error type.
    """
     # Define a dictionary that maps the error_type to column indices
    error_mapping = {
        "central": 0,  # The column with the central value
        "plus": 1,     # The column with the + error value
        "minus": 2     # The column with the - error value
    }
    check_error_type(error_type)
    check_evolution_order(evolution_order)
    # Get the column index corresponding to the error_type
    error_col_index = error_mapping.get(error_type, 0)  # Default to 'central' if error_type is invalid

    # Extracting parameter values
    A_pdf = MSTW_PDF["A_u"][evolution_order][0]
    eta_1 = MSTW_PDF["eta_1"][evolution_order][0]
    eta_2 = MSTW_PDF["eta_2"][evolution_order][0]
    epsilon = MSTW_PDF["epsilon_u"][evolution_order][0]
    gamma_pdf = MSTW_PDF["gamma_u"][evolution_order][0]

    if error_type == "central":
        result = pdf(x,A_pdf,eta_1,eta_2,epsilon,gamma_pdf)
    else:
    # Extracting errors
        delta_A_pdf  =  MSTW_PDF["A_u"][evolution_order][error_col_index]
        delta_eta_1 = MSTW_PDF["eta_1"][evolution_order][error_col_index]
        delta_eta_2 = MSTW_PDF["eta_2"][evolution_order][error_col_index]
        delta_epsilon = MSTW_PDF["epsilon_u"][evolution_order][error_col_index]
        delta_gamma_pdf = MSTW_PDF["gamma_u"][evolution_order][error_col_index]

        
        result = pdf_error(x,A_pdf,delta_A_pdf,eta_1,delta_eta_1,eta_2,delta_eta_2,epsilon,delta_epsilon,gamma_pdf,delta_gamma_pdf,error_type)
    return result

def dv_pdf(x, evolution_order="nlo",error_type="central"):
    """
    Compute the dv PDF and return either its central value or the corresponding value with error.

    Parameters
    ----------
    x : float
        The value of parton x.
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"

    Returns
    -------
    float
        The value of the dv based on the selected parameters and error type.
    """
    # Define a dictionary that maps the error_type to column indices
    error_mapping = {
        "central": 0,  # The column with the central value
        "plus": 1,     # The column with the + error value
        "minus": 2     # The column with the - error value
    }
    check_error_type(error_type)
    # Get the column index corresponding to the error_type
    error_col_index = error_mapping.get(error_type, 0)

    # Extracting parameter values
    A_pdf = MSTW_PDF["A_d"][evolution_order][0]
    eta_1 = MSTW_PDF["eta_3"][evolution_order][0]
    eta_2 = MSTW_PDF["eta_2"][evolution_order][0] + MSTW_PDF["eta_4-eta_2"][evolution_order][0]  # eta_4 ≡ eta_2 + (eta_4 - eta_2)
    epsilon = MSTW_PDF["epsilon_d"][evolution_order][0]
    gamma_pdf = MSTW_PDF["gamma_d"][evolution_order][0]

    if error_type == "central":
        result = pdf(x,A_pdf,eta_1,eta_2,epsilon,gamma_pdf)
    else:
        # Extracting errors
        delta_A_pdf  = MSTW_PDF["A_d"][evolution_order][error_col_index]
        delta_eta_1 = MSTW_PDF["eta_3"][evolution_order][error_col_index]
        delta_eta_2 = np.sign(MSTW_PDF["eta_4-eta_2"][evolution_order][error_col_index]) * np.sqrt(MSTW_PDF["eta_4-eta_2"][evolution_order][error_col_index]**2 + MSTW_PDF["eta_2"][evolution_order][error_col_index]**2)
        delta_epsilon = MSTW_PDF["epsilon_d"][evolution_order][error_col_index]
        delta_gamma_pdf = MSTW_PDF["gamma_d"][evolution_order][error_col_index]

        result = pdf_error(x,A_pdf,delta_A_pdf,eta_1,delta_eta_1,eta_2,delta_eta_2,epsilon,delta_epsilon,gamma_pdf,delta_gamma_pdf,error_type)
    return result

def sv_pdf(x, evolution_order="nlo",error_type="central"):
    """
    Compute the sv PDF and return either its central value or the corresponding value with error.

    Parameters
    ----------
    x : float
        The value of parton x.
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"

    Returns
    -------
    float
        The value of the sv based on the selected parameters and error type.
    """
    error_mapping = {
        "central": 0,
        "plus": 1,
        "minus": 2
    }
    check_error_type(error_type)
    error_col_index = error_mapping.get(error_type, 0)

    A_m = MSTW_PDF["A_-"][evolution_order][0]
    delta_m = MSTW_PDF["delta_-"][evolution_order][0]
    eta_m = MSTW_PDF["eta_-"][evolution_order][0]
    x_0 =  MSTW_PDF["x_0"][evolution_order][0]

    if error_type == "central":
        result = result = A_m * (x ** (delta_m - 1)) * ((1 - x) ** eta_m) * (1 -x/x_0)
    else:
        # Extracting errors
        delta_A_m  = MSTW_PDF["A_-"][evolution_order][error_col_index]
        delta_delta_m = MSTW_PDF["delta_-"][evolution_order][error_col_index]
        delta_eta_m =MSTW_PDF["eta_-"][evolution_order][error_col_index]
        delta_x_0 = MSTW_PDF["x_0"][evolution_order][error_col_index]

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

def S_pdf(x, evolution_order="nlo",error_type="central"):
    """
    Compute the S PDF and return either its central value or the corresponding value with error.

    Parameters
    ----------
    x : float
        The value of parton x.
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"

    Returns
    -------
    float
        The value of the S based on the selected parameters and error type.
    """
    error_mapping = {
        "central": 0,
        "plus": 1,
        "minus": 2
    }
    check_error_type(error_type)
    error_col_index = error_mapping.get(error_type, 0)

    # Extracting parameter values
    A_pdf      = MSTW_PDF["A_S"][evolution_order][0]
    eta_1      = MSTW_PDF["delta_S"][evolution_order][0]
    eta_2      = MSTW_PDF["eta_S"][evolution_order][0]
    epsilon    = MSTW_PDF["epsilon_S"][evolution_order][0]
    gamma_pdf  = MSTW_PDF["gamma_S"][evolution_order][0]

    if error_type == "central":
        result = pdf(x,A_pdf,eta_1,eta_2,epsilon,gamma_pdf)
    else:
    # Extracting errors
        delta_A_pdf  = MSTW_PDF["A_S"][evolution_order][error_col_index]
        delta_eta_1 = MSTW_PDF["delta_S"][evolution_order][error_col_index]
        delta_eta_2 = MSTW_PDF["eta_S"][evolution_order][error_col_index]
        delta_epsilon = MSTW_PDF["epsilon_S"][evolution_order][error_col_index]
        delta_gamma_pdf = MSTW_PDF["gamma_S"][evolution_order][error_col_index]

        result = pdf_error(x,A_pdf,delta_A_pdf,eta_1,delta_eta_1,eta_2,delta_eta_2,epsilon,delta_epsilon,gamma_pdf,delta_gamma_pdf,error_type)

    return result

def s_plus_pdf(x, evolution_order="nlo",error_type="central"):
    """
    Compute the s_plus PDF and return either its central value or the corresponding value with error.

    Parameters
    ----------
    x : float
        The value of parton x.
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"

    Returns
    -------
    float
        The value of the s_plus based on the selected parameters and error type.
    """
    error_mapping = {
        "central": 0,
        "plus": 1,
        "minus": 2
    }
    check_error_type(error_type)
    error_col_index = error_mapping.get(error_type, 0)

    A_pdf      = MSTW_PDF["A_+"][evolution_order][0]
    eta_1      = MSTW_PDF["delta_S"][evolution_order][0]
    eta_2      = MSTW_PDF["eta_+"][evolution_order][0]
    epsilon    = MSTW_PDF["epsilon_S"][evolution_order][0]
    gamma_pdf  = MSTW_PDF["gamma_S"][evolution_order][0]

    if error_type == "central":
        result = pdf(x,A_pdf,eta_1,eta_2,epsilon,gamma_pdf)
    else:
        # Extracting errors
        delta_A_pdf      = MSTW_PDF["A_+"][evolution_order][error_col_index]
        delta_eta_1      = MSTW_PDF["delta_S"][evolution_order][error_col_index]
        delta_eta_2      = MSTW_PDF["eta_+"][evolution_order][error_col_index]
        delta_epsilon    = MSTW_PDF["epsilon_S"][evolution_order][error_col_index]
        delta_gamma_pdf  = MSTW_PDF["gamma_S"][evolution_order][error_col_index]


        result = pdf_error(x,A_pdf,delta_A_pdf,eta_1,delta_eta_1,eta_2,delta_eta_2,epsilon,delta_epsilon,gamma_pdf,delta_gamma_pdf,error_type)

    return result

def Delta_pdf(x, evolution_order="nlo",error_type="central"):
    """
    Compute the Delta = ubar - dbar PDF and return either its central value or the corresponding value with error.

    Parameters
    ----------
    x : float
        The value of parton x.
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"

    Returns
    -------
    float
        The value of the Delta PDF based on the selected parameters and error type.
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

    A_Delta     = MSTW_PDF["A_Delta"][evolution_order][0]
    eta_Delta   = MSTW_PDF["eta_Delta"][evolution_order][0]
    eta_S       = MSTW_PDF["eta_S"][evolution_order][0]
    delta_Delta = MSTW_PDF["delta_Delta"][evolution_order][0]
    gamma_Delta = MSTW_PDF["gamma_Delta"][evolution_order][0]
    # Compute the Delta(x) PDF
    if error_type == "central":
        result = A_Delta * (x ** (eta_Delta - 1)) * (1 - x) ** (eta_S+2) * (1 + gamma_Delta*x + delta_Delta*x**2)
    else:
        dpdf_dA = (x ** (eta_Delta - 1)) * (1 - x) ** (eta_S+2) * (1 + gamma_Delta*x + delta_Delta*x**2)
        dpdf_deta_Delta = A_Delta * (x ** (eta_Delta - 1)) * (1 - x) ** (eta_S+2) * (1 + gamma_Delta*x + delta_Delta*x**2)*np.log(x)
        dpdf_deta_S = A_Delta * (x ** (eta_Delta - 1)) * (1 - x) ** (eta_S+2) * (1 + gamma_Delta*x + delta_Delta*x**2)*np.log(1-x)
        dpdf_dgamma_Delta = A_Delta * (x ** (eta_Delta)) * (1 - x) ** (eta_S+2)
        dpdf_ddeltaDelta = A_Delta * (x ** (eta_Delta + 1)) * (1 - x) ** (eta_S+2) 
        Delta_A = dpdf_dA * MSTW_PDF["A_Delta"][evolution_order][error_col_index]
        Delta_eta_Delta =  dpdf_deta_Delta * MSTW_PDF["eta_Delta"][evolution_order][error_col_index]
        Delta_eta_S = dpdf_deta_S * MSTW_PDF["eta_S"][evolution_order][error_col_index]
        Delta_gamma_Delta = dpdf_dgamma_Delta * MSTW_PDF["gamma_Delta"][evolution_order][error_col_index]
        Delta_delta_Delta = dpdf_ddeltaDelta * MSTW_PDF["delta_Delta"][evolution_order][error_col_index]
        result = np.sqrt(Delta_A**2 + Delta_delta_Delta**2 + Delta_eta_Delta**2 +Delta_eta_S**2+Delta_gamma_Delta**2 +Delta_delta_Delta**2)
    
    return result

def gluon_pdf(x, evolution_order="nlo",error_type="central"):
    """
    Compute the gluon PDF and return either its central value or the corresponding value with error.

    Parameters
    ----------
    x : float
        The value of parton x.
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"

    Returns
    -------
    float
        The value of the gluon based on the selected parameters and error type.
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

    # Extracting parameter values
    A_pdf     = MSTW_PDF["A_g"][evolution_order][0]
    eta_1     = MSTW_PDF["delta_g"][evolution_order][0]
    eta_2     = MSTW_PDF["eta_g"][evolution_order][0]
    epsilon   = MSTW_PDF["epsilon_g"][evolution_order][0]
    gamma_pdf = MSTW_PDF["gamma_g"][evolution_order][0]

    A_pdf_prime     = MSTW_PDF["A_g'"][evolution_order][0]
    eta_1_prime     = MSTW_PDF["delta_g'"][evolution_order][0]
    eta_2_prime     = MSTW_PDF["eta_g'"][evolution_order][0]

    if error_type == "central":
        result = pdf(x,A_pdf,eta_1,eta_2,epsilon,gamma_pdf)
        if evolution_order != "lo":
            result += pdf(x,A_pdf_prime,eta_1_prime,eta_2_prime,0,0)
    else:
        # Extracting errors
        delta_A_pdf      = MSTW_PDF["A_g"][evolution_order][error_col_index]
        delta_eta_1      = MSTW_PDF["delta_g"][evolution_order][error_col_index]
        delta_eta_2      = MSTW_PDF["eta_g"][evolution_order][error_col_index]
        delta_epsilon    = MSTW_PDF["epsilon_g"][evolution_order][error_col_index]
        delta_gamma_pdf  = MSTW_PDF["gamma_g"][evolution_order][error_col_index]
        result = pdf_error(x,A_pdf,delta_A_pdf,eta_1,delta_eta_1,eta_2,delta_eta_2,epsilon,delta_epsilon,gamma_pdf,delta_gamma_pdf,error_type)
        if evolution_order != "lo":
            # Extracting errors
            delta_A_prime_pdf     = MSTW_PDF["A_g'"][evolution_order][error_col_index]
            delta_eta_1_prime     = MSTW_PDF["delta_g'"][evolution_order][error_col_index]
            delta_eta_2_prime     = MSTW_PDF["eta_g'"][evolution_order][error_col_index]
            dpdf_dA_pdf = (1-x)**eta_2_prime*x**(eta_1_prime-1)
            dpdf_deta_1 = A_pdf_prime*(1-x)**eta_2_prime*x**(eta_1_prime-1)*np.log(x)
            dpdf_deta_2 = A_pdf_prime*(1-x)**eta_2_prime*x**(eta_1_prime-1)*np.log(1-x)

            #print(dpdf_dA_pdf,dpdf_deta_1,dpdf_deta_2)
            Delta_A_prime = dpdf_dA_pdf * delta_A_prime_pdf
            Delta_eta_1_prime = dpdf_deta_1 * delta_eta_1_prime
            Delta_eta_2_prime = dpdf_deta_2 * delta_eta_2_prime

            result += np.sqrt(Delta_A_prime**2+Delta_eta_1_prime**2+Delta_eta_2_prime**2)
    return result

def uv_minus_dv_pdf(x, evolution_order="nlo",error_type="central"):
    """
    Compute the uv - dv (non_singlet_isovector) PDF and return either its central value or the corresponding value with error.

    Parameters
    ----------
    x : float
        The value of parton x.
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"

    Returns
    -------
    float
        The value of the uv - dv PDF based on the selected parameters and error type.
    """
    uv = uv_pdf(x,evolution_order,error_type)
    dv= dv_pdf(x,evolution_order,error_type)
    if error_type == "central":
        result = uv-dv
    else:
        result = np.sqrt(uv**2+dv**2)
    return result

def uv_plus_dv_plus_S_pdf(x, evolution_order="nlo",error_type="central"):
    """
    Compute the uv + dv + S (quark singlet) PDF and return either its central value or the corresponding value with error.

    Parameters
    ----------
    x : float
        The value of parton x.
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"

    Returns
    -------
    float
        The value of the quark singlet PDF based on the selected parameters and error type.
    """
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

def plot_uv_pdf(x_0=1e-2,evolution_order="nlo",logplot=False,error_bars=True):
    """
    Plot the uv PDF over x.

    Parameters
    ----------
    x_0 : float, optional
        The value of minimum value of parton x. Default is 1e-2
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    logplot : bool, optional
        Whether to use a logarithmic scale on the x-axis. Default is False.
    error_bars : bool, optional
        Whether to display error bars corresponding to PDF uncertainties. Default is True.
    """
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

def plot_dv_pdf(x_0=1e-2,evolution_order="nlo",logplot=False,error_bars=True):
    """
    Plot the dv PDF over x.

    Parameters
    ----------
    x_0 : float, optional
        The value of minimum value of parton x. Default is 1e-2
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    logplot : bool, optional
        Whether to use a logarithmic scale on the x-axis. Default is False.
    error_bars : bool, optional
        Whether to display error bars corresponding to PDF uncertainties. Default is True.
    """
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

def plot_uv_minus_dv_pdf(x_0=1e-2,evolution_order="nlo",logplot=False,error_bars=True):
    """
    Plot the uv - dv (non_singlet_isovector) PDF over x.

    Parameters
    ----------
    x_0 : float, optional
        The value of minimum value of parton x. Default is 1e-2
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    logplot : bool, optional
        Whether to use a logarithmic scale on the x-axis. Default is False.
    error_bars : bool, optional
        Whether to display error bars corresponding to PDF uncertainties. Default is True.
    """
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

def plot_uv_plus_dv_plus_S_pdf(x_0=1e-2,evolution_order="nlo",logplot=False,error_bars=True):
    """
    Plot the uv + dv + S (quark singlet) PDF over x.

    Parameters
    ----------
    x_0 : float, optional
        The value of minimum value of parton x. Default is 1e-2
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    logplot : bool, optional
        Whether to use a logarithmic scale on the x-axis. Default is False.
    error_bars : bool, optional
        Whether to display error bars corresponding to PDF uncertainties. Default is True.
    """
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

def plot_gluon_pdf(x_0=1e-2,evolution_order="nlo",logplot=False,error_bars=True):
    """
    Plot the gluon PDF over x.

    Parameters
    ----------
    x_0 : float, optional
        The value of minimum value of parton x. Default is 1e-2
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    logplot : bool, optional
        Whether to use a logarithmic scale on the x-axis. Default is False.
    error_bars : bool, optional
        Whether to display error bars corresponding to PDF uncertainties. Default is True.
    """
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
