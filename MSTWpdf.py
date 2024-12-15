# Dependencies
import pandas as pd
import numpy as np

# Set display options
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000)        # Set width to avoid wrapping
pd.set_option('display.max_colwidth', None) # Show full content of each column

############################################
############################################
# Import MSTW PDF data
# Base path to main data directory
base_path = "/mnt/c/Users/flori/Documents/PostDoc/Data/GPD/"
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

# Extract LO, NLO and NLO columns
MSTWpdf_LO=MSTWpdf[["LO"]]
MSTWpdf_NLO=MSTWpdf[["NLO"]]
MSTWpdf_NNLO=MSTWpdf[["NNLO"]]

############################################
############################################

# Define the PDFs using Eqs. (6-12) in  0901.0002 
def uv_PDF(x, error_type="central"):
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
    
    # Compute the uv(x) PDF
    result = A_u * (x ** (eta_1 - 1)) * ((1 - x) ** eta_2) * (1 + epsilon_u * np.sqrt(x) + gamma_u * x)
    
    return result

def dv_PDF(x, error_type="central"):
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
    
    # Compute the dv(x) PDF
    result = A_d * (x ** (eta_3 - 1)) * ((1 - x) ** eta_4) * (1 + epsilon_d * np.sqrt(x) + gamma_d * x)
    
    return result

def sv_PDF(x, error_type="central"):
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

def Sv_PDF(x, error_type="central"):
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

def s_plus_PDF(x, error_type="central"):
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

def Delta_PDF(x, error_type="central"):
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


    # Compute the Delta(x) PDF
    result = A_Delta * (x ** (eta_Delta - 1)) * (1 - x) ** (eta_S+2) * (1 + gamma_Delta*x + delta_Delta*x**2)
    
    return result

def gluon_PDF(x, error_type="central"):
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

    # Compute the g(x) PDF
    # At LO this is the full expression
    result = A_g * (x ** (delta_g - 1)) * ((1 - x) ** eta_g) * (1 + epsilon_g * np.sqrt(x) + gamma_g * x)
    
    return result 