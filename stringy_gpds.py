# # Dependencies
import numpy as np
import mpmath as mp
import sympy
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.integrate import quad, trapezoid, odeint
from joblib import Parallel, delayed
from scipy.special import gamma, digamma
from scipy.interpolate import interp1d, RectBivariateSpline
import time
import re
import os

from mstw_pdf import MSTW_PDF
from aac_pdf import AAC_PDF

########################################
#### Currently enforced assumptions ####
########################################
# singlet_moment for B GPD set to zero #
# Normalizations of isoscalar_moment   #
# ubar = dbar                          #
# Delta_u_bar = Delta_s_bar = Delta_s  #
########################################




########################################
#### Dictionaries and data handling ####
####       Change as required       ####
########################################

# Parent directory for data
BASE_PATH = "/mnt/c/Users/flori/Documents/PostDoc/Data/stringy-gpds/"
# Subdirectories for cleaner file handling
IMPACT_PARAMETER_MOMENTS_PATH = BASE_PATH + "ImpactParameterMoments/"
MOMENTUM_SPACE_MOMENTS_PATH = BASE_PATH + "MomentumSpaceMoments/"
GPD_PATH = BASE_PATH + "GPDs/"
# Folder for generated plots
PLOT_PATH =  "/mnt/c/Users/flori/Documents/PostDoc/Plots/stringy-gpds/"

# Add some colors
saturated_pink = (1.0, 0.1, 0.6)  


PUBLICATION_MAPPING = {
    "2305.11117": ("cyan",2),
    "0705.4295": ("orange",2),
    "1908.10706": (saturated_pink,2),
    "2310.08484": ("darkblue",2),
    "2410.03539": ("green",2)
# Add more publication IDs and corresponding colors here
}

GPD_PUBLICATION_MAPPING = {
    # publication ID, GPD type, GPD label, eta, t ,mu
    ("2008.10573","non_singlet_isovector","Htilde",0.00, -0.69, 2.00): ("mediumturquoise","000_069_200"),
    ("2008.10573","non_singlet_isovector","Htilde",0.33, -0.69, 2.00): ("green","033_069_200"),
    ("2112.07519","non_singlet_isovector","Htilde",0.00, -0.39, 3.00): ("purple","000_039_300"),
    ("2008.10573","non_singlet_isovector","E",0.00, -0.69, 2.00): ("mediumturquoise","000_069_200"),
    ("2008.10573","non_singlet_isovector","E",0.33, -0.69, 2.00): ("green","033_069_200"),
    ("2312.10829","non_singlet_isovector","E",0.10, -0.23, 2.00): ("orange","010_023_200"),
    # No data:
    # ("","non_singlet_isoscalar","E",0.00, -0.00, 2.00): ("purple","000_000_200"),
    # ("","non_singlet_isoscalar","E",0.33, -0.69, 2.00): ("green","033_069_200"),
    # ("","non_singlet_isoscalar","E",0.10, -0.23, 2.00): ("orange","010_023_200"),
    # ("","non_singlet_isoscalar","E",0.33, -0.69, 2.00): (saturated_pink,"000_039_200"),
# Add more publication IDs and corresponding colors here
}

GPD_LABEL_MAP ={"H": "A",
                "E": "B",
                "Htilde": "Atilde"
                    }

def initialize_moment_to_function():
    # Define dictionary that maps conformal moments names and types to expressions
    global MOMENT_TO_FUNCTION
    MOMENT_TO_FUNCTION = {
    # Contains a Pair of moment_type and moment_label to match input PDF and evolution type
    ("non_singlet_isovector", "A"): (non_singlet_isovector_moment,"vector"),
    ("non_singlet_isovector", "B"): (non_singlet_isovector_moment,"vector"),
    ("non_singlet_isovector", "Atilde"): (non_singlet_isovector_moment,"axial"),
    ("non_singlet_isoscalar", "A"): (non_singlet_isoscalar_moment,"vector"),
    ("non_singlet_isoscalar", "B"): (non_singlet_isoscalar_moment,"vector"),
    ("non_singlet_isoscalar", "Atilde"): (non_singlet_isoscalar_moment,"axial"),
    ("singlet","A"): (singlet_moment, "vector"),
    ("singlet","B"): (singlet_moment, "vector"),
    ("singlet","Atilde"): (singlet_moment, "axial"),
    }

def get_regge_slope(moment_type,moment_label,evolve_type):
    """Set Regge slopes, modify manually
    
    Parameters:
    - moment_type (str.): non_singlet_isovector, non_singlet_isoscalar, singlet
    - moment_label (str.): A, Atilde...
    - evolve_type (str.): Type of evolution equation
    """
    check_moment_type_label(moment_type,moment_label)
    check_evolve_type(evolve_type)

    if evolve_type == "vector":
        if moment_type == "non_singlet_isovector":
            if moment_label == "A":
                # Value from the paper
                # alpha_prime = 1.069
                # Best fit
                alpha_prime = 0.658
                return alpha_prime
            if moment_label == "B":
                alpha_prime = 1.460
                return alpha_prime                
        if moment_type == "non_singlet_isoscalar":
            if moment_label == "A":
                # Value from the paper
                # alpha_prime = 0.891
                # Best fit
                alpha_prime = 0.957
                return alpha_prime
            if moment_label == "B":
                alpha_prime = 1.13
                return alpha_prime
        if moment_type == "singlet":
            if moment_label in ["A","B"]:
                alpha_prime_s = 1.828
                alpha_prime_T = 0.627
                alpha_prime_S = 4.277
                return alpha_prime_s, alpha_prime_T, alpha_prime_S
    elif evolve_type == "axial":
        if moment_type == "non_singlet_isovector":
            if moment_label == "Atilde":
                alpha_prime = 0.454387
                return alpha_prime
        if moment_type == "non_singlet_isoscalar":
            if moment_label == "Atilde":
                alpha_prime = 0.297522
                return alpha_prime
        if moment_type == "singlet":
            if moment_label in ["Atilde","Btilde"]:
                # Assuming that the exchange is solely
                # carried by the eta prime meson trajectory
                alpha_prime_s = 1.179
                alpha_prime_PV = 0.490
                alpha_prime_PS = 0.744
                return alpha_prime_s, alpha_prime_PV, alpha_prime_PS
    else:
        raise ValueError(f"Evolve type {evolve_type} for moment {moment_type} with label {moment_label} unavailable.")
            

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
    moment_path = MOMENTUM_SPACE_MOMENTS_PATH
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

def save_gpd_data(x_values, eta, t, mu,y_values,particle="quark",gpd_type="non_singlet_isovector",gpd_label="H",error_type="central"):
    """
    Save the function f(x, eta, t, mu) evaluated at x_values to a CSV file.
    """
    if len(x_values) != len(y_values):
        raise ValueError(f"x_values ({len(x_values)}) and y_values({len(y_values)}) are of unequal length")

    gpd_name = gpd_type + "_" + particle + "_GPD_" + gpd_label
    gpd_path = BASE_PATH + "/GPDs/"

    filename = os.path.join(gpd_path, generate_filename(eta, t, mu,gpd_name,error_type))
    data = np.column_stack((x_values, y_values))
    np.savetxt(filename, data, delimiter=",")
    print(f"Saved data to {filename}")

def load_gpd_data(eta, t, mu,particle="quark",gpd_type="non_singlet_isovector",gpd_label="H",error_type ="central"):
    """
    Load data from CSV if it exists, otherwise return None.
    """

    gpd_name = gpd_type + "_" + particle + "_GPD_" + gpd_label
    gpd_path = BASE_PATH + "/GPDs/"
    filename = os.path.join(gpd_path, generate_filename(eta, t, mu,gpd_name,error_type))
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
    if (pub_id,gpd_type,gpd_label,eta_in,t_in,mu_in) in GPD_PUBLICATION_MAPPING:
        (color,parameter_set) = GPD_PUBLICATION_MAPPING[(pub_id,gpd_type,gpd_label,eta_in,t_in,mu_in)]
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
    gpd_path = BASE_PATH + "/GPDs/"
    FILE_NAME = f"{gpd_name}_{pub_id}_{parameter_set}{error}.csv"
    FILE_PATH = f"{gpd_path}{FILE_NAME}"

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
#### Helper functions ####
##########################

def check_evolution_order(evolution_order):
    if evolution_order not in ["LO","NLO","NNLO"]:
        raise ValueError(f"Wrong evolution_order {evolution_order} for evolution equation")

def get_alpha_s(evolution_order="LO"):
    """
    Returns alpha_s at the input scale of 1 GeV from the MSTW PDF best fit.
    Note that the MSTW best fit obtains alpha_S(mu=1 GeV**2)=0.68183, different from the world average
    Parameters:
    - evolution_order (str. optional): LO, NLO or NNLO
    """
    check_evolution_order(evolution_order)
    index_alpha_s= MSTW_PDF[MSTW_PDF["Parameter"] == "alpha_S(Q0^2)"].index[0]
    alpha_s_in = MSTW_PDF[[evolution_order]].iloc[index_alpha_s,0][0]
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

def error_sign(error,error_type):
    check_error_type(error_type)
    if error_type == "minus":
        return -error
    else:
        return error

#####################################
### Input for Evolution Equations ###
#####################################
def evolve_alpha_s(mu, Nf = 3,evolution_order="LO"):
    """
    Evolve alpha_S=g**/(4pi) from some input scale mu_in to some other scale mu.
    Note that the MSTW best fit obtains alpha_S(mu=1 GeV**2)=0.68183, different from the world average
    
    Arguments:
    mu (float):  The momentum scale of the process
    Nf (int. optional) Number of active flavors
    evolution_order (str. optional): LO, NLO...
    Returns:
    The evolved value of alpha_s at mu**2
    """
    # Set parameters
    Nc = 3
    c_a = Nc
    mu_R = 1 # 1 GeV
    # Extract value of alpha_S at the renormalization point of mu_R**2 = 1 GeV**2
    alpha_s_in = get_alpha_s()

    if mu_R == mu:
        return alpha_s_in

    beta_0 = 2/3* Nf - 11/3 * c_a
    if evolution_order == "LO":
        beta_1 = 0
    else:
        c_f = (Nc**2-1)/(2*Nc)
        beta_1 = 10/3 * c_a * Nf + 2 * c_f * Nf -34/3 * c_a**2

    # Define the differential equation
    def beta_function(alpha_s, Q2):
        # alpha_s is the value of alpha_s/(4*pi) and ln_Q2 is the logarithmic scale
        d_alpha_s = (beta_0 * alpha_s**2/(4*np.pi) + beta_1 * alpha_s**3/(4*np.pi)**2)/Q2
        return d_alpha_s

    if evolution_order == "LO":
        log_term = np.log(mu**2 / mu_R**2)
        denominator = 1 - (alpha_s_in / (4 * np.pi)) * beta_0 * log_term
        result = alpha_s_in / denominator
    else:
        Q2_values = np.linspace(1,mu**2, 10000)

        solution, info = odeint(beta_function, alpha_s_in, Q2_values,full_output = 1)

        # Check if there was a problem with integration
        if info['message'] != 'Integration successful.':
            print("Integration Warning:", info)
            return None
        
        result = solution[-1]
        result = result[0]

    return result

def integral_pdf_regge(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,j,alpha_p,t):
        """ Returns the result of the integral of a Reggeized PDF \int pdf(x) x**(j-1 - alpha_p * t )
        of the form
        pdf(x) = A_pdf * x**eta_1(1-x)**eta_2 * ( 1 + epsilon * sqrt(x) + gamma_pdf * x )

        Parameters:
        - A_pdf (float): Magnitude of the input PDF.
        - eta_1 (float): Exponent determining low-x behavior of input PDF.
        - eta_2 (float): Exponent determining the large-x behavior of input PDF.
        - epsilon (float): Prefacotr of sqrt(x) determining intermetiate behavior of input PDF.
        - gamma_pdf (float): Prefactor of x determining intermetiate behavior of input PDF.
        - j (float): conformal spin.
        - alpha_p (float): Regge slope.
        - t (float): Mandelstam t (< 0 in physical region)
        """
        frac_1 = epsilon*gamma(eta_1+j-alpha_p*t -.5)/(gamma(eta_1+eta_2+j-alpha_p*t+.5))
        # frac_2 = (eta_1+eta_2-gamma_pdf+eta_1*gamma_pdf+j*(1+gamma_pdf)-(1+gamma_pdf)*alpha_p*t)*gamma(eta_1+j-alpha_p*t-1)/gamma(1+eta_1+eta_2+j-alpha_p*t)
        frac_2 = (eta_1+eta_2-gamma_pdf+eta_1*gamma_pdf+j*(1+gamma_pdf)+(1+gamma_pdf)*alpha_p*t)*gamma(eta_1+j-alpha_p*t-1)/gamma(1+eta_1+eta_2+j-alpha_p*t)
        result = A_pdf*gamma(1+eta_2)*(frac_1+frac_2)
        return result

def integral_polarized_pdf_regge(
                A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
                Delta_A_pdf,alpha,gamma_pol, lambda_pol,
                j,alpha_p,t,evolution_order="LO"
                ):
        """ Returns the result of the integral of a Reggeized polarized PDF \int polarized_pdf(x) x**(j-1 - alpha_p * t )
        of the form
        polarized_pdf(x) = A_pdf * x**alpha * ( 1 + gamma_pol * x**lambda_pol ) pdf_(x).
        The input PDF is of the form
        pdf(x) = A_pdf * x**eta_1(1-x)**eta_2 * ( 1 + epsilon * sqrt(x) + gamma_pdf * x )
        and taken at its central value, without error.

        Parameters:
        - A_pdf (float): Magnitude of the input PDF.
        - eta_1 (float): Exponent determining low-x behavior of input PDF.
        - eta_2 (float): Exponent determining the large-x behavior of input PDF.
        - epsilon (float): Prefactor of sqrt(x) determining intermediate behavior of input PDF.
        - gamma_pdf (float): Prefactor of x determining intermediate behavior of input PDF.
        - Delta_A_pdf (float): Magnitude of the polarized input PDF.
        - alpha (float):  Exponent determining low-x behavior of input polarized PDF.
        - gamma_pol (float): Parametrizing intermediate-x behavior of polarized PDF.
        - lambda_pol (float): Parametrizing intermediate-x behavior of polarized PDF.
        - j (float): conformal spin.
        - alpha_p (float): Regge slope.
        - t (float): Mandelstam t (< 0 in physical region)
        - evolution_order (str. optional): LO, NLO, NNLO
        """
        check_evolution_order(evolution_order)
        term1 = (
                A_pdf * Delta_A_pdf * gamma(eta_2 + 1) * (
                (
                        (gamma_pol * epsilon * gamma(eta_1 + j - alpha_p * t + alpha + lambda_pol - 0.5))
                        / gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + lambda_pol + 0.5)
                )
                + (
                        (epsilon * gamma(eta_1 + j - alpha_p * t + alpha - 0.5))
                        / gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + 0.5)
                )
                + (
                        (gamma_pol * gamma(eta_1 + j - alpha_p * t + alpha + lambda_pol - 1))
                        * (
                        alpha + lambda_pol + eta_1 * (gamma_pdf + 1)
                        + eta_2 + gamma_pdf * (alpha + lambda_pol + j - alpha_p * t - 1)
                        + j - alpha_p * t
                        )
                        / gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + lambda_pol + 1)
                )
                + (
                        (gamma(eta_1 + j - alpha_p * t + alpha - 1))
                        * (
                        alpha + eta_1 * (gamma_pdf + 1)
                        + eta_2 + gamma_pdf * (alpha + j - alpha_p * t - 1)
                        + j - alpha_p * t
                        )
                        / gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + 1)
                )
                )
        )

        if evolution_order == "LO":
            result = term1
        elif evolution_order == "NLO":
            term2 = - A_pdf * Delta_A_pdf * gamma_pol * gamma(eta_2 + 1)*(
                        (epsilon * gamma(eta_1 + j - alpha_p * t + alpha - 0.5))
                        / gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + 0.5)
                        + (gamma(eta_1 + j - alpha_p * t + alpha - 1)
                        * (
                        alpha + eta_1 * (gamma_pdf + 1)
                        + eta_2 + gamma_pdf * (alpha + j - alpha_p * t - 1)
                        + j - alpha_p * t)
                        )
                        / gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + 1)
            )
            result =term1+term2
        else:
            raise ValueError("Currently unsupported evolution type")
        
        return result

def integral_pdf_regge_error(A_pdf,delta_A_pdf,eta_1,delta_eta_1,eta_2,delta_eta_2,epsilon,delta_epsilon,gamma_pdf,delta_gamma_pdf,j,alpha_p,t, error_type="central"):
        """ Returns the result of error added in quadrature of the integral of a 
        Reggeized PDF f = \int pdf(x) x**(j-1 - alpha_p * t ) 
        error_pdf = sqrt(df/dA_pdf**2 * delta_A_pdf**2 + df/deta_1**2 * delta_eta_1**2 + ...)
        of the form 
        pdf(x) = A_pdf * x**eta_1(1-x)**eta_2 * ( 1 + epsilon * sqrt(x) + gamma_pdf * x )

        Parameters:
        - A_pdf (float): Magnitude of the input PDF.
        - delta_A_pdf (float): Error of A_pdf
        - eta_1 (float): Exponent determining low-x behavior of input PDF.
        - delta_eta_1 (float): Error of eta_1
        - eta_2 (float): Exponent determining the large-x behavior of input PDF.
        - delta_eta_2 (float): Error of eta_2
        - epsilon (float): Prefacotr of sqrt(x) determining intermetiate behavior of input PDF.
        - delta_epsilon (float): Error of delta_epsilon
        - gamma_pdf (float): Prefactor of x determining intermetiate behavior of input PDF.
        - delta_gamma_pdf (float): Error of gamma_pdf
        - j (float): conformal spin.
        - alpha_p (float): Regge slope.
        - t (float): Mandelstam t (< 0 in physical region)
        
        Returns:
        0 if the central value of the input PDF is picked.
        +- error_pdf if error_type ="plus" or "minus".
        """
        check_error_type(error_type)
        if error_type == "central":
                return 0
        def dpdf_dA_pdf(A_pdf, epsilon, eta_1, eta_2, j, t, alpha_p, gamma_pdf):
                frac_1 = epsilon*gamma(eta_1+j-alpha_p*t -.5)/(gamma(eta_1+eta_2+j-alpha_p*t+.5))
                frac_2 = (eta_1+eta_2-gamma_pdf+eta_1*gamma_pdf+j*(1+gamma_pdf)-(1+gamma_pdf)*alpha_p*t)*gamma(eta_1+j-alpha_p*t-1)/gamma(1+eta_1+eta_2+j-alpha_p*t)
                result = gamma(1+eta_2)*(frac_1+frac_2)
                return result

        def dpdf_deta_1(A_pdf, epsilon, eta_1, eta_2, j, t, alpha_p, gamma_pdf):
                term_1 = (epsilon * gamma(eta_1 + j - t * alpha_p - 0.5) * 
                digamma(eta_1 + j - t * alpha_p - 0.5) / 
                gamma(eta_1 + eta_2 + j - t * alpha_p + 0.5))

                term_2 = (epsilon * gamma(eta_1 + j - t * alpha_p - 0.5) * 
                        digamma(eta_1 + eta_2 + j - t * alpha_p + 0.5) / 
                        gamma(eta_1 + eta_2 + j - t * alpha_p + 0.5))

                term_3 = ((gamma_pdf + 1) * gamma(eta_1 + j - t * alpha_p - 1) / 
                        gamma(eta_1 + eta_2 + j - t * alpha_p + 1))

                factor = (eta_1 * (gamma_pdf + 1) + eta_2 + 
                        gamma_pdf * (j - alpha_p * t - 1) + j - alpha_p * t)

                term_4 = (gamma(eta_1 + j - t * alpha_p - 1) * digamma(eta_1 + j - t * alpha_p - 1) * factor /
                        gamma(eta_1 + eta_2 + j - t * alpha_p + 1))

                term_5 = (gamma(eta_1 + j - t * alpha_p - 1) * factor * 
                        digamma(eta_1 + eta_2 + j - t * alpha_p + 1) /
                        gamma(eta_1 + eta_2 + j - t * alpha_p + 1))

                return A_pdf * gamma(eta_2 + 1) * (term_1 - term_2 + term_3 + term_4 - term_5)
        def dpdf_deta_2(A_pdf, epsilon, eta_1, eta_2, j, t, alpha_p, gamma_pdf):
                term_1 = (epsilon * gamma(eta_1 + j - t * alpha_p - 0.5) / 
                        gamma(eta_1 + eta_2 + j - t * alpha_p + 0.5))

                factor = (eta_1 * (gamma_pdf + 1) + eta_2 + 
                        gamma_pdf * (j - alpha_p * t - 1) + j - alpha_p * t)

                term_2 = (gamma(eta_1 + j - t * alpha_p - 1) * factor / 
                        gamma(eta_1 + eta_2 + j - t * alpha_p + 1))

                term_3 = (-epsilon * gamma(eta_1 + j - t * alpha_p - 0.5) * 
                        digamma(eta_1 + eta_2 + j - t * alpha_p + 0.5) / 
                        gamma(eta_1 + eta_2 + j - t * alpha_p + 0.5))

                term_4 = (-gamma(eta_1 + j - t * alpha_p - 1) * factor * 
                        digamma(eta_1 + eta_2 + j - t * alpha_p + 1) /
                        gamma(eta_1 + eta_2 + j - t * alpha_p + 1))

                term_5 = (gamma(eta_1 + j - t * alpha_p - 1) /
                        gamma(eta_1 + eta_2 + j - t * alpha_p + 1))

                return (A_pdf * gamma(eta_2 + 1) * digamma(eta_2 + 1) * (term_1 + term_2) +
                        A_pdf * gamma(eta_2 + 1) * (term_3 + term_4 + term_5))

        def dpdf_depsilon(A_pdf, epsilon, eta_1, eta_2, j, t, alpha_p, gamma_pdf):
                term1 = A_pdf * gamma(eta_2 + 1) * gamma(eta_1 + j - alpha_p * t - 0.5)
                term2 = gamma(eta_1 + eta_2 + j - alpha_p * t + 0.5)
                return term1/term2
        def dpdf_dgamma(A_pdf, epsilon, eta_1, eta_2, j, t, alpha_p, gamma_pdf):
                term1 = A_pdf * gamma(eta_2 + 1) * (eta_1 + j - alpha_p * t - 1) * gamma(eta_1 + j - t * alpha_p - 1)
                term2 = gamma(eta_1 + eta_2 + j - t * alpha_p + 1)
                return term1/term2
        Delta_A_pdf = dpdf_dA_pdf(A_pdf, epsilon, eta_1, eta_2, j, t, alpha_p, gamma_pdf) * delta_A_pdf
        Delta_eta_1 = dpdf_deta_1(A_pdf, epsilon, eta_1, eta_2, j, t, alpha_p, gamma_pdf) * delta_eta_1
        Delta_eta_2 = dpdf_deta_2(A_pdf, epsilon, eta_1, eta_2, j, t, alpha_p, gamma_pdf) * delta_eta_2
        Delta_epsilon = dpdf_depsilon(A_pdf, epsilon, eta_1, eta_2, j, t, alpha_p, gamma_pdf) * delta_epsilon
        Delta_gamma_pdf = dpdf_dgamma(A_pdf, epsilon, eta_1, eta_2, j, t, alpha_p, gamma_pdf) * delta_gamma_pdf
        # Debug
        #print(Delta_A_pdf,Delta_eta_1,Delta_eta_2,Delta_epsilon,Delta_gamma_pdf)
        result = np.sqrt(Delta_A_pdf**2+Delta_eta_1**2+Delta_eta_2**2+Delta_epsilon**2+Delta_gamma_pdf**2)
        return result


def integral_polarized_pdf_regge_error(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
                                       Delta_A_pdf,err_Delta_A_pdf,alpha,err_alpha,gamma_pol,err_gamma_pol, lambda_pol,err_lambda_pol,
                                       j,alpha_p,t,evolution_order="LO", error_type="central"):
        check_evolution_order(evolution_order)
        if error_type == "central":
                return 0
        def dpol_pdf_dDelta_A_pdf(
                        A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
                        Delta_A_pdf,alpha,gamma_pol, lambda_pol,
                        j,alpha_p,t
        ):
                term1 = A_pdf * gamma(eta_2 + 1)
                
                term2 = (gamma_pol * epsilon * gamma(eta_1 + j - alpha_p * t + alpha + lambda_pol - 0.5)) / \
                        gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + lambda_pol + 0.5)
                
                term3 = (epsilon * gamma(eta_1 + j - alpha_p * t + alpha - 0.5)) / \
                        gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + 0.5)
                
                term4 = (gamma(eta_1 + j - alpha_p * t + alpha + lambda_pol - 1) * 
                        (gamma_pol + (gamma_pol * gamma_pdf * 
                        (alpha + lambda_pol + eta_1 + j - alpha_p * t - 1)) / 
                        (alpha + lambda_pol + eta_1 + eta_2 + j - alpha_p * t))) / \
                        gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + lambda_pol)
                
                term5 = (gamma(eta_1 + j - alpha_p * t + alpha - 1) * 
                        ((gamma_pdf * (alpha + eta_1 + j - alpha_p * t - 1)) / 
                        (alpha + eta_1 + eta_2 + j - alpha_p * t) + 1)) / \
                        gamma(eta_1 + eta_2 + j - alpha_p * t + alpha)
                if evolution_order == "LO":
                    term6 = 0
                elif evolution_order == "NLO":
                    term6 = - gamma_pol * (
                        term3 +
                        gamma(eta_1 + j - alpha_p * t + alpha - 1)/gamma(1+ eta_1 + eta_2 + j - alpha_p * t + alpha) * \
                        ((eta_2+eta_1*(1+gamma_pdf)+j -alpha_p * t + alpha + gamma_pdf * (-1+j-alpha_p *t + alpha)))
                    )
                
                return term1 * (term2 + term3 + term4 + term5 + term6)
        def dpol_pdf_dalpha(
                        A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
                        Delta_A_pdf,alpha,gamma_pol, lambda_pol,
                        j,alpha_p,t
        ):
                term1 = A_pdf * Delta_A_pdf * gamma(eta_2 + 1)
    
                term2 = (gamma_pol * epsilon * gamma(eta_1 + j - alpha_p * t + alpha + lambda_pol - 0.5) 
                        * digamma(eta_1 + j - alpha_p * t + alpha + lambda_pol - 0.5)) \
                        / gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + lambda_pol + 0.5)
                
                term3 = - (gamma_pol * epsilon * gamma(eta_1 + j - alpha_p * t + alpha + lambda_pol - 0.5) 
                                * digamma(eta_1 + eta_2 + j - alpha_p * t + alpha + lambda_pol + 0.5)) \
                        / gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + lambda_pol + 0.5)
                
                term4 = (epsilon * gamma(eta_1 + j - alpha_p * t + alpha - 0.5) 
                        * digamma(eta_1 + j - alpha_p * t + alpha - 0.5)) \
                        / gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + 0.5)
                
                term5 = - (epsilon * gamma(eta_1 + j - alpha_p * t + alpha - 0.5) 
                        * digamma(eta_1 + eta_2 + j - alpha_p * t + alpha + 0.5)) \
                        / gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + 0.5)
                
                term6 = (gamma_pol * (eta_2 + 1) * gamma_pdf * gamma(eta_1 + j - alpha_p * t + alpha + lambda_pol - 1)) \
                        / ((alpha + lambda_pol + eta_1 + eta_2 + j - alpha_p * t)**2 
                        * gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + lambda_pol))
                
                term7 = (gamma(eta_1 + j - alpha_p * t + alpha + lambda_pol - 1) 
                        * digamma(eta_1 + j - alpha_p * t + alpha + lambda_pol - 1)
                        * (gamma_pol + (gamma_pol * gamma_pdf * (alpha + lambda_pol + eta_1 + j - alpha_p * t - 1))
                                / (alpha + lambda_pol + eta_1 + eta_2 + j - alpha_p * t))) \
                        / gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + lambda_pol)
                
                term8 = - (gamma(eta_1 + j - alpha_p * t + alpha + lambda_pol - 1) 
                                * digamma(eta_1 + eta_2 + j - alpha_p * t + alpha + lambda_pol)
                                * (gamma_pol + (gamma_pol * gamma_pdf * (alpha + lambda_pol + eta_1 + j - alpha_p * t - 1))
                                / (alpha + lambda_pol + eta_1 + eta_2 + j - alpha_p * t))) \
                        / gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + lambda_pol)
                
                term9 = ((eta_2 + 1) * gamma_pdf * gamma(eta_1 + j - alpha_p * t + alpha - 1)) \
                        / ((alpha + eta_1 + eta_2 + j - alpha_p * t)**2 * gamma(eta_1 + eta_2 + j - alpha_p * t + alpha))
                
                term10 = (gamma(eta_1 + j - alpha_p * t + alpha - 1) * digamma(eta_1 + j - alpha_p * t + alpha - 1)
                        * (1 + (gamma_pdf * (alpha + eta_1 + j - alpha_p * t - 1))
                                / (alpha + eta_1 + eta_2 + j - alpha_p * t))) \
                        / gamma(eta_1 + eta_2 + j - alpha_p * t + alpha)
                
                term11 = - (gamma(eta_1 + j - alpha_p * t + alpha - 1) * digamma(eta_1 + eta_2 + j - alpha_p * t + alpha)
                                * (1 + (gamma_pdf * (alpha + eta_1 + j - alpha_p * t - 1))
                                / (alpha + eta_1 + eta_2 + j - alpha_p * t))) \
                        / gamma(eta_1 + eta_2 + j - alpha_p * t + alpha)
                if evolution_order == "LO":
                    term12 = 0
                elif evolution_order == "NLO":
                    term12 = gamma_pol * (- (term4 + term5) + gamma(eta_1 + j - alpha_p * t + alpha - 1) *(
                         -(eta_2 + 1) * gamma_pdf /(eta_1 + eta_2 + j - alpha_p * t + alpha) - 
                         (eta_2 + eta_1 * (1 + gamma_pdf) + j - alpha_p * t + alpha + gamma_pdf * (j - alpha_p * t + alpha -1)) * \
                         (digamma(eta_1 + j - alpha_p * t + alpha - 1)- digamma(eta_1 + eta_2 + j - alpha_p * t + alpha))
                    )/gamma(1 + eta_1 +eta_2 + j - alpha_p * t + alpha))

                return term1 * (term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10 + term11 + term12)
        
        
        def dpol_pdf_dgamma_pol(
                        A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
                        Delta_A_pdf,alpha,gamma_pol, lambda_pol,
                        j,alpha_p,t
        ):
                term1 = A_pdf * Delta_A_pdf * gamma(eta_2 + 1)
    
                term2 = (epsilon * gamma(eta_1 + j - alpha_p * t + alpha + lambda_pol - 0.5)) \
                        / gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + lambda_pol + 0.5)
                
                term3 = (gamma(eta_1 + j - alpha_p * t + alpha + lambda_pol - 1) * 
                        (- alpha_p * t  + alpha + lambda_pol + eta_1 * (gamma_pdf + 1) + eta_2 +
                        gamma_pdf * (- alpha_p * t  + alpha + lambda_pol + j - 1) + j)) \
                        / gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + lambda_pol + 1)
                
                if evolution_order == "LO":
                    term4 = 0
                elif evolution_order == "NLO":
                    term4 = - ((epsilon * gamma(eta_1 + j - alpha_p * t + alpha - 0.5)) \
                        / gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + 0.5)
                        + (eta_2 + eta_1*(1+gamma_pdf) +  j - alpha_p * t + alpha
                        + gamma_pdf * ( j - alpha_p * t + alpha - 1)) * gamma(eta_1 +  j - alpha_p * t + alpha - 1)/\
                        gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + 1)
                    )
                return term1 * (term2 + term3 + term4)
        
        def dpol_pdf_dlambda_pol(
                        A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
                        Delta_A_pdf,alpha,gamma_pol, lambda_pol,
                        j,alpha_p,t
        ):
                # Same for LO and NLO
                term1 = A_pdf * Delta_A_pdf * gamma(eta_2 + 1)
                
                term2 = (gamma_pol * epsilon * gamma(eta_1 + j - alpha_p * t + alpha + lambda_pol - 0.5) * 
                        digamma(eta_1 + j - alpha_p * t + alpha + lambda_pol - 0.5)) / \
                        gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + lambda_pol + 0.5)
                
                term3 = -(gamma_pol * epsilon * gamma(eta_1 + j - alpha_p * t + alpha + lambda_pol - 0.5) * 
                        digamma(eta_1 + eta_2 + j - alpha_p * t + alpha + lambda_pol + 0.5)) / \
                        gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + lambda_pol + 0.5)
                
                term4 = (gamma_pol * (eta_2 + 1) * gamma_pdf * gamma(eta_1 + j - alpha_p * t + alpha + lambda_pol - 1)) / \
                        ((alpha + lambda_pol + eta_1 + eta_2 + j - alpha_p * t)**2 * 
                        gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + lambda_pol))
                
                term5 = (gamma(eta_1 + j - alpha_p * t + alpha + lambda_pol - 1) * 
                        digamma(eta_1 + j - alpha_p * t + alpha + lambda_pol - 1) * 
                        (gamma_pol + (gamma_pol * gamma_pdf * 
                        (alpha + lambda_pol + eta_1 + j - alpha_p * t - 1)) / 
                        (alpha + lambda_pol + eta_1 + eta_2 + j - alpha_p * t))) / \
                        gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + lambda_pol)
                
                term6 = -(gamma(eta_1 + j - alpha_p * t + alpha + lambda_pol - 1) * 
                        digamma(eta_1 + eta_2 + j - alpha_p * t + alpha + lambda_pol) * 
                        (gamma_pol + (gamma_pol * gamma_pdf * 
                        (alpha + lambda_pol + eta_1 + j - alpha_p * t - 1)) / 
                        (alpha + lambda_pol + eta_1 + eta_2 + j - alpha_p * t))) / \
                        gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + lambda_pol)
                                
                return term1 * (term2 + term3 + term4 + term5 + term6)
        
        Delta_Delta_A_pdf = dpol_pdf_dDelta_A_pdf(
                        A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
                        Delta_A_pdf,alpha,gamma_pol, lambda_pol,
                        j,alpha_p,t
                        ) * err_Delta_A_pdf
        Delta_alpha = dpol_pdf_dalpha(
                        A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
                        Delta_A_pdf,alpha,gamma_pol, lambda_pol,
                        j,alpha_p,t
                        ) * err_alpha
        Delta_gamma_pol = dpol_pdf_dgamma_pol(
                        A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
                        Delta_A_pdf,alpha,gamma_pol, lambda_pol,
                        j,alpha_p,t
                        ) * err_gamma_pol
        Delta_lambda_pol= dpol_pdf_dlambda_pol(
                        A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
                        Delta_A_pdf,alpha,gamma_pol, lambda_pol,
                        j,alpha_p,t
                        ) * err_lambda_pol

        # Debug
        # print(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
        #                 Delta_A_pdf,alpha,gamma_pol, lambda_pol)
        # print(dpol_pdf_dDelta_A_pdf(
        #                 A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
        #                 Delta_A_pdf,alpha,gamma_pol, lambda_pol,
        #                 j,alpha_p,t
        # ))
        # print(dpol_pdf_dalpha(
        #                 A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
        #                 Delta_A_pdf,alpha,gamma_pol, lambda_pol,
        #                 j,alpha_p,t
        # ))
        # print(dpol_pdf_dgamma_pol(
        #                 A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
        #                 Delta_A_pdf,alpha,gamma_pol, lambda_pol,
        #                 j,alpha_p,t
        # ))
        # print(dpol_pdf_dlambda_pol(
        #                 A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
        #                 Delta_A_pdf,alpha,gamma_pol, lambda_pol,
        #                 j,alpha_p,t
        # ))

        result = np.sqrt(Delta_Delta_A_pdf**2+Delta_alpha**2+Delta_gamma_pol**2+Delta_lambda_pol**2)
        return result

def integral_uv_pdf_regge(j,eta,alpha_p,t, evolution_order = "LO", error_type="central"):
    """
    Result of the integral of the Reggeized uv(x) PDF based on the given LO parameters and selected errors.
    
    Arguments:
    - j (float) conformal spin,
    - eta (float): skewness (scalar or array)(placeholder for now),
    - alpha_p (float): Regge slope,
    - t (float): Mandelstam t (scalar or array),
    - evolution_order (str. optional): LO, NLO, NNLO
    - error_type: (str. optional) A string indicating whether to use 'central', 'plus', or 'minus' errors. Default is 'central'.
    
    Returns:
    The value of the Reggeized integral together with the error of uv(x) based on the selected parameters and error type.
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
    error_col_index = error_mapping.get(error_type) 

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

    pdf = integral_pdf_regge(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,j,alpha_p,t)

    if error_type != "central":
    # Extracting errors
        delta_A_pdf  = MSTW_PDF[[evolution_order]].iloc[index_A_u,0][error_col_index]
        delta_eta_1 = MSTW_PDF[[evolution_order]].iloc[index_eta_1,0][error_col_index]
        delta_eta_2 = MSTW_PDF[[evolution_order]].iloc[index_eta_2,0][error_col_index]
        delta_epsilon = MSTW_PDF[[evolution_order]].iloc[index_epsilon_u,0][error_col_index]
        delta_gamma_pdf = MSTW_PDF[[evolution_order]].iloc[index_gamma_u,0][error_col_index]

        
        pdf_error = integral_pdf_regge_error(A_pdf,delta_A_pdf,eta_1,delta_eta_1,eta_2,delta_eta_2,epsilon,delta_epsilon,gamma_pdf,delta_gamma_pdf,j,alpha_p,t,error_type)
        return pdf, pdf_error
    else:
        return pdf, 0

def integral_dv_pdf_regge(j,eta,alpha_p,t, evolution_order = "LO", error_type="central"):
    """
    Result of the integral of the Reggeized dv(x) PDF based on the given LO parameters and selected errors.
    
    Arguments:
    - j (float) conformal spin,
    - eta (float): skewness (scalar or array)(placeholder for now),
    - alpha_p (float): Regge slope,
    - t (float): Mandelstam t (scalar or array),
    - evolution_order (str. optional): LO, NLO, NNLO
    - error_type: (str. optional) A string indicating whether to use 'central', 'plus', or 'minus' errors. Default is 'central'.
    
    Returns:
    The value of the Reggeized integral together with the error of dv(x) based on the selected parameters and error type.
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
    error_col_index = error_mapping.get(error_type)

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

    pdf = integral_pdf_regge(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,j,alpha_p,t)

    if error_type != "central":
        # Extracting errors
        delta_A_pdf  = MSTW_PDF[[evolution_order]].iloc[index_A_d,0][error_col_index]
        delta_eta_1 = MSTW_PDF[[evolution_order]].iloc[index_eta_3,0][error_col_index]
        delta_eta_2 = np.sign(MSTW_PDF[[evolution_order]].iloc[index_eta_42, 0][error_col_index]) * np.sqrt(MSTW_PDF[[evolution_order]].iloc[index_eta_42, 0][error_col_index]**2+MSTW_PDF[[evolution_order]].iloc[index_eta_2, 0][error_col_index]**2)
        delta_epsilon = MSTW_PDF[[evolution_order]].iloc[index_epsilon_d,0][error_col_index]
        delta_gamma_pdf = MSTW_PDF[[evolution_order]].iloc[index_gamma_d,0][error_col_index]


        pdf_error = integral_pdf_regge_error(A_pdf,delta_A_pdf,eta_1,delta_eta_1,eta_2,delta_eta_2,epsilon,delta_epsilon,gamma_pdf,delta_gamma_pdf,j,alpha_p,t,error_type)
        return pdf, pdf_error
    else:
        return pdf, 0

def integral_sv_pdf_regge(j,eta,alpha_p,t, evolution_order = "LO", error_type="central"):
    """
    Result of the integral of the Reggeized sv(x) PDF based on the given LO parameters and selected errors.
    
    Arguments:
    - j (float) conformal spin,
    - eta (float): skewness (scalar or array)(placeholder for now),
    - alpha_p (float): Regge slope,
    - t (float): Mandelstam t (scalar or array),
    - evolution_order (str. optional): LO, NLO, NNLO
    - error_type: (str. optional) A string indicating whether to use 'central', 'plus', or 'minus' errors. Default is 'central'.
    
    Returns:
    The value of the Reggeized integral together with the error of sv(x) based on the selected parameters and error type.
    """
    # eta_1 = delta_minus, eta_2 = eta_minus, epsilon = 0, gamma = 0
    def integral_sv_pdf_regge(A_m,delta_m,eta_m,x_0,j,alpha_p,t):
        frac = gamma(1+eta_m)*gamma(j+delta_m-1-alpha_p*t)/(x_0*gamma(1+delta_m+eta_m+j-alpha_p*t))
        result = -A_m*(j-1-delta_m*(x_0-1)-x_0*(eta_m+j-alpha_p*t)-alpha_p*t)*frac
        return result
    def integral_sv_pdf_regge_error(A_m,delta_A_m,delta_m,delta_delta_m,eta_m,delta_eta_m,x_0,delta_x_0,j,alpha_p,t, error_type="central"):
        if error_type == "central":
            return 0
        def dpdf_dA_m(A_m, delta_m, eta_m,x_0, j, alpha_p,t):
            result = (
                    gamma(eta_m + 1) * gamma(delta_m + j - alpha_p * t - 1) *
                    ((delta_m + j - alpha_p * t - 1) / (delta_m + eta_m + j - alpha_p * t) - x_0)
                ) / (x_0 * gamma(delta_m + eta_m + j - alpha_p * t))
        
            return result
        def dpdf_ddelta_m(A_m,delta_m,eta_m,x_0,j,alpha_p,t):
            term_1 = (
                A_m * gamma(eta_m + 1) * 
                ((1 / (delta_m + eta_m + j - alpha_p * t)) -
                ((delta_m + j - alpha_p * t - 1) / (delta_m + eta_m + j - alpha_p * t) ** 2)) *
                gamma(delta_m + j - alpha_p * t - 1)
            ) / (x_0 * gamma(delta_m + eta_m + j - alpha_p * t))
            
            term_2 = (
                A_m * gamma(eta_m + 1) * gamma(delta_m + j - alpha_p * t - 1) *
                digamma(delta_m + j - alpha_p * t - 1) *
                ((delta_m + j - alpha_p * t - 1) / (delta_m + eta_m + j - alpha_p * t) - x_0)
            ) / (x_0 * gamma(delta_m + eta_m + j - alpha_p * t))
            
            term_3 = (
                A_m * gamma(eta_m + 1) * gamma(delta_m + j - alpha_p * t - 1) *
                digamma(delta_m + eta_m + j - alpha_p * t) *
                ((delta_m + j - alpha_p * t - 1) / (delta_m + eta_m + j - alpha_p * t) - x_0)
            ) / (x_0 * gamma(delta_m + eta_m + j - alpha_p * t))
            
            return term_1 + term_2 - term_3
        def dpdf_deta_m(A_m,delta_m,eta_m,x_0,j,alpha_p,t):
            term_1 = -(
                A_m * gamma(eta_m + 1) * (delta_m + j - alpha_p * t - 1) * gamma(delta_m + j - alpha_p * t - 1)
            ) / (x_0 * (delta_m + eta_m + j - alpha_p * t) ** 2 * gamma(delta_m + eta_m + j - alpha_p * t))
            
            term_2 = (
                A_m * gamma(eta_m + 1) * digamma(eta_m + 1) * gamma(delta_m + j - alpha_p * t - 1) *
                ((delta_m + j - alpha_p * t - 1) / (delta_m + eta_m + j - alpha_p * t) - x_0)
            ) / (x_0 * gamma(delta_m + eta_m + j - alpha_p * t))
            
            term_3 = (
                A_m * gamma(eta_m + 1) * gamma(delta_m + j - alpha_p * t - 1) *
                digamma(delta_m + eta_m + j - alpha_p * t) *
                ((delta_m + j - alpha_p * t - 1) / (delta_m + eta_m + j - alpha_p * t) - x_0)
            ) / (x_0 * gamma(delta_m + eta_m + j - alpha_p * t))
            
            return term_1 + term_2 - term_3
        
        def dpdf_dx_0(A_m,delta_m,eta_m,x_0,j,alpha_p,t):
            term_1 = -(
                A_m * gamma(eta_m + 1) * gamma(delta_m + j - alpha_p * t - 1) *
                ((delta_m + j - alpha_p * t - 1) / (delta_m + eta_m + j - alpha_p * t) - x_0)
            ) / (x_0 ** 2 * gamma(delta_m + eta_m + j - alpha_p * t))
            
            term_2 = -(
                A_m * gamma(eta_m + 1) * gamma(delta_m + j - alpha_p * t - 1)
            ) / (x_0 * gamma(delta_m + eta_m + j - alpha_p * t))
            
            return term_1 + term_2
        
        Delta_A_m = dpdf_dA_m(A_m,delta_m,eta_m,x_0,j,alpha_p,t) * delta_A_m
        Delta_delta_m = dpdf_ddelta_m(A_m,delta_m,eta_m,x_0,j,alpha_p,t) * delta_delta_m
        Delta_eta_m = dpdf_deta_m(A_m,delta_m,eta_m,x_0,j,alpha_p,t) * delta_eta_m
        Delta_x_0= dpdf_dx_0(A_m,delta_m,eta_m,x_0,j,alpha_p,t) * delta_x_0

        # Debug
        # print(dpdf_dA_m(A_m,delta_m,eta_m,x_0,j,alpha_p,t),dpdf_ddelta_m(A_m,delta_m,eta_m,x_0,j,alpha_p,t),dpdf_deta_m(A_m,delta_m,eta_m,x_0,j,alpha_p,t),dpdf_dx_0(A_m,delta_m,eta_m,x_0,j,alpha_p,t))
        # print(Delta_A_m,Delta_delta_m,Delta_eta_m,Delta_x_0)

        result = np.sqrt(Delta_A_m**2+Delta_delta_m**2+Delta_eta_m**2+Delta_x_0**2)
        return result
        
    # Check type
    check_error_type(error_type)

    error_mapping = {
        "central": 0,
        "plus": 1,
        "minus": 2
    }
    
    error_col_index = error_mapping.get(error_type)

    # delta_- fixed to 0.2
    index_A_m = MSTW_PDF[MSTW_PDF["Parameter"] == "A_-"].index[0]
    index_eta_m = MSTW_PDF[MSTW_PDF["Parameter"] == "eta_-"].index[0]
    index_x_0 = MSTW_PDF[MSTW_PDF["Parameter"] == "x_0"].index[0]

    # Extracting parameter values
    A_m = MSTW_PDF[[evolution_order]].iloc[index_A_m,0][0]
    delta_m = 0.2
    eta_m = MSTW_PDF[[evolution_order]].iloc[index_eta_m,0][0]
    x_0 = MSTW_PDF[[evolution_order]].iloc[index_x_0,0][0]

    pdf = integral_sv_pdf_regge(A_m,delta_m,eta_m,x_0,j,alpha_p,t)

    if error_type != "central":
    # Extracting errors
        delta_A_m  = MSTW_PDF[[evolution_order]].iloc[index_A_m,0][error_col_index]
        delta_delta_m = 0
        delta_eta_m = MSTW_PDF[[evolution_order]].iloc[index_eta_m,0][error_col_index]
        delta_x_0 = MSTW_PDF[[evolution_order]].iloc[index_x_0,0][error_col_index]
        # Debug
        # print(A_m,delta_m,eta_m,x_0)
        # print(delta_A_m,delta_delta_m,delta_eta_m,delta_x_0)

        pdf_error = integral_sv_pdf_regge_error(A_m,delta_A_m,delta_m,delta_delta_m,eta_m,delta_eta_m,x_0,delta_x_0,j,alpha_p,t, error_type)
        return pdf, pdf_error
    else:
        return pdf, 0

def integral_S_pdf_regge(j,eta,alpha_p,t, evolution_order = "LO", error_type="central"):
    """
    Result of the integral of the Reggeized Sv(x) PDF based on the given LO parameters and selected errors.
    
    Arguments:
    - j (float) conformal spin,
    - eta (float): skewness (scalar or array)(placeholder for now),
    - alpha_p (float): Regge slope,
    - t (float): Mandelstam t (scalar or array),
    - evolution_order (str. optional): LO, NLO, NNLO
    - error_type: (str. optional) A string indicating whether to use 'central', 'plus', or 'minus' errors. Default is 'central'.
    
    Returns:
    The value of the Reggeized integral together with the error of Sv(x) based on the selected parameters and error type.
    """
    # Check type
    check_error_type(error_type)
    
    error_mapping = {
        "central": 0,
        "plus": 1,
        "minus": 2
    }
    
    error_col_index = error_mapping.get(error_type)

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

    pdf = integral_pdf_regge(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,j,alpha_p,t)

    if error_type != "central":
    # Extracting errors
        delta_A_pdf  = MSTW_PDF[[evolution_order]].iloc[index_A_S,0][error_col_index]
        delta_eta_1 = MSTW_PDF[[evolution_order]].iloc[index_delta_S,0][error_col_index]
        delta_eta_2 = MSTW_PDF[[evolution_order]].iloc[index_eta_S,0][error_col_index]
        delta_epsilon = MSTW_PDF[[evolution_order]].iloc[index_epsilon_S,0][error_col_index]
        delta_gamma_pdf = MSTW_PDF[[evolution_order]].iloc[index_gamma_S,0][error_col_index]

        pdf_error = integral_pdf_regge_error(A_pdf,delta_A_pdf,eta_1,delta_eta_1,eta_2,delta_eta_2,epsilon,delta_epsilon,gamma_pdf,delta_gamma_pdf,j,alpha_p,t,error_type)
        return pdf, pdf_error
    else:
        return pdf, 0

def integral_s_plus_pdf_regge(j,eta,alpha_p,t, evolution_order = "LO", error_type="central"):
    """
    Result of the integral of the Reggeized s_+(x) PDF based on the given LO parameters and selected errors.

    Arguments:
    - j (float) conformal spin,
    - eta (float): skewness (scalar or array)(placeholder for now),
    - alpha_p (float): Regge slope,
    - t (float): Mandelstam t (scalar or array),
    - evolution_order (str. optional): LO, NLO, NNLO
    - error_type: (str. optional) A string indicating whether to use 'central', 'plus', or 'minus' errors. Default is 'central'.
    
    Returns:
    The value of the Reggeized integral together with the error of s_+(x) based on the selected parameters and error type.
    """
    # Check type
    check_error_type(error_type)
    
    error_mapping = {
        "central": 0,
        "plus": 1,
        "minus": 2
    }
    
    error_col_index = error_mapping.get(error_type)

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

    pdf = integral_pdf_regge(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,j,alpha_p,t)

    if error_type != "central":
        # Extracting errors
        delta_A_pdf  = MSTW_PDF[[evolution_order]].iloc[index_A_p,0][error_col_index]
        delta_eta_1 = MSTW_PDF[[evolution_order]].iloc[index_delta_S,0][error_col_index]
        delta_eta_2 = MSTW_PDF[[evolution_order]].iloc[index_eta_p,0][error_col_index]
        delta_epsilon = MSTW_PDF[[evolution_order]].iloc[index_epsilon_S,0][error_col_index]
        delta_gamma_pdf = MSTW_PDF[[evolution_order]].iloc[index_gamma_S,0][error_col_index]


        pdf_error = integral_pdf_regge_error(A_pdf,delta_A_pdf,eta_1,delta_eta_1,eta_2,delta_eta_2,epsilon,delta_epsilon,gamma_pdf,delta_gamma_pdf,j,alpha_p,t,error_type)
        return pdf, pdf_error
    else:
        return pdf, 0

def integral_Delta_pdf_regge(j,eta,alpha_p,t, evolution_order = "LO", error_type="central"):
    """
    Result of the integral of the Reggeized uv(x) PDF based on the given LO parameters and selected errors.
    
    Arguments:
    - j (float) conformal spin,
    - eta (float): skewness (scalar or array)(placeholder for now),
    - alpha_p (float): Regge slope,
    - t (float): Mandelstam t (scalar or array),
    - evolution_order (str. optional): LO, NLO, NNLO
    - error_type: (str. optional) A string indicating whether to use 'central', 'plus', or 'minus' errors. Default is 'central'.
    
    Returns:
    The value of the Reggeized integral together with the error of uv(x) based on the selected parameters and error type.
    """
    def integral_Delta_pdf_regge(A_Delta,eta_Delta,eta_S,gamma_Delta,delta_Delta,j,alpha_p,t):
        frac_1 = (2+eta_Delta+eta_S+j-alpha_p*t)*(3+eta_Delta+eta_S+j-alpha_p*t)
        frac_2 = gamma(3+eta_S)*gamma(j+eta_Delta-1-alpha_p*t)/(gamma(2+eta_Delta+eta_S+j-alpha_p*t))
        result = A_Delta*(1+((delta_Delta*(eta_Delta+j-alpha_p*t)+gamma_Delta*(3+eta_Delta+eta_S+j-alpha_p*t))*(eta_Delta+j-1+alpha_p*t))/frac_1)*frac_2
        return result
    def integral_Delta_pdf_regge_error(A_Delta,delta_A_Delta,eta_Delta,delta_eta_Delta,eta_S,delta_eta_S,gamma_Delta,delta_gamma_Delta,delta_Delta,delta_delta_Delta,j,alpha_p,t, error_type="central"):
        if error_type == "central":
             return 0
        def dpdf_dA_Delta(A_Delta,eta_Delta,eta_S,gamma_Delta,delta_Delta,j,alpha_p,t):
            term = (
            gamma(eta_S + 3) * gamma(eta_Delta + j - alpha_p * t - 1) *
            (
                (
                    (eta_Delta + j - alpha_p * t - 1) *
                    (
                        delta_Delta * (eta_Delta + j - alpha_p * t) +
                        gamma_Delta * (eta_Delta + eta_S + j - alpha_p * t + 3)
                    )
                ) / (
                    (eta_Delta + eta_S + j - alpha_p * t + 2) *
                    (eta_Delta + eta_S + j - alpha_p * t + 3)
                ) + 1
            )
            ) / gamma(eta_Delta + eta_S + j - alpha_p * t + 2)
        
            return term
        def dpdf_deta_Delta(A_Delta,eta_Delta,eta_S,gamma_Delta,delta_Delta,j,alpha_p,t):
            term_1 = (
                A_Delta * gamma(eta_S + 3) * gamma(eta_Delta + j - alpha_p * t - 1) *
                (
                    -(
                        (eta_Delta + j - alpha_p * t - 1) *
                        (
                            delta_Delta * (eta_Delta + j - alpha_p * t) +
                            gamma_Delta * (eta_Delta + eta_S + j - alpha_p * t + 3)
                        )
                    ) / (
                        (eta_Delta + eta_S + j - alpha_p * t + 2) ** 2 *
                        (eta_Delta + eta_S + j - alpha_p * t + 3)
                    )
                    - (
                        (eta_Delta + j - alpha_p * t - 1) *
                        (
                            delta_Delta * (eta_Delta + j - alpha_p * t) +
                            gamma_Delta * (eta_Delta + eta_S + j - alpha_p * t + 3)
                        )
                    ) / (
                        (eta_Delta + eta_S + j - alpha_p * t + 2) *
                        (eta_Delta + eta_S + j - alpha_p * t + 3) ** 2
                    )
                    + (
                        (delta_Delta + gamma_Delta) * (eta_Delta + j - alpha_p * t - 1)
                    ) / (
                        (eta_Delta + eta_S + j - alpha_p * t + 2) *
                        (eta_Delta + eta_S + j - alpha_p * t + 3)
                    )
                    + (
                        delta_Delta * (eta_Delta + j - alpha_p * t) +
                        gamma_Delta * (eta_Delta + eta_S + j - alpha_p * t + 3)
                    ) / (
                        (eta_Delta + eta_S + j - alpha_p * t + 2) *
                        (eta_Delta + eta_S + j - alpha_p * t + 3)
                    )
                )
            ) / gamma(eta_Delta + eta_S + j - alpha_p * t + 2)
        
            term_2 = (
                A_Delta * gamma(eta_S + 3) * gamma(eta_Delta + j - alpha_p * t - 1) *
                digamma(eta_Delta + j - alpha_p * t - 1) *
                (
                    (
                        (eta_Delta + j - alpha_p * t - 1) *
                        (
                            delta_Delta * (eta_Delta + j - alpha_p * t) +
                            gamma_Delta * (eta_Delta + eta_S + j - alpha_p * t + 3)
                        )
                    ) / (
                        (eta_Delta + eta_S + j - alpha_p * t + 2) *
                        (eta_Delta + eta_S + j - alpha_p * t + 3)
                    ) + 1
                )
            ) / gamma(eta_Delta + eta_S + j - alpha_p * t + 2)
            
            term_3 = (
                -A_Delta * gamma(eta_S + 3) * gamma(eta_Delta + j - alpha_p * t - 1) *
                digamma(eta_Delta + eta_S + j - alpha_p * t + 2) *
                (
                    (
                        (eta_Delta + j - alpha_p * t - 1) *
                        (
                            delta_Delta * (eta_Delta + j - alpha_p * t) +
                            gamma_Delta * (eta_Delta + eta_S + j - alpha_p * t + 3)
                        )
                    ) / (
                        (eta_Delta + eta_S + j - alpha_p * t + 2) *
                        (eta_Delta + eta_S + j - alpha_p * t + 3)
                    ) + 1
                )
            ) / gamma(eta_Delta + eta_S + j - alpha_p * t + 2)
            
            return term_1 + term_2 + term_3
        
        def dpdf_deta_S(A_Delta,eta_Delta,eta_S,gamma_Delta,delta_Delta,j,alpha_p,t):
            term_1 = (
                A_Delta * gamma(eta_S + 3) * gamma(eta_Delta + j - alpha_p * t - 1) *
                (
                    -(
                        (eta_Delta + j - alpha_p * t - 1) *
                        (
                            delta_Delta * (eta_Delta + j - alpha_p * t) +
                            gamma_Delta * (eta_Delta + eta_S + j - alpha_p * t + 3)
                        )
                    ) / (
                        (eta_Delta + eta_S + j - alpha_p * t + 2) ** 2 *
                        (eta_Delta + eta_S + j - alpha_p * t + 3)
                    )
                    - (
                        (eta_Delta + j - alpha_p * t - 1) *
                        (
                            delta_Delta * (eta_Delta + j - alpha_p * t) +
                            gamma_Delta * (eta_Delta + eta_S + j - alpha_p * t + 3)
                        )
                    ) / (
                        (eta_Delta + eta_S + j - alpha_p * t + 2) *
                        (eta_Delta + eta_S + j - alpha_p * t + 3) ** 2
                    )
                    + (
                        gamma_Delta * (eta_Delta + j - alpha_p * t - 1)
                    ) / (
                        (eta_Delta + eta_S + j - alpha_p * t + 2) *
                        (eta_Delta + eta_S + j - alpha_p * t + 3)
                    )
                )
            ) / gamma(eta_Delta + eta_S + j - alpha_p * t + 2)
            
            term_2 = (
                A_Delta * gamma(eta_S + 3) * digamma(eta_S + 3) * gamma(eta_Delta + j - alpha_p * t - 1) *
                (
                    (
                        (eta_Delta + j - alpha_p * t - 1) *
                        (
                            delta_Delta * (eta_Delta + j - alpha_p * t) +
                            gamma_Delta * (eta_Delta + eta_S + j - alpha_p * t + 3)
                        )
                    ) / (
                        (eta_Delta + eta_S + j - alpha_p * t + 2) *
                        (eta_Delta + eta_S + j - alpha_p * t + 3)
                    ) + 1
                )
            ) / gamma(eta_Delta + eta_S + j - alpha_p * t + 2)
            
            term_3 = (
                -A_Delta * gamma(eta_S + 3) * gamma(eta_Delta + j - alpha_p * t - 1) *
                digamma(eta_Delta + eta_S + j - alpha_p * t + 2) *
                (
                    (
                        (eta_Delta + j - alpha_p * t - 1) *
                        (
                            delta_Delta * (eta_Delta + j - alpha_p * t) +
                            gamma_Delta * (eta_Delta + eta_S + j - alpha_p * t + 3)
                        )
                    ) / (
                        (eta_Delta + eta_S + j - alpha_p * t + 2) *
                        (eta_Delta + eta_S + j - alpha_p * t + 3)
                    ) + 1
                )
            ) / gamma(eta_Delta + eta_S + j - alpha_p * t + 2)
            
            return term_1 + term_2 + term_3
        def dpdf_dgamma_Delta(A_Delta,eta_Delta,eta_S,gamma_Delta,delta_Delta,j,alpha_p,t):
            term_1 = gamma(eta_Delta + j - alpha_p * t)
            term_2 = gamma(3 + eta_Delta + eta_S + j - alpha_p * t)
            result = A_Delta * gamma(3+eta_S) * term_1 / term_2
            return result
        def dpdf_ddelta_Delta(A_Delta,eta_Delta,eta_S,gamma_Delta,delta_Delta,j,alpha_p,t):
            return (
                A_Delta * gamma(eta_S + 3) * (eta_Delta + j - alpha_p * t - 1) * (eta_Delta + j - alpha_p * t) * 
                gamma(eta_Delta + j - alpha_p * t - 1)
            ) / (
                (eta_Delta + eta_S + j - alpha_p * t + 2) *
                (eta_Delta + eta_S + j - alpha_p * t + 3) *
                gamma(eta_Delta + eta_S + j - alpha_p * t + 2)
            )
        # Debug
        # print(dpdf_dA_Delta(A_Delta,eta_Delta,eta_S,gamma_Delta,delta_Delta,j,alpha_p,t))
        # print(dpdf_deta_Delta(A_Delta,eta_Delta,eta_S,gamma_Delta,delta_Delta,j,alpha_p,t))
        # print(dpdf_deta_S(A_Delta,eta_Delta,eta_S,gamma_Delta,delta_Delta,j,alpha_p,t))
        # print(dpdf_dgamma_Delta(A_Delta,eta_Delta,eta_S,gamma_Delta,delta_Delta,j,alpha_p,t))
        # print(dpdf_ddelta_Delta(A_Delta,eta_Delta,eta_S,gamma_Delta,delta_Delta,j,alpha_p,t))
        # Check type
        Delta_A_Delta = dpdf_dA_Delta(A_Delta,eta_Delta,eta_S,gamma_Delta,delta_Delta,j,alpha_p,t) * delta_A_Delta
        Delta_eta_Delta = dpdf_deta_Delta(A_Delta,eta_Delta,eta_S,gamma_Delta,delta_Delta,j,alpha_p,t) * delta_eta_Delta
        Delta_eta_S = dpdf_deta_S(A_Delta,eta_Delta,eta_S,gamma_Delta,delta_Delta,j,alpha_p,t) * delta_eta_S
        Delta_gamma_Delta = dpdf_dgamma_Delta(A_Delta,eta_Delta,eta_S,gamma_Delta,delta_Delta,j,alpha_p,t) * delta_gamma_Delta
        Delta_delta_Delta = dpdf_ddelta_Delta(A_Delta,eta_Delta,eta_S,gamma_Delta,delta_Delta,j,alpha_p,t) * delta_delta_Delta

        result = np.sqrt(Delta_A_Delta**2+Delta_eta_Delta**2+Delta_eta_S**2+Delta_gamma_Delta**2+Delta_delta_Delta**2)
        return result
    
    check_error_type(error_type)

     # Define a dictionary that maps the error_type to column indices
    error_mapping = {
        "central": 0,  # The column with the central value
        "plus": 1,     # The column with the + error value
        "minus": 2     # The column with the - error value
    }
    
    # Get the column index corresponding to the error_type
    error_col_index = error_mapping.get(error_type) 

    # Get row index of entry
    index_A_Delta=MSTW_PDF[MSTW_PDF["Parameter"] == "A_Delta"].index[0]
    index_eta_Delta=MSTW_PDF[MSTW_PDF["Parameter"] == "eta_Delta"].index[0]
    index_eta_S=MSTW_PDF[MSTW_PDF["Parameter"] == "eta_S"].index[0]
    index_delta_Delta=MSTW_PDF[MSTW_PDF["Parameter"] == "delta_Delta"].index[0]
    index_gamma_Delta=MSTW_PDF[MSTW_PDF["Parameter"] == "gamma_Delta"].index[0]

    # Extracting parameter values
    A_Delta = MSTW_PDF[[evolution_order]].iloc[index_A_Delta,0][0]
    eta_Delta = MSTW_PDF[[evolution_order]].iloc[index_eta_Delta,0][0]
    eta_S = MSTW_PDF[[evolution_order]].iloc[index_eta_S,0][0]
    delta_Delta = MSTW_PDF[[evolution_order]].iloc[index_delta_Delta,0][0]
    gamma_Delta = MSTW_PDF[[evolution_order]].iloc[index_gamma_Delta,0][0]

    pdf = integral_Delta_pdf_regge(A_Delta,eta_Delta,eta_S,gamma_Delta,delta_Delta,j,alpha_p,t)

    if error_type != "central":
        # Extracting errors
        delta_A_Delta  = MSTW_PDF[[evolution_order]].iloc[index_A_Delta,0][error_col_index]
        delta_eta_Delta = MSTW_PDF[[evolution_order]].iloc[index_eta_Delta,0][error_col_index]
        delta_eta_S = MSTW_PDF[[evolution_order]].iloc[index_eta_S,0][error_col_index]
        delta_delta_Delta = MSTW_PDF[[evolution_order]].iloc[index_delta_Delta,0][error_col_index]
        delta_gamma_Delta = MSTW_PDF[[evolution_order]].iloc[index_gamma_Delta,0][error_col_index]

        pdf_error = integral_Delta_pdf_regge_error(A_Delta,delta_A_Delta,eta_Delta,delta_eta_Delta,eta_S,delta_eta_S,gamma_Delta,delta_gamma_Delta,delta_Delta,delta_delta_Delta,j,alpha_p,t, error_type)
        return pdf, pdf_error
    else:
        return pdf, 0

def integral_gluon_pdf_regge(j,eta,alpha_p,t, evolution_order = "LO", error_type="central"):
    """
    Result of the integral of the Reggeized g(x) PDF based on the given LO parameters and selected errors.

    Arguments:
    - j (float) conformal spin,
    - eta (float): skewness (scalar or array)(placeholder for now),
    - alpha_p (float): Regge slope,
    - t (float): Mandelstam t (scalar or array),
    - evolution_order (str. optional): LO, NLO, NNLO
    - error_type: (str. optional) A string indicating whether to use 'central', 'plus', or 'minus' errors. Default is 'central'.
    
    Returns:
    The value of the Reggeized integral together with the error of g(x) based on the selected parameters and error type.
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
    error_col_index = error_mapping.get(error_type) 

    # Get row index of entry
    index_A_g=MSTW_PDF[MSTW_PDF["Parameter"] == "A_g"].index[0]
    index_delta_g=MSTW_PDF[MSTW_PDF["Parameter"] == "delta_g"].index[0]
    index_eta_g=MSTW_PDF[MSTW_PDF["Parameter"] == "eta_g"].index[0]
    index_epsilon_g=MSTW_PDF[MSTW_PDF["Parameter"] == "epsilon_g"].index[0]
    index_gamma_g=MSTW_PDF[MSTW_PDF["Parameter"] == "gamma_g"].index[0]

    # Extracting parameter values
    A_pdf = MSTW_PDF[[evolution_order]].iloc[index_A_g,0][0]
    eta_1 = MSTW_PDF[[evolution_order]].iloc[index_delta_g,0][0]
    eta_2 = MSTW_PDF[[evolution_order]].iloc[index_eta_g,0][0]
    epsilon = MSTW_PDF[[evolution_order]].iloc[index_epsilon_g,0][0]
    gamma_pdf = MSTW_PDF[[evolution_order]].iloc[index_gamma_g,0][0]

    pdf = integral_pdf_regge(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,j,alpha_p,t)

    # Additional term at NLO and NNLO
    if evolution_order != "LO":
        # Get row index of entry
        index_A_g_prime=MSTW_PDF[MSTW_PDF["Parameter"] == "A_g'"].index[0]
        index_delta_g_prime=MSTW_PDF[MSTW_PDF["Parameter"] == "delta_g'"].index[0]
        index_eta_g_prime=MSTW_PDF[MSTW_PDF["Parameter"] == "eta_g'"].index[0]
        A_pdf_prime = MSTW_PDF[[evolution_order]].iloc[index_A_g_prime,0][0]
        eta_1_prime = MSTW_PDF[[evolution_order]].iloc[index_delta_g_prime,0][0]
        eta_2_prime = MSTW_PDF[[evolution_order]].iloc[index_eta_g_prime,0][0]
        nlo_term = A_pdf_prime * (eta_1_prime + eta_2_prime + j - alpha_p * t) * gamma(j - alpha_p *t + eta_1_prime-1)*gamma(1+eta_2_prime)/\
                gamma(j-alpha_p*t+eta_1_prime+eta_2_prime + 1)
        # print(pdf,nlo_term)
        pdf += nlo_term
    if error_type != "central":
    # Extracting errors
        delta_A_pdf  = MSTW_PDF[[evolution_order]].iloc[index_A_g,0][error_col_index]
        delta_eta_1 = MSTW_PDF[[evolution_order]].iloc[index_delta_g,0][error_col_index]
        delta_eta_2 = MSTW_PDF[[evolution_order]].iloc[index_eta_g,0][error_col_index]
        delta_epsilon = MSTW_PDF[[evolution_order]].iloc[index_epsilon_g,0][error_col_index]
        delta_gamma_pdf = MSTW_PDF[[evolution_order]].iloc[index_gamma_g,0][error_col_index]


        pdf_error = integral_pdf_regge_error(A_pdf,delta_A_pdf,eta_1,delta_eta_1,eta_2,delta_eta_2,epsilon,delta_epsilon,gamma_pdf,delta_gamma_pdf,j,alpha_p,t,error_type)
        if evolution_order != "LO":
            delta_A_prime_pdf  = MSTW_PDF[[evolution_order]].iloc[index_A_g_prime,0][error_col_index]
            delta_eta_1_prime= MSTW_PDF[[evolution_order]].iloc[index_delta_g_prime,0][error_col_index]
            delta_eta_2_prime = MSTW_PDF[[evolution_order]].iloc[index_eta_g_prime,0][error_col_index]
            
            # print(A_pdf,eta_1,eta_2,epsilon,gamma_pdf)
            # print(delta_A_pdf,delta_eta_1,delta_eta_2,delta_epsilon,delta_gamma_pdf)
            # print("-----")
            # print(A_pdf_prime,eta_1_prime,eta_2_prime)
            # print(delta_A_prime_pdf,delta_eta_1_prime,delta_eta_2_prime)

            dpdf_dA = gamma(j - alpha_p *t + eta_1_prime-1)*gamma(1+eta_2_prime)/\
                gamma(j-alpha_p*t+eta_1_prime+eta_2_prime)
            dpdf_deta_1 = (A_pdf_prime * gamma(1+eta_2_prime) * gamma(j - alpha_p *t + eta_1_prime-1)/\
                gamma(j-alpha_p*t+eta_1_prime+eta_2_prime) * \
            (digamma(eta_1_prime + 1 )-digamma(eta_1_prime + eta_2_prime + j - alpha_p * t))
            )
            dpdf_deta_2 = (A_pdf_prime * gamma(1+eta_2_prime) * gamma(j - alpha_p *t + eta_1_prime-1)/\
              gamma(j-alpha_p*t+eta_1_prime+eta_2_prime + 1) * \
             (1 + (eta_1_prime + eta_2_prime + j -alpha_p * t) * (digamma(eta_2_prime + 1) - digamma(eta_1_prime + eta_2_prime + j -alpha_p * t + 1)))
            )
            # print(dpdf_dA,dpdf_deta_1,dpdf_deta_2)
            Delta_A = dpdf_dA * delta_A_prime_pdf
            Delta_eta_1 = dpdf_deta_1 * delta_eta_1_prime
            Delta_eta_2 = dpdf_deta_2 * delta_eta_2_prime
            pdf_error+= np.sqrt(Delta_A**2+Delta_eta_1**2+Delta_eta_2**2)
        return pdf, pdf_error
    else:
        return pdf, 0

def integral_polarized_uv_pdf_regge(j,eta,alpha_p,t, evolution_order = "LO", error_type="central"):
        """
        Result of the integral of the Reggeized uv(x) PDF based on the given LO parameters and selected errors.

        Arguments:
        - j (float) conformal spin,
        - eta (float): skewness (scalar or array)(placeholder for now),
        - alpha_p (float): Regge slope,
        - t (float): Mandelstam t (scalar or array),
        - evolution_order (str. optional): LO, NLO, NNLO
        - error_type: (str. optional) A string indicating whether to use 'central', 'plus', or 'minus' errors. Default is 'central'.

        Returns:
        The value of the Reggeized integral together with the error of uv(x) based on the selected parameters and error type.
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

        # Get row index of entry
        index_delta_A_u=AAC_PDF[AAC_PDF["Parameter"] == "Delta_A_u"].index[0]
        index_alpha_u=AAC_PDF[AAC_PDF["Parameter"] == "alpha_u"].index[0]
        index_delta_lambda_u=AAC_PDF[AAC_PDF["Parameter"] == "Delta_lambda_u"].index[0]
        index_delta_gamma_u=AAC_PDF[AAC_PDF["Parameter"] == "Delta_gamma_u"].index[0]

        # Extracting central parameter values
        A_pdf = MSTW_PDF[[evolution_order]].iloc[index_A_u,0][0]
        eta_1 = MSTW_PDF[[evolution_order]].iloc[index_eta_1,0][0]
        eta_2 = MSTW_PDF[[evolution_order]].iloc[index_eta_2,0][0]
        epsilon = MSTW_PDF[[evolution_order]].iloc[index_epsilon_u,0][0]
        gamma_pdf = MSTW_PDF[[evolution_order]].iloc[index_gamma_u,0][0]
        # Extracting parameter values based on error type
        delta_A_pdf = AAC_PDF[[evolution_order]].iloc[index_delta_A_u,0][0]
        alpha = AAC_PDF[[evolution_order]].iloc[index_alpha_u,0][0]
        gamma_pol = AAC_PDF[[evolution_order]].iloc[index_delta_gamma_u,0][0]
        lambda_pol = AAC_PDF[[evolution_order]].iloc[index_delta_lambda_u,0][0]

        pdf = integral_polarized_pdf_regge(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
                                           delta_A_pdf,alpha,gamma_pol,lambda_pol,
                                           j,alpha_p,t,evolution_order)
        if error_type != "central":
            err_delta_A_pdf = AAC_PDF[[evolution_order]].iloc[index_delta_A_u,0][error_col_index]
            err_alpha = AAC_PDF[[evolution_order]].iloc[index_alpha_u,0][error_col_index]
            err_gamma_pol = AAC_PDF[[evolution_order]].iloc[index_delta_gamma_u,0][error_col_index]
            err_lambda_pol = AAC_PDF[[evolution_order]].iloc[index_delta_lambda_u,0][error_col_index]


            pdf_error = integral_polarized_pdf_regge_error(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
                                            delta_A_pdf,err_delta_A_pdf,alpha,err_alpha,gamma_pol,err_gamma_pol,lambda_pol,err_lambda_pol,
                                            j,alpha_p,t,evolution_order,error_type)
            return pdf, pdf_error
        else:
            return pdf, 0

def integral_polarized_dv_pdf_regge(j,eta,alpha_p,t, evolution_order = "LO", error_type="central"):
        """
        Result of the integral of the Reggeized dv(x) PDF based on the given LO parameters and selected errors.

        Arguments:
        - j (float) conformal spin,
        - eta (float): skewness (scalar or array)(placeholder for now),
        - alpha_p (float): Regge slope,
        - t (float): Mandelstam t (scalar or array),
        - evolution_order (str. optional): LO, NLO, NNLO
        - error_type: (str. optional) A string indicating whether to use 'central', 'plus', or 'minus' errors. Default is 'central'.

        Returns:
        The value of the Reggeized integral together with the error of dv(x) based on the selected parameters and error type.
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
        index_A_d = MSTW_PDF[MSTW_PDF["Parameter"] == "A_d"].index[0]
        index_eta_3 = MSTW_PDF[MSTW_PDF["Parameter"] == "eta_3"].index[0]
        index_eta_2=MSTW_PDF[MSTW_PDF["Parameter"] == "eta_2"].index[0]
        # Only eta_4-eta_2 given
        index_eta_42 = MSTW_PDF[MSTW_PDF["Parameter"] == "eta_4-eta_2"].index[0]
        index_epsilon_d = MSTW_PDF[MSTW_PDF["Parameter"] == "epsilon_d"].index[0]
        index_gamma_d = MSTW_PDF[MSTW_PDF["Parameter"] == "gamma_d"].index[0]

        # Get row index of entry
        index_delta_A_d=AAC_PDF[AAC_PDF["Parameter"] == "Delta_A_d"].index[0]
        index_alpha_d=AAC_PDF[AAC_PDF["Parameter"] == "alpha_d"].index[0]
        index_delta_lambda_d=AAC_PDF[AAC_PDF["Parameter"] == "Delta_lambda_d"].index[0]
        index_delta_gamma_d=AAC_PDF[AAC_PDF["Parameter"] == "Delta_gamma_d"].index[0]

        # Extracting central parameter values
        # Extracting parameter values
        A_pdf = MSTW_PDF[[evolution_order]].iloc[index_A_d,0][0]
        eta_1 = MSTW_PDF[[evolution_order]].iloc[index_eta_3,0][0]
        eta_2 = MSTW_PDF[[evolution_order]].iloc[index_eta_42, 0][0] + MSTW_PDF[[evolution_order]].iloc[index_eta_2, 0][0]
        epsilon = MSTW_PDF[[evolution_order]].iloc[index_epsilon_d,0][0]
        gamma_pdf = MSTW_PDF[[evolution_order]].iloc[index_gamma_d,0][0]

        # Extracting parameter values based on error type
        Delta_A_pdf = AAC_PDF[[evolution_order]].iloc[index_delta_A_d,0][0]
        alpha = AAC_PDF[[evolution_order]].iloc[index_alpha_d,0][0]
        gamma_pol = AAC_PDF[[evolution_order]].iloc[index_delta_gamma_d,0][0]
        lambda_pol = AAC_PDF[[evolution_order]].iloc[index_delta_lambda_d,0][0]

        pdf = integral_polarized_pdf_regge(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
                                           Delta_A_pdf,alpha,gamma_pol,lambda_pol,
                                           j,alpha_p,t,evolution_order)
        if error_type != "central":
            err_delta_A_pdf = AAC_PDF[[evolution_order]].iloc[index_delta_A_d,0][error_col_index]
            err_alpha = AAC_PDF[[evolution_order]].iloc[index_alpha_d,0][error_col_index]
            err_gamma_pol = AAC_PDF[[evolution_order]].iloc[index_delta_gamma_d,0][error_col_index]
            err_lambda_pol = AAC_PDF[[evolution_order]].iloc[index_delta_lambda_d,0][error_col_index]


            pdf_error = integral_polarized_pdf_regge_error(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
                                            Delta_A_pdf,err_delta_A_pdf,alpha,err_alpha,gamma_pol,err_gamma_pol,lambda_pol,err_lambda_pol,
                                            j,alpha_p,t,evolution_order,error_type)
            return pdf, pdf_error
        else:
            return pdf, 0

def integral_polarized_S_pdf_regge(j,eta,alpha_p,t, evolution_order = "LO", error_type="central"):
        """
        Result of the integral of the Reggeized S(x) PDF based on the given LO parameters and selected errors.
        
        Arguments:
        - j (float) conformal spin,
        - eta (float): skewness (scalar or array)(placeholder for now),
        - alpha_p (float): Regge slope,
        - t (float): Mandelstam t (scalar or array),
        - evolution_order (str. optional): LO, NLO, NNLO
        - error_type: (str. optional) A string indicating whether to use 'central', 'plus', or 'minus' errors. Default is 'central'.
        
        Returns:
        The value of the Reggeized integral together with the error of S(x) based on the selected parameters and error type.
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
        index_A_S=MSTW_PDF[MSTW_PDF["Parameter"] == "A_S"].index[0]
        index_delta_S=MSTW_PDF[MSTW_PDF["Parameter"] == "delta_S"].index[0]
        index_eta_S=MSTW_PDF[MSTW_PDF["Parameter"] == "eta_S"].index[0]
        index_epsilon_S=MSTW_PDF[MSTW_PDF["Parameter"] == "epsilon_S"].index[0]
        index_gamma_S=MSTW_PDF[MSTW_PDF["Parameter"] == "gamma_S"].index[0]

        # Get row index of entry
        index_delta_A_S=AAC_PDF[AAC_PDF["Parameter"] == "Delta_A_S"].index[0]
        index_alpha_S=AAC_PDF[AAC_PDF["Parameter"] == "alpha_S"].index[0]
        index_delta_lambda_S=AAC_PDF[AAC_PDF["Parameter"] == "Delta_lambda_S"].index[0]
        index_delta_gamma_S=AAC_PDF[AAC_PDF["Parameter"] == "Delta_gamma_S"].index[0]

        # Extracting central parameter values
        A_pdf = MSTW_PDF[[evolution_order]].iloc[index_A_S,0][0]
        eta_1 = MSTW_PDF[[evolution_order]].iloc[index_delta_S,0][0]
        eta_2 = MSTW_PDF[[evolution_order]].iloc[index_eta_S,0][0]
        epsilon = MSTW_PDF[[evolution_order]].iloc[index_epsilon_S,0][0]
        gamma_pdf = MSTW_PDF[[evolution_order]].iloc[index_gamma_S,0][0]
        # Extracting parameter values based on error type
        delta_A_pdf = AAC_PDF[[evolution_order]].iloc[index_delta_A_S,0][0]
        alpha = AAC_PDF[[evolution_order]].iloc[index_alpha_S,0][0]
        gamma_pol = AAC_PDF[[evolution_order]].iloc[index_delta_gamma_S,0][0]
        lambda_pol = AAC_PDF[[evolution_order]].iloc[index_delta_lambda_S,0][0]

        pdf = integral_polarized_pdf_regge(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
                                           delta_A_pdf,alpha,gamma_pol,lambda_pol,
                                           j,alpha_p,t,evolution_order)
        
        if error_type != "central":
            err_delta_A_pdf = AAC_PDF[[evolution_order]].iloc[index_delta_A_S,0][error_col_index]
            err_alpha = AAC_PDF[[evolution_order]].iloc[index_alpha_S,0][error_col_index]
            err_gamma_pol = AAC_PDF[[evolution_order]].iloc[index_delta_gamma_S,0][error_col_index]
            err_lambda_pol = AAC_PDF[[evolution_order]].iloc[index_delta_lambda_S,0][error_col_index]

            pdf_error = integral_polarized_pdf_regge_error(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
                                           delta_A_pdf,err_delta_A_pdf,alpha,err_alpha,gamma_pol,err_gamma_pol,lambda_pol,err_lambda_pol,
                                           j,alpha_p,t,evolution_order,error_type)
            return pdf, pdf_error
        else:
            return pdf, 0

def integral_polarized_gluon_pdf_regge(j,eta,alpha_p,t, evolution_order = "LO", error_type="central"):
        """
        Result of the integral of the Reggeized gluon(x) PDF based on the given LO parameters and selected errors.
        
        Arguments:
        - j (float) conformal spin,
        - eta (float): skewness (scalar or array)(placeholder for now),
        - alpha_p (float): Regge slope,
        - t (float): Mandelstam t (scalar or array),
        - evolution_order (str. optional): LO, NLO, NNLO
        - error_type: (str. optional) A string indicating whether to use 'central', 'plus', or 'minus' errors. Default is 'central'.
        
        Returns:
        The value of the Reggeized integral together with the error of gluon(x) based on the selected parameters and error type.
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

        # Get row index of entry
        index_delta_A_g=AAC_PDF[AAC_PDF["Parameter"] == "Delta_A_g"].index[0]
        index_alpha_g=AAC_PDF[AAC_PDF["Parameter"] == "alpha_g"].index[0]
        index_delta_lambda_g=AAC_PDF[AAC_PDF["Parameter"] == "Delta_lambda_g"].index[0]
        index_delta_gamma_g=AAC_PDF[AAC_PDF["Parameter"] == "Delta_gamma_g"].index[0]


        # Extracting central parameter values
        A_pdf = MSTW_PDF[[evolution_order]].iloc[index_A_g,0][0]
        eta_1 = MSTW_PDF[[evolution_order]].iloc[index_delta_g,0][0]
        eta_2 = MSTW_PDF[[evolution_order]].iloc[index_eta_g,0][0]
        epsilon = MSTW_PDF[[evolution_order]].iloc[index_epsilon_g,0][0]
        gamma_pdf = MSTW_PDF[[evolution_order]].iloc[index_gamma_g,0][0]
        # Extracting parameter values based on error type
        delta_A_pdf = AAC_PDF[[evolution_order]].iloc[index_delta_A_g,0][0]
        alpha = AAC_PDF[[evolution_order]].iloc[index_alpha_g,0][0]
        gamma_pol = AAC_PDF[[evolution_order]].iloc[index_delta_gamma_g,0][0]
        lambda_pol = AAC_PDF[[evolution_order]].iloc[index_delta_lambda_g,0][0]

        # print(A_pdf,eta_1,eta_2,epsilon,gamma_pdf)
        # print(delta_A_pdf,alpha,gamma_pol,lambda_pol)

        # We shift by j-1 such that the moments come out right
        # Thus, we need to add a factor of x when resumming the GPD
        pdf = integral_polarized_pdf_regge(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
                                           delta_A_pdf,alpha,gamma_pol,lambda_pol,
                                           j,alpha_p,t,evolution_order)
        if evolution_order != "LO":
            # Additional gluon contribution at NLO and NNLO that is not of the standard form
            index_A_g_prime=MSTW_PDF[MSTW_PDF["Parameter"] == "A_g'"].index[0]
            index_delta_g_prime=MSTW_PDF[MSTW_PDF["Parameter"] == "delta_g'"].index[0]
            index_eta_g_prime=MSTW_PDF[MSTW_PDF["Parameter"] == "eta_g'"].index[0]
            A_pdf_prime = MSTW_PDF[[evolution_order]].iloc[index_A_g_prime,0][0]
            eta_1_prime = MSTW_PDF[[evolution_order]].iloc[index_delta_g_prime,0][0]
            eta_2_prime = MSTW_PDF[[evolution_order]].iloc[index_eta_g_prime,0][0]

            pdf+= (A_pdf_prime * delta_A_pdf *gamma(1+eta_2_prime) *
                    (1-gamma_pol)*gamma(eta_1_prime + j + alpha - alpha_p * t - 1)/
                    gamma(eta_1_prime + eta_2_prime + j + alpha - alpha_p * t) +
                    gamma_pol * gamma(eta_1_prime + j + alpha - alpha_p * t + lambda_pol- 1)/
                    gamma(eta_1_prime + eta_2_prime + j + alpha - alpha_p * t + lambda_pol)
            )

        if error_type != "central":
            err_delta_A_pdf = AAC_PDF[[evolution_order]].iloc[index_delta_A_g,0][error_col_index]
            err_alpha = AAC_PDF[[evolution_order]].iloc[index_alpha_g,0][error_col_index]
            err_gamma_pol = AAC_PDF[[evolution_order]].iloc[index_delta_gamma_g,0][error_col_index]
            err_lambda_pol = AAC_PDF[[evolution_order]].iloc[index_delta_lambda_g,0][error_col_index]

            pdf_error = integral_polarized_pdf_regge_error(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
                                           delta_A_pdf,err_delta_A_pdf,alpha,err_alpha,gamma_pol,err_gamma_pol,lambda_pol,err_lambda_pol,
                                           j,alpha_p,t,evolution_order,error_type)
            if evolution_order != "LO":
                dpdf_dA = A_pdf_prime *gamma(1+eta_2_prime) * (
                    (1-gamma_pol)*gamma(eta_1_prime + j + alpha - alpha_p * t - 1)/
                    gamma(eta_1_prime + eta_2_prime + j + alpha - alpha_p * t) +
                    gamma_pol * gamma(eta_1_prime + j + alpha - alpha_p * t + lambda_pol- 1)/
                    gamma(eta_1_prime + eta_2_prime + j + alpha - alpha_p * t + lambda_pol)
                            )
                dpdf_dalpha = A_pdf_prime * delta_A_pdf * gamma(eta_2_prime + 1) * (
                            (
                                gamma_pol * gamma(eta_1_prime + j + alpha - alpha_p * t + lambda_pol - 1) *
                                (digamma(eta_1_prime + j + alpha - alpha_p * t + lambda_pol - 1) -
                                digamma(eta_1_prime + eta_2_prime + j + alpha - alpha_p * t + lambda_pol))
                            )/\
                            gamma(eta_1_prime + eta_2_prime + j + alpha - alpha_p * t + lambda_pol) + \
                            (
                            (gamma_pol - 1) * gamma(eta_1_prime + j + alpha - alpha_p * t - 1) *
                            (digamma(eta_1_prime + eta_2_prime + j + alpha - alpha_p * t) -
                            digamma(eta_1_prime + j + alpha - alpha_p * t - 1))
                            ) /\
                            gamma(eta_1_prime + eta_2_prime + j + alpha - alpha_p * t)
                            )
                dpdf_dgamma_pol = A_pdf_prime * delta_A_pdf * gamma(eta_2_prime + 1) * (
                                gamma(eta_1_prime + j + alpha - alpha_p * t + lambda_pol - 1) / \
                                gamma(eta_1_prime + eta_2_prime + j + alpha - alpha_p * t + lambda_pol) - \
                                gamma(eta_1_prime + j + alpha - alpha_p * t - 1) / \
                                gamma(eta_1_prime + eta_2_prime + j + alpha - alpha_p * t)
                                )
                dpdf_dlambda_pol = A_pdf_prime * delta_A_pdf * gamma_pol * gamma(eta_2_prime + 1) * (
                                gamma(eta_1_prime + j + alpha - alpha_p * t + lambda_pol - 1) *
                                (digamma(eta_1_prime + j + alpha - alpha_p * t + lambda_pol - 1) -
                                digamma(eta_1_prime + eta_2_prime + j + alpha - alpha_p * t + lambda_pol))
                                )/gamma(eta_1_prime + eta_2_prime + j + alpha - alpha_p * t + lambda_pol)
                Delta_A_pdf = dpdf_dA * err_delta_A_pdf
                Delta_alpha = dpdf_dalpha * err_alpha
                Delta_gamma_pol = dpdf_dgamma_pol * err_gamma_pol
                Delta_lambda_pol = dpdf_dlambda_pol * err_lambda_pol
                # print(dpdf_dA,dpdf_dalpha,dpdf_dgamma_pol,dpdf_dlambda_pol)
                pdf_error += np.sqrt(Delta_A_pdf**2 + Delta_alpha**2 + Delta_gamma_pol**2 +Delta_lambda_pol**2)
            return pdf, pdf_error
        else:
            return pdf, 0

# Define Reggeized conformal moments
def non_singlet_isovector_moment(j,eta,t, moment_label="A",evolve_type="vector", evolution_order="LO",error_type="central"):
    """
    Currently no skewness dependence!
    """
   # Check type
    check_error_type(error_type)
    check_evolution_order(evolution_order)
    check_moment_type_label("non_singlet_isovector",moment_label)
    check_evolve_type(evolve_type)

    alpha_prime = get_regge_slope("non_singlet_isovector",moment_label,evolve_type)

    if moment_label in ["A","B"]:
        if evolution_order == "LO":
            if moment_label == "A":
                norm, gud = 1, 1
            if moment_label == "B":
                norm, gud = 3.83651, 1
        elif evolution_order == "NLO":
           print("norm and gud is todo")
           return
        else:
           raise ValueError(f"Currently unsupporte evolution order {evolution_order}")
        uv, uv_error = integral_uv_pdf_regge(j,eta,alpha_prime,t,evolution_order,error_type)
        dv, dv_error = integral_dv_pdf_regge(j,eta,alpha_prime,t,evolution_order,error_type)
    elif moment_label =="Atilde":
       #norm, gu, gd = 1.29597 , 0.926, 0.341
       #result = norm * (gu * integral_polarized_uv_pdf_regge(j,eta,alpha_prime,t,error_type) - gd * integral_polarized_dv_pdf_regge(j,eta,alpha_prime,t,error_type))
       if evolution_order == "LO":
        norm, gud = 0.78682, 1.2723
       elif evolution_order == "NLO":
           print("norm and gud is todo")
           return
       else:
           raise ValueError(f"Currently unsupporte evolution order {evolution_order}")
       uv, uv_error = integral_polarized_uv_pdf_regge(j,eta,alpha_prime,t,evolution_order,error_type)
       dv, dv_error = integral_polarized_dv_pdf_regge(j,eta,alpha_prime,t,evolution_order,error_type)

    error = error_sign(np.sqrt(uv_error**2+dv_error**2),error_type)
    result = norm * gud * ( uv - dv + error )

    return result

def u_minus_d_pdf_regge(j,eta,t, evolution_order = "LO", error_type="central"):
    """ Currently only experimental function that does not set ubar=dbar"""
    # Check type
    check_error_type(error_type)
    # Value optmized for range -t < 5 GeV
    alpha_prime = 0.675606
    # Normalize to 1 at t = 0
    return 1.107*(integral_uv_pdf_regge(j,eta,alpha_prime,t,evolution_order,error_type)
                    -integral_dv_pdf_regge(j,eta,alpha_prime,t,evolution_order,error_type)
                    -integral_Delta_pdf_regge(j,alpha_prime,t,evolution_order,error_type))

def non_singlet_isoscalar_moment(j,eta,t, moment_label="A",evolve_type="vector", evolution_order = "LO", error_type="central"):
    """
    Currently no skewness dependence!
    """
    # Check type
    check_error_type(error_type)
    check_moment_type_label("non_singlet_isoscalar",moment_label)
    check_evolve_type(evolve_type)

    alpha_prime = get_regge_slope("non_singlet_isoscalar",moment_label,evolve_type)

    if moment_label in ["A","B"]:
        if evolution_order == "LO":
            if moment_label == "A":
                norm, gud = 1, 1
            if moment_label == "B":
                norm, gud = -0.122, 1
        elif evolution_order == "NLO":
            print("norm and gud is todo")
            return
        else:
            raise ValueError(f"Currently unsupporte evolution order {evolution_order}")
        uv, uv_error = integral_uv_pdf_regge(j,eta,alpha_prime,t,evolution_order,error_type)
        dv, dv_error = integral_dv_pdf_regge(j,eta,alpha_prime,t,evolution_order,error_type)
    elif moment_label =="Atilde":
       #norm, gu, gd = 0.783086 , 0.926, 0.341
       #result = norm * (gu * integral_polarized_uv_pdf_regge(j,eta,alpha_prime,t,error_type) + gd * integral_polarized_dv_pdf_regge(j,eta,alpha_prime,t,error_type))
        if evolution_order == "LO":
            norm, gud = 1.71407, 0.4044
        elif evolution_order == "NLO":
            print("norm and gud is todo")
            return
        else:
            raise ValueError(f"Currently unsupporte evolution order {evolution_order}")
        uv, uv_error = integral_polarized_uv_pdf_regge(j,eta,alpha_prime,t,evolution_order,error_type)
        dv, dv_error = integral_polarized_dv_pdf_regge(j,eta,alpha_prime,t,evolution_order,error_type)
    error = error_sign(np.sqrt(uv_error**2+dv_error**2),error_type)
    result = norm * gud * ( uv + dv + error )

    return result

def u_plus_d_pdf_regge(j,eta,t, evolution_order = "LO", error_type="central"):
    """ Currently only experimental function that does not set ubar=dbar"""
    # Check type
    check_error_type(error_type)
    # Value optmized for range -t < 5 GeV
    alpha_prime = 0.949256
    # Normalize to 1 at t = 0
    return 0.973*(integral_uv_pdf_regge(j,eta,alpha_prime,t,evolution_order,error_type)
                    +integral_dv_pdf_regge(j,eta,alpha_prime,t,evolution_order,error_type)
                    +integral_Delta_pdf_regge(j,alpha_prime,t,evolution_order,error_type))

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
    
def quark_singlet_regge_A(j,eta,t, Nf=3, alpha_prime_ud=0.891, moment_label="A", evolution_order ="LO", error_type="central"):
    if moment_label in ["A","B"]:
        uv, uv_error = integral_uv_pdf_regge(j,eta,alpha_prime_ud,t,evolution_order,error_type) 
        dv, dv_error = integral_dv_pdf_regge(j,eta,alpha_prime_ud,t,evolution_order,error_type)
        Spdf, Spdf_error = integral_S_pdf_regge(j,eta,alpha_prime_ud,t,evolution_order,error_type)
        # To Do
        Delta, Delta_error = integral_Delta_pdf_regge(j,eta,alpha_prime_ud,t,evolution_order,error_type)
        s_plus, s_plus_error = integral_s_plus_pdf_regge(j,eta,alpha_prime_ud,t,evolution_order,error_type)
    elif moment_label == "Atilde":
        uv, uv_error = integral_polarized_uv_pdf_regge(j,eta,alpha_prime_ud,t,evolution_order,error_type) 
        dv, dv_error = integral_polarized_dv_pdf_regge(j,eta,alpha_prime_ud,t,evolution_order,error_type)
        # To do
        #Delta = integral_polarized_Delta_pdf_regge(j,eta,alpha_prime_ud,t,error_type)
        Spdf, Spdf_error = integral_polarized_S_pdf_regge(j,eta,alpha_prime_ud,t,evolution_order,error_type)
        # To do
        #s_plus = integral_polarized_s_plus_pdf_regge(j,eta,alpha_prime_ud,t,error_type)
    else:
        raise ValueError(f"Unsupported moment label {moment_label}")
    if Nf == 3 or Nf == 4:
        error = np.sqrt(uv_error**2+dv_error**2+Spdf_error**2)
        result = uv + dv + Spdf
    elif Nf == 2:
        error = np.sqrt(uv_error**2+dv_error**2+Spdf_error**2+s_plus_error**2)
        result = uv + dv + Spdf - s_plus
    elif Nf == 1:
        error = .5*np.sqrt(4*uv_error**2+Spdf_error**2+s_plus_error**2+4*Delta_error**2)
        result = .5*(Spdf-s_plus+2*uv-2*Delta)
    else :
        raise ValueError("Currently only (integer) 1 <= Nf <= 3 supported")
    return result, error
    
def quark_singlet_regge_D(j,eta,t, Nf=3, alpha_prime_ud=0.891,alpha_prime_s=1.828, moment_label="A",evolution_order="LO", error_type="central"):
    if eta == 0:
        return 0, 0
    if moment_label in ["A","B"]:
        uv, uv_error = integral_uv_pdf_regge(j,eta,alpha_prime_ud,t,evolution_order,error_type) 
        dv, dv_error = integral_dv_pdf_regge(j,eta,alpha_prime_ud,t,evolution_order,error_type)
        Delta, Delta_error = integral_Delta_pdf_regge(j,eta,alpha_prime_ud,t,evolution_order,error_type)
        Sv, Sv_error = integral_S_pdf_regge(j,eta,alpha_prime_ud,t,evolution_order,error_type)
        s_plus, s_plus_error = integral_s_plus_pdf_regge(j,eta,alpha_prime_ud,t,evolution_order,error_type)

        uv_s, uv_s_error = integral_uv_pdf_regge(j,eta,alpha_prime_s,t,evolution_order,error_type) 
        dv_s, dv_s_error = integral_dv_pdf_regge(j,eta,alpha_prime_s,t,evolution_order,error_type)
        Sv_s, Sv_s_error = integral_S_pdf_regge(j,eta,alpha_prime_s,t,evolution_order,error_type)
        s_plus_s, s_plus_s_error = integral_s_plus_pdf_regge(j,eta,alpha_prime_s,t,evolution_order,error_type)
        Delta_s, Delta_s_error = integral_Delta_pdf_regge(j,eta,alpha_prime_s,t,evolution_order,error_type)

    elif moment_label == "Atilde":
        uv, uv_error  = integral_polarized_uv_pdf_regge(j,eta,alpha_prime_ud,t,evolution_order,error_type) 
        dv, dv_error = integral_polarized_dv_pdf_regge(j,eta,alpha_prime_ud,t,evolution_order,error_type)
        # To do
        #Delta = integral_polarized_Delta_pdf_regge(j,eta,alpha_prime_ud,t,error_type)
        Sv, Sv_error = integral_polarized_S_pdf_regge(j,eta,alpha_prime_ud,t,evolution_order,error_type)
        # To Do
        #s_plus = integral_polarized_s_plus_pdf_regge(j,eta,alpha_prime_ud,t,error_type)
        uv_s, uv_s_error = integral_polarized_uv_pdf_regge(j,eta,alpha_prime_s,t,evolution_order,error_type) 
        dv_s, dv_s_error = integral_polarized_dv_pdf_regge(j,eta,alpha_prime_s,t,evolution_order,error_type)
        Sv_s, Sv_s_error = integral_polarized_S_pdf_regge(j,eta,alpha_prime_s,t,evolution_order,error_type)
        # To do
        # s_plus_s = integral_polarized_s_plus_pdf_regge(j,eta,alpha_prime_s,t,error_type)
        # Delta_s = integral_polarized_Delta_pdf_regge(j,eta,alpha_prime_s,t,error_type)
    else:
        raise ValueError(f"Unsupported moment label {moment_label}")

    if Nf == 3 or Nf == 4:
        term_1 = uv + dv + Sv
        error_1 = np.sqrt(uv_error**2+dv_error**2+Sv_error**2) 
        term_2 = uv_s + dv_s + Sv_s 
        error_2 = np.sqrt(uv_s_error**2+dv_s_error**2+Sv_s_error**2) 
    elif Nf == 2:
        term_1 = uv + dv + Sv - s_plus
        error_1 = np.sqrt(uv_error**2+dv_error**2+Sv_error**2+s_plus_error**2)
        term_2 = uv_s + dv_s + Sv_s - s_plus_s
        error_2 = np.sqrt(uv_s_error**2+dv_s_error**2+Sv_s_error**2+s_plus_s_error**2) 
    elif Nf == 1:
        term_1 = .5*(Sv-s_plus+2*uv-2*Delta)
        error_1 = .5*np.sqrt(4*uv_error**2+Sv_error**2+s_plus_error**2+4*Delta_error**2)
        term_2 = .5*(Sv_s-s_plus_s+2*uv_s-2*Delta_s)
        error_2 = .5*np.sqrt(4*uv_s_error**2+Sv_s_error**2+s_plus_s_error**2+4*Delta_s_error**2)
    else :
        raise ValueError("Currently only (integer) 1 <= Nf <= 3 supported")
    error = (d_hat(j,eta,t)-1)*np.sqrt(error_1**2+error_2**2)
    result = (d_hat(j,eta,t)-1)*(term_1-term_2)
    return result, error

def quark_singlet_regge(j,eta,t,Nf=3,moment_label="A",evolve_type="vector",evolution_order="LO",error_type="central"):
    # Check type
    check_error_type(error_type)
    check_evolve_type(evolve_type)
    check_moment_type_label("singlet",moment_label)
    check_evolution_order(evolution_order)
    if moment_label == "B":
        prf = -1
    else:
        prf = +1

    # alpha_prime_ud = 0.891
    # alpha_prime_s = 1.828
    alpha_prime_ud = get_regge_slope("non_singlet_isoscalar",moment_label,evolve_type)
    alpha_prime_s, _, _ = get_regge_slope("singlet",moment_label,evolve_type)

    term_1, error_1 = quark_singlet_regge_A(j,eta,t,Nf,alpha_prime_ud,moment_label,evolution_order,error_type)
    term_2, error_2 = quark_singlet_regge_D(j,eta,t,Nf,alpha_prime_ud,alpha_prime_s,moment_label,evolution_order,error_type)
    sum_squared = error_1**2 + error_2**2
    error = np.frompyfunc(mp.sqrt, 1, 1)(sum_squared)
    #error = np.array(mp.sqrt(error_1**2+error_2**2))
    result = term_1 + prf * term_2
    return result, error

def gluon_regge_A(j,eta,t, alpha_prime_T = 0.627,moment_label="A", evolution_order="LO",error_type="central"):
    if moment_label == "A":
        result, error = integral_gluon_pdf_regge(j,eta,alpha_prime_T,t,evolution_order,error_type)
    elif moment_label =="Atilde":
        result, error = integral_polarized_gluon_pdf_regge(j,eta,alpha_prime_T,t,evolution_order,error_type)
    else:
        raise ValueError(f"Unsupported moment label {moment_label}")
    return result, error

def gluon_regge_D(j,eta,t, alpha_prime_T = 0.627, alpha_prime_S = 4.277,moment_label="A",evolution_order="LO", error_type="central"):
    # Check type
    check_error_type(error_type)
    check_moment_type_label("singlet",moment_label)
    if eta == 0:
        return 0, 0 
    else :
        term_1 = (d_hat(j,eta,t)-1)
        term_2, error_2 = gluon_regge_A(j,eta,t,alpha_prime_T,moment_label,evolution_order,error_type)
        if moment_label == "A":
            term_3, error_3 = integral_gluon_pdf_regge(j,eta,t,alpha_prime_S,evolution_order,error_type)
        elif moment_label =="Atilde":
            term_3, error_3 = integral_polarized_gluon_pdf_regge(j,eta,t,alpha_prime_S,evolution_order,error_type)
        else:
            raise ValueError(f"Unsupported moment label {moment_label}")
        error = term_1 * np.sqrt(error_2**2+error_3**2)
        result = term_1 * (term_2-term_3)
        return result, error
    
def gluon_singlet_regge(j,eta,t,moment_label="A",evolve_type="vector", evolution_order="LO",error_type="central"):
    # Check type
    check_error_type(error_type)
    check_evolve_type(evolve_type)
    check_moment_type_label("singlet",moment_label)

    if moment_label == "B":
        prf = -1
    else:
        prf = +1

    _, alpha_prime_T, alpha_prime_S = get_regge_slope("singlet",moment_label,evolve_type)
    term_1, error_1 = gluon_regge_A(j,eta,t,alpha_prime_T,moment_label,evolution_order,error_type)
    if eta == 0:
        result = term_1
        error = error_1
    else :
        term_2, error_2 = gluon_regge_D(j,eta,t,alpha_prime_T,alpha_prime_S,moment_label,evolution_order,error_type)
        sum_squared = error_1**2 + error_2**2
        error = np.frompyfunc(mp.sqrt, 1, 1)(sum_squared)
        #error = np.array(mp.sqrt(error_1**2+error_2**2))
        result = term_1 + prf * term_2
    return result, error

def singlet_moment(j,eta,t,Nf=3,moment_label="A",evolve_type="vector",solution="+",evolution_order="LO",error_type="central"):
    """
    Returns 0 if the moment_label = "B", in accordance with holography and quark model considerations. 
    Otherwise it returns the diagonal combination of quark + gluon moment.
    """
    # Check type
    check_error_type(error_type)
    check_evolve_type(evolve_type)

    if moment_label == "B":
        return 0

    # Switch sign
    if solution == "+":
        solution = "-"
    elif solution == "-":
        solution = "+"
    else:
        raise ValueError("Invalid solution type. Use '+' or '-'.")

    quark_in, quark_in_error = quark_singlet_regge(j,eta,t,Nf,moment_label,evolve_type,evolution_order,error_type)
    # Note: j/6 already included in gamma_qg and gamma_gg definitions
    gluon_prf = (gamma_qg(j-1,Nf,evolve_type,"LO")/
                    (gamma_qq(j-1,Nf,"singlet",evolve_type,"LO")-gamma_pm(j-1,Nf,evolve_type,solution)))
    gluon_in, gluon_in_error = gluon_singlet_regge(j,eta,t,moment_label,evolve_type,evolution_order,error_type)
    sum_squared = quark_in_error**2 + gluon_prf**2*gluon_in_error**2
    error = error_sign(np.frompyfunc(mp.sqrt, 1, 1)(sum_squared),error_type)
    #error = error_sign(np.array(mp.sqrt(quark_in_error**2+gluon_prf**2*gluon_in_error**2)),error_type)
    result = quark_in + gluon_prf * gluon_in + error
    return result

# Initialize the MOMENT_TO_FUNCTION dictionary
# after all functions are defined
initialize_moment_to_function()

################################
##### Evolution Equations ######
################################

def harmonic_number(l,j):
    if l == 1:
        euler_gamma = 0.5772156649
        result = digamma(j+1) + euler_gamma
    else:
        result = mp.zeta(l) - mp.zeta(l,j+1)
    return result

def harmonic_number_prime(l,j):
    if l == 1:
        raise ValueError(f"invalid value l = {l} for primed harmonic sum.")
    term1 = mp.zeta(l)
    term2 = mp.zeta(l,1+mp.floor(.5 * j))
    result = term1 - term2
    return result

def sommerfeld_watson_trapezoid(func,k_0=1,k_1=False,epsilon=.2,k_range=10,n_k=150,alternating_sum=False):
    # Use translational invariance to shift integration to be symmetric around 0
    k_vals = np.linspace(-k_range, k_range, n_k) 
    k_vals-= k_0.imag
    # If k_1 != False, the function values are shifted by the upper bound
    # such that the subtraction of two SWTs values correspond to the finite sum
    k_vals_trig = mp.re(k_0) - epsilon + 1j * k_vals
    if k_1:
        k_vals_summand = k_1 + 1 - epsilon + 1j * k_vals
        # print(k_vals_summand)
    else:
        k_vals_summand = k_vals_trig
    if alternating_sum:
        trig_term = np.array([mp.csc(mp.pi * k) for k in k_vals_trig])
    else:
        trig_term = np.array([mp.cot(mp.pi * k ) for k in k_vals_trig])
    f_vals = np.array([func(k) for k in k_vals_summand])
    integrand = -0.5 * trig_term * f_vals

    integrand_real = np.array([float(mp.re(x)) for x in integrand])
    integrand_imag = np.array([float(mp.im(x)) for x in integrand])

    integral_real = trapezoid(integrand_real, k_vals)
    integral_imag = trapezoid(integrand_imag, k_vals)

    integral = integral_real + 1j * integral_imag
    return integral

def fractional_finite_sum(func,k_0=1,k_1=1,epsilon=.2,k_range=10,n_k=150,alternating_sum=False):
    swt_term_1 = sommerfeld_watson_trapezoid(func,k_0 = k_0,k_1=False, epsilon=epsilon,k_range=k_range,n_k=n_k,alternating_sum=alternating_sum)
    swt_term_2 = sommerfeld_watson_trapezoid(func,k_0 = k_0,k_1=k_1, epsilon=epsilon,k_range=k_range,n_k=n_k,alternating_sum=alternating_sum)
    # print(swt_term_1,swt_term_2)
    ff_sum = swt_term_1 - swt_term_2
    return ff_sum

def harmonic_number_tilde(j):
    def summand_euler_gamma(k):
        result = 1/k**2
        return result
    def summand_digamma(k):
        result = mp.digamma(k + 1)/k**2
        return result
    if isinstance(j, (int, np.integer)):
        result = harmonic_sum_tilde(j)
    else:
        start_time = time.time()
        euler_gamma_terms = - digamma(1) * fractional_finite_sum(summand_euler_gamma,k_0=1,k_1=j,k_range=5,n_k=100)
        digamma_terms = fractional_finite_sum(summand_digamma,k_0=1,k_1=j,k_range=10,n_k=130)
        end_time = time.time()
        print("t1",end_time - start_time)
        result = euler_gamma_terms + digamma_terms
    return result

def harmonic_sum(l,j):
    return sum(1/k**l for k in range(1, j+1))

def harmonic_sum_prime(l,j):
    return 2**(l-1) * sum((1+(-1)**k)/k**l for k in range(1, j+1))

def harmonic_sum_tilde(j):
    return sum((-1)**k/k**2 * harmonic_sum(1,k) for k in range(1, j+1))

def gamma_qq(j,Nf=3,moment_type="non_singlet_isovector",evolve_type="vector",evolution_order="LO"):
    """
    Return conformal qq singlet anomalous dimension for conformal spin-j

    Arguments:
    - j (float): conformal spin
    - moment_type (str. optional): non_singlet_isovector, non_singlet_isoscalar, singlet
    - evolve_type (str. optional): vector or axial
    - evolution_order (str. optional): LO, NLO or NNLO
    """
    check_evolution_order(evolution_order)
    Nc = 3
    c_f = (Nc**2-1)/(2*Nc)
    t_f = .5

    def gamma_qq_lo(j):
        # Belitsky (4.152)
        result = - c_f * (-4*digamma(j+2)+4*digamma(1)+2/((j+1)*(j+2))+3)
        return result
    
    def gamma_qq_nlo(j):
        # Belitsky K.5
        c_a = Nc
        s_0 = harmonic_number(0,j+1)
        s_1 = harmonic_number(1,j+1)
        s_2 = harmonic_number(2,j+1)
        s_2_prime = harmonic_number_prime(2,j+1)
        s_tilde = harmonic_number_tilde(j+1)
        s_3_prime = harmonic_number_prime(3,j+1)
        term1 = (c_f**2 - .5 * c_f * c_a)*(
                    4 * (2 * j + 3)/((j + 1 )**2 * (j + 2)**2)*s_0
                    - 2 * (3 * j**3 +10 * j**2 + 11*j +3)/((j + 1 )**3 * (j + 2)**2)
                    + 4 * (2 * s_1 - 1/((j + 1)*(j + 2))) * (s_2 - s_2_prime)
                    + 16 * s_tilde + 6 * s_2 - .75 - 2 * s_3_prime
                    + 4 * (-1)**(j+1) * (2 * j**2 + 6 * j + 5)/((j + 1 )**3 * (j + 2)**3)
                )
        term2 = c_f * c_a *( s_1 * (134/9 + 2 * (2*j +3 )/((j+1)**2*(j+2)**2) )
                            - 4 * s_1 * s_2 + s_2 *(-13/3 +2/((j+1)*(j+2)))
                            -43/24 - 1/9 * (151*j**4 + 867 * j**3 +1792*j**2 + 1590*j + 523)/\
                            ((j+1)**3*(j+2)**3)
                            )
        term3 = c_f * t_f * Nf * (-40/90*s_1 +8/3*s_2 +1/3 +4/9 * (11*j**2 +27*j+13)/((j+1)**2*(j+2)**2))
        nlo_term = term1 + term2 + term3
        result = nlo_term
        return result
    
    if evolution_order == "LO":
        result = gamma_qq_lo(j)
        return result

    if moment_type in ["non_singlet_isovector","non_singlet_isoscalar"]:
        if evolution_order == "NLO":
            result = gamma_qq_nlo(j)
            return result
        else:
            raise ValueError(f"Currently unsupported evolution order {evolution_order}")
    
    elif moment_type == "singlet":
        if evolution_order == "NLO":
            # Belitsky K.6
            term1 = gamma_qq_nlo(j)
            if evolve_type == "vector":
                term2 = -(
                            4*c_f*t_f*Nf * (5 * j**5 + 57 * j**4 + 227 * j**3 + 427 * j**2 + 404 * j + 160)/
                            (j * (j + 1)**3 * (j + 2)**3 * (j + 3)**2)
                )
            if evolve_type == "axial":
                term2 = + 4 * c_f * t_f * Nf * ((j + 3) * (4 + 5 * j + 3 * j**2 + j**3)) / ((j + 1)**3 * (j + 2)**3)

            result = term1 + term2
            return result
        else:
            raise ValueError(f"Currently unsupported evolution order {evolution_order}")
    else:
        raise ValueError(f"Wrong moment type {moment_type}")
    
  

def gamma_qg(j, Nf=3, evolve_type = "vector",evolution_order="LO"):
    """
    Compute off-diagonal qg anomalous dimension
    Parameters:
    j -- conformal spin
    Nf -- Number of active flavors (default Nf = 3 )
    evolve_type -- "vector" or "axial"
    evolution_order : LO, NLO, NNLO
    Returns:
    Value of anomalous dimension
    """
    # Note addition factor of j/6 (see (K.1) in 0504030)
    Nc = 3
    c_a = Nc
    c_f = (Nc**2-1)/(2*Nc)
    t_f = .5
   
    if evolution_order == "LO":
        if evolve_type == "vector":
            result = -24*Nf*t_f*(j**2+3*j+4)/(j*(j+1)*(j+2)*(j+3))
        elif evolve_type == "axial":
            result = -24*Nf*t_f/((j+1)*(j+2))
        else:
            raise ValueError("evolve_type must be axial or vector")
    elif evolution_order == "NLO":
        # Belitsky K.7 and K.11
        s_1 = harmonic_number(1,j+1)
        s_2 = harmonic_number(2,j+1)
        s_2_prime = harmonic_number_prime(2,j+1)
        if evolve_type == "vector":
            term1 = -2 * c_a * t_f * Nf * (
                (-2 * s_1**2 + 2 * s_2 - 2 * s_2_prime) * ((j**2 + 3 * j + 4) / ((j + 1) * (j + 2) * (j + 3))) +
                (960 + 2835 * j + 4057 * j**2 + 3983 * j**3 + 3046 * j**4 + 1777 * j**5 +
                731 * j**6 + 195 * j**7 + 30 * j**8 + 2 * j**9) / (j * (j + 1)**3 * (j + 2)**3 * (j + 3)**3) +
                (-1)**(j + 1) * (141 + 165 * j + 92 * j**2 + 27 * j**3 + 3 * j**4) /
                ((j + 1) * (j + 2)**3 * (j + 3)**3) +
                8 * (2 * j + 5) / ((j + 2)**2 * (j + 3)**2) * s_1
            )
            term2 = -2 * c_f * t_f * Nf * (
                (2 * s_1**2 - 2 * s_2 + 5) * ((j**2 + 3 * j + 4) / ((j + 1) * (j + 2) * (j + 3))) -
                4 * s_1 / ((j + 1)**2) +
                (11 * j**4 + 70 * j**3 + 159 * j**2 + 160 * j + 64) /
                ((j + 1)**3 * (j + 2)**3 * (j + 3))
            )
        elif evolve_type == "axial":
            term1 = 2 * c_f * t_f * Nf * (
                - (j * (16 + 49 * j + 60 * j**2 + 30 * j**3 + 5 * j**4)) / ((j + 1)**3 * (j + 2)**3) +
                (4 * j / ((j + 1)**2 * (j + 2))) * s_1 -
                (2 * j / ((j + 1) * (j + 2))) * (s_1**2 - s_2)
            )

            term2 = 4 * c_a * t_f * Nf * (
                (8 + 4 * j - 7 * j**2 - 10 * j**3 - 6 * j**4 - j**5) / ((j + 1)**3 * (j + 2)**3) -
                (4 * s_1 / ((j + 1) * (j + 2)**2)) +
                (j / ((j + 1) * (j + 2))) * (s_1**2 - s_2 + s_2_prime)
            )
        else:
            raise ValueError("evolve_type must be axial or vector")
        result = term1 + term2
    else:
        raise ValueError(f"Currently unsupported evolution order {evolution_order}")

    # Match forward to Wilson anomalous dimension
    result*=j/6
    return result

def gamma_gq(j, Nf=3,evolve_type = "vector",evolution_order="LO"):
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
    c_a = Nc
    c_f = (Nc**2-1)/(2*Nc)
    t_f = .5
    
    if evolution_order == "LO":
        if evolve_type == "vector":
            result = -c_f*(j**2+3*j+4)/(3*(j+1)*(j+2))
        elif evolve_type == "axial":
            result = -c_f*j*(j+3)/(3*(j+1)*(j+2))
        else:
            raise ValueError("Type must be axial or vector")
    elif evolution_order == "NLO":
        # Belitsky K.8 and K.12
        s_1 = harmonic_number(1,j+1)
        s_2 = harmonic_number(2,j+1)
        s_2_prime = harmonic_number_prime(2,j+1)
        if evolve_type == "vector":
            term1 = -c_f**2 * (
                (-2 * s_1**2 + 10 * s_1 - 2 * s_2) * ((j**2 + 3 * j + 4) / (j * (j + 1) * (j + 2))) -
                4 * s_1 / ((j + 2)**2) -
                (12 * j**6 + 102 * j**5 + 373 * j**4 + 740 * j**3 + 821 * j**2 + 464 * j + 96) /
                (j * (j + 1)**3 * (j + 2)**3)
            )
            term2 = -2 * c_a * c_f * (
                (s_1**2 + s_2 - s_2_prime) * ((j**2 + 3 * j + 4) / (j * (j + 1) * (j + 2))) +
                (1296 + 10044 * j + 30945 * j**2 + 47954 * j**3 + 42491 * j**4 + 22902 * j**5 + 7515 * j**6 +
                1384 * j**7 + 109 * j**8) / (9 * j**2 * (j + 1)**3 * (j + 2)**2 * (j + 3)**2) +
                (-1)**(j + 1) * (8 + 9 * j + 4 * j**2 + j**3) / ((j + 1)**3 * (j + 2)**3) -
                (17 * j**4 + 68 * j**3 + 143 * j**2 + 128 * j + 24) / (3 * j**2 * (j + 1)**2 * (j + 2)) * s_1
            )
            term3 = -(8 / 3) * c_f * t_f * Nf * (
                (s_1 - 8 / 3) * ((j**2 + 3 * j + 4) / (j * (j + 1) * (j + 2))) + 1 / ((j + 2)**2)
            )
        elif evolve_type == "axial": 
            term1 = 8 * c_f * t_f * Nf * (
                ((j + 3) * (5 * j + 7)) / (9 * (j + 1) * (j + 2)**2) -
                ((j + 3) / (3 * (j + 1) * (j + 2))) * s_1
            )
            term2 = c_f**2 * (
                ((j + 3) * (3 * j + 4) * (3 + 14 * j + 12 * j**2 + 3 * j**3)) / ((j + 1)**3 * (j + 2)**3) -
                (2 * (j + 3) * (3 * j + 4)) / ((j + 1) * (j + 2)**2) * s_1 +
                (2 * (j + 3) / ((j + 1) * (j + 2))) * (s_1**2 + s_2)
            )
            term3 = 2 * c_a * c_f * (
                - (750 + 2380 * j + 3189 * j**2 + 2098 * j**3 + 651 * j**4 + 76 * j**5) / (9 * (j + 1)**3 * (j + 2)**3) +
                ((45 + 44 * j + 11 * j**2) / (3 * (j + 1)**2 * (j + 2))) * s_1 +
                ((j + 3) / ((j + 1) * (j + 2))) * (-s_1**2 - s_2 + s_2_prime)
            )
        else:
            raise ValueError("Type must be axial or vector")
        result = term1 + term2 + term3
    else:
        raise ValueError(f"Currently unsupported evolution order {evolution_order}")


    # Match forward to Wilson anomalous dimension
    result*=6/j
    return result

def gamma_gg(j, Nf = 3, evolve_type = "vector",evolution_order="LO"):
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
    c_a = Nc
    c_f = (Nc**2-1)/(2*Nc)
    t_f = .5
    beta_0 = 2/3* Nf - 11/3 * Nc

    if evolution_order == "LO":
            if evolve_type == "vector":
                result = -c_a*(-4*digamma(j+2)+4*digamma(1)+8*(j**2+3*j+3)/(j*(j+1)*(j+2)*(j+3))-beta_0/c_a)
            elif evolve_type == "axial":
                result = -c_a*(-4*digamma(j+2)+4*digamma(1)+8/((j+1)*(j+2))-beta_0/c_a)
            else:
                raise ValueError("Type must be axial or vector")
    elif evolution_order == "NLO":
        s_1 = harmonic_number(1,j+1)
        s_2_prime = harmonic_number_prime(2,j+1)
        s_3_prime = harmonic_number_prime(3,j+1)
        s_tilde = harmonic_number_tilde(j+1)
        if evolve_type == "vector":
            term1 = c_a * t_f * Nf * (
                (-40 / 9) * s_1 + (8 / 3) + (8 / 9) * (19 * j**4 + 114 * j**3 + 275 * j**2 + 312 * j + 138) /
                (j * (j + 1)**2 * (j + 2)**2 * (j + 3))
            )
            term2 = c_f * t_f * Nf * (
                2 + 4 * (2 * j**6 + 16 * j**5 + 51 * j**4 + 74 * j**3 + 41 * j**2 - 8 * j - 16) /
                (j * (j + 1)**3 * (j + 2)**3 * (j + 3))
            )
            term3 = c_a**2 * (
                (134 / 9) * s_1 + 16 * s_1 * (2 * j**5 + 15 * j**4 + 48 * j**3 + 81 * j**2 + 66 * j + 18) /
                (j**2 * (j + 1)**2 * (j + 2)**2 * (j + 3)**2) -
                (16 / 3) + 8 * s_2_prime * (j**2 + 3 * j + 3) / (j * (j + 1) * (j + 2) * (j + 3)) -
                4 * s_1 * s_2_prime +
                8 * s_tilde - s_3_prime -
                (1 / 9) * (457 * j**9 + 6855 * j**8 + 44428 * j**7 + 163542 * j**6) /
                (j**2 * (j + 1)**3 * (j + 2)**3 * (j + 3)**3) -
                (1 / 9) * (376129 * j**5 + 557883 * j**4 + 529962 * j**3 + 308808 * j**2 + 101088 * j + 15552) /
                (j**2 * (j + 1)**3 * (j + 2)**3 * (j + 3)**3)
            )

        elif evolve_type == "axial":
            term1 = 2 * c_f * t_f * Nf * (
                (8 + 30 * j + 70 * j**2 + 71 * j**3 + 35 * j**4 + 9 * j**5 + j**6) /
                ((j + 1)**3 * (j + 2)**3)
            )
            term2 = 8 * c_a * t_f * Nf * (
                (35 + 75 * j + 52 * j**2 + 18 * j**3 + 3 * j**4) / (9 * (j + 1)**2 * (j + 2)**2) -
                (5 / 9) * s_1
            )
            term3 = c_a**2 * (
                - (1768 + 5250 * j + 7075 * j**2 + 4974 * j**3 + 1909 * j**4 + 432 * j**5 + 48 * j**6) /
                (9 * (j + 1)**3 * (j + 2)**3) +
                (2 / 9) * (484 + 948 * j + 871 * j**2 + 402 * j**3 + 67 * j**4) / ((j + 1)**2 * (j + 2)**2) * s_1 +
                (8 / ((j + 1) * (j + 2))) * s_2_prime -
                4 * s_1 * s_2_prime - s_3_prime +
                8 * s_tilde
            )
        else:
            raise ValueError("Type must be axial or vector")
        result = term1 + term2 + term3
    else:
        raise ValueError(f"Currently unsupported evolution order {evolution_order}")

    return result

def d_element(j,k):
    """ Belistky (4.204)"""
    if j == k:
        raise ValueError("j and k must be unqual")
    result = - .5 * (1 + (-1)**(j-k)) * (2 * k + 3)/((j - k)*(j + k + 3))
    return result

def digamma_A(j,k):
    """ Belistky (4.212)"""
    if j == k:
        raise ValueError("j and k must be unqual")
    result = digamma(.5* (j + k + 4)) - digamma(.5 * (j-k)) + 2 * digamma(j - k) - digamma(j + 2) - digamma(1)
    return result

def discrete_theta(j,k):
    """Returns 1 if j > k and 0 otherwise
    """
    return int(j > k)

def conformal_anomaly_qq(j,k):
    """Belitsky (4.206). Equal for vector and axial """
    if j == k:
        raise ValueError("j and k must be unqual")
    Nc = 3
    c_f = (Nc**2-1)/(2*Nc)
    result =  -c_f * (1 + (-1)**(j - k)) * discrete_theta(j-2,k) * ((3 + 2 * k) / ((j - k) * (j + k + 3))) * (
        2 * digamma_A(j, k) + 
        (digamma_A(j, k) - digamma(j + 2) + digamma(1)) * ((j - k) * (j + k + 3)) / ((k + 1) * (k + 2))
        )
    return result

def conformal_anomaly_gq(j,k):
    """Belitsky (4.208). Equal for vector and axial """
    if j == k:
        raise ValueError("j and k must be unqual")
    Nc = 3
    c_f = (Nc**2-1)/(2*Nc)
    result = -c_f * (1 + (-1)**(j - k)) * discrete_theta(j-2,k) * (1 / 6) * ((3 + 2 * k) / ((k + 1) * (k + 2)))
    return result

def conformal_anomaly_gg(j,k):
    """Belitsky (4.209). Equal for vector and axial """
    if j == k:
        raise ValueError("j and k must be unqual")
    Nc = 3
    c_a = Nc
    result = (
            -c_a * (1 + (-1) ** (j - k)) * discrete_theta(j-2,k) *
            ((3 + 2 * k) / ((j - k) * (j + k + 3))) *
            (
                2 * digamma_A(j,k) +
                (digamma_A(j,k) - digamma(j + 2) + digamma(1)) *
                ((gamma(j + 4) * gamma(k)) / (gamma(j) * gamma(k + 4)) - 1) +
                2 * (j - k) * (j + k + 3) * (gamma(k) / gamma(k + 4))
            )
        )
    return result

def gamma_qq_nd(j,k, Nf=3, evolve_type = "vector",evolution_order="LO"):
    """ Belistky (4.203)"""
    if evolution_order == "LO":
        return 0
    if k >= j:
        return 0
    Nc = 3
    beta_0 = 2/3* Nf - 11/3 * Nc
    if evolution_order == "NLO":
        term1 = (gamma_qq(j,evolution_order="LO")-gamma_qq(k,evolution_order="LO"))* \
                (d_element(j,k) * (beta_0 - gamma_qq(k,evolution_order="LO")) + conformal_anomaly_qq(j,k))
        term2 = - (gamma_qg(j,Nf,evolve_type,"LO") - gamma_qg(k,Nf,evolve_type,"LO")) * d_element(j,k) * gamma_gq(j,Nf,evolve_type,"LO")
        term3 = gamma_qg(j,Nf,evolve_type,"LO") * conformal_anomaly_gq(j,k)
        result = term1 + term2 + term3
    else:
        raise ValueError(f"Currently unsupported evolution order {evolution_order}")
    return result

def gamma_qg_nd(j,k, Nf=3, evolve_type = "vector",evolution_order="LO"):
    """ Belistky (4.203)"""
    if evolution_order == "LO":
        return 0
    if k >= j:
        return 0
    Nc = 3
    beta_0 = 2/3* Nf - 11/3 * Nc
    if evolution_order == "NLO":
        term1 = (gamma_qg(j, Nf, evolve_type, "LO") - gamma_qg(k, Nf, evolve_type, "LO")) * \
                d_element(j, k) * (beta_0 - gamma_gg(k, Nf, evolve_type, "LO"))
        term2 = - (gamma_qq(j, evolution_order="LO") - gamma_qq(k, evolution_order="LO")) * \
                d_element(j, k) * gamma_qg(k, Nf, evolve_type, "LO")
        term3 = gamma_qg(j, Nf, evolve_type, "LO") * conformal_anomaly_gg(j, k)
        term4 = - conformal_anomaly_qq(j, k) * gamma_qg(k, Nf, evolve_type, "LO")
        result = term1 + term2 + term3 + term4
    else:
        raise ValueError(f"Currently unsupported evolution order {evolution_order}")
    return result

def gamma_gq_nd(j,k, Nf=3, evolve_type = "vector",evolution_order="LO"):
    """ Belistky (4.203)"""
    if evolution_order == "LO":
        return 0
    if k >= j:
        return 0
    Nc = 3
    beta_0 = 2/3* Nf - 11/3 * Nc
    if evolution_order == "NLO":
        term1 = (gamma_gq(j, Nf, evolve_type, "LO") - gamma_gq(k, Nf, evolve_type, "LO")) * \
                d_element(j, k) * (beta_0 - gamma_qq(k, evolution_order="LO"))
        term2 = - (gamma_gg(j, Nf, evolve_type, "LO") - gamma_gg(k, Nf, evolve_type, "LO")) * \
                d_element(j, k) * gamma_gq(k, Nf, evolve_type, "LO")
        term3 = gamma_gq(j, Nf, evolve_type, "LO") * conformal_anomaly_qq(j, k)
        term4 = - conformal_anomaly_gg(j, k) * gamma_gq(k, Nf, evolve_type, "LO")
        term5 = (gamma_gg(j, Nf, evolve_type, "LO") - gamma_qq(k,evolution_order="LO")) * conformal_anomaly_gq(j, k)

        result = term1 + term2 + term3 + term4 + term5
    else:
        raise ValueError(f"Currently unsupported evolution order {evolution_order}")
    return result

def gamma_gg_nd(j,k, Nf=3, evolve_type = "vector",evolution_order="LO"):
    """ Belistky (4.203)"""
    if evolution_order == "LO":
        return 0
    if k >= j:
        return 0
    Nc = 3
    beta_0 = 2/3* Nf - 11/3 * Nc
    if evolution_order == "NLO":
        term1 = (gamma_gg(j, Nf, evolve_type, "LO") - gamma_gg(k, Nf, evolve_type, "LO")) * \
                (d_element(j, k) * (beta_0 - gamma_gg(k, Nf, evolve_type, "LO")) + conformal_anomaly_gg(j, k))
        term2 = - (gamma_gq(j, Nf, evolve_type, "LO") - gamma_gq(k, Nf, evolve_type, "LO")) * \
                d_element(j, k) * gamma_qg(k, Nf, evolve_type, "LO")
        term3 = - conformal_anomaly_gq(j, k) * gamma_qg(k, Nf, evolve_type, "LO")
        result = term1 + term2 + term3
    else:
        raise ValueError(f"Currently unsupported evolution order {evolution_order}")
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
    
    base = gamma_qq(j,evolution_order="LO")+gamma_gg(j,Nf,evolve_type,evolution_order="LO")
    root = np.sqrt((gamma_qq(j,evolution_order="LO")-gamma_gg(j,Nf,evolve_type,evolution_order="LO"))**2+4*gamma_gq(j,Nf,evolve_type,evolution_order="LO")*gamma_qg(j,Nf,evolve_type,evolution_order="LO"))
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

    result = 1-gamma_pm(j,Nf,evolve_type,solution)/gamma_qq(j,evolution_order="LO")
    return result

def R_qq(j, Nf,moment_type,evolve_type):
    Nc = 3
    c_a = Nc
    c_f = (Nc**2-1)/(2*Nc)
    beta_0 = 2/3* Nf - 11/3 * Nc
    beta_1 = 10/3 * c_a * Nf + 2 * c_f * Nf -34/3 * c_a**2
    term1 = gamma_qq(j,Nf,moment_type,evolve_type,evolution_order="NLO")
    term2 = - .5 * beta_1/beta_0 * gamma_qq(j,Nf,moment_type,evolve_type,evolution_order="LO")
    result = term1 + term2
    return result

def R_qg(j, Nf,moment_type,evolve_type):
    Nc = 3
    c_a = Nc
    c_f = (Nc**2-1)/(2*Nc)
    beta_0 = 2/3* Nf - 11/3 * Nc
    beta_1 = 10/3 * c_a * Nf + 2 * c_f * Nf -34/3 * c_a**2
    term1 = gamma_qg(j,Nf,evolve_type,"NLO")
    term2 = - .5 * beta_1/beta_0 * gamma_qg(j,Nf,evolve_type,"LO")
    result = term1 + term2
    return result

def R_gq(j, Nf,moment_type,evolve_type):
    Nc = 3
    c_a = Nc
    c_f = (Nc**2-1)/(2*Nc)
    beta_0 = 2/3* Nf - 11/3 * Nc
    beta_1 = 10/3 * c_a * Nf + 2 * c_f * Nf -34/3 * c_a**2
    term1 = gamma_gq(j,Nf,evolve_type,"NLO")
    term2 = - .5 * beta_1/beta_0 * gamma_gq(j,Nf,evolve_type,"LO")
    result = term1 + term2
    return result

def R_gg(j, Nf,moment_type,evolve_type):
    Nc = 3
    c_a = Nc
    c_f = (Nc**2-1)/(2*Nc)
    beta_0 = 2/3* Nf - 11/3 * Nc
    beta_1 = 10/3 * c_a * Nf + 2 * c_f * Nf -34/3 * c_a**2
    term1 = gamma_gg(j,Nf,evolve_type,"NLO")
    term2 = - .5 * beta_1/beta_0 * gamma_gg(j,Nf,evolve_type,"LO")
    result = term1 + term2
    return result

def evolve_conformal_moment(j,eta,t,mu,Nf = 3,particle="quark",moment_type="non_singlet_isovector",moment_label ="A", evolution_order = "LO", error_type = "central"):
    """
    Evolve the conformal moment F_{j}^{+-} from some input scale mu_in to some other scale mu. 

    Arguments:
    j (float): conformal spin
    eta (float): skewness parameter
    t (float): Mandelstam t
    mu (float): Resolution scale
    Nf (int. optional): Number of active flavors (default Nf = 3)
    moment_type (str. optional): non_singlet_isovector, non_singlet_isoscalar, or singlet
    moment_label (str. optional): A(Tilde) B(Tilde) depending on H(Tilde) or E(Tilde) GPD etc.
    evolution_order (str. optional): LO, NLO
    error_type (str. optional): Choose central, upper or lower value for input PDF parameters

    Returns:
    The value of the evolved conformal moment at scale mu
    """
    
    check_particle_type(particle)
    check_moment_type_label(moment_type,moment_label)
    check_error_type(error_type)
    check_evolution_order(evolution_order)
    if particle == "gluon" and moment_type != "singlet":
        raise ValueError("Gluon is only singlet")

    # Set parameters
    Nc = 3
    beta_0 = 2/3* Nf - 11/3 * Nc

    # Extract fixed quantities
    alpha_s_in = get_alpha_s()
    alpha_s_evolved = evolve_alpha_s(mu,Nf,evolution_order)

    moment_in, evolve_type = MOMENT_TO_FUNCTION.get((moment_type, moment_label))

    ga_qq = gamma_qq(j-1,Nf,moment_type,evolve_type,evolution_order="LO")
    # Roots  of LO anomalous dimensions
    ga_p = gamma_pm(j-1,Nf,evolve_type,"+")
    ga_m = gamma_pm(j-1,Nf,evolve_type,"-")

    if moment_type == "singlet":
        moment_in_p = moment_in(j,eta,t, Nf, moment_label, evolve_type,"+",evolution_order,error_type)
        moment_in_m = moment_in(j,eta,t, Nf, moment_label, evolve_type,"-",evolution_order,error_type)
        ga_gq = gamma_gq(j-1,Nf, evolve_type,"LO")
        ga_qg = gamma_qg(j-1,Nf, evolve_type,"LO")
    if evolution_order != "LO":
        ga_gg = gamma_gg(j-1,Nf,evolve_type,"LO")
        r_qq = R_qq(j-1,Nf,moment_type,evolve_type)
        r_qg = R_qg(j-1,Nf,moment_type,evolve_type)
        r_gq = R_gq(j-1,Nf,moment_type,evolve_type)
        r_gg = R_gg(j-1,Nf,moment_type,evolve_type) 

    # Precompute alpha_s fraction:
    alpha_frac  = (alpha_s_in/alpha_s_evolved)    

    # Functions appearing in LO evolution
    def get_ga_moment(solution):
        # switch switch + <-> - when necessary
        if solution == "+":
            return ga_p, ga_m, moment_in_p
        elif solution == "-": 
            return ga_m, ga_p, moment_in_m
        else:
            raise ValueError(f"Wrong solution type: {solution}")

    def A_lo_quark(solution):
        # The switch also takes care of the relative minus sign
        ga_p, ga_m, moment = get_ga_moment(solution)
        result = (ga_qq - ga_m)/(ga_p - ga_m) * alpha_frac**(ga_p/beta_0) * moment
        return result
    
    def A_lo_gluon(solution):
        ga_p, ga_m, moment = get_ga_moment(solution)
        result = ga_gq/(ga_p - ga_m) * alpha_frac**(ga_p/beta_0) * moment
        return result

    def A_nlo_quark(solution):
        ga_p, ga_m, moment = get_ga_moment(solution)
        term1 = - (alpha_s_evolved - alpha_s_in)/(2*np.pi)/beta_0 * alpha_frac**(ga_p/beta_0) / \
                (ga_p - ga_m)**2 * ( moment )
        term2 = (ga_qq - ga_m) * (r_qq * (ga_qq-ga_m) + r_qg * ga_gq)
        term3 = ga_qg * (r_gq * (ga_qq - ga_m) + r_gg * ga_gq)
        result = term1 * (term2 + term3)
        return result
    
    def B_nlo_quark(solution):
        ga_p, ga_m, moment = get_ga_moment(solution)
        term1 = alpha_s_evolved/(2*np.pi)/(ga_m - ga_p + beta_0) * moment / (ga_p - ga_m)**2
        term2 = (1 - alpha_frac**((ga_m - ga_p + beta_0)/beta_0)) * alpha_frac**(ga_p/beta_0)
        term3 = ((ga_qq - ga_p) * (r_qq * (ga_qq - ga_m) + r_qg * ga_gq) + ga_qg * (r_gq * (ga_qq - ga_m) + r_gg * ga_gq))
        result = term1 * term2 * term3
        return result

    def A_nlo_gluon(solution):
        ga_p, ga_m, moment = get_ga_moment(solution)
        term1 = - (alpha_s_evolved - alpha_s_in)/(2*np.pi)/beta_0 * alpha_frac**(ga_p/beta_0) / \
                (ga_p - ga_m)**2 * ( moment )
        term2 = ga_gq * (r_qq * (ga_qq-ga_m) + r_qg * ga_gq)
        term3 = (ga_gg - ga_m) * (r_gq * (ga_qq - ga_m) + r_gg * ga_gq)
        result = term1 * (term2 + term3)
        return result

    def B_nlo_gluon(solution):
        ga_p, ga_m, moment = get_ga_moment(solution)
        term1 = alpha_s_evolved/(2*np.pi)/(ga_m - ga_p + beta_0) * moment / (ga_p - ga_m)**2
        term2 = (1 - alpha_frac**((ga_m - ga_p + beta_0)/beta_0)) * alpha_frac**(ga_p/beta_0)
        term3 = (ga_gq  * (r_qq * (ga_qq - ga_m) + r_qg * ga_gq) + (ga_gg - ga_p) * (r_gq * (ga_qq - ga_m) + r_gg * ga_gq) )
        result = term1 * term2 * term3
        return result

    def prf_T_nlo(k):
        ga_j_p, ga_j_m = ga_p, ga_m
        ga_k_p, ga_k_m = gamma_pm(k-1,Nf,evolve_type,"+"), gamma_pm(k-1,Nf,evolve_type,"-")
        alpha_term = alpha_s_evolved/(2*np.pi)
        ga_j_p_plus_ga_k_m = ga_j_p + ga_k_m + beta_0
        ga_j_p_minus_ga_k_m = ga_j_p - ga_k_m + beta_0
        ga_j_m_minus_ga_k_p = ga_j_m - ga_k_p
        ga_j_m_minus_ga_k_m = ga_j_m - ga_k_m
        ga_kk_times_ga_jj = (ga_k_p - ga_k_m)*(ga_j_p - ga_j_m)
        prf_T_1 = - alpha_term/ga_j_p_plus_ga_k_m * (1 - alpha_frac**(ga_j_p_plus_ga_k_m/beta_0))/ga_kk_times_ga_jj
        prf_T_2 = - alpha_term/ga_j_p_minus_ga_k_m * (1 - alpha_frac**(ga_j_p_minus_ga_k_m/beta_0))/ga_kk_times_ga_jj
        prf_T_3 = - alpha_term/ga_j_m_minus_ga_k_p * (1 - alpha_frac**(ga_j_m_minus_ga_k_p/beta_0))/ga_kk_times_ga_jj
        prf_T_4 = - alpha_term/ga_j_m_minus_ga_k_m * (1 - alpha_frac**(ga_j_m_minus_ga_k_m/beta_0))/ga_kk_times_ga_jj

        return prf_T_1, prf_T_2, prf_T_3, prf_T_4
    
    def T_nlo_quark():
        # Sum from k = 2 to j - 1
        # Note T = 0 for j=k
        quark_non_diagonal_part = 0
        ga_j_p, ga_j_m = ga_p, ga_m
        for k in range(2,j):
            ga_k_p, ga_k_m = gamma_pm(k-1,Nf,evolve_type,"+"), gamma_pm(k-1,Nf,evolve_type,"-")
            ga_qq_k = gamma_qq(k-1,evolution_order="LO")
            ga_gq_k = gamma_gq(k-1,Nf, evolve_type,"LO")
            ga_qg_k = gamma_qg(k-1,Nf, evolve_type,"LO")
            ga_qq_nd = gamma_qq_nd(j-1,k-1,Nf,evolve_type,"NLO")
            ga_qg_nd = gamma_qg_nd(j-1,k-1,Nf,evolve_type,"NLO")
            ga_gq_nd = gamma_gq_nd(j-1,k-1,Nf,evolve_type,"NLO")
            ga_gg_nd = gamma_gg_nd(j-1,k-1,Nf,evolve_type,"NLO")

            prf_T_1, prf_T_2, prf_T_3, prf_T_4 = prf_T_nlo(k)
            moment_k_p = moment_in(k,eta,t, Nf, moment_label, evolve_type,"+",evolution_order,error_type)
            moment_k_m = moment_in(k,eta,t, Nf, moment_label, evolve_type,"-",evolution_order,error_type)
            T_1_top = prf_T_1 * moment_k_p * (
                (ga_qq - ga_j_m) * ( ga_qq_nd * (ga_qq_k - ga_k_m) + ga_qg_nd * ga_gq_k )
                + ga_qg * ( ga_gq_nd * (ga_qq_k - ga_k_m) + ga_gg_nd * ga_gq_k )                             
            )          
            T_2_top = prf_T_2 * moment_k_m *  (
                (ga_qq - ga_j_m) * ( ga_qq_nd * (ga_qq_k - ga_k_p) + ga_qg_nd * ga_gq_k )
                + ga_qg * ( ga_gq_nd * (ga_qq_k - ga_k_p) + ga_gg_nd * ga_gq_k )                             
            )                    
            T_3_top = - prf_T_3 * moment_k_p *  (
                (ga_qq - ga_j_p) * ( ga_qq_nd * (ga_qq_k - ga_k_m) + ga_qg_nd * ga_gq_k )
                + ga_qg * ( ga_gq_nd * (ga_qq_k - ga_k_m) + ga_gg_nd * ga_gq_k )                             
            )
            T_4_top = prf_T_4 * moment_k_m *  (
                (ga_qq - ga_j_p) * ( ga_qq_nd * (ga_qq_k - ga_k_p) + ga_qg_nd * ga_gq_k )
                + ga_qg * ( ga_gq_nd * (ga_qq_k - ga_k_p) + ga_gg_nd * ga_gq_k )                             
            )
            # 
            quark_non_diagonal_part += eta**(j-k) * ( T_1_top + T_2_top + T_3_top + T_4_top)

        return quark_non_diagonal_part

    def T_nlo_gluon():
        # Sum from k = 2 to j - 1
        # Note T = 0 for j=k
        gluon_non_diagonal_part = 0
        ga_j_p, ga_j_m = ga_p, ga_m
        for k in range(2,j):
            ga_k_p, ga_k_m = gamma_pm(k-1,Nf,evolve_type,"+"), gamma_pm(k-1,Nf,evolve_type,"-")
            ga_qq_k = gamma_qq(k-1,evolution_order="LO")
            ga_gq_k = gamma_gq(k-1,Nf, evolve_type,"LO")
            ga_qg_k = gamma_qg(k-1,Nf, evolve_type,"LO")
            ga_qq_nd = gamma_qq_nd(j-1,k-1,Nf,evolve_type,"NLO")
            ga_qg_nd = gamma_qg_nd(j-1,k-1,Nf,evolve_type,"NLO")
            ga_gq_nd = gamma_gq_nd(j-1,k-1,Nf,evolve_type,"NLO")
            ga_gg_nd = gamma_gg_nd(j-1,k-1,Nf,evolve_type,"NLO")

            prf_T_1, prf_T_2, prf_T_3, prf_T_4 = prf_T_nlo(k)
            moment_k_p = moment_in(k,eta,t, Nf, moment_label, evolve_type,"+",evolution_order,error_type)
            moment_k_m = moment_in(k,eta,t, Nf, moment_label, evolve_type,"-",evolution_order,error_type)
            T_1_bot = prf_T_1 * moment_k_p * (
                ga_gq * ( ga_qq_nd * (ga_qq_k - ga_k_m) + ga_qg_nd * ga_gq_k )    
                + (ga_gg - ga_j_m) * ( ga_gq_nd * (ga_qq_k - ga_k_m) + ga_gg_nd * ga_gq_k )                         
            )
            T_2_bot = prf_T_2 * moment_k_m * (
                ga_gq * ( ga_qq_nd * (ga_qq_k - ga_k_p) + ga_qg_nd * ga_gq_k )    
                + (ga_gg - ga_j_m) * ( ga_gq_nd * (ga_qq_k - ga_k_p) + ga_gg_nd * ga_gq_k )                         
            )
            T_3_bot = - prf_T_3 * moment_k_p * (
                ga_gq * ( ga_qq_nd * (ga_qq_k - ga_k_m) + ga_qg_nd * ga_gq_k )    
                + (ga_gg - ga_j_p) * ( ga_gq_nd * (ga_qq_k - ga_k_m) + ga_gg_nd * ga_gq_k )   
            )                      
            T_4_bot = prf_T_4 * moment_k_m * (
                ga_gq * ( ga_qq_nd * (ga_qq_k - ga_k_p) + ga_qg_nd * ga_gq_k )    
                + (ga_gg - ga_j_p) * (ga_gq_nd * (ga_qq_k - ga_k_p) + ga_gg_nd * ga_gq_k )                         
            )
            gluon_non_diagonal_part = eta**(j-k) *  ( T_1_bot + T_2_bot + T_3_bot + T_4_bot )
        return gluon_non_diagonal_part
    # print("check")
    # print("lo",A_lo_quark("+") + A_lo_quark("-") + A_lo_gluon("+") + A_lo_gluon("-"))
    # print("nlo",A_nlo_quark("+") + A_nlo_quark("-") + A_nlo_gluon("+") + A_nlo_gluon("-") 
    #      + B_nlo_quark("+") + B_nlo_quark("-") + B_nlo_gluon("+") + B_nlo_gluon("-") + T_nlo_quark() + T_nlo_gluon())
    # if mu == 1:
    #     print("A_q_lo_+",A_lo_quark("+"))
    #     print("A_q_lo_-",A_lo_quark("-"))
    #     print("A_g_lo_+",A_lo_gluon("+"))
    #     print("A_g_lo_-",A_lo_gluon("-"))
    if evolution_order == "NLO" and mu != 1:
        print("------")
        print("A_q_+",A_nlo_quark("+"))
        print("A_q_-",A_nlo_quark("-"))
        print("A_g_+",A_nlo_gluon("+"))
        print("A_g_-",A_nlo_gluon("-"))
        print("B_q",B_nlo_quark("+"))
        print("C_q",B_nlo_quark("-"))
        print("B_g",B_nlo_gluon("+"))
        print("C_g",B_nlo_gluon("-"))
        print("------")

    if moment_type == "singlet":
        if particle == "quark":
            # Manually fix the scale to 0.51 @ mu = 2 GeV from 2310.08484
            if moment_label == "A":
                A0 = 1 #0.51/0.5618
            else:
                A0 = 1
            result = A_lo_quark("+") + A_lo_quark("-")
            if evolution_order == "NLO":
                diagonal_terms = A_nlo_quark("+") + A_nlo_quark("-") + B_nlo_quark("+") + B_nlo_quark("-")
                non_diagonal_terms = 0
                non_diagonal_terms = T_nlo_quark()
                # print("T_quark",non_diagonal_terms)
                result += diagonal_terms + non_diagonal_terms
        if particle == "gluon":
            # Manually fix the scale to 0.501 @ mu = 2 GeV from 2310.08484
            if moment_label == "A":
                A0 = 1 # 0.501/0.43807
            else:
                A0 = 1
            result = A_lo_gluon("+") + A_lo_gluon("-")
            if evolution_order == "NLO":
                diagonal_terms = A_nlo_gluon("+") + A_nlo_gluon("-") + B_nlo_gluon("+") + B_nlo_gluon("-")
                non_diagonal_terms = T_nlo_gluon()
                # print("T_gluon",non_diagonal_terms)
                result += diagonal_terms + non_diagonal_terms
        result *= A0
    elif moment_type in ["non_singlet_isovector","non_singlet_isoscalar"]:
        result = moment_in(j,eta,t,moment_label,evolve_type,evolution_order,error_type) * alpha_frac**(ga_qq/beta_0)   

    return result

def evolve_singlet_D(j,eta,t,mu,Nf=3,particle="quark",moment_label="A",evolution_order="LO",error_type="central"):
    check_particle_type(particle)
    check_moment_type_label("singlet",moment_label)
    if particle == "quark":
        # Manually fix the scale to 1.3 @ mu = 2 GeV from 2310.08484
        D0 = 1.3/1.037769
    else :
        # Manually fix the scale from holography (II.9) in 2204.08857
        D0 = 2.57/3.027868

    eta = 1 # Result is eta independent 
    term_1 = evolve_conformal_moment(j,eta,t,mu,Nf,particle,"singlet",moment_label,evolution_order,error_type)
    term_2 = evolve_conformal_moment(j,0,t,mu,Nf,particle,"singlet",moment_label,evolution_order,error_type)
    result = D0 * (term_1-term_2)/eta**2
    return result

def evolve_quark_non_singlet(j,eta,t,mu,Nf=3,moment_type="non_singlet_isovector",moment_label = "A",evolution_order="LO",error_type="central"):
    result = evolve_conformal_moment(j,eta,t,mu,Nf,"quark",moment_type,moment_label,evolution_order,error_type)
    return result

def evolve_quark_singlet(j,eta,t,mu,Nf=3,moment_label = "A",evolution_order="LO",error_type="central"):
    result = evolve_conformal_moment(j,eta,t,mu,Nf,"quark","singlet",moment_label,evolution_order,error_type)
    return result

def evolve_gluon_singlet(j,eta,t,mu,Nf=3,moment_label = "A",evolution_order="LO",error_type="central"):
    result = evolve_conformal_moment(j,eta,t,mu,Nf,"gluon","singlet",moment_label,evolution_order,error_type)
    return result

def evolve_quark_singlet_D(eta,t,mu,Nf=3,moment_label = "A",evolution_order="LO",error_type="central"):
    result = evolve_singlet_D(eta,t,mu,Nf,"quark",moment_label,evolution_order,error_type)
    return result

def evolve_gluon_singlet_D(j,eta,t,mu,Nf=3,moment_label = "A",evolution_order="LO",error_type="central"):
    result = evolve_singlet_D(eta,t,mu,Nf,"gluon",moment_label,evolution_order,error_type)
    return result

def fourier_transform_moment(j,eta,mu,b_vec,Nf=3,particle="quark",moment_type="non_singlet_isovector", moment_label="A",evolution_order="LO", Delta_max = 5,num_points=100, error_type="central"):
    """
    Optimized calculation of Fourier transformed moments using trapezoidal rule.

    Parameters:
    - j (float): Conformal spin
    - eta (float): Skewness parameter
    - mu (float): RGE scale
    - b_vec: (b_x, b_y), the vector for which to compute the result
    - Nf (int, optional): Number of active flavors. Default is 3.
    - particle (str. optional): "quark" or "gluon". Default is quark.
    - moment_type (str. optional): singlet, non_singlet_isovector or non_singlet_isoscalar. Default is non_singlet_isovector.
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
        moment = evolve_conformal_moment(j,eta,t,mu,Nf,particle,moment_type,moment_label,evolution_order,error_type)
        result = moment*np.exp(exponent)
        return result
    
    # Compute the integrand for each pair of (Delta_x, Delta_y) values
    integrand_values = integrand(Delta_x_grid, Delta_y_grid, b_x, b_y)
    # Perform the numerical integration using the trapezoidal rule for efficiency
    integral_result = np.real(trapezoid(trapezoid(integrand_values, Delta_x_vals, axis=0), Delta_y_vals))/((2*np.pi)**2)

    return integral_result
    

def inverse_fourier_transform_moment(j,eta,mu,Delta_vec,Nf=3,particle="quark",moment_type="non_singlet_isovector",moment_label="A",evolution_order="LO", 
                                     b_max = 9 ,num_points=500, Delta_max=10):
    """
    Sanity check for Fourier transform. The result should be the input moment.

    Parameters:
    - eta (float): Skewness parameter
    - mu (float): RGE scale
    - b_vec: (b_x, b_y), the vector for which to compute the result
    - Nf (int, optional): Number of active flavors. Default is 3.
    - moment_type (str. optional): non_singlet_isovector, non_singlet_isoscalar or flavor separated u, d. Default is non_singlet_isovector.
    - Delta_max (float, optional): maximum radius for the integration domain (limits the integration bounds)
    - num_points: number of points for discretizing the domain (adapt as needed)
    - error_type (str. optional): Whether to use central, plus or minus value of input PDF. Default is central.

    Returns:
    - The value of the Fourier transformed moment at (b_vec)
    """
    check_particle_type(particle)
    if moment_type not in ["singlet","non_singlet_isovector","non_singlet_isoscalar","u","d"]:
        raise ValueError(f"Wrong moment_type {moment_type}")

    Delta_x, Delta_y = Delta_vec
    # Limits of integration for Delta_x, Delta_y on a square grid
    x_min, x_max = -b_max, b_max
    y_min, y_max = -b_max, b_max
    # Discretize the grid (vectorized)
    b_x_vals = np.linspace(x_min, x_max, num_points)
    b_y_vals = np.linspace(y_min, y_max, num_points)

    def integrand(b_x,b_y,Delta_x,Delta_y):
        b_vec = (b_x,b_y)
        exponent = 1j * (b_x * Delta_x + b_y * Delta_y)
        moment = fourier_transform_moment(j,eta,mu,b_vec,Nf,particle,moment_type,moment_label,evolution_order,num_points=num_points,Delta_max=Delta_max)
        result = moment*np.exp(exponent)
        return result

    # Compute the integrand for each pair of (Delta_x, Delta_y) values
    integrand_values = np.array(Parallel(n_jobs=-1)(delayed(integrand)(b_x, b_y, Delta_x, Delta_y)
                                                 for b_y in b_y_vals
                                                 for b_x in b_x_vals))
    integrand_values = integrand_values.reshape((num_points, num_points))

    integral_result = trapezoid(trapezoid(integrand_values, b_x_vals, axis=1), b_y_vals,axis=0)

    return integral_result.real

def fourier_transform_transverse_moment(j,eta,mu,b_vec,Nf=3,particle="quark",moment_type="non_singlet_isovector",evolution_order="LO", Delta_max = 5,num_points=100, error_type="central"):
    """
    Optimized calculation of Fourier transformed moments for transversely polarized target using trapezoidal rule. 
    Automatically uses A_n and B_n moments with assumed nucleon mass of M_n = 0.93827 GeV

    Parameters:
    - j (float): Conformal spin
    - eta (float): Skewness parameter
    - mu (float): RGE scale
    - b_vec: (b_x, b_y), the vector for which to compute the result
    - Nf (int, optional): Number of active flavors. Default is 3.
    - particle (str. optional): "quark" or "gluon". Default is quark.
    - moment_type (str. optional): singlet, non_singlet_isovector or non_singlet_isoscalar. Default is non_singlet_isovector.
    - Delta_max (float, optional): maximum radius for the integration domain (limits the integration bounds)
    - num_points: number of points for discretizing the domain (adapt as needed)
    - error_type (str. optional): Whether to use central, plus or minus value of input PDF. Default is central.

    Returns:
    - The value of the Fourier transformed moment at (b_vec)
    """
    check_error_type(error_type)
    check_particle_type(particle)
    check_moment_type_label(moment_type,"A")
    check_moment_type_label(moment_type,"B")
    # Nucleon mass in GeV
    M_n = 0.93827

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
        moment_1 = evolve_conformal_moment(j,eta,t,mu,Nf,particle,moment_type,"A",evolution_order,error_type)
        moment_2 = 1j * Delta_y/(2*M_n) * evolve_conformal_moment(j,eta,t,mu,Nf,particle,moment_type,"B",evolution_order,error_type)
        moment = moment_1 + moment_2
        result = moment*np.exp(exponent)
        return result
    
    # Compute the integrand for each pair of (Delta_x, Delta_y) values
    integrand_values = integrand(Delta_x_grid, Delta_y_grid, b_x, b_y)
    # Perform the numerical integration using the trapezoidal rule for efficiency
    integral_result = np.real(trapezoid(trapezoid(integrand_values, Delta_x_vals, axis=0), Delta_y_vals))/((2*np.pi)**2)

    return integral_result

def fourier_transform_quark_gluon_helicity(eta,mu,b_vec,Nf=3,particle="quark",moment_type="non_singlet_isovector",evolution_order="LO", Delta_max = 10,num_points=100, error_type="central"):
    """
    Quark gluon helicity in impact parameter space

    Parameters:
    - eta (float): Skewness parameter
    - mu (float): RGE scale
    - b_vec: (b_x, b_y), the vector for which to compute the result
    - Nf (int, optional): Number of active flavors. Default is 3.
    - particle (str. optional): "quark" or "gluon"
    - moment_type (str. optional): non_singlet_isovector, non_singlet_isoscalar or flavor separated u, d. Default is non_singlet_isovector.
    - Delta_max (float, optional): maximum radius for the integration domain (limits the integration bounds)
    - num_points: number of points for discretizing the domain (adapt as needed)
    - error_type (str. optional): Whether to use central, plus or minus value of input PDF. Default is central.

    Returns:
    - The value of the quark gluon helicity at (b_vec)
    """
    check_particle_type(particle)
    if moment_type not in ["singlet","non_singlet_isovector","non_singlet_isoscalar","u","d"]:
        raise ValueError(f"Wrong moment_type {moment_type}")
    check_error_type(error_type)
    
    if moment_type != "singlet":
        n = 1
    else:
        n = 2

    if moment_type in ["singlet","non_singlet_isovector","non_singlet_isoscalar"]:
        result = fourier_transform_moment(n,eta,mu,b_vec,Nf,particle,moment_type,moment_label="Atilde",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type=error_type)/2
    elif moment_type in ["u","d"]:
        if moment_type == "u":
            prf = 1
        else:
            prf = -1
        moment_1 = fourier_transform_moment(1,eta,mu,b_vec,Nf,particle,moment_type="non_singlet_isoscalar",moment_label="Atilde",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type=error_type)/2
        moment_2 = fourier_transform_moment(1,eta,mu,b_vec,Nf,particle,moment_type="non_singlet_isovector",moment_label="Atilde",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type=error_type)/2
        result = (moment_1 + prf * moment_2)/2

    return result

def fourier_transform_quark_helicity(eta,mu,b_vec,Nf=3,moment_type="non_singlet_isovector",evolution_order="LO", Delta_max = 10,num_points=100, error_type="central"):
    result = fourier_transform_quark_gluon_helicity(eta,mu,b_vec,Nf,particle="quark",moment_type=moment_type,evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type=error_type)
    return result

def fourier_transform_gluon_helicity(eta,mu,b_vec,Nf=3,evolution_order="LO",Delta_max = 10,num_points=100, error_type="central"):
    result = fourier_transform_quark_gluon_helicity(eta,mu,b_vec,Nf,particle="gluon",moment_type="singlet",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type=error_type)
    return result

def fourier_transform_quark_helicity(eta,mu,b_vec,Nf=3,moment_type="non_singlet_isovector",evolution_order="LO", Delta_max = 10,num_points=100, error_type="central"):
    result = fourier_transform_quark_gluon_helicity(eta,mu,b_vec,Nf,particle="quark",moment_type=moment_type,evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type=error_type)
    return result

def fourier_transform_gluon_helicity(eta,mu,b_vec,Nf=3,evolution_order="LO",Delta_max = 10,num_points=100, error_type="central"):
    result = fourier_transform_quark_gluon_helicity(eta,mu,b_vec,Nf,particle="gluon",moment_type="singlet",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type=error_type)
    return result

def fourier_transform_spin_orbit_correlation(eta,mu,b_vec,Nf=3,evolution_order="LO",particle="quark",moment_type="non_singlet_isovector", Delta_max = 8,num_points=100, error_type="central"):
    """
    Spin-orbit correlation in impact parameter space

    Parameters:
    - eta (float): Skewness parameter
    - mu (float): RGE scale
    - b_vec: (b_x, b_y), the vector for which to compute the result
    - Nf (int, optional): Number of active flavors. Default is 3.
    - particle (str. optional): "quark" or "gluon"
    - moment_type (str. optional): non_singlet_isovector, non_singlet_isoscalar or flavor separated u, d. Default is non_singlet_isovector.
    - Delta_max (float, optional): maximum radius for the integration domain (limits the integration bounds)
    - num_points: number of points for discretizing the domain (adapt as needed)
    - error_type (str. optional): Whether to use central, plus or minus value of input PDF. Default is central.

    Returns:
    - The value of the spin-orbit correlation at (b_vec)
    """
    check_particle_type(particle)
    if moment_type not in ["singlet","non_singlet_isovector","non_singlet_isoscalar","u","d"]:
        raise ValueError(f"Wrong moment_type {moment_type}")
    check_error_type(error_type)

    if moment_type != "singlet":
        n = 1
    else: 
        n = 2

    if moment_type in ["singlet","non_singlet_isovector","non_singlet_isoscalar"]:
            term_1 = fourier_transform_moment(n+1,eta,mu,b_vec,Nf,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type="central")
            term_2 = fourier_transform_moment(n,eta,mu,b_vec,Nf,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type="central")
            moment = (term_1-term_2)/2
            if error_type != "central":      
                term_1_error = .5 * (fourier_transform_moment(n+1,eta,mu,b_vec,Nf,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type=error_type)
                                - term_1)
                term_2_error = .5 * (fourier_transform_moment(n,eta,mu,b_vec,Nf,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type=error_type)
                                - term_2)
                if error_type == "plus":
                    moment += np.sqrt(term_1_error**2+term_2_error**2)
                else:
                    moment -= np.sqrt(term_1_error**2+term_2_error**2)
            result = moment
            return result

    elif moment_type in ["u","d"]:
        if moment_type == "u":
            prf = 1
        else:
            prf = -1
            term_1 = fourier_transform_moment(2,eta,mu,b_vec,Nf,particle="quark",moment_type="non_singlet_isoscalar",moment_label="Atilde",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type="central")
            term_2 = fourier_transform_moment(1,eta,mu,b_vec,Nf,particle="quark",moment_type="non_singlet_isoscalar",moment_label="A",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type="central")
            moment_1 = (term_1-term_2)/2
            if error_type != "central":      
                term_1_error = (fourier_transform_moment(2,eta,mu,b_vec,Nf,particle="quark",moment_type="non_singlet_isoscalar",moment_label="Atilde",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type=error_type)
                                - term_1)
                term_2_error = (fourier_transform_moment(1,eta,mu,b_vec,Nf,particle="quark",moment_type="non_singlet_isoscalar",moment_label="A",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type=error_type)
                                - term_2)
                if error_type == "plus":
                    moment_1 += np.sqrt(term_1_error**2+term_2_error**2)
                else:
                    moment_1 -= np.sqrt(term_1_error**2+term_2_error**2)
            term_1 = fourier_transform_moment(2,eta,mu,b_vec,Nf,particle="quark",moment_type="non_singlet_isovector",moment_label="Atilde",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type="central")
            term_2 = fourier_transform_moment(1,eta,mu,b_vec,Nf,particle="quark",moment_type="non_singlet_isovector",moment_label="A",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type="central")
            moment_2 = (term_1-term_2)/2 
            if error_type != "central":      
                term_1_error = (fourier_transform_moment(2,eta,mu,b_vec,Nf,particle="quark",moment_type="non_singlet_isovector",moment_label="Atilde",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type=error_type)
                                - term_1)
                term_2_error = (fourier_transform_moment(1,eta,mu,b_vec,Nf,particle="quark",moment_type="non_singlet_isovector",moment_label="A",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type=error_type)
                                - term_2)
                if error_type == "plus":
                    moment_2 += np.sqrt(term_1_error**2+term_2_error**2)
                else:
                    moment_2 -= np.sqrt(term_1_error**2+term_2_error**2)
            moment = (moment_1 + prf * moment_2)/2
            result = moment
            return result
        
def fourier_transform_orbital_angular_momentum(eta,mu,b_vec,Nf=3,particle="quark",moment_type="non_singlet_isovector",evolution_order="LO", Delta_max = 7,num_points=100, error_type="central"):
    """
    Orbital angular momentum in impact parameter space

    Parameters:
    - eta (float): Skewness parameter
    - mu (float): RGE scale
    - b_vec: (b_x, b_y), the vector for which to compute the result
    - Nf (int, optional): Number of active flavors. Default is 3.
    - particle (str. optional): "quark" or "gluon"
    - moment_type (str. optional): non_singlet_isovector, non_singlet_isoscalar or flavor separated u, d. Default is non_singlet_isovector.
    - Delta_max (float, optional): maximum radius for the integration domain (limits the integration bounds)
    - num_points: number of points for discretizing the domain (adapt as needed)
    - error_type (str. optional): Whether to use central, plus or minus value of input PDF. Default is central.

    Returns:
    - The value of the orbital angular momentum at (b_vec)
    """
    check_particle_type(particle)
    if moment_type not in ["singlet","non_singlet_isovector","non_singlet_isoscalar","u","d"]:
        raise ValueError(f"Wrong moment_type {moment_type}")
    check_error_type(error_type)

    if moment_type != "singlet":
        n = 1
    else: 
        n = 2

    if moment_type in ["singlet","non_singlet_isovector","non_singlet_isoscalar"]:
            term_1 = fourier_transform_moment(n+1,eta,mu,b_vec,Nf,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type="central")
            term_2 = fourier_transform_moment(n+1,eta,mu,b_vec,Nf,particle=particle,moment_type=moment_type,moment_label="B",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type="central")
            term_3 = fourier_transform_moment(n,eta,mu,b_vec,Nf,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type="central")
            moment = (term_1+term_2-term_3)/2
            if error_type != "central":      
                term_1_error = .5 * (fourier_transform_moment(n+1,eta,mu,b_vec,Nf,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type=error_type)
                                - term_1)
                term_2_error = .5 * (fourier_transform_moment(n+1,eta,mu,b_vec,Nf,particle=particle,moment_type=moment_type,moment_label="B",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type=error_type)
                                - term_2)
                term_3_error = .5 * (fourier_transform_moment(n,eta,mu,b_vec,Nf,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type=error_type)
                                - term_3)
                if error_type == "plus":
                    moment += np.sqrt(term_1_error**2+term_2_error**2+term_3_error**2)
                else:
                    moment -= np.sqrt(term_1_error**2+term_2_error**2+term_3_error**2)
            result = moment
            return result

    elif moment_type in ["u","d"]:
        if moment_type == "u":
            prf = 1
        else:
            prf = -1
            term_1 = fourier_transform_moment(2,eta,mu,b_vec,Nf,particle="quark",moment_type="non_singlet_isoscalar",moment_label="A",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type="central")
            term_2 = fourier_transform_moment(2,eta,mu,b_vec,Nf,particle="quark",moment_type="non_singlet_isoscalar",moment_label="B",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type="central")
            term_3 = fourier_transform_moment(1,eta,mu,b_vec,Nf,particle="quark",moment_type="non_singlet_isoscalar",moment_label="Atilde",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type="central")
            moment_1 = (term_1+term_2-term_3)/2
            if error_type != "central":      
                term_1_error = (fourier_transform_moment(2,eta,mu,b_vec,Nf,particle="quark",moment_type="non_singlet_isoscalar",moment_label="A",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type=error_type)
                                - term_1)
                term_2_error = (fourier_transform_moment(2,eta,mu,b_vec,Nf,particle="quark",moment_type="non_singlet_isoscalar",moment_label="B",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type=error_type)
                                - term_2)
                term_3_error = (fourier_transform_moment(1,eta,mu,b_vec,Nf,particle="quark",moment_type="non_singlet_isoscalar",moment_label="Atilde",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type=error_type)
                                - term_3)
                if error_type == "plus":
                    moment_1 += np.sqrt(term_1_error**2+term_2_error**2+term_3_error**2)
                else:
                    moment_1 -= np.sqrt(term_1_error**2+term_2_error**2+term_3_error**2)
            term_1 = fourier_transform_moment(2,eta,mu,b_vec,Nf,particle="quark",moment_type="non_singlet_isovector",moment_label="A",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type="central")
            term_2 = fourier_transform_moment(2,eta,mu,b_vec,Nf,particle="quark",moment_type="non_singlet_isovector",moment_label="B",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type="central")
            term_3 = fourier_transform_moment(1,eta,mu,b_vec,Nf,particle="quark",moment_type="non_singlet_isovector",moment_label="Atilde",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type="central")
            moment_2 = (term_1+term_2-term_3)/2 
            if error_type != "central":      
                term_1_error = (fourier_transform_moment(2,eta,mu,b_vec,Nf,particle="quark",moment_type="non_singlet_isovector",moment_label="A",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type=error_type)
                                - term_1)
                term_2_error = (fourier_transform_moment(2,eta,mu,b_vec,Nf,particle="quark",moment_type="non_singlet_isovector",moment_label="B",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type=error_type)
                                - term_2)
                term_3_error = (fourier_transform_moment(1,eta,mu,b_vec,Nf,particle="quark",moment_type="non_singlet_isovector",moment_label="Atilde",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type=error_type)
                                - term_3)
                if error_type == "plus":
                    moment_2 += np.sqrt(term_1_error**2+term_2_error**2+term_3_error**2)
                else:
                    moment_2 -= np.sqrt(term_1_error**2+term_2_error**2+term_3_error**2)
            moment = (moment_1 + prf * moment_2)/2
            result = moment
            return result

def fourier_transform_quark_orbital_angular_momentum(eta,mu,b_vec,Nf=3,moment_type="non_singlet_isovector",evolution_order="LO", Delta_max = 7,num_points=100, error_type="central"):
    result = fourier_transform_orbital_angular_momentum(eta,mu,b_vec,Nf,particle="quark",moment_type=moment_type,evolution_order=evolution_order, Delta_max=Delta_max,num_points=num_points, error_type=error_type)
    return result

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

# Define get_j_base which contains real part of integration variable and associated parity
def get_j_base(particle="quark",moment_type="non_singlet_isovector", moment_label="A"):
    check_particle_type(particle)
    check_moment_type_label(moment_type,moment_label)

    if moment_label in ["A","B"]:
        if particle == "quark" and moment_type in ["non_singlet_isovector","non_singlet_isoscalar"]:
            j_base, parity = .95, "none"
        elif particle == "quark" and moment_type == "singlet":
            j_base, parity = 2.5, "odd"
        elif particle == "gluon" and moment_type == "singlet":
            j_base, parity = 2.1, "even"
    if moment_label == "Atilde":
        if particle == "quark" and moment_type in ["non_singlet_isovector","non_singlet_isoscalar"]:
            j_base, parity = .95, "none"
        if particle == "quark" and moment_type == "singlet":
            j_base, parity = 1.6, "even"
        if particle == "gluon" and moment_type == "singlet":
            j_base, parity = 1.6, "odd"
    
    return j_base, parity

def mellin_barnes_gpd(x, eta, t, mu, Nf=3, particle = "quark", moment_type="singlet",moment_label="A",evolution_order="LO", error_type="central",real_imag ="real",j_max = 15, n_jobs=-1):
    """
    Numerically evaluate the Mellin-Barnes integral parallel to the imaginary axis to obtain the corresponding GPD
    
    Parameters:
    - x (float): Parton x
    - eta (float): Skewness.
    - t (float): Mandelstam t
    - mu (float): Resolution scale
    - Nf (int 1<= Nf <=3 ): Number of flavors
    - particle (str): particle species (quark or gluon)
    - moment_type (str): singlet, non_singlet_isovector, non_singlet_isoscalar
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
    check_evolution_order(evolution_order)
    check_moment_type_label(moment_type,moment_label)
    
    if moment_type == "singlet":
        if particle == "quark":
            # Scale fixed by it's value at the input scale:
            # print(uv_PDF(1e-3)+dv_PDF(1e-3)+Sv_PDF(1e-3))
            if moment_label in "A":
                norm = 1.78160932e+03/ 1.61636674e+03
            else:
                norm = 1
        elif particle == "gluon":
            # Scale fixed by it's value at the input scale:
            # print(.1 gluon_PDF(.1))
            if moment_label == "A":
                norm = 0.86852857/9.93131764e-01
            else:
                norm = 1
    elif moment_type == "non_singlet_isovector":
        # Scale fixed by it's value at the input scale:
        # print(uv_minus_dv_PDF(1e-4))
        if moment_label == "A":
            norm = 152.92491544/153.88744991730528
        else:
            norm = 1
    elif moment_type == "non_singlet_isoscalar":
        # To Do
        #print("non_singlet_isoscalar norm is To Do")
        norm = 1

    j_base, parity = get_j_base(particle,moment_type,moment_label)

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
            if moment_type == "singlet":
                mom_val = evolve_quark_singlet(z,eta,t,mu,Nf,moment_label,evolution_order,error_type)
            else:
                mom_val = evolve_quark_non_singlet(z,eta,t,mu,Nf,moment_type,moment_label,evolution_order,error_type)
        else:
            # (-1) from shift in Sommerfeld-Watson transform
            mom_val = (-1) * evolve_gluon_singlet(z,eta,t,mu,Nf, moment_label,evolution_order,error_type)
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
        print(f"Warning: Large error estimate for (x,eta,t)={x,eta,t}: {error}")
    return norm * integral

# Check normalizations:
# vectorized_mellin_barnes_gpd=np.vectorize(mellin_barnes_gpd)
# x_Array = np.array([1e-2, 0.1, 0.3, 0.5, 0.7, 1]) 

# print(uv_minus_dv_PDF(x_Array)) 
# print(vectorized_mellin_barnes_gpd(x_Array,1e-5, -1e-4, 1, moment_type="non_singlet_isovector", moment_label="A", j_max=100, n_jobs =-1 ))

# print(uv_plus_dv_plus_S_PDF(x_Array))
# print(vectorized_mellin_barnes_gpd(x_Array, 1e-5, -1e-4, 1, particle="quark", moment_type="singlet", moment_label="A", j_max=100, n_jobs = -1 ))

# print(x_Array*gluon_PDF(x_Array))
# print(vectorized_mellin_barnes_gpd(x_Array, 1e-3, -1e-4, 1, particle="gluon", moment_type="singlet", moment_label="A", j_max=100, n_jobs = -1 ))
# del x_Array

################################
#### Additional Observables ####
################################

def spin_orbit_corelation(t,mu, particle="quark",moment_type="non_singlet_isovector",evolution_order="LO"):
    """ Returns the spin orbit correlation of moment_type including errors

    Parameters:
    - t (float): Mandelstam t
    - mu (float): The momentum scale of the process
    - moment_type (str. optional): The flavor dependence. Either non_singlet_isovector or non_singlet_isoscalar    
    """

    if moment_type not in ["singlet",
                           "non_singlet_isoscalar",
                           "non_singlet_isovector"]:
        raise ValueError(f"Wrong moment type {moment_type}")
    
    if moment_type != "singlet":
        n = 1
    else:
        n = 2

    term_1 = evolve_conformal_moment(n+1,0,t,mu,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,error_type="central")
    term_2 = evolve_conformal_moment(n,0,t,mu,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,error_type="central")
    result = (term_1 - term_2)/2

    term_1_plus = evolve_conformal_moment(n+1,0,t,mu,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,error_type="plus")
    term_2_plus = evolve_conformal_moment(n,0,t,mu,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,error_type="plus")
    error_plus = np.sqrt((term_1_plus-term_1)**2+(term_2_plus-term_2)**2)/2

    term_1_minus = evolve_conformal_moment(n+1,0,t,mu,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,error_type="minus")
    term_2_minus = evolve_conformal_moment(n,0,t,mu,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,error_type="minus")
    error_minus = np.sqrt((term_1_minus-term_1)**2+(term_2_minus-term_2)**2)/2

    # if np.abs(error_plus-error_minus)<1e-2:
    #     print(f"{moment_type} spin-orbit correlation: {result:.3f}(+- {error_plus:.3f}) at {mu} GeV")
    # else:
    #     print(f"{moment_type} spin-orbit correlation: {result:.3f}(+{error_plus:.3f})(-{error_minus:.3f}) at {mu} GeV")
    return result, error_plus, error_minus

def quark_gluon_spin(t,mu, particle="quark",moment_type="non_singlet_isovector",evolution_order="LO"):
    """ Returns the spin contribution of moment_type including errors

    Parameters:
    - t (float): Mandelstam t
    - mu (float): The momentum scale of the process
    - moment_type (str. optional): The flavor dependence. Either non_singlet_isovector or non_singlet_isoscalar    
    """

    if moment_type not in ["singlet",
                           "non_singlet_isoscalar",
                           "non_singlet_isovector"]:
        raise ValueError(f"Wrong moment type {moment_type}")
    
    if moment_type != "singlet":
        n = 1
    else:
        n = 2

    term_1 = evolve_conformal_moment(n+1,0,t,mu,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,error_type="central")
    term_2 = evolve_conformal_moment(n+1,0,t,mu,particle=particle,moment_type=moment_type,moment_label="B",evolution_order=evolution_order,error_type="central")
    result = (term_1 + term_2)/2

    term_1_plus = evolve_conformal_moment(n+1,0,t,mu,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,error_type="plus")
    term_2_plus = evolve_conformal_moment(n+1,0,t,mu,particle=particle,moment_type=moment_type,moment_label="B",evolution_order=evolution_order,error_type="plus")
    error_plus = np.sqrt((term_1_plus-term_1)**2+(term_2_plus-term_2)**2)/2

    term_1_minus = evolve_conformal_moment(n+1,0,t,mu,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,error_type="minus")
    term_2_minus = evolve_conformal_moment(n+1,0,t,mu,particle=particle,moment_type=moment_type,moment_label="B",evolution_order=evolution_order,error_type="minus")
    error_minus = np.sqrt((term_1_minus-term_1)**2+(term_2_minus-term_2)**2)/2

    return result, error_plus, error_minus

def total_spin(t,mu,particle="quark",moment_type="non_singlet_isovector",evolution_order="LO"):
    """ Returns the total spin of moment_type including errors

    Parameters:
    - t (float): Mandelstam t
    - mu (float): The momentum scale of the process
    - moment_type (str. optional): The flavor dependence. Either non_singlet_isovector or non_singlet_isoscalar    
    """
    if moment_type not in ["singlet",
                           "non_singlet_isoscalar",
                           "non_singlet_isovector"]:
        raise ValueError(f"Wrong moment type {moment_type}")
    
    if moment_type != "singlet":
        n = 1
    else:
        n = 2

    term_1 = evolve_conformal_moment(n+1,0,t,mu,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,error_type="central")
    term_2 = evolve_conformal_moment(n+1,0,t,mu,particle=particle,moment_type=moment_type,moment_label="B",evolution_order=evolution_order,error_type="central")
    result = (term_1 + term_2)/2

    term_1_plus = evolve_conformal_moment(n+1,0,t,mu,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,error_type="plus")
    term_2_plus = evolve_conformal_moment(n+1,0,t,mu,particle=particle,moment_type=moment_type,moment_label="B",evolution_order=evolution_order,error_type="plus")
    error_plus = np.sqrt((term_1_plus-term_1)**2+(term_2_plus-term_2)**2)/2

    term_1_minus = evolve_conformal_moment(n+1,0,t,mu,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,error_type="minus")
    term_2_minus = evolve_conformal_moment(n+1,0,t,mu,particle=particle,moment_type=moment_type,moment_label="B",evolution_order=evolution_order,error_type="minus")
    error_minus = np.sqrt((term_1_minus-term_1)**2+(term_2_minus-term_2)**2)/2
    
    return result, error_plus, error_minus

def orbital_angular_momentum(t,mu, particle="quark",moment_type="non_singlet_isovector",evolution_order="LO"):
    """ Returns the orbital angular momentum of moment_type including errors

    Parameters:
    - t (float): Mandelstam t
    - mu (float): The momentum scale of the process
    - moment_type (str. optional): The flavor dependence. Either non_singlet_isovector or non_singlet_isoscalar    
    """

    if moment_type not in ["singlet",
                           "non_singlet_isoscalar",
                           "non_singlet_isovector"]:
        raise ValueError(f"Wrong moment type {moment_type}")
    
    if moment_type != "singlet":
        n = 1
    else:
        n = 2

    term_1 = evolve_conformal_moment(n+1,0,t,mu,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,error_type="central")
    term_2 = evolve_conformal_moment(n+1,0,t,mu,particle=particle,moment_type=moment_type,moment_label="B",evolution_order=evolution_order,error_type="central")
    term_3 = evolve_conformal_moment(n,0,t,mu,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,error_type="central")
    result = (term_1 + term_2)/2 - term_3/2

    term_1_plus = evolve_conformal_moment(n+1,0,t,mu,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,error_type="plus")
    term_2_plus = evolve_conformal_moment(n+1,0,t,mu,particle=particle,moment_type=moment_type,moment_label="B",evolution_order=evolution_order,error_type="plus")
    term_3_plus = evolve_conformal_moment(n,0,t,mu,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,error_type="plus")
    error_plus = np.sqrt((term_1_plus-term_1)**2+(term_2_plus-term_2)**2+(term_3-term_3_plus)**2)/2

    term_1_minus = evolve_conformal_moment(n+1,0,t,mu,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,error_type="minus")
    term_2_minus = evolve_conformal_moment(n+1,0,t,mu,particle=particle,moment_type=moment_type,moment_label="B",evolution_order=evolution_order,error_type="minus")
    term_3_minus = evolve_conformal_moment(n,0,t,mu,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,error_type="minus")
    error_minus = np.sqrt((term_1_minus-term_1)**2+(term_2_minus-term_2)**2+(term_3-term_3_minus)**2)/2

    return result, error_plus, error_minus

def quark_gluon_helicity(t,mu, particle="quark",moment_type="non_singlet_isovector",evolution_order="LO"):
    """ Prints the quark helicity of moment_type including errors

    Parameters:
    - t (float): Mandelstam t
    - mu (float): The momentum scale of the process
    - moment_type (str. optional): The flavor dependence. Either non_singlet_isovector or non_singlet_isoscalar    
    """
    check_particle_type(particle)
    if moment_type not in ["singlet","non_singlet_isoscalar","non_singlet_isovector"]:
        raise ValueError(f"Wrong moment type {moment_type}")
    if particle == "gluon" and moment_type != "singlet":
        raise ValueError(f"Wrong moment_type {moment_type} for {particle}")
    if moment_type != "singlet":
        n = 1
    else:
        n = 2
    result = evolve_conformal_moment(n,0,t,mu,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,error_type="central")/2

    term_1 = evolve_conformal_moment(n,0,t,mu,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,error_type="plus")/2
    error_plus = abs(result - term_1)

    term_1 = evolve_conformal_moment(n,0,t,mu,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,error_type="minus")/2
    error_minus = abs(result - term_1)
    # if np.abs(error_plus-error_minus)<1e-2:
    #     print(f"{moment_type} quark helicity: {result:.3f}(\\pm {error_plus:.3f}) at {mu} GeV")
    # else:
    #     print(f"{moment_type} quark helicity: {result:.3f}(+{error_plus:.3f})(-{error_minus:.3f}) at {mu} GeV")
    return result, error_plus, error_minus

def quark_helicity(t,mu, moment_type="non_singlet_isovector",evolution_order="LO"):
    result, error_plus, error_minus = quark_gluon_helicity(t,mu,particle="quark",moment_type=moment_type,evolution_order=evolution_order)
    return result, error_plus, error_minus

def gluon_helicity(t,mu,evolution_order="LO"):
    result, error_plus, error_minus = quark_gluon_helicity(t,mu,particle="gluon",moment_type="singlet",evolution_order=evolution_order)
    return result, error_plus, error_minus

################################
####### Plot functions #########
################################


def plot_moment(n,eta,y_label,mu_in=2,t_max=3,Nf=3,particle="quark",moment_type="non_singlet_isovector", moment_label="A",evolution_order="LO", n_t=50):
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
    - moment_type (str): The type of moment (e.g., "non_singlet_isovector").
    - moment_label (str): The label of the moment (e.g., "A").
    - n_t (int, optional): Number of points for t_fine (default is 50).
    - num_columns (int, optional): Number of columns for the grid layout (default is 3).
    """
    check_particle_type(particle)
    check_moment_type_label(moment_type,moment_label)
    # Accessor functions for -t, values, and errors
    def t_values(moment_type, moment_label, pub_id):
        """Return the -t values for a given moment type, label, and publication ID."""
        data, n_to_row_map = load_lattice_moment_data(particle,moment_type, moment_label, pub_id)

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
    def compute_results(j, eta, t_vals, mu, Nf=3, particle="quark", moment_type="non_singlet_isovector", moment_label="A"):
        """Compute central, plus, and minus results for a given evolution function."""
        if moment_type != "D":
            results = Parallel(n_jobs=-1)(
                delayed(lambda t: float(evolve_conformal_moment(j, eta, t, mu, Nf, particle, moment_type, moment_label, evolution_order, "central")))(t)
                for t in t_vals
            )
            results_plus = Parallel(n_jobs=-1)(
                delayed(lambda t: float(evolve_conformal_moment(j, eta, t, mu, Nf, particle, moment_type, moment_label, evolution_order, "plus")))(t)
                for t in t_vals
            )
            results_minus = Parallel(n_jobs=-1)(
                delayed(lambda t: float(evolve_conformal_moment(j, eta, t, mu, Nf, particle, moment_type, moment_label, evolution_order, "minus")))(t)
                for t in t_vals
            )
            return results, results_plus, results_minus
        else:
            results = Parallel(n_jobs=-1)(
                delayed(lambda t: float(evolve_singlet_D(j, eta, t, mu, Nf, particle, moment_label, evolution_order, "central")))(t)
                for t in t_vals
            )
            results_plus = Parallel(n_jobs=-1)(
                delayed(lambda t: float(evolve_singlet_D(j, eta, t, mu, Nf, particle, moment_label, evolution_order, "plus")))(t)
                for t in t_vals
            )
            results_minus = Parallel(n_jobs=-1)(
                delayed(lambda t: float(evolve_singlet_D(j, eta, t, mu, Nf, particle, moment_label, evolution_order, "minus")))(t)
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
        data, n_to_row_map = load_lattice_moment_data(particle,moment_type, moment_label, pub_id)
        if data is None or n not in n_to_row_map:
            continue
        t_vals = t_values(moment_type, moment_label, pub_id)
        Fn0_vals = Fn0_values(n, particle, moment_type, moment_label, pub_id)
        Fn0_errs = Fn0_errors(n, particle, moment_type, moment_label, pub_id)
        ax.errorbar(t_vals, Fn0_vals, yerr=Fn0_errs, fmt='o', color=color, label=f"{pub_id}")

    # Add labels and formatting
    ax.set_xlabel("$-t\,[\mathrm{GeV}^2]$", fontsize=14)
    if particle == "gluon" and n == 2:
        ax.set_ylabel(f"$A_g(t,\mu = {mu_in}\,[\mathrm{{GeV}}])$", fontsize=14)
    elif particle == "quark" and moment_type == "singlet" and n == 2:
        ax.set_ylabel(f"$A_{{u+d+s}}(t,\mu = {mu_in}\,[\mathrm{{GeV}}])$", fontsize=14)
    else:
        ax.set_ylabel(f"{y_label}$(j={n}, \\eta=0, t, \\mu={mu_in}\, \\mathrm{{GeV}})$", fontsize=14)
    ax.legend()
    ax.grid(True, which="both")
    ax.set_xlim([0, t_max])
    
    plt.show()

def plot_moments_on_grid(eta, y_label, t_max=3, Nf=3, particle="quark", moment_type="non_singlet_isovector", moment_label="A",evolution_order="LO", n_t=50, num_columns=3,D_term = False,set_y_lim=False,y_0 = -1, y_1 =1):
    """
    Plots conformal moments vs. available lattice data.

    Parameters:
    - eta (float): Skewness parameter
    - y_label (str.): Label on y-axis
    - t_max (float, optional): Maximum value of -t
    - Nf (float, optional): Number of active flavors
    - particle (str. optional): quark or gluon
    - moment_type (str. optional): non_singlet_isovector, singlet...
    - moment_label (str. optional): A, B, Atilde,...
    - n_t (int. optional): Number of points for plot generation
    - num_columns (int. optional): Number of points for plot generation
    - D_term (bool, optional): Whether to plot the D term separately, which is computetd from the difference between skewless and skewness dependent moment
    - set_y_lim (bool, optional): Whether to manually set the limits on the y_axis
    - y_0 (float, optional): lower limit on y_axis
    - y_1 (float, optional): upper limit on y_axis
    """
    check_particle_type(particle)
    check_moment_type_label(moment_type, moment_label)

    if not D_term:
        data_moment_label = moment_label
    else:
        data_moment_label = "D"

    # Accessor functions for -t, values, and errors
    def t_values(data_moment_type, data_moment_label, pub_id):
        """Return the -t values for a given moment type, label, and publication ID."""
        data, n_to_row_map = load_lattice_moment_data(particle,data_moment_type, data_moment_label, pub_id)

        if data is None and n_to_row_map is None:
            print(f"No data found for {data_moment_type} {data_moment_label} {pub_id}. Skipping.")
            return None 
        
        if data is not None:
            return data[:, 0]
        else:
            print(f"Data is None for {data_moment_type} {data_moment_label} {pub_id}. Skipping.")
        return None  

    def compute_results(j, eta, t_vals, mu, Nf=3, particle="quark", moment_type="non_singlet_isovector", moment_label="A",evolution_order=evolution_order):
        """Compute central, plus, and minus results for a given evolution function."""
        if not D_term:
            results = Parallel(n_jobs=-1)(
                delayed(lambda t: float(evolve_conformal_moment(j, eta, t, mu, Nf, particle, moment_type, moment_label, evolution_order, "central")))(t)
                for t in t_vals
            )
            results_plus = Parallel(n_jobs=-1)(
                delayed(lambda t: float(evolve_conformal_moment(j, eta, t, mu, Nf, particle, moment_type, moment_label, evolution_order, "plus")))(t)
                for t in t_vals
            )
            results_minus = Parallel(n_jobs=-1)(
                delayed(lambda t: float(evolve_conformal_moment(j, eta, t, mu, Nf, particle, moment_type, moment_label, evolution_order, "minus")))(t)
                for t in t_vals
            )
            return results, results_plus, results_minus
        else:
            results = Parallel(n_jobs=-1)(
                delayed(lambda t: float(evolve_singlet_D(j, eta, t, mu, Nf, particle, moment_label, evolution_order, "central")))(t)
                for t in t_vals
            )
            results_plus = Parallel(n_jobs=-1)(
                delayed(lambda t: float(evolve_singlet_D(j, eta, t, mu, Nf, particle, moment_label, evolution_order, "plus")))(t)
                for t in t_vals
            )
            results_minus = Parallel(n_jobs=-1)(
                delayed(lambda t: float(evolve_singlet_D(j, eta, t, mu, Nf, particle, moment_label, evolution_order, "minus")))(t)
                for t in t_vals
            )
            return results, results_plus, results_minus
    if D_term:
        t_fine = np.linspace(-t_max, -1e-3, n_t)
    else:
        t_fine = np.linspace(-t_max, 0, n_t)

    # Initialize publication data
    publication_data = {}
    mu = None
    for pub_id, (color,mu) in PUBLICATION_MAPPING.items():
        data, n_to_row_map = load_lattice_moment_data(particle,moment_type, data_moment_label, pub_id)
        if data is None and n_to_row_map is None:
            continue
        num_n_values = (data.shape[1] - 1) // 2
        publication_data[pub_id] = num_n_values

    if mu is None:
        mu = 2 

    if publication_data:
        max_n_value = max(publication_data.values())
        if moment_type == "singlet":
            max_n_value+=1
    else:
        max_n_value = 4
    # Calculate rows for grid layout
    num_rows = (max_n_value + num_columns - 1) // num_columns

    # Create a figure for the grid of subplots (for displaying in notebook)
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_columns, figsize=(15, num_rows * 5))
    axes = axes.flatten()

    if moment_type == "singlet":
        n_0 = 2
    else:
        n_0 = 1
 
    for n in range(n_0, max_n_value + 1):
        ax = axes[n - 1]  # Select the appropriate axis
        
        # Compute results for the current n
        evolve_moment_central, evolve_moment_plus, evolve_moment_minus = compute_results(n, eta, t_fine, mu, Nf, particle, moment_type, moment_label,evolution_order)

        if publication_data:
            ax.plot(-t_fine, evolve_moment_central, color="blue", linewidth=2, label="This work")
        else:
            ax.plot(-t_fine, evolve_moment_central, color="blue", linewidth=2)
        ax.fill_between(-t_fine, evolve_moment_minus, evolve_moment_plus, color="blue", alpha=0.2)

        # Plot data from publications
        if publication_data:
            for pub_id, (color, mu) in PUBLICATION_MAPPING.items():
                data, n_to_row_map = load_lattice_moment_data(particle,moment_type, data_moment_label, pub_id)
                if data is None or n not in n_to_row_map:
                    continue
                t_vals = t_values(moment_type, data_moment_label, pub_id)
                Fn0_vals = Fn0_values(n, particle, moment_type, data_moment_label, pub_id)
                Fn0_errs = Fn0_errors(n, particle, moment_type, data_moment_label, pub_id)
                ax.errorbar(t_vals, Fn0_vals, yerr=Fn0_errs, fmt='o', color=color, label=f"{pub_id}")
            ax.legend()

        # Add labels and formatting
        ax.set_xlabel("$-t\,[\mathrm{GeV}^2]$", fontsize=14)
        ax.set_ylabel(f"{y_label}$(j={n}, \\eta=0, t, \\mu={mu}\, \\mathrm{{GeV}})$", fontsize=14)
 
        ax.grid(True, which="both")
        ax.set_xlim([0, t_max])
        if set_y_lim:
            ax.set_ylim([y_0,y_1])

        # Save each plot as a separate PDF (including publication data)
        pdf_path = f"{PLOT_PATH}{moment_type}_{particle}_{data_moment_label}_n_{n}.pdf"
        
        # Create a new figure to save the current plot as a PDF
        fig_single, ax_single = plt.subplots(figsize=(7, 5))  # New figure for saving each plot
        
        # Plot the RGE functions
        ax_single.plot(-t_fine, evolve_moment_central, color="blue", linewidth=2)
        ax_single.fill_between(-t_fine, evolve_moment_minus, evolve_moment_plus, color="blue", alpha=0.2)

        # Plot data from publications
        if publication_data:
            for pub_id, (color, mu) in PUBLICATION_MAPPING.items():
                data, n_to_row_map = load_lattice_moment_data(particle,moment_type, data_moment_label, pub_id)
                if data is None or n not in n_to_row_map:
                    continue
                t_vals = t_values(moment_type, data_moment_label, pub_id)
                Fn0_vals = Fn0_values(n, particle, moment_type, data_moment_label, pub_id)
                Fn0_errs = Fn0_errors(n, particle, moment_type, data_moment_label, pub_id)
                ax_single.errorbar(t_vals, Fn0_vals, yerr=Fn0_errs, fmt='o', color=color, label=f"{pub_id}")
            ax_single.legend()
        ax_single.set_xlabel("$-t\,[\mathrm{GeV}^2]$", fontsize=14)
        ax_single.set_ylabel(f"{y_label}$(j={n}, \\eta=0, t, \\mu={mu}\, \\mathrm{{GeV}})$", fontsize=14)
        ax_single.grid(True, which="both")
        ax_single.set_xlim([0, t_max])
        if set_y_lim:
            ax_single.set_ylim([y_0,y_1])
        plt.tight_layout()
        plt.savefig(pdf_path, format="pdf", bbox_inches="tight",dpi=600)
        plt.close(fig_single)

    # Remove any empty axes if max_n_value doesn't fill the grid
    for i in range(max_n_value, len(axes)):
        fig.delaxes(axes[i])

    if moment_type == "singlet":
        fig.delaxes(axes[0])

    # Adjust GridSpec to remove whitespace
    gs = plt.GridSpec(num_rows, num_columns, figure=fig)
    remaining_axes = fig.get_axes()  # Get all remaining axes

    for i, ax in enumerate(remaining_axes):
        ax.set_subplotspec(gs[i])  # Assign axes to new GridSpec slots

    # Show the full grid of subplots in the notebook
    plt.tight_layout()
    plt.show()

    # Close the figure to free up memory
    plt.close(fig)

def plot_moments_D_on_grid(t_max, mu, Nf=3,evolution_order="LO", n_t=50,display="both"):
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
    def compute_results_D(j, eta, t_vals, mu, Nf=3, particle="quark", moment_type="non_singlet_isovector", moment_label="A"):
        """Compute central, plus, and minus results for a given evolution function."""
        results = Parallel(n_jobs=-1)(
            delayed(lambda t: float(evolve_singlet_D(j, eta, t, mu, Nf, particle, moment_label, evolution_order, "central")))(t)
            for t in t_vals
        )
        results_plus = Parallel(n_jobs=-1)(
            delayed(lambda t: float(evolve_singlet_D(j, eta, t, mu, Nf, particle, moment_label, evolution_order, "plus")))(t)
            for t in t_vals
        )
        results_minus = Parallel(n_jobs=-1)(
            delayed(lambda t: float(evolve_singlet_D(j, eta, t, mu, Nf, particle, moment_label, evolution_order, "minus")))(t)
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

def plot_spin_orbit_correlation(particle="quark",evolution_order="LO",n_t = 50):
    """
    Generates plots of lattice data and spin orbit correlation
    
    Parameters:
    - n_t (int, optional): Number of points for t_fine (default is 50).
    """
    def compute_result(t_vals,mu,moment_type):
        parallel_results = Parallel(n_jobs=-1)(
            delayed(lambda t: spin_orbit_corelation(t,mu,particle,moment_type, evolution_order))(t)
            for t in t_vals)
        results, error_plus, error_minus = zip(*parallel_results)
        results = np.array(results)
        error_plus = np.array(error_plus)
        error_minus = np.array(error_minus)
        results_plus = results + error_plus
        results_minus = results - error_minus
        return results, results_plus, results_minus
    
    publications = [("2305.11117","2410.03539"),("0705.4295","0705.4295")]
    shapes = ['o','^']
    moment_types = ["non_singlet_isoscalar", "non_singlet_isovector","u","d"]
    labels = [r"$C_z^{{u+d}}(t)$", r"$C_z^{{u-d}}(t)$", r"$C_z^{{u}}(t)$", r"$C_z^{{d}}(t)$"] 
    colors = ["black","red","green","blue"]
    t_min, t_max = 0,0

    # Store which moment types are available
    moment_data = []
    for j, pub in enumerate(publications):
        for i, moment_type in enumerate(moment_types):
            t_values, val_data, err_data = load_Cz_data(particle,moment_type,pub[0],pub[1])
            if (t_values is None or (isinstance(t_values, np.ndarray) and t_values.size == 0)) and \
            (val_data is None or (isinstance(val_data, np.ndarray) and val_data.size == 0)) and \
            (err_data is None or (isinstance(err_data, np.ndarray) and err_data.size == 0)): 
                #print(f"No match for publications {pub} and {moment_type}")
                continue
            if moment_type not in moment_data:
                moment_data.append(moment_type)
            if np.max(t_values) > t_max:
                t_max = np.max(t_values)
            if np.min(t_values) < t_min:
                t_min = np.min(t_values)
            plt.errorbar(t_values, val_data, yerr=err_data, 
                        fmt=shapes[j], color=colors[i], 
                        markersize=6,capsize=5, capthick=1)
    if t_min == 0:
        t_min= 1e-4
    t_fine = np.linspace(-t_min,-t_max,n_t)

    for i, moment_type in enumerate(moment_data):
        results, results_plus, results_minus = compute_result(t_fine,1,moment_type)
        plt.plot(-t_fine,results,color=colors[i],linewidth=2, label=labels[i])
        plt.fill_between(-t_fine,results_minus,results_plus,color=colors[i],alpha=.2)
    #padding = .05 *  (t_max - t_min)
    padding = 0
    plt.xlim(t_min-padding,t_max+padding)
    plt.xlabel("$-t\,[\mathrm{GeV}^2]$")
    plt.legend(fontsize=10, markerscale=1.5)
    plt.grid(True)
    #plt.yscale('log') # set y axis to log scale
    #plt.xscale('log') # set x axis to log scale
    plt.tight_layout()

    FILE_PATH = PLOT_PATH + "Cz_over_t" +".pdf"
    plt.savefig(FILE_PATH,format="pdf",bbox_inches="tight",dpi=600)

    plt.show()

def plot_fourier_transform_moments(j,eta,mu,plot_title,Nf=3,particle="quark",moment_type="non_singlet_isovector", moment_label="A",evolution_order="LO", b_max = 2,Delta_max = 5,num_points=100,error_type="central"):
    """
    Generates a density plot of the 2D Fourier transfrom of RGE-evolved 
    conformal moments for a given moment type and label.
    
    Parameters:
    - j (float): Conformal spin
    - eta (float): Skewness parameter
    - mu (float): RGE scale
    - plot_title (str.): Title of the plot
    - particle (str. optional): "quark" or "gluon". Default is quark.
    - moment_type (str. optional): singlet, non_singlet_isovector or non_singlet_isoscalar. Default is non_singlet_isovector.
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
    fourier_transform_moment_values_flat = Parallel(n_jobs=-1)(delayed(fourier_transform_moment)(j,eta,mu,b_vec,Nf,particle,moment_type, moment_label, evolution_order,Delta_max,num_points,error_type) for b_vec in b_vecs)

    # Reshape the result back to the grid shape
    fourier_transform_moment_values = np.array(fourier_transform_moment_values_flat).reshape(b_x.shape)

    # Convert Gev^-1 to fm
    hbarc = 0.1975

    # Create the 2D density plot
    plt.figure(figsize=(8, 6))
    # Convert to fm, fm and fm^-2
    plt.pcolormesh(b_x*hbarc, b_y*hbarc, fourier_transform_moment_values/hbarc**2, shading='auto', cmap='viridis',rasterized=True)
    plt.colorbar()
    plt.xlabel(r'$b_x\,[\mathrm{fm}]$', fontsize=14)
    plt.ylabel(r'$b_y\,[\mathrm{fm}]$', fontsize=14)
    plt.title(f"{plot_title}$(j={j}, \\eta=0, t, \\mu={mu}\, \\mathrm{{GeV}})$", fontsize=14)
    plt.show()

def plot_fourier_transform_transverse_moments(j,eta,mu,Nf=3,particle="quark",moment_type="non_singlet_isovector",evolution_order="LO",
                                            b_max = 4.5,Delta_max = 7,num_points=100, n_b=100, interpolation=True, n_int=300,
                                            vmin = 0 , vmax = 1 , write_to_file = False, read_from_file = True):
    """
    Generates a density plot of the 2D Fourier transfrom of RGE-evolved 
    conformal moments for a given moment type and a transversely polarzied target.
    Automatically uses A and B moments.
    
    Parameters:
    - j (float): Conformal spin
    - eta (float): Skewness parameter
    - mu (float): RGE scale
    - particle (str. optional): "quark" or "gluon". Default is quark.
    - moment_type (str. optional): singlet, non_singlet_isovector or non_singlet_isoscalar. Default is non_singlet_isovector.
    - moment_label (str. optiona): Label of conformal moment, e.g. A
    - b_max (float, optional): Maximum b value for the vector b_vec=[b_x,b_y] (default is 2).
    - Delta_max (float, optional): Maximum value for Delta integration (default is 11).
    - num_points (float, optional): Number of intervals to split [-Delta_max, Delta_max] interval (default is 100).
    - n_b (int, optional): Number of points the interval [-b_max, b_max] is split into (default is 50).
    - interpolation (bool, optional): Interpolate data points on finer grid
    - n_int (int, optional): Number of points used for interpolation
    - vmin (float ,optioanl): Sets minimum value of colorbar
    - vmax (float, optional): Sets maximum value of colorbar
    - read_from_file (bool): Whether to load data from file system
    - write_to_file (bool): Whether to write data to file system
    """
    check_particle_type(particle)

    if moment_type not in ["non_singlet_isovector", "non_singlet_isoscalar", "u", "d", "all","singlet"]:
        raise ValueError(f"Wrong moment_type {moment_type}")

    if write_to_file and read_from_file:
        raise ValueError("write_to_file and read_from_file can't simultaneously be True")

    FILE_PATH = PLOT_PATH + "imp_param_transv_pol_moment_j_" + str(j) + "_" + moment_type  +".pdf"

    # Define the grid for b_vec
    b_x = np.linspace(-b_max, b_max, n_b)
    b_y = np.linspace(-b_max, b_max, n_b)
    b_x_grid, b_y_grid = np.meshgrid(b_x, b_y)
    b_vecs = np.array([b_x_grid.ravel(), b_y_grid.ravel()]).T

    # Convert GeV^-1 to fm
    hbarc = 0.1975
    b_x_fm = b_x * hbarc
    b_y_fm = b_y * hbarc

    moment_types = ["non_singlet_isovector", "non_singlet_isoscalar", "u", "d"] if moment_type == "all" else [moment_type]

    # Initialize cache to store Fourier transforms for "non_singlet_isovector" and "non_singlet_isoscalar"
    cache = {}

    # Determine figure layout
    if moment_type == "all":
        fig, axs = plt.subplots(1, len(moment_types), figsize=(len(moment_types) * 4, 4))
    else:
        fig, axs = plt.subplots(1, 1, figsize=(len(moment_types) * 4, 4))
        axs = np.array([[axs[0]], [axs[1]]])  # Make it a 2D array for consistency

    for i, mom_type in enumerate(moment_types):
        READ_WRITE_PATH = IMPACT_PARAMETER_MOMENTS_PATH + "imp_param_transv_pol_moment_j_" + str(j) + "_"  + mom_type 
        row, col = divmod(i, 4)  # Map index to subplot location
        ax = axs[col]

        title_map = {
            "non_singlet_isovector": "u-d",
            "non_singlet_isoscalar": "u+d",
            "u": "u",
            "d": "d"
        }
        title = title_map[mom_type]

        # Compute Fourier transform and cache the results for non_singlet_isovector and non_singlet_isoscalar
        if mom_type in ["non_singlet_isovector", "non_singlet_isoscalar"] or moment_type != "all":
            # Define the grid for b_vec
            b_x = np.linspace(-b_max, b_max, n_b)
            b_y = np.linspace(-b_max, b_max, n_b)
            b_x_grid, b_y_grid = np.meshgrid(b_x, b_y)
            b_vecs = np.array([b_x_grid.ravel(), b_y_grid.ravel()]).T

            b_x_fm = b_x * hbarc
            b_y_fm = b_y * hbarc
            if mom_type not in cache:
                if read_from_file:
                    file_name = generate_filename(eta,0,mu,READ_WRITE_PATH,"central")
                    b_x_fm, b_y_fm, fourier_transform_moment_values_flat = read_ft_from_csv(file_name)
                    n_b = len(fourier_transform_moment_values_flat)
                    b_x = np.linspace(-b_max, b_max, n_b)
                    b_y = np.linspace(-b_max, b_max, n_b)
                    # Exctract shape for reshaping
                    b_x_grid, b_y_grid = np.meshgrid(b_x, b_y)
                else:
                    fourier_transform_moment_values_flat = Parallel(n_jobs=-1)(delayed(fourier_transform_transverse_moment)(
                        j,eta, mu, b_vec, Nf, particle,mom_type, evolution_order, Delta_max, num_points, "central") for b_vec in b_vecs)
                    # Reshape
                    fourier_transform_moment_values_flat = np.array(fourier_transform_moment_values_flat).reshape(b_x_grid.shape)
                    # Convert to fm^-2
                    fourier_transform_moment_values_flat = fourier_transform_moment_values_flat/hbarc**2
                cache[mom_type] = fourier_transform_moment_values_flat

        if mom_type in ["u","d"]:
            if read_from_file:
                file_name = generate_filename(eta,0,mu,READ_WRITE_PATH,"central")
                b_x_fm, b_y_fm, _ = read_ft_from_csv(file_name)
            elif interpolation:
                b_x = np.linspace(-b_max, b_max, n_b)
                b_y = np.linspace(-b_max, b_max, n_b)
                b_x_fm = b_x * hbarc
                b_y_fm = b_y * hbarc

            if mom_type == "u":
                prf = 1
            if mom_type == "d":
                prf = -1 
            fourier_transform_moment_values_flat = (cache["non_singlet_isoscalar"] + prf * cache["non_singlet_isovector"]) / 2

        # Save write out values before interpolation
        if write_to_file:
            b_x_fm_write_out = b_x_fm
            b_y_fm_write_out = b_y_fm
            ft_write_out = fourier_transform_moment_values_flat

        if interpolation:
            ft_interpolation = RectBivariateSpline(b_x_fm, b_y_fm, fourier_transform_moment_values_flat)

            # Call the interpolation on a finer grid
            b_x = np.linspace(-b_max, b_max, n_int)
            b_y = np.linspace(-b_max, b_max, n_int)
            b_x_fm = b_x * hbarc
            b_y_fm = b_y * hbarc

            fourier_transform_moment_values_flat = ft_interpolation(b_x_fm, b_y_fm)

        # Generate 2D density plot
        im = ax.pcolormesh(b_x_fm, b_y_fm, fourier_transform_moment_values_flat, 
                            shading='auto', cmap='jet',vmin=vmin, vmax=vmax,rasterized=True)
        ax.set_xlabel(r'$b_x\,[\mathrm{fm}]$', fontsize=14)
        if i == 0:
            ax.set_ylabel(r'$b_y\,[\mathrm{fm}]$', fontsize=14)
        ax.set_title(rf"$\rho_{{{j},\perp}}^{{{title}}}$", fontsize=14)
        ax.set_xlim([-b_max * hbarc, b_max * hbarc])
        ax.set_ylim([-b_max * hbarc, b_max * hbarc])

        # Add colorbar only once per row
        if col == len(moment_types)-1:
            cbar_ax = fig.add_axes([ax.get_position().x1, ax.get_position().y0, 0.01, ax.get_position().height])
            fig.colorbar(im, cax=cbar_ax)

        if write_to_file:
            file_name = generate_filename(eta,0,mu,READ_WRITE_PATH,"central")
            save_ft_to_csv(b_x_fm_write_out,b_y_fm_write_out,ft_write_out,file_name)

        # Remove ticks and labels
        if i != 0 and moment_type == "all":
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_ylabel(None)

    plt.subplots_adjust(wspace=0, hspace=0)

    # File export
    plt.savefig(FILE_PATH,format="pdf",bbox_inches="tight",dpi=600)

    # Adjust layout and show the plot
    plt.show()
    plt.close()

def plot_fourier_transform_quark_spin_orbit_correlation(eta, mu, Nf=3, moment_type="non_singlet_isovector",evolution_order="LO", 
                                          b_max=4.5, Delta_max=10, num_points=100, n_b=50, interpolation = True,n_int=300,
                                          vmin = -1.8 , vmax = 1, ymin = -2, ymax = .3,
                                          plot_option="both",write_to_file = False, read_from_file = True):
    """
    Generates a density plot of the 2D Fourier transform of the quark spin-orbit correlation
    It also includes a 1D slice at b_y = 0.

    Parameters:
    - eta (float): Skewness parameter
    - mu (float): RGE scale
    - Nf (int, optional): Number of flavors (default is 3).
    - moment_type (str, optional): "non_singlet_isovector", "non_singlet_isoscalar", "u", "d" or "all" (default is "non_singlet_isovector").
    - b_max (float, optional): Maximum b value for the vector b_vec=[b_x,b_y] (default is 6 GeV = 1.185 fm ).
    - Delta_max (float, optional): Maximum value for Delta integration (default is 10).
    - num_points (int, optional): Number of intervals to split [-Delta_max, Delta_max] interval for trapezoid (default is 100).
    - n_b (int, optional): Number of points the interval [-b_max, b_max] is split into (default is 50).
    - interpolation (bool, optional): Interpolate data points on finer grid
    - n_int (int, optional): Number of points used for interpolation
    - vmin (float ,optioanl): Sets minimum value of colorbar
    - vmax (float, optional): Sets maximum value of colorbar
    - ymin (float, optional): Sets minimum value for lower plot of b_y = 0 slice
    - ymax (float, optional): Sets maximum value for lower plot of b_y = 0 slice
    - plot_option (str, optional): "upper", "lower", or "both" to control which plots are shown (default is "both").
    - read_from_file (bool): Whether to load data from file system
    - write_to_file (bool): Whether to write data to file system
    """
    
    particle = "quark"
    if moment_type not in ["non_singlet_isovector", "non_singlet_isoscalar", "u", "d", "all"]:
        raise ValueError(f"Wrong moment_type {moment_type}")

    if write_to_file and read_from_file:
        raise ValueError("write_to_file and read_from_file can't simultaneously be True")


    if moment_type not in ["non_singlet_isovector", "non_singlet_isoscalar", "u", "d", "all"]:
        raise ValueError(f"Wrong moment_type {moment_type}")

    FILE_PATH = PLOT_PATH + "imp_param_spin_orbit_" + moment_type  +".pdf"

    # Define the grid for b_vec
    b_x = np.linspace(-b_max, b_max, n_b)
    b_y = np.linspace(-b_max, b_max, n_b)
    b_x_grid, b_y_grid = np.meshgrid(b_x, b_y)
    b_vecs = np.array([b_x_grid.ravel(), b_y_grid.ravel()]).T

    # Convert GeV^-1 to fm
    hbarc = 0.1975
    b_x_fm = b_x * hbarc
    b_y_fm = b_y * hbarc

    moment_types = ["non_singlet_isovector", "non_singlet_isoscalar", "u", "d"] if moment_type == "all" else [moment_type]

    # Initialize cache to store Fourier transforms for "non_singlet_isovector" and "non_singlet_isoscalar"
    cache = {}

    # Determine figure layout
    if moment_type == "all":
        fig, axs = plt.subplots(2, len(moment_types), figsize=(len(moment_types)*3, 4.5), gridspec_kw={'height_ratios': [2, 1], 'hspace': 0})
    else:
        fig, axs = plt.subplots(2, 1, figsize=(3, 4.5), gridspec_kw={'height_ratios': [2, 1], 'hspace': 0})
        axs = np.array([[axs[0]], [axs[1]]])  # Make it a 2D array for consistency

    for i, mom_type in enumerate(moment_types):
        READ_WRITE_PATH = IMPACT_PARAMETER_MOMENTS_PATH + "imp_param_spin_orbit_" + mom_type 
        row, col = divmod(i, 4)  # Map index to subplot location
        ax = axs[0, col]
        ax_lower = axs[1, col]

        title_map = {
            "non_singlet_isovector": "u-d",
            "non_singlet_isoscalar": "u+d",
            "u": "u",
            "d": "d"
        }
        title = title_map[mom_type]

        # Compute Fourier transform and cache the results for non_singlet_isovector and non_singlet_isoscalar
        if mom_type in ["non_singlet_isovector", "non_singlet_isoscalar"] or moment_type != "all":
            # Define the grid for b_vec
            b_x = np.linspace(-b_max, b_max, n_b)
            b_y = np.linspace(-b_max, b_max, n_b)
            b_x_grid, b_y_grid = np.meshgrid(b_x, b_y)
            b_vecs = np.array([b_x_grid.ravel(), b_y_grid.ravel()]).T

            b_x_fm = b_x * hbarc
            b_y_fm = b_y * hbarc
            if mom_type not in cache:
                if read_from_file:
                    file_name = generate_filename(eta,0,mu,READ_WRITE_PATH,"central")
                    b_x_fm, b_y_fm, fourier_transform_moment_values_flat = read_ft_from_csv(file_name)
                    n_b = len(fourier_transform_moment_values_flat)
                    b_x = np.linspace(-b_max, b_max, n_b)
                    b_y = np.linspace(-b_max, b_max, n_b)
                    # Exctract shape for reshaping
                    b_x_grid, b_y_grid = np.meshgrid(b_x, b_y)
                else:
                    fourier_transform_moment_values_flat = Parallel(n_jobs=-1)(delayed(fourier_transform_spin_orbit_correlation)(
                        eta, mu, b_vec, Nf, particle,mom_type, evolution_order, Delta_max, num_points, "central") for b_vec in b_vecs)
                    # Reshape
                    fourier_transform_moment_values_flat = np.array(fourier_transform_moment_values_flat).reshape(b_x_grid.shape)
                    # Convert to fm^-2
                    fourier_transform_moment_values_flat = fourier_transform_moment_values_flat/hbarc**2
                cache[mom_type] = fourier_transform_moment_values_flat

            # Generate error bars for lower plot
            if plot_option in ["lower", "both"]:
                if read_from_file:
                    file_name = generate_filename(eta,0,mu,READ_WRITE_PATH,"plus")
                    _, _, fourier_transform_moment_values_flat_plus = read_ft_from_csv(file_name)
                    file_name = generate_filename(eta,0,mu,READ_WRITE_PATH,"minus")
                    _, _, fourier_transform_moment_values_flat_minus = read_ft_from_csv(file_name)
                else:
                    fourier_transform_moment_values_flat_plus = Parallel(n_jobs=-1)(delayed(fourier_transform_spin_orbit_correlation)(
                        eta, mu, b_vec, Nf, particle,mom_type, evolution_order, Delta_max, num_points, "plus") for b_vec in b_vecs)
                    fourier_transform_moment_values_flat_minus = Parallel(n_jobs=-1)(delayed(fourier_transform_spin_orbit_correlation)(
                        eta, mu, b_vec, Nf, particle, mom_type, evolution_order, Delta_max, num_points, "minus") for b_vec in b_vecs)
                    # Reshape
                    fourier_transform_moment_values_flat_plus = np.array(fourier_transform_moment_values_flat_plus).reshape(b_x_grid.shape)
                    # Convert to fm^-2
                    fourier_transform_moment_values_flat_plus = fourier_transform_moment_values_flat_plus/hbarc**2
                    # Reshape
                    fourier_transform_moment_values_flat_minus = np.array(fourier_transform_moment_values_flat_minus).reshape(b_x_grid.shape)
                    # Convert to fm^-2
                    fourier_transform_moment_values_flat_minus = fourier_transform_moment_values_flat_minus/hbarc**2

                # Cache
                cache[f"{mom_type}_plus"] = fourier_transform_moment_values_flat_plus
                cache[f"{mom_type}_minus"] = fourier_transform_moment_values_flat_minus

        if mom_type in ["u","d"]:
            if read_from_file:
                file_name = generate_filename(eta,0,mu,READ_WRITE_PATH,"central")
                b_x_fm, b_y_fm, _ = read_ft_from_csv(file_name)
            elif interpolation:
                b_x = np.linspace(-b_max, b_max, n_b)
                b_y = np.linspace(-b_max, b_max, n_b)
                b_x_fm = b_x * hbarc
                b_y_fm = b_y * hbarc

            error_plus = np.sqrt((cache["non_singlet_isoscalar_plus"]-cache["non_singlet_isoscalar"])**2
                                    + (cache["non_singlet_isovector_plus"]-cache["non_singlet_isovector"])**2)/2
            error_minus = np.sqrt((cache["non_singlet_isoscalar_minus"]-cache["non_singlet_isoscalar"])**2
                                    + (cache["non_singlet_isovector_minus"]-cache["non_singlet_isovector"])**2)/2
            if mom_type == "u":
                prf = 1
            if mom_type == "d":
                prf = -1 
            fourier_transform_moment_values_flat = (cache["non_singlet_isoscalar"] + prf * cache["non_singlet_isovector"]) / 2
            fourier_transform_moment_values_flat_plus = fourier_transform_moment_values_flat + error_plus
            fourier_transform_moment_values_flat_minus = fourier_transform_moment_values_flat - error_minus

        # Save write out values before interpolation
        if write_to_file:
            b_x_fm_write_out = b_x_fm
            b_y_fm_write_out = b_y_fm
            ft_write_out = fourier_transform_moment_values_flat
            if plot_option in ["lower","both"]:
                ft_write_out_plus = fourier_transform_moment_values_flat_plus
                ft_write_out_minus = fourier_transform_moment_values_flat_minus

        if interpolation:

            ft_interpolation = RectBivariateSpline(b_x_fm, b_y_fm, fourier_transform_moment_values_flat)
            ft_interpolation_plus = RectBivariateSpline(b_x_fm, b_y_fm, fourier_transform_moment_values_flat_plus)
            ft_interpolation_minus = RectBivariateSpline(b_x_fm, b_y_fm, fourier_transform_moment_values_flat_minus)

            # Call the interpolation on a finer grid
            b_x = np.linspace(-b_max, b_max, n_int)
            b_y = np.linspace(-b_max, b_max, n_int)
            b_x_fm = b_x * hbarc
            b_y_fm = b_y * hbarc

            fourier_transform_moment_values_flat = ft_interpolation(b_x_fm, b_y_fm)
            if plot_option in ["lower","both"]:
                fourier_transform_moment_values_flat_plus = ft_interpolation_plus(b_x_fm, b_y_fm)
                fourier_transform_moment_values_flat_minus = ft_interpolation_minus(b_x_fm, b_y_fm)

        # Upper plot: 2D density plot
        if plot_option in ["upper", "both"]:
            im = ax.pcolormesh(b_x_fm, b_y_fm, fourier_transform_moment_values_flat, 
                               shading='auto', cmap='jet', vmin=vmin, vmax=vmax,rasterized=True)
            ax.set_xlabel(r'$b_x\,[\mathrm{fm}]$', fontsize=14)
            if i == 0:
                ax.set_ylabel(r'$b_y\,[\mathrm{fm}]$', fontsize=14)
            ax.set_title(rf"$C_z^{{{title}}}$", fontsize=14)
            ax.set_xlim([-b_max * hbarc, b_max * hbarc])
            ax.set_ylim([-b_max * hbarc, b_max * hbarc])

            # Add colorbar only once per row
            if col == len(moment_types)-1:
                cbar_ax = fig.add_axes([ax.get_position().x1, ax.get_position().y0, 0.01, ax.get_position().height])
                fig.colorbar(im, cax=cbar_ax)

        # Lower plot: 1D slice at b_y = 0
        if plot_option in ["lower", "both"]:
            idx_by_0 = np.argmin(np.abs(b_y))
            ax_lower.plot(b_x_fm, fourier_transform_moment_values_flat[idx_by_0, :], color='black')
            ax_lower.fill_between(b_x_fm, 
                                  fourier_transform_moment_values_flat_minus[idx_by_0, :], 
                                  fourier_transform_moment_values_flat_plus[idx_by_0, :], 
                                  color='gray', alpha=0.5)
            ax_lower.set_xlabel(r'$b_x\,[\mathrm{fm}]$', fontsize=14)
            ax_lower.set_xlim([-b_max * hbarc, b_max * hbarc])
            ax_lower.set_ylim([ymin, ymax])

            if moment_type == "all" and i == 0:
                ax_lower.set_ylabel(rf'$C_z^{{q}}$', fontsize=14)
            elif i == 0:
                ax_lower.set_ylabel(rf'$C_z^{{{title}}}$', fontsize=14)
        # Remove ticks and labels
        if i != 0 and moment_type == "all":
            ax.set_yticks([])
            ax.set_yticklabels([])
            ax.set_ylabel(None)

            ax_lower.set_yticks([])
            ax_lower.set_yticklabels([])
        if write_to_file:
            file_name = generate_filename(eta,0,mu,READ_WRITE_PATH,"central")
            save_ft_to_csv(b_x_fm_write_out,b_y_fm_write_out,ft_write_out,file_name)
            if plot_option in ["lower", "both"]:
                file_name = generate_filename(eta,0,mu,READ_WRITE_PATH,"plus")
                save_ft_to_csv(b_x_fm_write_out,b_y_fm_write_out,ft_write_out_plus,file_name)
                file_name = generate_filename(eta,0,mu,READ_WRITE_PATH,"minus")
                save_ft_to_csv(b_x_fm_write_out,b_y_fm_write_out,ft_write_out_minus,file_name)
    plt.subplots_adjust(wspace=0, hspace=0)

    # File export
    plt.savefig(FILE_PATH,format="pdf",bbox_inches="tight",dpi=600)

    # Adjust layout and show the plot
    plt.show()
    plt.close()

def plot_fourier_transform_quark_helicity(eta, mu, Nf=3, moment_type="non_singlet_isovector",evolution_order="LO", 
                                          b_max=4.5, Delta_max=8, num_points=100, n_b=50, interpolation = True,n_int=300,
                                          vmin = -1.1 , vmax = 2.5, ymin = -0.5, ymax = 2.5,
                                          plot_option="both", read_from_file=True, write_to_file = False):
    """
    Generates a density plot of the 2D Fourier transform of RGE-evolved conformal moments.
    It also includes a 1D slice at b_y = 0.

    Parameters:
    - eta (float): Skewness parameter
    - mu (float): RGE scale
    - Nf (int, optional): Number of flavors (default is 3).
    - moment_type (str, optional): "non_singlet_isovector", "non_singlet_isoscalar", "u", "d" or "all" (default is "non_singlet_isovector").
    - b_max (float, optional): Maximum b value for the vector b_vec=[b_x,b_y] (default is 6 GeV = 1.185 fm ).
    - Delta_max (float, optional): Maximum value for Delta integration (default is 8).
    - num_points (int, optional): Number of intervals to split [-Delta_max, Delta_max] interval for trapezoid (default is 100).
    - n_b (int, optional): Number of points the interval [-b_max, b_max] is split into (default is 50).
    - interpolation (bool, optional): Interpolate data points on finer grid
    - n_int (int, optional): Number of points used for interpolation
    - vmin (float ,optioanl): Sets minimum value of colorbar
    - vmax (float, optional): Sets maximum value of colorbar
    - ymin (float, optional): Sets minimum value for lower plot of b_y = 0 slice
    - ymax (float, optional): Sets maximum value for lower plot of b_y = 0 slice
    - plot_option (str, optional): "upper", "lower", or "both" to control which plots are shown (default is "both").
    - read_from_file (bool): Whether to load data from file system
    - write_to_file (bool): Whether to write data to file system
    """   
    if write_to_file and read_from_file:
        raise ValueError("write_to_file and read_from_file can't simultaneously be True")


    if moment_type not in ["non_singlet_isovector", "non_singlet_isoscalar", "u", "d", "all"]:
        raise ValueError(f"Wrong moment_type {moment_type}")

    FILE_PATH = PLOT_PATH + "imp_param_helicity_" + moment_type +".pdf"
    

    # Define the grid for b_vec
    b_x = np.linspace(-b_max, b_max, n_b)
    b_y = np.linspace(-b_max, b_max, n_b)
    b_x_grid, b_y_grid = np.meshgrid(b_x, b_y)
    b_vecs = np.array([b_x_grid.ravel(), b_y_grid.ravel()]).T

    # Convert GeV^-1 to fm
    hbarc = 0.1975
    b_x_fm = b_x * hbarc
    b_y_fm = b_y * hbarc

    moment_types = ["non_singlet_isovector", "non_singlet_isoscalar", "u", "d"] if moment_type == "all" else [moment_type]

    # Initialize cache to store Fourier transforms for "non_singlet_isovector" and "non_singlet_isoscalar"
    cache = {}

    # Determine figure layout
    if moment_type == "all":
        fig, axs = plt.subplots(2, len(moment_types), figsize=(len(moment_types)*3, 4.5), gridspec_kw={'height_ratios': [2, 1], 'hspace': 0})
    else:
        fig, axs = plt.subplots(2, 1, figsize=(3, 4.5), gridspec_kw={'height_ratios': [2, 1], 'hspace': 0})
        axs = np.array([[axs[0]], [axs[1]]])  # Make it a 2D array for consistency

    for i, mom_type in enumerate(moment_types):
        READ_WRITE_PATH = IMPACT_PARAMETER_MOMENTS_PATH + "imp_param_helicity_" + mom_type
        row, col = divmod(i, 4)  # Map index to subplot location
        ax = axs[0, col]
        ax_lower = axs[1, col]

        title_map = {
            "non_singlet_isovector": "u-d",
            "non_singlet_isoscalar": "u+d",
            "u": "u",
            "d": "d"
        }
        title = title_map[mom_type]

        # Compute Fourier transform and cache the results for non_singlet_isovector and non_singlet_isoscalar
        if mom_type in ["non_singlet_isovector", "non_singlet_isoscalar"] or moment_type != "all":
            # Define the grid for b_vec
            b_x = np.linspace(-b_max, b_max, n_b)
            b_y = np.linspace(-b_max, b_max, n_b)
            b_x_grid, b_y_grid = np.meshgrid(b_x, b_y)
            b_vecs = np.array([b_x_grid.ravel(), b_y_grid.ravel()]).T

            b_x_fm = b_x * hbarc
            b_y_fm = b_y * hbarc
            if mom_type not in cache:
                if read_from_file:
                    file_name = generate_filename(eta,0,mu,READ_WRITE_PATH,"central")
                    b_x_fm, b_y_fm, fourier_transform_moment_values_flat = read_ft_from_csv(file_name)
                    n_b = len(fourier_transform_moment_values_flat)
                    b_x = np.linspace(-b_max, b_max, n_b)
                    b_y = np.linspace(-b_max, b_max, n_b)
                    # Exctract shape for reshaping
                    b_x_grid, b_y_grid = np.meshgrid(b_x, b_y)
                else:
                    fourier_transform_moment_values_flat = Parallel(n_jobs=-1)(delayed(fourier_transform_quark_helicity)(
                        eta, mu, b_vec, Nf, mom_type, evolution_order, Delta_max, num_points, "central") for b_vec in b_vecs)
                    # Reshape
                    fourier_transform_moment_values_flat = np.array(fourier_transform_moment_values_flat).reshape(b_x_grid.shape)
                    # Convert to fm^-2
                    fourier_transform_moment_values_flat = fourier_transform_moment_values_flat/hbarc**2
                # Cache
                cache[mom_type] = fourier_transform_moment_values_flat

                # Generate error bars for lower plot
                if plot_option in ["lower", "both"]:
                    if read_from_file:
                        file_name = generate_filename(eta,0,mu,READ_WRITE_PATH,"plus")
                        _, _, fourier_transform_moment_values_flat_plus = read_ft_from_csv(file_name)
                        file_name = generate_filename(eta,0,mu,READ_WRITE_PATH,"minus")
                        _, _, fourier_transform_moment_values_flat_minus = read_ft_from_csv(file_name)
                    else:
                        fourier_transform_moment_values_flat_plus = Parallel(n_jobs=-1)(delayed(fourier_transform_quark_helicity)(
                            eta, mu, b_vec, Nf, mom_type, evolution_order, Delta_max, num_points, "plus") for b_vec in b_vecs)
                        fourier_transform_moment_values_flat_minus = Parallel(n_jobs=-1)(delayed(fourier_transform_quark_helicity)(
                            eta, mu, b_vec, Nf, mom_type, evolution_order, Delta_max, num_points, "minus") for b_vec in b_vecs)
                        # Reshape
                        fourier_transform_moment_values_flat_plus = np.array(fourier_transform_moment_values_flat_plus).reshape(b_x_grid.shape)
                        # Convert to fm^-2
                        fourier_transform_moment_values_flat_plus = fourier_transform_moment_values_flat_plus/hbarc**2
                        # Reshape
                        fourier_transform_moment_values_flat_minus = np.array(fourier_transform_moment_values_flat_minus).reshape(b_x_grid.shape)
                        # Convert to fm^-2
                        fourier_transform_moment_values_flat_minus = fourier_transform_moment_values_flat_minus/hbarc**2
                    # Cache
                    cache[f"{mom_type}_plus"] = fourier_transform_moment_values_flat_plus
                    cache[f"{mom_type}_minus"] = fourier_transform_moment_values_flat_minus

        # Use cached values for u and d
        if mom_type in ["u","d"]:
            if read_from_file:
                file_name = generate_filename(eta,0,mu,READ_WRITE_PATH,"central")
                b_x_fm, b_y_fm, _ = read_ft_from_csv(file_name)
            elif interpolation:
                b_x = np.linspace(-b_max, b_max, n_b)
                b_y = np.linspace(-b_max, b_max, n_b)
                b_x_fm = b_x * hbarc
                b_y_fm = b_y * hbarc

            error_plus = np.sqrt((cache["non_singlet_isoscalar_plus"]-cache["non_singlet_isoscalar"])**2
                                    + (cache["non_singlet_isovector_plus"]-cache["non_singlet_isovector"])**2)/2
            error_minus = np.sqrt((cache["non_singlet_isoscalar_minus"]-cache["non_singlet_isoscalar"])**2
                                    + (cache["non_singlet_isovector_minus"]-cache["non_singlet_isovector"])**2)/2
            if mom_type == "u":
                prf = 1
            if mom_type == "d":
                prf = -1 
            fourier_transform_moment_values_flat = (cache["non_singlet_isoscalar"] + prf * cache["non_singlet_isovector"]) / 2
            fourier_transform_moment_values_flat_plus = fourier_transform_moment_values_flat + error_plus
            fourier_transform_moment_values_flat_minus = fourier_transform_moment_values_flat - error_minus

        # Save write out values before interpolation
        if write_to_file:
            b_x_fm_write_out = b_x_fm
            b_y_fm_write_out = b_y_fm
            ft_write_out = fourier_transform_moment_values_flat
            if plot_option in ["lower","both"]:
                ft_write_out_plus = fourier_transform_moment_values_flat_plus
                ft_write_out_minus = fourier_transform_moment_values_flat_minus

        if interpolation:
            ft_interpolation = RectBivariateSpline(b_x_fm, b_y_fm, fourier_transform_moment_values_flat)
            ft_interpolation_plus = RectBivariateSpline(b_x_fm, b_y_fm, fourier_transform_moment_values_flat_plus)
            ft_interpolation_minus = RectBivariateSpline(b_x_fm, b_y_fm, fourier_transform_moment_values_flat_minus)

            # Call the interpolation on a finer grid
            b_x = np.linspace(-b_max, b_max, n_int)
            b_y = np.linspace(-b_max, b_max, n_int)
            b_x_fm = b_x * hbarc
            b_y_fm = b_y * hbarc

            fourier_transform_moment_values_flat = ft_interpolation(b_x_fm, b_y_fm)
            fourier_transform_moment_values_flat_plus = ft_interpolation_plus(b_x_fm, b_y_fm)
            fourier_transform_moment_values_flat_minus = ft_interpolation_minus(b_x_fm, b_y_fm)

        # Upper plot: 2D density plot
        if plot_option in ["upper", "both"]:
            im = ax.pcolormesh(b_x_fm, b_y_fm, fourier_transform_moment_values_flat, 
                               shading='auto', cmap='jet', vmin=vmin, vmax=vmax,rasterized=True)
            ax.set_xlabel(r'$b_x\,[\mathrm{fm}]$', fontsize=14)
            if i == 0:
                ax.set_ylabel(r'$b_y\,[\mathrm{fm}]$', fontsize=14)
            ax.set_title(rf"$S_z^{{{title}}}$", fontsize=14)
            ax.set_xlim([-b_max * hbarc, b_max * hbarc])
            ax.set_ylim([-b_max * hbarc, b_max * hbarc])

            # Add colorbar only once per row
            if col == len(moment_types)-1:
                cbar_ax = fig.add_axes([ax.get_position().x1, ax.get_position().y0, 0.01, ax.get_position().height])
                fig.colorbar(im, cax=cbar_ax)

        # Lower plot: 1D slice at b_y = 0
        if plot_option in ["lower", "both"]:
            idx_by_0 = np.argmin(np.abs(b_y))
            ax_lower.plot(b_x_fm, fourier_transform_moment_values_flat[idx_by_0, :], color='black')
            ax_lower.fill_between(b_x_fm, 
                                  fourier_transform_moment_values_flat_minus[idx_by_0, :], 
                                  fourier_transform_moment_values_flat_plus[idx_by_0, :], 
                                  color='gray', alpha=0.5)
            ax_lower.set_xlabel(r'$b_x\,[\mathrm{fm}]$', fontsize=14)
            ax_lower.set_xlim([-b_max * hbarc, b_max * hbarc])
            ax_lower.set_ylim([ymin, ymax])

            if moment_type == "all" and i == 0:
                ax_lower.set_ylabel(rf'$S_z^{{q}}$', fontsize=14)
            elif i == 0:
                ax_lower.set_ylabel(rf'$S_z^{{{title}}}$', fontsize=14)
            # Remove ticks and labels
            if i != 0 and moment_type == "all":
                ax.set_yticks([])
                ax.set_yticklabels([])
                ax.set_ylabel(None)

                ax_lower.set_yticks([])
                ax_lower.set_yticklabels([])
        if write_to_file:
            file_name = generate_filename(eta,0,mu,READ_WRITE_PATH,"central")
            save_ft_to_csv(b_x_fm_write_out,b_y_fm_write_out,ft_write_out,file_name)
            if plot_option in ["lower", "both"]:
                file_name = generate_filename(eta,0,mu,READ_WRITE_PATH,"plus")
                save_ft_to_csv(b_x_fm_write_out,b_y_fm_write_out,ft_write_out_plus,file_name)
                file_name = generate_filename(eta,0,mu,READ_WRITE_PATH,"minus")
                save_ft_to_csv(b_x_fm_write_out,b_y_fm_write_out,ft_write_out_minus,file_name)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(FILE_PATH,format="pdf",bbox_inches="tight",dpi=600)

    # Adjust layout and show the plot
    plt.show()
    plt.close()

def plot_fourier_transform_singlet_helicity(eta, mu, Nf=3, particle = "gluon",evolution_order="LO",
                                          b_max=4.5, Delta_max=8, num_points=100, n_b=50, interpolation = True,n_int=300,
                                          vmin = -2.05 , vmax = 3.08, ymin= -2.05, ymax = 3.08,
                                          plot_option="both", read_from_file=True, write_to_file = False):
    """
    Generates a density plot of the 2D Fourier transform of RGE-evolved conformal moments.
    It also includes a 1D slice at b_y = 0.

    Parameters:
    - eta (float): Skewness parameter
    - mu (float): RGE scale
    - Nf (int, optional): Number of flavors (default is 3).
    - particle (str. optional): quark or gluon
    - moment_type (str, optional): "non_singlet_isovector", "non_singlet_isoscalar", "u", "d" or "all" (default is "non_singlet_isovector").
    - b_max (float, optional): Maximum b value for the vector b_vec=[b_x,b_y] (default is 6 GeV = 1.185 fm ).
    - Delta_max (float, optional): Maximum value for Delta integration (default is 8).
    - num_points (int, optional): Number of intervals to split [-Delta_max, Delta_max] interval for trapezoid (default is 100).
    - n_b (int, optional): Number of points the interval [-b_max, b_max] is split into (default is 50).
    - interpolation (bool, optional): Interpolate data points on finer grid
    - n_int (int, optional): Number of points for interpolation
    - vmin (float ,optional): Sets minimum value of colorbar
    - vmax (float, optional): Sets maximum value of colorbar
    - ymin (float, optional): Sets minimum value for lower plot of b_y = 0 slice
    - ymax (float, optional): Sets maximum value for lower plot of b_y = 0 slice
    - plot_option (str, optional): "upper", "lower", or "both" to control which plots are shown (default is "both").
    - read_from_file (bool): Whether to load data from file system
    - write_to_file (bool): Whether to write data to file system
    """   
    
    if write_to_file and read_from_file:
        raise ValueError("write_to_file and read_from_file can't simultaneously be True")
    
    title_map = {
            "gluon": ("g"),
            "quark": ("sea")
        }
    title = title_map[particle]
    
    FILE_PATH = PLOT_PATH + "imp_param_helicity_" + "singlet_" + particle + ".pdf"
    READ_WRITE_PATH = IMPACT_PARAMETER_MOMENTS_PATH + "imp_param_helicity_" + "singlet_" + particle

    # Define the grid for b_vec
    b_x = np.linspace(-b_max, b_max, n_b)
    b_y = np.linspace(-b_max, b_max, n_b)
    b_x_grid, b_y_grid = np.meshgrid(b_x, b_y)
    b_vecs = np.array([b_x_grid.ravel(), b_y_grid.ravel()]).T

    # Convert GeV^-1 to fm
    hbarc = 0.1975
    b_x_fm = b_x * hbarc
    b_y_fm = b_y * hbarc

    fig, axs = plt.subplots(2, 1, figsize=(3, 4.5), gridspec_kw={'height_ratios': [2, 1], 'hspace': 0})
    axs = np.array([[axs[0]], [axs[1]]])  # Make it a 2D array for consistency


    ax = axs[0, 0]
    ax_lower = axs[1, 0]

    if read_from_file:
        file_name = generate_filename(eta,0,mu,READ_WRITE_PATH,"central")
        b_x_fm, b_y_fm, fourier_transform_moment_values_flat = read_ft_from_csv(file_name)
    else:
        fourier_transform_moment_values_flat = Parallel(n_jobs=-1)(delayed(fourier_transform_quark_gluon_helicity)(
                    eta, mu, b_vec, Nf, particle,"singlet", evolution_order, Delta_max, num_points, "central") for b_vec in b_vecs)
        # Reshape
        fourier_transform_moment_values_flat = np.array([z.real for z in fourier_transform_moment_values_flat], dtype=np.float64).reshape(b_x_grid.shape)
        # Convert to fm^-2
        fourier_transform_moment_values_flat = fourier_transform_moment_values_flat/hbarc**2

    # Generate error bars for lower plot
    if plot_option in ["lower", "both"]:
        if read_from_file:
            file_name = generate_filename(eta,0,mu,READ_WRITE_PATH,"plus")
            _, _, fourier_transform_moment_values_flat_plus = read_ft_from_csv(file_name)
            file_name = generate_filename(eta,0,mu,READ_WRITE_PATH,"minus")
            _, _, fourier_transform_moment_values_flat_minus = read_ft_from_csv(file_name)
        else:
            fourier_transform_moment_values_flat_plus = Parallel(n_jobs=-1)(delayed(fourier_transform_quark_gluon_helicity)(
                eta, mu, b_vec, Nf, particle,"singlet", evolution_order, Delta_max, num_points, "plus") for b_vec in b_vecs)
            fourier_transform_moment_values_flat_minus = Parallel(n_jobs=-1)(delayed(fourier_transform_quark_gluon_helicity)(
                eta, mu, b_vec, Nf, particle,"singlet", evolution_order, Delta_max, num_points, "minus") for b_vec in b_vecs)
            # Reshape            
            fourier_transform_moment_values_flat_plus = np.array([z.real for z in fourier_transform_moment_values_flat_plus], dtype=np.float64).reshape(b_x_grid.shape)
            # Convert to fm^-2
            fourier_transform_moment_values_flat_plus = fourier_transform_moment_values_flat_plus/hbarc**2
            # Reshape
            fourier_transform_moment_values_flat_minus = np.array([z.real for z in fourier_transform_moment_values_flat_minus], dtype=np.float64).reshape(b_x_grid.shape)
            # Convert to fm^-2
            fourier_transform_moment_values_flat_minus = fourier_transform_moment_values_flat_minus/hbarc**2

    # Save write out values before interpolation
    if write_to_file:
        b_x_fm_write_out = b_x_fm
        b_y_fm_write_out = b_y_fm
        ft_write_out = fourier_transform_moment_values_flat
        if plot_option in ["lower","both"]:
            ft_write_out_plus = fourier_transform_moment_values_flat_plus
            ft_write_out_minus = fourier_transform_moment_values_flat_minus

    if interpolation:
        ft_interpolation = RectBivariateSpline(b_x_fm, b_y_fm, fourier_transform_moment_values_flat)
        ft_interpolation_plus = RectBivariateSpline(b_x_fm, b_y_fm, fourier_transform_moment_values_flat_plus)
        ft_interpolation_minus = RectBivariateSpline(b_x_fm, b_y_fm, fourier_transform_moment_values_flat_minus)

        # Call the interpolation on a finer grid
        b_x = np.linspace(-b_max, b_max, n_int)
        b_y = np.linspace(-b_max, b_max, n_int)
        b_x_fm = b_x * hbarc
        b_y_fm = b_y * hbarc

        fourier_transform_moment_values_flat = ft_interpolation(b_x_fm, b_y_fm)
        fourier_transform_moment_values_flat_plus = ft_interpolation_plus(b_x_fm, b_y_fm)
        fourier_transform_moment_values_flat_minus = ft_interpolation_minus(b_x_fm, b_y_fm)

    # Upper plot: 2D density plot
    if plot_option in ["upper", "both"]:
        im = ax.pcolormesh(b_x_fm, b_y_fm, fourier_transform_moment_values_flat, 
                            shading='auto', cmap='jet', vmin=vmin, vmax=vmax,rasterized=True)
        ax.set_xlabel(r'$b_x\,[\mathrm{fm}]$', fontsize=14)
        ax.set_ylabel(r'$b_y\,[\mathrm{fm}]$', fontsize=14)
        ax.set_xlim([-b_max * hbarc, b_max * hbarc])
        ax.set_ylim([-b_max * hbarc, b_max * hbarc])
        ax.set_title(rf"$S_z^{title}$", fontsize=14)

        # Add colorbar
        cbar_ax = fig.add_axes([ax.get_position().x1, ax.get_position().y0, 0.03, ax.get_position().height])
        fig.colorbar(im, cax=cbar_ax)

        # Lower plot: 1D slice at b_y = 0
        if plot_option in ["lower", "both"]:
            idx_by_0 = np.argmin(np.abs(b_y))
            ax_lower.plot(b_x_fm, fourier_transform_moment_values_flat[idx_by_0, :], color='black')
            ax_lower.fill_between(b_x_fm, 
                                  fourier_transform_moment_values_flat_minus[idx_by_0, :], 
                                  fourier_transform_moment_values_flat_plus[idx_by_0, :], 
                                  color='gray', alpha=0.5)
            ax_lower.set_xlabel(r'$b_x\,[\mathrm{fm}]$', fontsize=14)
            ax_lower.set_xlim([-b_max * hbarc, b_max * hbarc])
            
            ax_lower.set_ylim([ymin, ymax ])
            ax_lower.set_ylabel(rf'$S_z^{title}$', fontsize=14)

    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(FILE_PATH,format="pdf",bbox_inches="tight",dpi=600)

    if write_to_file:
        file_name = generate_filename(eta,0,mu,READ_WRITE_PATH,"central")
        save_ft_to_csv(b_x_fm_write_out,b_y_fm_write_out,ft_write_out,file_name)
        if plot_option in ["lower", "both"]:
            file_name = generate_filename(eta,0,mu,READ_WRITE_PATH,"plus")
            save_ft_to_csv(b_x_fm_write_out,b_y_fm_write_out,ft_write_out_plus,file_name)
            file_name = generate_filename(eta,0,mu,READ_WRITE_PATH,"minus")
            save_ft_to_csv(b_x_fm_write_out,b_y_fm_write_out,ft_write_out_minus,file_name)

    # Adjust layout and show the plot
    plt.show()
    plt.close()

def plot_fourier_transform_singlet_spin_orbit_correlation(eta, mu, Nf=3, particle = "gluon",evolution_order="LO",
                                          b_max=4.5, Delta_max=8, num_points=100, n_b=50, interpolation = True, n_int=300,
                                          vmin = -2.05 , vmax = 3.08, ymin= -2.05, ymax = 3.08,
                                          plot_option="both", read_from_file=True, write_to_file = False):
    """
    Generates a density plot of the 2D Fourier transform of RGE-evolved conformal moments.
    It also includes a 1D slice at b_y = 0.

    Parameters:
    - eta (float): Skewness parameter
    - mu (float): RGE scale
    - Nf (int, optional): Number of flavors (default is 3).
    - particle (str. optional): quark or gluon
    - b_max (float, optional): Maximum b value for the vector b_vec=[b_x,b_y] (default is 6 GeV = 1.185 fm ).
    - Delta_max (float, optional): Maximum value for Delta integration (default is 8).
    - num_points (int, optional): Number of intervals to split [-Delta_max, Delta_max] interval for trapezoid (default is 100).
    - n_b (int, optional): Number of points the interval [-b_max, b_max] is split into (default is 50).
    - interpolation (bool, optional): Interpolate data points on finer grid
    - n_int (int, optional): Number of points for interpolation
    - vmin (float ,optional): Sets minimum value of colorbar
    - vmax (float, optional): Sets maximum value of colorbar
    - ymin (float, optional): Sets minimum value for lower plot of b_y = 0 slice
    - ymax (float, optional): Sets maximum value for lower plot of b_y = 0 slice
    - plot_option (str, optional): "upper", "lower", or "both" to control which plots are shown (default is "both").
    - read_from_file (bool): Whether to load data from file system
    - write_to_file (bool): Whether to write data to file system
    """   
    
    if write_to_file and read_from_file:
        raise ValueError("write_to_file and read_from_file can't simultaneously be True")
    
    title_map = {
            "gluon": ("g"),
            "quark": ("sea")
        }
    title = title_map[particle]
    
    FILE_PATH = PLOT_PATH + "imp_param_" + "spin_orbit_singlet_" + particle + ".pdf"
    READ_WRITE_PATH = IMPACT_PARAMETER_MOMENTS_PATH + "imp_param_"  + "spin_orbit_singlet_" + particle

    # Define the grid for b_vec
    b_x = np.linspace(-b_max, b_max, n_b)
    b_y = np.linspace(-b_max, b_max, n_b)
    b_x_grid, b_y_grid = np.meshgrid(b_x, b_y)
    b_vecs = np.array([b_x_grid.ravel(), b_y_grid.ravel()]).T

    # Convert GeV^-1 to fm
    hbarc = 0.1975
    b_x_fm = b_x * hbarc
    b_y_fm = b_y * hbarc

    fig, axs = plt.subplots(2, 1, figsize=(3, 4.5), gridspec_kw={'height_ratios': [2, 1], 'hspace': 0})
    axs = np.array([[axs[0]], [axs[1]]])  # Make it a 2D array for consistency


    ax = axs[0, 0]
    ax_lower = axs[1, 0]

    if read_from_file:
        file_name = generate_filename(eta,0,mu,READ_WRITE_PATH,"central")
        b_x_fm, b_y_fm, fourier_transform_moment_values_flat = read_ft_from_csv(file_name)
    else:
        fourier_transform_moment_values_flat = Parallel(n_jobs=-1)(delayed(fourier_transform_spin_orbit_correlation)(
                    eta, mu, b_vec, Nf, particle, "singlet", evolution_order, Delta_max, num_points, "central") for b_vec in b_vecs)
        # Reshape
        fourier_transform_moment_values_flat = np.array([z.real for z in fourier_transform_moment_values_flat], dtype=np.float64).reshape(b_x_grid.shape)
        # Convert to fm^-2
        fourier_transform_moment_values_flat = fourier_transform_moment_values_flat/hbarc**2

    # Generate error bars for lower plot
    if plot_option in ["lower", "both"]:
        if read_from_file:
            file_name = generate_filename(eta,0,mu,READ_WRITE_PATH,"plus")
            _, _, fourier_transform_moment_values_flat_plus = read_ft_from_csv(file_name)
            file_name = generate_filename(eta,0,mu,READ_WRITE_PATH,"minus")
            _, _, fourier_transform_moment_values_flat_minus = read_ft_from_csv(file_name)
        else:
            fourier_transform_moment_values_flat_plus = Parallel(n_jobs=-1)(delayed(fourier_transform_spin_orbit_correlation)(
                eta, mu, b_vec, Nf, particle, "singlet", evolution_order, Delta_max, num_points, "plus") for b_vec in b_vecs)
            fourier_transform_moment_values_flat_minus = Parallel(n_jobs=-1)(delayed(fourier_transform_spin_orbit_correlation)(
                eta, mu, b_vec, Nf, particle, "singlet", evolution_order, Delta_max, num_points, "minus") for b_vec in b_vecs)
            # Reshape            
            fourier_transform_moment_values_flat_plus = np.array([z.real for z in fourier_transform_moment_values_flat_plus], dtype=np.float64).reshape(b_x_grid.shape)
            # Convert to fm^-2
            fourier_transform_moment_values_flat_plus = fourier_transform_moment_values_flat_plus/hbarc**2
            # Reshape
            fourier_transform_moment_values_flat_minus = np.array([z.real for z in fourier_transform_moment_values_flat_minus], dtype=np.float64).reshape(b_x_grid.shape)
            # Convert to fm^-2
            fourier_transform_moment_values_flat_minus = fourier_transform_moment_values_flat_minus/hbarc**2

    # Save write out values before interpolation
    if write_to_file:
        file_name = generate_filename(eta,0,mu,READ_WRITE_PATH,"central")
        save_ft_to_csv(b_x_fm,b_y_fm,fourier_transform_moment_values_flat,file_name)
        if plot_option in ["lower", "both"]:
            file_name = generate_filename(eta,0,mu,READ_WRITE_PATH,"plus")
            save_ft_to_csv(b_x_fm,b_y_fm,fourier_transform_moment_values_flat_plus,file_name)
            file_name = generate_filename(eta,0,mu,READ_WRITE_PATH,"minus")
            save_ft_to_csv(b_x_fm,b_y_fm,fourier_transform_moment_values_flat_minus,file_name)

    if interpolation:
        ft_interpolation = RectBivariateSpline(b_x_fm, b_y_fm, fourier_transform_moment_values_flat)
        ft_interpolation_plus = RectBivariateSpline(b_x_fm, b_y_fm, fourier_transform_moment_values_flat_plus)
        ft_interpolation_minus = RectBivariateSpline(b_x_fm, b_y_fm, fourier_transform_moment_values_flat_minus)

        # Call the interpolation on a finer grid
        b_x = np.linspace(-b_max, b_max, n_int)
        b_y = np.linspace(-b_max, b_max, n_int)
        b_x_fm = b_x * hbarc
        b_y_fm = b_y * hbarc

        fourier_transform_moment_values_flat = ft_interpolation(b_x_fm, b_y_fm)
        fourier_transform_moment_values_flat_plus = ft_interpolation_plus(b_x_fm, b_y_fm)
        fourier_transform_moment_values_flat_minus = ft_interpolation_minus(b_x_fm, b_y_fm)

    # Upper plot: 2D density plot
    if plot_option in ["upper", "both"]:
        im = ax.pcolormesh(b_x_fm, b_y_fm, fourier_transform_moment_values_flat, 
                            shading='auto', cmap='jet', vmin=vmin, vmax=vmax,rasterized=True)
        ax.set_xlabel(r'$b_x\,[\mathrm{fm}]$', fontsize=14)
        ax.set_ylabel(r'$b_y\,[\mathrm{fm}]$', fontsize=14)
        ax.set_xlim([-b_max * hbarc, b_max * hbarc])
        ax.set_ylim([-b_max * hbarc, b_max * hbarc])
        ax.set_title(rf"$C_z^{title}$", fontsize=14)

        # Add colorbar
        cbar_ax = fig.add_axes([ax.get_position().x1, ax.get_position().y0, 0.03, ax.get_position().height])
        fig.colorbar(im, cax=cbar_ax)

        # Lower plot: 1D slice at b_y = 0
        if plot_option in ["lower", "both"]:
            idx_by_0 = np.argmin(np.abs(b_y))
            ax_lower.plot(b_x_fm, fourier_transform_moment_values_flat[idx_by_0, :], color='black')
            ax_lower.fill_between(b_x_fm, 
                                  fourier_transform_moment_values_flat_minus[idx_by_0, :], 
                                  fourier_transform_moment_values_flat_plus[idx_by_0, :], 
                                  color='gray', alpha=0.5)
            ax_lower.set_xlabel(r'$b_x\,[\mathrm{fm}]$', fontsize=14)
            ax_lower.set_xlim([-b_max * hbarc, b_max * hbarc])
            
            ax_lower.set_ylim([ymin, ymax ])
            ax_lower.set_ylabel(rf'$C_z^{title}$', fontsize=14)

    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(FILE_PATH,format="pdf",bbox_inches="tight",dpi=600)

    # Adjust layout and show the plot
    plt.show()
    plt.close()


def plot_fourier_transform_quark_orbital_angular_momentum(eta, mu, Nf=3, moment_type="non_singlet_isovector",evolution_order="LO", 
                                          b_max=3, Delta_max=7, num_points=100, n_b=50, interpolation = True,n_int=300,
                                          vmin = -2 , vmax = 2, ymin = -2, ymax = .3,
                                          plot_option="both", read_from_file=True, write_to_file = False):
    """
    Generates a density plot of the 2D Fourier transform of RGE-evolved conformal moments.
    It also includes a 1D slice at b_y = 0.

    Parameters:
    - eta (float): Skewness parameter
    - mu (float): RGE scale
    - Nf (int, optional): Number of flavors (default is 3).
    - moment_type (str, optional): "non_singlet_isovector", "non_singlet_isoscalar", "u", "d" or "all" (default is "non_singlet_isovector").
    - b_max (float, optional): Maximum b value for the vector b_vec=[b_x,b_y] (default is 6 GeV = 1.185 fm ).
    - Delta_max (float, optional): Maximum value for Delta integration (default is 7).
    - num_points (int, optional): Number of intervals to split [-Delta_max, Delta_max] interval for trapezoid (default is 100).
    - n_b (int, optional): Number of points the interval [-b_max, b_max] is split into (default is 50).
    - interpolation (bool, optional): Interpolate data points on finer grid
    - n_int (int, optional): Number of points used for interpolation
    - vmin (float ,optioanl): Sets minimum value of colorbar
    - vmax (float, optional): Sets maximum value of colorbar
    - ymin (float, optional): Sets minimum value for lower plot of b_y = 0 slice
    - ymax (float, optional): Sets maximum value for lower plot of b_y = 0 slice
    - plot_option (str, optional): "upper", "lower", or "both" to control which plots are shown (default is "both").
    - read_from_file (bool): Whether to load data from file system
    - write_to_file (bool): Whether to write data to file system
    """   
    if write_to_file and read_from_file:
        raise ValueError("write_to_file and read_from_file can't simultaneously be True")


    if moment_type not in ["non_singlet_isovector", "non_singlet_isoscalar", "u", "d", "all"]:
        raise ValueError(f"Wrong moment_type {moment_type}")

    FILE_PATH = PLOT_PATH + "imp_param_oam_" + moment_type +".pdf"

    # Convert GeV^-1 to fm
    hbarc = 0.1975

    moment_types = ["non_singlet_isovector", "non_singlet_isoscalar", "u", "d"] if moment_type == "all" else [moment_type]

    # Initialize cache to store Fourier transforms for "non_singlet_isovector" and "non_singlet_isoscalar"
    cache = {}

    # Determine figure layout
    if moment_type == "all":
        fig, axs = plt.subplots(2, len(moment_types), figsize=(len(moment_types)*3, 4.5), gridspec_kw={'height_ratios': [2, 1], 'hspace': 0})
    else:
        fig, axs = plt.subplots(2, 1, figsize=(3, 4.5), gridspec_kw={'height_ratios': [2, 1], 'hspace': 0})
        axs = np.array([[axs[0]], [axs[1]]])  # Make it a 2D array for consistency

    for i, mom_type in enumerate(moment_types):
        READ_WRITE_PATH = IMPACT_PARAMETER_MOMENTS_PATH + "imp_param_oam_" + mom_type 
        # Update the grid to data contained in file
        if read_from_file:
            file_name = generate_filename(eta,0,mu,READ_WRITE_PATH,"central")
            b_x_fm, b_y_fm, fourier_transform_moment_values_flat = read_ft_from_csv(file_name)


        row, col = divmod(i, 4)  # Map index to subplot location
        ax = axs[0, col]
        ax_lower = axs[1, col]

        title_map = {
            "non_singlet_isovector": "u-d",
            "non_singlet_isoscalar": "u+d",
            "u": "u",
            "d": "d"
        }
        title = title_map[mom_type]

        # Compute Fourier transform and cache the results for non_singlet_isovector and non_singlet_isoscalar
        if mom_type in ["non_singlet_isovector", "non_singlet_isoscalar"] or moment_type != "all":
            # Define the grid for b_vec
            b_x = np.linspace(-b_max, b_max, n_b)
            b_y = np.linspace(-b_max, b_max, n_b)
            b_x_grid, b_y_grid = np.meshgrid(b_x, b_y)
            b_vecs = np.array([b_x_grid.ravel(), b_y_grid.ravel()]).T

            b_x_fm = b_x * hbarc
            b_y_fm = b_y * hbarc
            if mom_type not in cache:
                if read_from_file:
                    file_name = generate_filename(eta,0,mu,READ_WRITE_PATH,"central")
                    b_x_fm, b_y_fm, fourier_transform_moment_values_flat = read_ft_from_csv(file_name)
                    # Extract shape for reshaping
                    n_b = len(fourier_transform_moment_values_flat)
                    b_x = np.linspace(-b_max, b_max, n_b)
                    b_y = np.linspace(-b_max, b_max, n_b)
                    b_x_grid, b_y_grid = np.meshgrid(b_x, b_y)
                else:
                    fourier_transform_moment_values_flat = Parallel(n_jobs=-1)(delayed(fourier_transform_quark_orbital_angular_momentum)(
                        eta, mu, b_vec, Nf, mom_type, evolution_order, Delta_max, num_points, "central") for b_vec in b_vecs)
                    # Reshape
                    fourier_transform_moment_values_flat = np.array(fourier_transform_moment_values_flat).reshape(b_x_grid.shape)
                    # Convert to fm^-2
                    fourier_transform_moment_values_flat = fourier_transform_moment_values_flat/hbarc**2
                # Cache
                cache[mom_type] = fourier_transform_moment_values_flat

                # Generate error bars for lower plot
                if plot_option in ["lower", "both"]:
                    if read_from_file:
                        file_name = generate_filename(eta,0,mu,READ_WRITE_PATH,"plus")
                        _, _, fourier_transform_moment_values_flat_plus = read_ft_from_csv(file_name)
                        file_name = generate_filename(eta,0,mu,READ_WRITE_PATH,"minus")
                        _, _, fourier_transform_moment_values_flat_minus = read_ft_from_csv(file_name)
                    else:
                        fourier_transform_moment_values_flat_plus = Parallel(n_jobs=-1)(delayed(fourier_transform_quark_orbital_angular_momentum)(
                            eta, mu, b_vec, Nf, mom_type, evolution_order, Delta_max, num_points, "plus") for b_vec in b_vecs)
                        fourier_transform_moment_values_flat_minus = Parallel(n_jobs=-1)(delayed(fourier_transform_quark_orbital_angular_momentum)(
                            eta, mu, b_vec, Nf, mom_type, evolution_order, Delta_max, num_points, "minus") for b_vec in b_vecs)
                        # Reshape
                        fourier_transform_moment_values_flat_plus = np.array(fourier_transform_moment_values_flat_plus).reshape(b_x_grid.shape)
                        # Convert to fm^-2
                        fourier_transform_moment_values_flat_plus = fourier_transform_moment_values_flat_plus/hbarc**2
                        # Reshape
                        fourier_transform_moment_values_flat_minus = np.array(fourier_transform_moment_values_flat_minus).reshape(b_x_grid.shape)
                        # Convert to fm^-2
                        fourier_transform_moment_values_flat_minus = fourier_transform_moment_values_flat_minus/hbarc**2

                    cache[f"{mom_type}_plus"] = fourier_transform_moment_values_flat_plus
                    cache[f"{mom_type}_minus"] = fourier_transform_moment_values_flat_minus

        # Use cached values for u and d
        if mom_type in ["u","d"]:
            if read_from_file:
                file_name = generate_filename(eta,0,mu,READ_WRITE_PATH,"central")
                b_x_fm, b_y_fm, _ = read_ft_from_csv(file_name)
            elif interpolation:
                b_x = np.linspace(-b_max, b_max, n_b)
                b_y = np.linspace(-b_max, b_max, n_b)
                b_x_fm = b_x * hbarc
                b_y_fm = b_y * hbarc

            error_plus = np.sqrt((cache["non_singlet_isoscalar_plus"]-cache["non_singlet_isoscalar"])**2
                                    + (cache["non_singlet_isovector_plus"]-cache["non_singlet_isovector"])**2)/2
            error_minus = np.sqrt((cache["non_singlet_isoscalar_minus"]-cache["non_singlet_isoscalar"])**2
                                    + (cache["non_singlet_isovector_minus"]-cache["non_singlet_isovector"])**2)/2
            if mom_type == "u":
                prf = 1
            if mom_type == "d":
                prf = -1 
            fourier_transform_moment_values_flat = (cache["non_singlet_isoscalar"] + prf * cache["non_singlet_isovector"]) / 2
            fourier_transform_moment_values_flat_plus = fourier_transform_moment_values_flat + error_plus
            fourier_transform_moment_values_flat_minus = fourier_transform_moment_values_flat - error_minus

        # Save write out values before interpolation
        if write_to_file:
            b_x_fm_write_out = b_x_fm
            b_y_fm_write_out = b_y_fm
            ft_write_out = fourier_transform_moment_values_flat
            if plot_option in ["lower","both"]:
                ft_write_out_plus = fourier_transform_moment_values_flat_plus
                ft_write_out_minus = fourier_transform_moment_values_flat_minus

        if interpolation:
            ft_interpolation = RectBivariateSpline(b_x_fm, b_y_fm, fourier_transform_moment_values_flat)
            ft_interpolation_plus = RectBivariateSpline(b_x_fm, b_y_fm, fourier_transform_moment_values_flat_plus)
            ft_interpolation_minus = RectBivariateSpline(b_x_fm, b_y_fm, fourier_transform_moment_values_flat_minus)

            # Call the interpolation on a finer grid
            b_x = np.linspace(-b_max, b_max, n_int)
            b_y = np.linspace(-b_max, b_max, n_int)
            b_x_fm = b_x * hbarc
            b_y_fm = b_y * hbarc

            fourier_transform_moment_values_flat = ft_interpolation(b_x_fm, b_y_fm)
            fourier_transform_moment_values_flat_plus = ft_interpolation_plus(b_x_fm, b_y_fm)
            fourier_transform_moment_values_flat_minus = ft_interpolation_minus(b_x_fm, b_y_fm)


        # Upper plot: 2D density plot
        if plot_option in ["upper", "both"]:
            im = ax.pcolormesh(b_x_fm, b_y_fm, fourier_transform_moment_values_flat, 
                               shading='auto', cmap='jet',vmin=vmin, vmax=vmax,rasterized=True)
            ax.set_xlabel(r'$b_x\,[\mathrm{fm}]$', fontsize=14)
            if i == 0:
                ax.set_ylabel(r'$b_y\,[\mathrm{fm}]$', fontsize=14)
            ax.set_title(rf"$L_z^{{{title}}}$", fontsize=14)
            ax.set_xlim([-b_max * hbarc, b_max * hbarc])
            ax.set_ylim([-b_max * hbarc, b_max * hbarc])

            # Add colorbar only once per row
            if col == len(moment_types)-1:
                cbar_ax = fig.add_axes([ax.get_position().x1, ax.get_position().y0, 0.01, ax.get_position().height])
                fig.colorbar(im, cax=cbar_ax)

        # Lower plot: 1D slice at b_y = 0
        if plot_option in ["lower", "both"]:
            idx_by_0 = np.argmin(np.abs(b_y))
            ax_lower.plot(b_x_fm, fourier_transform_moment_values_flat[idx_by_0, :], color='black')
            ax_lower.fill_between(b_x_fm, 
                                  fourier_transform_moment_values_flat_minus[idx_by_0, :], 
                                  fourier_transform_moment_values_flat_plus[idx_by_0, :], 
                                  color='gray', alpha=0.5)
            ax_lower.set_xlabel(r'$b_x\,[\mathrm{fm}]$', fontsize=14)
            ax_lower.set_xlim([-b_max * hbarc, b_max * hbarc])
            ax_lower.set_ylim([ymin, ymax])

            if moment_type == "all" and i == 0:
                ax_lower.set_ylabel(rf'$L_z^{{q}}$', fontsize=14)
            elif i == 0:
                ax_lower.set_ylabel(rf'$L_z^{{{title}}}$', fontsize=14)
            # Remove ticks and labels
            if i != 0 and moment_type == "all":
                ax.set_yticks([])
                ax.set_yticklabels([])
                ax.set_ylabel(None)

                ax_lower.set_yticks([])
                ax_lower.set_yticklabels([])

        if write_to_file:
            file_name = generate_filename(eta,0,mu,READ_WRITE_PATH,"central")
            save_ft_to_csv(b_x_fm_write_out,b_y_fm_write_out,ft_write_out,file_name)
            if plot_option in ["lower", "both"]:
                file_name = generate_filename(eta,0,mu,READ_WRITE_PATH,"plus")
                save_ft_to_csv(b_x_fm_write_out,b_y_fm_write_out,ft_write_out_plus,file_name)
                file_name = generate_filename(eta,0,mu,READ_WRITE_PATH,"minus")
                save_ft_to_csv(b_x_fm_write_out,b_y_fm_write_out,ft_write_out_minus,file_name)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(FILE_PATH,format="pdf",bbox_inches="tight",dpi=600)

    # Adjust layout and show the plot
    plt.show()
    plt.close()


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

def plot_evolved_moment_over_j(eta,t,mu,Nf = 3,j_base = 3,particle="quark",moment_type="non_singlet_isovector",moment_label ="A",evolution_order="LO", 
                            error_type = "central", j_max=5, num_points=200, real_imag = "real"):
    """
    Plot the real and imaginary parts of the evolved conformal moment 
    over -j_max < k < j_max, with j transformed as j -> z = j_base + 1j * k.

    Arguments:
    particle -- "quark" or "gluon"
    moment_label -- Label of the moment (e.g., "A", "B")
    parity -- Parity information used to compute j_base
    eta -- Skewness parameter
    t -- Mandelstam t
    mu -- Resolution scale
    Nf -- Number of active flavors (default: 3)
    moment_type -- Type of moment ("non_singlet_isovector", "non_singlet_isoscalar", or "singlet")
    error_type -- Choose "central", "upper", or "lower" value for input PDF parameters
    j_max -- Maximum absolute value of k (default: 5)
    num_points -- Number of points for plotting (default: 200)
    """

    # Define k values
    
    
    if real_imag == "real":
        k_vals = np.linspace(j_base, j_max, num_points)
        z_vals = k_vals
    elif real_imag == "imag":
        k_vals = np.linspace(-j_max, j_max, num_points)
        z_vals = j_base +  1j * k_vals 

    # Evaluate the function for each z
    evolved_moment = np.array(
        Parallel(n_jobs=-1)(delayed(evolve_conformal_moment)(z, eta, t, mu, Nf, 
                                                        particle, moment_type, moment_label, evolution_order, error_type) for z in z_vals),
                dtype=complex)
    # evolved_moment = np.array([evolve_conformal_moment(z, eta, t, mu, Nf, 
    #                                                     particle, moment_type, moment_label, error_type) for z in z_vals])

    # Plot real and imaginary parts
    plt.figure(figsize=(8, 5))
    plt.plot(k_vals, evolved_moment.real, label="Re[evolved moment]", color='b')
    plt.plot(k_vals, evolved_moment.imag, label="Im[evolved moment]", color='r', linestyle='dashed')
    plt.xlabel(r"$k$")
    plt.ylabel(r"$\mathcal{F}(z)$")
    plt.title(f"Evolution of Conformal Moment for {particle}, {moment_label}")
    plt.axhline(0, color='black', linewidth=0.5, linestyle='dotted')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='dotted')
    plt.legend(fontsize=10, markerscale=1.5)
    plt.grid()
    plt.show()

def plot_conformal_partial_wave_over_j(x,eta,particle="quark",moment_type="non_singlet_isovector",moment_label ="A",parity="none"):
    """Plots the conformal parftial wave over conformal spin-j for given eta, particle and parity.

    Parameters:
    - j (float): conformal spin
    - eta (float): skewness
    - particle (str., optiona): quark or gluon. Default is quark
    - parity (str., optional): even, odd, or none. Default is none
    """
    check_particle_type(particle)
    check_parity(parity)

    j_base, parity = get_j_base(particle,moment_type,moment_label)
    k_values = np.linspace(-15, 15, 200)
    j_values = j_base + 1j * k_values
    y_values = np.array(Parallel(n_jobs=-1)(delayed(conformal_partial_wave)(j, x, eta , particle, parity) for j in j_values)
                        ,dtype=complex)

    # Create subplots for real and imaginary parts
    plt.figure(figsize=(10, 6))  # Adjust figure size for better visualization

    #plt.subplot(2, 1, 1)
    plt.plot(k_values, y_values.real)
    plt.xlabel("j")
    plt.ylabel("Real Part")
    plt.title(f"Real Part of Conformal Partial Wave for {particle} with Parity {parity}")

    #plt.subplot(2, 1, 2)
    plt.plot(k_values, y_values.imag)
    plt.xlabel("j")
    plt.ylabel("Imaginary Part")
    plt.title(f"Imaginary Part of Conformal Partial Wave for {particle} with Parity {parity}")

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()

def plot_mellin_barnes_gpd_integrand(x, eta, t, mu, Nf=3, particle="quark", moment_type="singlet", moment_label="A",evolution_order="LO", parity = "none", error_type="central", j_max=7.5):
    """
    Plot the real and imaginary parts of the integrand of the Mellin-Barnes integral over k with j = j_base + i*k.

    Parameters:
    - x, eta, t, mu: Physical parameters.
    - Nf (int): Number of flavors.
    - particle (str): Particle species ("quark" or "gluon").
    - moment_type (str. optional): Moment type ("singlet", "non_singlet_isovector", "non_singlet_isoscalar").
    - moment_label (str. optional): Moment label ("A", "Atilde", "B").
    - parity (str., optional)
    - error_type (str. optional): PDF value type ("central", "plus", "minus").
    - j_max (float. optional): Maximum value of imaginary part k for plotting.
    """
    check_parity(parity)
    check_error_type(error_type)
    check_particle_type(particle)
    check_moment_type_label(moment_type,moment_label)

    j_base, parity_check = get_j_base(particle,moment_type,moment_label)
    if parity != parity_check:
        print(f"Warning: Wrong parity of {parity} for moment_type of {moment_type} for particle {particle}")

    if eta == 0:
        eta = 1e-6

    def integrand_real(k):
        # Plot imag
        z = j_base + 1j * k
        # Plot real
        #z = k
        dz = 1j
        sin_term = mp.sin(np.pi * z)
        pw_val = conformal_partial_wave(z, x, eta, particle, parity)
        if particle == "quark":
            if moment_type == "singlet":
                mom_val = evolve_quark_singlet(z, eta, t, mu, Nf, moment_label, evolution_order, error_type)
            else:
                mom_val = evolve_quark_non_singlet(z, eta, t, mu, Nf, moment_type, moment_label, evolution_order, error_type)
        else:
            mom_val = evolve_gluon_singlet(z, eta, t, mu, Nf, moment_label, evolution_order, error_type)
        result = -0.5j * dz * pw_val * mom_val / sin_term
        return result.real

    def integrand_imag(k):
        # Plot imag
        z = j_base + 1j * k
        # Plot real
        #z = k
        dz = 1j
        sin_term = mp.sin(np.pi * z)
        pw_val = conformal_partial_wave(z, x, eta, particle, parity)
        if particle == "quark":
            if moment_type == "singlet":
                mom_val = evolve_quark_singlet(z, eta, t, mu, Nf, moment_label, evolution_order, error_type)
            else:
                mom_val = evolve_quark_non_singlet(z, eta, t, mu, Nf, moment_type, moment_label, evolution_order, error_type)
        else:
            mom_val = (-1) * evolve_gluon_singlet(z, eta, t, mu, Nf,moment_label, evolution_order, error_type)
        result = -0.5j * dz * pw_val * mom_val / sin_term
        return result.imag

    print(f"Integrand at j_max={j_max}")
    print(integrand_real(j_max))
    print(integrand_imag(j_max))

    # Define k range for plotting
    k_values = np.linspace(-j_max, j_max, 300)
    #k_values = np.linspace(1.1, 10, 300)
    #k_values = np.linspace(j_base,j_max,300)
    # Parallel computation of real and imaginary parts
    real_values = Parallel(n_jobs=-1)(delayed(integrand_real)(k) for k in k_values)
    imag_values = Parallel(n_jobs=-1)(delayed(integrand_imag)(k) for k in k_values)

    # Plotting
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(k_values, real_values, label="Real Part", color="blue")
    ax[0].set_ylabel("Real Part")
    ax[0].legend()
    ax[0].grid()
    #ax[0].set_ylim([-30,40])

    ax[1].plot(k_values, imag_values, label="Imaginary Part", color="red")
    ax[1].set_xlabel("k")
    ax[1].set_ylabel("Imaginary Part")
    ax[1].legend()
    ax[1].grid()

    plt.suptitle("Integrand Real and Imaginary Parts")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_gpd_data(Nf=3,particle="quark",gpd_type="non_singlet_isovector",gpd_label="Htilde",evolution_order="LO",n_int=300,n_gpd=50,sampling=True,n_init=os.cpu_count(), 
                  plot_gpd =True, error_bars=True,write_to_file = False, read_from_file = True, plot_legend = True, y_0 = -1e-1, y_1 =3):
    """
    ADD CORRECT SELECTION OF GPD TYPE: CURRENTLY RUNS OVER ALL DATA

    Plots a GPD of gpd_type and gpd_label over x together with the available data on the file system. Adjust GPD_LABEL_MAP and y_label_map as desired.

    Parameters:
    - Nf (int, optional): Number of active flavors
    - particle (str. optional): quark or gluon
    - gpd_type (str. optional): non_singlet_isovector, singlet,...
    - gpd_label (str. optional): H, E, Htilde...
    - n_int (int. optional): Number of points used for interpolation
    - n_gpd (int. optional): Number of data points generated
    - sampling (Bool, optional): Whether to sample the GPD an add more points in important areas
    - n_init (int. optional): Number of points used for sampling
    - plot_gpd (Bool, optional): Plot GPD or only data
    - error_bars (Bool, optional): Generate error bars for GPD
    - write_to_file (Bool, optional): Write the generated data points on the file system
    - read_from_file (Bool, optional): Read data from file system. Skips GPD data generation
    - plot_legend (Bool, optional): Show plot legend
    - y_0 (float, optional): Lower bound on y axis
    - y_1 (float, optional): Upper bound on y axis
    """
    def compute_result(x, eta,t,mu,error_type="central"):
        return mellin_barnes_gpd(x, eta, t, mu, Nf,particle,gpd_type,moment_label, evolution_order, real_imag="real", error_type=error_type,n_jobs=1)
    
    y_label_map = {
            ("non_singlet_isovector","Htilde"): "$\\widetilde{{H}}^{{u-d}}(x,\eta,t;\mu)$",
            ("non_singlet_isoscalar", "Htilde"): "$\\widetilde{{H}}^{{u+d}}(x,\eta,t;\mu)$",
            ("u","Htilde"): "$\\widetilde{{H}}^{{u}}(x,\eta,t;\mu)$",
            ("d","Htilde"): "$\\widetilde{{H}}^{{d}}(x,\eta,t;\mu)$",
            ("non_singlet_isovector","E"): "$E^{{u-d}}(x,\eta,t;\mu)$",
            ("non_singlet_isoscalar", "E"): "$E^{{u+d}}(x,\eta,t;\mu)$",
            ("u","E"): "$E^{{u}}(x,\eta,t;\mu)$",
            ("d","E"): "$E^{{d}}(x,\eta,t;\mu)$"
    }

    if (gpd_type, gpd_label) in y_label_map:
        y_label = y_label_map[(gpd_type, gpd_label)]
    else:
        print(f"Key ({gpd_type}, {gpd_label}) not found in y_label_map - abort")
        return
    if (gpd_label) in GPD_LABEL_MAP:
        moment_label = GPD_LABEL_MAP[gpd_label]
    else:
        print(f"Key {gpd_label} not found in GPD_LABEL_MAP - abort")
        return

    if write_to_file and read_from_file:
        raise ValueError("write_to_file and read_from_file can't simultaneously be True")

    # Initialize plot
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    for (pub_id,gpd_type_data,gpd_label_data,eta,t,mu), (color,_) in GPD_PUBLICATION_MAPPING.items():
        # Check whether type and label agree with input
        if gpd_type_data != gpd_type or gpd_label_data != gpd_label:
            continue
        gpd_interpolation={} # Initialize dictionary
        for error_type in ["central","plus","minus"]:
            x_values, gpd_values = load_lattice_gpd_data(eta,t,mu,particle,gpd_type,gpd_label,pub_id,error_type)
            if x_values is None:
                continue
            x_values = np.around(x_values, decimals=4)
            gpd_values = np.around(gpd_values, decimals=4)
            x_min = np.min(x_values)
            x_max = np.max(x_values)
            x_fine = np.linspace(x_min, x_max, n_int)

            gpd_interpolation[error_type] = np.zeros_like(x_fine)
            # Separate interpolation for DGLAP and ERBL region
            for i, x in enumerate(x_fine): 
                region_mask = (
                    (x <= -eta) & (x_values <= -eta) | 
                    (-eta <= x <= eta) & ( (x_values >= -eta) & (x_values <= eta)) |
                    (x >= eta) & (x_values >= eta)  
                )

                x_region = x_values[region_mask]
                y_region = gpd_values[region_mask]

                if len(x_region) > 1:
                    interpolate_function = interp1d(x_region, y_region,bounds_error=False, fill_value="extrapolate")
                    gpd_interpolation[error_type][i] = interpolate_function(x)
                else:
                    gpd_interpolation[error_type][i] = np.interp(x, x_region, y_region)
        if sampling and plot_gpd and not read_from_file:
            x_val = np.linspace(x_min,x_max,n_init)
            # Measure time for sampling initial points
            start_time_sampling = time.time()
            results = Parallel(n_jobs=-1)(delayed(compute_result)(x,eta,t,mu) for x in x_val)
            end_time_sampling = time.time()

            # Compute differences and scale intervals
            diffs = np.abs(np.diff(results))
            # Ensure diffs is not zero for power
            diffs += 1e-6
            scaled_intervals = np.power(diffs, 0.5) / np.sum(np.power(diffs, 0.5))
            cumulative_intervals = np.cumsum(np.insert(scaled_intervals, 0, 0))
            cumulative_intervals = x_min + (x_max - x_min) * (
                cumulative_intervals - cumulative_intervals[0]) / (
                    cumulative_intervals[-1] - cumulative_intervals[0])
            x_val = np.interp(np.linspace(x_min, x_max, n_gpd), cumulative_intervals, x_val)
            # Output sampling time
            print(f"Time for initial sampling for parameters (eta,t) = ({eta,t}): {end_time_sampling - start_time_sampling:.6f} seconds")
        elif not read_from_file:
            x_val = np.linspace(x_min, x_max, n_gpd)

            # Add crossover points
            if eta not in x_val:
                x_val = np.append(x_val,eta)
            if - eta not in x_val and x_min < 0:
                x_val = np.append(x_val,-eta)
            x_val = np.sort(x_val)


        indices = []
        x_lin = np.linspace(x_min,x_max,n_gpd)
        # Prepare data for interpolation
        for x in x_lin:
            # Compute the absolute difference between x and each element in x_values
            diff = np.abs(x_fine- x)
            
            # Find indices where the difference is within the tolerance
            close_indices = np.where(diff <= 1e-1)[0]
            
            # If there's a match, append only one index (the first closest match)
            if len(close_indices) > 0:
                # Find the index of the closest value (the smallest difference)
                closest_index = int(close_indices[np.argmin(diff[close_indices])])
                indices.append(closest_index)
            else:
                # Handle case when no close match is found, can append None or skip
                indices.append(None)
        
        valid_indices = [i for i in indices if i is not None]

        gpd_errorbar_plot_values = gpd_interpolation["central"][valid_indices]
        error_p = gpd_interpolation["plus"][valid_indices] - gpd_errorbar_plot_values
        error_m = gpd_errorbar_plot_values - gpd_interpolation["minus"][valid_indices]
        gpd_errorbar_plot_errors = np.sqrt(error_p**2+error_m**2)/2

        if plot_gpd:
            # Measure time for adaptive grid computation
            start_time_adaptive = time.time()
            if read_from_file:
                x_val, results = load_gpd_data(eta,t,mu,particle,gpd_type,gpd_label)
                if x_val is None:
                    print(f"No data for {gpd_type} {gpd_label} at (eta,t,mu) = {eta},{t},{mu} - abort ")
                    return 
                    #raise ValueError("No data found on system. Change write_to_file = True")
            else:
                results = Parallel(n_jobs=-1)(delayed(compute_result)(x,eta,t,mu) for x in x_val)

            # Error bar computations
            if error_bars:
                if read_from_file:
                    x_plus, results_plus = load_gpd_data(eta,t,mu,particle,gpd_type,gpd_label,"plus")
                    x_minus,results_minus = load_gpd_data(eta,t,mu,particle,gpd_type,gpd_label,"minus")
                    if not np.array_equal(x_plus, x_minus) or not np.array_equal(x_plus, x_val):
                        raise ValueError("Mismatch in x-values between error data files")
                else:
                    results_plus = Parallel(n_jobs=-1)(delayed(compute_result)(x,eta,t,mu, error_type="plus") for x in x_val)
                    results_minus = Parallel(n_jobs=-1)(delayed(compute_result)(x,eta,t,mu, error_type="minus") for x in x_val)
            else:
                results_plus = results
                results_minus = results
            end_time_adaptive = time.time()

            # Write to file system
            if write_to_file:
                save_gpd_data(x_val,eta,t,mu,results,particle,gpd_type,gpd_label)
                save_gpd_data(x_val,eta,t,mu,results_plus,particle,gpd_type,gpd_label,"plus")
                save_gpd_data(x_val,eta,t,mu,results_minus,particle,gpd_type,gpd_label,"minus")

            print(f"Time for plot computation for parameters (eta,t) = ({eta,t}): {end_time_adaptive - start_time_adaptive:.6f} seconds")

            if error_bars:
                ax.plot(x_val, results,label=(f"$\\eta={eta:.2f}$, "
                        f"$t={t:.2f} \\text{{ GeV}}^2$, "), color=color)#f"$\\mu={mu:.2f} \\text{{ GeV}}$"), color=color)
                ax.fill_between(x_val,results_minus,results_plus,color=color,alpha=.2)
            else:
                ax.plot(x_val, results,label=(f"$\\eta={eta:.2f}$, "
                        f"$t={t:.2f} \\text{{ GeV}}^2$, "), color=color)#f"$\\mu={mu:.2f} \\text{{ GeV}}$"), color=color)
            ax.errorbar(x_lin, gpd_errorbar_plot_values, yerr=gpd_errorbar_plot_errors, fmt='o', color=color,markersize = 4)
        else: 
        # Errorbar plot for interpolation
            plt.errorbar(x_lin, gpd_errorbar_plot_values, yerr=gpd_errorbar_plot_errors, fmt='o', color=color,label=(f"$\\eta={eta:.2f}$, "
                        f"$t={t:.2f} \\text{{ GeV}}^2$, "))#f"$\\mu={mu:.2f} \\text{{ GeV}}$"),markersize = 4)

        ax.axvline(x=eta, linestyle='--', color = color)   
        ax.axvline(x=-eta, linestyle='--', color = color)

        # Interpolation line
        #plt.plot(x_fine, gpd_interpolation["central"], '-', color=color,label=f'$\eta = {eta},\ t ={t:.2f}\\ GeV $')
                 #,\ \mu = {mu:.2f}\\ GeV$')

        # Filled area for interpolation uncertainty
        #plt.fill_between(x_fine, gpd_interpolation["minus"], gpd_interpolation["plus"], alpha=0.2, color=color) 


    ax.set_ylim(y_0, y_1)
    ax.set_xlabel("x")  # Add axis labels
    ax.set_ylabel(y_label, fontsize =14)
    if plot_legend:
        ax.legend(fontsize=10, markerscale=1.5)
    ax.grid(True)  # Add a grid for better readability
    FILE_PATH = PLOT_PATH + gpd_type + "_" + particle + "_GPD_" + gpd_label +"_comparison" + ".pdf"
    fig.savefig(FILE_PATH,format="pdf",bbox_inches="tight",dpi=600)

def plot_singlet_quark_gpd(eta, t, mu, Nf=3, moment_label="A",evolution_order="LO", real_imag="real", sampling=True, n_init=os.cpu_count(), n_points=20, x_0=1e-2, x_1=1, error_bars=True):
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
    check_moment_type_label("singlet",moment_label)
    # Ensure x_0 < x_1 for a valid range
    if x_0 >= x_1:
        raise ValueError("x_0 must be less than x_1.")

    if x_0 <= 0:
        raise ValueError("x_0 must be greater than zero.")

    def compute_result(x, error_type="central"):
        return mellin_barnes_gpd(x, eta, t, mu, Nf,particle="quark",moment_type="singlet",moment_label=moment_label, evolution_order=evolution_order, real_imag=real_imag, error_type=error_type,n_jobs=1)

    if sampling:

        x_values = np.linspace(x_0, x_1, n_init)

        # Measure time for sampling initial points
        start_time_sampling = time.time()
        results = Parallel(n_jobs=-1)(delayed(compute_result)(x,eta,t,mu) for x in x_values)
        end_time_sampling = time.time()

        # Compute differences and scale intervals
        diffs = np.abs(np.diff(results))
        # Ensure diffs is not zero for power
        diffs += 1e-6
        scaled_intervals = np.power(diffs, 0.5) / np.sum(np.power(diffs, 0.5))
        cumulative_intervals = np.cumsum(np.insert(scaled_intervals, 0, 0))
        cumulative_intervals = x_0 + (x_1 - x_0) * (
            cumulative_intervals - cumulative_intervals[0]) / (
                cumulative_intervals[-1] - cumulative_intervals[0])

        # Output sampling time
        print(f"Time for initial sampling for parameters (eta,t) = ({eta,t}): {end_time_sampling - start_time_sampling:.6f} seconds")

    # Measure time for adaptive grid computation
    start_time_adaptive = time.time()
    if sampling:
        x_values = np.interp(np.linspace(x_0, x_1, n_points), cumulative_intervals, x_values)
    else:
        x_values = np.linspace(x_0, x_1, n_points)

    # Add crossover points
    if eta not in x_values:
        x_values = np.append(x_values,eta)
    if - eta not in x_values and x_0 < 0:
        x_values = np.append(x_values,-eta)
    x_values = np.sort(x_values)

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
        ("real", real_parts, real_errors_minus, real_errors_plus, 'b'),
        ("imag", imag_parts, imag_errors_minus, imag_errors_plus, 'r')
    ]

    # Plot real and/or imaginary parts
    for part, data, errors_minus, errors_plus, color in plot_parts:
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

    if moment_label == "A":
            title = "singlet Sea Quark GPD"
    elif moment_label == "Atilde":
            title = "Axial singlet Sea Quark GPD"
    else:
        print("No title defined.")

    # Set the title based on real_imag
    if real_imag == "real":
        plt.title(f'{title}')
    elif real_imag == "imag":
        plt.title(f'Imaginary Part of {title}')
    elif real_imag == "both":
        plt.title(f'Real and Imaginary Part of {title}')

    # Add vertical lines to separate DGLAP from ERBL region
    plt.axvline(x=eta, linestyle='--')   
    plt.axvline(x=-eta, linestyle='--')

    plt.xlim(x_0, x_1)
    plt.xlabel('x')
    plt.legend(fontsize=10, markerscale=1.5)
    plt.grid(True)
    plt.show()

def plot_non_singlet_quark_gpd(eta, t, mu, Nf=3, moment_type="non_singlet_isovector",moment_label="A",evolution_order="LO",real_imag="real", sampling=True, n_init=os.cpu_count(), n_points=30, x_0=-1, x_1=1, error_bars=True):
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
    check_moment_type_label(moment_type,moment_label)
    # Ensure x_0 < x_1 for a valid range
    if x_0 >= x_1:
        raise ValueError("x_0 must be less than x_1.")

    # Validate real_imag input
    if real_imag not in ("real", "imag", "both"):
        raise ValueError("Invalid option for real_imag. Choose from 'real', 'imag', or 'both'.")

    def compute_result(x, error_type="central"):
        return mellin_barnes_gpd(x, eta, t, mu, Nf, particle="quark",moment_type=moment_type,moment_label=moment_label, evolution_order=evolution_order,real_imag=real_imag, error_type=error_type, n_jobs=1)

    if sampling:

        x_values = np.linspace(x_0, x_1, n_init)

        # Measure time for sampling initial points
        start_time_sampling = time.time()
        results = Parallel(n_jobs=-1)(delayed(compute_result)(x,eta,t,mu) for x in x_values)
        end_time_sampling = time.time()

        # Compute differences and scale intervals
        diffs = np.abs(np.diff(results))
        # Ensure diffs is not zero for power
        diffs += 1e-6
        scaled_intervals = np.power(diffs, 0.5) / np.sum(np.power(diffs, 0.5))
        cumulative_intervals = np.cumsum(np.insert(scaled_intervals, 0, 0))
        cumulative_intervals = x_0 + (x_1 - x_0) * (
            cumulative_intervals - cumulative_intervals[0]) / (
                cumulative_intervals[-1] - cumulative_intervals[0])

        # Output sampling time
        print(f"Time for initial sampling for parameters (eta,t) = ({eta,t}): {end_time_sampling - start_time_sampling:.6f} seconds")

    # Measure time for adaptive grid computation
    start_time_adaptive = time.time()
    if sampling:
        x_values = np.interp(np.linspace(x_0, x_1, n_points), cumulative_intervals, x_values)
    else:
        x_values = np.linspace(x_0, x_1, n_points)

    # Add crossover points
    if eta not in x_values:
        x_values = np.append(x_values,eta)
    if - eta not in x_values and x_0 < 0:
        x_values = np.append(x_values,-eta)
    x_values = np.sort(x_values)

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
        ("real", real_parts, real_errors_minus, real_errors_plus, 'b'),
        ("imag", imag_parts, imag_errors_minus, imag_errors_plus, 'r')
    ]


    # Plot real and/or imaginary parts
    for part, data, errors_minus, errors_plus, color in plot_parts:
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
    if moment_label == "A":
        if moment_type == "non_singlet_isovector":
            title = "Non-singlet Isovector"
        elif moment_type == "non_singlet_isoscalar":
            title = "Non-singlet Isoscalar"
    elif moment_label == "Atilde":
        if moment_type == "non_singlet_isovector":
            title = "Axial Non-singlet Isovector"
        elif moment_type == "non_singlet_isoscalar":
            title = "Axial Non-singlet Isoscalar"
    else:
        print("No title defined.")

    # Set the title based on real_imag
    if real_imag == "real":
        plt.title(f'{title} Quark GPD')
    elif real_imag == "imag":
        plt.title(f'Imaginary Part of {title} Quark GPD')
    elif real_imag == "both":
        plt.title(f'Real and Imaginary Part of {title} Quark GPD')

    # Add vertical lines to separate DGLAP from ERBL region
    plt.axvline(x=eta, linestyle='--')   
    plt.axvline(x=-eta, linestyle='--')

    plt.xlim(x_0, x_1)
    plt.xlabel('x')
    plt.legend(fontsize=10, markerscale=1.5)
    plt.grid(True)
    plt.show()
    
def plot_gluon_gpd(eta, t, mu, Nf=3,moment_label="A",evolution_order="LO", real_imag="real", sampling=True, n_init=os.cpu_count(), n_points=20, x_0=1e-2, x_1=1, error_bars=True):
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
        return mellin_barnes_gpd(x, eta, t, mu, Nf, particle="gluon",moment_type="singlet",moment_label=moment_label, evolution_order=evolution_order,real_imag=real_imag, error_type=error_type,n_jobs=1)

    if sampling:

        x_values = np.linspace(x_0, x_1, n_init)

        # Measure time for sampling initial points
        start_time_sampling = time.time()
        results = Parallel(n_jobs=-1)(delayed(compute_result)(x,eta,t,mu) for x in x_values)
        end_time_sampling = time.time()

        # Compute differences and scale intervals
        diffs = np.abs(np.diff(results))
        # Ensure diffs is not zero for power
        diffs += 1e-6
        scaled_intervals = np.power(diffs, 0.5) / np.sum(np.power(diffs, 0.5))
        cumulative_intervals = np.cumsum(np.insert(scaled_intervals, 0, 0))
        cumulative_intervals = x_0 + (x_1 - x_0) * (
            cumulative_intervals - cumulative_intervals[0]) / (
                cumulative_intervals[-1] - cumulative_intervals[0])

        # Output sampling time
        print(f"Time for initial sampling for parameters (eta,t) = ({eta,t}): {end_time_sampling - start_time_sampling:.6f} seconds")

    # Measure time for adaptive grid computation
    start_time_adaptive = time.time()
    if sampling:
        x_values = np.interp(np.linspace(x_0, x_1, n_points), cumulative_intervals, x_values)
    else:
        x_values = np.linspace(x_0, x_1, n_points)

    # Add crossover points
    if eta not in x_values:
        x_values = np.append(x_values,eta)
    if - eta not in x_values and x_0 < 0:
        x_values = np.append(x_values,-eta)
    x_values = np.sort(x_values)

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
        ("real", real_parts, real_errors_minus, real_errors_plus, 'b'),
        ("imag", imag_parts, imag_errors_minus, imag_errors_plus, 'r')
    ]

    # Plot real and/or imaginary parts
    for part, data, errors_minus, errors_plus, color in plot_parts:
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

    if moment_label == "A":
            title = "singlet Gluon GPD"
    elif moment_label == "Atilde":
            title = "Axial singlet Gluon GPD"
    else:
        print("No title defined.")

    # Set the title based on real_imag
    if real_imag == "real":
        plt.title(f'{title}')
    elif real_imag == "imag":
        plt.title(f'Imaginary Part of {title}')
    elif real_imag == "both":
        plt.title(f'Real and Imaginary Part of {title}')

    # Add vertical lines to separate DGLAP from ERBL region
    plt.axvline(x=eta, linestyle='--')   
    plt.axvline(x=-eta, linestyle='--')

    plt.xlim(x_0, x_1)
    plt.xlabel('x')
    plt.legend(fontsize=10, markerscale=1.5)
    plt.grid(True)
    plt.show()

def plot_gpds(eta_array, t_array, mu_array, colors, Nf=3, particle="quark",gpd_type="non_singlet_isovector",gpd_label="H",evolution_order="LO",sampling=True, n_init=os.cpu_count(), n_points=50, x_0=-1, x_1=1, y_0 = -1e-2, y_1 = 3, 
              error_bars=True, plot_legend = False,write_to_file=True,read_from_file=False):
    """
    Plots a given GPD using the kinematical parameters contained in eta_array, t_array, mu_array
    and corresponding colors with dynamically adjusted x intervals, including error bars.

    The function supports both positive and negative values of parton x though for singlet it defaults to x > 0.

    Options to read/write from/to file system are included.

    Parameters:
    - eta_array (array float): Array containing skewness values
    - t_array (array float): Array containing  t values
    - mu_array (array float): Array containing mu values
    - colors (array str.): Array containing colors for associated values
    - Nf (int, optional): Number of active flavors
    - gpd_type (str. optional): non_singlet_isovector,...
    - gpd_label (str. optional): H, E, ...
    - sampling (Bool, optional): Choose whether to plot using importance sampling  
    - n_init (int, optional): Sampling size, default is available number of CPUs
    - n_points (int, optional): Number of plot points
    - x_0 (float,optional): Lower bound on parton x
    - x_1 (float,optional): Upper bound on parton x
    - y_0 (float, optioanl): LOwer bound on y axis
    - y_1 (float, optional): Upper bound on y axis
    - error_bars (bool): Compute error bars as well
    - plot_legend (bool): Show plot legend
    - write_to_file (Bool, optional): Write data to file system
    - read_from_file (Bool, optional): Read data from file system
    """
    ylabel_map = {
        "non_singlet_isovector": {
            "H": "$H_{u-d}(x,\\eta,t;\\mu)$",
            "Htilde": r"$\widetilde{H}_{u-d}(x,\eta,t;\mu)$",
            "E": "$E_{u-d}(x,\\eta,t;\\mu)$"
        },
        "non_singlet_isoscalar": {
            "H": "$H_{u+d}(x,\\eta,t;\\mu)$",
            "Htilde": r"$\widetilde{H}_{u+d}(x,\eta,t;\mu)$",
            "E": "$E_{u+d}(x,\\eta,t;\\mu)$"
        },
        "singlet": {
            "H": {
                "quark": "$H_{u+d+s}(x,\\eta,t;\\mu)$",
                "gluon": "$H_{g}(x,\\eta,t;\\mu)$"
            },
            "Htilde": {
                "quark": r"$\widetilde{H}_{u+d+s}(x,\eta,t;\mu)$",
                "gluon": r"$\widetilde{H}_{g}(x,\eta,t;\mu)$"
            },
            "E": {
                "quark": "$E_{u+d+s}(x,\\eta,t;\\mu)$",
                "gluon": "$E_{g}(x,\\eta,t;\\mu)$"
            }
        }
    }

    moment_type = gpd_type
    if (gpd_label) in GPD_LABEL_MAP:
        moment_label = GPD_LABEL_MAP[gpd_label]
    else:
        print(f"Key {gpd_label} not found in GPD_LABEL_MAP - abort")
        return
    
    check_particle_type(particle)
    check_moment_type_label(moment_type,moment_label)

    if write_to_file and read_from_file:
        raise ValueError("write_to_file and read_from_file can't simultaneously be True")

    if len({len(eta_array), len(t_array), len(mu_array), len(colors)}) > 1:
        print("Arrays containing kinematical variables have unequal lengths - abort")
        return

    # eta_array = [1e-3,0.1,1/3]
    # t_array = [-1e-3,-0.23,-0.69]
    # mu_array = [2,2,2]
    # colors = ["purple","orange","green"]

    if moment_type == "singlet":
        x_0 = 2e-3

    def compute_result(x, eta,t,mu,error_type="central"):
        return mellin_barnes_gpd(x, eta, t, mu, Nf,particle,moment_type,moment_label, evolution_order=evolution_order, real_imag="real", error_type=error_type,n_jobs=1)

    if read_from_file:
        sampling = False

    # Initialize plot
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    for eta, t, mu, color in zip(eta_array,t_array,mu_array,colors):
        if sampling:

            x_values = np.linspace(x_0, x_1, n_init)

            # Measure time for sampling initial points
            start_time_sampling = time.time()
            results = Parallel(n_jobs=-1)(delayed(compute_result)(x,eta,t,mu) for x in x_values)
            end_time_sampling = time.time()

            # Compute differences and scale intervals
            diffs = np.abs(np.diff(results))
            # Ensure diffs is not zero for power
            diffs += 1e-6
            scaled_intervals = np.power(diffs, 0.5) / np.sum(np.power(diffs, 0.5))
            cumulative_intervals = np.cumsum(np.insert(scaled_intervals, 0, 0))
            cumulative_intervals = x_0 + (x_1 - x_0) * (
                cumulative_intervals - cumulative_intervals[0]) / (
                    cumulative_intervals[-1] - cumulative_intervals[0])

            # Output sampling time
            print(f"Time for initial sampling for parameters (eta,t,mu) = ({eta,t,mu}): {end_time_sampling - start_time_sampling:.6f} seconds")

        # Measure time for adaptive grid computation
        start_time_adaptive = time.time()
        if sampling:
            x_values = np.interp(np.linspace(x_0, x_1, n_points), cumulative_intervals, x_values)
        else:
            x_values = np.linspace(x_0, x_1, n_points)

        # Add crossover points
        if eta not in x_values:
            x_values = np.append(x_values,eta)
        if - eta not in x_values and x_0 < 0:
            x_values = np.append(x_values,-eta)
        x_values = np.sort(x_values)
        if read_from_file:
                x_values = None
                x_values, results = load_gpd_data(eta,t,mu,particle,gpd_type,gpd_label)
                if x_values is None:
                    raise ValueError("No data found on system. Change write_to_file = True")
        else:
            results = Parallel(n_jobs=-1)(delayed(compute_result)(x,eta,t,mu) for x in x_values)

        # Error bar computations
        if error_bars:
            if read_from_file:
                x_plus, results_plus = load_gpd_data(eta,t,mu,particle,gpd_type,gpd_label,"plus")
                x_minus,results_minus = load_gpd_data(eta,t,mu,particle,gpd_type,gpd_label,"minus")
            else:
                results_plus = Parallel(n_jobs=-1)(delayed(compute_result)(x,eta,t,mu, error_type="plus") for x in x_values)
                results_minus = Parallel(n_jobs=-1)(delayed(compute_result)(x,eta,t,mu, error_type="minus") for x in x_values)
        else:
            results_plus = results
            results_minus = results
        end_time_adaptive = time.time()

        # Output plot generation time
        print(f"Time for plot computation for parameters (eta,t,mu) = ({eta,t,mu}): {end_time_adaptive - start_time_adaptive:.6f} seconds")

        if error_bars:
            ax.plot(x_values, results,label=(f"$\\eta={eta:.2f}$, "
                    f"$t={t:.2f} \\text{{ GeV}}^2$"), color=color)
            ax.fill_between(x_values,results_minus,results_plus,color=color,alpha=.2)
        else:
            ax.plot(x_values, results,label=(f"$\\eta={eta:.2f}$, "
                    f"$t={t:.2f} \\text{{ GeV}}^2$"), color=color)
        # Add vertical lines to separate DGLAP from ERBL region
        ax.axvline(x=eta, linestyle='--', color = color)   
        ax.axvline(x=-eta, linestyle='--', color = color)

        if write_to_file:
            save_gpd_data(x_values,eta,t,mu,results,particle,gpd_type,gpd_label)
            save_gpd_data(x_values,eta,t,mu,results_plus,particle,gpd_type,gpd_label,"plus")
            save_gpd_data(x_values,eta,t,mu,results_minus,particle,gpd_type,gpd_label,"minus")

    ax.set_xlim(x_0, x_1)
    ax.set_ylim(y_0,y_1)
    ax.set_xlabel('x')

    if moment_type == "singlet":
        ylabel = ylabel_map[moment_type][gpd_label][particle]
    else:
        ylabel = ylabel_map[moment_type][gpd_label]    

    ax.set_ylabel(ylabel, fontsize=14)
    if plot_legend:
        ax.legend(fontsize=10, markerscale=1.5)
    ax.grid(True)
    # Export
    FILE_PATH = PLOT_PATH +  gpd_type + "_" + particle + "_GPD_" + gpd_label +".pdf"
    fig.savefig(FILE_PATH,format="pdf",bbox_inches="tight",dpi=600)
    print(f"File saved to {FILE_PATH}")

