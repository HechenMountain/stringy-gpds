# Dependencies
import numpy as np
import os
import csv

# mpmath precision set in config
from .config import mp

from scipy.integrate import quad, odeint, trapezoid, fixed_quad
from scipy.interpolate import interp1d, RectBivariateSpline

from itertools import product
from joblib import Parallel, delayed
from tqdm import tqdm

from .mstw_pdf import get_alpha_s

from . import config as cfg
from . import helpers as hp
from . import regge as reg
from . import special as sp
from . import adim

#############################################
####   Currently enforced assumptions    ####
#############################################
# singlet_moment for B GPD set to zero      #
# Normalizations of isoscalar_moment        #
# ubar = dbar                               #
# Delta_u_bar = Delta_s_bar = Delta_s       #
#############################################

#############################################
# NLO anomalous dimensions are interpolated #
# but LO not. Though the code is there to   #
# also use interpolation for those          #
#############################################

#####################################
### Input for Evolution Equations ###
#####################################

def evolve_alpha_s(mu, evolution_order="nlo"):
    """
    Evolve the strong coupling constant alpha_s = g**2 / (4 pi) from an input scale μ_in to another scale μ.

    Note that the MSTW lo best fit determines alpha_s(mu**2 = 1 GeV**2) \approx 0.68183, which differs from the current world average.

    Parameters
    ----------
    mu : float
        The resolution scale in GeV
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"

    Returns
    -------
    float
        The value of the strong coupling constant alpha_s at the given scale.
    """
    # Set parameters
    mu_R = 1 # 1 GeV from MSTW and AAC PDFs
    # Extract value of alpha_S at the renormalization point of mu_R**2 = 1 GeV**2
    alpha_s_in = get_alpha_s(evolution_order)

    if mu_R == mu:
        return alpha_s_in
    if evolution_order == "lo":
        beta_1 = 0
    else:
        beta_1 = cfg.BETA_1

    # Define the differential equation
    def beta_function(alpha_s, Q2):
        # alpha_s is the value of alpha_s/(4*pi) and ln_Q2 is the logarithmic scale
        d_alpha_s = (cfg.BETA_0 * alpha_s**2/(4*np.pi) + beta_1 * alpha_s**3/(4*np.pi)**2)/Q2
        return d_alpha_s

    if evolution_order == "lo":
        log_term = np.log(mu**2 / mu_R**2)
        denominator = 1 - (alpha_s_in / (4 * np.pi)) * cfg.BETA_0 * log_term
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

##############################
### Moment parametrization ###
##############################

def non_singlet_isovector_moment(j,eta,t, moment_label="A", evolution_order="nlo",error_type="central"):
    """
    Compute the non-singlet isovector moment for given conformal spin and kinematics. Currently skewness independent

    Parameters
    ----------
    j : complex
        Conformal spin.
    eta : float
        Skewness parameter.
    t : float
        Mandelstam variable t 
    moment_label : str, optional
        A(tilde), B(tilde) depending on H(tilde) or E(tilde) GPD etc. Default is "A".
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"

    Returns
    -------
    complex or float
        The value of the moment
    """
    # Check type
    hp.check_error_type(error_type)
    hp.check_evolution_order(evolution_order)
    hp.check_moment_type_label("non_singlet_isovector",moment_label)

    alpha_prime = hp.get_regge_slope("non_singlet_isovector",moment_label,evolution_order)

    if moment_label in ["A","B"]:
        uv, uv_error = reg.uv_pdf_regge(j,eta,alpha_prime,t,evolution_order,error_type)
        dv, dv_error = reg.dv_pdf_regge(j,eta,alpha_prime,t,evolution_order,error_type)
    elif moment_label =="Atilde":
       uv, uv_error = reg.polarized_uv_pdf_regge(j,eta,alpha_prime,t,evolution_order,error_type)
       dv, dv_error = reg.polarized_dv_pdf_regge(j,eta,alpha_prime,t,evolution_order,error_type)

    norm = hp.get_moment_normalizations("non_singlet_isovector",moment_label,evolution_order)
    sum_squared = uv_error**2+dv_error**2
    error = abs(mp.sqrt(sum_squared))
    error = hp.error_sign(error,error_type)
    result = norm * ( uv - dv + error )

    return result

def non_singlet_isoscalar_moment(j,eta,t, moment_label="A", evolution_order = "nlo", error_type="central"):
    """
    Compute the non-singlet isoscalar moment for given conformal spin and kinematics. Currently skewness independent

    Parameters
    ----------
    j : complex
        Conformal spin.
    eta : float
        Skewness parameter.
    t : float
        Mandelstam variable t 
    moment_label : str, optional
        A(tilde), B(tilde) depending on H(tilde) or E(tilde) GPD etc. Default is "A".
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"

    Returns
    -------
    complex or float
        The value of the moment
    """

    # Check type
    hp.check_error_type(error_type)
    hp.check_moment_type_label("non_singlet_isoscalar",moment_label)

    alpha_prime = hp.get_regge_slope("non_singlet_isoscalar",moment_label,evolution_order)

    if moment_label in ["A","B"]:
        uv, uv_error = reg.uv_pdf_regge(j,eta,alpha_prime,t,evolution_order,error_type)
        dv, dv_error = reg.dv_pdf_regge(j,eta,alpha_prime,t,evolution_order,error_type)
    elif moment_label =="Atilde":
        uv, uv_error = reg.polarized_uv_pdf_regge(j,eta,alpha_prime,t,evolution_order,error_type)
        dv, dv_error = reg.polarized_dv_pdf_regge(j,eta,alpha_prime,t,evolution_order,error_type)

    norm = hp.get_moment_normalizations("non_singlet_isoscalar",moment_label,evolution_order)
    sum_squared = uv_error**2+dv_error**2
    error = abs(mp.sqrt(sum_squared))
    error = hp.error_sign(error,error_type)
    result = norm * ( uv + dv + error )

    return result

def d_hat(j,eta,t):
    """
    Skewness dependent kinematical factor for Reggeized spin-j exchanges.

    Parameters
    ----------
    j : complex
        conformal spin
    eta : float
        skewness
    t : float
        Mandelstam t (< 0)

    Returns
    -------
    float
        The value of d_hat (= 1 for eta == 0)
    """

    m_N = 0.93827 # Nucleon mass in GeV
    if eta == 0:
        return 1
    # For large imaginary parts the convergence is very slow
    # so we cut off at too large values
    if abs(j.imag) > 20:
        j = mp.mpc(j.real, 20 * mp.sign(j.imag))
    # Check whether call is safe
    dble = 2 * j.real  
    if dble % 2 == 1:
        raise ValueError(f"d_hat called at pole")
    t = -1e-12 if t == 0 else t

    if mp.im(j) < 0:
        j = mp.conj(j)
        result = mp.hyp2f1(-j/2, -(j-1)/2, 1/2 - j, - 4 * m_N**2/t * eta**2)
        result = mp.conj(result)
    else:
        result = mp.hyp2f1(-j/2, -(j-1)/2, 1/2 - j, - 4 * m_N**2/t * eta**2)
    return result
    
def quark_singlet_regge_A(j,eta,t, alpha_prime_ud=0.891, moment_label="A", evolution_order ="nlo", error_type="central"):
    """
    Compute the reggeized A term of the quark PDFs for given conformal spin and kinematics.

    Parameters
    ----------
    j : complex
        Conformal spin.
    eta : float
        Skewness parameter.
    t : float
        Mandelstam variable t 
    alpha_prime_ud : float
        Regge slope
    moment_label : str, optional
        A(tilde), B(tilde) depending on H(tilde) or E(tilde) GPD etc. Default is "A".
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"

    Returns
    -------
    complex or float
        The value of the moment
    """
    if moment_label in ["A","B"]:
        uv, uv_error = reg.uv_pdf_regge(j,eta,alpha_prime_ud,t,evolution_order,error_type) 
        dv, dv_error = reg.dv_pdf_regge(j,eta,alpha_prime_ud,t,evolution_order,error_type)
        Spdf, Spdf_error = reg.S_pdf_regge(j,eta,alpha_prime_ud,t,evolution_order,error_type)
        # To Do
        Delta, Delta_error = reg.Delta_pdf_regge(j,eta,alpha_prime_ud,t,evolution_order,error_type)
        s_plus, s_plus_error = reg.s_plus_pdf_regge(j,eta,alpha_prime_ud,t,evolution_order,error_type)
    elif moment_label == "Atilde":
        uv, uv_error = reg.polarized_uv_pdf_regge(j,eta,alpha_prime_ud,t,evolution_order,error_type) 
        dv, dv_error = reg.polarized_dv_pdf_regge(j,eta,alpha_prime_ud,t,evolution_order,error_type)
        # To do
        #Delta = polarized_Delta_pdf_regge(j,eta,alpha_prime_ud,t,error_type)
        Spdf, Spdf_error = reg.polarized_S_pdf_regge(j,eta,alpha_prime_ud,t,evolution_order,error_type)
        # To do
        #s_plus = polarized_s_plus_pdf_regge(j,eta,alpha_prime_ud,t,error_type)
    else:
        raise ValueError(f"Unsupported moment label {moment_label}")
    if cfg.N_F == 3 or cfg.N_F == 4:
        sum_squared = uv_error**2+dv_error**2+Spdf_error**2
        result = uv + dv + Spdf
    elif cfg.N_F == 2:
        sum_squared = uv_error**2+dv_error**2+Spdf_error**2+s_plus_error**2
        result = uv + dv + Spdf - s_plus
    elif cfg.N_F == 1:
        sum_squared = .5*(4*uv_error**2+Spdf_error**2+s_plus_error**2+4*Delta_error**2)
        result = .5*(Spdf-s_plus+2*uv-2*Delta)
    else :
        raise ValueError("Currently only (integer) 1 <= cfg.N_F <= 3 supported")
    error = abs(mp.sqrt(sum_squared))
    return result, error
    
def quark_singlet_regge_D(j,eta,t,  alpha_prime_ud=0.891,alpha_prime_s=1.828, moment_label="A",evolution_order="nlo", error_type="central"):
    """
    Compute the reggeized A term of the quark PDFs for given conformal spin and kinematics.

    Parameters
    ----------
    j : complex
        Conformal spin.
    eta : float
        Skewness parameter.
    t : float
        Mandelstam variable t 
    alpha_prime_ud : float
        Regge slope
    alpha_prime_s : float
        D-term Regge slope
    moment_label : str, optional
        A(tilde), B(tilde) depending on H(tilde) or E(tilde) GPD etc. Default is "A".
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"

    Returns
    -------
    complex or float
        The value of the moment
    """
    if eta == 0:
        return 0, 0
    if moment_label in ["A","B"]:
        uv, uv_error = reg.uv_pdf_regge(j,eta,alpha_prime_ud,t,evolution_order,error_type) 
        dv, dv_error = reg.dv_pdf_regge(j,eta,alpha_prime_ud,t,evolution_order,error_type)
        Delta, Delta_error = reg.Delta_pdf_regge(j,eta,alpha_prime_ud,t,evolution_order,error_type)
        Sv, Sv_error = reg.S_pdf_regge(j,eta,alpha_prime_ud,t,evolution_order,error_type)
        s_plus, s_plus_error = reg.s_plus_pdf_regge(j,eta,alpha_prime_ud,t,evolution_order,error_type)

        uv_s, uv_s_error = reg.uv_pdf_regge(j,eta,alpha_prime_s,t,evolution_order,error_type) 
        dv_s, dv_s_error = reg.dv_pdf_regge(j,eta,alpha_prime_s,t,evolution_order,error_type)
        Sv_s, Sv_s_error = reg.S_pdf_regge(j,eta,alpha_prime_s,t,evolution_order,error_type)
        s_plus_s, s_plus_s_error = reg.s_plus_pdf_regge(j,eta,alpha_prime_s,t,evolution_order,error_type)
        Delta_s, Delta_s_error = reg.Delta_pdf_regge(j,eta,alpha_prime_s,t,evolution_order,error_type)

    elif moment_label == "Atilde":
        uv, uv_error  = reg.polarized_uv_pdf_regge(j,eta,alpha_prime_ud,t,evolution_order,error_type) 
        dv, dv_error = reg.polarized_dv_pdf_regge(j,eta,alpha_prime_ud,t,evolution_order,error_type)
        # To do
        #Delta = polarized_Delta_pdf_regge(j,eta,alpha_prime_ud,t,error_type)
        Sv, Sv_error = reg.polarized_S_pdf_regge(j,eta,alpha_prime_ud,t,evolution_order,error_type)
        # To Do
        #s_plus = polarized_s_plus_pdf_regge(j,eta,alpha_prime_ud,t,error_type)
        uv_s, uv_s_error = reg.polarized_uv_pdf_regge(j,eta,alpha_prime_s,t,evolution_order,error_type) 
        dv_s, dv_s_error = reg.polarized_dv_pdf_regge(j,eta,alpha_prime_s,t,evolution_order,error_type)
        Sv_s, Sv_s_error = reg.polarized_S_pdf_regge(j,eta,alpha_prime_s,t,evolution_order,error_type)
        # To do
        # s_plus_s = polarized_s_plus_pdf_regge(j,eta,alpha_prime_s,t,error_type)
        # Delta_s = polarized_Delta_pdf_regge(j,eta,alpha_prime_s,t,error_type)
    else:
        raise ValueError(f"Unsupported moment label {moment_label}")

    if cfg.N_F == 3 or cfg.N_F == 4:
        term_1 = uv + dv + Sv
        sum_squared_1 = uv_error**2+dv_error**2+Sv_error**2
        term_2 = uv_s + dv_s + Sv_s 
        sum_squared_2 = uv_s_error**2+dv_s_error**2+Sv_s_error**2
    elif cfg.N_F == 2:
        term_1 = uv + dv + Sv - s_plus
        sum_squared_1 = uv_error**2+dv_error**2+Sv_error**2+s_plus_error**2
        term_2 = uv_s + dv_s + Sv_s - s_plus_s
        sum_squared_2 = uv_s_error**2+dv_s_error**2+Sv_s_error**2+s_plus_s_error**2
    elif cfg.N_F == 1:
        term_1 = .5*(Sv-s_plus+2*uv-2*Delta)
        sum_squared_1 = .5*(4*uv_error**2+Sv_error**2+s_plus_error**2+4*Delta_error**2)
        term_2 = .5*(Sv_s-s_plus_s+2*uv_s-2*Delta_s)
        sum_squared_2 = .5*(4*uv_s_error**2+Sv_s_error**2+s_plus_s_error**2+4*Delta_s_error**2)
    else :
        raise ValueError("Currently only (integer) 1 <= cfg.N_F <= 3 supported")
    sum_squared = mp.sqrt(sum_squared_1**2 + sum_squared_2**2)
    error = abs(mp.sqrt(sum_squared))
    error = (d_hat(j,eta,t)-1)*error
    result = (d_hat(j,eta,t)-1)*(term_1-term_2)
    return result, error

def quark_singlet_regge(j,eta,t,moment_label="A",evolution_order="nlo",error_type="central"):
    """
    Compute the reggeized A and D term of the quark PDFs for given conformal spin and kinematics. The Regge slopes
    are defined in the config file.

    Parameters
    ----------
    j : complex
        Conformal spin.
    eta : float
        Skewness parameter.
    t : float
        Mandelstam variable t 
    moment_label : str, optional
        A(tilde), B(tilde) depending on H(tilde) or E(tilde) GPD etc. Default is "A".
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"

    Returns
    -------
    complex or float
        The value of the moment
    """
    # Check type
    hp.check_error_type(error_type)
    hp.check_moment_type_label("singlet",moment_label)
    hp.check_evolution_order(evolution_order)

    if moment_label == "B":
        prf = -1
    else:
        prf = +1

    alpha_prime_ud, alpha_prime_s, _, _ = hp.get_regge_slope("singlet",moment_label,evolution_order)
    norm_A, norm_D, _, _ = hp.get_moment_normalizations("singlet",moment_label,evolution_order)

    term_1, error_1 = quark_singlet_regge_A(j,eta,t,alpha_prime_ud,moment_label,evolution_order,error_type)
    if eta == 0:
        result = norm_A * term_1
        error = norm_A * error_1
    else :
        term_2, error_2 = quark_singlet_regge_D(j,eta,t,alpha_prime_ud,alpha_prime_s,moment_label,evolution_order,error_type)
        sum_squared = norm_A**2 * error_1**2 + norm_D**2 * error_2**2
        error = abs(mp.sqrt(sum_squared))
        result = norm_A * term_1 + norm_D * prf * term_2
    return result, error

def gluon_singlet_regge_A(j,eta,t, alpha_prime_T = 0.627,moment_label="A", evolution_order="nlo",error_type="central"):
    """
    Compute the reggeized A term of the gluon PDFs for given conformal spin and kinematics.

    Parameters
    ----------
    j : complex
        Conformal spin.
    eta : float
        Skewness parameter.
    t : float
        Mandelstam variable t 
    alpha_prime_T : float
        Regge slope
    moment_label : str, optional
        A(tilde), B(tilde) depending on H(tilde) or E(tilde) GPD etc. Default is "A".
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"

    Returns
    -------
    complex or float
        The value of the moment
    """
    if moment_label == "A":
        result, error = reg.gluon_pdf_regge(j,eta,alpha_prime_T,t,evolution_order,error_type)
    elif moment_label =="Atilde":
        result, error = reg.polarized_gluon_pdf_regge(j,eta,alpha_prime_T,t,evolution_order,error_type)
    else:
        raise ValueError(f"Unsupported moment label {moment_label}")
    return result, error

def gluon_singlet_regge_D(j,eta,t, alpha_prime_T = 0.627, alpha_prime_S = 4.277,moment_label="A",evolution_order="nlo", error_type="central"):
    """
    Compute the reggeized D term of the quark PDFs for given conformal spin and kinematics.

    Parameters
    ----------
    j : complex
        Conformal spin.
    eta : float
        Skewness parameter.
    t : float
        Mandelstam variable t 
    alpha_prime_ud : float
        Regge slope
    alpha_prime_S : float
        D-term Regge slope
    moment_label : str, optional
        A(tilde), B(tilde) depending on H(tilde) or E(tilde) GPD etc. Default is "A".
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"

    Returns
    -------
    complex or float
        The value of the moment
    """
    # Check type
    hp.check_error_type(error_type)
    hp.check_moment_type_label("singlet",moment_label)
    if eta == 0:
        return 0, 0 
    else :
        term_1 = (d_hat(j,eta,t)-1)
        term_2, error_2 = gluon_singlet_regge_A(j,eta,t,alpha_prime_T,moment_label,evolution_order,error_type)
        if moment_label == "A":
            term_3, error_3 = reg.gluon_pdf_regge(j,eta,t,alpha_prime_S,evolution_order,error_type)
        elif moment_label =="Atilde":
            term_3, error_3 = reg.polarized_gluon_pdf_regge(j,eta,t,alpha_prime_S,evolution_order,error_type)
        else:
            raise ValueError(f"Unsupported moment label {moment_label}")
        sum_squared = error_2**2+error_3**2
        error = abs(mp.sqrt(sum_squared))
        error = term_1 * error
        result = term_1 * (term_2-term_3)
        return result, error
    
def gluon_singlet_regge(j,eta,t,moment_label="A",evolution_order="nlo",error_type="central"):
    """
    Compute the reggeized A and D term of the quark PDFs for given conformal spin and kinematics.

    Parameters
    ----------
    j : complex
        Conformal spin.
    eta : float
        Skewness parameter.
    t : float
        Mandelstam variable t 
    moment_label : str, optional
        A(tilde), B(tilde) depending on H(tilde) or E(tilde) GPD etc. Default is "A".
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"

    Returns
    -------
    complex or float
        The value of the moment
    """
    # Check type
    hp.check_error_type(error_type)
    hp.check_moment_type_label("singlet",moment_label)

    if moment_label == "B":
        prf = -1
    else:
        prf = +1

    _, _, alpha_prime_T, alpha_prime_S = hp.get_regge_slope("singlet",moment_label,evolution_order)
    _, _, norm_A, norm_D = hp.get_moment_normalizations("singlet",moment_label,evolution_order)

    term_1, error_1 = gluon_singlet_regge_A(j,eta,t,alpha_prime_T,moment_label,evolution_order,error_type)
    if eta == 0:
        result = norm_A * term_1
        error = norm_A * error_1
    else :
        term_2, error_2 = gluon_singlet_regge_D(j,eta,t,alpha_prime_T,alpha_prime_S,moment_label,evolution_order,error_type)
        sum_squared = norm_A**2 * error_1**2 + norm_D**2 * error_2**2
        error = abs(mp.sqrt(sum_squared))
        result = norm_A * term_1 + norm_D * prf * term_2
    return result, error

def singlet_moment(j,eta,t,moment_label="A",solution="+",evolution_order="nlo",error_type="central",interpolation=True):
    """
    Compute the diagonal combination of quark and gluon singlet moments for given conformal spin and kinematics.

    Returns 0 if the moment_label = "B", in accordance with holography and quark model considerations. 
    Otherwise it returns the diagonal combination of quark + gluon moment. Error for singlet_moment at j = 1
    for solution "-" unreliable because of pole in gamma. Better reconstruct evolved moment from GPD.

    Parameters
    ----------
    j : complex
        Conformal spin.
    eta : float
        Skewness parameter.
    t : float
        Mandelstam variable t.
    moment_label : str, optional
        A(tilde), B(tilde) depending on H(tilde) or E(tilde) GPD etc. Default is "A".
    solution : string
        + or - depending on which solution to pick
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"

    Returns
    -------
    complex or float
        The value of the moment.
    """
    if moment_label == "B":
        return 0, 0
    # Check type
    hp.check_error_type(error_type)

    # Switch sign
    if solution == "+":
        solution = "-"
    elif solution == "-":
        solution = "+"
    else:
        raise ValueError("Invalid solution type. Use '+' or '-'.")
    
    evolve_type = hp.get_evolve_type(moment_label)
    # Eigenvalue of anomalous dimension matrix
    index  = 0 if solution == "+" else 1
    ga_pm = adim.gamma_pm(j-1,evolve_type,solution,interpolation=interpolation)[index]

    quark_prf = .5 
    quark_in, quark_in_error = quark_singlet_regge(j,eta,t,moment_label,evolution_order,error_type)
    # Note: j/6 already included in adim.gamma_qg and adim.gamma_qg definitions
    gluon_prf = .5 * (adim.gamma_qg(j-1,evolve_type,"lo",interpolation=interpolation)/
                    (adim.gamma_qq(j-1,"singlet",evolve_type,"lo",interpolation=interpolation)-ga_pm))
    gluon_in, gluon_in_error = gluon_singlet_regge(j,eta,t,moment_label,evolution_order,error_type)
    sum_squared = quark_prf**2 * quark_in_error**2 + gluon_prf**2*gluon_in_error**2
    error = abs(mp.sqrt(sum_squared))
    result = quark_prf * quark_in + gluon_prf * gluon_in
    return result, error

################################
##### Evolution Equations ######
################################

evolve_moment_interpolation = {}
if cfg.INTERPOLATE_MOMENTS:
    for particle,moment_type, moment_label, evolution_order in product(
        cfg.PARTICLES, cfg.MOMENTS, cfg.LABELS, cfg.ORDERS):
        if particle == "gluon" and moment_type != "singlet":
            continue
        params = {
            # Dummy to not regenerate the
            # mixed input singlet moment
            "solution": ".",
            "particle": particle,
            "moment_type": moment_type,
            "moment_label": moment_label,
            "evolution_order": evolution_order,
            "error_type": "central"
        }
        selected_triples = [
            (eta, t, mu)
            for eta, t, mu in zip(cfg.ETA_ARRAY, cfg.T_ARRAY, cfg.MU_ARRAY)
        ]
        evolve_moment_interpolation[(particle,moment_type,moment_label,evolution_order,"central")] = [
            hp.build_moment_interpolator(eta, t, mu, **params)
            for eta, t, mu in selected_triples
        ]

@hp.mpmath_vectorize
def evolve_conformal_moment(j,eta,t,mu,A0=1,particle="quark",moment_type="non_singlet_isovector",moment_label ="A", evolution_order = "nlo", error_type = "central",interpolation=True):
    """
    Evolve the conformal moment F_{j}^{+-} from some input scale mu_in to some other scale mu.

    Parameters
    ----------
    j : float
        Conformal spin.
    eta : float
        Skewness parameter.
    t : float
        Mandelstam t.
    mu : float
        Resolution scale.
    A0 : float, optional
        Normalization factor (default A0 = 1).
    particle : str, optional
        "quark" or "gluon". Default is "quark".
    moment_type : str, optional
        "non_singlet_isovector", "non_singlet_isoscalar", or "singlet".
    moment_label : str, optional
        A(tilde), B(tilde) depending on H(tilde) or E(tilde) GPD etc. Default is "A".
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"
    interpolation : bool, optional
        Use interpolated values for anomalous dimensions.

    Returns
    -------
    float or mpc
        The value of the evolved conformal moment at scale mu.

    Note
    ----
    Decorated using mpmath_vectorize from helpers.py for vectorized calls
    """

    hp.check_particle_type(particle)
    hp.check_moment_type_label(moment_type,moment_label)
    hp.check_error_type(error_type)
    hp.check_evolution_order(evolution_order)
    if particle == "gluon" and moment_type != "singlet":
        raise ValueError("Gluon is only singlet")
    
    if cfg.INTERPOLATE_MOMENTS and j.imag != 0:
        selected_triples = [
            (eta_, t_, mu_)
            for eta_, t_, mu_ in zip(cfg.ETA_ARRAY, cfg.T_ARRAY, cfg.MU_ARRAY)
        ]
        index = selected_triples.index((eta, t, mu))
        key = (particle,moment_type,moment_label,evolution_order,error_type)
        interp = evolve_moment_interpolation[key][index]
        # Interpolator generated with fixed j_re = get_j_base(...)
        return interp(j)

    if error_type != "central" and j.imag != 0:
        raise ValueError("Error propagation for complex spin-j not supported")

    # Extract fixed quantities
    alpha_s_in = get_alpha_s(evolution_order)
    alpha_s_evolved = evolve_alpha_s(mu,evolution_order)

    evolve_type = hp.get_evolve_type(moment_label)

    if moment_type == "non_singlet_isovector":
        moment_in = non_singlet_isovector_moment(j,eta,t,moment_label,evolution_order,error_type)
    elif moment_type == "non_singlet_isoscalar":
        moment_in = non_singlet_isoscalar_moment(j,eta,t,moment_label,evolution_order,error_type) 
    elif moment_type == "singlet":
        # Roots  of lo anomalous dimensions
        ga_p, ga_m = adim.gamma_pm(j-1,evolve_type,interpolation=interpolation)
        moment_in_p, error_p = singlet_moment(j,eta,t,  moment_label,"+",evolution_order,error_type,interpolation=interpolation)
        moment_in_m, error_m = singlet_moment(j,eta,t,  moment_label,"-",evolution_order,error_type,interpolation=interpolation)
        ga_gq = adim.gamma_gq(j-1, evolve_type,"lo",interpolation=interpolation)
        ga_qg = adim.gamma_qg(j-1, evolve_type,"lo",interpolation=interpolation)
        if evolution_order != "lo":
            ga_gg = adim.gamma_gg(j-1,evolve_type,"lo",interpolation=interpolation)
            r_qq = adim.R_qq(j-1,evolve_type,interpolation=interpolation)
            r_qg = adim.R_qg(j-1,evolve_type,interpolation=interpolation)
            r_gq = adim.R_gq(j-1,evolve_type,interpolation=interpolation)
            r_gg = adim.R_gg(j-1,evolve_type,interpolation=interpolation) 
    # Precompute alpha_s fraction:
    alpha_frac  = (alpha_s_in/alpha_s_evolved)    

    ga_qq = adim.gamma_qq(j-1,moment_type,evolve_type,evolution_order="lo",interpolation=interpolation)

    # Functions appearing in evolution
    def get_gammas(solution):
        # switch + <-> - when necessary
        if solution == "+":
            return ga_p, ga_m
        elif solution == "-": 
            return ga_m, ga_p
        else:
            raise ValueError(f"Wrong solution type: {solution}")

    def E_non_singlet_lo():
        return alpha_frac**(ga_qq/cfg.BETA_0)

    def E_non_singlet_nlo(j):
        ga_qq = adim.gamma_qq(j,moment_type,evolve_type,"lo",interpolation=interpolation)
        ga_qq_nlo = adim.gamma_qq(j,moment_type,evolve_type,"nlo",interpolation=interpolation)
        result = alpha_frac**(ga_qq/cfg.BETA_0) * (1 + (.5 * cfg.BETA_1/cfg.BETA_0**2 *  ga_qq - ga_qq_nlo/cfg.BETA_0) * \
                                      (alpha_s_evolved - alpha_s_in)/(2*np.pi)
                                      )
        return result
    def B_non_singlet_nlo(k):
        gamma_term = (ga_qq - adim.gamma_qq(k,moment_type,evolve_type,"lo",interpolation=interpolation) + cfg.BETA_0)
        ga_nd = adim.gamma_qq_nd(j-1,k,evolve_type,evolution_order,interpolation=interpolation)
        result = alpha_s_evolved/(2*np.pi) * ga_nd/gamma_term * (
            1 - alpha_frac**(gamma_term/cfg.BETA_0)
        )
        if j - 1 == k:
            result += 1
        return result

    def EB_non_singlet_nlo(k):
        # Combinede function to call in fractional_finite_sum
        if moment_type == "non_singlet_isovector":
            moment_k  = non_singlet_isovector_moment(k,eta,t,moment_label,evolution_order,error_type)
        else:
            moment_k  = non_singlet_isoscalar_moment(k,eta,t,moment_label,evolution_order,error_type)
        non_diagonal_terms = eta**(j - k) * E_non_singlet_nlo(k-1) * B_non_singlet_nlo(k-1)
        non_diagonal_terms = non_diagonal_terms * moment_k
        return non_diagonal_terms
    
    def A_lo_quark(solution):
        # The switch also takes care of the relative minus sign
        ga_p, ga_m = get_gammas(solution)
        result = (ga_qq - ga_m)/(ga_p - ga_m) * alpha_frac**(ga_p/cfg.BETA_0) * 2
        return result
    
    def A_lo_gluon(solution):
        ga_p, ga_m = get_gammas(solution)
        result = ga_gq/(ga_p - ga_m) * alpha_frac**(ga_p/cfg.BETA_0) * 2
        return result

    def A_quark_nlo(solution):
        ga_p, ga_m = get_gammas(solution)
        term1 = - (alpha_s_evolved - alpha_s_in)/(2*mp.pi)/cfg.BETA_0 * alpha_frac**(ga_p/cfg.BETA_0) / \
                (ga_p - ga_m)**2 * (2)
        term2 = (ga_qq - ga_m) * (r_qq * (ga_qq-ga_m) + r_qg * ga_gq)
        term3 = ga_qg * (r_gq * (ga_qq - ga_m) + r_gg * ga_gq)
        result = term1 * (term2 + term3)
        return result
    
    def B_quark_nlo(solution):
        ga_p, ga_m = get_gammas(solution)
        term1 = alpha_s_evolved/(2*mp.pi)/(ga_m - ga_p + cfg.BETA_0) * 2 / (ga_p - ga_m)**2
        term2 = (1 - alpha_frac**((ga_m - ga_p + cfg.BETA_0)/cfg.BETA_0)) * alpha_frac**(ga_p/cfg.BETA_0)
        term3 = ((ga_qq - ga_p) * (r_qq * (ga_qq - ga_m) + r_qg * ga_gq) + ga_qg * (r_gq * (ga_qq - ga_m) + r_gg * ga_gq))
        result = term1 * term2 * term3
        return result

    def A_gluon_nlo(solution):
        ga_p, ga_m = get_gammas(solution)
        term1 = - (alpha_s_evolved - alpha_s_in)/(2*mp.pi)/cfg.BETA_0 * alpha_frac**(ga_p/cfg.BETA_0) / \
                (ga_p - ga_m)**2 * (2)
        term2 = ga_gq * (r_qq * (ga_qq-ga_m) + r_qg * ga_gq)
        term3 = (ga_gg - ga_m) * (r_gq * (ga_qq - ga_m) + r_gg * ga_gq)
        result = term1 * (term2 + term3)
        return result

    def B_gluon_nlo(solution):
        ga_p, ga_m = get_gammas(solution)
        term1 = alpha_s_evolved/(2*mp.pi)/(ga_m - ga_p + cfg.BETA_0) * 2 / (ga_p - ga_m)**2
        term2 = (1 - alpha_frac**(((ga_m - ga_p + cfg.BETA_0)/cfg.BETA_0))) * alpha_frac**(ga_p/cfg.BETA_0)
        term3 = (ga_gq  * (r_qq * (ga_qq - ga_m) + r_qg * ga_gq) + (ga_gg - ga_p) * (r_gq * (ga_qq - ga_m) + r_gg * ga_gq) )
        result = term1 * term2 * term3
        return result

    def prf_T_nlo(k):
        ga_j_p, ga_j_m = ga_p, ga_m
        ga_k_p, ga_k_m = adim.gamma_pm(k-1,evolve_type,interpolation=interpolation)
        alpha_term = alpha_s_evolved/(2*mp.pi)
        ga_1 = ga_j_p - ga_k_p + cfg.BETA_0
        ga_2 = ga_j_p - ga_k_m + cfg.BETA_0
        ga_3 = ga_j_m - ga_k_p + cfg.BETA_0
        ga_4 = ga_j_m - ga_k_m + cfg.BETA_0
        ga_kk_jj = (ga_k_p - ga_k_m)*(ga_j_p - ga_j_m)
        prf_T_1 = - alpha_term/ga_1 * (1 - alpha_frac**(ga_1/cfg.BETA_0))/ga_kk_jj
        prf_T_2 = - alpha_term/ga_2 * (1 - alpha_frac**(ga_2/cfg.BETA_0))/ga_kk_jj
        prf_T_3 = - alpha_term/ga_3 * (1 - alpha_frac**(ga_3/cfg.BETA_0))/ga_kk_jj
        prf_T_4 = - alpha_term/ga_4 * (1 - alpha_frac**(ga_4/cfg.BETA_0))/ga_kk_jj

        return prf_T_1, prf_T_2, prf_T_3, prf_T_4

    # T1 and T3 go with "+" solution, T2 and T4 go with "-" solution
    def T_quark_nlo(k):
        # Note T = 0 for j=k
        ga_j_p, ga_j_m = ga_p, ga_m
    
        ga_k_p, ga_k_m = adim.gamma_pm(k-1,evolve_type,interpolation=interpolation)
        ga_qq_k = adim.gamma_qq(k-1,evolution_order="lo",interpolation=interpolation)
        ga_gq_k = adim.gamma_gq(k-1, evolve_type,"lo",interpolation=interpolation)
        ga_qq_nd = adim.gamma_qq_nd(j-1,k-1,evolve_type,"nlo",interpolation=interpolation)
        ga_qg_nd = adim.gamma_qg_nd(j-1,k-1,evolve_type,"nlo",interpolation=interpolation)
        ga_gq_nd = adim.gamma_gq_nd(j-1,k-1,evolve_type,"nlo",interpolation=interpolation)
        ga_gg_nd = adim.gamma_gg_nd(j-1,k-1,evolve_type,"nlo",interpolation=interpolation)

        prf_T_1, prf_T_2, prf_T_3, prf_T_4 = prf_T_nlo(k)

        moment_k_p, error_k_p = singlet_moment(k,eta,t,  moment_label,"+",evolution_order,error_type,interpolation=interpolation)
        moment_k_m, error_k_m = singlet_moment(k,eta,t,  moment_label,"-",evolution_order,error_type,interpolation=interpolation)

        T_1_top = prf_T_1 * 2 * (
            (ga_qq - ga_j_m) * ( ga_qq_nd * (ga_qq_k - ga_k_m) + ga_qg_nd * ga_gq_k )
            + ga_qg * ( ga_gq_nd * (ga_qq_k - ga_k_m) + ga_gg_nd * ga_gq_k )                             
        )          
        T_2_top = prf_T_2 * 2 *  (
            (ga_qq - ga_j_m) * ( ga_qq_nd * (ga_qq_k - ga_k_p) + ga_qg_nd * ga_gq_k )
            + ga_qg * ( ga_gq_nd * (ga_qq_k - ga_k_p) + ga_gg_nd * ga_gq_k )                             
        )                    
        T_3_top = - prf_T_3 * (-2) *  (
            (ga_qq - ga_j_p) * ( ga_qq_nd * (ga_qq_k - ga_k_m) + ga_qg_nd * ga_gq_k )
            + ga_qg * ( ga_gq_nd * (ga_qq_k - ga_k_m) + ga_gg_nd * ga_gq_k )                             
        )
        T_4_top = prf_T_4 * 2 *  (
            (ga_qq - ga_j_p) * ( ga_qq_nd * (ga_qq_k - ga_k_p) + ga_qg_nd * ga_gq_k )
            + ga_qg * ( ga_gq_nd * (ga_qq_k - ga_k_p) + ga_gg_nd * ga_gq_k )                             
        )
        plus_terms = T_1_top + T_3_top
        minus_terms = T_2_top + T_4_top

        quark_non_diagonal_part = eta**(j-k) * (plus_terms * moment_k_p + minus_terms * moment_k_m)
        if isinstance(j, (int, np.integer)) and isinstance(k, (int, np.integer)):
            sum_squared = (eta**(j-k) * plus_terms * error_k_p)**2 + (eta**(j-k) * minus_terms * error_k_m)**2
            quark_non_diagonal_errors = abs(mp.sqrt(sum_squared))
            return quark_non_diagonal_part, quark_non_diagonal_errors
        else:
            return quark_non_diagonal_part

    def T_gluon_nlo(k):
        # Note T = 0 for j=k
        ga_j_p, ga_j_m = ga_p, ga_m
        ga_k_p, ga_k_m = adim.gamma_pm(k-1,evolve_type,interpolation=interpolation)
        ga_qq_k = adim.gamma_qq(k-1,evolution_order="lo",interpolation=interpolation)
        ga_gq_k = adim.gamma_gq(k-1, evolve_type,"lo",interpolation=interpolation)
        ga_qq_nd = adim.gamma_qq_nd(j-1,k-1,evolve_type,"nlo",interpolation=interpolation)
        ga_qg_nd = adim.gamma_qg_nd(j-1,k-1,evolve_type,"nlo",interpolation=interpolation)
        ga_gq_nd = adim.gamma_gq_nd(j-1,k-1,evolve_type,"nlo",interpolation=interpolation)
        ga_gg_nd = adim.gamma_gg_nd(j-1,k-1,evolve_type,"nlo",interpolation=interpolation)

        prf_T_1, prf_T_2, prf_T_3, prf_T_4 = prf_T_nlo(k)

        moment_k_p, error_k_p = singlet_moment(k,eta,t,  moment_label,"+",evolution_order,error_type,interpolation=interpolation)
        moment_k_m, error_k_m = singlet_moment(k,eta,t,  moment_label,"-",evolution_order,error_type,interpolation=interpolation)

        T_1_bot = prf_T_1 * 2 * (
            ga_gq * ( ga_qq_nd * (ga_qq_k - ga_k_m) + ga_qg_nd * ga_gq_k )    
            + (ga_gg - ga_j_m) * ( ga_gq_nd * (ga_qq_k - ga_k_m) + ga_gg_nd * ga_gq_k )                         
        )
        T_2_bot = prf_T_2 * 2  * (
            ga_gq * ( ga_qq_nd * (ga_qq_k - ga_k_p) + ga_qg_nd * ga_gq_k )    
            + (ga_gg - ga_j_m) * ( ga_gq_nd * (ga_qq_k - ga_k_p) + ga_gg_nd * ga_gq_k )                         
        )
        T_3_bot = - prf_T_3 * (-2) * (
            ga_gq * ( ga_qq_nd * (ga_qq_k - ga_k_m) + ga_qg_nd * ga_gq_k )    
            + (ga_gg - ga_j_p) * ( ga_gq_nd * (ga_qq_k - ga_k_m) + ga_gg_nd * ga_gq_k )   
        )                      
        T_4_bot = prf_T_4 * 2 * (
            ga_gq * ( ga_qq_nd * (ga_qq_k - ga_k_p) + ga_qg_nd * ga_gq_k )    
            + (ga_gg - ga_j_p) * (ga_gq_nd * (ga_qq_k - ga_k_p) + ga_gg_nd * ga_gq_k )                         
        )
        plus_terms = T_1_bot + T_3_bot
        minus_terms = T_2_bot + T_4_bot

        gluon_non_diagonal_part = eta**(j-k) * (plus_terms * moment_k_p + minus_terms * moment_k_m)
        if isinstance(j, (int, np.integer)) and isinstance(k, (int, np.integer)):
            sum_squared = (eta**(j-k) * plus_terms * error_k_p)**2 + (eta**(j-k) * minus_terms * error_k_m)**2
            gluon_non_diagonal_errors = abs(mp.sqrt(sum_squared))
            return gluon_non_diagonal_part, gluon_non_diagonal_errors
        else:
            return gluon_non_diagonal_part
        
    # Initialize non-diagonal terms in evolution
    non_diagonal_terms = 0
    non_diagonal_errors = 0
    non_diagonal_terms_alt = 0

    if moment_type == "singlet":
        if particle == "quark":
            result = A_lo_quark("+") * moment_in_p + A_lo_quark("-") * moment_in_m
            sum_squared =  (A_lo_quark("+") * error_p)**2 + (A_lo_quark("-") * error_m)**2
            error = abs(mp.sqrt(sum_squared))
            result += hp.error_sign(error,error_type)
            if evolution_order == "nlo":
                plus_terms = A_quark_nlo("+") + B_quark_nlo("+")
                minus_terms = A_quark_nlo("-") + B_quark_nlo("-")
                diagonal_terms = plus_terms * moment_in_p + minus_terms * moment_in_m
                sum_squared = plus_terms**2 * error_p**2 + minus_terms**2 * error_m**2
                diagonal_errors = abs(mp.sqrt(sum_squared))
                if isinstance(j, (int, np.integer)) and eta != 0:
                    for k in range(2,j - 2 + 1):
                        non_diagonal_terms += T_quark_nlo(k)[0]
                        non_diagonal_errors += T_quark_nlo(k)[1]
                    # Exactly resum for real j
                    non_diagonal_terms_alt = 0
                elif eta != 0 and cfg.ND_EVOLVED_COMPLEX_MOMENT:
                    # ND evolution comes with a factor of 1 + (-1)**(j-k)
                    # that needs to be treated separately
                    non_diagonal_terms = sp.fractional_finite_sum(T_quark_nlo,k_0=2,k_1=j - 2 + 1,n_tuple=1)
                    # (-1)**(-k), treated here, (-1)**j in mellin_barnes_gpd later on
                    non_diagonal_terms_alt = sp.fractional_finite_sum(T_quark_nlo,k_0=2,k_1=j - 2 + 1,n_tuple=1,alternating_sum=True)
                error = diagonal_errors + non_diagonal_errors
                # The term without (-1)**j can be added to the diagonal part
                result += diagonal_terms + non_diagonal_terms + hp.error_sign(error,error_type)
        if particle == "gluon":
            result = A_lo_gluon("+") * moment_in_p + A_lo_gluon("-") * moment_in_m
            sum_squared =  (A_lo_gluon("+") * error_p)**2 + (A_lo_gluon("-") * error_m)**2
            error = abs(mp.sqrt(sum_squared))
            result += hp.error_sign(error,error_type)
            if evolution_order == "nlo":
                plus_terms = A_gluon_nlo("+") + B_gluon_nlo("+")
                minus_terms = A_gluon_nlo("-")  + B_gluon_nlo("-")
                diagonal_terms =  plus_terms * moment_in_p + minus_terms * moment_in_m
                sum_squared = plus_terms**2 * error_p**2 + minus_terms**2 * error_m**2
                diagonal_errors = abs(mp.sqrt(sum_squared))
                if isinstance(j, (int, np.integer)) and eta != 0:
                    for k in range(2,j - 2 + 1):
                        non_diagonal_terms += T_gluon_nlo(k)[0]
                        non_diagonal_errors += T_gluon_nlo(k)[1]
                    # Exactly resum for real j
                    non_diagonal_terms_alt = 0
                elif eta != 0 and cfg.ND_EVOLVED_COMPLEX_MOMENT:
                    # ND evolution comes with a factor of 1 + (-1)**(j-k)
                    # that needs to be treated separately
                    non_diagonal_terms = sp.fractional_finite_sum(T_gluon_nlo,k_0=2,k_1=j - 2 + 1,n_tuple=1)
                    # (-1)**(-k), treated here, (-1)**j in mellin_barnes_gpd later on
                    non_diagonal_terms_alt = sp.fractional_finite_sum(T_gluon_nlo,k_0=2,k_1=j - 2 + 1,n_tuple=1,alternating_sum=True)
                error = diagonal_errors + non_diagonal_errors
                # The term without (-1)**j can be added to the diagonal part
                result += diagonal_terms + non_diagonal_terms + hp.error_sign(error,error_type)

    elif moment_type in ["non_singlet_isovector","non_singlet_isoscalar"]: 
        if evolution_order == "lo":
            result = moment_in * E_non_singlet_lo()
        elif evolution_order == "nlo":
            result = moment_in * E_non_singlet_nlo(j-1)
            non_diagonal_terms = 0
            if isinstance(j, (int, np.integer)) and eta != 0:
                for k in range(1,j - 2 + 1):
                    non_diagonal_terms += EB_non_singlet_nlo(k)
                # Exactly resum for real j
                non_diagonal_terms_alt = 0
            elif eta != 0 and cfg.ND_EVOLVED_COMPLEX_MOMENT:
                # ND evolution comes with a factor of 1 + (-1)**(j-k)
                # that needs to be treated separately
                non_diagonal_terms = sp.fractional_finite_sum(EB_non_singlet_nlo,k_0=1,k_1=j - 2 + 1,n_tuple=1)
                # (-1)**(-k), treated here, (-1)**j in mellin_barnes_gpd later on
                non_diagonal_terms_alt = sp.fractional_finite_sum(EB_non_singlet_nlo,k_0=1,k_1=j - 2 + 1,n_tuple=1,alternating_sum=True)
            # The term without (-1)**j can be added to the diagonal part
            result += non_diagonal_terms

    result *= A0
    non_diagonal_terms_alt *= A0
    # Return real value when called for real j
    if result.imag == 0 or isinstance(j, (int, np.integer)):
        return float(result.real)
    # Need to keep separated for mellin_barnes_gpd
    return result, non_diagonal_terms_alt

def dipole_moment(n,eta,t,mu,particle="quark",moment_type="non_singlet_isovector",moment_label="A",evolution_order="nlo",error_type="central",lattice=False):
    """
    Get the dipole form for the evolved conformal moment F_{n}^{+-} obtained by fit.dipole_fit_moment

    Parameters
    ----------
    n : int
        Conformal spin.
    eta : float
        Skewness parameter.
    t : float
        Mandelstam t.
    mu : float
        Resolution scale.
    particle : str, optional
        "quark" or "gluon". Default is "quark".
    moment_type : str, optional
        non_singlet_isovector, non_singlet_isoscalar, or singlet.
    moment_label : str, optional
        A(tilde), B(tilde) depending on H(tilde) or E(tilde) GPD etc. Default is "A".
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"
    lattice : bool, optional
        Can also be used to get the dipole form of lattice moments

    Returns
    -------
    float
        The value of the evolved conformal moment at scale mu in dipole form.
    """
    def dipole_form(t, A_D, m_D2): 
        return A_D / (1 - t / m_D2)**2
    
    def parse_csv_params(csv_path, particle, moment_type, moment_label, n, evolution_order):
        csv_path = cfg.Path(csv_path)
        key = (particle, moment_type, moment_label, str(n), evolution_order)

        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if lattice:
                    row_key = (row["particle"], row["moment_type"], row["moment_label"], row["n"], row["pub_id"])
                else:
                    row_key = (row["particle"], row["moment_type"], row["moment_label"], row["n"], row["evolution_order"])
                if row_key == key:
                    return {
                        "A_D": float(row["A_D"]),
                        "m_D2": float(row["m_D2"])
                    }

        raise ValueError(f"No matching entry found in {csv_path} for key: {key}")
    def get_dipole_from_csv(t,csv_path, particle, moment_type, moment_label, n, evolution_order):
        params = parse_csv_params(csv_path,particle, moment_type, moment_label, n, evolution_order)
        if 'A_D' not in params or 'm_D2' not in params:
            print("Required parameters 'A_D' or 'm_D2' not found in CSV - abort")
            return None
        return dipole_form(t, params['A_D'], params['m_D2'])
    
    if cfg.N_F != 3:
        print("Warning: Currently no distinction on file system for cfg.N_F != 3")
    if lattice:
        if error_type != "central":
            file_path = cfg.MOMENTUM_SPACE_MOMENTS_PATH / f"dipole_moments_{evolution_order}_{error_type}.csv"
        else:
            file_path = cfg.MOMENTUM_SPACE_MOMENTS_PATH / f"dipole_moments_{evolution_order}.csv"
    else:
        prefix = "dipole_moments"
        file_path = hp.generate_filename(eta,0,mu,cfg.MOMENTUM_SPACE_MOMENTS_PATH / prefix,error_type )
    result = get_dipole_from_csv(t,file_path, particle, moment_type, moment_label, n, evolution_order)
    return result

def evolve_singlet_D(j,eta,t,mu,D0=1,particle="quark",moment_label="A",evolution_order="nlo",error_type="central"):
    """
    Helper function to extract the evolved D-term moment. For documentation see evolve_conformal_moment.
    """
    hp.check_particle_type(particle)
    hp.check_moment_type_label("singlet",moment_label)
    if j == 2:
        eta = 1 # Result is eta independent 
    term_1 = evolve_conformal_moment(j,eta,t,mu,1,particle,"singlet",moment_label,evolution_order,error_type)
    term_2 = evolve_conformal_moment(j,0,t,mu,1,particle,"singlet",moment_label,evolution_order,error_type)
    result = D0 * (term_1-term_2)/eta**2
    return result

def evolve_quark_non_singlet(j,eta,t,mu,A0=1,moment_type="non_singlet_isovector",moment_label = "A",evolution_order="nlo",error_type="central"):
    """
    Helper function to extract the quark non-singlet moments. For documentation see evolve_conformal_moment.
    """
    result = evolve_conformal_moment(j,eta,t,mu,A0,"quark",moment_type,moment_label,evolution_order,error_type)
    return result

def evolve_quark_singlet(j,eta,t,mu,A0=1,moment_label = "A",evolution_order="nlo",error_type="central"):
    """
    Helper function to extract the quark singlet moment. For documentation see evolve_conformal_moment.
    """
    result = evolve_conformal_moment(j,eta,t,mu,A0,"quark","singlet",moment_label,evolution_order,error_type)
    return result

def evolve_gluon_singlet(j,eta,t,mu,A0=1,moment_label = "A",evolution_order="nlo",error_type="central"):
    """
    Helper function to extract the gluon singlet moment. For documentation see evolve_conformal_moment.
    """
    result = evolve_conformal_moment(j,eta,t,mu,A0,"gluon","singlet",moment_label,evolution_order,error_type)
    return result

def evolve_quark_singlet_D(eta,t,mu,D0=1,moment_label = "A",evolution_order="nlo",error_type="central"):
    """
    Helper function to extract the evolved quark D-term moment. For documentation see evolve_conformal_moment.
    """
    result = evolve_singlet_D(eta,t,mu,D0,"quark",moment_label,evolution_order,error_type)
    return result

def evolve_gluon_singlet_D(j,eta,t,mu,D0=1,moment_label = "A",evolution_order="nlo",error_type="central"):
    """
    Helper function to extract the evolved gluon D-term moment. For documentation see evolve_conformal_moment.
    """
    result = evolve_singlet_D(j,eta,t,mu,D0,"gluon",moment_label,evolution_order,error_type)
    return result

def fourier_transform_moment(n,eta,mu,b_vec,A0=1,particle="quark",moment_type="non_singlet_isovector", moment_label="A",evolution_order="nlo", Delta_max = 5,num_points=100, error_type="central",dipole_form=True):
    """
    Compute Fourier transformed moments using trapezoidal rule.

    Parameters
    ----------
    n : int
        Conformal spin.
    eta : float
        Skewness parameter.
    t : float
        Mandelstam t.
    mu : float
        Resolution scale.
    A0 : float, optional
        Normalization factor (default A0 = 1).
    particle : str, optional
        "quark" or "gluon". Default is "quark".
    moment_type : str, optional
        non_singlet_isovector, non_singlet_isoscalar, or singlet.
    moment_label : str, optional
        A(tilde), B(tilde) depending on H(tilde) or E(tilde) GPD etc. Default is "A".
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    Delta_max : float, optional
        Maximal momentum transfer to cut off integration
    num_points : int, optional
        Number of grid points used for trapezoidal rule is num_points**2
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"
    dipole_form : bool, optional
        Use dipole fit for faster integration.

    Returns
    -------
    float
        The value of the Fourier transformed moment at `b_vec`.
    """

    hp.check_error_type(error_type)
    hp.check_particle_type(particle)
    hp.check_moment_type_label(moment_type,moment_label)
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
        if eta == 0:
            t = -(Delta_x**2+Delta_y**2)
        else:
            m_N = 0.93827
            t = - (4 * eta**2 * m_N**2 + (Delta_x**2+Delta_y**2)) / (1-eta**2)
        exponent = -1j * (b_x * Delta_x + b_y * Delta_y)
        if dipole_form:
            moment = dipole_moment(n,eta,t,mu,particle,moment_type,moment_label,evolution_order,error_type)
        else:
            moment = evolve_conformal_moment(n,eta,t,mu,A0,particle,moment_type,moment_label,evolution_order,error_type)
        result = moment * np.exp(exponent)
        return result
    
    # Compute the integrand for each pair of (Delta_x, Delta_y) values
    integrand_values = integrand(Delta_x_grid, Delta_y_grid, b_x, b_y)
    # Perform the numerical integration using the trapezoidal rule for efficiency
    integral_result = np.real(trapezoid(trapezoid(integrand_values, Delta_x_vals, axis=0), Delta_y_vals))/((2*np.pi)**2)

    return integral_result
    
def inverse_fourier_transform_moment(n,eta,mu,Delta_vec,particle="quark",moment_type="non_singlet_isovector",moment_label="A",evolution_order="nlo", 
                                     b_max = 9 ,num_points=500, Delta_max=10):
    """
    Sanity check for Fourier transform. The result should be the input moment.

    Parameters
    ----------
    n : int
        Conformal spin.
    eta : float
        Skewness parameter.
    Delta_vec : vector float
        Momentum transfer
    mu : float
        Resolution scale.
    particle : str, optional
        "quark" or "gluon". Default is "quark".
    moment_type : str, optional
        non_singlet_isovector, non_singlet_isoscalar, or singlet.
    moment_label : str, optional
        A(tilde), B(tilde) depending on H(tilde) or E(tilde) GPD etc. Default is "A".
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    b_max : float, optional
        Maximal impact parameter to cut off integration
    num_points : int, optional
        Number of grid points used for trapezoidal rule is num_points**2
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"
    dipole_form : bool, optional
        Use dipole fit for faster integration.

    Returns
    -------
    float
        The value of the inverse Fourier transformed moment at `Delta_vec`.
    """
    hp.check_particle_type(particle)
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
        moment = fourier_transform_moment(n,eta,mu,b_vec,particle,moment_type,moment_label,evolution_order,num_points=num_points,Delta_max=Delta_max)
        result = moment * np.exp(exponent)
        return result

    # Compute the integrand for each pair of (Delta_x, Delta_y) values
    integrand_values = np.array(Parallel(n_jobs=-1)(delayed(integrand)(b_x, b_y, Delta_x, Delta_y)
                                                 for b_y in b_y_vals
                                                 for b_x in b_x_vals))
    integrand_values = integrand_values.reshape((num_points, num_points))

    integral_result = trapezoid(trapezoid(integrand_values, b_x_vals, axis=1), b_y_vals,axis=0)

    return integral_result.real

def fourier_transform_transverse_moment(n,eta,mu,b_vec,A0=1,particle="quark",moment_type="non_singlet_isovector",evolution_order="nlo", Delta_max = 5,num_points=100, error_type="central",dipole_form=True):
    """
    Compute Fourier transformed transverse moments using trapezoidal rule. Currently only supports unpolarized moments.

    Parameters
    ----------
    n : int
        Conformal spin.
    eta : float
        Skewness parameter.
    t : float
        Mandelstam t.
    mu : float
        Resolution scale.
    A0 : float, optional
        Normalization factor (default A0 = 1).
    particle : str, optional
        "quark" or "gluon". Default is "quark".
    moment_type : str, optional
        non_singlet_isovector, non_singlet_isoscalar, or singlet.
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    Delta_max : float, optional
        Maximal momentum transfer to cut off integration
    num_points : int, optional
        Number of grid points used for trapezoidal rule is num_points**2
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"
    dipole_form : bool, optional
        Use dipole fit for faster integration.

    Returns
    -------
    float
        The value of the Fourier transformed transverse moment at `b_vec`.
    """
    hp.check_error_type(error_type)
    hp.check_particle_type(particle)
    hp.check_moment_type_label(moment_type,"A")
    hp.check_moment_type_label(moment_type,"B")
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
        if dipole_form:
            moment_1 = dipole_moment(n,eta,t,mu,particle,moment_type,"A",evolution_order,error_type)
            moment_2 = 1j * Delta_y/(2*M_n) * dipole_moment(n,eta,t,mu,particle,moment_type,"B",evolution_order,error_type)
        else:
            moment_1 = evolve_conformal_moment(n,eta,t,mu,A0,particle,moment_type,"A",evolution_order,error_type)
            moment_2 = 1j * Delta_y/(2*M_n) * evolve_conformal_moment(n,eta,t,mu,A0,particle,moment_type,"B",evolution_order,error_type)
        moment = moment_1 + moment_2
        result = moment * np.exp(exponent)
        return result

    # Compute the integrand for each pair of (Delta_x, Delta_y) values
    integrand_values = integrand(Delta_x_grid, Delta_y_grid, b_x, b_y)
    # Perform the numerical integration using the trapezoidal rule for efficiency
    integral_result = np.real(trapezoid(trapezoid(integrand_values, Delta_x_vals, axis=0), Delta_y_vals))/((2*np.pi)**2)

    return integral_result

def fourier_transform_quark_gluon_helicity(eta,mu,b_vec,particle="quark",moment_type="non_singlet_isovector",evolution_order="nlo", Delta_max = 10,num_points=100, error_type="central"):
    """
    Quark gluon helicity in impact parameter space in GeV^2. For documentation see fourier_transform_moment.
    """
    def ft_moment(b_vec,moment_type,error_type):
        return fourier_transform_moment(n=1,eta=eta,mu=mu,b_vec=b_vec,
                                        particle=particle,moment_type=moment_type,
                                        moment_label="Atilde",evolution_order=evolution_order,
                                        Delta_max=Delta_max,num_points=num_points,error_type=error_type)
    hp.check_particle_type(particle)
    if moment_type not in ["singlet","non_singlet_isovector","non_singlet_isoscalar","u","d"]:
        raise ValueError(f"Wrong moment_type {moment_type}")
    hp.check_error_type(error_type)

    if moment_type in ["singlet","non_singlet_isovector","non_singlet_isoscalar"]:
        result = ft_moment(b_vec,moment_type,error_type=error_type)/2
    elif moment_type in ["u","d"]:
        if moment_type == "u":
            prf = 1
        else:
            prf = -1
        moment_1 = ft_moment(b_vec,moment_type="non_singlet_isoscalar",error_type="central")/2
        moment_2 = ft_moment(b_vec,moment_type="non_singlet_isovector",error_type="central")/2
        result = (moment_1 + prf * moment_2)/2
        if error_type != "central":
            error_1 = .5 * ft_moment(b_vec,moment_type="non_singlet_isoscalar",error_type=error_type) - moment_1
            error_2 = .5 * ft_moment(b_vec,moment_type="non_singlet_isovector",error_type=error_type) - moment_2
            error = np.sqrt(error_1**2 + error_2**2)
            result += hp.error_sign(error,error_type)

    return result

def fourier_transform_quark_helicity(eta,mu,b_vec,moment_type="non_singlet_isovector",evolution_order="nlo", Delta_max = 10,num_points=100, error_type="central"):
    """
    Helper function to get Fourier transformed quark helicity. For documentation see fourier_transform_moment.
    """
    result = fourier_transform_quark_gluon_helicity(eta,mu,b_vec,particle="quark",moment_type=moment_type,evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type=error_type)
    return result

def fourier_transform_gluon_helicity(eta,mu,b_vec,evolution_order="nlo",Delta_max = 10,num_points=100, error_type="central"):
    """
    Helper function to get Fourier transformed gluon helicity. For documentation see fourier_transform_moment.
    """
    result = fourier_transform_quark_gluon_helicity(eta,mu,b_vec,particle="gluon",moment_type="singlet",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type=error_type)
    return result

def fourier_transform_spin_orbit_correlation(eta,mu,b_vec,evolution_order="nlo",particle="quark",moment_type="non_singlet_isovector", Delta_max = 8,num_points=100, error_type="central"):
    """
    Spin-orbit correlation in impact parameter space in GeV^2. For documentation see fourier_transform_moment.
    """
    def ft_moment(n,b_vec,moment_type,moment_label,error_type):
        return fourier_transform_moment(n=n,eta=eta,mu=mu,b_vec=b_vec,
                                        particle=particle,moment_type=moment_type,
                                        moment_label=moment_label,evolution_order=evolution_order,
                                        Delta_max=Delta_max,num_points=num_points,error_type=error_type)
    hp.check_particle_type(particle)
    if moment_type not in ["singlet","non_singlet_isovector","non_singlet_isoscalar","u","d"]:
        raise ValueError(f"Wrong moment_type {moment_type}")
    hp.check_error_type(error_type)

    if moment_type in ["singlet","non_singlet_isovector","non_singlet_isoscalar"]:
            term_1 = ft_moment(2,b_vec,moment_type=moment_type,moment_label="Atilde",error_type="central")
            term_2 = ft_moment(1,b_vec,moment_type=moment_type,moment_label="A",error_type="central")
            moment = (term_1-term_2)/2
            if error_type != "central":      
                term_1_error = .5 * (ft_moment(2,b_vec,moment_type=moment_type,moment_label="Atilde",error_type=error_type)
                                - term_1)
                term_2_error = .5 * (ft_moment(1,b_vec,moment_type=moment_type,moment_label="A",error_type=error_type)
                                - term_2)
                error = np.sqrt(term_1_error**2+term_2_error**2)
                moment += hp.error_sign(error,error_type)
            result = moment
            return result

    elif moment_type in ["u","d"]:
        if moment_type == "u":
            prf = 1
        else:
            prf = -1
            term_1 = ft_moment(2,b_vec,moment_type="non_singlet_isoscalar",moment_label="Atilde",error_type="central")
            term_2 = ft_moment(1,b_vec,moment_type="non_singlet_isoscalar",moment_label="A",error_type="central")
            moment_1 = (term_1-term_2)/2
            if error_type != "central":      
                term_1_error = .5 * (ft_moment(2,b_vec,moment_type="non_singlet_isoscalar",moment_label="Atilde",error_type=error_type)
                                - term_1)
                term_2_error = .5 * (ft_moment(2,b_vec,moment_type="non_singlet_isoscalar",moment_label="A",error_type=error_type)
                                - term_2)
                error_1 = np.sqrt(term_1_error**2+term_2_error**2)
                moment_1 += hp.error_sign(error,error_type)
    
            term_1 = ft_moment(2,b_vec,moment_type="non_singlet_isovector",moment_label="Atilde",error_type="central")
            term_2 = ft_moment(1,b_vec,moment_type="non_singlet_isovector",moment_label="A",error_type="central")
            moment_2 = (term_1-term_2)/2 
            moment = (moment_1 + prf * moment_2)/2
            if error_type != "central":      
                term_1_error = .5 * (ft_moment(2,b_vec,moment_type="non_singlet_isovector",moment_label="Atilde",error_type=error_type)
                                - term_1)
                term_2_error = .5 * (ft_moment(1,b_vec,moment_type="non_singlet_isovector",moment_label="A",error_type=error_type)
                                - term_2)
                error_2 = np.sqrt(term_1_error**2+term_2_error**2)
                error = np.sqrt(error_1**2 + error_2**2)
                moment += hp.error_sign(error,error_type)
            result = moment
            return result
        
def fourier_transform_orbital_angular_momentum(eta,mu,b_vec,particle="quark",moment_type="non_singlet_isovector",evolution_order="nlo", Delta_max = 7,num_points=100, error_type="central"):
    """
    Orbital angular momentum in impact parameter space in GeV^2. For documentation see fourier_transform_moment.
    """
    def ft_moment(n,b_vec,moment_type,moment_label,error_type):
        return fourier_transform_moment(n=n,eta=eta,mu=mu,b_vec=b_vec,
                                        particle=particle,moment_type=moment_type,
                                        moment_label=moment_label,evolution_order=evolution_order,
                                        Delta_max=Delta_max,num_points=num_points,error_type=error_type)
    hp.check_particle_type(particle)
    if moment_type not in ["singlet","non_singlet_isovector","non_singlet_isoscalar","u","d"]:
        raise ValueError(f"Wrong moment_type {moment_type}")
    hp.check_error_type(error_type)

    if moment_type in ["singlet","non_singlet_isovector","non_singlet_isoscalar"]:
            term_1 = ft_moment(2,b_vec,moment_type=moment_type,moment_label="A",error_type="central")
            term_2 = ft_moment(2,b_vec,moment_type=moment_type,moment_label="B",error_type="central")
            term_3 = ft_moment(1,b_vec,moment_type=moment_type,moment_label="Atilde",error_type="central")
            moment = (term_1+term_2-term_3)/2
            if error_type != "central":   
                term_1_error = .5 * (ft_moment(2,b_vec,moment_type=moment_type,moment_label="A",error_type=error_type) - term_1)
                term_2_error = .5 * (ft_moment(2,b_vec,moment_type=moment_type,moment_label="B",error_type=error_type) - term_2)
                term_3_error = .5 * (ft_moment(1,b_vec,moment_type=moment_type,moment_label="Atilde",error_type=error_type) - term_3)
                error = np.sqrt(term_1_error**2+term_2_error**2+term_3_error**2)
                moment += hp.error_sign(error,error_type)
            result = moment
            return result

    elif moment_type in ["u","d"]:
        if moment_type == "u":
            prf = 1
        else:
            prf = -1
            term_1 = ft_moment(2,b_vec,moment_type="non_singlet_isoscalar",moment_label="A",error_type="central")
            term_2 = ft_moment(2,b_vec,moment_type="non_singlet_isoscalar",moment_label="B",error_type="central")
            term_3 = ft_moment(1,b_vec,moment_type="non_singlet_isoscalar",moment_label="Atilde",error_type="central")
            moment_1 = (term_1+term_2-term_3)/2
            if error_type != "central":      
                term_1_error = .5 * (ft_moment(2,b_vec,moment_type="non_singlet_isoscalar",moment_label="A",error_type=error_type) - term_1)
                term_2_error = .5 * (ft_moment(2,b_vec,moment_type="non_singlet_isoscalar",moment_label="B",error_type=error_type) - term_2)
                term_3_error = .5 * (ft_moment(1,b_vec,moment_type="non_singlet_isoscalar",moment_label="Atilde",error_type=error_type) - term_3)
                error_1 = np.sqrt(term_1_error**2+term_2_error**2+term_3_error**2)

            term_1 = ft_moment(2,b_vec,moment_type="non_singlet_isovector",moment_label="A",error_type="central")
            term_2 = ft_moment(2,b_vec,moment_type="non_singlet_isovector",moment_label="B",error_type="central")
            term_3 = ft_moment(1,b_vec,moment_type="non_singlet_isovector",moment_label="Atilde",error_type="central")
            moment_2 = (term_1+term_2-term_3)/2
            moment = (moment_1 + prf * moment_2)/2
            if error_type != "central":      
                term_1_error = .5 * (ft_moment(2,b_vec,moment_type="non_singlet_isovector",moment_label="A",error_type=error_type) - term_1)
                term_2_error = .5 * (ft_moment(2,b_vec,moment_type="non_singlet_isovector",moment_label="B",error_type=error_type) - term_2)
                term_3_error = .5 * (ft_moment(1,b_vec,moment_type="non_singlet_isovector",moment_label="Atilde",error_type=error_type) - term_3)
                error_2 = np.sqrt(term_1_error**2+term_2_error**2+term_3_error**2)
                error = np.sqrt(error_1**2 + error_2**2)
                moment += hp.error_sign(error,error_type)
            result = moment
            return result

def fourier_transform_quark_orbital_angular_momentum(eta,mu,b_vec,moment_type="non_singlet_isovector",evolution_order="nlo", Delta_max = 7,num_points=100, error_type="central"):
    """
    Helper function to get quark orbital angular momentum. For documentation see fourier_transform_moment.
    """
    result = fourier_transform_orbital_angular_momentum(eta,mu,b_vec,particle="quark",moment_type=moment_type,evolution_order=evolution_order, Delta_max=Delta_max,num_points=num_points, error_type=error_type)
    return result

################################
#### Mellin-Barnes Integral ####
################################

# Define conformal partial waves
@hp.mpmath_vectorize
def conformal_partial_wave(j, x, eta, particle = "quark", parity="none"):
    """
    Calculate the conformal partial waves for quark and gluon GPDs and generate their
    respective "even" or "odd" combinations.

    Parameters
    ----------
    j : complex
        Conformal spin.
    x : float
        Value of parton x.
    eta : float
        Value of skewness.
    particle : str, optional
        "quark" or "gluon". Default is "quark".
    parity : str, optional
        The parity of the function. Either "even", "odd", or "none". Default is "none".

    Returns
    -------
    mpc
        Value of even or odd combination of conformal quark partial waves.

    Raises
    ------
    ValueError
        If the `parity` argument is not "even", "odd", or "none".

    Notes
    -----
    Decorated using mpmath_vectorize from helpers.py for vectorized calls
    """
    hp.check_particle_type(particle)
    hp.check_parity(parity)
    if parity not in ["even", "odd","none"]:
        raise ValueError("Parity must be even, odd or none")
    
    if particle == "quark":
        def cal_P(x,eta):
            gamma_term = 2.0**j * mp.gamma(1.5 + j) / (mp.gamma(0.5) * mp.gamma(j))
            arg = (1 + x / eta)
            hyp = mp.hyp2f1(-j, j + 1, 2, 0.5 * arg)
            result = 1 / eta**j  * arg * hyp * gamma_term
            return result
        def cal_Q(x,eta):
            sin_term = mp.sin(mp.pi * j) / mp.pi
            hyp = mp.hyp2f1(0.5 * j, 0.5 * (j + 1), 1.5 + j, (eta / x)**2) 
            result = 1 / x**j * hyp * sin_term
            return result
    else:   
        def cal_P(x,eta):
            gamma_term = 2.0**(j-1) * mp.gamma(1.5 + j) / (mp.gamma(0.5) * mp.gamma(j-1))
            arg = (1. + x / eta)
            hyp = mp.hyp2f1(-j, j + 1, 3, 0.5 * arg)
            result = 1 / eta**(j-1) * arg**2 * hyp * gamma_term
            return result
        def cal_Q(x,eta):
            sin_term = mp.sin(mp.pi * (j+1))  / mp.pi 
            hyp = mp.hyp2f1(0.5 * (j-1), 0.5 * j, 1.5 + j, (eta / x)**2) 
            result = 1 / x**(j-1) * hyp * sin_term
            return result

    def p_j(x):
        # Initialize P_term and Q_term with zero
        P_term = 0
        Q_term = 0        
        if eta - np.abs(x) >= 0 : 
            # Note continuity at x = eta gives cal_P = cal_Q
            P_term =  cal_P(x,eta)
        elif x - eta > 0 :
            Q_term = cal_Q(x,eta)
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
    """
    Get the real value of j and corresponding parity used for the 
    conformal partial waves in the Mellin-Barnes integration of the GPD

    Parameters
    ----------
    particle : str, optional
        "quark" or "gluon". Default is "quark".
    moment_type : str, optional
        non_singlet_isovector, non_singlet_isoscalar, or singlet.
    moment_label : str, optional
        A(tilde), B(tilde) depending on H(tilde) or E(tilde) GPD etc. Default is "A".

    Returns
    -------
    j_base : float
        Real part of j used for Mellin-Barnes integration
    parity : str
        The partinent parity label.

    Raises
    ------
    ValueError
        If the `parity` argument is not "even", "odd", or "none".

    """
    hp.check_particle_type(particle)
    hp.check_moment_type_label(moment_type,moment_label)

    if moment_label in ["A","B"]:
        # Vector exchange
        if particle == "quark" and moment_type in ["non_singlet_isovector","non_singlet_isoscalar"]:
            j_base, parity = 0.95, "none"
        # Tensor exchanges
        elif particle == "quark" and moment_type == "singlet":
            j_base, parity = 1.95, "odd"
        elif particle == "gluon" and moment_type == "singlet":
            j_base, parity = 1.95, "even"
    elif moment_label == "Atilde":
        # Vector exchange
        if particle == "quark" and moment_type in ["non_singlet_isovector","non_singlet_isoscalar"]:
            j_base, parity = 0.95, "none"
        # Vector exchanges
        if particle == "quark" and moment_type == "singlet":
            j_base, parity = 0.95, "even"
        if particle == "gluon" and moment_type == "singlet":
            j_base, parity = 0.95, "odd"
    else:
        raise ValueError(f"Wrong moment type {moment_type} and/or label {moment_label} for particle {particle}")
    
    return j_base, parity

@cfg.memory.cache
def estimate_gpd_error(eta,t,mu,particle,moment_type,moment_label,evolution_order,error_type):
    """
    Estimate the relative error of the GPD moment based on input moment uncertainties.

    Parameters
    ----------
    eta : float
        Skewness parameter.
    t : float
        Mandelstam t.
    mu : float
        Resolution scale.
    particle : str
        "quark" or "gluon". Default is "quark".
    moment_type : str
        non_singlet_isovector, non_singlet_isoscalar, or singlet.
    moment_label : str
        A(tilde), B(tilde) depending on H(tilde) or E(tilde) GPD etc.
    evolution_order : str, optional
        "lo", "nlo",... .
    error_type : str
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"
        
    Returns
    -------
    float
        The relative error estimate for the specified GPD moment.

    Notes
    -----
    This function is cached using `joblib.Memory` for performance.
    """

    if error_type == "central":
        return 1
    @hp.mpmath_vectorize
    def compute_rel_error(i):
        central_term = evolve_conformal_moment(
            j=i, eta=eta, t=t, mu=mu,
            particle=particle, moment_type=moment_type,
            moment_label=moment_label,
            evolution_order=evolution_order,
            error_type="central"
            )
        error_term = evolve_conformal_moment(
            j=i, eta=eta, t=t, mu=mu,
            particle=particle, moment_type=moment_type,
            moment_label=moment_label,
            evolution_order=evolution_order,
            error_type=error_type
        )
        frac = (1 - min(abs(error_term),abs(central_term)) / max(abs(error_term),abs(central_term)))
        return frac
    
    j_b, _ = get_j_base(particle,moment_type,moment_label)
    j_b = int(np.ceil(j_b))

    # Use first 10 moments to get estimate
    max_iter = 10
    i_values = np.array([2*k + j_b for k in range(max_iter)])
    gpd_err = 0
    results = compute_rel_error(i_values)

    # Take average error
    gpd_err = sum(results)/len(results)
    # Return average error in percent
    gpd_err = 1 + hp.error_sign(gpd_err,error_type)

    return gpd_err

gpd_errors = {}
for particle,moment_type, moment_label, evolution_order, error_type in product(
    cfg.PARTICLES, cfg.MOMENTS, cfg.LABELS, cfg.ORDERS, cfg.ERRORS):
    if particle == "gluon" and moment_type != "singlet":
        continue
    params = {
        "particle": particle,
        "moment_type": moment_type,
        "moment_label": moment_label,
        "evolution_order": evolution_order,
        "error_type": error_type,
    }
    selected_triples = [
        (eta, t, mu)
        for eta, t, mu in zip(cfg.ETA_ARRAY, cfg.T_ARRAY, cfg.MU_ARRAY)
    ]
    gpd_errors[(particle,moment_type,moment_label, evolution_order, error_type)] = [
        estimate_gpd_error(eta, t, mu, **params)
        for eta, t, mu in selected_triples
    ]

def mellin_barnes_gpd(x, eta, t, mu,  A0=1 ,particle = "quark", moment_type="non_singlet_isovector",moment_label="A",evolution_order="nlo", error_type="central",real_imag ="real",j_max = 15, n_jobs=1):
    """
    Numerically evaluate the Mellin-Barnes integral parallel to the imaginary axis 
    to obtain the corresponding GPD.

    Parameters
    ----------
    x : float
        Parton momentum fraction.
    eta : float
        Skewness parameter.
    t : float
        Mandelstam t.
    mu : float
        Resolution scale.
    A0 : float, optional
        Overall scale.
    particle : str, optional
        "quark" or "gluon". Default is "quark".
    moment_type : str, optional
        non_singlet_isovector, non_singlet_isoscalar, or singlet.
    moment_label : str, optional
        A(tilde), B(tilde) depending on H(tilde) or E(tilde) GPD etc.
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"
    real_imag : str, optional
        Choose to compute "real", "imag", or "both" parts of the result.
    j_max : float, optional
        Integration range parallel to the imaginary axis.
    n_jobs : int, optional
        Number of subregions (and processes) the integral is split into.
    n_k : int, optional
        Number of sampling points within the interval [-j_max, j_max].

    Returns
    -------
    complex or float
        The value of the Mellin-Barnes integral with real and imaginary parts.

    Notes
    -----
    For low `x` and/or `eta`, it is recommended to divide the integration region using n_jobs.
    """

    hp.check_particle_type(particle)
    hp.check_error_type(error_type)
    hp.check_evolution_order(evolution_order)
    hp.check_moment_type_label(moment_type,moment_label)

    # When error_type == "central" gpd_rel_error is 1, 
    # so we don't need to distinguish
    key = (particle, moment_type, moment_label, evolution_order, error_type)
    if key not in gpd_errors:
        raise ValueError("No error estimates for GPDs have been computed. Modify PARTICLES, MOMENTS,... in config file")
    selected_triples = [
        (eta_, t_, mu_)
        for eta_, t_, mu_ in zip(cfg.ETA_ARRAY, cfg.T_ARRAY, cfg.MU_ARRAY)
    ]
    index = selected_triples.index((eta, t, mu))
    gpd_rel_error = gpd_errors[key][index]

    j_base, parity = get_j_base(particle,moment_type,moment_label)

    @hp.mpmath_vectorize
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
        sin_term = mp.sin(mp.pi * z)
        # We double the sine here since its (-1)**(2 * j) from the non-diagonal evolution
        sin2_term = mp.sin(2 * mp.pi * z)/2
        pw_val = conformal_partial_wave(z, x, eta, particle, parity)

        if particle == "quark":
            if moment_type == "singlet":
                mom_val = evolve_quark_singlet(z,eta,t,mu,A0,moment_label,evolution_order,error_type="central")
            else:
                mom_val = evolve_quark_non_singlet(z,eta,t,mu,A0,moment_type,moment_label,evolution_order,error_type="central")
        else:
            mom_val = evolve_gluon_singlet(z,eta,t,mu,A0,moment_label,evolution_order,error_type="central")
            # (-1) from shift in Sommerfeld-Watson transform
            mom_val = tuple(-x for x in mom_val)
        # Now re-introduce (1 + (-1)**j)
        # First piece contains diagonal + non-alternating non-diagonal piece
        # second piece is alternating non-diagonal piece
        result = -.5j * dz * pw_val * (mom_val[0] / sin_term + mom_val[1] / sin2_term)
        if real_imag == 'real':
            return np.float64(mp.re(result))
        elif real_imag == 'imag':
            return np.float64(mp.im(result))
        elif real_imag == 'both':
            return result
        else:
            raise ValueError("real_imag must be either 'real', 'imag', or 'both'")
        
    def find_integration_bound(integrand, j_max, tolerance=1e-2, step=10, max_iterations=50):
        """
        Find an appropriate upper integration bound

        Parameters
        ----------
        integrand : function
            The function to be integrated.
        tolerance : str, optional
            The desired tolerance for the integrand's absolute value. Default is "1e-2".
        step : float, optional
            The increment to increase the integration bound in each step. Default is 10.
        max_iterations : int, optional
            The maximum number of iterations to perform. Default is 50.

        Returns
        -------
        float
            The determined upper integration bound.

        Raises
        ------
        ValueError
            If the maximum number of iterations is reached without finding a suitable bound.
        """

        iterations = 0

        while abs(integrand(j_max, "real")) > tolerance and iterations < max_iterations:
            j_max += step
            iterations += 1

        if iterations == max_iterations:
            raise ValueError(f"Maximum number of iterations reached at (x,eta,t) = {x,eta,t} without finding a suitable bound. Increase initial value of j_max")

        # Check for rapid oscillations
        if abs(integrand(j_max,  "real") - integrand(j_max + 2,  "real")) > tolerance:
            while abs(integrand(j_max,  "real")) > tolerance and iterations < max_iterations:
                j_max += step
                iterations += 1

            if iterations == max_iterations:
                raise ValueError(f"Maximum number of iterations reached at (x,eta,t) = {x,eta,t} without finding a suitable bound. Increase initial value of j_max")
        if j_max > 250:
            print(f"Warning j_max={j_max} is large, adjust corresponding base value in get_j_base")
        return j_max

    # Function to integrate over a subinterval of k 
    def integrate_subinterval(k_values, real_imag):
        """
        Integrate the integrand over the specified subinterval and return either 
        the real, imaginary part, or both.

        Parameters
        ----------
        k_values : array-like
            A list or array containing the minimum and maximum k values.
        real_imag : str
            Specifies whether to return "real", "imag", or "both".

        Returns
        -------
        tuple
            If `real_imag` is "real":
                (real_part, error)
            If `real_imag` is "imag":
                (imag_part, error)
            If `real_imag` is "both":
                (real_part, real_error, imag_part, imag_error)
        """

        k_min = k_values[0]
        k_max = k_values[-1]

        if real_imag == 'real':
            # integral, error = quad(lambda k: integrand(k, 'real'), k_min, k_max, limit = 200)
            integral, _ = fixed_quad(lambda k: integrand(k, 'real'), k_min, k_max, n = 150)
            integral_low, _ = fixed_quad(lambda k: integrand(k, 'real'), k_min, k_max, n = 80)
            error = abs(integral-integral_low)
            # Use symmetry of the real part of the integrand
            integral *= 2
            error *= 2
            return integral, error
        elif real_imag == 'imag':
            # integral, error = quad(lambda k: integrand(k, 'imag'), k_min, k_max, limit = 100)
            integral, _ = fixed_quad(lambda k: integrand(k, 'imag'), k_min, k_max, n = 150)
            integral_low, _ = fixed_quad(lambda k: integrand(k, 'imag'), k_min, k_max, n = 80)
            error = abs(integral-integral_low)
            return integral, error
        elif real_imag == 'both':
            # integral, error = mp.quad(lambda k: integrand(k, 'both'), -k_max, k_max, error=True)
            integral, _ = fixed_quad(lambda k: integrand(k, 'both'), k_min, k_max, n = 150)
            integral_low, _ = fixed_quad(lambda k: integrand(k, 'both'), k_min, k_max, n = 80)
            real_integral, imag_integral = np.float64(mp.re(integral)), np.float64(mp.im(integral))
            error = abs(integral.real-integral_low.real) + 1j * abs(integral.imag-integral_low.imag)
            real_error, imag_error = np.float64(mp.re(error)), np.float64(mp.im(error))
            # real_integral, real_error = quad(lambda k: integrand(k, 'real'), k_min, k_max, limit = 200)
            # imag_integral, imag_error = quad(lambda k: integrand(k, 'imag'), k_min, k_max, limit = 200)
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
    if np.abs(error) > 1e-2:
        print(f"Warning: Large error estimate for (x,eta,t)={x,eta,t}: {error}")
    # Mutliply with the relative error
    integral *= gpd_rel_error
    return float(integral) if real_imag in ["real","imag"] else float(integral.real) + 1j * float(integral.imag)

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

def spin_orbit_corelation(eta,t,mu, A0 = 1, particle="quark",moment_type="non_singlet_isovector",evolution_order="nlo"):
    """ 
    Returns the spin orbit correlation C_z of moment_type including errors

    Parameters
    ----------
    eta : float
        Skewness parameter.
    t : float
        Mandelstam t.
    mu : float
        Resolution scale.
    A0 : float, optional
        Overall scale.
    particle : str, optional
        "quark" or "gluon". Default is "quark".
    moment_type : str, optional
        non_singlet_isovector, non_singlet_isoscalar, or singlet.
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo" 

    Returns
    -------
    tuple
        A tuple containing the result, upper error estimate, and lower error estimate.

    """

    if moment_type not in ["singlet",
                           "non_singlet_isoscalar",
                           "non_singlet_isovector"]:
        raise ValueError(f"Wrong moment type {moment_type}")

    term_1 = evolve_conformal_moment(2,eta,t,mu,A0,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,error_type="central")
    term_1_plus = evolve_conformal_moment(2,eta,t,mu,A0,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,error_type="plus")
    term_1_minus = evolve_conformal_moment(2,eta,t,mu,A0,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,error_type="minus")

    term_2 = evolve_conformal_moment(1,eta,t,mu,A0,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,error_type="central")
    term_2_plus = evolve_conformal_moment(1,eta,t,mu,A0,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,error_type="plus")
    term_2_minus = evolve_conformal_moment(1,eta,t,mu,A0,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,error_type="minus")

    result = (term_1 - term_2)/2
    error_plus = np.sqrt((term_1_plus-term_1)**2+(term_2_plus-term_2)**2)/2
    error_minus = np.sqrt((term_1_minus-term_1)**2+(term_2_minus-term_2)**2)/2

    return result, error_plus, error_minus

def total_spin(eta,t,mu,A0=1,particle="quark",moment_type="non_singlet_isovector",evolution_order="nlo"):
    """ 
    Returns the total spin J of moment_type including errors

    Parameters
    ----------
    eta : float
        Skewness parameter.
    t : float
        Mandelstam t.
    mu : float
        Resolution scale.
    A0 : float, optional
        Overall scale.
    particle : str, optional
        "quark" or "gluon". Default is "quark".
    moment_type : str, optional
        non_singlet_isovector, non_singlet_isoscalar, or singlet.
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"  
        
    Returns
    -------
    tuple
        A tuple containing the result, upper error estimate, and lower error estimate.

    """
    if moment_type not in ["singlet",
                           "non_singlet_isoscalar",
                           "non_singlet_isovector"]:
        raise ValueError(f"Wrong moment type {moment_type}")

    term_1 = evolve_conformal_moment(2,eta,t,mu,A0,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,error_type="central")
    term_2 = evolve_conformal_moment(2,eta,t,mu,A0,particle=particle,moment_type=moment_type,moment_label="B",evolution_order=evolution_order,error_type="central")
    result = (term_1 + term_2)/2

    term_1_plus = evolve_conformal_moment(2,eta,t,mu,A0,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,error_type="plus")
    term_2_plus = evolve_conformal_moment(2,eta,t,mu,A0,particle=particle,moment_type=moment_type,moment_label="B",evolution_order=evolution_order,error_type="plus")
    error_plus = np.sqrt((term_1_plus-term_1)**2+(term_2_plus-term_2)**2)/2

    term_1_minus = evolve_conformal_moment(2,eta,t,mu,A0,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,error_type="minus")
    term_2_minus = evolve_conformal_moment(2,eta,t,mu,A0,particle=particle,moment_type=moment_type,moment_label="B",evolution_order=evolution_order,error_type="minus")
    error_minus = np.sqrt((term_1_minus-term_1)**2+(term_2_minus-term_2)**2)/2

    return result, error_plus, error_minus

def orbital_angular_momentum(eta,t,mu,A0=1, particle="quark",moment_type="non_singlet_isovector",evolution_order="nlo"):
    """ 
    Returns the orbital angular momentum L_z of moment_type including errors

    Parameters
    ----------
    eta : float
        Skewness parameter.
    t : float
        Mandelstam t.
    mu : float
        Resolution scale.
    A0 : float, optional
        Overall scale.
    particle : str, optional
        "quark" or "gluon". Default is "quark".
    moment_type : str, optional
        non_singlet_isovector, non_singlet_isoscalar, or singlet.
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
        
    Returns
    -------
    tuple
        A tuple containing the result, upper error estimate, and lower error estimate.

    """
    hp.check_particle_type(particle)
    if moment_type not in ["singlet",
                           "non_singlet_isoscalar",
                           "non_singlet_isovector"]:
        raise ValueError(f"Wrong moment type {moment_type}")

    term_1 = evolve_conformal_moment(2,eta,t,mu,A0,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,error_type="central")
    term_2 = evolve_conformal_moment(2,eta,t,mu,A0,particle=particle,moment_type=moment_type,moment_label="B",evolution_order=evolution_order,error_type="central")

    term_1_plus = evolve_conformal_moment(2,eta,t,mu,A0,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,error_type="plus")
    term_2_plus = evolve_conformal_moment(2,eta,t,mu,A0,particle=particle,moment_type=moment_type,moment_label="B",evolution_order=evolution_order,error_type="plus")

    term_1_minus = evolve_conformal_moment(2,eta,t,mu,A0,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,error_type="minus")
    term_2_minus = evolve_conformal_moment(2,eta,t,mu,A0,particle=particle,moment_type=moment_type,moment_label="B",evolution_order=evolution_order,error_type="minus")

    term_3 = evolve_conformal_moment(1,eta,t,mu,A0,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,error_type="central")
    term_3_plus = evolve_conformal_moment(1,eta,t,mu,A0,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,error_type="plus")
    term_3_minus = evolve_conformal_moment(1,eta,t,mu,A0,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,error_type="minus")

    result = (term_1 + term_2)/2 - term_3/2
    error_plus = np.sqrt((term_1_plus-term_1)**2+(term_2_plus-term_2)**2+(term_3-term_3_plus)**2)/2
    error_minus = np.sqrt((term_1_minus-term_1)**2+(term_2_minus-term_2)**2+(term_3-term_3_minus)**2)/2

    return result, error_plus, error_minus

def quark_gluon_helicity(eta,t,mu,A0=1, particle="quark",moment_type="non_singlet_isovector",evolution_order="nlo"):
    """ 
    Returns the helicity of moment_type including errors

    Parameters
    ----------
    eta : float
        Skewness parameter.
    t : float
        Mandelstam t.
    mu : float
        Resolution scale.
    A0 : float, optional
        Overall scale.
    particle : str, optional
        "quark" or "gluon". Default is "quark".
    moment_type : str, optional
        non_singlet_isovector, non_singlet_isoscalar, or singlet.
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
        
    Returns
    -------
    tuple
        A tuple containing the result, upper error estimate, and lower error estimate.

    """
    hp.check_particle_type(particle)
    if moment_type not in ["singlet","non_singlet_isoscalar","non_singlet_isovector"]:
        raise ValueError(f"Wrong moment type {moment_type}")
    if particle == "gluon" and moment_type != "singlet":
        raise ValueError(f"Wrong moment_type {moment_type} for {particle}")

    result = evolve_conformal_moment(1,eta,t,mu,A0,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,error_type="central")/2

    term_1 = evolve_conformal_moment(1,eta,t,mu,A0,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,error_type="plus")/2
    error_plus = abs(result - term_1)

    term_1 = evolve_conformal_moment(1,eta,t,mu,A0,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,error_type="minus")/2
    error_minus = abs(result - term_1)
    return result, error_plus, error_minus

def quark_helicity(eta,t,mu,A0=1, moment_type="non_singlet_isovector",evolution_order="nlo"):
    """
    Helper function to get quark helicity. For documentation see quark_gluon_helicity
    """
    result, error_plus, error_minus = quark_gluon_helicity(eta,t,mu,A0,particle="quark",moment_type=moment_type,evolution_order=evolution_order)
    return result, error_plus, error_minus

def gluon_helicity(eta,t,mu,A0=1,evolution_order="nlo"):
    """
    Helper function to get gluon helicity. For documentation see quark_gluon_helicity
    """
    result, error_plus, error_minus = quark_gluon_helicity(eta,t,mu,A0,particle="gluon",moment_type="singlet",evolution_order=evolution_order)
    return result, error_plus, error_minus