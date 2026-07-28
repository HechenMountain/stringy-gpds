import numpy as np
import mpmath as mp
import os
import csv

from . import helpers as hp
from . import core
from . import adim
from . import regge as reg
from . import config as cfg
from .unpolarized_pdf import get_alpha_s

from scipy.optimize import curve_fit, least_squares
from scipy.special import betaln
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def dipole_fit_lattice_moments(n,particle,moment_type,moment_label,pub_id,error_type="central",plot_fit=False, write_to_file=True):
    """
    Generates a dipole fit to the corresponding lattice moment

    Parameters
    ----------
    n : int
        Conformal spin.
    particle : str
        "quark" or "gluon". Default is "quark".
    moment_type : str
        non_singlet_isovector, non_singlet_isoscalar, or singlet.
    moment_label : str
        A(tilde), B(tilde) depending on H(tilde) or E(tilde) GPD etc.
    pub_id: str
        ArXiv identifier
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"
    plot_fit : bool, optional
        Whether to plot the fit vs the data. Default is False
    write_to_file : bool, optional
        If True, writes the fit results to 'dipole_moments_pub_id_eta_t_mu.csv'.
    """
    # Accessor functions for -t, values, and errors
    def t_values(moment_type, moment_label, pub_id):
        """Return the -t values for a given moment type, label, and publication ID."""
        data, n_to_row_map = hp.load_lattice_moment_data(particle,moment_type, moment_label, pub_id)

        if data is None and n_to_row_map is None:
            print(f"No data found for {moment_type} {moment_label} {pub_id}. Skipping.")
            return None 
        
        if data is not None:
            # Safely access data[:, 0] since data is not None
            return data[:, 0]
        else:
            print(f"Data is None for {moment_type} {moment_label} {pub_id}. Skipping.")
        return None

    def dipole_form(t, A_D, m_D2):
        return A_D / (1 - t / m_D2)**2

    data, n_to_row_map = hp.load_lattice_moment_data(particle,moment_type, moment_label, pub_id)
    if data is None or n not in n_to_row_map:
        raise ValueError(f"No data on file system for {particle} {moment_type} {moment_label} in {pub_id}")
    
    t_vals = -t_values(moment_type, moment_label, pub_id)
    # Extract values and errors
    Fn0_vals = hp.Fn0_values(n, particle, moment_type, moment_label, pub_id)
    if error_type != "central":
        Fn0_errs = hp.Fn0_errors(n, particle, moment_type, moment_label, pub_id)
        Fn0_vals += hp.error_sign(Fn0_errs,error_type)

    # Initial parameter guess: A_D ~ max(f_vals), m_D2 ~ 1.0 
    initial_guess = [np.max(data), 1.0]
    bounds = ([-np.inf, 0], [np.inf, np.inf])

    popt, pcov = curve_fit(dipole_form, t_vals, Fn0_vals, p0=initial_guess,bounds=bounds)
    AD_fit, m_D2_fit = popt
    if plot_fit:
        t_fit = np.linspace(0, -max(abs(t_vals)), 100)
        f_fit = dipole_form(t_fit, *popt)
        # Plot data and fit
        plt.figure(figsize=(8, 5))
        plt.plot(-t_vals, Fn0_vals, 'o', label='Data')
        plt.plot(-t_fit, f_fit, '-')
        plt.xlabel('-t')
        plt.ylabel('f(t)')
        plt.ylim([0,1.1 * AD_fit])
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    if write_to_file:
        if error_type != "central":
            file_path = cfg.MOMENTUM_SPACE_MOMENTS_PATH / f"dipole_moments_{pub_id}_{error_type}.csv"
        else:
            file_path = cfg.MOMENTUM_SPACE_MOMENTS_PATH / f"dipole_moments_{pub_id}.csv"
        hp.update_dipole_csv(
            file_path=file_path,
            n=n,
            particle=particle,
            moment_type=moment_type,
            moment_label=moment_label,
            # use pub_id as key
            evolution_order=pub_id,
            A_D=AD_fit,
            m_D2=m_D2_fit,
            lattice=True
        )

def dipole_fit_moment(n,eta,mu,particle="quark",moment_type="non_singlet_isovector",moment_label="A",evolution_order="nlo",error_type="central",plot_fit=False,write_to_file=True):
    """
    Generates a dipole fit to the first moment of the corresponding singlet GPD.

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
    plot_fit : bool, optional
        Whether to plot the fit vs the data. Default is False
    write_to_file : bool, optional
        If True, writes the fit results to 'dipole_moments_eta_t_mu.csv'.

    Returns
    -------
    tuple
        A tuple containing the dipole scale and mass squared (float).
    """

    gpd_label = cfg.INVERTED_GPD_LABEL_MAP.get(moment_label)
    if gpd_label is None:
        print(f"Value {moment_label} not found in GPD_LABEL_MAP - abort")
        return
    def dipole_form(t, A_D, m_D2):
        return A_D / (1 - t / m_D2)**2
    n_int = os.cpu_count() if os.cpu_count() > 10 else 10

    t_vals = np.linspace(-1e-6,-10,n_int)
    f_vals = Parallel(n_jobs=-1)(
            delayed(lambda t: float(core.evolve_conformal_moment(n, eta, t, mu, 1,
                                                                  particle=particle, moment_type=moment_type,
                                                                  moment_label=moment_label, evolution_order=evolution_order,
                                                                  error_type=error_type).real))(t)
            for t in t_vals
        )
    f_vals = np.array(f_vals)
    # check for crossing behavior
    num_pos = np.sum(f_vals > 0)
    num_neg = np.sum(f_vals < 0) 

    if num_pos and num_neg:
        print("Warning: zero-crossing detected")
        print(f"(n, eta, mu, particle, moment_type, moment_label, evolution_order,error_type) = {n, eta, mu, particle, moment_type, moment_label, evolution_order,error_type}.")
        print(f"Positive values:{num_pos}, Negative values: {num_neg}")

    # Discard either region
    if num_pos >= num_neg:
        mask = f_vals > 0
    else:
        mask = f_vals < 0

    t_vals = t_vals[mask]
    f_vals = f_vals[mask]

    if all(val == 0 for val in f_vals):
        AD_fit, m_D2_fit = 0, 1
    else:
        # Initial parameter guess: A_D ~ max(f_vals), m_D2 ~ 1.0 
        initial_guess = [np.max(f_vals), 1.0]
        bounds = ([-np.inf, 0], [np.inf, np.inf])
        popt, pcov = curve_fit(dipole_form, t_vals, f_vals, p0=initial_guess,bounds=bounds)
        AD_fit, m_D2_fit = popt

    if plot_fit:
        t_fit = np.linspace(0, -10, 100)
        f_fit = dipole_form(t_fit, *popt)
        # Plot data and fit
        plt.figure(figsize=(8, 5))
        plt.plot(-t_vals, f_vals, 'o', label='Data')
        plt.plot(-t_fit, f_fit, '-')
        plt.title(f'{moment_type} {particle} {moment_label} {error_type}')
        plt.xlabel('-t')
        plt.ylabel('f(t)')
        plt.ylim([0,1.1 * AD_fit])
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    if write_to_file:
        prefix = "dipole_moments"
        file_path = hp.generate_filename(eta,0,mu,cfg.MOMENTUM_SPACE_MOMENTS_PATH / prefix,error_type)
        hp.update_dipole_csv(
            file_path=file_path,
            n=n,
            particle=particle,
            moment_type=moment_type,
            moment_label=moment_label,
            evolution_order=evolution_order,
            A_D=AD_fit,
            m_D2=m_D2_fit
        )
    return AD_fit, m_D2_fit


def quark_singlet_regge_fit(n,eta,t,alpha_prime_ud, alpha_prime_s,norm_A, norm_D,moment_label="A",evolution_order="nlo",error_type="central"):
    """
    Reggeized quark singlet moment with unfixed Regge slopes for fit procedure.

    Parameters
    ----------
    n : int
        Conformal spin.
    eta : float
        Skewness parameter.
    t : float
        Mandelstam t.
    alpha_prime_ud : float
        A-term Regge slope
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
    float
        The Reggeized moment for the given parameters
    """
    # Check type
    hp.check_error_type(error_type)
    hp.check_moment_type_label("singlet",moment_label)
    hp.check_evolution_order(evolution_order)

    if moment_label == "B":
        prf = -1
    else:
        prf = +1

    term_1, error_1 = core.quark_singlet_regge_A(n,eta,t,alpha_prime_ud,moment_label,evolution_order,error_type)
    term_2, error_2 = core.quark_singlet_regge_D(n,eta,t,alpha_prime_ud,alpha_prime_s,moment_label,evolution_order,error_type)
    sum_squared = norm_A**2 * error_1**2 + norm_D**2 * error_2**2
    # error = np.frompyfunc(abs, 1, 1)(mp.sqrt(sum_squared))
    error = abs(mp.sqrt(sum_squared))
    result = norm_A * term_1 + norm_D * prf * term_2

    return result, error

def gluon_singlet_regge_fit(n,eta,t,alpha_prime_T, alpha_prime_S,norm_A, norm_D ,moment_label="A", evolution_order="nlo",error_type="central"):
    """
    Reggeized quark singlet moment with unfixed Regge slopes for fit procedure.

    Parameters
    ----------
    n : int
        Conformal spin.
    eta : float
        Skewness parameter.
    t : float
        Mandelstam t.
    alpha_prime_T : float
        A-term Regge slope
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
    float
        The Reggeized moment for the given parameters
    """
    # Check type
    hp.check_error_type(error_type)
    hp.check_moment_type_label("singlet",moment_label)

    if moment_label == "B":
        prf = -1
    else:
        prf = +1

    term_1, error_1 = core.gluon_singlet_regge_A(n,eta,t,alpha_prime_T,moment_label,evolution_order,error_type)
    if eta == 0:
        result = norm_A * term_1
        error = norm_A * error_1
    else :
        term_2, error_2 = core.gluon_singlet_regge_D(n,eta,t,alpha_prime_T,alpha_prime_S,moment_label,evolution_order,error_type)
        sum_squared = norm_A**2 * error_1**2 + norm_D**2 * error_2**2
        # error = np.frompyfunc(abs, 1, 1)(mp.sqrt(sum_squared))
        error = abs(mp.sqrt(sum_squared))
        result = norm_A * term_1 + norm_D * prf * term_2
    return result, error

def singlet_moment_fit(j,eta,t,alpha_prime_ud, alpha_prime_s,norm_Aq, norm_Dq,alpha_prime_T, alpha_prime_S,norm_Ag, norm_Dg,
                       moment_label="A",solution="+",evolution_order="nlo",error_type="central",interpolation=True):
    """
    Reggeized singlet moment with unfixed Regge slopes for fit procedure.

    Parameters
    ----------
    n : int
        Conformal spin.
    eta : float
        Skewness parameter.
    t : float
        Mandelstam t.
    alpha_prime_ud : float
        Quark A-term Regge slope
    alpha_prime_s : float
        Quark D-term Regge slope
    norm_Aq : float
        Quark singlet A-term norm
    norm_Dq : float
        Quark singlet D-term norm
    alpha_prime_T : float
        Gluon A-term Regge slope
    alpha_prime_S : float
        Gluon D-term Regge slope
    norm_Ag : float
        Gluon singlet A-term norm
    norm_Dg : float
        Gluon singlet D-term norm
    moment_label : str, optional
        A(tilde), B(tilde) depending on H(tilde) or E(tilde) GPD etc. Default is "A".
    solution : str, optional
        "+" or "-" solution
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"
    interpolation : bool, optional
        Whether to interpolate anomalous dimension

    Returns
    -------
    float
        The Reggeized moment for the given parameters

    Note
    ----
    Returns 0 if the moment_label = "B", in accordance with holography and quark model considerations. 
    Otherwise it returns the diagonal combination of quark + gluon moment. Error for singlet_moment at j = 1
    for solution "-" unreliable because of pole in gamma. Better to reconstruct evolved moment from GPD.
    """
    if moment_label == "B":
        return 0, 0
    # Check type
    hp.check_error_type(error_type)

    evolve_type = hp.get_evolve_type(moment_label)

    # Switch sign
    if solution == "+":
        solution = "-"
    elif solution == "-":
        solution = "+"
    else:
        raise ValueError("Invalid solution type. Use '+' or '-'.")

    index  = 0 if solution == "+" else 1
    ga_pm = adim.gamma_pm(j-1,evolve_type,solution,interpolation=interpolation)[index]

    quark_prf = .5 
    quark_in, quark_in_error = quark_singlet_regge_fit(j,eta,t,alpha_prime_ud, alpha_prime_s,norm_Aq, norm_Dq,moment_label,evolution_order,error_type)
    # Note: j/6 already included in gamma_qg and gamma_gg definitions
    gluon_prf = .5 * (adim.gamma_qg(j-1,evolve_type,"lo",interpolation=interpolation)/
                    (adim.gamma_qq(j-1,"singlet",evolve_type,"lo",interpolation=interpolation)-ga_pm))
    gluon_in, gluon_in_error = gluon_singlet_regge_fit(j,eta,t,alpha_prime_T, alpha_prime_S,norm_Ag, norm_Dg,moment_label,evolution_order,error_type)
    # print(solution,gluon_prf)
    sum_squared = quark_prf**2 * quark_in_error**2 + gluon_prf**2*gluon_in_error**2
    # print("->",quark_in,quark_in_error,quark_prf)
    # print("->",gluon_in,gluon_in_error,gluon_prf)
    # error = np.frompyfunc(abs, 1, 1)(mp.sqrt(sum_squared))
    error = abs(mp.sqrt(sum_squared))
    result = quark_prf * quark_in + gluon_prf * gluon_in
    return result, error

def evolve_singlet_fit(eta,t,mu,alpha_prime_ud, alpha_prime_s,norm_Aq, norm_Dq,alpha_prime_T, alpha_prime_S,norm_Ag, norm_Dg,
                       particle="quark",moment_label ="A", evolution_order = "nlo", error_type = "central",interpolation=True):  
    """
    Reggeized evolved singlet moment with unfixed Regge slopes for fit procedure.

    Parameters
    ----------
    eta : float
        Skewness parameter.
    t : float
        Mandelstam t.
    mu : float
        Resolution scale mu
    alpha_prime_ud : float
        Quark A-term Regge slope
    alpha_prime_s : float
        Quark D-term Regge slope
    norm_Aq : float
        Quark singlet A-term norm
    norm_Dq : float
        Quark singlet D-term norm
    alpha_prime_T : float
        Gluon A-term Regge slope
    alpha_prime_S : float
        Gluon D-term Regge slope
    norm_Ag : float
        Gluon singlet A-term norm
    norm_Dg : float
        Gluon singlet D-term norm
    particle : str, optional
        "quark" or "gluon"
    moment_label : str, optional
        A(tilde), B(tilde) depending on H(tilde) or E(tilde) GPD etc. Default is "A".
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"
    interpolation : bool, optional
        Whether to interpolate anomalous dimension

    Returns
    -------
    float
        The evolved Reggeized moment for the given parameters
    """
    hp.check_particle_type(particle)
    hp.check_moment_type_label("singlet",moment_label)
    hp.check_error_type(error_type)
    hp.check_evolution_order(evolution_order)

    j = 2

    # Set parameters
    
    # Extract fixed quantities
    alpha_s_in = get_alpha_s(evolution_order)
    alpha_s_evolved = core.evolve_alpha_s(mu,evolution_order)

    evolve_type = hp.get_evolve_type(moment_label)

    ga_qq = adim.gamma_qq(j-1,"singlet",evolve_type,evolution_order="nlo",interpolation=interpolation)

    # Roots  of lo anomalous dimensions
    ga_p, ga_m = adim.gamma_pm(j-1,evolve_type,interpolation=interpolation)
    moment_in_p, error_p = singlet_moment_fit(j,eta,t,alpha_prime_ud, alpha_prime_s,norm_Aq, norm_Dq,alpha_prime_T, alpha_prime_S,norm_Ag, norm_Dg,
                                                moment_label,"+",evolution_order,error_type,interpolation=interpolation)
    moment_in_m, error_m = singlet_moment_fit(j,eta,t,alpha_prime_ud, alpha_prime_s,norm_Aq, norm_Dq,alpha_prime_T, alpha_prime_S,norm_Ag, norm_Dg,
                                                moment_label,"-",evolution_order,error_type,interpolation=interpolation)
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
    
    # Functions appearing in evolution
    def get_gammas(solution):
        # switch + <-> - when necessary
        if solution == "+":
            return ga_p, ga_m
        elif solution == "-": 
            return ga_m, ga_p
        else:
            raise ValueError(f"Wrong solution type: {solution}")
    
    def A_lo_quark(solution):
        # The switch also takes care of the relative minus sign
        ga_p, ga_m = get_gammas(solution)
        result = (ga_qq - ga_m)/(ga_p - ga_m) * alpha_frac**(ga_p/cfg.BETA_0) * 2
        # print(ga_p,ga_m,(ga_qq - ga_m)/(ga_p - ga_m))
        return result
    
    def A_lo_gluon(solution):
        ga_p, ga_m = get_gammas(solution)
        result = ga_gq/(ga_p - ga_m) * alpha_frac**(ga_p/cfg.BETA_0) * 2
        # print("gluon",solution,result/moment)
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
        term2 = (1 - alpha_frac**((ga_m - ga_p + cfg.BETA_0)/cfg.BETA_0)) * alpha_frac**(ga_p/cfg.BETA_0)
        term3 = (ga_gq  * (r_qq * (ga_qq - ga_m) + r_qg * ga_gq) + (ga_gg - ga_p) * (r_gq * (ga_qq - ga_m) + r_gg * ga_gq) )
        result = term1 * term2 * term3
        return result

    if particle == "quark":
        result = A_lo_quark("+") * moment_in_p + A_lo_quark("-") * moment_in_m
        sum_squared =  (A_lo_quark("+") * error_p)**2 + (A_lo_quark("-") * error_m)**2
        # error = np.frompyfunc(abs, 1, 1)(mp.sqrt(sum_squared))
        error = abs(mp.sqrt(sum_squared))
        result += hp.error_sign(error,error_type)
        if evolution_order == "nlo":
            plus_terms = A_quark_nlo("+") + B_quark_nlo("+")
            minus_terms = A_quark_nlo("-") + B_quark_nlo("-")
            diagonal_terms = plus_terms * moment_in_p + minus_terms * moment_in_m
            sum_squared = plus_terms**2 * error_p**2 + minus_terms**2 * error_m**2
            # diagonal_errors = np.frompyfunc(abs, 1, 1)(mp.sqrt(sum_squared))
            diagonal_errors = abs(mp.sqrt(sum_squared))
            error = diagonal_errors 
            result += diagonal_terms + hp.error_sign(error,error_type)
    if particle == "gluon":
        result = A_lo_gluon("+") * moment_in_p + A_lo_gluon("-") * moment_in_m
        sum_squared =  (A_lo_gluon("+") * error_p)**2 + (A_lo_gluon("-") * error_m)**2
        # error = np.frompyfunc(abs, 1, 1)(mp.sqrt(sum_squared))
        error = abs(mp.sqrt(sum_squared))
        result += hp.error_sign(error,error_type)
        if evolution_order == "nlo":
            plus_terms = A_gluon_nlo("+") + B_gluon_nlo("+")
            minus_terms = A_gluon_nlo("-")  + B_gluon_nlo("-")
            diagonal_terms =  plus_terms * moment_in_p + minus_terms * moment_in_m
            sum_squared = plus_terms**2 * error_p**2 + minus_terms**2 * error_m**2
            # diagonal_errors = np.frompyfunc(abs, 1, 1)(mp.sqrt(sum_squared))
            diagonal_errors = abs(mp.sqrt(sum_squared))
            error = diagonal_errors
            result += diagonal_terms + hp.error_sign(error,error_type)

    # Return real value when called for real j
    if mp.im(result) == 0:
        return np.float64(mp.re(result))
    return result

def evolve_singlet_D_fit(eta,t,mu,alpha_prime_ud, alpha_prime_s,norm_Aq, norm_Dq,alpha_prime_T, alpha_prime_S,norm_Ag, norm_Dg,
                       particle="quark",moment_label ="A", evolution_order = "nlo", error_type = "central",interpolation=True):
    """
    Reggeized evolved singlet D moment with unfixed Regge slopes for fit procedure.

    Parameters
    ----------
    eta : float
        Skewness parameter.
    t : float
        Mandelstam t.
    mu : float
        Resolution scale mu
    alpha_prime_ud : float
        Quark A-term Regge slope
    alpha_prime_s : float
        Quark D-term Regge slope
    norm_Aq : float
        Quark singlet A-term norm
    norm_Dq : float
        Quark singlet D-term norm
    alpha_prime_T : float
        Gluon A-term Regge slope
    alpha_prime_S : float
        Gluon D-term Regge slope
    norm_Ag : float
        Gluon singlet A-term norm
    norm_Dg : float
        Gluon singlet D-term norm
    particle : str, optional
        "quark" or "gluon"
    moment_label : str, optional
        A(tilde), B(tilde) depending on H(tilde) or E(tilde) GPD etc. Default is "A".
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"
    interpolation : bool, optional
        Whether to interpolate anomalous dimension

    Returns
    -------
    float
        The evolved Reggeized D moment for the given parameters
    """
    hp.check_particle_type(particle)
    hp.check_moment_type_label("singlet",moment_label)
    term_1 = evolve_singlet_fit(eta,t,mu,alpha_prime_ud, alpha_prime_s,norm_Aq, norm_Dq,alpha_prime_T, alpha_prime_S,norm_Ag, norm_Dg,
                       particle=particle,moment_label=moment_label, evolution_order = evolution_order, error_type = error_type,interpolation=interpolation)
    term_2 = evolve_singlet_fit(0,t,mu,alpha_prime_ud, alpha_prime_s,norm_Aq, norm_Dq,alpha_prime_T, alpha_prime_S,norm_Ag, norm_Dg,
                       particle=particle,moment_label=moment_label, evolution_order = evolution_order, error_type = error_type,interpolation=interpolation)
    result = (term_1-term_2)/eta**2
    # Discard numerically tiny imaginary residue from interpolated anomalous dimensions
    if mp.im(result) != 0:
        result = np.float64(mp.re(result))
    return result


def fit_non_singlet_slopes(evolution_order="nlo",error_type="central",plot = True):
    """
    Fit non-singlet slopes to dipole form of lattice form factors
    Dipole parameters are hard-coded. Modify as needed.

    Parameters
    ----------
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"
    plot : bool, optional
        Show a plot of fit and data

    Note
    -------
    Prints the best-fit parameters and optionally shows plots.
    """
    m_F12 = 0.71
    mu_p = 2.7928
    mu_n = -1.913
    mp2 = 0.9382**2
    m_a_iv_2 = 1.322**2
    m_a_is_2 = 1.736**2
    gA_ud_iv = 1.2723
    gA_ud_is = 0.416
    def Ge_p(t):
        return 1/(1-t/m_F12)**2
    def Gm_p(t):
        return mu_p * Ge_p(t)
    def Ge_n(t):
        return 0
    def Gm_n(t):
        return mu_n * Ge_p(t)
    def F_1p(t):
        num = Ge_p(t) - t/(4*mp2)*Gm_p(t)
        den = 1 - t/(4*mp2)
        return num/den
    def F_1n(t):
        num = Ge_n(t) - t/(4*mp2)*Gm_n(t)
        den = 1 - t/(4*mp2)
        return num/den
    def F_2p(t):
        num = Gm_p(t) - Ge_p(t)
        den = 1 - t/(4*mp2)
        return num/den
    def F_2n(t):
        num = Gm_n(t) - Ge_n(t)
        den = 1 - t/(4*mp2)
        return num/den
    def G_a_isovector(t):
        return gA_ud_iv/(1-t/(m_a_iv_2))**2
    def G_a_isoscalar(t):
        return gA_ud_is/(1-t/(m_a_is_2))**2
    
    def non_singlet_isovector(t, norm, alpha_p):
        uv, err_u = reg.uv_pdf_regge(1,0,alpha_p,t,evolution_order,error_type=error_type)
        dv, err_d = reg.dv_pdf_regge(1,0,alpha_p,t,evolution_order,error_type=error_type)
        err = hp.error_sign(np.sqrt(err_u**2 + err_d**2),error_type)
        return norm * (uv - dv + err)
    def non_singlet_isovector_vec(t_arr, norm, alpha_p):
        return np.array([
           non_singlet_isovector(t,norm,alpha_p)
            for t in t_arr
        ], dtype=float)
    def non_singlet_isoscalar(t, norm, alpha_p):
        uv, err_u = reg.uv_pdf_regge(1,0,alpha_p,t,evolution_order,error_type=error_type)
        dv, err_d = reg.dv_pdf_regge(1,0,alpha_p,t,evolution_order,error_type=error_type)
        err = hp.error_sign(np.sqrt(err_u**2 + err_d**2),error_type)
        return norm * (uv + dv + err)
    def non_singlet_isoscalar_vec(t_arr, norm, alpha_p):
        return np.array([
           non_singlet_isoscalar(t,norm,alpha_p)
            for t in t_arr
        ], dtype=float)
    def polarized_non_singlet_isovector(t, norm, alpha_p):
        uv, err_u = reg.polarized_uv_pdf_regge(1,0,alpha_p,t,evolution_order,error_type=error_type)
        dv, err_d = reg.polarized_dv_pdf_regge(1,0,alpha_p,t,evolution_order,error_type=error_type)
        err = hp.error_sign(np.sqrt(err_u**2 + err_d**2),error_type)
        return norm * (uv - dv + err)
    def polarized_non_singlet_isovector_vec(t_arr, norm, alpha_p):
        return np.array([
           polarized_non_singlet_isovector(t,norm,alpha_p)
            for t in t_arr
        ], dtype=float)
    def polarized_non_singlet_isoscalar(t, norm, alpha_p):
        uv, err_u = reg.polarized_uv_pdf_regge(1,0,alpha_p,t,evolution_order,error_type=error_type)
        dv, err_d = reg.polarized_dv_pdf_regge(1,0,alpha_p,t,evolution_order,error_type=error_type)
        err = hp.error_sign(np.sqrt(err_u**2 + err_d**2),error_type)
        return norm * (uv + dv + err)
    def polarized_non_singlet_isoscalar_vec(t_arr, norm, alpha_p):
        return np.array([
           polarized_non_singlet_isoscalar(t,norm,alpha_p)
            for t in t_arr
        ], dtype=float)
    
    t_vals = np.linspace(0,-3,200)

    pseudo_data_isovector_A = F_1p(t_vals) - F_1n(t_vals)
    pseudo_data_isoscalar_A = 3 * (F_1p(t_vals) + F_1n(t_vals))
    pseudo_data_isovector_B = F_2p(t_vals) - F_2n(t_vals)
    pseudo_data_isoscalar_B = 3 * (F_2p(t_vals) + F_2n(t_vals))
    pseudo_data_isovector_Atilde = G_a_isovector(t_vals)
    pseudo_data_isoscalar_Atilde = G_a_isoscalar(t_vals)
    # Perform fits
    # A fits (non-polarized)[
    popt_A_iv, pcov_A_iv = curve_fit(non_singlet_isovector_vec, t_vals, pseudo_data_isovector_A, p0=[1,0.6], bounds=([1,.1],[1.0001,3]))
    popt_A_is, pcov_A_is = curve_fit(non_singlet_isoscalar_vec, t_vals, pseudo_data_isoscalar_A, p0=[1,1],bounds=([.1,.1],[5,3]))

    # B fits (non-polarized)
    popt_B_iv, pcov_B_iv = curve_fit(non_singlet_isovector_vec, t_vals, pseudo_data_isovector_B, p0=[4,1.5],bounds=([2,1],[6,3]))
    popt_B_is, pcov_B_is = curve_fit(non_singlet_isoscalar_vec, t_vals, pseudo_data_isoscalar_B, p0=[-0.1,1.1],bounds=([-2,1],[-.1,3]))

    # Atilde fits (polarized)
    popt_Atilde_iv, pcov_Atilde_iv = curve_fit(polarized_non_singlet_isovector_vec, t_vals, pseudo_data_isovector_Atilde, p0=[0.8,1],bounds=([.5,.1],[2,3]))
    popt_Atilde_is, pcov_Atilde_is = curve_fit(polarized_non_singlet_isoscalar_vec, t_vals, pseudo_data_isoscalar_Atilde, p0=[1.7,0.3],bounds=([.2,.1],[3,1.5]))

    # # Print or return the results
    print("Fitted parameters:")
    print(f"A isovector: norm = {popt_A_iv[0]:.4f}, alpha_p = {popt_A_iv[1]:.4f}")
    print(f"A isoscalar: norm = {popt_A_is[0]:.4f}, alpha_p = {popt_A_is[1]:.4f}")
    print(f"B isovector: norm = {popt_B_iv[0]:.4f}, alpha_p = {popt_B_iv[1]:.4f}")
    print(f"B isoscalar: norm = {popt_B_is[0]:.4f}, alpha_p = {popt_B_is[1]:.4f}")
    print(f"Atilde isovector: norm = {popt_Atilde_iv[0]:.4f}, alpha_p = {popt_Atilde_iv[1]:.4f}")
    print(f"Atilde isoscalar: norm = {popt_Atilde_is[0]:.4f}, alpha_p = {popt_Atilde_is[1]:.4f}")

    results = {
        ("non_singlet_isovector", "A"): tuple(popt_A_iv),
        ("non_singlet_isoscalar", "A"): tuple(popt_A_is),
        ("non_singlet_isovector", "B"): tuple(popt_B_iv),
        ("non_singlet_isoscalar", "B"): tuple(popt_B_is),
        ("non_singlet_isovector", "Atilde"): tuple(popt_Atilde_iv),
        ("non_singlet_isoscalar", "Atilde"): tuple(popt_Atilde_is),
    }

    if plot:
        plt.figure()
        plt.plot(-t_vals, pseudo_data_isovector_A, 'o', label='Pseudo-data A (iv)')
        plt.plot(-t_vals, non_singlet_isovector_vec(t_vals, *popt_A_iv), '-', label='Fit A (iv)')
        plt.xlabel("t [GeV²]")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    if plot:
        plt.figure()
        plt.plot(-t_vals, pseudo_data_isoscalar_A, 'o', label='Pseudo-data A (is)')
        plt.plot(-t_vals, non_singlet_isoscalar_vec(t_vals, *popt_A_is), '-', label='Fit A (is)')
        plt.xlabel("t [GeV²]")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
    
    if plot:
        plt.figure()
        plt.plot(-t_vals, pseudo_data_isovector_B, 'o', label='Pseudo-data B (iv)')
        plt.plot(-t_vals, non_singlet_isovector_vec(t_vals, *popt_B_iv), '-', label='Fit B (iv)')
        plt.xlabel("t [GeV²]")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    if plot:
        plt.figure()
        plt.plot(-t_vals, pseudo_data_isoscalar_B, 'o', label='Pseudo-data B (is)')
        plt.plot(-t_vals, non_singlet_isoscalar_vec(t_vals, *popt_B_is), '-', label='Fit B (is)')
        plt.xlabel("t [GeV²]")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    if plot:
        plt.figure()
        plt.plot(-t_vals, pseudo_data_isovector_Atilde, 'o', label='Pseudo-data Atilde (iv)')
        plt.plot(-t_vals, polarized_non_singlet_isovector_vec(t_vals, *popt_Atilde_iv), '-', label='Fit Atilde (iv)')
        plt.xlabel("t [GeV²]")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    if plot:
        plt.figure()
        plt.plot(-t_vals, pseudo_data_isoscalar_Atilde, 'o', label='Pseudo-data Atilde (is)')
        plt.plot(-t_vals, polarized_non_singlet_isoscalar_vec(t_vals, *popt_Atilde_is), '-', label='Fit Atilde (is)')
        plt.xlabel("t [GeV²]")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    return results

def fit_singlet_slopes_A(evolution_order="nlo",plot=True,fix_norm_A=True):
    """
    Fit singlet A slopes to dipole form of lattice form factors.
    Dipole parameters are hard-coded. Modify as needed.

    Parameters
    ----------
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"
    plot : bool, optional
        Show a plot of fit and data
    fix_norm_A : bool, optional
        Keep the quark and gluon A normalizations at 1, so that only the t-dependence is
        fitted and the forward limit stays the prediction of the input PDFs:
        A_q(0) = <x>_q and A_g(0) = <x>_g, hence A_q(0) + A_g(0) = 1 by the momentum sum
        rule. Setting it to False lets the fit trade the normalization against the shape,
        which reproduces the lattice A_q(0) better but breaks the sum rule. Default is True.

    Note
    -------
    Prints the best-fit parameters and optionally shows plots. The D normalizations are
    always fitted; they are not constrained by a sum rule.
    """
    # Dipole form from table III in 2310.08484
    g_A = 0.501
    m_A_g2 = 1.262**2
    g_D = -2.572
    m_D_g2 = 0.538**2

    q_A = 0.510
    m_A_q2 = 1.477**2
    q_D = -1.30
    m_D_q2 = 0.81**2

    def gluon_A(t):
        return g_A/(1-t/m_A_g2)**2
    def quark_A(t):
        return q_A/(1-t/m_A_q2)**2
    def gluon_D(t):
        return g_D/(1-t/m_D_g2)**2
    def quark_D(t):
        return q_D/(1-t/m_D_q2)**2

    def gluon_singlet_A(t,norm,alpha_p):
        res, _ = core.gluon_singlet_regge_A(2,0,t,alpha_p,moment_label="A",evolution_order=evolution_order)
        return norm*res
    def gluon_singlet_A_vec(t_arr,norm,alpha_p):
        return np.array([
            float(gluon_singlet_A(t,norm,alpha_p).real)
            for t in np.atleast_1d(t_arr)
        ], dtype=float)
    def quark_singlet_A(t,norm,alpha_p):
        res, _ = core.quark_singlet_regge_A(2,0,t,alpha_p,moment_label="A",evolution_order=evolution_order)
        return norm*res
    def quark_singlet_A_vec(t_arr,norm,alpha_p):
        return np.array([
            float(quark_singlet_A(t,norm,alpha_p).real)
            for t in np.atleast_1d(t_arr)
        ], dtype=float)

    # Generate pseudo data
    t_vals = np.linspace(-1e-6,-2,100)

    pseudo_data_gluon_A = gluon_A(t_vals)
    pseudo_data_quark_A = quark_A(t_vals)
    pseudo_data_gluon_D = gluon_D(t_vals)
    pseudo_data_quark_D = quark_D(t_vals)

    # norm fixed to 1 -> the forward limit is the momentum fraction of the input PDF
    norm_bounds = ([1,1.0001] if fix_norm_A else [.2,1.5])
    norm_0 = 1. if fix_norm_A else .5
    popt_A_g, pcov_A_g = curve_fit(gluon_singlet_A_vec, t_vals, pseudo_data_gluon_A, p0=[norm_0,0.6],
                                   bounds=([norm_bounds[0],.05],[norm_bounds[1],1.5]))
    print(f"gluon A: norm = {popt_A_g[0]:.4f}, alpha_p = {popt_A_g[1]:.4f}")
    popt_A_q, pcov_A_q = curve_fit(quark_singlet_A_vec, t_vals, pseudo_data_quark_A, p0=[norm_0,0.8],
                                   bounds=([norm_bounds[0],.05],[norm_bounds[1],1.5]))
    print(f"quark A: norm = {popt_A_q[0]:.4f}, alpha_p = {popt_A_q[1]:.4f}")

    alpha_p_T = popt_A_g[1]
    alpha_p_ud = popt_A_q[1]
    
    # j = 2 independent of eta
    def gluon_singlet_D(t,norm,alpha_p_S):
        res, _ = core.gluon_singlet_regge_D(2, 1, t, alpha_p_T, alpha_p_S, moment_label="A", evolution_order=evolution_order)
        return norm*res
    def gluon_singlet_D_vec(t_arr, norm, alpha_p_S):
        return np.array([
            gluon_singlet_D(t,norm,alpha_p_S)
            for t in t_arr
        ], dtype=float)
    
    def quark_singlet_D(t,norm,alpha_p_s):
        res, _ = core.quark_singlet_regge_D(2,1,t,alpha_p_ud,alpha_p_s,moment_label="A",evolution_order=evolution_order)
        return norm*res
    def quark_singlet_D_vec(t_arr, norm, alpha_p_S):
        return np.array([
            quark_singlet_D(t,norm,alpha_p_S)
            for t in t_arr
        ], dtype=float)

    popt_D_g, pcov_D_g = curve_fit(gluon_singlet_D_vec, t_vals, pseudo_data_gluon_D, p0=[1,4.2], bounds=([.1,3],[2,6]))
    print(f"gluon D: norm = {popt_D_g[0]:.4f}, alpha_p = {popt_D_g[1]:.4f}")
    popt_D_q, pcov_D_q = curve_fit(quark_singlet_D_vec, t_vals, pseudo_data_quark_D, p0=[2,1], bounds=([.2,1],[3,3]))
    print(f"quark D: norm = {popt_D_q[0]:.4f}, alpha_p = {popt_D_q[1]:.4f}")

    # Use less data points for plot
    t_vals = np.linspace(-1e-6,-2,50)
    pseudo_data_gluon_A = gluon_A(t_vals)
    pseudo_data_quark_A = quark_A(t_vals)
    pseudo_data_gluon_D = gluon_D(t_vals)
    pseudo_data_quark_D = quark_D(t_vals)

    if plot:
        plt.figure()
        plt.plot(-t_vals, pseudo_data_gluon_A, 'o', label='Pseudo-data A_g')
        plt.plot(-t_vals, gluon_singlet_A(t_vals, *popt_A_g), '-', label='A_g')
        plt.xlabel("t [GeV²]")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    if plot:
        plt.figure()
        plt.plot(-t_vals, pseudo_data_quark_A, 'o', label='Pseudo-data A_q')
        plt.plot(-t_vals, quark_singlet_A(t_vals, *popt_A_q), '-', label='A_q')
        plt.xlabel("t [GeV²]")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    if plot:
        plt.figure()
        plt.plot(-t_vals, pseudo_data_gluon_D, 'o', label='Pseudo-data D_g')
        plt.plot(-t_vals, gluon_singlet_D_vec(t_vals, *popt_D_g), '-', label='D_g')
        plt.xlabel("t [GeV²]")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    if plot:
        plt.figure()
        plt.plot(-t_vals, pseudo_data_quark_D, 'o', label='Pseudo-data A_g')
        plt.plot(-t_vals, quark_singlet_D_vec(t_vals, *popt_D_q), '-', label='D_q')
        plt.xlabel("t [GeV²]")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    return {"quark_A": tuple(popt_A_q), "gluon_A": tuple(popt_A_g),
            "quark_D": tuple(popt_D_q), "gluon_D": tuple(popt_D_g)}

def fit_singlet_slopes_Atilde(evolution_order="nlo",plot=True):
    """
    Fit singlet Atilde slopes to dipole form of lattice form factors.
    Dipole parameters are hard-coded. Modify as needed.

    Parameters
    ----------
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"
    plot : bool, optional
        Show a plot of fit and data

    Note
    -------
    Prints the best-fit parameters and optionally shows plots.
    """
    # Dipole form from table III in 1703.06703
    # GP u + d + s
    # q_A = 19.1505
    # m_A_q2 = 0.458175**2
    # p0=[161,7]
    # bounds=([100,3],[300,10])
    # GA u + d + s
    q_A = 0.495328
    m_A_q2 = 1.51972**2
    p0=[5,.5]
    bounds=([3,.2],[10,3])
    def quark_A(t):
        return q_A/(1-t/m_A_q2)**2

    def compute_quark(t,norm, alpha_p):
        res, _ = core.quark_singlet_regge_A(2,0,t,alpha_p,moment_label="Atilde",evolution_order=evolution_order)
        return norm*float(res.real)
    def quark_singlet_A(t_vals,norm,alpha_p):
        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(compute_quark)(t, norm, alpha_p)
            for t in t_vals
        )
        return results
    
    # Generate pseudo data
    t_vals = np.linspace(-1e-6,-10,10)
    pseudo_data_quark_A = quark_A(t_vals)
    popt_A_q, pcov_A_q = curve_fit(quark_singlet_A, t_vals, pseudo_data_quark_A, p0=p0, bounds=bounds)
    print(f"quark A: norm = {popt_A_q[0]:.4f}, alpha_p = {popt_A_q[1]:.4f}")

    # Use more data points for plot
    t_vals = np.linspace(-1e-6,-10,50)
    pseudo_data_quark_A = quark_A(t_vals)

    if plot:
        plt.figure()
        plt.plot(-t_vals, pseudo_data_quark_A, 'o', label='Pseudo-data A_q')
        plt.plot(-t_vals, quark_singlet_A(t_vals, *popt_A_q), '-', label='A_q')
        plt.xlabel("t [GeV²]")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    return tuple(popt_A_q)

##############################
### Fit to evolved moments ###
### Work in progress...    ###
##############################

def fit_singlet_slopes_2(evolution_order="nlo",plot=True):
    # Dipole form from table III in 2310.08484
    g_A = 0.501
    m_A_g2 = 1.262**2
    g_D = -2.572
    m_D_g2 = 0.538**2

    q_A = 0.510
    m_A_q2 = 1.477**2
    q_D = -1.30
    m_D_q2 = 0.81**2

    if evolution_order == "lo":
        alpha_prime_ud = 0.9426
    elif evolution_order == "nlo":
        alpha_prime_ud = 0.9492

    def gluon_A(t):
        return g_A/(1-t/m_A_g2)**2
    def quark_A(t):
        return q_A/(1-t/m_A_q2)**2
    def gluon_D(t):
        return g_D/(1-t/m_D_g2)**2
    def quark_D(t):
        return q_D/(1-t/m_D_q2)**2
    
    t_vals = np.linspace(-1e-6,-2,25)

    def gluon_singlet_A(t, norm_Aq, alpha_prime_T, norm_Ag):
        return evolve_singlet_fit(0,t,2,alpha_prime_ud=alpha_prime_ud,particle="gluon",
                                alpha_prime_s=0,norm_Aq=norm_Aq,alpha_prime_T=alpha_prime_T,alpha_prime_S=0,norm_Ag=norm_Ag,norm_Dg=0,evolution_order=evolution_order)
    def quark_singlet_A(t, norm_Aq, alpha_prime_T, norm_Ag):
        return evolve_singlet_fit(0,t,2,alpha_prime_ud=alpha_prime_ud,particle="quark",
                                   alpha_prime_s=0,norm_Aq=norm_Aq,alpha_prime_T=alpha_prime_T,alpha_prime_S=0,norm_Ag=norm_Ag,norm_Dg=0,evolution_order=evolution_order)
    def compute_moments(t, norm_Aq, alpha_prime_T, norm_Ag):
        quark = quark_singlet_A(t, norm_Aq, alpha_prime_T, norm_Ag)
        gluon = gluon_singlet_A(t, norm_Aq, alpha_prime_T, norm_Ag)
        return gluon, quark

    def singlet_A(t_vals, norm_Aq, alpha_prime_T, norm_Ag):
        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(compute_moments)(t, norm_Aq, alpha_prime_T, norm_Ag)
            for t in t_vals
        )
        gluon, quark = zip(*results)
        return np.concatenate([gluon, quark])

    # Generate pseudo data
    pseudo_data_gluon_A = gluon_A(t_vals)
    pseudo_data_quark_A = quark_A(t_vals)
    pseudo_data_singlet_A = np.concatenate([pseudo_data_gluon_A, pseudo_data_quark_A])
    pseudo_data_gluon_D = gluon_D(t_vals)
    pseudo_data_quark_D = quark_D(t_vals)
    pseudo_data_singlet_D = np.concatenate([pseudo_data_gluon_D, pseudo_data_quark_D])

    popt, pcov = curve_fit(
        singlet_A,
        t_vals,
        pseudo_data_singlet_A,
        p0=[0.9,0.6, 1.3],
        bounds=([0.5, 0.3, 0.7], [1.5, 1, 1.7])
    )
    # Extract fitted params
    norm_Aq, alpha_prime_T, norm_Ag = popt
    print(f"Singlet A:  alpha'_T = {alpha_prime_T:.4f}, norm_Ag = {norm_Ag:.4f}, norm_Aq = {norm_Aq:.4f}")
   

    def quark_singlet_D(t_vals,alpha_prime_s, norm_Dq, alpha_prime_S, norm_Dg):
        res = np.array([evolve_singlet_D_fit(1,t,2,alpha_prime_ud, alpha_prime_s,norm_Aq, norm_Dq,alpha_prime_T, alpha_prime_S,norm_Ag, norm_Dg,
                       particle="quark",moment_label="A", evolution_order = evolution_order, error_type = "central",interpolation=True)
                       for t in t_vals
        ], dtype=np.float64)
        return res
    
    def gluon_singlet_D(t_vals,alpha_prime_s, norm_Dq, alpha_prime_S, norm_Dg):
        res = np.array([evolve_singlet_D_fit(1,t,2,alpha_prime_ud, alpha_prime_s,norm_Aq, norm_Dq,alpha_prime_T, alpha_prime_S,norm_Ag, norm_Dg,
                       particle="gluon",moment_label="A", evolution_order = evolution_order, error_type = "central",interpolation=True)
                       for t in t_vals
        ], dtype=np.float64)
        return res
    def singlet_D(t_vals, alpha_prime_s, norm_Dq, alpha_prime_S, norm_Dg):
        gluon = gluon_singlet_D(t_vals, alpha_prime_s, norm_Dq, alpha_prime_S, norm_Dg)
        quark = quark_singlet_D(t_vals, alpha_prime_s, norm_Dq, alpha_prime_S, norm_Dg)
        return np.concatenate([gluon, quark])

    popt_D, pcov_D = curve_fit(
        singlet_D,
        t_vals,
        pseudo_data_singlet_D,
        p0=[1.9,1, 5, 1],
        bounds=([1, 0.2, 2,0.2], [3, 3, 6.0,3])
    )
    # Extract fitted params
    alpha_prime_s, norm_Dq, alpha_prime_S, norm_Dg = popt_D
    print(f"Singlet D:  alpha'_s = {alpha_prime_s:.4f}, norm_Dq = {norm_Dq:.4f}, alpha'_S = {alpha_prime_S:.4f},  norm_Dg = {norm_Dg:.4f}")

    # Use less data points for plot
    t_vals = np.linspace(-1e-6,-2,10)
    pseudo_data_gluon_A = gluon_A(t_vals)
    pseudo_data_quark_A = quark_A(t_vals)
    pseudo_data_gluon_D = gluon_D(t_vals)
    pseudo_data_quark_D = quark_D(t_vals)

    if plot:
        fitted_gluon_A = gluon_singlet_A(t_vals, *popt)
        fitted_quark_A = quark_singlet_A(t_vals, *popt)
        plt.figure()
        plt.plot(-t_vals, pseudo_data_gluon_A, 'o', label='Pseudo-data $A_g$')
        plt.plot(-t_vals, fitted_gluon_A, '-', label='Fit $A_g$')
        plt.plot(-t_vals, pseudo_data_quark_A, 'o', label='Pseudo-data $A_q$')
        plt.plot(-t_vals, fitted_quark_A, '-', label='Fit $A_q$')
        plt.xlabel(r"$-t$ [GeV$^2$]")
        plt.ylabel(r"$A_{q,g}(t)$")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    if plot:
        fitted_gluon_D = gluon_singlet_D(t_vals, *popt_D)
        fitted_quark_D = quark_singlet_D(t_vals, *popt_D)
        plt.figure()
        plt.plot(-t_vals, pseudo_data_gluon_D, 'o', label='Pseudo-data $D_g$')
        plt.plot(-t_vals, fitted_gluon_D, '-', label='Fit $D_g$')
        plt.plot(-t_vals, pseudo_data_quark_D, 'o', label='Pseudo-data $D_q$')
        plt.plot(-t_vals, fitted_quark_D, '-', label='Fit $D_q$')
        plt.xlabel(r"$-t$ [GeV$^2$]")
        plt.ylabel(r"$D_{q,g}(t)$")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return

def fit_singlet_D_slopes(norm_Aq,alpha_prime_T,norm_Ag,evolution_order="nlo",plot=True,alpha_prime_ud=None):
    # Dipole form from table III in 2310.08484
    g_D = -2.572
    m_D_g2 = 0.538**2

    q_D = -1.30
    m_D_q2 = 0.81**2

    # Use the quark singlet A slope for self-consistency with the runtime
    # config (REGGE_SLOPES["singlet"][...][0]); legacy fallback values below.
    if alpha_prime_ud is None:
        if evolution_order == "lo":
            alpha_prime_ud = 0.9426
        elif evolution_order == "nlo":
            alpha_prime_ud = 0.9492

    def gluon_D(t):
        return g_D/(1-t/m_D_g2)**2
    def quark_D(t):
        return q_D/(1-t/m_D_q2)**2

    # Generate pseudo data
    t_vals = np.linspace(-1e-6,-2,25)

    def quark_singlet_D(t_vals,alpha_prime_s, norm_Dq, alpha_prime_S, norm_Dg):
        res = np.array([evolve_singlet_D_fit(1,t,2,alpha_prime_ud, alpha_prime_s,norm_Aq, norm_Dq,alpha_prime_T, alpha_prime_S,norm_Ag, norm_Dg,
                       particle="quark",moment_label="A", evolution_order = evolution_order, error_type = "central",interpolation=True)
                       for t in t_vals
        ], dtype=np.float64)
        # def poly(alpha_p, t):
        #     return -1/(1-t/alpha_p)**2
        # res = np.array([norm_Dq * poly(alpha_prime_s,t) + norm_Dg * poly(alpha_prime_S,t)
        #                for t in t_vals
        # ], dtype=np.float64)
        return res
    
    def gluon_singlet_D(t_vals,alpha_prime_s, norm_Dq, alpha_prime_S, norm_Dg):
        res = np.array([evolve_singlet_D_fit(1,t,2,alpha_prime_ud, alpha_prime_s,norm_Aq, norm_Dq,alpha_prime_T, alpha_prime_S,norm_Ag, norm_Dg,
                       particle="gluon",moment_label="A", evolution_order = evolution_order, error_type = "central",interpolation=True)
                       for t in t_vals
        ], dtype=np.float64)
        # def poly(alpha_p, t):
        #     return - 1/(1-t/alpha_p)**2
        # res = np.array([norm_Dq * poly(alpha_prime_s,t) + norm_Dg * poly(alpha_prime_S,t)
        #                for t in t_vals
        # ], dtype=np.float64)
        return res
    def singlet_D(t_vals, alpha_prime_s, norm_Dq, alpha_prime_S, norm_Dg):
        gluon = gluon_singlet_D(t_vals, alpha_prime_s, norm_Dq, alpha_prime_S, norm_Dg)
        quark = quark_singlet_D(t_vals, alpha_prime_s, norm_Dq, alpha_prime_S, norm_Dg)
        return np.concatenate([gluon, quark])

    pseudo_data_gluon_D = gluon_D(t_vals)
    pseudo_data_quark_D = quark_D(t_vals)
    pseudo_data_singlet_D = np.concatenate([pseudo_data_gluon_D, pseudo_data_quark_D])

    popt, pcov = curve_fit(
        singlet_D,
        t_vals,
        pseudo_data_singlet_D,
        p0=[1.9,1, 5, 1],
        bounds=([1, 0.2, 2,0.2], [3, 3, 6.0,3])
    )
    # Extract fitted params
    alpha_prime_s, norm_Dq, alpha_prime_S, norm_Dg = popt
    print(f"Singlet D:  alpha'_s = {alpha_prime_s:.4f}, norm_Dq = {norm_Dq:.4f}, alpha'_S = {alpha_prime_S:.4f},  norm_Dg = {norm_Dg:.4f}")

    # Use less data points for plot
    t_vals = np.linspace(-1e-6,-2,10)
    pseudo_data_gluon_D = gluon_D(t_vals)
    pseudo_data_quark_D = quark_D(t_vals)


    if plot:
        fitted_gluon_D = gluon_singlet_D(t_vals, *popt)
        fitted_quark_D = quark_singlet_D(t_vals, *popt)
        plt.figure()
        plt.plot(-t_vals, pseudo_data_gluon_D, 'o', label='Pseudo-data $D_g$')
        plt.plot(-t_vals, fitted_gluon_D, '-', label='Fit $D_g$')
        plt.plot(-t_vals, pseudo_data_quark_D, 'o', label='Pseudo-data $D_q$')
        plt.plot(-t_vals, fitted_quark_D, '-', label='Fit $D_q$')
        plt.xlabel(r"$-t$ [GeV$^2$]")
        plt.ylabel(r"$D_{q,g}(t)$")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return tuple(popt)


####################################################
### Fits of the input PDF parameters to PDF data ###
### (writes the pdfs/<pdf_set>[_POL].csv files)  ###
####################################################
#
# Data format (one row per point), as delivered by the GUMP program:
#
#   x, t, mu, f, delta f, GPD type, flavor
#
# with GPD type 0 = unpolarized, 2 = polarized, and flavor in u, d, s, g.
# Quarks are tabulated for both signs of x with
#   unpolarized: f(x) = q(x),        f(-x) = -qbar(x)
#   polarized:   f(x) = Delta q(x),  f(-x) = +Delta qbar(x)
# (charge-parity of the corresponding GPD), and the gluon entry is x*f_g(x).
# Only t = 0 and mu = cfg.MU_INPUT rows are used.
#
# The fitted functional forms are exactly those of unpolarized_pdf.py / polarized_pdf.py, on which the
# analytic Regge/Mellin integrals in regge.py rely:
#   f(x)       = A x^(eta_1-1) (1-x)^eta_2 (1 + epsilon sqrt(x) + gamma x)      (uv, dv, S, g, s+sbar)
#   Delta(x)   = A_D x^(eta_D-1) (1-x)^(eta_S+2) (1 + gamma_D x + delta_D x^2), Delta = dbar - ubar
#   (s-sbar)   = A_- x^(delta_- - 1) (1-x)^eta_- (1 - x/x_0)
#   Delta f(x) = F(x) f(x),  F(x) = dA x^alpha (1 + dgamma (x^lambda - 1))      (nlo/nnlo)
#                                 = dA x^alpha (1 + dgamma x^lambda)            (lo)


def _beta(a, b):
    """Euler beta function (positive arguments)."""
    return np.exp(betaln(a, b))


def _pdf_form(x, A, eta_1, eta_2, epsilon, gamma_pdf):
    """Input PDF, identical to unpolarized_pdf.pdf."""
    return A * x ** (eta_1 - 1) * (1 - x) ** eta_2 * (1 + epsilon * np.sqrt(x) + gamma_pdf * x)


def _pdf_mellin(n, A, eta_1, eta_2, epsilon, gamma_pdf):
    """int_0^1 dx x^n f(x): n = 0 number sum rule, n = 1 momentum sum rule."""
    a = eta_1 + n
    return A * (_beta(a, eta_2 + 1) + epsilon * _beta(a + 0.5, eta_2 + 1)
                + gamma_pdf * _beta(a + 1, eta_2 + 1))


def _delta_form(x, A_D, eta_D, eta_S, gamma_D, delta_D):
    """Delta = dbar - ubar, identical to unpolarized_pdf.Delta_pdf."""
    return A_D * x ** (eta_D - 1) * (1 - x) ** (eta_S + 2) * (1 + gamma_D * x + delta_D * x ** 2)


def _delta_mellin(n, A_D, eta_D, eta_S, gamma_D, delta_D):
    a, b = eta_D + n, eta_S + 3
    return A_D * (_beta(a, b) + gamma_D * _beta(a + 1, b) + delta_D * _beta(a + 2, b))


def _sv_x0(delta_m, eta_m):
    """x_0 fixed by the strangeness number sum rule int_0^1 dx (s - sbar) = 0."""
    return _beta(delta_m + 1, eta_m + 1) / _beta(delta_m, eta_m + 1)


def _sv_form(x, A_m, delta_m, eta_m, x_0):
    """s - sbar, identical to unpolarized_pdf.sv_pdf."""
    return A_m * x ** (delta_m - 1) * (1 - x) ** eta_m * (1 - x / x_0)


def _polarized_factor(x, delta_A, alpha, delta_gamma, delta_lambda, evolution_order):
    """Polarized factor F(x) = Delta f(x)/f(x), identical to polarized_pdf.polarized_pdf."""
    if evolution_order != "lo":
        return delta_A * x ** alpha * (1 + delta_gamma * (x ** delta_lambda - 1))
    return delta_A * x ** alpha * (1 + delta_gamma * x ** delta_lambda)


def _lsq(residual, p0, bounds, name, verbose=True):
    """Least-squares fit returning (params, sigmas, covariance, chi2/dof)."""
    res = least_squares(residual, p0, bounds=bounds, x_scale="jac", max_nfev=20000)
    if not res.success:
        raise RuntimeError(f"{name}: fit failed ({res.message})")
    dof = len(res.fun) - len(res.x)
    chi2red = 2 * res.cost / dof
    # Covariance from the Gauss-Newton approximation, inflated when the functional form does
    # not describe the data (chi2/dof > 1) so that the quoted errors stay conservative.
    cov = np.linalg.inv(res.jac.T @ res.jac) * max(1.0, chi2red)
    sigma = np.sqrt(np.diag(cov))
    if verbose:
        print(f"{name:10s} chi2/dof = {chi2red:8.3f}   params = {np.array2string(res.x, precision=4)}")
    return res.x, sigma, cov, chi2red


def _propagated_sigma(f, p, cov):
    """1-sigma error of a scalar function f(p) of the fitted parameters."""
    g = np.zeros(len(p))
    for i in range(len(p)):
        h = 1e-6 * max(1.0, abs(p[i]))
        pp, pm = np.array(p, float), np.array(p, float)
        pp[i] += h
        pm[i] -= h
        g[i] = (f(pp) - f(pm)) / (2 * h)
    return float(np.sqrt(g @ cov @ g))


def _param_row(name, lo, nlo, nnlo=None):
    """CSV row 'name, lo, +error, -error, nlo, +error, -error, nnlo, +error, -error'."""
    nnlo = nlo if nnlo is None else nnlo
    out = [f'"{name}"']
    for central, sigma in (lo, nlo, nnlo):
        out += [f"{central:.6g}", f"{abs(sigma):.4g}", f"{-abs(sigma):.4g}"]
    return ",".join(out) + "\n"


def read_pdf_parameter_csv(path):
    """
    Read an input PDF parameter file (pdfs/<pdf_set>.csv or <pdf_set>_POL.csv) into a dictionary.

    Parameters
    ----------
    path : str or Path
        Path to the parameter CSV.

    Returns
    -------
    dict
        {parameter: {"lo": array([central, +error, -error]), "nlo": ..., "nnlo": ...}}
    """
    out = {}
    with open(path, "r", newline="") as file:
        next(file)
        for row in csv.reader(file):
            values = np.array(row[1:10], dtype=float)
            out[row[0]] = {"lo": values[0:3], "nlo": values[3:6], "nnlo": values[6:9]}
    return out


def load_pdf_data(data_path):
    """
    Parse a GUMP-format PDF data file into the flavor combinations of the input parametrization.

    Parameters
    ----------
    data_path : str or Path
        CSV with columns x, t, mu, f, delta f, GPD type, flavor (see the notes above).

    Returns
    -------
    dict
        Arrays on a common x-grid (x > 0). Unpolarized: uv, dv, ubar, dbar, s, sbar, s_plus,
        sv, S, Delta, g. Polarized (prefix D): Duv, Ddv, Dubar, Ddbar, Ds, Dsbar, Ds_plus,
        DS, Dg. Every entry k carries its error in k + "_err". "has_strange" flags whether
        the data set resolves the strange quark.

    Note
    ----
    Without strange entries the sea has to be completed by the kappa-prescription of the
    caller (s = sbar = kappa/2 (ubar + dbar)); see fit_input_pdfs.
    """
    rows = np.genfromtxt(data_path, delimiter=",", names=True, dtype=None, encoding="utf-8",
                         deletechars="", replace_space="_")
    names = rows.dtype.names
    x_col, f_col, error_col = names[0], names[3], names[4]
    type_col, flavor_col = names[5], names[6]

    def branch(gpd_type, flavor, sign):
        mask = ((rows[type_col] == gpd_type) & (rows[flavor_col] == flavor)
                & (np.sign(rows[x_col]) == sign))
        sub = rows[mask]
        order = np.argsort(np.abs(sub[x_col]))
        return np.abs(sub[x_col])[order], sub[f_col][order], sub[error_col][order]

    x, u, u_error = branch(0, "u", +1)
    _, u_m, u_m_error = branch(0, "u", -1)
    _, d, d_error = branch(0, "d", +1)
    _, d_m, d_m_error = branch(0, "d", -1)
    x_g, f_g, f_g_error = branch(0, "g", +1)
    if not (np.allclose(x, x_g) and len(x) == len(u_m)):
        raise ValueError("Inconsistent x-grids between flavors in the PDF data")

    has_strange = bool(np.any(rows[flavor_col] == "s"))
    # f(-x) = -qbar(x) for the unpolarized distributions
    ubar, dbar = -u_m, -d_m
    data = {
        "x": x, "has_strange": has_strange,
        "uv": u + u_m, "uv_err": np.hypot(u_error, u_m_error),
        "dv": d + d_m, "dv_err": np.hypot(d_error, d_m_error),
        "ubar": ubar, "ubar_err": u_m_error,
        "dbar": dbar, "dbar_err": d_m_error,
        "Delta": dbar - ubar, "Delta_err": np.hypot(u_m_error, d_m_error),
        "g": f_g / x, "g_err": f_g_error / x,
    }
    if has_strange:
        _, s, s_error = branch(0, "s", +1)
        _, s_m, s_m_error = branch(0, "s", -1)
        sbar = -s_m
        data.update({
            "s": s, "s_err": s_error,
            "sbar": sbar, "sbar_err": s_m_error,
            "s_plus": s + sbar, "s_plus_err": np.hypot(s_error, s_m_error),
            "sv": s - sbar, "sv_err": np.hypot(s_error, s_m_error),
            # S = 2(ubar + dbar) + s + sbar
            "S": 2 * (ubar + dbar) + s + sbar,
            "S_err": np.sqrt(4 * u_m_error ** 2 + 4 * d_m_error ** 2
                             + s_error ** 2 + s_m_error ** 2),
        })

    # f(-x) = +Delta qbar(x) for the polarized distributions
    _, pol_u, pol_u_error = branch(2, "u", +1)
    _, pol_u_m, pol_u_m_error = branch(2, "u", -1)
    _, pol_d, pol_d_error = branch(2, "d", +1)
    _, pol_d_m, pol_d_m_error = branch(2, "d", -1)
    _, pol_g, pol_g_error = branch(2, "g", +1)
    data.update({
        "Duv": pol_u - pol_u_m, "Duv_err": np.hypot(pol_u_error, pol_u_m_error),
        "Ddv": pol_d - pol_d_m, "Ddv_err": np.hypot(pol_d_error, pol_d_m_error),
        "Dubar": pol_u_m, "Dubar_err": pol_u_m_error,
        "Ddbar": pol_d_m, "Ddbar_err": pol_d_m_error,
        "Dg": pol_g / x, "Dg_err": pol_g_error / x,
    })
    if has_strange:
        _, pol_s, pol_s_error = branch(2, "s", +1)
        _, pol_s_m, pol_s_m_error = branch(2, "s", -1)
        data.update({
            "Ds": pol_s, "Ds_err": pol_s_error,
            "Dsbar": pol_s_m, "Dsbar_err": pol_s_m_error,
            "Ds_plus": pol_s + pol_s_m, "Ds_plus_err": np.hypot(pol_s_error, pol_s_m_error),
            "DS": 2 * (pol_u_m + pol_d_m) + pol_s + pol_s_m,
            "DS_err": np.sqrt(4 * pol_u_m_error ** 2 + 4 * pol_d_m_error ** 2
                              + pol_s_error ** 2 + pol_s_m_error ** 2),
        })
    return data


def input_carriers(parameters, x, evolution_order):
    """
    Reconstruct the unpolarized input PDF combinations from a parameter dictionary.

    Parameters
    ----------
    parameters : dict
        Output of read_pdf_parameter_csv (an unpolarized <pdf_set>.csv).
    x : array_like
        Parton x values.
    evolution_order : str
        "lo", "nlo", "nnlo".

    Returns
    -------
    dict
        uv, dv, S, Delta, g, s_plus, sv and the per-flavor sea combinations
        ubar, dbar, s, sbar. These are the carriers the polarized factors multiply
        (Delta f = F(x) f(x)), matching polarized_pdf.polarized_*_pdf.
    """
    def value(key):
        return parameters[key][evolution_order][0]

    uv = _pdf_form(x, value("A_u"), value("eta_1"), value("eta_2"),
                   value("epsilon_u"), value("gamma_u"))
    dv = _pdf_form(x, value("A_d"), value("eta_3"), value("eta_2") + value("eta_4-eta_2"),
                   value("epsilon_d"), value("gamma_d"))
    S = _pdf_form(x, value("A_S"), value("delta_S"), value("eta_S"),
                  value("epsilon_S"), value("gamma_S"))
    Delta = _delta_form(x, value("A_Delta"), value("eta_Delta"), value("eta_S"),
                        value("gamma_Delta"), value("delta_Delta"))
    s_plus = _pdf_form(x, value("A_+"), value("delta_S"), value("eta_+"),
                       value("epsilon_S"), value("gamma_S"))
    sv = _sv_form(x, value("A_-"), value("delta_-"), value("eta_-"), value("x_0"))
    gluon = _pdf_form(x, value("A_g"), value("delta_g"), value("eta_g"),
                      value("epsilon_g"), value("gamma_g"))
    if evolution_order != "lo":
        gluon = gluon + _pdf_form(x, value("A_g'"), value("delta_g'"), value("eta_g'"), 0.0, 0.0)
    return {"uv": uv, "dv": dv, "S": S, "Delta": Delta, "g": gluon, "s_plus": s_plus, "sv": sv,
            "ubar": (-2 * Delta + S - s_plus) / 4, "dbar": (2 * Delta + S - s_plus) / 4,
            "s": (s_plus + sv) / 2, "sbar": (s_plus - sv) / 2}


def fit_input_pdfs(data_path, pdf_set, kappa=0.5, alpha_s=None, plot=True, write_to_file=True):
    """
    Fit the unpolarized input PDF parameters (MSTW schema) and write pdfs/<pdf_set>.csv.

    Parameters
    ----------
    data_path : str or Path
        PDF data file in the GUMP format (see load_pdf_data).
    pdf_set : str
        Name of the parameter set. The output is written to cfg.PDF_PATH / f"{pdf_set}.csv"
        and is selected at runtime with pdf_set = "<pdf_set>" in user_config.py.
    kappa : float, optional
        Only used if the data set has no strange entries: s = sbar = kappa/2 (ubar + dbar),
        i.e. S = (2 + kappa)(ubar + dbar) with A_+ = kappa/(2 + kappa) A_S, eta_+ = eta_S,
        A_- = 0. Default is 0.5 (neutrino dimuon range 0.4-0.5).
    alpha_s : dict, optional
        {"lo": ..., "nlo": ..., "nnlo": ...}, the value of alpha_S at the input scale
        cfg.MU_INPUT. Default: inherited from the currently configured PDF set, which is
        correct as long as that set has the same input scale. After changing cfg.MU_INPUT,
        evolve the old value with core.evolve_alpha_s and pass it here.
    plot : bool, optional
        Show data, fit and pulls. Default is True.
    write_to_file : bool, optional
        Write cfg.PDF_PATH / f"{pdf_set}.csv". Default is True.

    Returns
    -------
    dict
        Fitted parameters and 1-sigma errors per flavor combination.

    Note
    ----
    A_u, A_d and A_g are not fitted but fixed by the number (int uv = 2, int dv = 1) and
    momentum (int x [uv + dv + S + g] = 1) sum rules, and x_0 by int (s - sbar) = 0.
    The LO column uses the single-term gluon (as does the MSTW LO fit), NLO and NNLO the
    two-term gluon. Fitting a data set extracted at NLO into the LO column is a pragmatic
    refit, not a genuine LO extraction.
    """
    data = load_pdf_data(data_path)
    x = data["x"]
    if alpha_s is None:
        alpha_s = {order: get_alpha_s(order) for order in ("lo", "nlo", "nnlo")}
    results = {}

    # ---- uv, dv: normalizations fixed by the number sum rules ----
    def fit_valence(name, values, errors, number):
        def A_of(p):
            return number / _pdf_mellin(0, 1.0, *p)

        def residual(p):
            return (_pdf_form(x, A_of(p), *p) - values) / errors

        p, sigma, cov, _ = _lsq(residual, [0.7, 3.5, 1.0, 5.0],
                                ([0.05, 1.0, -30, -30], [2.0, 12.0, 60, 150]), name)
        return dict(A=A_of(p), sA=_propagated_sigma(A_of, p, cov), p=p, s=sigma)

    results["uv"] = fit_valence("uv", data["uv"], data["uv_err"], 2.0)
    results["dv"] = fit_valence("dv", data["dv"], data["dv_err"], 1.0)

    # ---- S = 2(ubar + dbar) + s + sbar ----
    if data["has_strange"]:
        S_values, S_errors = data["S"], data["S_err"]
    else:
        S_values = (2 + kappa) * (data["ubar"] + data["dbar"])
        S_errors = (2 + kappa) * np.hypot(data["ubar_err"], data["dbar_err"])

    p_S, s_S, _, _ = _lsq(lambda p: (_pdf_form(x, *p) - S_values) / S_errors,
                          [0.6, -0.1, 8.0, -2.0, 5.0],
                          ([1e-3, -0.45, 2.0, -30, -30], [50.0, 0.5, 20.0, 60, 150]), "S")
    results["S"] = dict(p=p_S, s=s_S)
    eta_S = p_S[2]

    # ---- Delta = dbar - ubar (shares eta_S with the S fit) ----
    def residual_Delta(p):
        return (_delta_form(x, p[0], p[1], eta_S, p[2], p[3]) - data["Delta"]) / data["Delta_err"]

    p_D, s_D, _, _ = _lsq(residual_Delta, [8.0, 2.0, 5.0, -30.0],
                          ([1e-3, 0.3, -50, -300], [200.0, 4.0, 100, 300]), "Delta")
    int_Delta = float(_delta_mellin(0, p_D[0], p_D[1], eta_S, p_D[2], p_D[3]))
    results["Delta"] = dict(p=p_D, s=s_D, integral=int_Delta)

    # ---- strange: s + sbar shares delta_S, epsilon_S and gamma_S with S (MSTW) ----
    if data["has_strange"]:
        def residual_s_plus(p):  # p = (A_+, eta_+)
            model = _pdf_form(x, p[0], p_S[1], p[1], p_S[3], p_S[4])
            return (model - data["s_plus"]) / data["s_plus_err"]

        p_sp, s_sp, _, _ = _lsq(residual_s_plus, [0.2, 8.0], ([1e-4, 2.0], [50.0, 25.0]), "s+sbar")
        A_plus, eta_plus = p_sp
        sA_plus, seta_plus = s_sp

        # s - sbar: delta_- is fixed (extreme correlation with A_- and eta_-, as in MSTW)
        # and x_0 follows from the strangeness number sum rule
        delta_minus = 0.2

        def residual_sv(p):  # p = (A_-, eta_-)
            model = _sv_form(x, p[0], delta_minus, p[1], _sv_x0(delta_minus, p[1]))
            return (model - data["sv"]) / data["sv_err"]

        # s - sbar is concentrated at small x, so eta_- runs large (~40); keep the bound loose
        p_sv, s_sv, _, _ = _lsq(residual_sv, [0.1, 8.0], ([-20.0, 1.0], [20.0, 60.0]), "s-sbar")
        A_minus, eta_minus = p_sv
        sA_minus, seta_minus = s_sv
        x_0 = float(_sv_x0(delta_minus, eta_minus))
    else:
        # kappa-prescription: s + sbar = kappa (ubar + dbar) = kappa/(2 + kappa) S
        A_plus, sA_plus = kappa / (2 + kappa) * p_S[0], kappa / (2 + kappa) * s_S[0]
        eta_plus, seta_plus = eta_S, s_S[2]
        A_minus, sA_minus, seta_minus = 0.0, 0.0, 0.0
        delta_minus, eta_minus = 0.2, 10.0
        x_0 = 0.017414  # inert (A_- = 0), kept finite to avoid a division by zero
    results["strange"] = dict(A_plus=A_plus, eta_plus=eta_plus, A_minus=A_minus,
                              delta_minus=delta_minus, eta_minus=eta_minus, x_0=x_0)

    # ---- gluon: A_g fixed by the momentum sum rule ----
    momentum_uv = _pdf_mellin(1, results["uv"]["A"], *results["uv"]["p"])
    momentum_dv = _pdf_mellin(1, results["dv"]["A"], *results["dv"]["p"])
    momentum_S = _pdf_mellin(1, *p_S)
    momentum_g = 1.0 - momentum_uv - momentum_dv - momentum_S
    print(f"momentum fractions: uv = {momentum_uv:.4f}, dv = {momentum_dv:.4f}, "
          f"S = {momentum_S:.4f} -> gluon = {momentum_g:.4f}")

    def A_g_of(p):  # single-term gluon (LO column)
        return momentum_g / _pdf_mellin(1, 1.0, *p)

    p_g, s_g, cov_g, _ = _lsq(lambda p: (_pdf_form(x, A_g_of(p), *p) - data["g"]) / data["g_err"],
                              [0.2, 4.0, -2.0, 3.0],
                              ([-0.85, 1.0, -30, -30], [1.5, 15.0, 60, 150]), "g (lo)")
    results["g_lo"] = dict(A=A_g_of(p_g), sA=_propagated_sigma(A_g_of, p_g, cov_g), p=p_g, s=s_g)

    # two-term gluon (NLO/NNLO): p = (delta_g, eta_g, epsilon_g, gamma_g, A_g', delta_g', eta_g')
    def A_g_two_term_of(p):
        momentum_prime = _pdf_mellin(1, p[4], p[5], p[6], 0.0, 0.0)
        return (momentum_g - momentum_prime) / _pdf_mellin(1, 1.0, *p[:4])

    def gluon_two_term(p):
        return _pdf_form(x, A_g_two_term_of(p), *p[:4]) + _pdf_form(x, p[4], p[5], p[6], 0.0, 0.0)

    p_g2, s_g2, cov_g2, _ = _lsq(lambda p: (gluon_two_term(p) - data["g"]) / data["g_err"],
                                 [0.2, 4.0, -2.0, 3.0, -1.0, -0.3, 25.0],
                                 ([-0.85, 1.0, -30, -30, -10, -0.85, 2.0],
                                  [1.5, 15.0, 60, 150, 10, 1.5, 40.0]), "g (nlo)")
    results["g_nlo"] = dict(A=A_g_two_term_of(p_g2),
                            sA=_propagated_sigma(A_g_two_term_of, p_g2, cov_g2), p=p_g2, s=s_g2)

    if plot:
        curves = {
            "uv": (data["uv"], data["uv_err"],
                   _pdf_form(x, results["uv"]["A"], *results["uv"]["p"])),
            "dv": (data["dv"], data["dv_err"],
                   _pdf_form(x, results["dv"]["A"], *results["dv"]["p"])),
            "S": (S_values, S_errors, _pdf_form(x, *p_S)),
            "Delta": (data["Delta"], data["Delta_err"],
                      _delta_form(x, p_D[0], p_D[1], eta_S, p_D[2], p_D[3])),
            "g (lo)": (data["g"], data["g_err"], _pdf_form(x, results["g_lo"]["A"], *p_g)),
            "g (nlo)": (data["g"], data["g_err"], gluon_two_term(p_g2)),
        }
        if data["has_strange"]:
            curves["s+sbar"] = (data["s_plus"], data["s_plus_err"],
                                _pdf_form(x, A_plus, p_S[1], eta_plus, p_S[3], p_S[4]))
            curves["s-sbar"] = (data["sv"], data["sv_err"],
                                _sv_form(x, A_minus, delta_minus, eta_minus, x_0))
        _plot_pdf_fits(x, curves, f"{pdf_set}: unpolarized input PDFs")

    if write_to_file:
        uv, dv = results["uv"], results["dv"]
        gluon_lo, gluon_nlo = results["g_lo"], results["g_nlo"]
        lines = ["Parameter,LO[central_value,+error,-error],NLO[central_value,+error,-error],"
                 "NNLO[central_value,+error,-error]\n"]
        lines += _param_row("alpha_S(Q0^2)", (alpha_s["lo"], 0), (alpha_s["nlo"], 0),
                            (alpha_s["nnlo"], 0))
        lines += _param_row("A_u", (uv["A"], uv["sA"]), (uv["A"], uv["sA"]))
        for i, name in enumerate(["eta_1", "eta_2", "epsilon_u", "gamma_u"]):
            lines += _param_row(name, (uv["p"][i], uv["s"][i]), (uv["p"][i], uv["s"][i]))
        lines += _param_row("A_d", (dv["A"], dv["sA"]), (dv["A"], dv["sA"]))
        lines += _param_row("eta_3", (dv["p"][0], dv["s"][0]), (dv["p"][0], dv["s"][0]))
        # dv_pdf reconstructs eta_4 = eta_2 + (eta_4 - eta_2)
        eta_42 = dv["p"][1] - uv["p"][1]
        lines += _param_row("eta_4-eta_2", (eta_42, dv["s"][1]), (eta_42, dv["s"][1]))
        for i, name in [(2, "epsilon_d"), (3, "gamma_d")]:
            lines += _param_row(name, (dv["p"][i], dv["s"][i]), (dv["p"][i], dv["s"][i]))
        for i, name in enumerate(["A_S", "delta_S", "eta_S", "epsilon_S", "gamma_S"]):
            lines += _param_row(name, (p_S[i], s_S[i]), (p_S[i], s_S[i]))
        lines += _param_row("int_0^1_dx_Delta(x,Q0^2)", (int_Delta, 0), (int_Delta, 0))
        for i, name in enumerate(["A_Delta", "eta_Delta", "gamma_Delta", "delta_Delta"]):
            lines += _param_row(name, (p_D[i], s_D[i]), (p_D[i], s_D[i]))
        # the gluon rows differ per order: single-term at LO, two-term at NLO/NNLO
        lines += _param_row("A_g", (gluon_lo["A"], gluon_lo["sA"]),
                            (gluon_nlo["A"], gluon_nlo["sA"]))
        for i, name in enumerate(["delta_g", "eta_g", "epsilon_g", "gamma_g"]):
            lines += _param_row(name, (gluon_lo["p"][i], gluon_lo["s"][i]),
                                (gluon_nlo["p"][i], gluon_nlo["s"][i]))
        # The LO values of the primed gluon are unused (gluon_pdf drops the term for "lo"),
        # but they have to stay away from the poles of the Gamma functions in the Regge moments
        for i, (name, lo_placeholder) in enumerate([("A_g'", 0.0), ("delta_g'", 0.5),
                                                    ("eta_g'", 5.0)]):
            lines += _param_row(name, (lo_placeholder, 0),
                                (gluon_nlo["p"][4 + i], gluon_nlo["s"][4 + i]))
        lines += _param_row("A_+", (A_plus, sA_plus), (A_plus, sA_plus))
        lines += _param_row("eta_+", (eta_plus, seta_plus), (eta_plus, seta_plus))
        lines += _param_row("A_-", (A_minus, sA_minus), (A_minus, sA_minus))
        lines += _param_row("delta_-", (delta_minus, 0), (delta_minus, 0))
        lines += _param_row("eta_-", (eta_minus, seta_minus), (eta_minus, seta_minus))
        lines += _param_row("x_0", (x_0, 0), (x_0, 0))
        # GM-VFNS parameters of the MSTW schema: unused here, kept for schema completeness
        lines += _param_row("alpha_S(M_Z^2)", (0.13939, 0), (0.12018, 0), (0.11707, 0))
        lines += _param_row("r_1", (-0.39484, 0), (-0.57631, 0), (-0.80834, 0))
        lines += _param_row("r_2", (-1.0719, 0), (0.81878, 0), (1.2669, 0))
        lines += _param_row("r_3", (-0.28973, 0), (-0.083208, 0), (0.15098, 0))

        file_path = cfg.PDF_PATH / f"{pdf_set}.csv"
        file_path.write_text("".join(lines))
        print(f"wrote {file_path}")

    momentum_g_nlo = (_pdf_mellin(1, results["g_nlo"]["A"], *p_g2[:4])
                      + _pdf_mellin(1, p_g2[4], p_g2[5], p_g2[6], 0.0, 0.0))
    print(f"sum rules: int uv = {_pdf_mellin(0, results['uv']['A'], *results['uv']['p']):.6f}, "
          f"int dv = {_pdf_mellin(0, results['dv']['A'], *results['dv']['p']):.6f}, "
          f"momentum (nlo) = {momentum_uv + momentum_dv + momentum_S + momentum_g_nlo:.6f}, "
          f"int Delta = {int_Delta:.4f}")
    return results


def fit_polarized_input_pdfs(data_path, pdf_set, kappa=0.5, plot=True, write_to_file=True):
    """
    Fit the polarized input factors (AAC schema) and write pdfs/<pdf_set>_POL.csv.

    Parameters
    ----------
    data_path : str or Path
        PDF data file in the GUMP format (see load_pdf_data).
    pdf_set : str
        Name of the parameter set. The unpolarized carriers are read back from
        cfg.PDF_PATH / f"{pdf_set}.csv", so run fit_input_pdfs first.
    kappa : float, optional
        Only used if the data set has no strange entries (mirror prescription
        Delta s + Delta sbar = kappa (Delta ubar + Delta dbar)). Default is 0.5.
    plot : bool, optional
        Show data, fit and pulls. Default is True.
    write_to_file : bool, optional
        Write cfg.PDF_PATH / f"{pdf_set}_POL.csv". Default is True.

    Returns
    -------
    dict
        {(row, evolution_order): (params, sigmas)} with
        params = (Delta_A, alpha, Delta_gamma, Delta_lambda).

    Note
    ----
    The model is multiplicative, Delta f = F(x) f(x), so every factor is fitted with its
    unpolarized carrier held fixed at the values of pdf_set: run this after fit_input_pdfs
    and re-run it whenever the unpolarized set changes. The written errors are
    band-calibrated: the code propagates parameter errors in uncorrelated quadrature, which
    for these strongly correlated fits would grossly overestimate the band, so the sigmas
    are rescaled such that the quadrature reproduces the true correlated 1-sigma band of
    Delta f over the fitted x-range.
    """
    data = load_pdf_data(data_path)
    x = data["x"]
    parameters = read_pdf_parameter_csv(cfg.PDF_PATH / f"{pdf_set}.csv")

    def carriers(evolution_order):
        return input_carriers(parameters, x, evolution_order)

    # AAC row -> (data combination, unpolarized carrier)
    if data["has_strange"]:
        targets = {
            "u": ("Duv", "uv"), "d": ("Ddv", "dv"), "g": ("Dg", "g"),
            "ubar": ("Dubar", "ubar"), "dbar": ("Ddbar", "dbar"),
            "s": ("Ds", "s"), "sbar": ("Dsbar", "sbar"),
            "s_plus": ("Ds_plus", "s_plus"), "S": ("DS", "S"),
        }
    else:
        # mirror-kappa: Delta S = (2 + kappa)(Delta ubar + Delta dbar), and the strange rows
        # share the light-sea factor because Delta s is proportional to it
        data["Dsea"] = data["Dubar"] + data["Ddbar"]
        data["Dsea_err"] = np.hypot(data["Dubar_err"], data["Ddbar_err"])
        data["DS"] = (2 + kappa) * data["Dsea"]
        data["DS_err"] = (2 + kappa) * data["Dsea_err"]
        targets = {
            "u": ("Duv", "uv"), "d": ("Ddv", "dv"), "g": ("Dg", "g"),
            "ubar": ("Dubar", "ubar"), "dbar": ("Ddbar", "dbar"),
            "s": ("Dsea", "s"), "sbar": ("Dsea", "sbar"),
            "s_plus": ("Dsea", "s_plus"), "S": ("DS", "S"),
        }

    fitted = {}
    for evolution_order in ("lo", "nlo"):
        carrier = carriers(evolution_order)
        for row, (data_key, carrier_key) in targets.items():
            fitted[(row, evolution_order)] = _fit_polarized_factor(
                x, data[data_key], data[data_key + "_err"], carrier[carrier_key],
                evolution_order, row)

    # positivity diagnostics |Delta f| <= f
    x_grid = np.geomspace(1e-5, 0.999, 400)
    print("positivity, max |Delta f / f| on [1e-5, 0.999] (nlo):")
    for row in targets:
        ratio = np.abs(_polarized_factor(x_grid, *fitted[(row, "nlo")][0], "nlo"))
        print(f"  {row:7s}: {ratio.max():8.3f} at x = {x_grid[np.argmax(ratio)]:.3g}")

    if plot:
        carrier = carriers("nlo")
        curves = {}
        for row, (data_key, carrier_key) in targets.items():
            factor = _polarized_factor(x, *fitted[(row, "nlo")][0], "nlo")
            curves["D" + row] = (data[data_key], data[data_key + "_err"],
                                 factor * carrier[carrier_key])
        _plot_pdf_fits(x, curves, f"{pdf_set}: polarized input factors (nlo)")

    if write_to_file:
        lines = ["Parameter,LO[central_value,+error,-error],NLO[central_value,+error,-error],"
                 "NNLO[central_value,+error,-error]\n"]
        for row in ["u", "d", "ubar", "g", "dbar", "s", "sbar", "s_plus", "S"]:
            p_lo, sigma_lo = fitted[(row, "lo")]
            p_nlo, sigma_nlo = fitted[(row, "nlo")]
            for i, name in enumerate([f"Delta_A_{row}", f"alpha_{row}",
                                      f"Delta_gamma_{row}", f"Delta_lambda_{row}"]):
                lines += _param_row(name, (p_lo[i], sigma_lo[i]), (p_nlo[i], sigma_nlo[i]))
        file_path = cfg.PDF_PATH / f"{pdf_set}_POL.csv"
        file_path.write_text("".join(lines))
        print(f"wrote {file_path}")
    return fitted


def _fit_polarized_factor(x, values, errors, carrier, evolution_order, name):
    """Fit a single polarized factor; returns (params, band-calibrated sigmas)."""
    def residual(p):
        return (_polarized_factor(x, *p, evolution_order) * carrier - values) / errors

    # the (Delta_gamma, Delta_lambda) direction is nearly degenerate -> multi-start
    lower = [-5000.0, 0.0, -400.0, 0.02]
    upper = [5000.0, 3.0, 400.0, 6.0]
    starts = [[0.5, 0.3, 0.5, 1.0], [0.5, 0.3, 0.0, 1.0], [-0.5, 0.7, 0.0, 1.0],
              [-2.5, 0.3, 2.0, 0.5], [2.5, 0.3, 2.0, 0.5], [-5.0, 1.0, 1.0, 3.0],
              [5.0, 1.0, 1.0, 3.0]]
    best = None
    for p0 in starts:
        result = least_squares(residual, p0, bounds=(lower, upper), x_scale="jac", max_nfev=50000)
        if best is None or result.cost < best.cost:
            best = result

    # a Delta_lambda pinned at a bound leaves the (Delta_gamma, Delta_lambda) direction
    # degenerate and its covariance meaningless -> refit with Delta_lambda frozen
    frozen = min(abs(best.x[3] - lower[3]), abs(best.x[3] - upper[3])) < 1e-6
    if frozen:
        delta_lambda = best.x[3]
        result = least_squares(lambda p: residual([p[0], p[1], p[2], delta_lambda]), best.x[:3],
                               bounds=(lower[:3], upper[:3]), x_scale="jac", max_nfev=50000)
        chi2red = 2 * result.cost / (len(result.fun) - len(result.x))
        cov = np.zeros((4, 4))
        cov[:3, :3] = np.linalg.inv(result.jac.T @ result.jac) * max(1.0, chi2red)
        popt = np.append(result.x, delta_lambda)
        sigma = np.append(np.sqrt(np.diag(cov)[:3]), 0.0)
    else:
        chi2red = 2 * best.cost / (len(best.fun) - len(best.x))
        cov = np.linalg.inv(best.jac.T @ best.jac) * max(1.0, chi2red)
        popt, sigma = best.x, np.sqrt(np.diag(cov))

    # band calibration (see the notes of fit_polarized_input_pdfs)
    gradient = np.zeros((4, len(x)))
    for i in range(4):
        h = 1e-6 * max(1.0, abs(popt[i]))
        plus, minus = np.array(popt, float), np.array(popt, float)
        plus[i] += h
        minus[i] -= h
        gradient[i] = (_polarized_factor(x, *plus, evolution_order)
                       - _polarized_factor(x, *minus, evolution_order)) / (2 * h) * carrier
    band_correlated = np.sqrt(np.einsum("ik,ij,jk->k", gradient, cov, gradient))
    band_quadrature = np.sqrt(np.einsum("ik,i,ik->k", gradient, sigma ** 2, gradient))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(band_correlated > 0, band_quadrature / band_correlated, 1.0)
    calibration = max(float(np.median(ratio[np.isfinite(ratio) & (ratio > 0)])), 1.0)
    sigma = sigma / calibration

    tag = " [Delta_lambda frozen]" if frozen else ""
    print(f"{name + ' (' + evolution_order + ')':14s} chi2/dof = {chi2red:8.3f}   "
          f"params = {np.array2string(popt, precision=4)}   "
          f"(band calibration /{calibration:.2f}){tag}")
    return popt, sigma


def _plot_pdf_fits(x, curves, title):
    """Data, fit and pulls for each entry of curves = {name: (values, errors, model)}."""
    n = len(curves)
    fig, axes = plt.subplots(2, n, figsize=(3.6 * n, 6.5), sharex=True, squeeze=False,
                             gridspec_kw={"height_ratios": [3, 1]})
    for i, (name, (values, errors, model)) in enumerate(curves.items()):
        ax, ax_pull = axes[0][i], axes[1][i]
        ax.errorbar(x, x * values, yerr=x * errors, fmt=".", ms=3, label="data")
        ax.plot(x, x * model, "-", lw=1.2, label="fit")
        ax.set_xscale("log")
        ax.set_title(f"x {name}(x)", fontsize=10)
        ax.legend(fontsize=8)
        ax_pull.plot(x, (model - values) / errors, ".", ms=3)
        ax_pull.axhline(0, color="k", lw=0.5)
        ax_pull.set_xscale("log")
        ax_pull.set_xlabel("x")
        ax_pull.set_ylabel("pull")
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()