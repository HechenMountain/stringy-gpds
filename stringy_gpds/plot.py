import numpy as np
import matplotlib.pyplot as plt
import time
import os

from . import config as cfg
from . import helpers as hp
from . import core
from . import mstw_pdf as mstw
from . import aac_pdf as aac

# mpmath precision set in config
from .config import mp

from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.interpolate import RectBivariateSpline, interp1d

#########################
####### Plot PDFs #######
#########################

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
    hp.check_evolution_order(evolution_order)
    vectorized_uv_pdf = np.vectorize(mstw.uv_pdf)
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
    hp.check_evolution_order(evolution_order)
    vectorized_dv_pdf = np.vectorize(mstw.dv_pdf)
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
    hp.check_evolution_order(evolution_order)
    vectorized_uv_minus_dv_pdf = np.vectorize(mstw.uv_minus_dv_pdf)
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
    hp.check_evolution_order(evolution_order)
    vectorized_uv_plus_dv_plus_S_pdf = np.vectorize(mstw.uv_plus_dv_plus_S_pdf)
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
    hp.check_evolution_order(evolution_order)
    vectorized_gluon_pdf = np.vectorize(mstw.gluon_pdf)
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

def plot_polarized_uv_pdf(x_0=1e-2,evolution_order="nlo",logplot = False,error_bars=True):
    """
    Plot the polarized uv PDF at a fixed value of x.

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
    hp.check_evolution_order(evolution_order)
    vectorized_polarized_uv_pdf = np.vectorize(aac.polarized_uv_pdf)
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

def plot_polarized_dv_pdf(x_0=1e-2,evolution_order="nlo",logplot = False,error_bars=True):
    """
    Plot the polarized dv PDF at a fixed value of x.

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
    hp.check_evolution_order(evolution_order)
    vectorized_polarized_dv_pdf = np.vectorize(aac.polarized_dv_pdf)
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


def plot_polarized_ubar_pdf(x_0=1e-2,evolution_order="nlo",logplot = False,error_bars=True):
    """
    Plot the polarized ubar PDF at a fixed value of x.

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
    hp.check_evolution_order(evolution_order)
    vectorized_polarized_ubar_pdf = np.vectorize(aac.polarized_ubar_pdf)
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

def plot_polarized_uv_minus_dv_pdf(x_0=1e-2,evolution_order="nlo",logplot = False,error_bars=True):
    """
    Plot the polarized uv - dv (non_singlet_isovector) PDF at a fixed value of x.

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
    hp.check_evolution_order(evolution_order)
    vectorized_polarized_uv_minus_dv_pdf = np.vectorize(aac.polarized_uv_minus_dv_pdf)
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

def plot_polarized_uv_plus_dv_plus_S_pdf(x_0=1e-2,evolution_order="nlo",logplot = False,error_bars=True):
    """
    Plot the polarized uv + dv + S (singlet quark) PDF at a fixed value of x.

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
    hp.check_evolution_order(evolution_order)
    vectorized_polarized_uv_plus_dv_plus_S_pdf = np.vectorize(aac.polarized_uv_plus_dv_plus_S_pdf)
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

def plot_polarized_gluon_pdf(x_0=1e-2,y_0=-1,y_1=1,evolution_order="nlo",logplot = False,error_bars=True):
    """
    Plot the polarized gluon PDF at a fixed value of x.

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
    hp.check_evolution_order(evolution_order)
    vectorized_polarized_gluon_pdf = np.vectorize(aac.polarized_gluon_pdf)
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

############################
####### Plot Moments #######
############################

def plot_moment(n, eta, y_label, mu_in=2, t_max=3, A0=1, particle="quark",
                moment_type="non_singlet_isovector", moment_label="A",
                evolution_order="nlo", n_t=50):
    """
    Generate a plot comparing lattice data with RGE-evolved results for a given moment type and label.

    Parameters
    ----------
    n : int
        Conformal spin.
    eta : float
        Skewness parameter.
    y_label : str
        Label for the y-axis of the plot.
    mu_in : float, optional
        Resolution scale.
    t_max : float, optional
        Maximum value of -t shown on the x-axis. Default is 3.
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
    n_t : int, optional
        Number of sampling points in t-space. Default is 50.

    Returns
    -------
    None

    Notes
    -----
    This function is intended for visualization in a jupyter notebook.
    """

    hp.check_particle_type(particle)
    hp.check_moment_type_label(moment_type,moment_label)
    # Accessor functions for -t, values, and errors
    def t_values(moment_type, moment_label, pub_id):
        """Return the -t values for a given moment type, label, and publication ID."""
        data, n_to_row_map = hp.read_lattice_moment_data(particle,moment_type, moment_label, pub_id)

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
    def compute_results(j, eta, t_vals, mu,  particle="quark", moment_type="non_singlet_isovector", moment_label="A"):
        """Compute central, plus, and minus results for a given evolution function."""
        if moment_type != "D":
            results = Parallel(n_jobs=-1)(
                delayed(lambda t: float(core.evolve_conformal_moment(j, eta, t, mu,  A0, particle, moment_type, moment_label, evolution_order, "central").real))(t)
                for t in t_vals
            )
            results_plus = Parallel(n_jobs=-1)(
                delayed(lambda t: float(core.evolve_conformal_moment(j, eta, t, mu,  A0, particle, moment_type, moment_label, evolution_order, "plus").real))(t)
                for t in t_vals
            )
            results_minus = Parallel(n_jobs=-1)(
                delayed(lambda t: float(core.evolve_conformal_moment(j, eta, t, mu,  A0, particle, moment_type, moment_label, evolution_order, "minus").real))(t)
                for t in t_vals
            )
            return results, results_plus, results_minus
        else:
            results = Parallel(n_jobs=-1)(
                delayed(lambda t: float(core.evolve_singlet_D(j, eta, t, mu,  A0, particle, moment_label, evolution_order, "central").real))(t)
                for t in t_vals
            )
            results_plus = Parallel(n_jobs=-1)(
                delayed(lambda t: float(core.evolve_singlet_D(j, eta, t, mu,  A0, particle, moment_label, evolution_order, "plus").real))(t)
                for t in t_vals
            )
            results_minus = Parallel(n_jobs=-1)(
                delayed(lambda t: float(core.evolve_singlet_D(j, eta, t, mu,  A0, particle, moment_label, evolution_order, "minus").real))(t)
                for t in t_vals
            )
            return results, results_plus, results_minus

    # Define the finer grid for t-values
    T_Fine = np.linspace(-t_max, 0, n_t)
    
    # Initialize the figure and axes for subplots
    fig, ax = plt.subplots(figsize=(7, 7)) 

    evolve_moment_central, evolve_moment_plus, evolve_moment_minus = compute_results(n,eta,T_Fine,mu_in,particle,moment_type,moment_label)
    
    # Plot the RGE functions
    ax.plot(-T_Fine, evolve_moment_central, color="blue", linewidth=2, label="This work")
    ax.fill_between(-T_Fine, evolve_moment_minus, evolve_moment_plus, color="blue", alpha=0.2)
    
    # Plot data from publications

    for pub_id, (color,mu) in cfg.PUBLICATION_MAPPING.items():
        if mu != mu_in:
            continue
        data, n_to_row_map = hp.read_lattice_moment_data(particle,moment_type, moment_label, pub_id)
        if data is None or n not in n_to_row_map:
            continue
        t_vals = t_values(moment_type, moment_label, pub_id)
        Fn0_vals = hp.Fn0_values(n, particle, moment_type, moment_label, pub_id)
        Fn0_errs = hp.Fn0_errors(n, particle, moment_type, moment_label, pub_id)
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

def plot_moments_on_grid(eta, y_label, t_max=3, A0=1, particle="quark", moment_type="non_singlet_isovector", moment_label="A",evolution_order="nlo", n_t=50, num_columns=3,D_term = False,set_y_lim=False,y_0 = -1, y_1 =1):
    """
    Generate a plot comparing lattice data with RGE-evolved results for a given moment type and label.

    Parameters
    ----------
    n : int
        Conformal spin.
    eta : float
        Skewness parameter.
    y_label : str
        Label for the y-axis of the plot.
    mu_in : float, optional
        Resolution scale.
    t_max : float, optional
        Maximum value of -t shown on the x-axis. Default is 3.
    A0 : float, optional
        Normalization factor (default A0 = 1).
    particle : str, optional
        "quark" or "gluon". Default is "quark".
    moment_type : str, optional
        "non_singlet_isovector", "non_singlet_isoscalar", or "singlet".
    moment_label : str, optional
        A(tilde), B(tilde) depending on H(tilde) or E(tilde) GPD etc. Default is "A".
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo".
    n_t : int, optional
        Number of sampling points in t-space. Default is 50.
    num_columns : int, optional
        Number of columns. Default is 3
    D_term : bool, optional
        Extract the D-term from the evolved moment
    set_y_lim : bool, optional
        Manually control y-axis limits. Default is False.
    y_0 : float, optional
        Lower bound on y-axis. Only has effect when set_y_lim = True. Default is -1
    y_1 : float, optional
        Upper bound on y-axis. Only has effect when set_y_lim = True. Default is 1

    Returns
    -------
    None

    Notes
    -----
    This function is intended for visualization in a jupyter notebook. The plots are saved to 
    cfg.PLOT_PATH / f"{moment_type}_{particle}_{data_moment_label}_n_{n}.pdf"
    Where PLOT_PATH is defined in config.py
    """
    hp.check_particle_type(particle)
    hp.check_moment_type_label(moment_type, moment_label)

    if not D_term:
        data_moment_label = moment_label
    else:
        data_moment_label = "D"

    # Accessor functions for -t, values, and errors
    def t_values(data_moment_type, data_moment_label, pub_id):
        """Return the -t values for a given moment type, label, and publication ID."""
        data, n_to_row_map = hp.read_lattice_moment_data(particle,data_moment_type, data_moment_label, pub_id)

        if data is None and n_to_row_map is None:
            print(f"No data found for {data_moment_type} {data_moment_label} {pub_id}. Skipping.")
            return None 
        
        if data is not None:
            return data[:, 0]
        else:
            print(f"Data is None for {data_moment_type} {data_moment_label} {pub_id}. Skipping.")
        return None  

    def compute_results(j,t_vals):
        """Compute central, plus, and minus results for a given evolution function."""
        if not D_term:
            results = Parallel(n_jobs=-1)(
                delayed(lambda t: float(core.evolve_conformal_moment(j, eta, t, mu,  A0, particle, moment_type, moment_label, evolution_order, "central").real))(t)
                for t in t_vals
            )
            results_plus = Parallel(n_jobs=-1)(
                delayed(lambda t: float(core.evolve_conformal_moment(j, eta, t, mu,  A0, particle, moment_type, moment_label, evolution_order, "plus").real))(t)
                for t in t_vals
            )
            results_minus = Parallel(n_jobs=-1)(
                delayed(lambda t: float(core.evolve_conformal_moment(j, eta, t, mu,  A0, particle, moment_type, moment_label, evolution_order, "minus").real))(t)
                for t in t_vals
            )
            return results, results_plus, results_minus
        else:
            results = Parallel(n_jobs=-1)(
                delayed(lambda t: float(core.evolve_singlet_D(j, eta, t, mu, A0, particle, moment_label, evolution_order, "central").real))(t)
                for t in t_vals
            )
            results_plus = Parallel(n_jobs=-1)(
                delayed(lambda t: float(core.evolve_singlet_D(j, eta, t, mu, A0, particle, moment_label, evolution_order, "plus").real))(t)
                for t in t_vals
            )
            results_minus = Parallel(n_jobs=-1)(
                delayed(lambda t: float(core.evolve_singlet_D(j, eta, t, mu, A0, particle, moment_label, evolution_order, "minus").real))(t)
                for t in t_vals
            )
            return results, results_plus, results_minus
    if D_term:
        T_Fine = np.linspace(-t_max, -1e-3, n_t)
    else:
        T_Fine = np.linspace(-t_max, 0, n_t)

    # Initialize publication data
    publication_data = {}
    mu = None
    for pub_id, (color,mu) in cfg.PUBLICATION_MAPPING.items():
        data, n_to_row_map = hp.read_lattice_moment_data(particle,moment_type, data_moment_label, pub_id)
        if data is None and n_to_row_map is None:
            continue
        num_n_values = (data.shape[1] - 1) // 2
        publication_data[pub_id] = num_n_values

    if mu is None:
        mu = 2 

    if publication_data:
        max_n_value = max(publication_data.values())
        # Exclude first unpolarized singlet moment
        if moment_type == "singlet" and moment_label == "A":
            max_n_value+=1
    else:
        max_n_value = 4
    # Calculate rows for grid layout
    num_rows = (max_n_value + num_columns - 1) // num_columns

    # Create a figure for the grid of subplots (for displaying in notebook)
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_columns, figsize=(15, num_rows * 5))
    axes = axes.flatten()

    if moment_type == "singlet" and moment_label == "A":
        n_0 = 2
    else:
        n_0 = 1
 
    for n in range(n_0, max_n_value + 1):
        ax = axes[n - 1]  # Select the appropriate axis
        
        # Compute results for the current n
        evolve_moment_central, evolve_moment_plus, evolve_moment_minus = compute_results(n, T_Fine)

        if publication_data:
            ax.plot(-T_Fine, evolve_moment_central, color="blue", linewidth=2, label="This work")
        else:
            ax.plot(-T_Fine, evolve_moment_central, color="blue", linewidth=2)
        ax.fill_between(-T_Fine, evolve_moment_minus, evolve_moment_plus, color="blue", alpha=0.2)

        # Plot data from publications
        if publication_data:
            for pub_id, (color, mu) in cfg.PUBLICATION_MAPPING.items():
                data, n_to_row_map = hp.read_lattice_moment_data(particle,moment_type, data_moment_label, pub_id)
                if data is None or n not in n_to_row_map:
                    continue
                t_vals = t_values(moment_type, data_moment_label, pub_id)
                Fn0_vals = hp.Fn0_values(n, particle, moment_type, data_moment_label, pub_id)
                Fn0_errs = hp.Fn0_errors(n, particle, moment_type, data_moment_label, pub_id)
                ax.errorbar(t_vals, Fn0_vals, yerr=Fn0_errs, fmt='o', color=color, label=f"{pub_id}")
            ax.legend()

        # Add labels and formatting
        ax.set_xlabel("$-t\,[\mathrm{GeV}^2]$", fontsize=14)
        ax.set_ylabel(f"{y_label}$(j={n}, \\eta=0, t, \\mu={mu}\, \\mathrm{{GeV}})$", fontsize=14)
 
        ax.grid(True, which="both")
        ax.set_xlim([0, t_max])
        if set_y_lim:
            ax.set_ylim([y_0,y_1])

        # Save each plot as a separate PDF
        pdf_path = cfg.PLOT_PATH / f"{moment_type}_{particle}_{data_moment_label}_n_{n}.pdf"
        
        # Create a new figure to save the current plot as individual PDFs
        fig_single, ax_single = plt.subplots(figsize=(7, 5))  # New figure for saving each plot
        
        # Plot the functions
        ax_single.plot(-T_Fine, evolve_moment_central, color="blue", linewidth=2)
        ax_single.fill_between(-T_Fine, evolve_moment_minus, evolve_moment_plus, color="blue", alpha=0.2)

        # Plot data from publications
        if publication_data:
            for pub_id, (color, mu) in cfg.PUBLICATION_MAPPING.items():
                data, n_to_row_map = hp.read_lattice_moment_data(particle,moment_type, data_moment_label, pub_id)
                if data is None or n not in n_to_row_map:
                    continue
                t_vals = t_values(moment_type, data_moment_label, pub_id)
                Fn0_vals = hp.Fn0_values(n, particle, moment_type, data_moment_label, pub_id)
                Fn0_errs = hp.Fn0_errors(n, particle, moment_type, data_moment_label, pub_id)
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

    if moment_type == "singlet" and moment_label == "A":
        # Delete empty plot for unpolarized singlet
        fig.delaxes(axes[0])

    # Adjust GridSpec to remove whitespace
    gs = plt.GridSpec(num_rows, num_columns, figure=fig)
    remaining_axes = fig.get_axes()  # Get all remaining axes

    for i, ax in enumerate(remaining_axes):
        ax.set_subplotspec(gs[i])  # Assign axes to new GridSpec slots

    # Show the full grid of subplots in the notebook
    plt.tight_layout()
    plt.show()

    # Close the figure
    plt.close(fig)

################################
#### RGE evolved Quantities ####
################################

def plot_spin_orbit_correlation(eta,mu,particle="quark",evolution_order="nlo",n_t = 50):
    """
    Generate a plot of the spin-orbit correlation over mu for given parameters and compares to lattice data.

    Parameters
    ----------
    eta : float
        Skewness parameter.
    mu : float
        Resolution scale in GeV.
    particle : str, optional
        "quark" or "gluon". Default is "quark".
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    n_t : int, optional
        Number of sampling points for the t-grid. Default is 50.

    Returns
    -------
    None

    Notes
    -----
    This function plots the spin-orbit correlation extracted from the 
    corresponding GPD moments and compares to available lattice data. 

    Lattice publication IDs are hard-coded in this function. Modify, if necessary.

    Saves the plot as cfg.PLOT_PATH / "Cz_over_t.pdf" where PLOT_PATH is set in config.py
    """
    def compute_result(t_vals,moment_type):
        parallel_results = Parallel(n_jobs=-1)(
            delayed(lambda t: core.spin_orbit_correlation(eta=eta,t=t,mu=mu,
                                                    particle=particle,moment_type=moment_type,
                                                    evolution_order=evolution_order))(t)
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
    moment_types = ["non_singlet_isoscalar", "non_singlet_isovector"]
    labels = [r"$C_z^{{u+d}}(t)$", r"$C_z^{{u-d}}(t)$"] 
    colors = ["black","red"]
    t_min, t_max = 0,0

    # Store which moment types are available
    moment_data = []
    for j, pub in enumerate(publications):
        for i, moment_type in enumerate(moment_types):
            t_values, val_data, err_data = hp.read_Cz_data(particle,moment_type,pub[0],pub[1])
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
    T_Fine = np.linspace(-t_min,-t_max,n_t)

    for i, moment_type in enumerate(moment_data):
        results, results_plus, results_minus = compute_result(T_Fine,moment_type)
        plt.plot(-T_Fine,results,color=colors[i],linewidth=2, label=labels[i])
        plt.fill_between(-T_Fine,results_minus,results_plus,color=colors[i],alpha=.2)
    #padding = .05 *  (t_max - t_min)
    padding = 0
    plt.xlim(t_min-padding,t_max+padding)
    plt.xlabel("$-t\,[\mathrm{GeV}^2]$")
    plt.legend(fontsize=14, markerscale=1.5)
    plt.grid(True)
    #plt.yscale('log') # set y axis to log scale
    #plt.xscale('log') # set x axis to log scale
    plt.tight_layout()

    FILE_PATH = cfg.PLOT_PATH / "Cz_over_t.pdf"
    plt.savefig(FILE_PATH,format="pdf",bbox_inches="tight",dpi=600)

    plt.show()

def plot_orbital_angular_momentum(eta,mu,particle="quark",evolution_order="nlo",n_t = 50):
    """
    Generate a plot of the orbital angular momentum over mu for given parameters and compares to lattice data.

    Parameters
    ----------
    eta : float
        Skewness parameter.
    mu : float
        Resolution scale in GeV.
    particle : str, optional
        "quark" or "gluon". Default is "quark".
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    n_t : int, optional
        Number of sampling points for the t-grid. Default is 50.

    Returns
    -------
    None

    Notes
    -----
    This function plots the orbital angular momentum extracted from the 
    corresponding GPD moments and compares to available lattice data. 

    Lattice publication IDs are hard-coded in this function. Modify, if necessary.

    Saves the plot as cfg.PLOT_PATH / "Lz_over_t.pdf" where PLOT_PATH is set in config.py
    """
    def compute_result(t_vals,moment_type):
        parallel_results = Parallel(n_jobs=-1)(
            delayed(lambda t: core.orbital_angular_momentum(eta=eta,t=t,mu=mu,
                                                    particle=particle,moment_type=moment_type,
                                                    evolution_order=evolution_order))(t)
            for t in t_vals)
        results, error_plus, error_minus = zip(*parallel_results)
        results = np.array(results)
        error_plus = np.array(error_plus)
        error_minus = np.array(error_minus)
        results_plus = results + error_plus
        results_minus = results - error_minus
        return results, results_plus, results_minus
    
    publications = [("2305.11117","2305.11117","2410.03539"),("0705.4295","0705.4295","0705.4295")]
    shapes = ['o','^']
    moment_types = ["non_singlet_isoscalar", "non_singlet_isovector"]
    labels = [r"$L_z^{{u+d}}(t)$", r"$L_z^{{u-d}}(t)$"] 
    colors = ["black","red"]
    t_min, t_max = 0,0

    # Store which moment types are available
    moment_data = []
    for j, pub in enumerate(publications):
        for i, moment_type in enumerate(moment_types):
            t_values, val_data, err_data = hp.read_Lz_data(particle,moment_type,pub[0],pub[1],pub[2])
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
    T_Fine = np.linspace(-t_min,-t_max,n_t)

    for i, moment_type in enumerate(moment_data):
        results, results_plus, results_minus = compute_result(T_Fine,moment_type)
        plt.plot(-T_Fine,results,color=colors[i],linewidth=2, label=labels[i])
        plt.fill_between(-T_Fine,results_minus,results_plus,color=colors[i],alpha=.2)
    #padding = .05 *  (t_max - t_min)
    padding = 0
    plt.xlim(t_min-padding,t_max+padding)
    plt.xlabel("$-t\,[\mathrm{GeV}^2]$")
    plt.legend(fontsize=14, markerscale=1.5)
    plt.grid(True)
    #plt.yscale('log') # set y axis to log scale
    #plt.xscale('log') # set x axis to log scale
    plt.tight_layout()

    FILE_PATH = cfg.PLOT_PATH / "Lz_over_t.pdf"
    plt.savefig(FILE_PATH,format="pdf",bbox_inches="tight",dpi=600)

    plt.show()

############################
####   Plots in impact  ####
####   parameter space  ####
############################

def plot_fourier_transform_moments(n, eta, mu, plot_title="", particle="quark",
                                   moment_type="non_singlet_isovector", moment_label="A",
                                   evolution_order="nlo", b_max=2, Delta_max=5,
                                   error_type="central"):
    """
    Generate a 2D density plot of the Fourier transform of RGE-evolved conformal moments
    for a given moment type and label in impact parameter space..

    Parameters
    ----------
    n : int
        Conformal spin.
    eta : float
        Skewness parameter.
    mu : float
        Resolution scale in GeV.
    plot_title : str, optional
        Title of the plot.
    particle : str, optional
        "quark" or "gluon". Default is "quark".
    moment_type : str, optional
        "non_singlet_isovector", "non_singlet_isoscalar", or "singlet".
    moment_label : str, optional
        A(tilde), B(tilde) depending on H(tilde) or E(tilde) GPD etc. Default is "A".
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    b_max : float, optional
        Maximum value of impact parameter in Gev^-1 used for the plot.
    Delta_max : float, optional
        Upper bound for the momentum magnitude in the integration. Default is 5 GeV.
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"

    Returns
    -------
    None

    Notes
    -----
    This function creates a 2D density plot of the Fourier-transformed conformal moments and displays
    the resulting distribution as a density plot. Intended for use in a jupyter notebook.
    Does not save a plot.
    """
    hp.check_particle_type(particle)
    hp.check_moment_type_label(moment_type,moment_label)
    # Define the grid for b_vec
    b_x = np.linspace(-b_max, b_max, 50)  # Range of x-component of b_vec
    b_y = np.linspace(-b_max, b_max, 50)  # Range of y-component of b_vec
    b_x, b_y = np.meshgrid(b_x, b_y)  # Create a grid of (b_x, b_y)
    # Flatten the grid for parallel processing
    b_vecs = np.array([b_x.ravel(), b_y.ravel()]).T

    def ft_moment(b_vec):
        return core.fourier_transform_moment(n=n,eta=eta,mu=mu,b_vec=b_vec,
                                        particle=particle,moment_type=moment_type, moment_label=moment_label, 
                                        evolution_order=evolution_order,Delta_max=Delta_max,error_type=error_type,dipole_form=True)
    # Parallel computation using joblib
    fourier_transform_moment_values_flat = Parallel(n_jobs=-1)(delayed(ft_moment)(b_vec) for b_vec in b_vecs)

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
    plt.title(f"{plot_title}$(j={n}, \\eta=0, t, \\mu={mu}\, \\mathrm{{GeV}})$", fontsize=14)
    plt.show()

def plot_fourier_transform_transverse_moments(n, eta, mu, particle="quark",
                                              moment_type="non_singlet_isovector", evolution_order="nlo",
                                              b_max=4.5, Delta_max=7, num_points=100, n_b=100,
                                              interpolation=True, n_int=300,
                                              vmin=0, vmax=1,
                                              write_to_file=False, read_from_file=True):
    """
    Generate a 2D density plot of the RGE-evolved conformal moments
    for a transversely polarized target in impact parameter space..

    Parameters
    ----------
    n : int
        Conformal spin.
    eta : float
        Skewness parameter.
    mu : float
        Resolution scale in GeV.
    particle : str, optional
        "quark" or "gluon". Default is "quark".
    moment_type : str, optional
        "non_singlet_isovector", "non_singlet_isoscalar", or "singlet".
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo".
    b_max : float, optional
        Maximum value of impact parameter in Gev^-1 used for the plot.
    Delta_max : float, optional
        Upper bound for the momentum magnitude in the integration. Default is 5 GeV.
    num_points : int, optional
        Number of points used in trapezoidal integration. Default is 100.
    n_b : int, optional
        Number of points for discretizing the transverse position plane. Default is 100.
    interpolation : bool, optional
        Whether to interpolate data to plot on a finer grid. Default is True.
    n_int : int, optional
        Number of interpolation points. Default is 300.
    vmin : float, optional
        Minimum value of the colorbar. Default is 0.
    vmax : float, optional
        Maximum value of the colorbar. Default is 1.
    write_to_file : bool, optional
        Whether to save the computed data to the file system. Default is False.
    read_from_file : bool, optional
        Whether to load data from file system if available. Default is True.

    Returns
    -------
    None

    Notes
    -----
    This function creates a 2D density plot of the Fourier-transformed conformal
    moments for a transversely polarized target. 
    
    The plot is saved under
    cfg.PLOT_PATH / f"imp_param_transv_pol_moment_j_{n}_{moment_type}_{eta}_{t}_{mu}.pdf" where 
    PLOT_PATH is defined in config.py

    If write_to_file is set to true the data is saved under 
    cfg.IMPACT_PARAMETER_MOMENTS_PATH / f"imp_param_transv_pol_moment_j_{n}_{mom_type}_{eta}_{t}_{mu}.csv"
    where IMPACT_PARAMETER_MOMENTS_PATH is defined in config.py

    If both `write_to_file` and `read_from_file` are True, a ValueError is raised.
    """
    def ft_transverse_moment(b_vec):
        return core.fourier_transform_transverse_moment(n=n,eta=eta, mu=mu, b_vec=b_vec,  A0=1,
                                                   particle=particle,moment_type=mom_type, evolution_order=evolution_order, 
                                                   Delta_max=Delta_max, num_points=num_points, error_type="central",dipole_form=True)
    hp.check_particle_type(particle)

    if moment_type not in ["non_singlet_isovector", "non_singlet_isoscalar", "u", "d", "all","singlet"]:
        raise ValueError(f"Wrong moment_type {moment_type}")

    if write_to_file and read_from_file:
        raise ValueError("write_to_file and read_from_file can't simultaneously be True")

    prfx =  f"imp_param_transv_pol_moment_j_{n}_{moment_type}"
    file_name = hp.generate_filename(eta,0,mu,prfx,file_ext="pdf")
    FILE_PATH = cfg.PLOT_PATH / file_name

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
        READ_WRITE_PATH = cfg.IMPACT_PARAMETER_MOMENTS_PATH / f"imp_param_transv_pol_moment_j_{n}_{mom_type}"
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
                    file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"central")
                    b_x_fm, b_y_fm, fourier_transform_moment_values_flat = hp.read_ft_from_csv(file_name)
                    n_b = len(fourier_transform_moment_values_flat)
                    b_x = np.linspace(-b_max, b_max, n_b)
                    b_y = np.linspace(-b_max, b_max, n_b)
                    # Exctract shape for reshaping
                    b_x_grid, b_y_grid = np.meshgrid(b_x, b_y)
                else:
                    fourier_transform_moment_values_flat = Parallel(n_jobs=-1)(delayed(ft_transverse_moment)(b_vec)
                         for b_vec in b_vecs)
                    # Reshape
                    fourier_transform_moment_values_flat = np.array(fourier_transform_moment_values_flat).reshape(b_x_grid.shape)
                    # Convert to fm^-2
                    fourier_transform_moment_values_flat = fourier_transform_moment_values_flat/hbarc**2
                cache[mom_type] = fourier_transform_moment_values_flat

        if mom_type in ["u","d"]:
            if read_from_file:
                file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"central")
                b_x_fm, b_y_fm, _ = hp.read_ft_from_csv(file_name)
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
        ax.set_title(rf"$\rho_{{{n},\perp}}^{{{title}}}$", fontsize=14)
        ax.set_xlim([-b_max * hbarc, b_max * hbarc])
        ax.set_ylim([-b_max * hbarc, b_max * hbarc])

        # Add colorbar only once per row
        if col == len(moment_types)-1:
            cbar_ax = fig.add_axes([ax.get_position().x1, ax.get_position().y0, 0.01, ax.get_position().height])
            fig.colorbar(im, cax=cbar_ax)

        if write_to_file:
            file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"central")
            hp.save_ft_to_csv(b_x_fm_write_out,b_y_fm_write_out,ft_write_out,file_name)

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


def plot_fourier_transform_transverse_moments_grid(n_max, eta, mu,
                                                   particle="quark", interpolation=True,
                                                   n_int=300, vmin=0, vmax=1):
    """
    Generate a grid of 2D density plots showing the Fourier transforms of 
    RGE-evolved conformal moments for a transversely polarized proton.

    The A and B moments are used automatically. All precomputed data tables 
    must have the same `b_max` and be present on the file system. Use 
    `plot_fourier_transform_transverse_moments` to generate them.

    Parameters
    ----------
    n_max : int
        Maximum conformal spin to include in the grid (plots for all n = 1, ..., n_max).
    eta : float
        Skewness parameter.
    mu : float
        Resolution scale in GeV.
    particle : str, optional
        "quark" or "gluon". Default is "quark".
    interpolation : bool, optional
        Whether to interpolate data onto a finer grid for smoother plots. Default is True.
    n_int : int, optional
        Number of points used in each interpolation dimension. Default is 300.
    vmin : float, optional
        Minimum value for the colormap (colorbar scale). Default is 0.
    vmax : float, optional
        Maximum value for the colormap (colorbar scale). Default is 1.

    Returns
    -------
    None

    Notes
    -----
    This function creates a 2D density plot of the Fourier-transformed conformal
    moments for a transversely polarized target. The plot is saved under
    cfg.PLOT_PATH / "imp_param_transv_pol_moments_{eta}_{t}_{mu}.pdf"
    where PLOT_PATH is defined in config.py.
    """

    def get_subplot_positions_and_heights(n_rows,n_cols):
        """
        Returns the positions and heights of each subplot in the grid.
        
        Returns:
        - A list of row heights and positions for each subplot.
        """
        row_positions_and_heights = []
        
        # Step 1: Create a hidden figure to determine subplot heights without labels/ticks
        fig_hidden, axs_hidden = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows*4))
        fig_hidden.subplots_adjust(wspace=0, hspace=0)  # Remove extra spacing
        
        for row in range(n_rows):
            # Get the bounding box (position and height) of the last column in each row
            bbox = axs_hidden[row,-1].get_position()
            row_positions_and_heights.append((bbox.x0, bbox.y0, bbox.width, bbox.height))  # Position (x0, y0), width, and height
        
        plt.close(fig_hidden)  # Close the hidden figure
        
        return row_positions_and_heights

            
    hp.check_particle_type(particle)

    if len(vmin)<n_max or len(vmax)<n_max:
        raise ValueError("Supply vmin and vmax as arrays of length n_max")

    prfx = "imp_param_transv_pol_moments"
    file_name = hp.generate_filename(eta,0,mu,prfx,file_ext="pdf")
    FILE_PATH = cfg.PLOT_PATH / file_name

    moment_types = ["non_singlet_isovector", "non_singlet_isoscalar", "u", "d"]

    # Initialize cache to store Fourier transforms for "non_singlet_isovector" and "non_singlet_isoscalar"
    #cache = {}
    cache = {j: {mom_type: None for mom_type in moment_types} for j in range(1, n_max + 1)}

    # Determine figure layout
    fig, axs = plt.subplots(n_max, len(moment_types), figsize=(len(moment_types) * 4, n_max*4))
    row_positions_and_heights = get_subplot_positions_and_heights(n_max,len(moment_types))

    title_map = {
        "non_singlet_isovector": "u-d",
        "non_singlet_isoscalar": "u+d",
        "u": "u",
        "d": "d"
    }
    hbarc = 0.1975

    for j in range(1, n_max + 1):
        for i, mom_type in enumerate(moment_types):
            READ_WRITE_PATH = cfg.IMPACT_PARAMETER_MOMENTS_PATH /  f"imp_param_transv_pol_moment_j_{j}_{mom_type}"
            row, col = j-1, i
            ax = axs[row, col]

            title = title_map[mom_type]

            # Compute Fourier transform and cache the results for non_singlet_isovector and non_singlet_isoscalar
            if mom_type in ["non_singlet_isovector", "non_singlet_isoscalar"]:
                if cache[j][mom_type] is None:
                    file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"central")
                    b_x_fm, b_y_fm, fourier_transform_moment_values_flat = hp.read_ft_from_csv(file_name)
                    b_max = max(b_x_fm)/hbarc
                    n_b = len(fourier_transform_moment_values_flat)
                    b_x = np.linspace(-b_max, b_max, n_b)
                    b_y = np.linspace(-b_max, b_max, n_b)
                    cache[j][mom_type] = fourier_transform_moment_values_flat

            if mom_type in ["u","d"]:
                file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"central")
                b_x_fm, b_y_fm, _ = hp.read_ft_from_csv(file_name)

                if mom_type == "u":
                    prf = 1
                if mom_type == "d":
                    prf = -1 
                fourier_transform_moment_values_flat = (cache[j]["non_singlet_isoscalar"] + prf * cache[j]["non_singlet_isovector"]) / 2

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
                                shading='auto', cmap='jet',vmin=vmin[j-1], vmax=vmax[j-1],rasterized=True)
            ax.set_xlabel(r'$b_x\,[\mathrm{fm}]$', fontsize=14)
            if i == 0:
                ax.set_ylabel(r'$b_y\,[\mathrm{fm}]$', fontsize=14)
            if j == 1:
                ax.set_title(rf"$\rho_{{n,\perp}}^{{{title}}}$", fontsize=14)

            ax.set_xlim([-b_max * hbarc, b_max * hbarc])
            ax.set_ylim([-b_max * hbarc, b_max * hbarc])
            
            if col == len(moment_types)-1:
                # print(ax.get_position().x1, ax.get_position().y0,ax.get_position().height)
                # Get positions without labels
                x0, y0, width, height = row_positions_and_heights[row]
                # We shift the x position by the width of the plot
                # such that it attaches to the right
                x0 += width
                cbar_ax = fig.add_axes([x0, y0, 0.01, height])
                fig.colorbar(im, cax=cbar_ax)
            if col == 0:
                ax.text(
                    0.05, 0.95,  
                    rf"$n={j}$",  
                    transform=ax.transAxes, 
                    ha='left', va='top', 
                    fontsize=14, color='black', fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.3')  # Adds a semi-transparent background
                )

            # Remove ticks and labels
            if i != 0:
                ax.set_yticks([])
                ax.set_yticklabels([])
                ax.set_ylabel(None)
            if j != n_max:
                ax.set_xticks([])
                ax.set_xticklabels([])
                ax.set_xlabel(None)

    plt.subplots_adjust(wspace=0, hspace=0)


    # File export
    plt.savefig(FILE_PATH,format="pdf",bbox_inches="tight",dpi=600)

    # Adjust layout and show the plot
    plt.show()
    plt.close()

def plot_fourier_transform_quark_spin_orbit_correlation(eta, mu,  moment_type="non_singlet_isovector",evolution_order="nlo", 
                                          b_max=4.5, Delta_max=10, n_b=50, interpolation = True,n_int=300,
                                          vmin = -1.8 , vmax = 1, ymin = -2, ymax = .3,
                                          plot_option="both",write_to_file = False, read_from_file = True):
    """
    Generate a 2D density plot of the quark spin-orbit correlation in impact parameter space.

    Parameters
    ----------
    eta : float
        Skewness parameter.
    mu : float
        Resolution scale in GeV.
    moment_type : str, optional
        "non_singlet_isovector", "non_singlet_isoscalar", or "singlet".
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo".
    b_max : float, optional
        Maximum value of impact parameter in Gev^-1 used for the plot.
    Delta_max : float, optional
        Upper bound for the momentum magnitude in the integration. Default is 5 GeV.
    n_b : int, optional
        Number of points for discretizing the transverse position plane. Default is 100.
    interpolation : bool, optional
        Whether to interpolate data to plot on a finer grid. Default is True.
    n_int : int, optional
        Number of interpolation points. Default is 300.
    vmin : float, optional
        Minimum value of the colorbar. Default is 0.
    vmax : float, optional
        Maximum value of the colorbar. Default is 1.
    ymin : float, optional
        Minimum value on y-axis used in by = 0 slice of lower plot.
    ymax : float, optional
        Maximum value on y-axis used in by = 0 slice of lower plot.   
    plot_option : str, optional
        Either "lower" (only by = 0 slice) or "both" (Density plot and by = 0 slice)
    write_to_file : bool, optional
        Whether to save the computed data to the file system. Default is False.
    read_from_file : bool, optional
        Whether to load data from file system if available. Default is True.

    Returns
    -------
    None

    Notes
    -----
    This function creates a 2D density plot of the Fourier-transformed quark spin-orbit correlation.
    The plot is saved under
    cfg.PLOT_PATH / f"imp_param_spin_orbit_{moment_type}.pdf" where 
    PLOT_PATH is defined in config.py

    If write_to_file is set to true the data is saved under 
    cfg.IMPACT_PARAMETER_MOMENTS_PATH / f"imp_param_spin_orbit_{mom_type}.csv"
    where IMPACT_PARAMETER_MOMENTS_PATH is defined in config.py

    If both `write_to_file` and `read_from_file` are True, a ValueError is raised.
    """
    def ft_spin_orbit(b_vec,moment_type,error_type):
        return core.fourier_transform_spin_orbit_correlation(eta=eta, mu=mu, b_vec=b_vec,  
                                                        particle=particle,moment_type=moment_type, evolution_order=evolution_order,
                                                        Delta_max=Delta_max, error_type=error_type,dipole_form=True)
    particle = "quark"
    if moment_type not in ["non_singlet_isovector", "non_singlet_isoscalar", "u", "d", "all"]:
        raise ValueError(f"Wrong moment_type {moment_type}")

    if write_to_file and read_from_file:
        raise ValueError("write_to_file and read_from_file can't simultaneously be True")


    if moment_type not in ["non_singlet_isovector", "non_singlet_isoscalar", "u", "d", "all"]:
        raise ValueError(f"Wrong moment_type {moment_type}")
    
    prfx = f"imp_param_spin_orbit_{moment_type}"
    file_name = hp.generate_filename(eta,0,mu,prfx,file_ext="pdf")
    FILE_PATH = cfg.PLOT_PATH / file_name


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
        
    title_map = {
        "non_singlet_isovector": "u-d",
        "non_singlet_isoscalar": "u+d",
        "u": "u",
        "d": "d"
    }
    for i, mom_type in enumerate(moment_types):
        READ_WRITE_PATH = cfg.IMPACT_PARAMETER_MOMENTS_PATH / f"imp_param_spin_orbit_{mom_type}"
        row, col = divmod(i, 4)  # Map index to subplot location
        ax = axs[0, col]
        ax_lower = axs[1, col]

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
                    file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"central")
                    b_x_fm, b_y_fm, fourier_transform_moment_values_flat = hp.read_ft_from_csv(file_name)
                    n_b = len(fourier_transform_moment_values_flat)
                    b_x = np.linspace(-b_max, b_max, n_b)
                    b_y = np.linspace(-b_max, b_max, n_b)
                    # Exctract shape for reshaping
                    b_x_grid, b_y_grid = np.meshgrid(b_x, b_y)
                else:
                    fourier_transform_moment_values_flat = Parallel(n_jobs=-1)(delayed(ft_spin_orbit)(
                        b_vec,moment_type=mom_type,error_type="central") for b_vec in b_vecs)
                    # Reshape
                    fourier_transform_moment_values_flat = np.array(fourier_transform_moment_values_flat).reshape(b_x_grid.shape)
                    # Convert to fm^-2
                    fourier_transform_moment_values_flat = fourier_transform_moment_values_flat/hbarc**2
                cache[mom_type] = fourier_transform_moment_values_flat

            # Generate error bars for lower plot
            if plot_option in ["lower", "both"]:
                if read_from_file:
                    file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"plus")
                    _, _, fourier_transform_moment_values_flat_plus = hp.read_ft_from_csv(file_name)
                    file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"minus")
                    _, _, fourier_transform_moment_values_flat_minus = hp.read_ft_from_csv(file_name)
                else:
                    fourier_transform_moment_values_flat_plus = Parallel(n_jobs=-1)(delayed(ft_spin_orbit)(
                        b_vec,moment_type=mom_type,error_type="plus") for b_vec in b_vecs)
                    fourier_transform_moment_values_flat_minus = Parallel(n_jobs=-1)(delayed(ft_spin_orbit)(
                        b_vec,moment_type=mom_type,error_type="minus") for b_vec in b_vecs)
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
                file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"central")
                b_x_fm, b_y_fm, fourier_transform_moment_values_flat = hp.read_ft_from_csv(file_name)
                file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"plus")
                _, _, fourier_transform_moment_values_flat_plus = hp.read_ft_from_csv(file_name)
                file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"minus")
                _, _, fourier_transform_moment_values_flat_minus = hp.read_ft_from_csv(file_name)
            # Compute from cache
            elif moment_type == "all":
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
            b_x = np.linspace(-b_max, b_max, n_b)
            b_y = np.linspace(-b_max, b_max, n_b)
            b_x_fm = b_x * hbarc
            b_y_fm = b_y * hbarc
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

            if moment_type == "all":
                # Colorbar width is dynamic
                width = 0.01
            else:
                width = 0.05

            # Add colorbar only once per row
            if col == len(moment_types)-1:
                cbar_ax = fig.add_axes([ax.get_position().x1, ax.get_position().y0, width, ax.get_position().height])
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
            file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"central")
            hp.save_ft_to_csv(b_x_fm_write_out,b_y_fm_write_out,ft_write_out,file_name)
            if plot_option in ["lower", "both"]:
                file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"plus")
                hp.save_ft_to_csv(b_x_fm_write_out,b_y_fm_write_out,ft_write_out_plus,file_name)
                file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"minus")
                hp.save_ft_to_csv(b_x_fm_write_out,b_y_fm_write_out,ft_write_out_minus,file_name)
    plt.subplots_adjust(wspace=0, hspace=0)

    # File export
    plt.savefig(FILE_PATH,format="pdf",bbox_inches="tight",dpi=600)

    # Adjust layout and show the plot
    plt.show()
    plt.close()

def plot_fourier_transform_quark_helicity(eta, mu,  moment_type="non_singlet_isovector",evolution_order="nlo", 
                                          b_max=4.5, Delta_max=8, n_b=50, interpolation = True,n_int=300,
                                          vmin = -1.1 , vmax = 2.5, ymin = -0.5, ymax = 2.5,
                                          plot_option="both", read_from_file=True, write_to_file = False):
    """
    Generate a 2D density plot of the quark helicity in impact parameter space.

    Parameters
    ----------
    eta : float
        Skewness parameter.
    mu : float
        Resolution scale in GeV.
    moment_type : str, optional
        "non_singlet_isovector", "non_singlet_isoscalar", or "singlet".
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo".
    b_max : float, optional
        Maximum value of impact parameter in Gev^-1 used for the plot.
    Delta_max : float, optional
        Upper bound for the momentum magnitude in the integration. Default is 5 GeV.
    n_b : int, optional
        Number of points for discretizing the transverse position plane. Default is 100.
    interpolation : bool, optional
        Whether to interpolate data to plot on a finer grid. Default is True.
    n_int : int, optional
        Number of interpolation points. Default is 300.
    vmin : float, optional
        Minimum value of the colorbar. Default is 0.
    vmax : float, optional
        Maximum value of the colorbar. Default is 1.
    ymin : float, optional
        Minimum value on y-axis used in by = 0 slice of lower plot.
    ymax : float, optional
        Maximum value on y-axis used in by = 0 slice of lower plot.   
    plot_option : str, optional
        Either "lower" (only by = 0 slice) or "both" (Density plot and by = 0 slice)
    write_to_file : bool, optional
        Whether to save the computed data to the file system. Default is False.
    read_from_file : bool, optional
        Whether to load data from file system if available. Default is True.

    Returns
    -------
    None

    Notes
    -----
    This function creates a 2D density plot of the Fourier-transformed quark helicity.

    The plot is saved under
    cfg.PLOT_PATH / f"imp_param_helicity_{moment_type}.pdf" where 
    PLOT_PATH is defined in config.py

    If write_to_file is set to true the data is saved under 
    cfg.IMPACT_PARAMETER_MOMENTS_PATH / f"imp_param_helicity_{mom_type}.csv"
    where IMPACT_PARAMETER_MOMENTS_PATH is defined in config.py

    If both `write_to_file` and `read_from_file` are True, a ValueError is raised.
    """ 
    def ft_quark_helicity(b_vec,moment_type,error_type):
        return core.fourier_transform_quark_helicity(eta=eta, mu=mu, b_vec=b_vec,  
                                                moment_type=moment_type, evolution_order=evolution_order,
                                                Delta_max=Delta_max, error_type=error_type,dipole_form=True)
    if write_to_file and read_from_file:
        raise ValueError("write_to_file and read_from_file can't simultaneously be True")

    if moment_type not in ["non_singlet_isovector", "non_singlet_isoscalar", "u", "d", "all"]:
        raise ValueError(f"Wrong moment_type {moment_type}")

    prfx = f"imp_param_helicity_{moment_type}"
    file_name = hp.generate_filename(eta,0,mu,prfx,file_ext="pdf")
    FILE_PATH = cfg.PLOT_PATH / file_name
    

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
        READ_WRITE_PATH = cfg.IMPACT_PARAMETER_MOMENTS_PATH / f"imp_param_helicity_{mom_type}"
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
                    file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"central")
                    b_x_fm, b_y_fm, fourier_transform_moment_values_flat = hp.read_ft_from_csv(file_name)
                    n_b = len(fourier_transform_moment_values_flat)
                    b_x = np.linspace(-b_max, b_max, n_b)
                    b_y = np.linspace(-b_max, b_max, n_b)
                    # Exctract shape for reshaping
                    b_x_grid, b_y_grid = np.meshgrid(b_x, b_y)
                else:
                    fourier_transform_moment_values_flat = Parallel(n_jobs=-1)(delayed(ft_quark_helicity)(
                        b_vec, moment_type = mom_type, error_type = "central") for b_vec in b_vecs)
                    # Reshape
                    fourier_transform_moment_values_flat = np.array(fourier_transform_moment_values_flat).reshape(b_x_grid.shape)
                    # Convert to fm^-2
                    fourier_transform_moment_values_flat = fourier_transform_moment_values_flat/hbarc**2
                # Cache
                cache[mom_type] = fourier_transform_moment_values_flat

                # Generate error bars for lower plot
                if plot_option in ["lower", "both"]:
                    if read_from_file:
                        file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"plus")
                        _, _, fourier_transform_moment_values_flat_plus = hp.read_ft_from_csv(file_name)
                        file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"minus")
                        _, _, fourier_transform_moment_values_flat_minus = hp.read_ft_from_csv(file_name)
                    else:
                        fourier_transform_moment_values_flat_plus =  Parallel(n_jobs=-1)(delayed(ft_quark_helicity)(
                        b_vec, moment_type = mom_type, error_type = "plus") for b_vec in b_vecs)
                        fourier_transform_moment_values_flat_minus =  Parallel(n_jobs=-1)(delayed(ft_quark_helicity)(
                        b_vec, moment_type = mom_type, error_type = "minus") for b_vec in b_vecs)
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
                file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"central")
                b_x_fm, b_y_fm, fourier_transform_moment_values_flat = hp.read_ft_from_csv(file_name)
                file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"plus")
                _, _, fourier_transform_moment_values_flat_plus = hp.read_ft_from_csv(file_name)
                file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"minus")
                _, _, fourier_transform_moment_values_flat_minus = hp.read_ft_from_csv(file_name)
            else:
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
            b_x = np.linspace(-b_max, b_max, n_b)
            b_y = np.linspace(-b_max, b_max, n_b)
            b_x_fm = b_x * hbarc
            b_y_fm = b_y * hbarc
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

            if moment_type == "all":
                # Colorbar width is dynamic
                width = 0.01
            else:
                width = 0.05

            # Add colorbar only once per row
            if col == len(moment_types)-1:
                cbar_ax = fig.add_axes([ax.get_position().x1, ax.get_position().y0, width, ax.get_position().height])
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
            file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"central")
            hp.save_ft_to_csv(b_x_fm_write_out,b_y_fm_write_out,ft_write_out,file_name)
            if plot_option in ["lower", "both"]:
                file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"plus")
                hp.save_ft_to_csv(b_x_fm_write_out,b_y_fm_write_out,ft_write_out_plus,file_name)
                file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"minus")
                hp.save_ft_to_csv(b_x_fm_write_out,b_y_fm_write_out,ft_write_out_minus,file_name)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(FILE_PATH,format="pdf",bbox_inches="tight",dpi=600)

    # Adjust layout and show the plot
    plt.show()
    plt.close()

def plot_fourier_transform_singlet_helicity(eta, mu,  particle = "gluon",evolution_order="nlo",
                                          b_max=4.5, Delta_max=8, n_b=50, interpolation = True,n_int=300,
                                          vmin = -2.05 , vmax = 3.08, ymin= -2.05, ymax = 3.08,
                                          plot_option="both", read_from_file=True, write_to_file = False):
    """
    Generate a 2D density plot of the quark helicity in impact parameter space.

    Parameters
    ----------
    eta : float
        Skewness parameter.
    mu : float
        Resolution scale in GeV.
    moment_type : str, optional
        "non_singlet_isovector", "non_singlet_isoscalar", or "singlet".
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo".
    b_max : float, optional
        Maximum value of impact parameter in Gev^-1 used for the plot.
    Delta_max : float, optional
        Upper bound for the momentum magnitude in the integration. Default is 5 GeV.
    n_b : int, optional
        Number of points for discretizing the transverse position plane. Default is 100.
    interpolation : bool, optional
        Whether to interpolate data to plot on a finer grid. Default is True.
    n_int : int, optional
        Number of interpolation points. Default is 300.
    vmin : float, optional
        Minimum value of the colorbar. Default is 0.
    vmax : float, optional
        Maximum value of the colorbar. Default is 1.
    ymin : float, optional
        Minimum value on y-axis used in by = 0 slice of lower plot.
    ymax : float, optional
        Maximum value on y-axis used in by = 0 slice of lower plot.   
    plot_option : str, optional
        Either "lower" (only by = 0 slice) or "both" (Density plot and by = 0 slice)
    write_to_file : bool, optional
        Whether to save the computed data to the file system. Default is False.
    read_from_file : bool, optional
        Whether to load data from file system if available. Default is True.

    Returns
    -------
    None

    Notes
    -----
    This function creates a 2D density plot of the Fourier-transformed quark helicity.

    The plot is saved under
    cfg.PLOT_PATH / f"imp_param_helicity_singlet_{particle}.pdf" where 
    PLOT_PATH is defined in config.py.

    If write_to_file is set to true the data is saved under 
    cfg.IMPACT_PARAMETER_MOMENTS_PATH / f"imp_param_helicity_singlet_{particle}.csv"
    where IMPACT_PARAMETER_MOMENTS_PATH is defined in config.py

    If both `write_to_file` and `read_from_file` are True, a ValueError is raised.
    """ 
    def ft_singlet_helicity(b_vec,error_type):
        return core.fourier_transform_helicity(eta=eta, mu=mu, b_vec=b_vec, 
                                                      particle=particle,moment_type="singlet", evolution_order=evolution_order,
                                                      Delta_max=Delta_max, error_type=error_type,dipole_form=True)
    if write_to_file and read_from_file:
        raise ValueError("write_to_file and read_from_file can't simultaneously be True")
    
    title_map = {
            "gluon": ("g"),
            "quark": ("sea")
        }
    title = title_map[particle]
    
    prfx = f"imp_param_helicity_singlet_{particle}"
    file_name = hp.generate_filename(eta,0,mu,prfx,file_ext="pdf")
    FILE_PATH = cfg.PLOT_PATH / file_name

    READ_WRITE_PATH = cfg.IMPACT_PARAMETER_MOMENTS_PATH / f"imp_param_helicity_singlet_{particle}"

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
        file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"central")
        b_x_fm, b_y_fm, fourier_transform_moment_values_flat = hp.read_ft_from_csv(file_name)
    else:
        fourier_transform_moment_values_flat = Parallel(n_jobs=-1)(delayed(ft_singlet_helicity)(
                    b_vec, error_type = "central") for b_vec in b_vecs)
        # Reshape
        fourier_transform_moment_values_flat = np.array([z.real for z in fourier_transform_moment_values_flat], dtype=np.float64).reshape(b_x_grid.shape)
        # Convert to fm^-2
        fourier_transform_moment_values_flat = fourier_transform_moment_values_flat/hbarc**2

    # Generate error bars for lower plot
    if plot_option in ["lower", "both"]:
        if read_from_file:
            file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"plus")
            _, _, fourier_transform_moment_values_flat_plus = hp.read_ft_from_csv(file_name)
            file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"minus")
            _, _, fourier_transform_moment_values_flat_minus = hp.read_ft_from_csv(file_name)
        else:
            fourier_transform_moment_values_flat_plus = Parallel(n_jobs=-1)(delayed(ft_singlet_helicity)(
                    b_vec, error_type = "plus") for b_vec in b_vecs)
            fourier_transform_moment_values_flat_minus = Parallel(n_jobs=-1)(delayed(ft_singlet_helicity)(
                    b_vec, error_type = "minus") for b_vec in b_vecs)
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
        ax.set_title(rf'$S_z^{{{title}}}$', fontsize=14)

        # Add colorbar
        cbar_ax = fig.add_axes([ax.get_position().x1, ax.get_position().y0, 0.05, ax.get_position().height])
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
            ax_lower.set_ylabel(rf'$S_z^{{{title}}}$', fontsize=14)

    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(FILE_PATH,format="pdf",bbox_inches="tight",dpi=600)

    if write_to_file:
        file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"central")
        hp.save_ft_to_csv(b_x_fm_write_out,b_y_fm_write_out,ft_write_out,file_name)
        if plot_option in ["lower", "both"]:
            file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"plus")
            hp.save_ft_to_csv(b_x_fm_write_out,b_y_fm_write_out,ft_write_out_plus,file_name)
            file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"minus")
            hp.save_ft_to_csv(b_x_fm_write_out,b_y_fm_write_out,ft_write_out_minus,file_name)

    # Adjust layout and show the plot
    plt.show()
    plt.close()

def plot_fourier_transform_singlet_spin_orbit_correlation(eta, mu,  particle = "gluon",evolution_order="nlo",
                                          b_max=4.5, Delta_max=8, n_b=50, interpolation = True, n_int=300,
                                          vmin = -2.05 , vmax = 3.08, ymin= -2.05, ymax = 3.08,
                                          plot_option="both", read_from_file=True, write_to_file = False):
    """
    Generate a 2D density plot of the singlet spin-orbit correlation in impact parameter space.

    Parameters
    ----------
    eta : float
        Skewness parameter.
    mu : float
        Resolution scale in GeV.
    moment_type : str, optional
        "non_singlet_isovector", "non_singlet_isoscalar", or "singlet".
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo".
    b_max : float, optional
        Maximum value of impact parameter in Gev^-1 used for the plot.
    Delta_max : float, optional
        Upper bound for the momentum magnitude in the integration. Default is 5 GeV.
    n_b : int, optional
        Number of points for discretizing the transverse position plane. Default is 100.
    interpolation : bool, optional
        Whether to interpolate data to plot on a finer grid. Default is True.
    n_int : int, optional
        Number of interpolation points. Default is 300.
    vmin : float, optional
        Minimum value of the colorbar. Default is 0.
    vmax : float, optional
        Maximum value of the colorbar. Default is 1.
    ymin : float, optional
        Minimum value on y-axis used in by = 0 slice of lower plot.
    ymax : float, optional
        Maximum value on y-axis used in by = 0 slice of lower plot.   
    plot_option : str, optional
        Either "lower" (only by = 0 slice) or "both" (Density plot and by = 0 slice)
    write_to_file : bool, optional
        Whether to save the computed data to the file system. Default is False.
    read_from_file : bool, optional
        Whether to load data from file system if available. Default is True.

    Returns
    -------
    None

    Notes
    -----
    This function creates a 2D density plot of the Fourier-transformed quark helicity.

    The plot is saved under
    cfg.PLOT_PATH / f"imp_param_spin_orbit_singlet_{particle}.pdf" where 
    PLOT_PATH is defined in config.py.

    If write_to_file is set to true the data is saved under 
    cfg.IMPACT_PARAMETER_MOMENTS_PATH / f"imp_param_spin_orbit_singlet_{particle}.csv"
    where IMPACT_PARAMETER_MOMENTS_PATH is defined in config.py

    If both `write_to_file` and `read_from_file` are True, a ValueError is raised.
    """  
    def ft_singlet_spin_orbit(b_vec,error_type):
        return core.fourier_transform_spin_orbit_correlation(eta=eta, mu=mu, b_vec=b_vec, 
                                                      particle=particle,moment_type="singlet", evolution_order=evolution_order,
                                                      Delta_max=Delta_max, error_type=error_type,dipole_form=True)
    if write_to_file and read_from_file:
        raise ValueError("write_to_file and read_from_file can't simultaneously be True")
    
    title_map = {
            "gluon": ("g"),
            "quark": ("sea")
        }
    title = title_map[particle]
    
    prfx = f"imp_param_spin_orbit_singlet_{particle}"
    file_name = hp.generate_filename(eta,0,mu,prfx,file_ext="pdf")
    FILE_PATH = cfg.PLOT_PATH / file_name

    READ_WRITE_PATH = cfg.IMPACT_PARAMETER_MOMENTS_PATH / f"imp_param_spin_orbit_singlet_{particle}"

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
        file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"central")
        b_x_fm, b_y_fm, fourier_transform_moment_values_flat = hp.read_ft_from_csv(file_name)
    else:
        fourier_transform_moment_values_flat = Parallel(n_jobs=-1)(delayed(ft_singlet_spin_orbit)(
                    b_vec, error_type = "central") for b_vec in b_vecs)
        # Reshape
        fourier_transform_moment_values_flat = np.array([z.real for z in fourier_transform_moment_values_flat], dtype=np.float64).reshape(b_x_grid.shape)
        # Convert to fm^-2
        fourier_transform_moment_values_flat = fourier_transform_moment_values_flat/hbarc**2

    # Generate error bars for lower plot
    if plot_option in ["lower", "both"]:
        if read_from_file:
            file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"plus")
            _, _, fourier_transform_moment_values_flat_plus = hp.read_ft_from_csv(file_name)
            file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"minus")
            _, _, fourier_transform_moment_values_flat_minus = hp.read_ft_from_csv(file_name)
        else:
            fourier_transform_moment_values_flat_plus = Parallel(n_jobs=-1)(delayed(ft_singlet_spin_orbit)(
                    b_vec, error_type = "plus") for b_vec in b_vecs)
            fourier_transform_moment_values_flat_minus = Parallel(n_jobs=-1)(delayed(ft_singlet_spin_orbit)(
                    b_vec, error_type = "minus") for b_vec in b_vecs)
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
        file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"central")
        hp.save_ft_to_csv(b_x_fm,b_y_fm,fourier_transform_moment_values_flat,file_name)
        if plot_option in ["lower", "both"]:
            file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"plus")
            hp.save_ft_to_csv(b_x_fm,b_y_fm,fourier_transform_moment_values_flat_plus,file_name)
            file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"minus")
            hp.save_ft_to_csv(b_x_fm,b_y_fm,fourier_transform_moment_values_flat_minus,file_name)

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
        cbar_ax = fig.add_axes([ax.get_position().x1, ax.get_position().y0, 0.05, ax.get_position().height])
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


def plot_fourier_transform_quark_orbital_angular_momentum(eta, mu,  moment_type="non_singlet_isovector",evolution_order="nlo", 
                                          b_max=3, Delta_max=7, n_b=50, interpolation = True,n_int=300,
                                          vmin = -2 , vmax = 2, ymin = -2, ymax = .3,
                                          plot_option="both", read_from_file=True, write_to_file = False):
    """
    Generate a 2D density plot of the quark orbital angular momentum in impact parameter space.

    Parameters
    ----------
    eta : float
        Skewness parameter.
    mu : float
        Resolution scale in GeV.
    moment_type : str, optional
        "non_singlet_isovector", "non_singlet_isoscalar", or "singlet".
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo".
    b_max : float, optional
        Maximum value of impact parameter in Gev^-1 used for the plot.
    Delta_max : float, optional
        Upper bound for the momentum magnitude in the integration. Default is 5 GeV.
    n_b : int, optional
        Number of points for discretizing the transverse position plane. Default is 100.
    interpolation : bool, optional
        Whether to interpolate data to plot on a finer grid. Default is True.
    n_int : int, optional
        Number of interpolation points. Default is 300.
    vmin : float, optional
        Minimum value of the colorbar. Default is 0.
    vmax : float, optional
        Maximum value of the colorbar. Default is 1.
    ymin : float, optional
        Minimum value on y-axis used in by = 0 slice of lower plot.
    ymax : float, optional
        Maximum value on y-axis used in by = 0 slice of lower plot.   
    plot_option : str, optional
        Either "lower" (only by = 0 slice) or "both" (Density plot and by = 0 slice)
    write_to_file : bool, optional
        Whether to save the computed data to the file system. Default is False.
    read_from_file : bool, optional
        Whether to load data from file system if available. Default is True.

    Returns
    -------
    None

    Notes
    -----
    This function creates a 2D density plot of the Fourier-transformed quark orbital angular momentum.

    The plot is saved under
    cfg.PLOT_PATH / f"imp_param_oam_{moment_type}.pdf" where 
    PLOT_PATH is defined in config.py.

    If write_to_file is set to true the data is saved under 
    cfg.IMPACT_PARAMETER_MOMENTS_PATH / f"imp_param_oam_{mom_type}.csv" 
    where IMPACT_PARAMETER_MOMENTS_PATH is defined in config.py

    If both `write_to_file` and `read_from_file` are True, a ValueError is raised.
    """     
    def ft_oam(b_vec,moment_type,error_type):
        return core.fourier_transform_quark_orbital_angular_momentum(eta=eta, mu=mu, b_vec=b_vec, 
                                                      moment_type=moment_type, evolution_order=evolution_order,
                                                      Delta_max=Delta_max, error_type=error_type,dipole_form=True)
    if write_to_file and read_from_file:
        raise ValueError("write_to_file and read_from_file can't simultaneously be True")


    if moment_type not in ["non_singlet_isovector", "non_singlet_isoscalar", "u", "d", "all"]:
        raise ValueError(f"Wrong moment_type {moment_type}")

    prfx = f"imp_param_oam_{moment_type}"
    file_name = hp.generate_filename(eta,0,mu,prfx,file_ext="pdf")
    FILE_PATH = cfg.PLOT_PATH / file_name

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
        READ_WRITE_PATH = cfg.IMPACT_PARAMETER_MOMENTS_PATH / f"imp_param_oam_{mom_type}" 
        # Update the grid to data contained in file
        if read_from_file:
            file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"central")
            b_x_fm, b_y_fm, fourier_transform_moment_values_flat = hp.read_ft_from_csv(file_name)


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
                    file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"central")
                    b_x_fm, b_y_fm, fourier_transform_moment_values_flat = hp.read_ft_from_csv(file_name)
                    # Extract shape for reshaping
                    n_b = len(fourier_transform_moment_values_flat)
                    b_x = np.linspace(-b_max, b_max, n_b)
                    b_y = np.linspace(-b_max, b_max, n_b)
                    b_x_grid, b_y_grid = np.meshgrid(b_x, b_y)
                else:
                    fourier_transform_moment_values_flat = Parallel(n_jobs=-1)(delayed(ft_oam)(
                        b_vec,moment_type = mom_type, error_type =  "central") for b_vec in b_vecs)
                    # Reshape
                    fourier_transform_moment_values_flat = np.array(fourier_transform_moment_values_flat).reshape(b_x_grid.shape)
                    # Convert to fm^-2
                    fourier_transform_moment_values_flat = fourier_transform_moment_values_flat/hbarc**2
                # Cache
                cache[mom_type] = fourier_transform_moment_values_flat

                # Generate error bars for lower plot
                if plot_option in ["lower", "both"]:
                    if read_from_file:
                        file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"plus")
                        _, _, fourier_transform_moment_values_flat_plus = hp.read_ft_from_csv(file_name)
                        file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"minus")
                        _, _, fourier_transform_moment_values_flat_minus = hp.read_ft_from_csv(file_name)
                    else:
                        fourier_transform_moment_values_flat_plus = Parallel(n_jobs=-1)(delayed(ft_oam)(
                        b_vec,moment_type = mom_type, error_type =  "plus") for b_vec in b_vecs)
                        fourier_transform_moment_values_flat_minus = Parallel(n_jobs=-1)(delayed(ft_oam)(
                        b_vec,moment_type = mom_type, error_type =  "minus") for b_vec in b_vecs)
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
                file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"central")
                b_x_fm, b_y_fm, fourier_transform_moment_values_flat = hp.read_ft_from_csv(file_name)
                file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"plus")
                _, _, fourier_transform_moment_values_flat_plus = hp.read_ft_from_csv(file_name)
                file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"minus")
                _, _, fourier_transform_moment_values_flat_minus = hp.read_ft_from_csv(file_name)
            else:
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
            b_x = np.linspace(-b_max, b_max, n_b)
            b_y = np.linspace(-b_max, b_max, n_b)
            b_x_fm = b_x * hbarc
            b_y_fm = b_y * hbarc
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

            if moment_type == "all":
                # Colorbar width is dynamic
                width = 0.01
            else:
                width = 0.05

            # Add colorbar only once per row
            if col == len(moment_types)-1:
                cbar_ax = fig.add_axes([ax.get_position().x1, ax.get_position().y0, width, ax.get_position().height])
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
            file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"central")
            hp.save_ft_to_csv(b_x_fm_write_out,b_y_fm_write_out,ft_write_out,file_name)
            if plot_option in ["lower", "both"]:
                file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"plus")
                hp.save_ft_to_csv(b_x_fm_write_out,b_y_fm_write_out,ft_write_out_plus,file_name)
                file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"minus")
                hp.save_ft_to_csv(b_x_fm_write_out,b_y_fm_write_out,ft_write_out_minus,file_name)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(FILE_PATH,format="pdf",bbox_inches="tight",dpi=600)

    # Adjust layout and show the plot
    plt.show()
    plt.close()

def plot_fourier_transform_singlet_orbital_angular_momentum(eta, mu,  particle = "gluon",evolution_order="nlo",
                                          b_max=4.5, Delta_max=8, n_b=50, interpolation = True, n_int=300,
                                          vmin = -2.05 , vmax = 3.08, ymin= -2.05, ymax = 3.08,
                                          plot_option="both", read_from_file=True, write_to_file = False):
    """
    Generate a 2D density plot of the singlet orbital angular momentum in impact parameter space.

    Parameters
    ----------
    eta : float
        Skewness parameter.
    mu : float
        Resolution scale in GeV.
    moment_type : str, optional
        "non_singlet_isovector", "non_singlet_isoscalar", or "singlet".
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo".
    b_max : float, optional
        Maximum value of impact parameter in Gev^-1 used for the plot.
    Delta_max : float, optional
        Upper bound for the momentum magnitude in the integration. Default is 5 GeV.
    n_b : int, optional
        Number of points for discretizing the transverse position plane. Default is 100.
    interpolation : bool, optional
        Whether to interpolate data to plot on a finer grid. Default is True.
    n_int : int, optional
        Number of interpolation points. Default is 300.
    vmin : float, optional
        Minimum value of the colorbar. Default is 0.
    vmax : float, optional
        Maximum value of the colorbar. Default is 1.
    ymin : float, optional
        Minimum value on y-axis used in by = 0 slice of lower plot.
    ymax : float, optional
        Maximum value on y-axis used in by = 0 slice of lower plot.   
    plot_option : str, optional
        Either "lower" (only by = 0 slice) or "both" (Density plot and by = 0 slice)
    write_to_file : bool, optional
        Whether to save the computed data to the file system. Default is False.
    read_from_file : bool, optional
        Whether to load data from file system if available. Default is True.

    Returns
    -------
    None

    Notes
    -----
    This function creates a 2D density plot of the Fourier-transformed singlet orbital angular momentum.

    The plot is saved under
    cfg.PLOT_PATH / f"imp_param_oam_singlet_{particle}.pdf" where 
    PLOT_PATH is defined in config.py.

    If write_to_file is set to true the data is saved under 
    cfg.IMPACT_PARAMETER_MOMENTS_PATH / f"imp_param_oam_singlet_{particle}.csv"
    where IMPACT_PARAMETER_MOMENTS_PATH is defined in config.py

    If both `write_to_file` and `read_from_file` are True, a ValueError is raised.
    """  
    def ft_oam(b_vec,error_type):
            return core.fourier_transform_singlet_orbital_angular_momentum(eta=eta, mu=mu, b_vec=b_vec, 
                                                                    particle=particle, evolution_order=evolution_order,
                                                                    Delta_max=Delta_max, error_type=error_type,dipole_form=True)
    if write_to_file and read_from_file:
        raise ValueError("write_to_file and read_from_file can't simultaneously be True")
    
    title_map = {
            "gluon": ("g"),
            "quark": ("sea")
        }
    title = title_map[particle]

    prfx = f"imp_param_oam_singlet_{particle}"
    file_name = hp.generate_filename(eta,0,mu,prfx,file_ext="pdf")
    FILE_PATH = cfg.PLOT_PATH / file_name
    READ_WRITE_PATH = cfg.IMPACT_PARAMETER_MOMENTS_PATH / f"imp_param_oam_singlet_{particle}"

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
        file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"central")
        b_x_fm, b_y_fm, fourier_transform_moment_values_flat = hp.read_ft_from_csv(file_name)
    else:
        fourier_transform_moment_values_flat = Parallel(n_jobs=-1)(delayed(ft_oam)(
                    b_vec, error_type = "central") for b_vec in b_vecs)
        # Reshape
        fourier_transform_moment_values_flat = np.array([z.real for z in fourier_transform_moment_values_flat], dtype=np.float64).reshape(b_x_grid.shape)
        # Convert to fm^-2
        fourier_transform_moment_values_flat = fourier_transform_moment_values_flat/hbarc**2

    # Generate error bars for lower plot
    if plot_option in ["lower", "both"]:
        if read_from_file:
            file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"plus")
            _, _, fourier_transform_moment_values_flat_plus = hp.read_ft_from_csv(file_name)
            file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"minus")
            _, _, fourier_transform_moment_values_flat_minus = hp.read_ft_from_csv(file_name)
        else:
            fourier_transform_moment_values_flat_plus = Parallel(n_jobs=-1)(delayed(ft_oam)(
                    b_vec, error_type = "plus") for b_vec in b_vecs)
            fourier_transform_moment_values_flat_minus = Parallel(n_jobs=-1)(delayed(ft_oam)(
                    b_vec, error_type = "minus") for b_vec in b_vecs)
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
        file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"central")
        hp.save_ft_to_csv(b_x_fm,b_y_fm,fourier_transform_moment_values_flat,file_name)
        if plot_option in ["lower", "both"]:
            file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"plus")
            hp.save_ft_to_csv(b_x_fm,b_y_fm,fourier_transform_moment_values_flat_plus,file_name)
            file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"minus")
            hp.save_ft_to_csv(b_x_fm,b_y_fm,fourier_transform_moment_values_flat_minus,file_name)

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
        ax.set_title(rf"$L_z^{{{title}}}$", fontsize=14)

        # Add colorbar
        cbar_ax = fig.add_axes([ax.get_position().x1, ax.get_position().y0, 0.05, ax.get_position().height])
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
            ax_lower.set_ylabel(rf'$L_z^{{{title}}}$', fontsize=14)

    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(FILE_PATH,format="pdf",bbox_inches="tight",dpi=600)

    # Adjust layout and show the plot
    plt.show()
    plt.close()


#################
### Plot GPDs ###
#################

def plot_gpd_data(particle="quark", gpd_type="non_singlet_isovector", gpd_label="Htilde", evolution_order="nlo",
                  n_int=300, n_gpd=50, sampling=True, n_init=os.cpu_count(),
                  plot_gpd=True, error_bars=True, write_to_file=False, read_from_file=True,
                  plot_legend=True, y_0=-1e-1, y_1=3):
    """
    Plot the GPD and available lattice data for the specified GPD type and label.

    Currently runs over all data defined in `GPD_PUBLICATION_MAPPING`.

    It is recommended to use plot_gpds to create the data and then use this function with
    read_from_file = True. Though the data generation using this function is also possible.

    Parameters
    ----------
    particle : str, optional
        "quark" or "gluon". Default is "quark".
    gpd_type : str, optional
        Type of GPD, e.g. "non_singlet_isovector", "singlet". Default is "non_singlet_isovector".
    gpd_label : str, optional
        GPD label, e.g. "H", "E", "Htilde". Default is "Htilde".
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    n_int : int, optional
        Number of points used for interpolation. Default is 300.
    n_gpd : int, optional
        Number of GPD data points to generate. Default is 50.
    sampling : bool, optional
        Whether to apply importance sampling in x for the GPD. Default is True.
    n_init : int, optional
        Number of initial sampling points. Default is number of CPU cores.
    plot_gpd : bool, optional
        Whether to plot the GPD curve. Default is True.
    error_bars : bool, optional
        Whether to show error bars for the GPD. Default is True.
    write_to_file : bool, optional
        Write generated GPD data to filesystem. Default is False.
    read_from_file : bool, optional
        Read GPD data from filesystem. Default is True.
    plot_legend : bool, optional
        Display the plot legend. Default is True.
    y_0 : float, optional
        Lower limit of the y-axis. Default is -0.1.
    y_1 : float, optional
        Upper limit of the y-axis. Default is 3.

    Returns
    -------
    None

    Notes
    -----
    The function reads from or writes the GPD data to the file system.

    Data is extracted from publications defined in `GPD_PUBLICATION_MAPPING` in config.py .

    The lattice data is interpolated such that is evaluated at the same x-value as the 
    computed GPD.

    May need to adapt `GPD_LABEL_MAP` and `y_label_map` for custom formatting.

    The plot is saved under 
    cfg.PLOT_PATH / f"{gpd_type}_{particle}_GPD_{gpd_label}_comparison.pdf"
    where PLOT_PATH is defined in config.py

    If write_to_file is True the data is saved as csv using save_gpd_data
    in helpers.py
    """
    def compute_result(x, eta,t,mu,error_type="central"):
        return core.mellin_barnes_gpd(x, eta, t, mu, particle,gpd_type,moment_label, evolution_order, real_imag="real", error_type=error_type,n_jobs=1)
    
    y_label_map = {
            ("non_singlet_isovector","H"): "$H^{{u-d}}(x,\eta,t;\mu)$",
            ("non_singlet_isoscalar","H"): "$H^{{u+d}}(x,\eta,t;\mu)$",
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
    if (gpd_label) in cfg.GPD_LABEL_MAP:
        moment_label = cfg.GPD_LABEL_MAP[gpd_label]
    else:
        print(f"Key {gpd_label} not found in GPD_LABEL_MAP - abort")
        return

    if write_to_file and read_from_file:
        raise ValueError("write_to_file and read_from_file can't simultaneously be True")

    # Initialize plot
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    for (pub_id,gpd_type_data,gpd_label_data,eta,t,mu), (color,_) in cfg.GPD_PUBLICATION_MAPPING.items():
        # Check whether type and label agree with input
        if gpd_type_data != gpd_type or gpd_label_data != gpd_label:
            continue
        gpd_interpolation={} # Initialize dictionary
        for error_type in ["central","plus","minus"]:
            x_values, gpd_values = hp.read_lattice_gpd_data(eta,t,mu,particle,gpd_type,gpd_label,pub_id,error_type)
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
                x_val, results = hp.read_gpd_data(eta,t,mu,particle,gpd_type,gpd_label,evolution_order)
                if x_val is None:
                    print(f"No data for {gpd_type} {gpd_label} at (eta,t,mu) = {eta},{t},{mu} - abort ")
                    return 
                    #raise ValueError("No data found on system. Change write_to_file = True")
            else:
                results = Parallel(n_jobs=-1)(delayed(compute_result)(x,eta,t,mu) for x in x_val)

            # Error bar computations
            if error_bars:
                if read_from_file:
                    x_plus, results_plus = hp.read_gpd_data(eta,t,mu,particle,gpd_type,gpd_label,evolution_order,"plus")
                    x_minus,results_minus = hp.read_gpd_data(eta,t,mu,particle,gpd_type,gpd_label,evolution_order,"minus")
                    if not np.array_equal(x_plus, x_minus) or not np.array_equal(x_plus, x_val):
                        raise ValueError(f"Mismatch in x-values between error data files: {gpd_type} {gpd_label} at (eta,t,mu) = {eta},{t},{mu}")
                else:
                    results_plus = Parallel(n_jobs=-1)(delayed(compute_result)(x,eta,t,mu, error_type="plus") for x in x_val)
                    results_minus = Parallel(n_jobs=-1)(delayed(compute_result)(x,eta,t,mu, error_type="minus") for x in x_val)
            else:
                results_plus = results
                results_minus = results
            end_time_adaptive = time.time()

            # Write to file system
            if write_to_file:
                hp.save_gpd_data(x_val,eta,t,mu,results,particle,gpd_type,gpd_label)
                hp.save_gpd_data(x_val,eta,t,mu,results_plus,particle,gpd_type,gpd_label,"plus")
                hp.save_gpd_data(x_val,eta,t,mu,results_minus,particle,gpd_type,gpd_label,"minus")

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
    FILE_PATH = cfg.PLOT_PATH / f"{gpd_type}_{particle}_GPD_{gpd_label}_comparison.pdf"

    fig.savefig(FILE_PATH,format="pdf",bbox_inches="tight",dpi=600)

def plot_gpds(eta_array, t_array, mu_array, colors, A0=1, particle="quark",
              gpd_type="non_singlet_isovector", gpd_label="H", evolution_order="nlo",
              sampling=True, n_init=os.cpu_count(), n_points=100,
              x_0=-1, x_1=1, y_0=-1e-2, y_1=3,
              error_bars=True, plot_legend=False,
              write_to_file=True, read_from_file=False):
    """
    Plot GPDs using given kinematical parameters and optional error bands.

    Generates plots of a Generalized Parton Distribution (GPD) based on multiple
    kinematical configurations defined by `eta_array`, `t_array`, and `mu_array`.
    The function supports error bands, dynamic x-limits, and reading/writing to disk.

    Parameters
    ----------
    eta_array : array_like
        Array of skewness values (η).
    t_array : array_like
        Array of momentum transfer squared values (t).
    mu_array : array_like
        Array of resolution scales (μ).
    colors : array_like of str
        Colors associated with each GPD curve.
    A0 : float, optional
        Manual rescaling factor for the GPDs. Default is 1.
    particle : str, optional
        "quark" or "gluon". Default is "quark".
    gpd_type : str, optional
        Type of GPD, e.g., "non_singlet_isovector", "singlet". Default is "non_singlet_isovector".
    gpd_label : str, optional
        Label of the GPD (e.g., "H", "E", "Htilde"). Default is "H".
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    sampling : bool, optional
        Whether to apply importance sampling in x. Default is True.
    n_init : int, optional
        Number of initial sampling points. Default is number of available CPU cores.
    n_points : int, optional
        Number of x-values used for plotting each curve. Default is 100.
    x_0 : float, optional
        Lower limit of x-axis. Default is -1.
    x_1 : float, optional
        Upper limit of x-axis. Default is 1.
    y_0 : float, optional
        Lower limit of y-axis. Default is -1e-2.
    y_1 : float, optional
        Upper limit of y-axis. Default is 3.
    error_bars : bool, optional
        Whether to compute and display error bands. Default is True.
    plot_legend : bool, optional
        Whether to include a plot legend. Default is False.
    write_to_file : bool, optional
        Write generated GPD data to filesystem. Default is False.
    read_from_file : bool, optional
        Read GPD data from filesystem. Default is True.

    Returns
    -------
    None

    Notes
    -----
    The lengt of the input arrays must be equal.

    For singlet GPDs, the function automatically applies a lower x cutoff of 1e-2.

    The plot is saved under 
    cfg.PLOT_PATH / f"{gpd_type}_{particle}_GPD_{gpd_label}.pdf"
    where PLOT_PATH is defined in config.py

    If write_to_file is True the data is saved as csv using save_gpd_data
    in helpers.py

    If both `write_to_file` and `read_from_file` are True, a ValueError is raised.
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
    if (gpd_label) in cfg.GPD_LABEL_MAP:
        moment_label = cfg.GPD_LABEL_MAP[gpd_label]
    else:
        print(f"Key {gpd_label} not found in cfg.GPD_LABEL_MAP - abort")
        return
    hp.check_particle_type(particle)
    hp.check_moment_type_label(moment_type,moment_label)

    if write_to_file and read_from_file:
        raise ValueError("write_to_file and read_from_file can't simultaneously be True")

    if len({len(eta_array), len(t_array), len(mu_array), len(colors)}) > 1:
        print("Arrays containing kinematical variables have unequal lengths - abort")
        return

    if moment_type == "singlet":
        x_0 = 1e-2

    def compute_result(x, eta,t,mu):
        return core.mellin_barnes_gpd(x, eta, t, mu, A0,particle,moment_type,moment_label, evolution_order=evolution_order, real_imag="real", error_type="central",n_jobs=1)

    if read_from_file:
        sampling = False

    # Initialize plot
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    for eta, t, mu, color in zip(eta_array,t_array,mu_array,colors):
        if sampling:
            x_values = np.linspace(x_0, x_1, n_init)

            # Measure time for sampling initial points
            start_time_sampling = time.time()
            results = Parallel(n_jobs=n_init)(delayed(compute_result)(x,eta,t,mu) for x in x_values)
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
        if eta not in x_values and eta >= x_0:
            x_values = np.append(x_values,eta)
        if - eta not in x_values and x_0 < 0:
            x_values = np.append(x_values,-eta)
        # Sort
        x_values = np.sort(x_values)
        # Exclude x = 0 as it is unphysical
        x_values = x_values[x_values != 0]
        if read_from_file:
                x_values = None
                x_values, results = hp.read_gpd_data(eta,t,mu,particle,gpd_type,gpd_label,evolution_order)
                if x_values is None:
                    raise ValueError(f"No data found on system for {gpd_type,gpd_label,eta,t,mu}. Change write_to_file = True")
        else:
            # results = Parallel(n_jobs=-1)(delayed(compute_result)(x,eta,t,mu) for x in x_values)
            with hp.tqdm_joblib(tqdm(total=len(x_values))) as progress_bar:
                results = Parallel(n_jobs=-1)(delayed(compute_result)(x,eta,t,mu)
                                              for x in x_values)
            if write_to_file:
                hp.save_gpd_data(x_values,eta,t,mu,results,particle,gpd_type,gpd_label,evolution_order)
        # Error bar computations
        if error_bars:
            if read_from_file:
                x_plus, results_plus = hp.read_gpd_data(eta,t,mu,particle,gpd_type,gpd_label,evolution_order,"plus")
                x_minus,results_minus = hp.read_gpd_data(eta,t,mu,particle,gpd_type,gpd_label,evolution_order,"minus")
            else:
                # Get keys for error types
                key_p = (particle, moment_type, moment_label, evolution_order, "plus")
                key_m = (particle, moment_type, moment_label, evolution_order, "minus")
                # Check whether they exist
                if key_p not in core.gpd_errors or key_m not in core.gpd_errors:
                    print(key_p)
                    print(key_m)
                    raise ValueError(f"No error estimates for {particle,moment_type,moment_label,evolution_order} GPDs have been computed. Modify PARTICLES, MOMENTS,... in config file")
                selected_triples = [
                    (eta_, t_, mu_)
                    for eta_, t_, mu_ in zip(cfg.ETA_ARRAY, cfg.T_ARRAY, cfg.MU_ARRAY)
                ]
                # Get corresponding index for kinematic triple eta, t, mu
                index = selected_triples.index((eta, t, mu))
                # Get error estimate
                # print(type(results))
                # print("---")
                # print(type(core.gpd_errors[key_p][index]),type(core.gpd_errors[key_p][index]))
                gpd_rel_error_p = core.gpd_errors[key_p][index]
                gpd_rel_error_m = core.gpd_errors[key_m][index]
                # Type conversion
                results = np.array(results)
                # Multiply relative error with central value
                results_plus = results * gpd_rel_error_p
                results_minus = results * gpd_rel_error_m
                # Save to csv
                if write_to_file:
                    hp.save_gpd_data(x_values,eta,t,mu,results_plus,particle,gpd_type,gpd_label,evolution_order,"plus")
                    hp.save_gpd_data(x_values,eta,t,mu,results_minus,particle,gpd_type,gpd_label,evolution_order,"minus")
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
    FILE_PATH = cfg.PLOT_PATH / f"{gpd_type}_{particle}_GPD_{gpd_label}.pdf"
    fig.savefig(FILE_PATH,format="pdf",bbox_inches="tight",dpi=600)
    print(f"File saved to {FILE_PATH}")

############################
## Plot Mellin-Barnes     ##
## integral related stuff ##
############################
def plot_conformal_partial_wave(j,eta,particle="quark",parity="none"):
    """
    Plot the conformal partial wave for a given conformal spin j, skewness eta, 
    particle type, and parity.

    Parameters
    ----------
    j : float
        Conformal spin.
    eta : float
        Skewness parameter.
    particle : str, optional
        "quark" or "gluon". Default is "quark".
    parity : str, optional
        Parity of the partial wave: "even", "odd", or "none". Default is "none".

    Returns
    -------
    None

    Notes
    -----
    This function generates a plot of the complex-valued conformal partial wave.
    Intended for use in a Jupyter notebook.
    """
    hp.check_particle_type(particle)
    hp.check_parity(parity)

    x_values = np.linspace(-1, 1, 200)
    y_values = Parallel(n_jobs=-1)(delayed(core.conformal_partial_wave)(j, x, eta , particle, parity) for x in x_values)

    # Separate real and imaginary parts
    y_values_real = [float(y.real) for y in y_values]
    y_values_imag = [float(y.imag) for y in y_values]

    # Create subplots for real and imaginary parts
    plt.figure(figsize=(10, 6))  

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

def plot_evolved_moment_over_j(eta, t, mu, j_base=3, particle="quark",
                                moment_type="non_singlet_isovector", moment_label="A",
                                evolution_order="nlo", error_type="central",
                                j_max=5, num_points=200):
    """
    Plot the real and imaginary parts of the evolved conformal moment as a function of
    complex conformal spin j = j_base + i * k, with -j_max < k < j_max.

    Parameters
    ----------
    eta : float
        Skewness parameter.
    t : float
        Mandelstam t.
    mu : float
        Resolution scale.
    j_base : float, optional
        Real part of integration contour. E.g. as defined in core.get_j_base
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
    j_max : float, optional
        Maximum value of the imaginary part of j (i.e., k). Default is 5.
    num_points : int, optional
        Number of points in the k-range used for plotting. Default is 200.

    Returns
    -------
    None

    Notes
    -----
    Visualizes the analytically continued evolved moment in the complex-j plane.

    Intended for use in a jupyter Notebook. Does not save a plot.
    """

    # Define k values
    k_vals = np.linspace(-j_max, j_max, num_points)
    z_vals = j_base +  1j * k_vals 

    # Evaluate the function for each z
    evolved_moment = np.array(
        Parallel(n_jobs=-1)(delayed(core.evolve_conformal_moment)(z, eta, t, mu, 1,
                                                        particle, moment_type, moment_label, evolution_order, error_type) for z in z_vals),
                dtype=complex)

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

def plot_conformal_partial_wave_over_j(x, eta, particle="quark",
                                       moment_type="non_singlet_isovector",
                                       moment_label="A"):
    """
    Plot the conformal partial wave as a function of complex conformal spin-j for a given x and eta.

    Parameters
    ----------
    x : float
        Parton momentum fraction.
    eta : float
        Skewness parameter.
    particle : str, optional
        "quark" or "gluon". Default is "quark".
    moment_type : str, optional
        "non_singlet_isovector", "non_singlet_isoscalar", or "singlet".
    moment_label : str, optional
        A(tilde), B(tilde) depending on H(tilde) or E(tilde) GPD etc. Default is "A".

    Returns
    -------
    None

    Notes
    -----
    The real part of j as well as the parity of the conformal
    partial wave is obtained from get_j_base.

    This function evaluates the conformal partial wave in the complex-j plane
    and plots its real and imaginary parts for fixed parton x.

    Intended for use in a jupyter Notebook. Does not save a plot.
    """

    hp.check_particle_type(particle)

    j_base, parity = core.get_j_base(particle,moment_type,moment_label)
    k_values = np.linspace(-15, 15, 200)
    j_values = j_base + 1j * k_values
    y_values = np.array(Parallel(n_jobs=-1)(delayed(core.conformal_partial_wave)(j, x, eta , particle, parity) for j in j_values)
                        ,dtype=complex)

    # Create subplots for real and imaginary parts
    plt.figure(figsize=(10, 6))  

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

def plot_conformal_partial_wave_over_x(j_b, eta, particle="quark",
                                       moment_type="non_singlet_isovector",
                                       moment_label="A"):
    """
    Plot the conformal partial wave as a function of parton momentum fraction x 
    for a given conformal spin j_b and skewness eta.

    Parameters
    ----------
    j_b : float
        Conformal spin-j used as base for evaluation.
    eta : float
        Skewness parameter.
    particle : str, optional
        "quark" or "gluon". Default is "quark".
    moment_type : str, optional
        "non_singlet_isovector", "non_singlet_isoscalar", or "singlet".
    moment_label : str, optional
        A(tilde), B(tilde) depending on H(tilde) or E(tilde) GPD etc. Default is "A".

    Returns
    -------
    None

    Notes
    -----
    This function evaluates the conformal partial wave for fixed conformal spin-j as a function of parton x, 
    and plots its real and imaginary components.

    Intended for use in a jupyter Notebook. Does not save a plot.
    """

    hp.check_particle_type(particle)
    
    _, parity = core.get_j_base(particle,moment_type,moment_label)
    x_values = np.linspace(1e-2, .99, 100)
    y_values = np.array(Parallel(n_jobs=-1)(delayed(core.conformal_partial_wave)(j_b, x, eta , particle, parity) for x in x_values)
                        ,dtype=complex)

    # Create subplots for real and imaginary parts
    plt.figure(figsize=(10, 6))  

    #plt.subplot(2, 1, 1)
    plt.plot(x_values, y_values.real)
    plt.xlabel("x")
    plt.ylabel("Real Part")
    plt.title(f"Real Part of Conformal Partial Wave for {particle} with Parity {parity}")

    #plt.subplot(2, 1, 2)
    plt.plot(x_values, y_values.imag)
    plt.xlabel("x")
    plt.ylabel("Imaginary Part")
    plt.title(f"Imaginary Part of Conformal Partial Wave for {particle} with Parity {parity}")

    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()

def plot_mellin_barnes_gpd_integrand(x, eta, t, mu,
                                     particle="quark", moment_type="singlet",
                                     moment_label="A", evolution_order="nlo",
                                     parity="none", error_type="central",
                                     j_max=7.5, n_j=150):
    """
    Plot the real and imaginary parts of the integrand of the Mellin-Barnes integral 
    over complex conformal spin j = j_base + i * k.

    Parameters
    ----------
    x : float
        Parton momentum fraction.
    eta : float
        Skewness parameter.
    t : float
        Mandelstam t.
    mu : float
        Renormalization scale in GeV.
    particle : str, optional
        "quark" or "gluon". Default is "quark".
    moment_type : str, optional
        "non_singlet_isovector", "non_singlet_isoscalar", or "singlet".
    moment_label : str, optional
        A(tilde), B(tilde) depending on H(tilde) or E(tilde) GPD etc. Default is "A".
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    parity : str, optional
        "even", "odd", or "none". Default is "none".
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"
    j_max : float, optional
        Maximum value of Im(j) = k used in the plot. The integration contour is along 
        j = j_base + i * k with -j_max < k < j_max. Default is 7.5.
    n_j : int, optional
        Number of points along Im(j). Default is 150.

    Returns
    -------
    None

    Notes
    -----
    This function evaluates and plots the real and imaginary parts of 
    the Mellin-Barnes integrand used to reconstruct the GPD. 

    Intended for use in a jupyter Notebook. Does not save a plot.
    """
    hp.check_parity(parity)
    hp.check_error_type(error_type)
    hp.check_particle_type(particle)
    hp.check_moment_type_label(moment_type,moment_label)

    j_base, parity_check = core.get_j_base(particle,moment_type,moment_label)
    if parity != parity_check:
        print(f"Warning: Wrong parity of {parity} for moment_type of {moment_type} for particle {particle}")

    def integrand_real(k):
        # Plot imag
        z = j_base + 1j * k
        # Plot real
        #z = k
        dz = 1j
        sin_term = mp.sin(np.pi * z)
        # We double the sine here since its (-1)**(2 * j) from the non-diagonal evolution
        sin2_term = mp.sin(2 * mp.pi * z)/2
        pw_val = core.conformal_partial_wave(z, x, eta, particle, parity)
        if particle == "quark":
            if moment_type == "singlet":
                mom_val = core.evolve_quark_singlet(z, eta, t, mu,1, moment_label, evolution_order, error_type)
            else:
                mom_val = core.evolve_quark_non_singlet(z, eta, t, mu,1, moment_type, moment_label, evolution_order, error_type)
        else:
            mom_val = core.evolve_gluon_singlet(z, eta, t, mu, 1, moment_label, evolution_order, error_type)
        result = -.5j * dz * pw_val * (mom_val[0] / sin_term + mom_val[1] / sin2_term)
        return result.real

    def integrand_imag(k):
        # Plot imag
        z = j_base + 1j * k
        # Plot real
        #z = k
        dz = 1j
        sin_term = mp.sin(mp.pi * z)
        # We double the sine here since its (-1)**(2 * j) from the non-diagonal evolution
        sin2_term = mp.sin(2 * mp.pi * z)/2
        pw_val = core.conformal_partial_wave(z, x, eta, particle, parity)
        if particle == "quark":
            if moment_type == "singlet":
                mom_val = core.evolve_quark_singlet(z, eta, t, mu,1, moment_label, evolution_order, error_type)
            else:
                mom_val = core.evolve_quark_non_singlet(z, eta, t, mu,1, moment_type, moment_label, evolution_order, error_type)
        else:
            mom_val = core.evolve_gluon_singlet(z, eta, t, mu,1,moment_label, evolution_order, error_type)
            mom_val = tuple(-x for x in mom_val)
        result = -.5j * dz * pw_val * (mom_val[0] / sin_term + mom_val[1] / sin2_term)
        return result.imag

    print(f"Integrand at j_max={j_max}")
    print(integrand_real(j_max))
    print(integrand_imag(j_max))
    # Define k range for plotting
    k_values = np.linspace(0, j_max, n_j)
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