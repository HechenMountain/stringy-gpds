import numpy as np
import matplotlib.pyplot as plt
import time
import os

from . import config as cfg
from . import helpers as hp
from . import core
# mpmath precision set in config
from .config import mp

from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.interpolate import RectBivariateSpline, interp1d

############################
####### Plot Moments #######
############################

def plot_moment(n,eta,y_label,mu_in=2,t_max=3,A0=1,particle="quark",moment_type="non_singlet_isovector", moment_label="A",evolution_order="nlo", n_t=50):
    """
    Generates plots of lattice data and RGE-evolved functions for a given moment type and label. Unless there is a different scale
    defined in PUBLICATION_MAPPING, the default is mu = 2 GeV.
    
    Parameters:
    - n (int): conformal spin
    - eta (float): skewness parameter
    - y_label (str.): the label of the y axis
    - mu_in (float, optional): The resolution scale. (default is 2 GeV).
    - t_max (float, optional): Maximum t value for the x-axis (default is 3).
    - A0 (float, optional): Overall scale
    - particle (str., optional): Either quark or gluon
    - moment_type (str): The type of moment (e.g., "non_singlet_isovector").
    - moment_label (str): The label of the moment (e.g., "A").
    - n_t (int, optional): Number of points for T_Fine (default is 50).
    - num_columns (int, optional): Number of columns for the grid layout (default is 3).
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
    Plots conformal moments vs. available lattice data.

    Parameters:
    - eta (float): Skewness parameter
    - y_label (str.): Label on y-axis
    - t_max (float, optional): Maximum value of -t
    - A0 (float, optional): Overall scale, default is 1
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

        # Save each plot as a separate PDF (including publication data)
        pdf_path = cfg.PLOT_PATH / f"{moment_type}_{particle}_{data_moment_label}_n_{n}.pdf"
        
        # Create a new figure to save the current plot as a PDF
        fig_single, ax_single = plt.subplots(figsize=(7, 5))  # New figure for saving each plot
        
        # Plot the RGE functions
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

############################
####   Plots in impact  ####
####   parameter space  ####
############################

def plot_spin_orbit_correlation(eta,mu,particle="quark",evolution_order="nlo",n_t = 50):
    """
    Generates plots of lattice data and spin orbit correlation
    
    Parameters:
    - n_t (int, optional): Number of points for T_Fine (default is 50).
    """
    def compute_result(t_vals,moment_type):
        parallel_results = Parallel(n_jobs=-1)(
            delayed(lambda t: core.spin_orbit_corelation(eta=eta,t=t,mu=mu,
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
    Generates plots of lattice data and orbital angular momentum
    
    Parameters:
    - n_t (int, optional): Number of points for T_Fine (default is 50).
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

def plot_fourier_transform_moments(n,eta,mu,plot_title="",particle="quark",moment_type="non_singlet_isovector", moment_label="A",evolution_order="nlo", b_max = 2,Delta_max = 5,num_points=100,error_type="central"):
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
                                        evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type=error_type,dipole_form=True)
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

def plot_fourier_transform_transverse_moments(n,eta,mu,particle="quark",moment_type="non_singlet_isovector",evolution_order="nlo",
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
    - n_b (int, optional): Number of points the interval [-b_max, b_max] is split into (default is 100).
    - interpolation (bool, optional): Interpolate data points on finer grid
    - n_int (int, optional): Number of points used for interpolation
    - vmin (float ,optioanl): Sets minimum value of colorbar
    - vmax (float, optional): Sets maximum value of colorbar
    - read_from_file (bool): Whether to load data from file system
    - write_to_file (bool): Whether to write data to file system
    """
    def ft_transverse_moment(b_vec):
        return core.fourier_transform_transverse_moment(n=n,eta=eta, mu=mu, b_vec=b_vec,  A0=1,
                                                   particle=particle,moment_type=mom_type, evolution_order=evolution_order, 
                                                   Delta_max=Delta_max, num_points=num_points, error_type="central")
    hp.check_particle_type(particle)

    if moment_type not in ["non_singlet_isovector", "non_singlet_isoscalar", "u", "d", "all","singlet"]:
        raise ValueError(f"Wrong moment_type {moment_type}")

    if write_to_file and read_from_file:
        raise ValueError("write_to_file and read_from_file can't simultaneously be True")

    FILE_PATH = cfg.PLOT_PATH / f"imp_param_transv_pol_moment_j_{n}_{moment_type}.pdf"

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

def plot_fourier_transform_quark_spin_orbit_correlation(eta, mu,  moment_type="non_singlet_isovector",evolution_order="nlo", 
                                          b_max=4.5, Delta_max=10, num_points=100, n_b=50, interpolation = True,n_int=300,
                                          vmin = -1.8 , vmax = 1, ymin = -2, ymax = .3,
                                          plot_option="both",write_to_file = False, read_from_file = True):
    """
    Generates a density plot of the 2D Fourier transform of the quark spin-orbit correlation
    It also includes a 1D slice at b_y = 0.

    Parameters:
    - eta (float): Skewness parameter
    - mu (float): RGE scale
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
    def ft_spin_orbit(b_vec,moment_type,error_type):
        return core.fourier_transform_spin_orbit_correlation(eta=eta, mu=mu, b_vec=b_vec,  
                                                        particle=particle,moment_type=moment_type, evolution_order=evolution_order,
                                                        Delta_max=Delta_max, num_points=num_points, error_type=error_type)
    particle = "quark"
    if moment_type not in ["non_singlet_isovector", "non_singlet_isoscalar", "u", "d", "all"]:
        raise ValueError(f"Wrong moment_type {moment_type}")

    if write_to_file and read_from_file:
        raise ValueError("write_to_file and read_from_file can't simultaneously be True")


    if moment_type not in ["non_singlet_isovector", "non_singlet_isoscalar", "u", "d", "all"]:
        raise ValueError(f"Wrong moment_type {moment_type}")

    FILE_PATH = cfg.PLOT_PATH / f"imp_param_spin_orbit_{moment_type}.pdf"


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
        READ_WRITE_PATH = cfg.IMPACT_PARAMETER_MOMENTS_PATH / f"imp_param_spin_orbit_{mom_type}"
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
                b_x_fm, b_y_fm, _ = hp.read_ft_from_csv(file_name)
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
                                          b_max=4.5, Delta_max=8, num_points=100, n_b=50, interpolation = True,n_int=300,
                                          vmin = -1.1 , vmax = 2.5, ymin = -0.5, ymax = 2.5,
                                          plot_option="both", read_from_file=True, write_to_file = False):
    """
    Generates a density plot of the 2D Fourier transform of RGE-evolved conformal moments.
    It also includes a 1D slice at b_y = 0.

    Parameters:
    - eta (float): Skewness parameter
    - mu (float): RGE scale
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
    def ft_quark_helicity(b_vec,moment_type,error_type):
        return core.fourier_transform_quark_helicity(eta=eta, mu=mu, b_vec=b_vec,  
                                                moment_type=moment_type, evolution_order=evolution_order,
                                                Delta_max=Delta_max, num_points=num_points, error_type=error_type)
    if write_to_file and read_from_file:
        raise ValueError("write_to_file and read_from_file can't simultaneously be True")

    if moment_type not in ["non_singlet_isovector", "non_singlet_isoscalar", "u", "d", "all"]:
        raise ValueError(f"Wrong moment_type {moment_type}")

    FILE_PATH = cfg.PLOT_PATH / f"imp_param_helicity_{moment_type}.pdf"
    

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
                b_x_fm, b_y_fm, _ = hp.read_ft_from_csv(file_name)
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
                                          b_max=4.5, Delta_max=8, num_points=100, n_b=50, interpolation = True,n_int=300,
                                          vmin = -2.05 , vmax = 3.08, ymin= -2.05, ymax = 3.08,
                                          plot_option="both", read_from_file=True, write_to_file = False):
    """
    Generates a density plot of the 2D Fourier transform of RGE-evolved conformal moments.
    It also includes a 1D slice at b_y = 0.

    Parameters:
    - eta (float): Skewness parameter
    - mu (float): RGE scale
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
    def ft_singlet_helicity(b_vec,error_type):
        return core.fourier_transform_quark_gluon_helicity(eta=eta, mu=mu, b_vec=b_vec, 
                                                      particle=particle,moment_type="singlet", evolution_order=evolution_order,
                                                      Delta_max=Delta_max, num_points=num_points, error_type=error_type)
    if write_to_file and read_from_file:
        raise ValueError("write_to_file and read_from_file can't simultaneously be True")
    
    title_map = {
            "gluon": ("g"),
            "quark": ("sea")
        }
    title = title_map[particle]
    
    FILE_PATH = cfg.PLOT_PATH / f"imp_param_helicity_singlet_{particle}.pdf"
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
                                          b_max=4.5, Delta_max=8, num_points=100, n_b=50, interpolation = True, n_int=300,
                                          vmin = -2.05 , vmax = 3.08, ymin= -2.05, ymax = 3.08,
                                          plot_option="both", read_from_file=True, write_to_file = False):
    """
    Generates a density plot of the 2D Fourier transform of RGE-evolved conformal moments.
    It also includes a 1D slice at b_y = 0.

    Parameters:
    - eta (float): Skewness parameter
    - mu (float): RGE scale
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
    def ft_singlet_spin_orbit(b_vec,error_type):
        return core.fourier_transform_spin_orbit_correlation(eta=eta, mu=mu, b_vec=b_vec, 
                                                      particle=particle,moment_type="singlet", evolution_order=evolution_order,
                                                      Delta_max=Delta_max, num_points=num_points, error_type=error_type)
    if write_to_file and read_from_file:
        raise ValueError("write_to_file and read_from_file can't simultaneously be True")
    
    title_map = {
            "gluon": ("g"),
            "quark": ("sea")
        }
    title = title_map[particle]
    
    FILE_PATH = cfg.PLOT_PATH / f"imp_param_spin_orbit_singlet_{particle}.pdf"
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


def plot_fourier_transform_quark_orbital_angular_momentum(eta, mu,  moment_type="non_singlet_isovector",evolution_order="nlo", 
                                          b_max=3, Delta_max=7, num_points=100, n_b=50, interpolation = True,n_int=300,
                                          vmin = -2 , vmax = 2, ymin = -2, ymax = .3,
                                          plot_option="both", read_from_file=True, write_to_file = False):
    """
    Generates a density plot of the 2D Fourier transform of RGE-evolved conformal moments.
    It also includes a 1D slice at b_y = 0.

    Parameters:
    - eta (float): Skewness parameter
    - mu (float): RGE scale
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
    def ft_oam(b_vec,moment_type,error_type):
        return core.fourier_transform_quark_orbital_angular_momentum(eta=eta, mu=mu, b_vec=b_vec, 
                                                      moment_type=moment_type, evolution_order=evolution_order,
                                                      Delta_max=Delta_max, num_points=num_points, error_type=error_type)
    if write_to_file and read_from_file:
        raise ValueError("write_to_file and read_from_file can't simultaneously be True")


    if moment_type not in ["non_singlet_isovector", "non_singlet_isoscalar", "u", "d", "all"]:
        raise ValueError(f"Wrong moment_type {moment_type}")

    FILE_PATH = cfg.PLOT_PATH / f"imp_param_oam_{moment_type}.pdf"

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
                b_x_fm, b_y_fm, _ = hp.read_ft_from_csv(file_name)
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
                                          b_max=4.5, Delta_max=8, num_points=100, n_b=50, interpolation = True, n_int=300,
                                          vmin = -2.05 , vmax = 3.08, ymin= -2.05, ymax = 3.08,
                                          plot_option="both", read_from_file=True, write_to_file = False):
    """
    Generates a density plot of the 2D Fourier transform of RGE-evolved conformal moments.
    It also includes a 1D slice at b_y = 0.

    Parameters:
    - eta (float): Skewness parameter
    - mu (float): RGE scale
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
    def ft_oam(b_vec,error_type):
            return core.fourier_transform_quark_orbital_angular_momentum(eta=eta, mu=mu, b_vec=b_vec, 
                                                                    moment_type="singlet", evolution_order=evolution_order,
                                                                    Delta_max=Delta_max, num_points=num_points, error_type=error_type)
    if write_to_file and read_from_file:
        raise ValueError("write_to_file and read_from_file can't simultaneously be True")
    
    title_map = {
            "gluon": ("g"),
            "quark": ("sea")
        }
    title = title_map[particle]
    FILE_PATH = cfg.PLOT_PATH / f"imp_param_oam_singlet_{particle}.pdf"
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
        ax.set_title(rf"$L_z^{title}$", fontsize=14)

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
            ax_lower.set_ylabel(rf'$L_z^{title}$', fontsize=14)

    fig.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(FILE_PATH,format="pdf",bbox_inches="tight",dpi=600)

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
    hp.check_particle_type(particle)
    hp.check_parity(parity)

    x_values = np.linspace(-1, 1, 200)
    y_values = Parallel(n_jobs=-1)(delayed(core.conformal_partial_wave)(j, x, eta , particle, parity) for x in x_values)

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

#####################
##### Plot GPDs #####
#####################

def plot_gpd_data(particle="quark",gpd_type="non_singlet_isovector",gpd_label="Htilde",evolution_order="nlo",n_int=300,n_gpd=50,sampling=True,n_init=os.cpu_count(), 
                  plot_gpd =True, error_bars=True,write_to_file = False, read_from_file = True, plot_legend = True, y_0 = -1e-1, y_1 =3):
    """
    ADD CORRECT SELECTION OF GPD TYPE: CURRENTLY RUNS OVER ALL DATA

    Plots a GPD of gpd_type and gpd_label over x together with the available data on the file system. Adjust GPD_LABEL_MAP and y_label_map as desired.

    Parameters:
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

def plot_gpds(eta_array, t_array, mu_array, colors,A0=1,  particle="quark",gpd_type="non_singlet_isovector",gpd_label="H",evolution_order="nlo",sampling=True, n_init=os.cpu_count(), n_points=100, x_0=-1, x_1=1, y_0 = -1e-2, y_1 = 3, 
              error_bars=True,plot_legend = False,write_to_file=True,read_from_file=False):
    """
    Generates data for a given GPD using the kinematical parameters contained in eta_array, t_array, mu_array
    and corresponding colors with dynamically adjusted x intervals, including error bars.

    The function supports both positive and negative values of parton x though for singlet it defaults to x > 0.

    Options to read/write from/to file system are included.

    Parameters:
    - eta_array (array float): Array containing skewness values
    - t_array (array float): Array containing  t values
    - mu_array (array float): Array containing mu values
    - colors (array str.): Array containing colors for associated values
    - A0 (float, optional): Manually adjust scale
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

    Note
    ----
    For singlet we cut-off at x0 = 1e-2
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
                if key_p or key_m not in core.gpd_errors:
                    raise ValueError("No error estimates for GPDs have been computed. Modify PARTICLES, MOMENTS,... in config file")
                selected_triples = [
                    (eta_, t_, mu_)
                    for eta_, t_, mu_ in zip(cfg.ETA_ARRAY, cfg.T_ARRAY, cfg.MU_ARRAY)
                ]
                # Get corresponding index for kinematic triple eta, t, mu
                index = selected_triples.index((eta, t, mu))
                # Get error estimate
                gpd_rel_error_p = core.gpd_errors[key_p][index]
                gpd_rel_error_m = core.gpd_errors[key_m][index]
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
    # fig.savefig(FILE_PATH,format="pdf",bbox_inches="tight",dpi=600)
    # print(f"File saved to {FILE_PATH}")

############################
## Plot Mellin-Barnes     ##
## integral related stuff ##
############################

def plot_evolved_moment_over_j(eta,t,mu,Nf = 3,j_base = 3,particle="quark",moment_type="non_singlet_isovector",moment_label ="A",evolution_order="nlo", 
                            error_type = "central", j_max=5, num_points=200):
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
    k_vals = np.linspace(-j_max, j_max, num_points)
    z_vals = j_base +  1j * k_vals 

    # Evaluate the function for each z
    evolved_moment = np.array(
        Parallel(n_jobs=-1)(delayed(core.core.evolve_conformal_moment)(z, eta, t, mu, Nf, 1,
                                                        particle, moment_type, moment_label, evolution_order, error_type) for z in z_vals),
                dtype=complex)
    # evolved_moment = np.array([core.evolve_conformal_moment(z, eta, t, mu, Nf, 
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

def plot_conformal_partial_wave_over_j(x,eta,particle="quark",moment_type="non_singlet_isovector",moment_label ="A"):
    """Plots the conformal parftial wave over conformal spin-j for given eta, particle and parity.

    Parameters:
    - j (float): conformal spin
    - eta (float): skewness
    - particle (str., optiona): quark or gluon. Default is quark
    """
    hp.check_particle_type(particle)

    j_base, parity = core.get_j_base(particle,moment_type,moment_label)
    k_values = np.linspace(-15, 15, 200)
    j_values = j_base + 1j * k_values
    y_values = np.array(Parallel(n_jobs=-1)(delayed(core.conformal_partial_wave)(j, x, eta , particle, parity) for j in j_values)
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

def plot_conformal_partial_wave_over_x(j_b,eta,particle="quark",moment_type="non_singlet_isovector",moment_label ="A"):
    """Plots the conformal parftial wave over conformal spin-j for given eta, particle and parity.

    Parameters:
    - j_b (float): conformal spin
    - eta (float): skewness
    - particle (str., optiona): quark or gluon. Default is quark
    """
    hp.check_particle_type(particle)
    
    _, parity = core.get_j_base(particle,moment_type,moment_label)
    x_values = np.linspace(1e-2, .99, 100)
    y_values = np.array(Parallel(n_jobs=-1)(delayed(core.conformal_partial_wave)(j_b, x, eta , particle, parity) for x in x_values)
                        ,dtype=complex)

    # Create subplots for real and imaginary parts
    plt.figure(figsize=(10, 6))  # Adjust figure size for better visualization

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

def plot_mellin_barnes_gpd_integrand(x, eta, t, mu, Nf=3, particle="quark", moment_type="singlet", moment_label="A",evolution_order="nlo", parity = "none", error_type="central", j_max=7.5,n_j=150):
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

def plot_fourier_transform_transverse_moments_grid(j_max,eta,mu,particle="quark", interpolation=True,n_int=300, vmin = 0 , vmax = 1 ):
    """
    Generates a density plot of the 2D Fourier transfrom of RGE-evolved 
    conformal moments for a given moment type and a transversely polarzied target.
    Automatically uses A and B moments. Code requires all tables to have same b_max
    and be available on the file system. To generate the data use the function
    plot_fourier_transform_transverse_moments.
    
    Parameters:
    - j_max (float): Maximal Conformal spin
    - eta (float): Skewness parameter
    - mu (float): RGE scale
    - particle (str. optional): "quark" or "gluon". Default is quark.
    - moment_label (str. optiona): Label of conformal moment, e.g. A
    - interpolation (bool, optional): Interpolate data points on finer grid
    - n_int (int, optional): Number of points used for interpolation
    - vmin (float ,optioanl): Sets minimum value of colorbar
    - vmax (float, optional): Sets maximum value of colorbar
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

    if len(vmin)<j_max or len(vmax)<j_max:
        raise ValueError("Supply vmin and vmax as arrays of length j_max")

    FILE_PATH = cfg.PLOT_PATH / "imp_param_transv_pol_moments.pdf"

    moment_types = ["non_singlet_isovector", "non_singlet_isoscalar", "u", "d"]

    # Initialize cache to store Fourier transforms for "non_singlet_isovector" and "non_singlet_isoscalar"
    #cache = {}
    cache = {j: {mom_type: None for mom_type in moment_types} for j in range(1, j_max + 1)}

    # Determine figure layout
    fig, axs = plt.subplots(j_max, len(moment_types), figsize=(len(moment_types) * 4, j_max*4))
    row_positions_and_heights = get_subplot_positions_and_heights(j_max,len(moment_types))

    title_map = {
        "non_singlet_isovector": "u-d",
        "non_singlet_isoscalar": "u+d",
        "u": "u",
        "d": "d"
    }
    hbarc = 0.1975

    for j in range(1, j_max + 1):
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
            if j != j_max:
                ax.set_xticks([])
                ax.set_xticklabels([])
                ax.set_xlabel(None)

    plt.subplots_adjust(wspace=0, hspace=0)


    # File export
    plt.savefig(FILE_PATH,format="pdf",bbox_inches="tight",dpi=600)

    # Adjust layout and show the plot
    plt.show()
    plt.close()