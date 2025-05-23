import numpy as np
import config as cfg
import helpers as hp
import matplotlib.pyplot as plt
import stringy_gpds as sgpds

from joblib import Parallel, delayed
from scipy.interpolate import RectBivariateSpline
from config import mp

def plot_evolved_moment_over_j(eta,t,mu,Nf = 3,j_base = 3,particle="quark",moment_type="non_singlet_isovector",moment_label ="A",evolution_order="LO", 
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
        Parallel(n_jobs=-1)(delayed(sgpds.evolve_conformal_moment)(z, eta, t, mu, Nf, 1,
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
    hp.check_particle_type(particle)
    hp.check_parity(parity)

    j_base, parity = sgpds.get_j_base(particle,moment_type,moment_label)
    k_values = np.linspace(-15, 15, 200)
    j_values = j_base + 1j * k_values
    y_values = np.array(Parallel(n_jobs=-1)(delayed(sgpds.conformal_partial_wave)(j, x, eta , particle, parity) for j in j_values)
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

def plot_mellin_barnes_gpd_integrand(x, eta, t, mu, Nf=3, particle="quark", moment_type="singlet", moment_label="A",evolution_order="LO", parity = "none", error_type="central", j_max=7.5,n_j=150):
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

    j_base, parity_check = sgpds.get_j_base(particle,moment_type,moment_label)
    if parity != parity_check:
        print(f"Warning: Wrong parity of {parity} for moment_type of {moment_type} for particle {particle}")

    def integrand_real(k):
        # Plot imag
        z = j_base + 1j * k
        # Plot real
        #z = k
        dz = 1j
        sin_term = mp.sin(mp.pi * z)
        pw_val = sgpds.conformal_partial_wave(z, x, eta, particle, parity)
        if particle == "quark":
            if moment_type == "singlet":
                mom_val = sgpds.evolve_quark_singlet(z, eta, t, mu, Nf,1, moment_label, evolution_order, error_type)
            else:
                mom_val = sgpds.evolve_quark_non_singlet(z, eta, t, mu, Nf,1, moment_type, moment_label, evolution_order, error_type)
        else:
            mom_val = sgpds.evolve_gluon_singlet(z, eta, t, mu, Nf,1, moment_label, evolution_order, error_type)
        result = -0.5j * dz * pw_val * mom_val / sin_term
        return result.real

    def integrand_imag(k):
        # Plot imag
        z = j_base + 1j * k
        # Plot real
        #z = k
        dz = 1j
        sin_term = mp.sin(mp.pi * z)
        pw_val = sgpds.conformal_partial_wave(z, x, eta, particle, parity)
        if particle == "quark":
            if moment_type == "singlet":
                mom_val = sgpds.evolve_quark_singlet(z, eta, t, mu, Nf,1, moment_label, evolution_order, error_type)
            else:
                mom_val = sgpds.evolve_quark_non_singlet(z, eta, t, mu, Nf,1, moment_type, moment_label, evolution_order, error_type)
        else:
            mom_val = (-1) * sgpds.evolve_gluon_singlet(z, eta, t, mu, Nf,1,moment_label, evolution_order, error_type)
        result = -0.5j * dz * pw_val * mom_val / sin_term
        return result.imag

    print(f"Integrand at j_max={j_max}")
    print(integrand_real(j_max))
    print(integrand_imag(j_max))

    # Define k range for plotting
    k_values = np.linspace(-j_max, j_max, n_j)
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
            READ_WRITE_PATH = cfg.IMPACT_PARAMETER_MOMENTS_PATH + "imp_param_transv_pol_moment_j_" + str(j) + "_"  + mom_type 
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