import numpy as np
import mpmath as mp
import helpers as hp
import stringy_gpds as sgpds
import config as cfg
import csv
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mstw_pdf import get_alpha_s
from joblib import Parallel, delayed

def dipole_fit_moment(n,eta,mu,Nf=3,particle="quark",moment_type="non_singlet_isovector",moment_label="Atilde",evolution_order="LO",error_type="central",plot_fit=False,write_to_file=True):
    """
    Generates a dipole fit to the first moment of the corresponding singlet GPD.

    Parameters:
    - eta (float): Skewness parameter
    - mu (float): Resolution scale
    - particle (str. optional): quark, gluon
    - gpd_label (str. optional): Atilde,...
    - plot_fit (bool optional): Whether to show a plot
    - write_to_file (bool optional): Write the fit results to a csv table
    - error_type (str. optional): central, plus, minus
    """
    def update_dipole_csv(file_path, particle, moment_type, moment_label, n, evolution_order, A_D, m_D2):
        file_path = cfg.Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        header = ["particle", "moment_type", "moment_label", "n", "evolution_order", "A_D", "m_D2"]
        key = (particle, moment_type, moment_label, str(n), evolution_order)

        rows = []
        found = False

        if file_path.exists():
            with open(file_path, newline='') as f:
                reader = csv.reader(f)
                rows = list(reader)

            # Ensure header exists
            if rows and rows[0] != header:
                raise ValueError("CSV file header does not match expected format.")

            # Check and update existing row
            for i, row in enumerate(rows[1:], start=1):
                if tuple(row[:5]) == key:
                    rows[i] = [particle, moment_type, moment_label, str(n), evolution_order, f"{A_D:.5f}", f"{m_D2:.5f}"]
                    found = True
                    break

        if not found:
            if not rows:
                rows.append(header)
            rows.append([particle, moment_type, moment_label, str(n), evolution_order, f"{A_D:.5f}", f"{m_D2:.5f}"])

        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
    gpd_label = cfg.INVERTED_GPD_LABEL_MAP.get(moment_label)
    if gpd_label is None:
        print(f"Value {moment_label} not found in GPD_LABEL_MAP - abort")
        return
    def dipole_form(t, A_D, m_D2):
        return A_D / (1 - t / m_D2)**2
    
    if moment_type == "singlet" and n == 1:
        # Reconstruct first singlet moment from evolved GPDs
        t_vals, f_vals = sgpds.first_singlet_moment(eta=eta,mu=mu,particle=particle,gpd_label=gpd_label,evolution_order=evolution_order,error_type=error_type)
    else:
        t_vals = np.linspace(-1e-6,-3,16)
        f_vals = Parallel(n_jobs=-1)(
                delayed(lambda t: float(sgpds.evolve_conformal_moment(n, eta, t, mu, Nf, 1,particle=particle, moment_type=moment_type, moment_label=moment_label, evolution_order=evolution_order, error_type=error_type).real))(t)
                for t in t_vals
            )
    # Initial parameter guess: A_D ~ max(f_vals), m_D2 ~ 1.0 
    initial_guess = [np.max(f_vals), 1.0]
    bounds = ([-np.inf, 0], [np.inf, np.inf])

    popt, pcov = curve_fit(dipole_form, t_vals, f_vals, p0=initial_guess,bounds=bounds)
    AD_fit, m_D2_fit = popt

    if plot_fit:
        t_fit = np.linspace(0, -3, 100)
        f_fit = dipole_form(t_fit, *popt)
        # Plot data and fit
        plt.figure(figsize=(8, 5))
        plt.plot(-t_vals, f_vals, 'o', label='Data')
        plt.plot(-t_fit, f_fit, '-')
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
        update_dipole_csv(
            file_path=file_path,
            particle=particle,
            moment_type=moment_type,
            moment_label=moment_label,
            n=n,
            evolution_order=evolution_order,
            A_D=AD_fit,
            m_D2=m_D2_fit
        )
    return AD_fit, m_D2_fit


def quark_singlet_regge_fit(j,eta,t,alpha_prime_ud, alpha_prime_s,norm_A, norm_D,Nf=3,moment_label="A",evolve_type="vector",evolution_order="LO",error_type="central"):
    # Check type
    hp.check_error_type(error_type)
    hp.check_evolve_type(evolve_type)
    hp.check_moment_type_label("singlet",moment_label)
    hp.check_evolution_order(evolution_order)
    if moment_label == "B":
        prf = -1
    else:
        prf = +1

    term_1, error_1 = sgpds.quark_singlet_regge_A(j,eta,t,Nf,alpha_prime_ud,moment_label,evolution_order,error_type)
    term_2, error_2 = sgpds.quark_singlet_regge_D(j,eta,t,Nf,alpha_prime_ud,alpha_prime_s,moment_label,evolution_order,error_type)
    sum_squared = norm_A**2 * error_1**2 + norm_D**2 * error_2**2
    # error = np.frompyfunc(abs, 1, 1)(mp.sqrt(sum_squared))
    error = abs(mp.sqrt(sum_squared))
    result = norm_A * term_1 + norm_D * prf * term_2

    return result, error

def gluon_singlet_regge_fit(j,eta,t,alpha_prime_T, alpha_prime_S,norm_A, norm_D ,moment_label="A",evolve_type="vector", evolution_order="LO",error_type="central"):
    # Check type
    hp.check_error_type(error_type)
    hp.check_evolve_type(evolve_type)
    hp.check_moment_type_label("singlet",moment_label)

    if moment_label == "B":
        prf = -1
    else:
        prf = +1

    term_1, error_1 = sgpds.gluon_singlet_regge_A(j,eta,t,alpha_prime_T,moment_label,evolution_order,error_type)
    if eta == 0:
        result = norm_A * term_1
        error = norm_A * error_1
    else :
        term_2, error_2 = sgpds.gluon_singlet_regge_D(j,eta,t,alpha_prime_T,alpha_prime_S,moment_label,evolution_order,error_type)
        sum_squared = norm_A**2 * error_1**2 + norm_D**2 * error_2**2
        # error = np.frompyfunc(abs, 1, 1)(mp.sqrt(sum_squared))
        error = abs(mp.sqrt(sum_squared))
        result = norm_A * term_1 + norm_D * prf * term_2
    return result, error

def singlet_moment_fit(j,eta,t,alpha_prime_ud, alpha_prime_s,norm_Aq, norm_Dq,alpha_prime_T, alpha_prime_S,norm_Ag, norm_Dg,
                       Nf=3,moment_label="A",evolve_type="vector",solution="+",evolution_order="LO",error_type="central",interpolation=True):
    """
    Returns 0 if the moment_label = "B", in accordance with holography and quark model considerations. 
    Otherwise it returns the diagonal combination of quark + gluon moment. Error for singlet_moment at j = 1
    for solution "-" unreliable because of pole in gamma. Better reconstruct evolved moment from GPD.
    """
    if moment_label == "B":
        return 0, 0
    # Check type
    hp.check_error_type(error_type)
    hp.check_evolve_type(evolve_type)

    # Switch sign
    if solution == "+":
        solution = "-"
    elif solution == "-":
        solution = "+"
    else:
        raise ValueError("Invalid solution type. Use '+' or '-'.")

    quark_prf = .5 
    quark_in, quark_in_error = quark_singlet_regge_fit(j,eta,t,alpha_prime_ud, alpha_prime_s,norm_Aq, norm_Dq,Nf,moment_label,evolve_type,evolution_order,error_type)
    # Note: j/6 already included in gamma_qg and gamma_gg definitions
    gluon_prf = .5 * (sgpds.gamma_qg(j-1,Nf,evolve_type,"LO",interpolation=interpolation)/
                    (sgpds.gamma_qq(j-1,Nf,"singlet",evolve_type,"LO",interpolation=interpolation)-sgpds.gamma_pm(j-1,Nf,evolve_type,solution,interpolation=interpolation)))
    gluon_in, gluon_in_error = gluon_singlet_regge_fit(j,eta,t,alpha_prime_T, alpha_prime_S,norm_Ag, norm_Dg,moment_label,evolve_type,evolution_order,error_type)
    # print(solution,gluon_prf)
    sum_squared = quark_prf**1 * quark_in_error**2 + gluon_prf**2*gluon_in_error**2
    # print("->",quark_in,quark_in_error,quark_prf)
    # print("->",gluon_in,gluon_in_error,gluon_prf)
    # error = np.frompyfunc(abs, 1, 1)(mp.sqrt(sum_squared))
    error = abs(mp.sqrt(sum_squared))
    result = quark_prf * quark_in + gluon_prf * gluon_in
    return result, error

def evolve_singlet_fit(eta,t,mu,alpha_prime_ud, alpha_prime_s,norm_Aq, norm_Dq,alpha_prime_T, alpha_prime_S,norm_Ag, norm_Dg,
                       Nf=3,particle="quark",moment_label ="A", evolution_order = "LO", error_type = "central",interpolation=True):  
    hp.check_particle_type(particle)
    hp.check_moment_type_label("singlet",moment_label)
    hp.check_error_type(error_type)
    hp.check_evolution_order(evolution_order)

    j = 2

    # Set parameters
    Nc = 3
    c_a = Nc
    c_f = (Nc**2-1)/(2*Nc)
    beta_0 = 2/3* Nf - 11/3 * Nc
    # Extract fixed quantities
    alpha_s_in = get_alpha_s(evolution_order)
    alpha_s_evolved = sgpds.evolve_alpha_s(mu,Nf,evolution_order)

    if moment_label in ["A","B"]:
        evolve_type = "vector"
    elif moment_label in ["Atilde","Btilde"]:
        evolve_type = "axial"

    ga_qq = sgpds.gamma_qq(j-1,Nf,"singlet",evolve_type,evolution_order="LO",interpolation=interpolation)

    # Roots  of LO anomalous dimensions
    ga_p = sgpds.gamma_pm(j-1,Nf,evolve_type,"+",interpolation=interpolation)
    ga_m = sgpds.gamma_pm(j-1,Nf,evolve_type,"-",interpolation=interpolation)
    moment_in_p, error_p = singlet_moment_fit(j,eta,t,alpha_prime_ud, alpha_prime_s,norm_Aq, norm_Dq,alpha_prime_T, alpha_prime_S,norm_Ag, norm_Dg,
                                               Nf, moment_label, evolve_type,"+",evolution_order,error_type,interpolation=interpolation)
    moment_in_m, error_m = singlet_moment_fit(j,eta,t,alpha_prime_ud, alpha_prime_s,norm_Aq, norm_Dq,alpha_prime_T, alpha_prime_S,norm_Ag, norm_Dg,
                                               Nf, moment_label, evolve_type,"-",evolution_order,error_type,interpolation=interpolation)
    ga_gq = sgpds.gamma_gq(j-1,Nf, evolve_type,"LO",interpolation=interpolation)
    ga_qg = sgpds.gamma_qg(j-1,Nf, evolve_type,"LO",interpolation=interpolation)
    if evolution_order != "LO":
        ga_gg = sgpds.gamma_gg(j-1,Nf,evolve_type,"LO",interpolation=interpolation)
        r_qq = sgpds.R_qq(j-1,Nf,evolve_type,interpolation=interpolation)
        r_qg = sgpds.R_qg(j-1,Nf,evolve_type,interpolation=interpolation)
        r_gq = sgpds.R_gq(j-1,Nf,evolve_type,interpolation=interpolation)
        r_gg = sgpds.R_gg(j-1,Nf,evolve_type,interpolation=interpolation) 

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
        result = (ga_qq - ga_m)/(ga_p - ga_m) * alpha_frac**(ga_p/beta_0) * 2
        # print(ga_p,ga_m,(ga_qq - ga_m)/(ga_p - ga_m))
        return result
    
    def A_lo_gluon(solution):
        ga_p, ga_m = get_gammas(solution)
        result = ga_gq/(ga_p - ga_m) * alpha_frac**(ga_p/beta_0) * 2
        # print("gluon",solution,result/moment)
        return result

    def A_quark_nlo(solution):
        ga_p, ga_m = get_gammas(solution)
        term1 = - (alpha_s_evolved - alpha_s_in)/(2*mp.pi)/beta_0 * alpha_frac**(ga_p/beta_0) / \
                (ga_p - ga_m)**2 * (2)
        term2 = (ga_qq - ga_m) * (r_qq * (ga_qq-ga_m) + r_qg * ga_gq)
        term3 = ga_qg * (r_gq * (ga_qq - ga_m) + r_gg * ga_gq)
        result = term1 * (term2 + term3)
        return result
    
    def B_quark_nlo(solution):
        ga_p, ga_m = get_gammas(solution)
        term1 = alpha_s_evolved/(2*mp.pi)/(ga_m - ga_p + beta_0) * 2 / (ga_p - ga_m)**2
        term2 = (1 - alpha_frac**((ga_m - ga_p + beta_0)/beta_0)) * alpha_frac**(ga_p/beta_0)
        term3 = ((ga_qq - ga_p) * (r_qq * (ga_qq - ga_m) + r_qg * ga_gq) + ga_qg * (r_gq * (ga_qq - ga_m) + r_gg * ga_gq))
        result = term1 * term2 * term3
        return result

    def A_gluon_nlo(solution):
        ga_p, ga_m = get_gammas(solution)
        term1 = - (alpha_s_evolved - alpha_s_in)/(2*mp.pi)/beta_0 * alpha_frac**(ga_p/beta_0) / \
                (ga_p - ga_m)**2 * (2)
        term2 = ga_gq * (r_qq * (ga_qq-ga_m) + r_qg * ga_gq)
        term3 = (ga_gg - ga_m) * (r_gq * (ga_qq - ga_m) + r_gg * ga_gq)
        result = term1 * (term2 + term3)
        return result

    def B_gluon_nlo(solution):
        ga_p, ga_m = get_gammas(solution)
        term1 = alpha_s_evolved/(2*mp.pi)/(ga_m - ga_p + beta_0) * 2 / (ga_p - ga_m)**2
        term2 = (1 - alpha_frac**((ga_m - ga_p + beta_0)/beta_0)) * alpha_frac**(ga_p/beta_0)
        term3 = (ga_gq  * (r_qq * (ga_qq - ga_m) + r_qg * ga_gq) + (ga_gg - ga_p) * (r_gq * (ga_qq - ga_m) + r_gg * ga_gq) )
        result = term1 * term2 * term3
        return result

    if particle == "quark":
        result = A_lo_quark("+") * moment_in_p + A_lo_quark("-") * moment_in_m
        sum_squared =  (A_lo_quark("+") * error_p)**2 + (A_lo_quark("-") * error_m)**2
        # error = np.frompyfunc(abs, 1, 1)(mp.sqrt(sum_squared))
        error = abs(mp.sqrt(sum_squared))
        result += hp.error_sign(error,error_type)
        if evolution_order == "NLO":
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
        if evolution_order == "NLO":
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
                       Nf=3,particle="quark",moment_label ="A", evolution_order = "LO", error_type = "central",interpolation=True):
    hp.check_particle_type(particle)
    hp.check_moment_type_label("singlet",moment_label)
    term_1 = evolve_singlet_fit(eta,t,mu,alpha_prime_ud, alpha_prime_s,norm_Aq, norm_Dq,alpha_prime_T, alpha_prime_S,norm_Ag, norm_Dg,
                       Nf=Nf,particle=particle,moment_label=moment_label, evolution_order = evolution_order, error_type = error_type,interpolation=interpolation)
    term_2 = evolve_singlet_fit(0,t,mu,alpha_prime_ud, alpha_prime_s,norm_Aq, norm_Dq,alpha_prime_T, alpha_prime_S,norm_Ag, norm_Dg,
                       Nf=Nf,particle=particle,moment_label=moment_label, evolution_order = evolution_order, error_type = error_type,interpolation=interpolation)
    result = (term_1-term_2)/eta**2
    return result


def fit_non_singlet_slopes(evolution_order="LO",error_type="central",plot = True):
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
        uv, err_u = sgpds.integral_uv_pdf_regge(1,0,alpha_p,t,evolution_order,error_type=error_type)
        dv, err_d = sgpds.integral_dv_pdf_regge(1,0,alpha_p,t,evolution_order,error_type=error_type)
        err = hp.error_sign(np.sqrt(err_u**2 + err_d**2),error_type)
        return norm * (uv - dv + err)
    def non_singlet_isovector_vec(t_arr, norm, alpha_p):
        return np.array([
           non_singlet_isovector(t,norm,alpha_p)
            for t in t_arr
        ], dtype=float)
    def non_singlet_isoscalar(t, norm, alpha_p):
        uv, err_u = sgpds.integral_uv_pdf_regge(1,0,alpha_p,t,evolution_order,error_type=error_type)
        dv, err_d = sgpds.integral_dv_pdf_regge(1,0,alpha_p,t,evolution_order,error_type=error_type)
        err = hp.error_sign(np.sqrt(err_u**2 + err_d**2),error_type)
        return norm * (uv + dv + err)
    def non_singlet_isoscalar_vec(t_arr, norm, alpha_p):
        return np.array([
           non_singlet_isoscalar(t,norm,alpha_p)
            for t in t_arr
        ], dtype=float)
    def polarized_non_singlet_isovector(t, norm, alpha_p):
        uv, err_u = sgpds.integral_polarized_uv_pdf_regge(1,0,alpha_p,t,evolution_order,error_type=error_type)
        dv, err_d = sgpds.integral_polarized_dv_pdf_regge(1,0,alpha_p,t,evolution_order,error_type=error_type)
        err = hp.error_sign(np.sqrt(err_u**2 + err_d**2),error_type)
        return norm * (uv - dv + err)
    def polarized_non_singlet_isovector_vec(t_arr, norm, alpha_p):
        return np.array([
           polarized_non_singlet_isovector(t,norm,alpha_p)
            for t in t_arr
        ], dtype=float)
    def polarized_non_singlet_isoscalar(t, norm, alpha_p):
        uv, err_u = sgpds.integral_polarized_uv_pdf_regge(1,0,alpha_p,t,evolution_order,error_type=error_type)
        dv, err_d = sgpds.integral_polarized_dv_pdf_regge(1,0,alpha_p,t,evolution_order,error_type=error_type)
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

    return

def fit_singlet_slopes_A(Nf=3,evolution_order="LO",plot=True):
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
        res, _ = sgpds.gluon_singlet_regge_A(2,0,t,alpha_p,moment_label="A",evolution_order=evolution_order)
        return norm*res
    def quark_singlet_A(t,norm,alpha_p):
        res, _ = sgpds.quark_singlet_regge_A(2,0,t,Nf,alpha_p,moment_label="A",evolution_order=evolution_order)
        return norm*res
    
    # Generate pseudo data
    t_vals = np.linspace(-1e-6,-2,100)

    pseudo_data_gluon_A = gluon_A(t_vals)
    pseudo_data_quark_A = quark_A(t_vals)
    pseudo_data_gluon_D = gluon_D(t_vals)
    pseudo_data_quark_D = quark_D(t_vals)
    
    popt_A_g, pcov_A_g = curve_fit(gluon_singlet_A, t_vals, pseudo_data_gluon_A, p0=[.5,0.6], bounds=([.2,.4],[1.5,1.5]))
    print(f"gluon A: norm = {popt_A_g[0]:.4f}, alpha_p = {popt_A_g[1]:.4f}")
    popt_A_q, pcov_A_q = curve_fit(quark_singlet_A, t_vals, pseudo_data_quark_A, p0=[.5,0.8], bounds=([.2,.4],[1.5,1.5]))
    print(f"quark A: norm = {popt_A_q[0]:.4f}, alpha_p = {popt_A_q[1]:.4f}")

    alpha_p_T = popt_A_g[1]
    alpha_p_ud = popt_A_q[1]
    
    # j = 2 independent of eta
    def gluon_singlet_D(t,norm,alpha_p_S):
        res, _ = sgpds.gluon_singlet_regge_D(2, 1, t, alpha_p_T, alpha_p_S, moment_label="A", evolution_order=evolution_order)
        return norm*res
    def gluon_singlet_D_vec(t_arr, norm, alpha_p_S):
        return np.array([
            gluon_singlet_D(t,norm,alpha_p_S)
            for t in t_arr
        ], dtype=float)
    
    def quark_singlet_D(t,norm,alpha_p_s):
        res, _ = sgpds.quark_singlet_regge_D(2,1,t,Nf,alpha_p_ud,alpha_p_s,moment_label="A",evolution_order=evolution_order)
        return norm*res
    def quark_singlet_D_vec(t_arr, norm, alpha_p_S):
        return np.array([
            quark_singlet_D(t,norm,alpha_p_S)
            for t in t_arr
        ], dtype=float)
    
    # t_2 = np.linspace(-1e-6,-1,10)
    # print(gluon_singlet_D_vec(t_2,1,6))
    # return 

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

    return

def fit_singlet_slopes_Atilde(Nf=3,evolution_order="LO",plot=True):
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
        res, _ = sgpds.quark_singlet_regge_A(2,0,t,Nf,alpha_p,moment_label="Atilde",evolution_order=evolution_order)
        return norm*float(res.real)
    def quark_singlet_A(t_vals,norm,alpha_p):
        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(compute_quark)(t, norm, alpha_p)
            for t in t_vals
        )
        return results
    
    # Generate pseudo data
    t_vals = np.linspace(-1e-6,-2,10)
    pseudo_data_quark_A = quark_A(t_vals)
    popt_A_q, pcov_A_q = curve_fit(quark_singlet_A, t_vals, pseudo_data_quark_A, p0=p0, bounds=bounds)
    print(f"quark A: norm = {popt_A_q[0]:.4f}, alpha_p = {popt_A_q[1]:.4f}")

    # Use less data points for plot
    t_vals = np.linspace(-1e-6,-2,50)
    pseudo_data_quark_A = quark_A(t_vals)

    if plot:
        plt.figure()
        plt.plot(-t_vals, pseudo_data_quark_A, 'o', label='Pseudo-data A_q')
        plt.plot(-t_vals, quark_singlet_A(t_vals, *popt_A_q), '-', label='A_q')
        plt.xlabel("t [GeV²]")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

    return



# Work in progress...
def fit_singlet_slopes_2(Nf=3,evolution_order="LO",plot=True):
    # Dipole form from table III in 2310.08484
    g_A = 0.501
    m_A_g2 = 1.262**2
    g_D = -2.572
    m_D_g2 = 0.538**2

    q_A = 0.510
    m_A_q2 = 1.477**2
    q_D = -1.30
    m_D_q2 = 0.81**2

    if evolution_order == "LO":
        alpha_prime_ud = 0.9426
    elif evolution_order == "NLO":
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
                       Nf=3,particle="quark",moment_label="A", evolution_order = evolution_order, error_type = "central",interpolation=True)
                       for t in t_vals
        ], dtype=np.float64)
        return res
    
    def gluon_singlet_D(t_vals,alpha_prime_s, norm_Dq, alpha_prime_S, norm_Dg):
        res = np.array([evolve_singlet_D_fit(1,t,2,alpha_prime_ud, alpha_prime_s,norm_Aq, norm_Dq,alpha_prime_T, alpha_prime_S,norm_Ag, norm_Dg,
                       Nf=3,particle="gluon",moment_label="A", evolution_order = evolution_order, error_type = "central",interpolation=True)
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

def fit_singlet_D_slopes(norm_Aq,alpha_prime_T,norm_Ag,Nf=3,evolution_order="LO",plot=True):
    # Dipole form from table III in 2310.08484
    g_D = -2.572
    m_D_g2 = 0.538**2

    q_D = -1.30
    m_D_q2 = 0.81**2

    if evolution_order == "LO":
        alpha_prime_ud = 0.9426
    elif evolution_order == "NLO":
        alpha_prime_ud = 0.9492

    def gluon_D(t):
        return g_D/(1-t/m_D_g2)**2
    def quark_D(t):
        return q_D/(1-t/m_D_q2)**2

    # Generate pseudo data
    t_vals = np.linspace(-1e-6,-2,25)

    def quark_singlet_D(t_vals,alpha_prime_s, norm_Dq, alpha_prime_S, norm_Dg):
        res = np.array([evolve_singlet_D_fit(1,t,2,alpha_prime_ud, alpha_prime_s,norm_Aq, norm_Dq,alpha_prime_T, alpha_prime_S,norm_Ag, norm_Dg,
                       Nf=3,particle="quark",moment_label="A", evolution_order = evolution_order, error_type = "central",interpolation=True)
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
                       Nf=3,particle="gluon",moment_label="A", evolution_order = evolution_order, error_type = "central",interpolation=True)
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


    return