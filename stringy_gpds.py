# Dependencies
import numpy as np
# import mpmath as mp
from config import mp
import matplotlib.pyplot as plt
import time
import os
import csv


from scipy.integrate import quad, odeint, trapezoid, fixed_quad
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy.optimize import curve_fit

from itertools import product

from joblib import Parallel, delayed
from tqdm import tqdm

from mstw_pdf import MSTW_PDF,get_alpha_s
from aac_pdf import AAC_PDF

import config as cfg
import helpers as hp

#############################################
####   Currently enforced assumptions    ####
#############################################
# singlet_moment for B GPD set to zero      #
# Normalizations of isoscalar_moment        #
# ubar = dbar                               #
# Delta_u_bar = Delta_s_bar = Delta_s       #
# Hard bypass for non-diag. evolution since #
# j-k %2 != 0. See (138) in hep-ph/0703179  #
# The magnitude of the nd piece is below    #
# our error estimate. So we discard it      #
#############################################

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
    alpha_s_in = get_alpha_s(evolution_order)
    # print("alpha_in:",alpha_s_in)
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
        - t (float, or array): Mandelstam t (< 0 in physical region)
        """
        frac_1 = epsilon*mp.gamma(eta_1+j-alpha_p*t -.5)/(mp.gamma(eta_1+eta_2+j-alpha_p*t+.5))
        frac_2 = (eta_1+eta_2-gamma_pdf+eta_1*gamma_pdf+j*(1+gamma_pdf)-(1+gamma_pdf)*alpha_p*t)*mp.gamma(eta_1+j-alpha_p*t-1)/mp.gamma(1+eta_1+eta_2+j-alpha_p*t)
        result = A_pdf*mp.gamma(1+eta_2)*(frac_1+frac_2)
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
        hp.check_evolution_order(evolution_order)
        term1 = (
                A_pdf * Delta_A_pdf * mp.gamma(eta_2 + 1) * (
                (
                        (gamma_pol * epsilon * mp.gamma(eta_1 + j - alpha_p * t + alpha + lambda_pol - 0.5))
                        / mp.gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + lambda_pol + 0.5)
                )
                + (
                        (epsilon * mp.gamma(eta_1 + j - alpha_p * t + alpha - 0.5))
                        / mp.gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + 0.5)
                )
                + (
                        (gamma_pol * mp.gamma(eta_1 + j - alpha_p * t + alpha + lambda_pol - 1))
                        * (
                        alpha + lambda_pol + eta_1 * (gamma_pdf + 1)
                        + eta_2 + gamma_pdf * (alpha + lambda_pol + j - alpha_p * t - 1)
                        + j - alpha_p * t
                        )
                        / mp.gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + lambda_pol + 1)
                )
                + (
                        (mp.gamma(eta_1 + j - alpha_p * t + alpha - 1))
                        * (
                        alpha + eta_1 * (gamma_pdf + 1)
                        + eta_2 + gamma_pdf * (alpha + j - alpha_p * t - 1)
                        + j - alpha_p * t
                        )
                        / mp.gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + 1)
                )
                )
        )

        if evolution_order == "LO":
            result = term1
        elif evolution_order == "NLO":
            term2 = - A_pdf * Delta_A_pdf * gamma_pol * mp.gamma(eta_2 + 1)*(
                        (epsilon * mp.gamma(eta_1 + j - alpha_p * t + alpha - 0.5))
                        / mp.gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + 0.5)
                        + (mp.gamma(eta_1 + j - alpha_p * t + alpha - 1)
                        * (
                        alpha + eta_1 * (gamma_pdf + 1)
                        + eta_2 + gamma_pdf * (alpha + j - alpha_p * t - 1)
                        + j - alpha_p * t)
                        )
                        / mp.gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + 1)
            )
            result = term1 + term2
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

        hp.check_error_type(error_type)
        if error_type == "central":
                return 0
        def dpdf_dA_pdf(A_pdf, epsilon, eta_1, eta_2, j, t, alpha_p, gamma_pdf):
                frac_1 = epsilon*mp.gamma(eta_1+j-alpha_p*t -.5)/(mp.gamma(eta_1+eta_2+j-alpha_p*t+.5))
                frac_2 = (eta_1+eta_2-gamma_pdf+eta_1*gamma_pdf+j*(1+gamma_pdf)-(1+gamma_pdf)*alpha_p*t)*mp.gamma(eta_1+j-alpha_p*t-1)/mp.gamma(1+eta_1+eta_2+j-alpha_p*t)
                result = mp.gamma(1+eta_2)*(frac_1+frac_2)
                return result

        def dpdf_deta_1(A_pdf, epsilon, eta_1, eta_2, j, t, alpha_p, gamma_pdf):
                term_1 = (epsilon * mp.gamma(eta_1 + j - t * alpha_p - 0.5) * 
                        mp.digamma(eta_1 + j - t * alpha_p - 0.5) / 
                        mp.gamma(eta_1 + eta_2 + j - t * alpha_p + 0.5))

                term_2 = (epsilon * mp.gamma(eta_1 + j - t * alpha_p - 0.5) * 
                        mp.digamma(eta_1 + eta_2 + j - t * alpha_p + 0.5) / 
                        mp.gamma(eta_1 + eta_2 + j - t * alpha_p + 0.5))

                term_3 = ((gamma_pdf + 1) * mp.gamma(eta_1 + j - t * alpha_p - 1) / 
                        mp.gamma(eta_1 + eta_2 + j - t * alpha_p + 1))

                factor = (eta_1 * (gamma_pdf + 1) + eta_2 + 
                        gamma_pdf * (j - alpha_p * t - 1) + j - alpha_p * t)

                term_4 = (mp.gamma(eta_1 + j - t * alpha_p - 1) * mp.digamma(eta_1 + j - t * alpha_p - 1) * factor /
                        mp.gamma(eta_1 + eta_2 + j - t * alpha_p + 1))

                term_5 = (mp.gamma(eta_1 + j - t * alpha_p - 1) * factor * 
                        mp.digamma(eta_1 + eta_2 + j - t * alpha_p + 1) /
                        mp.gamma(eta_1 + eta_2 + j - t * alpha_p + 1))

                return A_pdf * mp.gamma(eta_2 + 1) * (term_1 - term_2 + term_3 + term_4 - term_5)
        def dpdf_deta_2(A_pdf, epsilon, eta_1, eta_2, j, t, alpha_p, gamma_pdf):
                term_1 = (epsilon * mp.gamma(eta_1 + j - t * alpha_p - 0.5) / 
                        mp.gamma(eta_1 + eta_2 + j - t * alpha_p + 0.5))

                factor = (eta_1 * (gamma_pdf + 1) + eta_2 + 
                        gamma_pdf * (j - alpha_p * t - 1) + j - alpha_p * t)

                term_2 = (mp.gamma(eta_1 + j - t * alpha_p - 1) * factor / 
                        mp.gamma(eta_1 + eta_2 + j - t * alpha_p + 1))

                term_3 = (-epsilon * mp.gamma(eta_1 + j - t * alpha_p - 0.5) * 
                        mp.digamma(eta_1 + eta_2 + j - t * alpha_p + 0.5) / 
                        mp.gamma(eta_1 + eta_2 + j - t * alpha_p + 0.5))

                term_4 = (-mp.gamma(eta_1 + j - t * alpha_p - 1) * factor * 
                        mp.digamma(eta_1 + eta_2 + j - t * alpha_p + 1) /
                        mp.gamma(eta_1 + eta_2 + j - t * alpha_p + 1))

                term_5 = (mp.gamma(eta_1 + j - t * alpha_p - 1) /
                        mp.gamma(eta_1 + eta_2 + j - t * alpha_p + 1))

                return (A_pdf * mp.gamma(eta_2 + 1) * mp.digamma(eta_2 + 1) * (term_1 + term_2) +
                        A_pdf * mp.gamma(eta_2 + 1) * (term_3 + term_4 + term_5))

        def dpdf_depsilon(A_pdf, epsilon, eta_1, eta_2, j, t, alpha_p, gamma_pdf):
                term1 = A_pdf * mp.gamma(eta_2 + 1) * mp.gamma(eta_1 + j - alpha_p * t - 0.5)
                term2 = mp.gamma(eta_1 + eta_2 + j - alpha_p * t + 0.5)
                return term1/term2
        
        def dpdf_dgamma(A_pdf, epsilon, eta_1, eta_2, j, t, alpha_p, gamma_pdf):
                term1 = A_pdf * mp.gamma(eta_2 + 1) * (eta_1 + j - alpha_p * t - 1) * mp.gamma(eta_1 + j - t * alpha_p - 1)
                term2 = mp.gamma(eta_1 + eta_2 + j - t * alpha_p + 1)
                return term1/term2
        
        Delta_A_pdf = dpdf_dA_pdf(A_pdf, epsilon, eta_1, eta_2, j, t, alpha_p, gamma_pdf) * delta_A_pdf
        Delta_eta_1 = dpdf_deta_1(A_pdf, epsilon, eta_1, eta_2, j, t, alpha_p, gamma_pdf) * delta_eta_1
        Delta_eta_2 = dpdf_deta_2(A_pdf, epsilon, eta_1, eta_2, j, t, alpha_p, gamma_pdf) * delta_eta_2
        Delta_epsilon = dpdf_depsilon(A_pdf, epsilon, eta_1, eta_2, j, t, alpha_p, gamma_pdf) * delta_epsilon
        Delta_gamma_pdf = dpdf_dgamma(A_pdf, epsilon, eta_1, eta_2, j, t, alpha_p, gamma_pdf) * delta_gamma_pdf
        # Debug
        #print(Delta_A_pdf,Delta_eta_1,Delta_eta_2,Delta_epsilon,Delta_gamma_pdf)
        # result = abs(mp.sqrt(Delta_A_pdf**2+Delta_eta_1**2+Delta_eta_2**2+Delta_epsilon**2+Delta_gamma_pdf**2))
        sum_squared = Delta_A_pdf**2+Delta_eta_1**2+Delta_eta_2**2+Delta_epsilon**2+Delta_gamma_pdf**2
        result = abs(mp.sqrt(sum_squared))
        return result


def integral_polarized_pdf_regge_error(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
                                       Delta_A_pdf,err_Delta_A_pdf,alpha,err_alpha,gamma_pol,err_gamma_pol, lambda_pol,err_lambda_pol,
                                       j,alpha_p,t,evolution_order="LO", error_type="central"):
        hp.check_evolution_order(evolution_order)
        if error_type == "central":
                return 0
        def dpol_pdf_dDelta_A_pdf(
                        A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
                        Delta_A_pdf,alpha,gamma_pol, lambda_pol,
                        j,alpha_p,t
        ):
                term1 = A_pdf * mp.gamma(eta_2 + 1)
                
                term2 = (gamma_pol * epsilon * mp.gamma(eta_1 + j - alpha_p * t + alpha + lambda_pol - 0.5)) / \
                        mp.gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + lambda_pol + 0.5)
                
                term3 = (epsilon * mp.gamma(eta_1 + j - alpha_p * t + alpha - 0.5)) / \
                        mp.gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + 0.5)
                
                term4 = (mp.gamma(eta_1 + j - alpha_p * t + alpha + lambda_pol - 1) * 
                        (gamma_pol + (gamma_pol * gamma_pdf * 
                        (alpha + lambda_pol + eta_1 + j - alpha_p * t - 1)) / 
                        (alpha + lambda_pol + eta_1 + eta_2 + j - alpha_p * t))) / \
                        mp.gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + lambda_pol)
                
                term5 = (mp.gamma(eta_1 + j - alpha_p * t + alpha - 1) * 
                        ((gamma_pdf * (alpha + eta_1 + j - alpha_p * t - 1)) / 
                        (alpha + eta_1 + eta_2 + j - alpha_p * t) + 1)) / \
                        mp.gamma(eta_1 + eta_2 + j - alpha_p * t + alpha)
                if evolution_order == "LO":
                    term6 = 0
                elif evolution_order == "NLO":
                    term6 = - gamma_pol * (
                        term3 +
                        mp.gamma(eta_1 + j - alpha_p * t + alpha - 1)/mp.gamma(1+ eta_1 + eta_2 + j - alpha_p * t + alpha) * \
                        ((eta_2+eta_1*(1+gamma_pdf)+j -alpha_p * t + alpha + gamma_pdf * (-1+j-alpha_p *t + alpha)))
                    )
                
                return term1 * (term2 + term3 + term4 + term5 + term6)
        def dpol_pdf_dalpha(
                        A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
                        Delta_A_pdf,alpha,gamma_pol, lambda_pol,
                        j,alpha_p,t
        ):
                term1 = A_pdf * Delta_A_pdf * mp.gamma(eta_2 + 1)
    
                term2 = (gamma_pol * epsilon * mp.gamma(eta_1 + j - alpha_p * t + alpha + lambda_pol - 0.5) 
                        * mp.digamma(eta_1 + j - alpha_p * t + alpha + lambda_pol - 0.5)) \
                        / mp.gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + lambda_pol + 0.5)
                
                term3 = - (gamma_pol * epsilon * mp.gamma(eta_1 + j - alpha_p * t + alpha + lambda_pol - 0.5) 
                                * mp.digamma(eta_1 + eta_2 + j - alpha_p * t + alpha + lambda_pol + 0.5)) \
                        / mp.gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + lambda_pol + 0.5)
                
                term4 = (epsilon * mp.gamma(eta_1 + j - alpha_p * t + alpha - 0.5) 
                        * mp.digamma(eta_1 + j - alpha_p * t + alpha - 0.5)) \
                        / mp.gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + 0.5)
                
                term5 = - (epsilon * mp.gamma(eta_1 + j - alpha_p * t + alpha - 0.5) 
                        * mp.digamma(eta_1 + eta_2 + j - alpha_p * t + alpha + 0.5)) \
                        / mp.gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + 0.5)
                
                term6 = (gamma_pol * (eta_2 + 1) * gamma_pdf * mp.gamma(eta_1 + j - alpha_p * t + alpha + lambda_pol - 1)) \
                        / ((alpha + lambda_pol + eta_1 + eta_2 + j - alpha_p * t)**2 
                        * mp.gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + lambda_pol))
                
                term7 = (mp.gamma(eta_1 + j - alpha_p * t + alpha + lambda_pol - 1) 
                        * mp.digamma(eta_1 + j - alpha_p * t + alpha + lambda_pol - 1)
                        * (gamma_pol + (gamma_pol * gamma_pdf * (alpha + lambda_pol + eta_1 + j - alpha_p * t - 1))
                                / (alpha + lambda_pol + eta_1 + eta_2 + j - alpha_p * t))) \
                        / mp.gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + lambda_pol)
                
                term8 = - (mp.gamma(eta_1 + j - alpha_p * t + alpha + lambda_pol - 1) 
                                * mp.digamma(eta_1 + eta_2 + j - alpha_p * t + alpha + lambda_pol)
                                * (gamma_pol + (gamma_pol * gamma_pdf * (alpha + lambda_pol + eta_1 + j - alpha_p * t - 1))
                                / (alpha + lambda_pol + eta_1 + eta_2 + j - alpha_p * t))) \
                        / mp.gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + lambda_pol)
                
                term9 = ((eta_2 + 1) * gamma_pdf * mp.gamma(eta_1 + j - alpha_p * t + alpha - 1)) \
                        / ((alpha + eta_1 + eta_2 + j - alpha_p * t)**2 * mp.gamma(eta_1 + eta_2 + j - alpha_p * t + alpha))
                
                term10 = (mp.gamma(eta_1 + j - alpha_p * t + alpha - 1) * mp.digamma(eta_1 + j - alpha_p * t + alpha - 1)
                        * (1 + (gamma_pdf * (alpha + eta_1 + j - alpha_p * t - 1))
                                / (alpha + eta_1 + eta_2 + j - alpha_p * t))) \
                        / mp.gamma(eta_1 + eta_2 + j - alpha_p * t + alpha)
                
                term11 = - (mp.gamma(eta_1 + j - alpha_p * t + alpha - 1) * mp.digamma(eta_1 + eta_2 + j - alpha_p * t + alpha)
                                * (1 + (gamma_pdf * (alpha + eta_1 + j - alpha_p * t - 1))
                                / (alpha + eta_1 + eta_2 + j - alpha_p * t))) \
                        / mp.gamma(eta_1 + eta_2 + j - alpha_p * t + alpha)
                if evolution_order == "LO":
                    term12 = 0
                elif evolution_order == "NLO":
                    term12 = gamma_pol * (- (term4 + term5) + mp.gamma(eta_1 + j - alpha_p * t + alpha - 1) *(
                         -(eta_2 + 1) * gamma_pdf /(eta_1 + eta_2 + j - alpha_p * t + alpha) - 
                         (eta_2 + eta_1 * (1 + gamma_pdf) + j - alpha_p * t + alpha + gamma_pdf * (j - alpha_p * t + alpha -1)) * \
                         (mp.digamma(eta_1 + j - alpha_p * t + alpha - 1)- mp.digamma(eta_1 + eta_2 + j - alpha_p * t + alpha))
                    )/mp.gamma(1 + eta_1 +eta_2 + j - alpha_p * t + alpha))

                return term1 * (term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10 + term11 + term12)
        
        
        def dpol_pdf_dgamma_pol(
                        A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
                        Delta_A_pdf,alpha,gamma_pol, lambda_pol,
                        j,alpha_p,t
        ):
                term1 = A_pdf * Delta_A_pdf * mp.gamma(eta_2 + 1)
    
                term2 = (epsilon * mp.gamma(eta_1 + j - alpha_p * t + alpha + lambda_pol - 0.5)) \
                        / mp.gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + lambda_pol + 0.5)
                
                term3 = (mp.gamma(eta_1 + j - alpha_p * t + alpha + lambda_pol - 1) * 
                        (- alpha_p * t  + alpha + lambda_pol + eta_1 * (gamma_pdf + 1) + eta_2 +
                        gamma_pdf * (- alpha_p * t  + alpha + lambda_pol + j - 1) + j)) \
                        / mp.gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + lambda_pol + 1)
                
                if evolution_order == "LO":
                    term4 = 0
                elif evolution_order == "NLO":
                    term4 = - ((epsilon * mp.gamma(eta_1 + j - alpha_p * t + alpha - 0.5)) \
                        / mp.gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + 0.5)
                        + (eta_2 + eta_1*(1+gamma_pdf) +  j - alpha_p * t + alpha
                        + gamma_pdf * ( j - alpha_p * t + alpha - 1)) * mp.gamma(eta_1 +  j - alpha_p * t + alpha - 1)/\
                        mp.gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + 1)
                    )
                return term1 * (term2 + term3 + term4)
        
        def dpol_pdf_dlambda_pol(
                        A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
                        Delta_A_pdf,alpha,gamma_pol, lambda_pol,
                        j,alpha_p,t
        ):
                # Same for LO and NLO
                term1 = A_pdf * Delta_A_pdf * mp.gamma(eta_2 + 1)
                
                term2 = (gamma_pol * epsilon * mp.gamma(eta_1 + j - alpha_p * t + alpha + lambda_pol - 0.5) * 
                        mp.digamma(eta_1 + j - alpha_p * t + alpha + lambda_pol - 0.5)) / \
                        mp.gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + lambda_pol + 0.5)
                
                term3 = -(gamma_pol * epsilon * mp.gamma(eta_1 + j - alpha_p * t + alpha + lambda_pol - 0.5) * 
                        mp.digamma(eta_1 + eta_2 + j - alpha_p * t + alpha + lambda_pol + 0.5)) / \
                        mp.gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + lambda_pol + 0.5)
                
                term4 = (gamma_pol * (eta_2 + 1) * gamma_pdf * mp.gamma(eta_1 + j - alpha_p * t + alpha + lambda_pol - 1)) / \
                        ((alpha + lambda_pol + eta_1 + eta_2 + j - alpha_p * t)**2 * 
                        mp.gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + lambda_pol))
                
                term5 = (mp.gamma(eta_1 + j - alpha_p * t + alpha + lambda_pol - 1) * 
                        mp.digamma(eta_1 + j - alpha_p * t + alpha + lambda_pol - 1) * 
                        (gamma_pol + (gamma_pol * gamma_pdf * 
                        (alpha + lambda_pol + eta_1 + j - alpha_p * t - 1)) / 
                        (alpha + lambda_pol + eta_1 + eta_2 + j - alpha_p * t))) / \
                        mp.gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + lambda_pol)
                
                term6 = -(mp.gamma(eta_1 + j - alpha_p * t + alpha + lambda_pol - 1) * 
                        mp.digamma(eta_1 + eta_2 + j - alpha_p * t + alpha + lambda_pol) * 
                        (gamma_pol + (gamma_pol * gamma_pdf * 
                        (alpha + lambda_pol + eta_1 + j - alpha_p * t - 1)) / 
                        (alpha + lambda_pol + eta_1 + eta_2 + j - alpha_p * t))) / \
                        mp.gamma(eta_1 + eta_2 + j - alpha_p * t + alpha + lambda_pol)
                                
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
        # print(Delta_Delta_A_pdf,Delta_alpha,Delta_gamma_pol,Delta_lambda_pol)
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

        # result = abs(mp.sqrt(Delta_Delta_A_pdf**2+Delta_alpha**2+Delta_gamma_pol**2+Delta_lambda_pol**2))
        sum_squared = Delta_Delta_A_pdf**2+Delta_alpha**2+Delta_gamma_pol**2+Delta_lambda_pol**2
        result = abs(mp.sqrt(sum_squared))
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
    hp.check_error_type(error_type)

     # Define a dictionary that maps the error_type to column indices
    error_mapping = {
        "central": 0,  # The column with the central value
        "plus": 1,     # The column with the + error value
        "minus": 2     # The column with the - error value
    }
    
    # Get the column index corresponding to the error_type
    error_col_index = error_mapping.get(error_type) 

    # Extracting parameter values
    A_pdf     = MSTW_PDF["A_u"][evolution_order][0]
    eta_1     = MSTW_PDF["eta_1"][evolution_order][0]
    eta_2     = MSTW_PDF["eta_2"][evolution_order][0]
    epsilon   = MSTW_PDF["epsilon_u"][evolution_order][0]
    gamma_pdf = MSTW_PDF["gamma_u"][evolution_order][0]

    pdf = integral_pdf_regge(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,j,alpha_p,t)

    if error_type != "central":
    # Extracting errors
        delta_A_pdf  = MSTW_PDF["A_u"][evolution_order][error_col_index]
        delta_eta_1 = MSTW_PDF["eta_1"][evolution_order][error_col_index]
        delta_eta_2 = MSTW_PDF["eta_2"][evolution_order][error_col_index]
        delta_epsilon = MSTW_PDF["epsilon_u"][evolution_order][error_col_index]
        delta_gamma_pdf = MSTW_PDF["gamma_u"][evolution_order][error_col_index]

        
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
    hp.check_error_type(error_type)

     # Define a dictionary that maps the error_type to column indices
    error_mapping = {
        "central": 0,  # The column with the central value
        "plus": 1,     # The column with the + error value
        "minus": 2     # The column with the - error value
    }
    
    # Get the column index corresponding to the error_type
    error_col_index = error_mapping.get(error_type)

    A_pdf     = MSTW_PDF["A_d"][evolution_order][0]
    eta_1     = MSTW_PDF["eta_3"][evolution_order][0]
    eta_2     = MSTW_PDF["eta_2"][evolution_order][0] + MSTW_PDF["eta_4-eta_2"][evolution_order][0]  # eta_4 ≡ eta_2 + (eta_4 - eta_2)
    epsilon   = MSTW_PDF["epsilon_d"][evolution_order][0]
    gamma_pdf = MSTW_PDF["gamma_d"][evolution_order][0]

    pdf = integral_pdf_regge(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,j,alpha_p,t)

    if error_type != "central":
        # Extracting errors
        delta_A_pdf  = MSTW_PDF["A_d"][evolution_order][error_col_index]
        delta_eta_1 = MSTW_PDF["eta_3"][evolution_order][error_col_index]
        delta_eta_2 = np.sign(MSTW_PDF["eta_4-eta_2"][evolution_order][error_col_index]) * np.sqrt(MSTW_PDF["eta_4-eta_2"][evolution_order][error_col_index]**2 + MSTW_PDF["eta_2"][evolution_order][error_col_index]**2)
        delta_epsilon = MSTW_PDF["epsilon_d"][evolution_order][error_col_index]
        delta_gamma_pdf = MSTW_PDF["gamma_d"][evolution_order][error_col_index]


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
        frac = mp.gamma(1+eta_m)*mp.gamma(j+delta_m-1-alpha_p*t)/(x_0*mp.gamma(1+delta_m+eta_m+j-alpha_p*t))
        result = -A_m*(j-1-delta_m*(x_0-1)-x_0*(eta_m+j-alpha_p*t)-alpha_p*t)*frac
        return result
    def integral_sv_pdf_regge_error(A_m,delta_A_m,delta_m,delta_delta_m,eta_m,delta_eta_m,x_0,delta_x_0,j,alpha_p,t, error_type="central"):
        if error_type == "central":
            return 0
        def dpdf_dA_m(A_m, delta_m, eta_m,x_0, j, alpha_p,t):
            result = (
                    mp.gamma(eta_m + 1) * mp.gamma(delta_m + j - alpha_p * t - 1) *
                    ((delta_m + j - alpha_p * t - 1) / (delta_m + eta_m + j - alpha_p * t) - x_0)
                ) / (x_0 * mp.gamma(delta_m + eta_m + j - alpha_p * t))
        
            return result
        def dpdf_ddelta_m(A_m,delta_m,eta_m,x_0,j,alpha_p,t):
            term_1 = (
                A_m * mp.gamma(eta_m + 1) * 
                ((1 / (delta_m + eta_m + j - alpha_p * t)) -
                ((delta_m + j - alpha_p * t - 1) / (delta_m + eta_m + j - alpha_p * t) ** 2)) *
                mp.gamma(delta_m + j - alpha_p * t - 1)
            ) / (x_0 * mp.gamma(delta_m + eta_m + j - alpha_p * t))
            
            term_2 = (
                A_m * mp.gamma(eta_m + 1) * mp.gamma(delta_m + j - alpha_p * t - 1) *
                mp.digamma(delta_m + j - alpha_p * t - 1) *
                ((delta_m + j - alpha_p * t - 1) / (delta_m + eta_m + j - alpha_p * t) - x_0)
            ) / (x_0 * mp.gamma(delta_m + eta_m + j - alpha_p * t))
            
            term_3 = (
                A_m * mp.gamma(eta_m + 1) * mp.gamma(delta_m + j - alpha_p * t - 1) *
                mp.digamma(delta_m + eta_m + j - alpha_p * t) *
                ((delta_m + j - alpha_p * t - 1) / (delta_m + eta_m + j - alpha_p * t) - x_0)
            ) / (x_0 * mp.gamma(delta_m + eta_m + j - alpha_p * t))
            
            return term_1 + term_2 - term_3
        def dpdf_deta_m(A_m,delta_m,eta_m,x_0,j,alpha_p,t):
            term_1 = -(
                A_m * mp.gamma(eta_m + 1) * (delta_m + j - alpha_p * t - 1) * mp.gamma(delta_m + j - alpha_p * t - 1)
            ) / (x_0 * (delta_m + eta_m + j - alpha_p * t) ** 2 * mp.gamma(delta_m + eta_m + j - alpha_p * t))
            
            term_2 = (
                A_m * mp.gamma(eta_m + 1) * mp.digamma(eta_m + 1) * mp.gamma(delta_m + j - alpha_p * t - 1) *
                ((delta_m + j - alpha_p * t - 1) / (delta_m + eta_m + j - alpha_p * t) - x_0)
            ) / (x_0 * mp.gamma(delta_m + eta_m + j - alpha_p * t))
            
            term_3 = (
                A_m * mp.gamma(eta_m + 1) * mp.gamma(delta_m + j - alpha_p * t - 1) *
                mp.digamma(delta_m + eta_m + j - alpha_p * t) *
                ((delta_m + j - alpha_p * t - 1) / (delta_m + eta_m + j - alpha_p * t) - x_0)
            ) / (x_0 * mp.gamma(delta_m + eta_m + j - alpha_p * t))
            
            return term_1 + term_2 - term_3
        
        def dpdf_dx_0(A_m,delta_m,eta_m,x_0,j,alpha_p,t):
            term_1 = -(
                A_m * mp.gamma(eta_m + 1) * mp.gamma(delta_m + j - alpha_p * t - 1) *
                ((delta_m + j - alpha_p * t - 1) / (delta_m + eta_m + j - alpha_p * t) - x_0)
            ) / (x_0 ** 2 * mp.gamma(delta_m + eta_m + j - alpha_p * t))
            
            term_2 = -(
                A_m * mp.gamma(eta_m + 1) * mp.gamma(delta_m + j - alpha_p * t - 1)
            ) / (x_0 * mp.gamma(delta_m + eta_m + j - alpha_p * t))
            
            return term_1 + term_2
        
        Delta_A_m = dpdf_dA_m(A_m,delta_m,eta_m,x_0,j,alpha_p,t) * delta_A_m
        Delta_delta_m = dpdf_ddelta_m(A_m,delta_m,eta_m,x_0,j,alpha_p,t) * delta_delta_m
        Delta_eta_m = dpdf_deta_m(A_m,delta_m,eta_m,x_0,j,alpha_p,t) * delta_eta_m
        Delta_x_0= dpdf_dx_0(A_m,delta_m,eta_m,x_0,j,alpha_p,t) * delta_x_0

        # Debug
        # print(dpdf_dA_m(A_m,delta_m,eta_m,x_0,j,alpha_p,t),dpdf_ddelta_m(A_m,delta_m,eta_m,x_0,j,alpha_p,t),dpdf_deta_m(A_m,delta_m,eta_m,x_0,j,alpha_p,t),dpdf_dx_0(A_m,delta_m,eta_m,x_0,j,alpha_p,t))
        # print(Delta_A_m,Delta_delta_m,Delta_eta_m,Delta_x_0)

        result = abs(mp.sqrt(Delta_A_m**2+Delta_delta_m**2+Delta_eta_m**2+Delta_x_0**2))
        return result
        
    # Check type
    hp.check_error_type(error_type)

    error_mapping = {
        "central": 0,
        "plus": 1,
        "minus": 2
    }
    
    error_col_index = error_mapping.get(error_type)

    # Extracting parameter values
    A_m = MSTW_PDF["A_-"][evolution_order][0]
    delta_m = MSTW_PDF["delta_-"][evolution_order][0]
    eta_m = MSTW_PDF["eta_-"][evolution_order][0]
    x_0 = MSTW_PDF["x_0"][evolution_order][0]

    pdf = integral_sv_pdf_regge(A_m,delta_m,eta_m,x_0,j,alpha_p,t)

    if error_type != "central":
    # Extracting errors
        delta_A_m  = MSTW_PDF["A_-"][evolution_order][error_col_index]
        delta_delta_m = MSTW_PDF["delta_-"][evolution_order][error_col_index]
        delta_eta_m = MSTW_PDF["eta_-"][evolution_order][error_col_index]
        delta_x_0 = MSTW_PDF["x_0"][evolution_order][error_col_index]
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
    hp.check_error_type(error_type)
    
    error_mapping = {
        "central": 0,
        "plus": 1,
        "minus": 2
    }
    
    error_col_index = error_mapping.get(error_type)

    A_pdf      = MSTW_PDF["A_S"][evolution_order][0]
    eta_1      = MSTW_PDF["delta_S"][evolution_order][0]
    eta_2      = MSTW_PDF["eta_S"][evolution_order][0]
    epsilon    = MSTW_PDF["epsilon_S"][evolution_order][0]
    gamma_pdf  = MSTW_PDF["gamma_S"][evolution_order][0]

    pdf = integral_pdf_regge(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,j,alpha_p,t)

    if error_type != "central":
    # Extracting errors
        delta_A_pdf  = MSTW_PDF["A_S"][evolution_order][error_col_index]
        delta_eta_1 = MSTW_PDF["delta_S"][evolution_order][error_col_index]
        delta_eta_2 = MSTW_PDF["eta_S"][evolution_order][error_col_index]
        delta_epsilon = MSTW_PDF["epsilon_S"][evolution_order][error_col_index]
        delta_gamma_pdf = MSTW_PDF["gamma_S"][evolution_order][error_col_index]

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
    hp.check_error_type(error_type)
    
    error_mapping = {
        "central": 0,
        "plus": 1,
        "minus": 2
    }
    
    error_col_index = error_mapping.get(error_type)

    A_pdf      = MSTW_PDF["A_+"][evolution_order][0]
    eta_1      = MSTW_PDF["delta_S"][evolution_order][0]
    eta_2      = MSTW_PDF["eta_+"][evolution_order][0]
    epsilon    = MSTW_PDF["epsilon_S"][evolution_order][0]
    gamma_pdf  = MSTW_PDF["gamma_S"][evolution_order][0]

    pdf = integral_pdf_regge(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,j,alpha_p,t)

    if error_type != "central":
        # Extracting errors
        delta_A_pdf      = MSTW_PDF["A_+"][evolution_order][error_col_index]
        delta_eta_1      = MSTW_PDF["delta_S"][evolution_order][error_col_index]
        delta_eta_2      = MSTW_PDF["eta_+"][evolution_order][error_col_index]
        delta_epsilon    = MSTW_PDF["epsilon_S"][evolution_order][error_col_index]
        delta_gamma_pdf  = MSTW_PDF["gamma_S"][evolution_order][error_col_index]

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
        frac_2 = mp.gamma(3+eta_S)*mp.gamma(j+eta_Delta-1-alpha_p*t)/(mp.gamma(2+eta_Delta+eta_S+j-alpha_p*t))
        result = A_Delta*(1+((delta_Delta*(eta_Delta+j-alpha_p*t)+gamma_Delta*(3+eta_Delta+eta_S+j-alpha_p*t))*(eta_Delta+j-1+alpha_p*t))/frac_1)*frac_2
        return result
    def integral_Delta_pdf_regge_error(A_Delta,delta_A_Delta,eta_Delta,delta_eta_Delta,eta_S,delta_eta_S,gamma_Delta,delta_gamma_Delta,delta_Delta,delta_delta_Delta,j,alpha_p,t, error_type="central"):
        if error_type == "central":
             return 0
        def dpdf_dA_Delta(A_Delta,eta_Delta,eta_S,gamma_Delta,delta_Delta,j,alpha_p,t):
            term = (
            mp.gamma(eta_S + 3) * mp.gamma(eta_Delta + j - alpha_p * t - 1) *
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
            ) / mp.gamma(eta_Delta + eta_S + j - alpha_p * t + 2)
        
            return term
        def dpdf_deta_Delta(A_Delta,eta_Delta,eta_S,gamma_Delta,delta_Delta,j,alpha_p,t):
            term_1 = (
                A_Delta * mp.gamma(eta_S + 3) * mp.gamma(eta_Delta + j - alpha_p * t - 1) *
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
            ) / mp.gamma(eta_Delta + eta_S + j - alpha_p * t + 2)
        
            term_2 = (
                A_Delta * mp.gamma(eta_S + 3) * mp.gamma(eta_Delta + j - alpha_p * t - 1) *
                mp.digamma(eta_Delta + j - alpha_p * t - 1) *
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
            ) / mp.gamma(eta_Delta + eta_S + j - alpha_p * t + 2)
            
            term_3 = (
                -A_Delta * mp.gamma(eta_S + 3) * mp.gamma(eta_Delta + j - alpha_p * t - 1) *
                mp.digamma(eta_Delta + eta_S + j - alpha_p * t + 2) *
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
            ) / mp.gamma(eta_Delta + eta_S + j - alpha_p * t + 2)
            
            return term_1 + term_2 + term_3
        
        def dpdf_deta_S(A_Delta,eta_Delta,eta_S,gamma_Delta,delta_Delta,j,alpha_p,t):
            term_1 = (
                A_Delta * mp.gamma(eta_S + 3) * mp.gamma(eta_Delta + j - alpha_p * t - 1) *
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
            ) / mp.gamma(eta_Delta + eta_S + j - alpha_p * t + 2)
            
            term_2 = (
                A_Delta * mp.gamma(eta_S + 3) * mp.digamma(eta_S + 3) * mp.gamma(eta_Delta + j - alpha_p * t - 1) *
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
            ) / mp.gamma(eta_Delta + eta_S + j - alpha_p * t + 2)
            
            term_3 = (
                -A_Delta * mp.gamma(eta_S + 3) * mp.gamma(eta_Delta + j - alpha_p * t - 1) *
                mp.digamma(eta_Delta + eta_S + j - alpha_p * t + 2) *
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
            ) / mp.gamma(eta_Delta + eta_S + j - alpha_p * t + 2)
            
            return term_1 + term_2 + term_3
        def dpdf_dgamma_Delta(A_Delta,eta_Delta,eta_S,gamma_Delta,delta_Delta,j,alpha_p,t):
            term_1 = mp.gamma(eta_Delta + j - alpha_p * t)
            term_2 = mp.gamma(3 + eta_Delta + eta_S + j - alpha_p * t)
            result = A_Delta * mp.gamma(3+eta_S) * term_1 / term_2
            return result
        def dpdf_ddelta_Delta(A_Delta,eta_Delta,eta_S,gamma_Delta,delta_Delta,j,alpha_p,t):
            return (
                A_Delta * mp.gamma(eta_S + 3) * (eta_Delta + j - alpha_p * t - 1) * (eta_Delta + j - alpha_p * t) * 
                mp.gamma(eta_Delta + j - alpha_p * t - 1)
            ) / (
                (eta_Delta + eta_S + j - alpha_p * t + 2) *
                (eta_Delta + eta_S + j - alpha_p * t + 3) *
                mp.gamma(eta_Delta + eta_S + j - alpha_p * t + 2)
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

        # result = abs(mp.sqrt(Delta_A_Delta**2+Delta_eta_Delta**2+Delta_eta_S**2+Delta_gamma_Delta**2+Delta_delta_Delta**2))
        sum_squared = Delta_A_Delta**2+Delta_eta_Delta**2+Delta_eta_S**2+Delta_gamma_Delta**2+Delta_delta_Delta**2
        result = abs(mp.sqrt(sum_squared))
        return result
    
    hp.check_error_type(error_type)

     # Define a dictionary that maps the error_type to column indices
    error_mapping = {
        "central": 0,  # The column with the central value
        "plus": 1,     # The column with the + error value
        "minus": 2     # The column with the - error value
    }
    
    # Get the column index corresponding to the error_type
    error_col_index = error_mapping.get(error_type) 

    A_Delta     = MSTW_PDF["A_Delta"][evolution_order][0]
    eta_Delta   = MSTW_PDF["eta_Delta"][evolution_order][0]
    eta_S       = MSTW_PDF["eta_S"][evolution_order][0]
    delta_Delta = MSTW_PDF["delta_Delta"][evolution_order][0]
    gamma_Delta = MSTW_PDF["gamma_Delta"][evolution_order][0]


    pdf = integral_Delta_pdf_regge(A_Delta,eta_Delta,eta_S,gamma_Delta,delta_Delta,j,alpha_p,t)

    if error_type != "central":
        # Extracting errors
        delta_A_Delta      = MSTW_PDF["A_Delta"][evolution_order][error_col_index]
        delta_eta_Delta    = MSTW_PDF["eta_Delta"][evolution_order][error_col_index]
        delta_eta_S        = MSTW_PDF["eta_S"][evolution_order][error_col_index]
        delta_delta_Delta  = MSTW_PDF["delta_Delta"][evolution_order][error_col_index]
        delta_gamma_Delta  = MSTW_PDF["gamma_Delta"][evolution_order][error_col_index]

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
    hp.check_error_type(error_type)
    
     # Define a dictionary that maps the error_type to column indices
    error_mapping = {
        "central": 0,  # The column with the central value
        "plus": 1,     # The column with the + error value
        "minus": 2     # The column with the - error value
    }
    
    # Get the column index corresponding to the error_type
    error_col_index = error_mapping.get(error_type) 

    A_pdf     = MSTW_PDF["A_g"][evolution_order][0]
    eta_1     = MSTW_PDF["delta_g"][evolution_order][0]
    eta_2     = MSTW_PDF["eta_g"][evolution_order][0]
    epsilon   = MSTW_PDF["epsilon_g"][evolution_order][0]
    gamma_pdf = MSTW_PDF["gamma_g"][evolution_order][0]


    pdf = integral_pdf_regge(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,j,alpha_p,t)

    # Additional term at NLO and NNLO
    if evolution_order != "LO":
        # Get row index of entry
        A_pdf_prime     = MSTW_PDF["A_g'"][evolution_order][0]
        eta_1_prime     = MSTW_PDF["delta_g'"][evolution_order][0]
        eta_2_prime     = MSTW_PDF["eta_g'"][evolution_order][0]

        nlo_term = A_pdf_prime * (eta_1_prime + eta_2_prime + j - alpha_p * t) * mp.gamma(j - alpha_p *t + eta_1_prime-1)*mp.gamma(1+eta_2_prime)/\
                mp.gamma(j-alpha_p*t+eta_1_prime+eta_2_prime + 1)
        # print(pdf,nlo_term)
        pdf += nlo_term
    if error_type != "central":
    # Extracting errors
        delta_A_pdf      = MSTW_PDF["A_g"][evolution_order][error_col_index]
        delta_eta_1      = MSTW_PDF["delta_g"][evolution_order][error_col_index]
        delta_eta_2      = MSTW_PDF["eta_g"][evolution_order][error_col_index]
        delta_epsilon    = MSTW_PDF["epsilon_g"][evolution_order][error_col_index]
        delta_gamma_pdf  = MSTW_PDF["gamma_g"][evolution_order][error_col_index]

        pdf_error = integral_pdf_regge_error(A_pdf,delta_A_pdf,eta_1,delta_eta_1,eta_2,delta_eta_2,epsilon,delta_epsilon,gamma_pdf,delta_gamma_pdf,j,alpha_p,t,error_type)
        if evolution_order != "LO":
            delta_A_prime_pdf     = MSTW_PDF["A_g'"][evolution_order][error_col_index]
            delta_eta_1_prime     = MSTW_PDF["delta_g'"][evolution_order][error_col_index]
            delta_eta_2_prime     = MSTW_PDF["eta_g'"][evolution_order][error_col_index]

            # print(A_pdf,eta_1,eta_2,epsilon,gamma_pdf)
            # print(delta_A_pdf,delta_eta_1,delta_eta_2,delta_epsilon,delta_gamma_pdf)
            # print("-----")
            # print(A_pdf_prime,eta_1_prime,eta_2_prime)
            # print(delta_A_prime_pdf,delta_eta_1_prime,delta_eta_2_prime)

            dpdf_dA = mp.gamma(j - alpha_p *t + eta_1_prime-1)*mp.gamma(1+eta_2_prime)/\
                mp.gamma(j-alpha_p*t+eta_1_prime+eta_2_prime)
            dpdf_deta_1 = (A_pdf_prime * mp.gamma(1+eta_2_prime) * mp.gamma(j - alpha_p *t + eta_1_prime-1)/\
                mp.gamma(j-alpha_p*t+eta_1_prime+eta_2_prime) * \
            (mp.digamma(eta_1_prime + 1 )-mp.digamma(eta_1_prime + eta_2_prime + j - alpha_p * t))
            )
            dpdf_deta_2 = (A_pdf_prime * mp.gamma(1+eta_2_prime) * mp.gamma(j - alpha_p *t + eta_1_prime-1)/\
              mp.gamma(j-alpha_p*t+eta_1_prime+eta_2_prime + 1) * \
             (1 + (eta_1_prime + eta_2_prime + j -alpha_p * t) * (mp.digamma(eta_2_prime + 1) - mp.digamma(eta_1_prime + eta_2_prime + j -alpha_p * t + 1)))
            )
            # print(dpdf_dA,dpdf_deta_1,dpdf_deta_2)
            Delta_A = dpdf_dA * delta_A_prime_pdf
            Delta_eta_1 = dpdf_deta_1 * delta_eta_1_prime
            Delta_eta_2 = dpdf_deta_2 * delta_eta_2_prime
            sum_squared = Delta_A**2+Delta_eta_1**2+Delta_eta_2**2
            result = abs(mp.sqrt(sum_squared))
            pdf_error += result
            # pdf_error+= abs(mp.sqrt(Delta_A**2+Delta_eta_1**2+Delta_eta_2**2))
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
        hp.check_error_type(error_type)

        # Define a dictionary that maps the error_type to column indices
        error_mapping = {
        "central": 0,  # The column with the central value
        "plus": 1,     # The column with the + error value
        "minus": 2     # The column with the - error value
        }

        # Get the column index corresponding to the error_type
        error_col_index = error_mapping.get(error_type, 0)  # Default to 'central' if error_type is invalid

        A_pdf     = MSTW_PDF["A_u"][evolution_order][0]
        eta_1     = MSTW_PDF["eta_1"][evolution_order][0]
        eta_2     = MSTW_PDF["eta_2"][evolution_order][0]
        epsilon   = MSTW_PDF["epsilon_u"][evolution_order][0]
        gamma_pdf = MSTW_PDF["gamma_u"][evolution_order][0]

        delta_A_pdf = AAC_PDF["Delta_A_u"][evolution_order][0]
        alpha       = AAC_PDF["alpha_u"][evolution_order][0]
        gamma_pol   = AAC_PDF["Delta_gamma_u"][evolution_order][0]
        lambda_pol  = AAC_PDF["Delta_lambda_u"][evolution_order][0]

        pdf = integral_polarized_pdf_regge(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
                                           delta_A_pdf,alpha,gamma_pol,lambda_pol,
                                           j,alpha_p,t,evolution_order)
        if error_type != "central":
            err_delta_A_pdf = AAC_PDF["Delta_A_u"][evolution_order][error_col_index]
            err_alpha       = AAC_PDF["alpha_u"][evolution_order][error_col_index]
            err_gamma_pol   = AAC_PDF["Delta_gamma_u"][evolution_order][error_col_index]
            err_lambda_pol  = AAC_PDF["Delta_lambda_u"][evolution_order][error_col_index]

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
        hp.check_error_type(error_type)

        # Define a dictionary that maps the error_type to column indices
        error_mapping = {
        "central": 0,  # The column with the central value
        "plus": 1,     # The column with the + error value
        "minus": 2     # The column with the - error value
        }

        # Get the column index corresponding to the error_type
        error_col_index = error_mapping.get(error_type, 0)  # Default to 'central' if error_type is invalid

        A_pdf     = MSTW_PDF["A_d"][evolution_order][0]
        eta_1     = MSTW_PDF["eta_3"][evolution_order][0]
        eta_2     = MSTW_PDF["eta_2"][evolution_order][0] + MSTW_PDF["eta_4-eta_2"][evolution_order][0]  # eta_4 ≡ eta_2 + (eta_4 - eta_2)
        epsilon   = MSTW_PDF["epsilon_d"][evolution_order][0]
        gamma_pdf = MSTW_PDF["gamma_d"][evolution_order][0]

        Delta_A_pdf = AAC_PDF["Delta_A_d"][evolution_order][0]
        alpha       = AAC_PDF["alpha_d"][evolution_order][0]
        gamma_pol   = AAC_PDF["Delta_gamma_d"][evolution_order][0]
        lambda_pol  = AAC_PDF["Delta_lambda_d"][evolution_order][0]

        pdf = integral_polarized_pdf_regge(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
                                           Delta_A_pdf,alpha,gamma_pol,lambda_pol,
                                           j,alpha_p,t,evolution_order)
        if error_type != "central":
            err_delta_A_pdf = AAC_PDF["Delta_A_d"][evolution_order][error_col_index]
            err_alpha       = AAC_PDF["alpha_d"][evolution_order][error_col_index]
            err_gamma_pol   = AAC_PDF["Delta_gamma_d"][evolution_order][error_col_index]
            err_lambda_pol  = AAC_PDF["Delta_lambda_d"][evolution_order][error_col_index]

            pdf_error = integral_polarized_pdf_regge_error(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
                                            Delta_A_pdf,err_delta_A_pdf,alpha,err_alpha,gamma_pol,err_gamma_pol,lambda_pol,err_lambda_pol,
                                            j,alpha_p,t,evolution_order,error_type)
            return pdf, pdf_error
        else:
            return pdf, 0

def integral_polarized_S_pdf_regge(j,eta,alpha_p,t, evolution_order = "LO", error_type="central"):
        """
        Result of the integral of the Reggeized S(x) PDF based on the given LO parameters and selected errors. 
        We are assuming \Delta u = \Delta d = \Delta s = s
        
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
        hp.check_error_type(error_type)

        # Define a dictionary that maps the error_type to column indices
        error_mapping = {
                "central": 0,  # The column with the central value
                "plus": 1,     # The column with the + error value
                "minus": 2     # The column with the - error value
        }
        
        # Get the column index corresponding to the error_type
        error_col_index = error_mapping.get(error_type, 0)  # Default to 'central' if error_type is invalid

        A_pdf     = MSTW_PDF["A_S"][evolution_order][0]
        eta_1     = MSTW_PDF["delta_S"][evolution_order][0]
        eta_2     = MSTW_PDF["eta_S"][evolution_order][0]
        epsilon   = MSTW_PDF["epsilon_S"][evolution_order][0]
        gamma_pdf = MSTW_PDF["gamma_S"][evolution_order][0]

        delta_A_pdf = AAC_PDF["Delta_A_S"][evolution_order][0]
        alpha       = AAC_PDF["alpha_S"][evolution_order][0]
        gamma_pol   = AAC_PDF["Delta_gamma_S"][evolution_order][0]
        lambda_pol  = AAC_PDF["Delta_lambda_S"][evolution_order][0]

        pdf = integral_polarized_pdf_regge(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
                                            delta_A_pdf,alpha,gamma_pol,lambda_pol,
                                           j,alpha_p,t,evolution_order)
        if error_type != "central":
            err_delta_A_pdf = AAC_PDF["Delta_A_S"][evolution_order][error_col_index]
            err_alpha       = AAC_PDF["alpha_S"][evolution_order][error_col_index]
            err_gamma_pol   = AAC_PDF["Delta_gamma_S"][evolution_order][error_col_index]
            err_lambda_pol  = AAC_PDF["Delta_lambda_S"][evolution_order][error_col_index]
            pdf_error = integral_polarized_pdf_regge_error(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
                                           delta_A_pdf,err_delta_A_pdf,alpha,err_alpha,gamma_pol,err_gamma_pol,lambda_pol,err_lambda_pol,
                                           j,alpha_p,t,evolution_order,error_type)
            # Polarized sea quark pdf extremely sensitive to parametrization
            # Enforcing standard form combined with Gaussian error propagation
            # gives a huge error that is not compatible with the results by AAC
            # so we enforce the same scale for now
            pdf_error /= 5.20
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
        hp.check_error_type(error_type)

        # Define a dictionary that maps the error_type to column indices
        error_mapping = {
                "central": 0,  # The column with the central value
                "plus": 1,     # The column with the + error value
                "minus": 2     # The column with the - error value
        }
        
        # Get the column index corresponding to the error_type
        error_col_index = error_mapping.get(error_type, 0)  # Default to 'central' if error_type is invalid

        # Extracting central parameter values
        A_pdf = MSTW_PDF["A_g"][evolution_order][0]
        eta_1 = MSTW_PDF["delta_g"][evolution_order][0]
        eta_2 = MSTW_PDF["eta_g"][evolution_order][0]
        epsilon = MSTW_PDF["epsilon_g"][evolution_order][0]
        gamma_pdf = MSTW_PDF["gamma_g"][evolution_order][0]
        # Extracting parameter values based on error type
        delta_A_pdf = AAC_PDF["Delta_A_g"][evolution_order][0]
        alpha = AAC_PDF["alpha_g"][evolution_order][0]
        gamma_pol = AAC_PDF["Delta_gamma_g"][evolution_order][0]
        lambda_pol = AAC_PDF["Delta_lambda_g"][evolution_order][0]

        pdf = integral_polarized_pdf_regge(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
                                           delta_A_pdf,alpha,gamma_pol,lambda_pol,
                                           j,alpha_p,t,evolution_order)
        if evolution_order != "LO":
            # Additional gluon contribution at NLO and NNLO that is not of the LO form
            A_pdf_prime   = MSTW_PDF["A_g'"][evolution_order][0]
            eta_1_prime   = MSTW_PDF["delta_g'"][evolution_order][0]
            eta_2_prime   = MSTW_PDF["eta_g'"][evolution_order][0]


            pdf += A_pdf_prime * delta_A_pdf *mp.gamma(1+eta_2_prime) * (
                    (1-gamma_pol)*mp.gamma(eta_1_prime + j + alpha - alpha_p * t - 1)/
                    mp.gamma(eta_1_prime + eta_2_prime + j + alpha - alpha_p * t) +
                    gamma_pol * mp.gamma(eta_1_prime + j + alpha - alpha_p * t + lambda_pol- 1)/
                    mp.gamma(eta_1_prime + eta_2_prime + j + alpha - alpha_p * t + lambda_pol)
            )

        if error_type != "central":
            err_delta_A_pdf = AAC_PDF["Delta_A_g"][evolution_order][error_col_index]
            err_alpha = AAC_PDF["alpha_g"][evolution_order][error_col_index]
            err_gamma_pol = AAC_PDF["Delta_gamma_g"][evolution_order][error_col_index]
            err_lambda_pol = AAC_PDF["Delta_lambda_g"][evolution_order][error_col_index]
            # print(err_delta_A_pdf,err_alpha,err_gamma_pol,err_lambda_pol)

            pdf_error = integral_polarized_pdf_regge_error(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
                                           delta_A_pdf,err_delta_A_pdf,alpha,err_alpha,gamma_pol,err_gamma_pol,lambda_pol,err_lambda_pol,
                                           j,alpha_p,t,evolution_order,error_type)
            if evolution_order != "LO":
                dpdf_dA = A_pdf_prime *mp.gamma(1+eta_2_prime) * (
                    (1-gamma_pol)*mp.gamma(eta_1_prime + j + alpha - alpha_p * t - 1)/
                    mp.gamma(eta_1_prime + eta_2_prime + j + alpha - alpha_p * t) +
                    gamma_pol * mp.gamma(eta_1_prime + j + alpha - alpha_p * t + lambda_pol- 1)/
                    mp.gamma(eta_1_prime + eta_2_prime + j + alpha - alpha_p * t + lambda_pol)
                            )
                dpdf_dalpha = A_pdf_prime * delta_A_pdf * mp.gamma(eta_2_prime + 1) * (
                            (
                                gamma_pol * mp.gamma(eta_1_prime + j + alpha - alpha_p * t + lambda_pol - 1) *
                                (mp.digamma(eta_1_prime + j + alpha - alpha_p * t + lambda_pol - 1) -
                                mp.digamma(eta_1_prime + eta_2_prime + j + alpha - alpha_p * t + lambda_pol))
                            )/\
                            mp.gamma(eta_1_prime + eta_2_prime + j + alpha - alpha_p * t + lambda_pol) + \
                            (
                            (gamma_pol - 1) * mp.gamma(eta_1_prime + j + alpha - alpha_p * t - 1) *
                            (mp.digamma(eta_1_prime + eta_2_prime + j + alpha - alpha_p * t) -
                            mp.digamma(eta_1_prime + j + alpha - alpha_p * t - 1))
                            ) /\
                            mp.gamma(eta_1_prime + eta_2_prime + j + alpha - alpha_p * t)
                            )
                dpdf_dgamma_pol = A_pdf_prime * delta_A_pdf * mp.gamma(eta_2_prime + 1) * (
                                mp.gamma(eta_1_prime + j + alpha - alpha_p * t + lambda_pol - 1) / \
                                mp.gamma(eta_1_prime + eta_2_prime + j + alpha - alpha_p * t + lambda_pol) - \
                                mp.gamma(eta_1_prime + j + alpha - alpha_p * t - 1) / \
                                mp.gamma(eta_1_prime + eta_2_prime + j + alpha - alpha_p * t)
                                )
                dpdf_dlambda_pol = A_pdf_prime * delta_A_pdf * gamma_pol * mp.gamma(eta_2_prime + 1) * (
                                mp.gamma(eta_1_prime + j + alpha - alpha_p * t + lambda_pol - 1) *
                                (mp.digamma(eta_1_prime + j + alpha - alpha_p * t + lambda_pol - 1) -
                                mp.digamma(eta_1_prime + eta_2_prime + j + alpha - alpha_p * t + lambda_pol))
                                )/mp.gamma(eta_1_prime + eta_2_prime + j + alpha - alpha_p * t + lambda_pol)
                Delta_A_pdf = dpdf_dA * err_delta_A_pdf
                Delta_alpha = dpdf_dalpha * err_alpha
                Delta_gamma_pol = dpdf_dgamma_pol * err_gamma_pol
                Delta_lambda_pol = dpdf_dlambda_pol * err_lambda_pol
                # print(dpdf_dA,dpdf_dalpha,dpdf_dgamma_pol,dpdf_dlambda_pol)
                # print(Delta_A_pdf,Delta_alpha,Delta_gamma_pol,Delta_lambda_pol)
                sum_squared = Delta_A_pdf**2 + Delta_alpha**2 + Delta_gamma_pol**2 +Delta_lambda_pol**2
                result = abs(mp.sqrt(sum_squared))
                pdf_error += result
                # pdf_error += abs(mp.sqrt(Delta_A_pdf**2 + Delta_alpha**2 + Delta_gamma_pol**2 +Delta_lambda_pol**2))
                # Polarized gluon pdf extremely sensitive to parametrization
                # Enforcing standard form combined with Gaussian error propagation
                # gives a huge error that is not compatible with the results by AAC
                # so we enforce the same scale for now
                pdf_error /= 5.20
            return pdf, pdf_error
        else:
            return pdf, 0

# Define Reggeized conformal moments
non_singlet_interpolation = {}
if cfg.INTERPOLATE_INPUT_MOMENTS:
    for moment_type, moment_label, evolution_order, error_type in product(["non_singlet_isovector","non_singlet_isoscalar"],cfg.LABELS, cfg.ORDERS, cfg.ERRORS):
        if moment_type not in cfg.MOMENTS:
            continue
        params = {
            "solution": ".",
            "particle": "quark",
            "moment_type": moment_type,
            "moment_label": moment_label,
            "evolution_order": evolution_order,
            "error_type": error_type,
        }
        selected_triples = [
            (eta, t, mu)
            for eta, t, mu in zip(cfg.ETA_ARRAY, cfg.T_ARRAY, cfg.MU_ARRAY)
            if mu == 1
        ]

        non_singlet_interpolation[(moment_type, moment_label, evolution_order, error_type)] = [
            hp.build_moment_interpolator(eta, t, mu, **params)
            for eta, t, mu in selected_triples
        ]
def non_singlet_isovector_moment(j,eta,t, moment_label="A",evolve_type="vector", evolution_order="LO",error_type="central"):
    """
    Currently no skewness dependence!
    """
   # Check type
    hp.check_error_type(error_type)
    hp.check_evolution_order(evolution_order)
    hp.check_moment_type_label("non_singlet_isovector",moment_label)
    hp.check_evolve_type(evolve_type)

    if cfg.INTERPOLATE_INPUT_MOMENTS and isinstance(j,(complex,mp.mpc)):
        key = ("non_singlet_isovector",moment_label, evolution_order, error_type)
        selected_triples = [
            (eta_, t_, mu_)
            for eta_, t_, mu_ in zip(cfg.ETA_ARRAY, cfg.T_ARRAY, cfg.MU_ARRAY)
            if mu_ == 1
        ]
        index = selected_triples.index((eta, t, 1)) 
        interp = non_singlet_interpolation[key][index]
        return interp(j)

    alpha_prime = hp.get_regge_slope("non_singlet_isovector",moment_label,evolve_type)

    if moment_label in ["A","B"]:
        uv, uv_error = integral_uv_pdf_regge(j,eta,alpha_prime,t,evolution_order,error_type)
        dv, dv_error = integral_dv_pdf_regge(j,eta,alpha_prime,t,evolution_order,error_type)
    elif moment_label =="Atilde":
       uv, uv_error = integral_polarized_uv_pdf_regge(j,eta,alpha_prime,t,evolution_order,error_type)
       dv, dv_error = integral_polarized_dv_pdf_regge(j,eta,alpha_prime,t,evolution_order,error_type)

    norm = hp.get_moment_normalizations("non_singlet_isovector",moment_label,evolve_type,evolution_order)
    sum_squared = uv_error**2+dv_error**2
    error = abs(mp.sqrt(sum_squared))
    error = hp.error_sign(error,error_type)
    result = norm * ( uv - dv + error )

    return result

def u_minus_d_pdf_regge(j,eta,t, evolution_order = "LO", error_type="central"):
    """ Currently only experimental function that does not set ubar=dbar"""
    # Check type
    hp.check_error_type(error_type)
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
    hp.check_error_type(error_type)
    hp.check_moment_type_label("non_singlet_isoscalar",moment_label)
    hp.check_evolve_type(evolve_type)
    if cfg.INTERPOLATE_INPUT_MOMENTS and isinstance(j,(complex,mp.mpc)):
        key = ("non_singlet_isoscalar",moment_label, evolution_order, error_type)
        selected_triples = [
            (eta_, t_, mu_)
            for eta_, t_, mu_ in zip(cfg.ETA_ARRAY, cfg.T_ARRAY, cfg.MU_ARRAY)
            if mu_ == 1
        ]
        index = selected_triples.index((eta, t, 1)) 
        interp = non_singlet_interpolation[key][index]
        return interp(j)

    alpha_prime = hp.get_regge_slope("non_singlet_isoscalar",moment_label,evolve_type)

    if moment_label in ["A","B"]:
        uv, uv_error = integral_uv_pdf_regge(j,eta,alpha_prime,t,evolution_order,error_type)
        dv, dv_error = integral_dv_pdf_regge(j,eta,alpha_prime,t,evolution_order,error_type)
    elif moment_label =="Atilde":
        uv, uv_error = integral_polarized_uv_pdf_regge(j,eta,alpha_prime,t,evolution_order,error_type)
        dv, dv_error = integral_polarized_dv_pdf_regge(j,eta,alpha_prime,t,evolution_order,error_type)

    norm = hp.get_moment_normalizations("non_singlet_isoscalar",moment_label,evolve_type,evolution_order)
    sum_squared = uv_error**2+dv_error**2
    error = abs(mp.sqrt(sum_squared))
    error = hp.error_sign(error,error_type)
    result = norm * ( uv + dv + error )

    return result

def u_plus_d_pdf_regge(j,eta,t, evolution_order = "LO", error_type="central"):
    """ Currently only experimental function that does not set ubar=dbar"""
    # Check type
    hp.check_error_type(error_type)
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
        t = -1e-12 if t == 0 else t
        if mp.im(j) < 0:
            j = mp.conj(j)
            result = mp.hyp2f1(-j/2, -(j-1)/2, 1/2 - j, - 4 * m_N**2/t * eta**2)
            result = mp.conj(result)
        else:
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
        sum_squared = uv_error**2+dv_error**2+Spdf_error**2
        result = uv + dv + Spdf
    elif Nf == 2:
        sum_squared = uv_error**2+dv_error**2+Spdf_error**2+s_plus_error**2
        result = uv + dv + Spdf - s_plus
    elif Nf == 1:
        sum_squared = .5*(4*uv_error**2+Spdf_error**2+s_plus_error**2+4*Delta_error**2)
        result = .5*(Spdf-s_plus+2*uv-2*Delta)
    else :
        raise ValueError("Currently only (integer) 1 <= Nf <= 3 supported")
    error = abs(mp.sqrt(sum_squared))
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
        sum_squared_1 = uv_error**2+dv_error**2+Sv_error**2
        term_2 = uv_s + dv_s + Sv_s 
        sum_squared_2 = uv_s_error**2+dv_s_error**2+Sv_s_error**2
    elif Nf == 2:
        term_1 = uv + dv + Sv - s_plus
        sum_squared_1 = uv_error**2+dv_error**2+Sv_error**2+s_plus_error**2
        term_2 = uv_s + dv_s + Sv_s - s_plus_s
        sum_squared_2 = uv_s_error**2+dv_s_error**2+Sv_s_error**2+s_plus_s_error**2
    elif Nf == 1:
        term_1 = .5*(Sv-s_plus+2*uv-2*Delta)
        sum_squared_1 = .5*(4*uv_error**2+Sv_error**2+s_plus_error**2+4*Delta_error**2)
        term_2 = .5*(Sv_s-s_plus_s+2*uv_s-2*Delta_s)
        sum_squared_2 = .5*(4*uv_s_error**2+Sv_s_error**2+s_plus_s_error**2+4*Delta_s_error**2)
    else :
        raise ValueError("Currently only (integer) 1 <= Nf <= 3 supported")
    sum_squared = mp.sqrt(sum_squared_1**2 + sum_squared_2**2)
    # error = np.frompyfunc(abs,1,1)(mp.sqrt(sum_squared))
    error = abs(mp.sqrt(sum_squared))
    error = (d_hat(j,eta,t)-1)*error
    result = (d_hat(j,eta,t)-1)*(term_1-term_2)
    return result, error

def quark_singlet_regge(j,eta,t,Nf=3,moment_label="A",evolve_type="vector",evolution_order="LO",error_type="central"):
    # Check type
    hp.check_error_type(error_type)
    hp.check_evolve_type(evolve_type)
    hp.check_moment_type_label("singlet",moment_label)
    hp.check_evolution_order(evolution_order)
    if moment_label == "B":
        prf = -1
    else:
        prf = +1

    alpha_prime_ud, alpha_prime_s, _, _ = hp.get_regge_slope("singlet",moment_label,evolve_type)
    norm_A, norm_D, _, _ = hp.get_moment_normalizations("singlet",moment_label,evolve_type,evolution_order)

    term_1, error_1 = quark_singlet_regge_A(j,eta,t,Nf,alpha_prime_ud,moment_label,evolution_order,error_type)
    if eta == 0:
        result = norm_A * term_1
        error = norm_A * error_1
    else :
        term_2, error_2 = quark_singlet_regge_D(j,eta,t,Nf,alpha_prime_ud,alpha_prime_s,moment_label,evolution_order,error_type)
        sum_squared = norm_A**2 * error_1**2 + norm_D**2 * error_2**2
        # error = np.frompyfunc(abs, 1, 1)(mp.sqrt(sum_squared))
        error = abs(mp.sqrt(sum_squared))
        result = norm_A * term_1 + norm_D * prf * term_2

    return result, error

def gluon_singlet_regge_A(j,eta,t, alpha_prime_T = 0.627,moment_label="A", evolution_order="LO",error_type="central"):
    if moment_label == "A":
        result, error = integral_gluon_pdf_regge(j,eta,alpha_prime_T,t,evolution_order,error_type)
    elif moment_label =="Atilde":
        result, error = integral_polarized_gluon_pdf_regge(j,eta,alpha_prime_T,t,evolution_order,error_type)
    else:
        raise ValueError(f"Unsupported moment label {moment_label}")
    return result, error

def gluon_singlet_regge_D(j,eta,t, alpha_prime_T = 0.627, alpha_prime_S = 4.277,moment_label="A",evolution_order="LO", error_type="central"):
    # Check type
    hp.check_error_type(error_type)
    hp.check_moment_type_label("singlet",moment_label)
    if eta == 0:
        return 0, 0 
    else :
        term_1 = (d_hat(j,eta,t)-1)
        term_2, error_2 = gluon_singlet_regge_A(j,eta,t,alpha_prime_T,moment_label,evolution_order,error_type)
        if moment_label == "A":
            term_3, error_3 = integral_gluon_pdf_regge(j,eta,t,alpha_prime_S,evolution_order,error_type)
        elif moment_label =="Atilde":
            term_3, error_3 = integral_polarized_gluon_pdf_regge(j,eta,t,alpha_prime_S,evolution_order,error_type)
        else:
            raise ValueError(f"Unsupported moment label {moment_label}")
        sum_squared = error_2**2+error_3**2
        # error = np.frompyfunc(abs, 1, 1)(mp.sqrt(sum_squared))
        error = abs(mp.sqrt(sum_squared))
        error = term_1 * error
        result = term_1 * (term_2-term_3)
        return result, error
    
def gluon_singlet_regge(j,eta,t,moment_label="A",evolve_type="vector", evolution_order="LO",error_type="central"):
    # Check type
    hp.check_error_type(error_type)
    hp.check_evolve_type(evolve_type)
    hp.check_moment_type_label("singlet",moment_label)

    if moment_label == "B":
        prf = -1
    else:
        prf = +1

    _, _, alpha_prime_T, alpha_prime_S = hp.get_regge_slope("singlet",moment_label,evolve_type)
    _, _, norm_A, norm_D = hp.get_moment_normalizations("singlet",moment_label,evolve_type,evolution_order)

    term_1, error_1 = gluon_singlet_regge_A(j,eta,t,alpha_prime_T,moment_label,evolution_order,error_type)
    if eta == 0:
        result = norm_A * term_1
        error = norm_A * error_1
    else :
        term_2, error_2 = gluon_singlet_regge_D(j,eta,t,alpha_prime_T,alpha_prime_S,moment_label,evolution_order,error_type)
        sum_squared = norm_A**2 * error_1**2 + norm_D**2 * error_2**2
        # error = np.frompyfunc(abs, 1, 1)(mp.sqrt(sum_squared))
        error = abs(mp.sqrt(sum_squared))
        result = norm_A * term_1 + norm_D * prf * term_2
    return result, error

singlet_interpolation = {}
if cfg.INTERPOLATE_INPUT_MOMENTS:
    for solution, moment_label, evolution_order, error_type in product(["+","-"],cfg.LABELS, cfg.ORDERS, cfg.ERRORS):
        if "singlet" not in cfg.MOMENTS:
            continue
        params = {
            "solution": solution,
            "particle": "quark",
            "moment_type": "singlet",
            "moment_label": moment_label,
            "evolution_order": evolution_order,
            "error_type": error_type,
        }
        selected_triples = [
            (eta, t, mu)
            for eta, t, mu in zip(cfg.ETA_ARRAY, cfg.T_ARRAY, cfg.MU_ARRAY)
            if mu == 1
        ]

        singlet_interpolation[(solution, moment_label, evolution_order, error_type)] = [
            hp.build_moment_interpolator(eta, t, mu, **params)
            for eta, t, mu in selected_triples
        ]
def singlet_moment(j,eta,t,Nf=3,moment_label="A",evolve_type="vector",solution="+",evolution_order="LO",error_type="central",interpolation=True):
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
    if cfg.INTERPOLATE_INPUT_MOMENTS and isinstance(j,(complex,mp.mpc)):
        key = (solution,moment_label, evolution_order, "central")
        selected_triples = [
            (eta_, t_, mu_)
            for eta_, t_, mu_ in zip(cfg.ETA_ARRAY, cfg.T_ARRAY, cfg.MU_ARRAY)
            if mu_ == 1
        ]
        index = selected_triples.index((eta, t, 1)) 
        interp = singlet_interpolation[key][index]
        if error_type == "central":
            return interp(j), 0
        else:
            key_err = (solution,moment_label, evolution_order, error_type)
            interp_err = singlet_interpolation[key_err][index]
            return interp(j), interp_err(j)


    # Switch sign
    if solution == "+":
        solution = "-"
    elif solution == "-":
        solution = "+"
    else:
        raise ValueError("Invalid solution type. Use '+' or '-'.")

    quark_prf = .5 
    quark_in, quark_in_error = quark_singlet_regge(j,eta,t,Nf,moment_label,evolve_type,evolution_order,error_type)
    # Note: j/6 already included in gamma_qg and gamma_gg definitions
    gluon_prf = .5 * (gamma_qg(j-1,Nf,evolve_type,"LO",interpolation=interpolation)/
                    (gamma_qq(j-1,Nf,"singlet",evolve_type,"LO",interpolation=interpolation)-gamma_pm(j-1,Nf,evolve_type,solution,interpolation=interpolation)))
    gluon_in, gluon_in_error = gluon_singlet_regge(j,eta,t,moment_label,evolve_type,evolution_order,error_type)
    # print(solution,gluon_prf)
    sum_squared = quark_prf**1 * quark_in_error**2 + gluon_prf**2*gluon_in_error**2
    # print("->",quark_in,quark_in_error,quark_prf)
    # print("->",gluon_in,gluon_in_error,gluon_prf)
    # error = np.frompyfunc(abs, 1, 1)(mp.sqrt(sum_squared))
    error = abs(mp.sqrt(sum_squared))
    result = quark_prf * quark_in + gluon_prf * gluon_in
    return result, error


################################
##### Evolution Equations ######
################################

def harmonic_sum(l,j):
    if j < 0:
        print("Warning: sum called for negative iterator bound")
    sign = (-1 if l < 0 else 1)
    l_abs = abs(l)
    return sum((sign)**k/k**l_abs for k in range(1, j+1))

def harmonic_sum_prime(l,j):
    return 2**(l-1) * sum((1+(-1)**k)/k**l for k in range(1, j+1))

def harmonic_sum_tilde(j):
    return sum((-1)**k/k**2 * harmonic_sum(1,k) for k in range(1, j+1))

def fractional_finite_sum(func,k_0=1,k_1=1,epsilon=.2,k_range=10,n_k=300,alternating_sum=False, n_tuple = 1, plot_integrand = False,
                          full_range=True,trap=False,n_jobs=1,error_est=True):
    """
    Computes the fractional finite sume of a one-parameter function by either using mpmath quad (trap = False) or a trapezoidal rule (trap = True).
    Can handle tuples of functions as input.
    """
    if not trap and not plot_integrand:
        integral = [0] * n_tuple
        for i in range(n_tuple):
            @hp.mpmath_vectorize
            def integrand_quad(x):
                if full_range:
                    k = mp.re(k_0) - epsilon + 1j * (mp.tan(x) - mp.im(k_0))
                    # i already accounted for below
                    dk = mp.sec(x)**2
                else:
                    k = mp.re(k_0) - epsilon + 1j * (x - mp.im(k_0))
                    # i already accounted for below
                    dk = 1
                if alternating_sum:
                    trig_term = mp.csc(mp.pi * k)
                    alt_sign = power_minus_1(k_1 + 1 - mp.re(k_0))
                else:
                    trig_term = mp.cot(mp.pi * k )
                    alt_sign = 1
                if n_tuple == 1:
                    f1 = func(k)
                    f2 = func(k+k_1+1-mp.re(k_0))
                else:
                    f1 = func(k)[i]
                    f2 = func(k+k_1+1-mp.re(k_0))[i]
                res = -.5 * trig_term * (f1 - alt_sign * f2) * dk
                return res
            if full_range:
                eps = 1e-6
                b1 = - (np.pi/2 - eps)
                b2 = np.pi/2 - eps
            else:
                b1 = -k_range
                b2 = k_range
            int_tmp, _ = fixed_quad(lambda k: integrand_quad(k), b1, b2, n = 100)
            if error_est:
                # Estimate error
                int_tmp_50, _ = fixed_quad(lambda k: integrand_quad(k), b1, b2, n = 50)
                err_tmp = abs(int_tmp - int_tmp_50)
                if err_tmp > 1e-4:
                    print(f"Warning: large error={err_tmp} detected in fractional_finite_sum")
            # Discard small imaginary part
            int_tmp = int_tmp.real if abs(int_tmp.imag) < 1e-6 else int_tmp 
            integral[i] += int_tmp
    else:
        # Trapezoidal rule
        # start_time = time.time()
        k_vals = np.linspace(-k_range, k_range, n_k) 
        k_vals = k_vals - mp.im(k_0)
        k_vals_trig = mp.re(k_0) - epsilon + 1j * k_vals
        # k_vals_shift = k_1 + 1 - epsilon + 1j * k_vals
        k_vals_shift = k_vals_trig + k_1 + 1 - mp.re(k_0)

        if alternating_sum:
            trig_term = np.array([mp.csc(mp.pi * k) for k in k_vals_trig])
        else:
            trig_term = np.array([mp.cot(mp.pi * k ) for k in k_vals_trig])

        if plot_integrand:
            n_jobs = -1
        f_vals = np.array(Parallel(n_jobs=n_jobs)(delayed(func)(k) for k in k_vals_trig))
        f_vals_shift = np.array(Parallel(n_jobs=n_jobs)(delayed(func)(k) for k in k_vals_shift))

        if n_tuple == 1:
            f_vals = f_vals.reshape(-1, 1)
            f_vals_shift = f_vals_shift.reshape(-1, 1)

        if alternating_sum:
            # Perform index shift on (-1)^k
            # alt_sign = (-1)**(mp.re(k_0) + mp.re(k_1) + 1)
            # alt_sign = (-1)**(k_1)
            alt_sign = power_minus_1(k_1 + 1 - mp.re(k_0))
            f_vals_shift = np.array([x * alt_sign for x in f_vals_shift], dtype=object)

        # Compute the difference
        f_diff = f_vals - f_vals_shift

        # Initialize array for tuple solution
        integral = [0] * n_tuple
        if plot_integrand:
            plt.figure(figsize=(10, 6))
        for i in range(n_tuple):
            integrand = -0.5 * trig_term * f_diff[:,i]
            # Compute integrand
            integrand_real = np.array([float(mp.re(x)) for x in integrand])
            integrand_imag = np.array([float(mp.im(x)) for x in integrand])

            if plot_integrand:
                print("integrands at k_max:",integrand_real[-1],integrand_imag[-1])
                if n_tuple == 1:
                    plt.plot(k_vals,integrand_real,label=f"Re")
                    plt.plot(k_vals,integrand_imag,label=f"Im")
                else:
                    plt.plot(k_vals,integrand_real,label=f"real_{i}")
                    plt.plot(k_vals,integrand_imag,label=f"imag_{i}")
                plt.grid(True)
                plt.legend()
                plt.show()
                if n_tuple == 1:
                    # Return a scalar for n_tuple = 1
                    return integral[0]
                else:
                    return integral
            else:
                integral_real = trapezoid(integrand_real, k_vals)
                integral_imag = trapezoid(integrand_imag, k_vals)
                # end_time = time.time()
                # print("t0",end_time - start_time)
                integral[i] += integral_real + 1j * integral_imag
                # end_time = time.time()
                # print("trapezoidal with alternating_sum =",alternating_sum,integral[i],end_time - start_time)

    if n_tuple == 1:
        # Return a scalar for n_tuple = 1
        return integral[0]
    else:
        return integral
    
def polygamma_asymptotic(n, z, terms=5):
    if n == 0:
        # Degenerate case: digamma
        result = mp.ln(z) - 1 / (2 * z)
        for k in range(1, terms + 1):
            bernoulli_2k = mp.bernoulli(2 * k)
            result -= bernoulli_2k / (2 * k * z**(2 * k))
        return result

    # General polygamma case
    result = 0
    sign = (-1)**(n + 1)
    for k in range(terms):
        bernoulli_2k = mp.bernoulli(2 * k)
        if bernoulli_2k == 0 and k > 0:
            continue  # Bernoulli numbers vanish for odd index > 1
        coeff = mp.fac(2 * k + n - 1) / mp.fac(k)
        term = coeff * bernoulli_2k / z**(2 * k + n)
        result += term
    return sign * result

# Generate interpolators
harmonic_number_1_interpolation = hp.build_harmonic_interpolator(1)
harmonic_number_2_interpolation = hp.build_harmonic_interpolator(2)
harmonic_number_3_interpolation = hp.build_harmonic_interpolator(3)
harmonic_number_m2_interpolation = hp.build_harmonic_interpolator(-2)
harmonic_number_m3_interpolation = hp.build_harmonic_interpolator(-3)
# Pick interpolation
def harmonic_number_interpolation(l):
    harmonic_dictionary = {
    1: harmonic_number_1_interpolation,
    2: harmonic_number_2_interpolation,
    3: harmonic_number_3_interpolation,
    -2: harmonic_number_m2_interpolation,
    -3: harmonic_number_m3_interpolation
    }
    if l in harmonic_dictionary:
        return harmonic_dictionary[l]
    else: raise ValueError(f"Index l not in dictionary. Generate table and modify module.")

def harmonic_number(l,j,k_range=3,n_k=300,plot_integrand=False,trap=False,interpolation=True):
    if interpolation:
        interp = harmonic_number_interpolation(l)
        return interp(j)
    if j.imag == 0 and j.real == int(j.real):
        j_int = int(j.real)
        if j_int >= 1:
            result = harmonic_sum(l, j_int)
        elif j_int == 0:
            result = 0
        else:
            result = harmonic_sum(l, -(j_int + 1))
    elif l == 1:
        euler_gamma = 0.5772156649
        if abs(j) < 25:
            result = mp.digamma(j+1) + euler_gamma
        else:
        # Approximate with asymptotic series expansion
            result = polygamma_asymptotic(0,j+1,terms=1) + euler_gamma
    elif l > 1:
        if abs(j) < 25:
            result = mp.zeta(l) - mp.zeta(l,j+1)
        else:
        # Approximate with asymptotic series expansion
            result = mp.zeta(l) - (-1)**l / mp.factorial(l-1) * polygamma_asymptotic(l-1,j+1,terms=1)
    elif l == -1:
        if abs(j) < 25:
            result = power_minus_1(j) * mp.lerchphi(-1, 1, 1 + j) - mp.log(2)
        else:
        # Approximate with asymptotic series expansion
        # Lerch transcendent [-1,1,j+1]
            lerch_phi = .5 * (polygamma_asymptotic(0,1+.5*j,terms=1) - polygamma_asymptotic(0,.5*(1+j),terms=1))
            result = power_minus_1(j) * lerch_phi - mp.log(2)
    elif l < -1:
        m = abs(l)
        if abs(j) < 25:
            result = mp.polylog(m,-1) + power_minus_1(j) * 2**(-m) * (
                mp.zeta(m,(j+1)/2) - mp.zeta(m,1+.5*j)
            )
        else: 
        # Approximate with asymptotic series expansion
            zeta_1 = polygamma_asymptotic(m-1,(j+1)/2,terms=1)
            zeta_2 = polygamma_asymptotic(m-1,1+.5*j,terms=1)
            result = mp.polylog(m,-1) + power_minus_1(j) * 2**(-m) * (-1)**m / mp.factorial(m-1)* (
                zeta_1 - zeta_2
            )
    else:
        def func(i):
            return (1 / i**abs(l))
        if l < 0:
            alternating_sum = True
        else:
            alternating_sum = False
        result = fractional_finite_sum(func,k_1=j,k_range=k_range,n_k=n_k,alternating_sum=alternating_sum,plot_integrand=plot_integrand,trap=trap)
    return result

def harmonic_number_prime(l,j):
    if isinstance(j, (int, np.integer)):
        return harmonic_sum_prime(l,j)
    if l == 1:
        raise ValueError(f"invalid value l = {l} for primed harmonic sum.")
    term1 = mp.zeta(l)
    term2 = mp.zeta(l,1+mp.floor(.5 * j))
    result = term1 - term2
    return result

def harmonic_number_tilde(j,k_range=10,n_k=500,epsilon=.2):
    def stilde(k):
        return (mp.digamma(k + 1) - mp.digamma(1))/k**2
    if isinstance(j, (int, np.integer)):
        result = harmonic_sum_tilde(j)
    else:
        result = fractional_finite_sum(stilde,k_0=1,k_1=j,alternating_sum=True,k_range=k_range,n_k=n_k,epsilon=epsilon)
    return result

# Generate interpolator
nested_harmonic_1_2_interpolation = hp.build_harmonic_interpolator([1,2])
nested_harmonic_1_m2_interpolation = hp.build_harmonic_interpolator([1,-2])
nested_harmonic_2_1_interpolation = hp.build_harmonic_interpolator([2,1])
# Pick interpolation
def nested_harmonic_interpolation(indices):
    indices = tuple(int(i) for i in indices)
    if indices == (1,2):
        return nested_harmonic_1_2_interpolation
    elif indices == (1,-2):
        return nested_harmonic_1_m2_interpolation
    elif indices == (2,1):
        return nested_harmonic_2_1_interpolation
    else:
        raise ValueError(f"Generated table for interpolation of indices = {indices} and include in module.")
    
# @cfg.memory.cache
def nested_harmonic_number(indices, j,interpolation=True,n_k=100,k_range=10,epsilon=1e-1,trap=False):
    """
    Nested harmonic sum for j over indices. If interpolation is true, tabulated values are being used.
    The integration strategy including parameters can be chosen. Currently only handles indices > 0 since
    for an intermediate index < 0 there is an alternating and a non_alternating pieces in the series.
    These can be handled manually
    """
    # Convert to tuple such that the checks always compare the same type
    indices = tuple(int(i) for i in indices)
    
    # Analytical results are faster when not interpolated
    # (92)
    if indices == (1,1):
        result = harmonic_number(1,j)**2 + harmonic_number(2,j)
        result /= 2
    # (93)
    elif indices == (1,1,1):
        result = (harmonic_number(1,j)**3 + 3 * harmonic_number(1,j) * harmonic_number(2,j)
                   + 2 * harmonic_number(3,j))
        result /= 6
        return result
    # Return harmonic number for single index
    elif len(indices) == 1:
        result = harmonic_number(indices[0], j,n_k=n_k,k_range=k_range,trap=trap)
    # Use interpolated tables when no analytical solutions are available
    elif interpolation:
        # interp = hp.build_harmonic_interpolator(indices)
        interp = nested_harmonic_interpolation(indices)
        result = interp(j)
    # Explicitly compute result
    elif isinstance(j, (int, np.integer)):
        m, *rest = indices
        total = 0.0
        for i in range(1, j + 1):
            factor = (1 / i**abs(m)) * ((-1)**i if m < 0 else 1)
            inner_sum = nested_harmonic_number(rest, i,interpolation=False,k_range=k_range,n_k=n_k,epsilon=epsilon,trap=trap)
            total += factor * inner_sum
        result = total
        return total
    else:
        m, *rest = indices
        if m < 0:
            alternating_sum = True
        else:
            alternating_sum = False
        def func(i):
            factor = (1 / i**abs(m)) 
            inner_sum = nested_harmonic_number(rest,i,interpolation=False,k_range=k_range,n_k=n_k,epsilon=epsilon,trap=trap)
            return factor * inner_sum
        result = fractional_finite_sum(func,k_1=j,alternating_sum=alternating_sum, n_k=n_k,k_range=k_range,epsilon=epsilon,trap=trap)
    return result

        
def d_weight(m,k,n):
    """
    Sum of harmonic sum weigths. Equivalent to difference of two harmonic sums. Note that N in 
    Nucl.Phys.B 889 (2014) 351-400 is j + 1 here.
    """
    result = 1/(n+k)**m
    return result 

# Generate the interpolator
gamma_qq_lo_interpolation = hp.build_gamma_interpolator("qq","non_singlet_isovector","vector",evolution_order="LO")
def gamma_qq_lo(j,interpolation=True):
    if interpolation:
        interp = gamma_qq_lo_interpolation 
        return interp(j)
    Nc = 3
    c_f = (Nc**2-1)/(2*Nc)
    t_f = .5
    # Belitsky (4.152)
    result = - c_f * (-4*mp.digamma(j+2)+4*mp.digamma(1)+2/((j+1)*(j+2))+3)
    return result

# Generate the interpolators
gamma_qq_non_singlet_vector_nlo_interpolation = hp.build_gamma_interpolator("qq","non_singlet_isovector","vector",evolution_order="NLO")
gamma_qq_non_singlet_axial_nlo_interpolation = hp.build_gamma_interpolator("qq","non_singlet_isovector","axial",evolution_order="NLO")
gamma_qq_singlet_vector_nlo_interpolation = hp.build_gamma_interpolator("qq","singlet","vector",evolution_order="NLO")
gamma_qq_singlet_axial_nlo_interpolation = hp.build_gamma_interpolator("qq","singlet","axial",evolution_order="NLO")

# Pick correct interpolation
def gamma_qq_nlo_interpolation(moment_type,evolve_type):
    if moment_type != "singlet":
        return gamma_qq_non_singlet_vector_nlo_interpolation if evolve_type == "vector" else gamma_qq_non_singlet_axial_nlo_interpolation
    elif moment_type == "singlet":
        return gamma_qq_singlet_vector_nlo_interpolation if evolve_type == "vector" else gamma_qq_singlet_axial_nlo_interpolation
    else:
        raise ValueError(f"Wrong moment_type {moment_type}")
# @cfg.memory.cache
def gamma_qq_nlo(j,Nf=3,moment_type="non_singlet_isovector",evolve_type="vector",interpolation=True):
    if interpolation:
        # interp = hp.build_gamma_interpolator("qq",moment_type,evolve_type,evolution_order="NLO")
        interp = gamma_qq_nlo_interpolation(moment_type,evolve_type)
        result = interp(j)
        return result
    Nc = 3
    c_f = (Nc**2-1)/(2*Nc)
    t_f = .5
    c_a = Nc

    # Nucl.Phys.B 691 (2004) 129-181
    # Note that N there is j + 1 here
    # p is +: N -> N + 1 -> j + 2
    # m is -: N -> N - 1 -> j
    s_1 = harmonic_number(1,j+1)
    s_2 = harmonic_number(2,j+1)
    s_3_p = harmonic_number(3,j+2)
    s_3_m = harmonic_number(3,j)
    s_m3 = harmonic_number(-3,j+1)
    s_1_p = harmonic_number(1,j+2)
    s_1_m = harmonic_number(1,j)
    s_2_p = harmonic_number(2,j+2)
    s_2_pp = harmonic_number(2,j+3)
    s_2_m = harmonic_number(2,j)
    s_1_m2_p = nested_harmonic_number([1,-2],j+2)
    s_1_m2_m = nested_harmonic_number([1,-2],j)
    s_1_2_p = nested_harmonic_number([1,2],j+2)
    s_1_2_m = nested_harmonic_number([1,2],j)
    s_2_1_p = nested_harmonic_number([2,1],j+2)
    s_2_1_m = nested_harmonic_number([2,1],j)

    if moment_type == "singlet":
        s_1_mm = harmonic_number(1,j-1)
        s_1_pp = harmonic_number(1,j+3)

    # Nucl.Phys.B 688 (2004) 101-134
    # Note different beta function convention
    # so we reverse the sign
    # (3.5)
    term1 = - 4 * c_a * c_f * (2 * s_3_p - 17/24 - 2 * s_m3 - 28/3 * s_1
                            + 151/18 * (s_1_m + s_1_p) + 2 * (s_1_m2_m + s_1_m2_p) - 11/6 * (s_2_m + s_2_p)
                            )
    # Factor 2 because we insert t_f
    term2 = - 8 * c_f * t_f * Nf * (1/12 + 4/3 * s_1 - (11/9 * (s_1_m + s_1_p) - 1/3*(s_2_m + s_2_p)) )
    term3 = - 4 * c_f**2 * (4*s_m3 + 2*s_1 + 2 * s_2 -3/8 + (s_2_m + 2 * s_3_m)
                            - ((s_1_m + s_1_p) + 4 * (s_1_m2_m +s_1_m2_p) + 2 * (s_1_2_m + s_1_2_p) + 2 * (s_2_1_m + s_2_1_p) + (s_3_m + s_3_p))
    )
    
    if moment_type != "singlet":
        if evolve_type == "vector":
            # enforce conservation
            term1+= (7 * c_a *c_f - 14 * c_f**2)
        elif evolve_type == "axial":
            term1 += - 16 * c_f * (c_f - .5 * c_a) * (
                (s_2_m - s_2_p) - (s_3_m - s_3_p)
                - 2 * (s_1_m + s_1_p - 2 * s_1)
            )
        else:
            raise ValueError(f"Wrong evolve_type {evolve_type}")

    elif moment_type == "singlet":
        if evolve_type == "vector":
            # Nucl.Phys.B 691 (2004) 129-181
            # Note different beta function convention
            # so we reverse the sign
            term1 += - 8 * c_f * t_f * Nf * (
                20/9 * (s_1_mm - s_1_m) - (56/9*(s_1_p - s_1_pp) + 8/3 * ((s_2_p - s_2_pp)) )
                + (8*(s_1 - s_1_p)-4 * (s_2 - s_2_p)) - (2 * (s_1_m - s_1_p) + (s_2_m - s_2_p) + 2 * (s_3_m - s_3_p))
            )
        if evolve_type == "axial":
            # Nucl.Phys.B 889 (2014) 351-400
            # Note different beta function convention
            # so we reverse the sign
            # (4.4)
            eta = 1/((j+1)*(j+2))
            d02 = d_weight(2,0,j + 1)
            d03 = d_weight(3,0,j + 1)
            term1 += - 4 * c_f * Nf * (-5 * eta + 3 * eta**2 + 2 * eta**3 + 4 * d02 - 4 * d03)
    else:
        raise ValueError(f"Wrong moment_type {moment_type}")
    
    result = term1 + term2 + term3
    # Nucl.Phys.B 688 (2004) 101-134 and 
    # Nucl.Phys.B 889 (2014) 351-400 
    # defines Mellin moment without factor 1/2
    result*=2

    return result

def gamma_qq(j,Nf=3,moment_type="non_singlet_isovector",evolve_type="vector",evolution_order="LO",interpolation=True):
    """
    Returns conformal qq singlet or non-singlet anomalous dimension for conformal spin-j

    Arguments:
    - j (float): conformal spin
    - moment_type (str. optional): non_singlet_isovector, non_singlet_isoscalar, singlet
    - evolve_type (str. optional): vector or axial
    - evolution_order (str. optional): LO, NLO or NNLO
    - interpolation (bool, optional): Use tabulated values for interpolation (only beyond LO)
    """
    if evolution_order == "LO":
        return gamma_qq_lo(j,interpolation)
    elif evolution_order == "NLO":
        return gamma_qq_nlo(j,Nf,moment_type,evolve_type,interpolation)
    else:
        raise ValueError(f"Wrong evolution_order {evolution_order}")

# Generate the interpolator
gamma_qg_vector_lo_interpolation = hp.build_gamma_interpolator("qg","singlet","vector",evolution_order="LO")
gamma_qg_axial_lo_interpolation = hp.build_gamma_interpolator("qg","singlet","axial",evolution_order="LO")
# Pick the correct interpolation
def gamma_qg_lo_interpolation(evolve_type):
    return gamma_qg_vector_lo_interpolation if evolve_type == "vector" else gamma_qg_axial_lo_interpolation

def gamma_qg_lo(j, Nf=3, evolve_type = "vector",interpolation=True):
    if interpolation:
        interp = gamma_qg_lo_interpolation(evolve_type)
        return interp(j)
    t_f = .5
    # Note additional factor of j/6 at LO (see (K.1) in 0504030)
    if j == 0:
        j = 1e-12
    if evolve_type == "vector":
        result = -24*Nf*t_f*(j**2+3*j+4)/(j*(j+1)*(j+2)*(j+3))
    elif evolve_type == "axial":
        result = -24*Nf*t_f/((j+1)*(j+2))
    else:
        raise ValueError("evolve_type must be axial or vector")
    # Match forward to Wilson anomalous dimension
    result*=j/6
    return result

# Generate the interpolator
gamma_qg_singlet_vector_nlo_interpolation = hp.build_gamma_interpolator("qg","singlet","vector",evolution_order="NLO")
gamma_qg_singlet_axial_nlo_interpolation = hp.build_gamma_interpolator("qg","singlet","axial",evolution_order="NLO")
# Pick the correct interpolation
def gamma_qg_nlo_interpolation(evolve_type):
    return gamma_qg_singlet_vector_nlo_interpolation if evolve_type == "vector" else gamma_qg_singlet_axial_nlo_interpolation

# @cfg.memory.cache
def gamma_qg_nlo(j, Nf=3, evolve_type = "vector",interpolation=True):
    if interpolation:
        # interp = hp.build_gamma_interpolator("qg","singlet",evolve_type,"NLO")
        interp = gamma_qg_nlo_interpolation(evolve_type)
        result = interp(j)
        return result
    Nc = 3
    c_a = Nc
    c_f = (Nc**2-1)/(2*Nc)
    t_f = .5

    s_1 = harmonic_number(1,j+1)
    s_2 = harmonic_number(2,j+1)
    s_1_1_1 = nested_harmonic_number([1,1,1],j+1)
    s_1_1_1_m = nested_harmonic_number([1,1,1],j)
    s_1_1_1_p = nested_harmonic_number([1,1,1],j+2)
    s_1_m2 = nested_harmonic_number([1,-2],j+1)
    s_1_m2_m = nested_harmonic_number([1,-2],j)
    s_1_m2_p = nested_harmonic_number([1,-2],j+2)
    s_1_2 = nested_harmonic_number([1,2],j+1)
    s_1_2_m = nested_harmonic_number([1,2],j)
    s_1_2_p = nested_harmonic_number([1,2],j+2)
    s_2_1 = nested_harmonic_number([2,1],j+1)
    s_2_1_m = nested_harmonic_number([2,1],j)
    s_2_1_p = nested_harmonic_number([2,1],j+2)
    s_1_m = harmonic_number(1,j)
    s_1_mm = harmonic_number(1,j-1)
    s_1_p = harmonic_number(1,j+2)
    s_1_pp = harmonic_number(1,j+3)
    s_2 = harmonic_number(2,j+1)
    s_2_m = harmonic_number(2,j)
    s_2_p = harmonic_number(2,j+2)
    s_2_pp = harmonic_number(2,j+3)
    s_1_1 = nested_harmonic_number([1,1],j+1)
    s_1_1_p = nested_harmonic_number([1,1],j+2)
    s_3 = harmonic_number(3,j+1)
    s_3_m = harmonic_number(3,j)
    s_3_p = harmonic_number(3,j+2)

    s_1_1_pp =nested_harmonic_number([1,1],j+3)
    s_1_m2_pp = nested_harmonic_number([1,-2],j+3)
    s_1_1_1_pp = nested_harmonic_number([1,1,1],j+3)
    s_1_2_pp = nested_harmonic_number([1,2],j+3)
    s_2_1_pp = nested_harmonic_number([2,1],j+3)
    s_3_pp = harmonic_number(3,j+3)


    if evolve_type == "vector":
        # Nucl.Phys.B 691 (2004) 129-181
        # Note different beta function convention
        # so we reverse the sign
        term1 = - 4 * c_a * Nf * (
            20/9 * (s_1_mm - s_1_m) - (2* (s_1_m - s_1_p) + (s_2_m - s_2_p) + 2 * (s_3_m - s_3_p))
            - (218/9*(s_1_p-s_1_pp) + 4 * (s_1_1_p-s_1_1_pp) + 44/3 * (s_2_p - s_2_pp))
            + (27*(s_1-s_1_p) + 4 * (s_1_1 - s_1_1_p) - 7 * (s_2 - s_2_p) - 2 * (s_3 - s_3_p))
            - 2 * ((s_1_m2_m + 4 * s_1_m2_p - 2 * s_1_m2_pp - 3 * s_1_m2) + (s_1_1_1_m + 4 * s_1_1_1_p - 2 * s_1_1_1_pp - 3 * s_1_1_1))
        )
        term2 = - 8 * c_f * t_f * Nf * (
            2 * (5*(s_1_p - s_1_pp ) + 2 * (s_1_1_p - s_1_1_pp) - 2 * (s_2_p - s_2_pp) + (s_3_p - s_3_pp))
            - (43/2 * (s_1-s_1_p) + 4 * (s_1_1 - s_1_1_p) - 7/2 * (s_2 - s_2_p))
            + (7 * (s_1_m - s_1_p) - 1.5 * (s_2_m - s_2_p))
            + 2 * ((s_1_1_1_m + 4 * s_1_1_1_p -2 * s_1_1_1_pp - 3 * s_1_1_1)
                    -(s_1_2_m + 4 * s_1_2_p -2 * s_1_2_pp - 3 * s_1_2)
                    -(s_2_1_m + 4 * s_2_1_p -2 * s_2_1_pp - 3 * s_2_1)
                    +.5 * (s_3_m + 4 * s_3_p -2 * s_3_pp - 3 * s_3))
        )
    elif evolve_type == "axial":
        # Nucl.Phys.B 889 (2014) 351-400
        # Note different beta function convention
        # so we reverse the sign
        d1 = d_weight(1,1,j+1)
        d0 = d_weight(1,0,j+1)
        d02 = d_weight(2,0,j+1)
        d03 = d_weight(3,0,j+1)
        d12 = d_weight(2,1,j+1)
        d13 = d_weight(3,1,j+1)
        Delta_pqg = (2 * d1 - d0)
        s_m2 = harmonic_number(-2,j+1)
        # (4.5)
        term1 = - 8 * c_f * t_f * Nf * (
            2 * Delta_pqg * (s_1_1 - s_2) - 2 * (2 * d0 - d02 - 2 * d1) * s_1
            - 11 * d0 + 9/2 * d02 - d03 + 27/2 * d1 + 4 * d12 - 2 * d13
        )
        term2 = - 4 * c_a * Nf * (
            - 2 * Delta_pqg * (s_m2 + s_1_1) + 4 * (d0 - d1 - d12) * s_1
            + 12 * d0 - d02 - 2 * d03 - 11 * d1 - 12 * d12 - 12 * d13
        )
    else:
        raise ValueError("evolve_type must be axial or vector")
    result = term1 + term2
    # Nucl.Phys.B 889 (2014) 351-400 defines Mellin moment
    # without factor 1/2
    result*=2
    return result

def gamma_qg(j,Nf=3,evolve_type="vector",evolution_order="LO",interpolation=True):
    """
    Returns conformal qg singlet anomalous dimension for conformal spin-j

    Arguments:
    - j (float): conformal spin
    - Nf (int,. optional): Number of active flavors
    - evolve_type (str. optional): vector or axial
    - evolution_order (str. optional): LO, NLO or NNLO
    - interpolation (bool, optional): Use tabulated values for interpolation (only beyond LO)
    """
    if evolution_order == "LO":
        return gamma_qg_lo(j,Nf,evolve_type,interpolation)
    elif evolution_order == "NLO":
        return gamma_qg_nlo(j,Nf,evolve_type,interpolation)
    else:
        raise ValueError(f"Wrong evolution_order {evolution_order}")
    
# Generate the interpolator
gamma_gq_vector_lo_interpolation = hp.build_gamma_interpolator("gq","singlet","vector",evolution_order="LO")
gamma_gq_axial_lo_interpolation = hp.build_gamma_interpolator("gq","singlet","axial",evolution_order="LO")
# Pick the correct interpolation
def gamma_gq_lo_interpolation(evolve_type):
    return gamma_gq_vector_lo_interpolation if evolve_type == "vector" else gamma_gq_axial_lo_interpolation
def gamma_gq_lo(j,evolve_type="vector",interpolation=True):
    if interpolation:
        interp = gamma_gq_lo_interpolation(evolve_type)
        return interp(j)
    Nc = 3
    c_f = (Nc**2-1)/(2*Nc)
    if j == 0:
        j = 1e-12
    if evolve_type == "vector":
        result = -c_f*(j**2+3*j+4)/(3*(j+1)*(j+2))
    elif evolve_type == "axial":
        result = -c_f*j*(j+3)/(3*(j+1)*(j+2))
    else:
        raise ValueError("Type must be axial or vector")
    # Match forward to Wilson anomalous dimension
    result*=6/j
    return result

# Generate the interpolator
gamma_gq_singlet_vector_nlo_interpolation = hp.build_gamma_interpolator("gq","singlet","vector",evolution_order="NLO")
gamma_gq_singlet_axial_nlo_interpolation = hp.build_gamma_interpolator("gq","singlet","axial",evolution_order="NLO")
# Pick the correct interpolation
def gamma_gq_nlo_interpolation(evolve_type):
    return gamma_gq_singlet_vector_nlo_interpolation if evolve_type == "vector" else gamma_gq_singlet_axial_nlo_interpolation

# @cfg.memory.cache
def gamma_gq_nlo(j, Nf=3,evolve_type = "vector",interpolation=True):
    if interpolation:
        # interp = hp.build_gamma_interpolator("gq","singlet",evolve_type,evolution_order="NLO")
        interp = gamma_gq_nlo_interpolation(evolve_type)
        result = interp(j)
        return result

    Nc = 3
    c_a = Nc
    c_f = (Nc**2-1)/(2*Nc)
    t_f = .5

    s_1 = harmonic_number(1,j+1)
    s_2 = harmonic_number(2,j+1)

    s_1_1_1 = nested_harmonic_number([1,1,1],j+1)
    s_1_1_1_m = nested_harmonic_number([1,1,1],j)
    s_1_1_1_mm = nested_harmonic_number([1,1,1],j-1)
    s_1_1_1_p = nested_harmonic_number([1,1,1],j+2)
    s_1_m2 = nested_harmonic_number([1,-2],j+1)
    s_1_m2_m = nested_harmonic_number([1,-2],j)
    s_1_m2_mm = nested_harmonic_number([1,-2],j-1)
    s_1_m2_p = nested_harmonic_number([1,-2],j+2)
    s_1_2 = nested_harmonic_number([1,2],j+1)
    s_1_2_m = nested_harmonic_number([1,2],j)
    s_1_2_mm = nested_harmonic_number([1,2],j-1)
    s_1_2_p = nested_harmonic_number([1,2],j+2)
    s_2_1 = nested_harmonic_number([2,1],j+1)
    s_2_1_m = nested_harmonic_number([2,1],j)
    s_2_1_mm = nested_harmonic_number([2,1],j-1)
    s_2_1_p = nested_harmonic_number([2,1],j+2)
    s_1_m = harmonic_number(1,j)
    s_1_mm = harmonic_number(1,j-1)
    s_1_p = harmonic_number(1,j+2)
    s_1_pp = harmonic_number(1,j+3)
    s_2 = harmonic_number(2,j+1)
    s_2_m = harmonic_number(2,j)
    s_2_p = harmonic_number(2,j+2)
    s_2_pp = harmonic_number(2,j+3)
    s_1_1 = nested_harmonic_number([1,1],j+1)
    s_1_1_m = nested_harmonic_number([1,1],j)
    s_1_1_mm = nested_harmonic_number([1,1],j-1)
    s_1_1_p = nested_harmonic_number([1,1],j+2)
    s_3 = harmonic_number(3,j+1)
    s_3_m = harmonic_number(3,j)
    s_3_p = harmonic_number(3,j+2)

    if evolve_type == "vector":
        # Nucl.Phys.B 691 (2004) 129-181
        # Note different beta function convention
        # so we reverse the sign
        term1 = - 4 * c_a * c_f * (
            2* ((2 * s_1_1_1_mm - 4 * s_1_1_1_m - s_1_1_1_p + 3 *s_1_1_1)
                - (2 * s_1_m2_mm - 4 * s_1_m2_m - s_1_m2_p + 3 *s_1_m2) 
                - (2 * s_1_2_mm - 4 * s_1_2_m - s_1_2_p + 3 *s_1_2)
                - (2 * s_2_1_mm - 4 * s_2_1_m - s_2_1_p + 3 *s_2_1))
            + (2 * (s_1 - s_1_p) - 13 * (s_1_1 - s_1_1_p) - 7 * (s_2 - s_2_p) - 2 * (s_3 - s_3_p))
            + (s_1_mm - 2 * s_1_m + s_1_p) - 22/3 * (s_1_1_mm - 2 * s_1_1_m + s_1_1_p)
            + 4 * (7/9 * (s_1_m - s_1_p) + 3 * (s_2_m - s_2_p) + (s_3_m - s_3_p))
            + (44/9 * (s_1_p - s_1_pp) + 8/3 * (s_2_p - s_2_pp)) 
        )
        term2 = - 8 * c_f * t_f * Nf * (
            (4/3 * (s_1_1_mm - 2 * s_1_1_m + s_1_1_p) -20/9 * (s_1_mm - 2 * s_1_m + s_1_p) )
            - (4 * (s_1 - s_1_p) - 2 * (s_1_1 - s_1_1_p))
        )
        term3 = - 4 * c_f**2 *(
            3 * (2 * s_1_1_mm - 4 * s_1_1_m - s_1_1_p + 3 * s_1_1) - 2 * (2 * s_1_1_1_mm - 4 * s_1_1_1_m - s_1_1_1_p + 3 * s_1_1_1)
            - ((s_1 - s_1_p) - 2 * (s_1_1 - s_1_1_p) + 1.5 * (s_2 - s_2_p) - 3 * (s_3 - s_3_p))
            - (5/2 * (s_1_m - s_1_p) + 2 * (s_2_m - s_2_p) + 2 * (s_3_m - s_3_p))
        )
    elif evolve_type == "axial": 
        # Nucl.Phys.B 889 (2014) 351-400
        # Note different beta function convention
        # so we reverse the sign
        d1 = d_weight(1,1,j+1)
        d0 = d_weight(1,0,j+1)
        d02 = d_weight(2,0,j+1)
        d03 = d_weight(3,0,j+1)
        d12 = d_weight(2,1,j+1)
        d13 = d_weight(3,1,j+1)
        s_m2 = harmonic_number(-2,j+1)
        Delta_pgq = (2 * d0 - d1)
        # (4.6)
        term1 = - 16/9 * c_f * t_f * Nf * (
            3 * Delta_pgq * s_1 - 4 * d0 - d1 - 3 * d12
        )
        term2 = - 4 * c_f**2 * (
            - Delta_pgq * (2* s_1_1 - s_1) + 2 *(d1 + d12) * s_1
            -17/2 * d0 + 2 * d02 + 2 * d03 + 4 * d1 + .5 * d12 + d13
        )
        term3 = - 4 * c_a * c_f *(
            2* Delta_pgq * (s_1_1 - s_m2 - s_2) - (10/3 * d0 + 4 * d02 + 1/3 * d1) * s_1
            + 41/9 * d0 - 4 * d02 + 4 * d03 + 35/9 * d1 + 38/3 * d12 + 6 * d13
        )
    else:
        raise ValueError("Type must be axial or vector")
    result = term1 + term2 + term3
    # Nucl.Phys.B 889 (2014) 351-400 defines Mellin moment
    # without factor 1/2
    result*=2
    return result

def gamma_gq(j,Nf=3,evolve_type="vector",evolution_order="LO",interpolation=True):
    """
    Returns conformal gq singlet anomalous dimension for conformal spin-j

    Arguments:
    - j (float): conformal spin
    - Nf (int,. optional): Number of active flavors
    - evolve_type (str. optional): vector or axial
    - evolution_order (str. optional): LO, NLO or NNLO
    - interpolation (bool, optional): Use tabulated values for interpolation (only beyond LO)
    """
    if evolution_order == "LO":
        return gamma_gq_lo(j,evolve_type,interpolation)
    elif evolution_order == "NLO":
        return gamma_gq_nlo(j,Nf,evolve_type,interpolation)
    else:
        raise ValueError(f"Wrong evolution_order {evolution_order}")
# Generate the interpolator
gamma_gg_vector_lo_interpolation = hp.build_gamma_interpolator("gg","singlet","vector",evolution_order="LO")
gamma_gg_axial_lo_interpolation = hp.build_gamma_interpolator("gg","singlet","axial",evolution_order="LO")
# Pick the correct interpolation
def gamma_gg_lo_interpolation(evolve_type):
    return gamma_gg_vector_lo_interpolation if evolve_type == "vector" else gamma_gg_axial_lo_interpolation

def gamma_gg_lo(j,Nf=3,evolve_type="vector",interpolation=True):
    if interpolation:
        interp = gamma_gg_lo_interpolation(evolve_type)
        return interp(j)
    Nc = 3
    c_a = Nc
    beta_0 = 2/3* Nf - 11/3 * Nc
    if j == 0:
        j = 1e-12
    if evolve_type == "vector":
        result = -c_a*(-4*mp.digamma(j+2)+4*mp.digamma(1)+8*(j**2+3*j+3)/(j*(j+1)*(j+2)*(j+3))-beta_0/c_a)
    elif evolve_type == "axial":
        result = -c_a*(-4*mp.digamma(j+2)+4*mp.digamma(1)+8/((j+1)*(j+2))-beta_0/c_a)
    else:
        raise ValueError("Type must be axial or vector")
    return result

# Generate the interpolator
gamma_gg_singlet_vector_nlo_interpolation = hp.build_gamma_interpolator("gg","singlet","vector",evolution_order="NLO")
gamma_gg_singlet_axial_nlo_interpolation = hp.build_gamma_interpolator("gg","singlet","axial",evolution_order="NLO")
# Pick the correct interpolation
def gamma_gg_nlo_interpolation(evolve_type):
    return gamma_gg_singlet_vector_nlo_interpolation if evolve_type == "vector" else gamma_gg_singlet_axial_nlo_interpolation
# @cfg.memory.cache
def gamma_gg_nlo(j, Nf = 3, evolve_type = "vector",interpolation=True):
    if interpolation:
        # interp = hp.build_gamma_interpolator("gg","singlet",evolve_type,evolution_order="NLO")
        interp = gamma_gg_nlo_interpolation(evolve_type)
        result = interp(j)
        return result
    
    Nc = 3
    c_a = Nc
    c_f = (Nc**2-1)/(2*Nc)
    t_f = .5

    s_1 = harmonic_number(1,j+1)
    s_1_m2 = nested_harmonic_number([1,-2],j+1)
    s_1_m2_m = nested_harmonic_number([1,-2],j)
    s_1_m2_mm = nested_harmonic_number([1,-2],j-1)
    s_1_m2_p = nested_harmonic_number([1,-2],j+2)
    s_1_2 = nested_harmonic_number([1,2],j+1)
    s_1_2_m = nested_harmonic_number([1,2],j)
    s_1_2_mm = nested_harmonic_number([1,2],j-1)
    s_1_2_p = nested_harmonic_number([1,2],j+2)
    s_2_1 = nested_harmonic_number([2,1],j+1)
    s_2_1_m = nested_harmonic_number([2,1],j)
    s_2_1_mm = nested_harmonic_number([2,1],j-1)
    s_2_1_p = nested_harmonic_number([2,1],j+2)
    s_1_m = harmonic_number(1,j)
    s_1_mm = harmonic_number(1,j-1)
    s_1_p = harmonic_number(1,j+2)
    s_1_pp = harmonic_number(1,j+3)
    s_2 = harmonic_number(2,j+1)
    s_2_m = harmonic_number(2,j)
    s_2_p = harmonic_number(2,j+2)
    s_2_pp = harmonic_number(2,j+3)
    s_3 = harmonic_number(3,j+1)
    s_3_m = harmonic_number(3,j)
    s_3_p = harmonic_number(3,j+2)
    s_1_m2_pp = nested_harmonic_number([1,-2],j+3)
    s_1_2_pp = nested_harmonic_number([1,2],j+3)
    s_2_1_pp = nested_harmonic_number([2,1],j+3)
    s_m3 = harmonic_number(-3,j+1)
    s_3_pp = harmonic_number(3,j+3)

    if evolve_type == "vector":
        # Nucl.Phys.B 691 (2004) 129-181
        # Note different beta function convention
        # so we reverse the sign
        term1 = - 4 * c_a * Nf * (
            2/3 - 16/3 * s_1 -23/9 * (s_1_mm + s_1_pp) + 14/3 * (s_1_m + s_1_p) + 2/3 * (s_2_m - s_2_p)
        )
        term2 = - 4 * c_a**2 * (
            + 2 * s_m3 - 8/3  - 14/3 * s_1 + 2 * s_3 
            - 4 * ((s_1_m2_mm - 2 * s_1_m2_m - 2 * s_1_m2_p + s_1_m2_pp + 3 * s_1_m2)
                    + (s_1_2_mm - 2 * s_1_2_m - 2 * s_1_2_p + s_1_2_pp + 3 * s_1_2)
                    + (s_2_1_mm - 2 * s_2_1_m - 2 * s_2_1_p + s_2_1_pp + 3 * s_2_1)
                )
            + 8/3 * (s_2_p - s_2_pp) - 4  * (3 * (s_2_m - 3 * s_2_p + s_2_pp + s_2) -  (s_3_m - 3 * s_3_p + s_3_pp + s_3)) 
            + 109/18 * (s_1_m + s_1_p) + 61/3 * (s_2_m - s_2_p)
        )
        term3 = - 8 * c_f * t_f * Nf * (
            .5 + 2/3 * (s_1_mm - 13 * s_1_m - s_1_p - 5 * s_1_pp + 18 * s_1)
            + (3 * s_2_m - 5 * s_2_p + 2 * s_2) - 2 * (s_3_m - s_3_p)
        )
    elif evolve_type == "axial":
        # Nucl.Phys.B 889 (2014) 351-400
        # Note different beta function convention
        # so we reverse the sign
        d02 = d_weight(2,0,j+1)
        d03 = d_weight(3,0,j+1)
        s_m2 = harmonic_number(-2,j+1)
        eta = 1/((j+1)*(j+2))
        # (4.7)
        term1 = - 8 * c_f * t_f * Nf * (
            -.5 - 7 * eta + 5 * eta**2 + 2 * eta**3 + 6 * d02 - 4 * d03
        )
        term2 = - 4/3 * c_a * Nf *(
            10/3 * s_1 - 2 - 26/3 * eta + 2 * eta**2
        )
        term3 = -  4 * c_a**2 * (
            4 * (s_1_m2 + s_1_2 + s_2_1) - 2 * (s_3 +s_m3) - 67/9 * s_1 + 8/3
            - 8 * eta * (s_2 + s_m2) + 8 * (2 * eta + eta**2 - 2 * d02) * s_1
            + 901/18 * eta - 149/3 * eta**2 -24 * eta**3 - 32 * (d02 - d03)
        )
    else:
        raise ValueError("Type must be axial or vector")
    result = term1 + term2 + term3
    # Nucl.Phys.B 889 (2014) 351-400 defines Mellin moment
    # without factor 1/2
    result*=2
    return result

def gamma_gg(j,Nf=3,evolve_type="vector",evolution_order="LO",interpolation=True):
    """
    Returns conformal gg singlet anomalous dimension for conformal spin-j

    Arguments:
    - j (float): conformal spin
    - Nf (int,. optional): Number of active flavors
    - evolve_type (str. optional): vector or axial
    - evolution_order (str. optional): LO, NLO or NNLO
    - interpolation (bool, optional): Use tabulated values for interpolation (only beyond LO)
    """
    if evolution_order == "LO":
        return gamma_gg_lo(j,Nf,evolve_type,interpolation)
    elif evolution_order == "NLO":
        return gamma_gg_nlo(j,Nf,evolve_type,interpolation)
    else:
        raise ValueError(f"Wrong evolution_order {evolution_order}")

def power_minus_1(j):
    if mp.im(j) < 0:
        result = mp.mpc(-1,0)**(mp.conj(j))
        return mp.conj(result)
    else:
        result = mp.mpc(-1,0)**j
        return result

def d_element(j,k):
    """ Belistky (4.204)"""
    if j == k:
        raise ValueError("j and k must be unequal")
    result = - .5 * (1 + power_minus_1(j-k)) * (2 * k + 3)/((j - k)*(j + k + 3))
    return result

def digamma_A(j,k):
    """ Belistky (4.212)"""
    if j == k:
        raise ValueError("j and k must be unqual")
    result = mp.digamma(.5* (j + k + 4)) - mp.digamma(.5 * (j-k)) + 2 * mp.digamma(j - k) - mp.digamma(j + 2) - mp.digamma(1)
    return result

def heaviside_theta(j,k):
    """Returns 1 if j > k and 0 otherwise.
    """
    if j.imag == 0 and k.imag == 0:
        return int(j.real > k.real)
    # smooth out
    eps = 0.1
    jmk = j - k
    # Use Fermi-Dirac distribution to approximate
    re = 1/(1+mp.exp(-jmk.real/eps))
    im = 1/(1+mp.exp(-jmk.imag/eps))
    return re * im
    
def nd_projector(j,k):
    """
    Product of (1 + (-1)**(j-k)) * heaviside_theta(j-2,k).
    is 2 when j - k is non-zero and even and 0 otherwise.
    """
    # result = (1 + (-1)**(j.real - k.real)) * heaviside_theta(j.real-2,k.real)
    # return result
    jmk = j - k
    return 2 if int(jmk.real) % 2 == 0 else 0

def conformal_anomaly_qq(j,k):
    """Belitsky (4.206). Equal for vector and axial """
    if j == k:
        raise ValueError("j and k must be unqual")
    Nc = 3
    c_f = (Nc**2-1)/(2*Nc)
    result =  -c_f * nd_projector(j,k) * ((3 + 2 * k) / ((j - k) * (j + k + 3))) * (
        2 * digamma_A(j, k) + 
        (digamma_A(j, k) - mp.digamma(j + 2) + mp.digamma(1)) * ((j - k) * (j + k + 3)) / ((k + 1) * (k + 2))
        )
    return result

def conformal_anomaly_gq(j,k):
    """Belitsky (4.208). Equal for vector and axial """
    if j == k:
        raise ValueError("j and k must be unqual")
    Nc = 3
    c_f = (Nc**2-1)/(2*Nc)
    result = -c_f * nd_projector(j,k) * (1 / 6) * ((3 + 2 * k) / ((k + 1) * (k + 2)))
    return result

def conformal_anomaly_gg(j,k):
    """Belitsky (4.209). Equal for vector and axial """
    if j == k:
        raise ValueError("j and k must be unqual")
    Nc = 3
    c_a = Nc
    result = (
            -c_a * nd_projector(j,k) *
            ((3 + 2 * k) / ((j - k) * (j + k + 3))) *
            (
                2 * digamma_A(j,k) +
                (digamma_A(j,k) - mp.digamma(j + 2) + mp.digamma(1)) *
                ((mp.gamma(j + 4) * mp.gamma(k)) / (mp.gamma(j) * mp.gamma(k + 4)) - 1) +
                2 * (j - k) * (j + k + 3) * (mp.gamma(k) / mp.gamma(k + 4))
            )
        )
    return result

def gamma_qq_nd(j,k, Nf=3, evolve_type = "vector",evolution_order="LO",interpolation=True):
    """ Belistky (4.203)"""
    if evolution_order == "LO":
        return 0
    if isinstance(j, (int)) and  isinstance(k, (int)) and k >= j:
        return 0
    Nc = 3
    beta_0 = 2/3* Nf - 11/3 * Nc
    
    if evolution_order == "NLO":
        term1 = (gamma_qq(j,evolution_order="LO",interpolation=interpolation)-gamma_qq(k,evolution_order="LO",interpolation=interpolation))* \
                (d_element(j,k) * (beta_0 - gamma_qq(k,evolution_order="LO",interpolation=interpolation)) + conformal_anomaly_qq(j,k))
        term2 = - (gamma_qg(j,Nf,evolve_type,"LO",interpolation=interpolation) - 
                   gamma_qg(k,Nf,evolve_type,evolution_order="LO",interpolation=interpolation)) * d_element(j,k) * gamma_gq(j,Nf,evolve_type,evolution_order="LO",interpolation=interpolation)
        term3 = gamma_qg(j,Nf,evolve_type,"LO",interpolation=interpolation) * conformal_anomaly_gq(j,k)
        result = term1 + term2 + term3
    else:
        raise ValueError(f"Currently unsupported evolution order {evolution_order}")
    return result

def gamma_qg_nd(j,k, Nf=3, evolve_type = "vector",evolution_order="LO",interpolation=True):
    """ Belistky (4.203)"""
    if evolution_order == "LO":
        return 0
    if isinstance(j, (int)) and  isinstance(k, (int)) and k >= j:
        return 0
    Nc = 3
    beta_0 = 2/3* Nf - 11/3 * Nc
    
    if evolution_order == "NLO":
        term1 = (gamma_qg(j, Nf, evolve_type, "LO",interpolation=interpolation) - gamma_qg(k, Nf, evolve_type, "LO",interpolation=interpolation)) * \
                d_element(j, k) * (beta_0 - gamma_gg(k, Nf, evolve_type, "LO",interpolation=interpolation))
        term2 = - (gamma_qq(j, evolution_order="LO",interpolation=interpolation) - gamma_qq(k, evolution_order="LO",interpolation=interpolation)) * \
                d_element(j, k) * gamma_qg(k, Nf, evolve_type, "LO",interpolation=interpolation)
        term3 = gamma_qg(j, Nf, evolve_type, evolution_order="LO") * conformal_anomaly_gg(j, k)
        term4 = - conformal_anomaly_qq(j, k) * gamma_qg(k, Nf, evolve_type, evolution_order="LO",interpolation=interpolation)
        result = term1 + term2 + term3 + term4
    else:
        raise ValueError(f"Currently unsupported evolution order {evolution_order}")
    return result

def gamma_gq_nd(j,k, Nf=3, evolve_type = "vector",evolution_order="LO",interpolation=True):
    """ Belistky (4.203)"""
    if evolution_order == "LO":
        return 0
    if isinstance(j, (int)) and  isinstance(k, (int)) and k >= j:
        return 0
    Nc = 3
    beta_0 = 2/3* Nf - 11/3 * Nc
    
    if evolution_order == "NLO":
        term1 = (gamma_gq(j, Nf, evolve_type, evolution_order="LO",interpolation=interpolation) - gamma_gq(k, Nf, evolve_type, evolution_order="LO",interpolation=interpolation)) * \
                d_element(j, k) * (beta_0 - gamma_qq(k, evolution_order="LO",interpolation=interpolation))
        term2 = - (gamma_gg(j, Nf, evolve_type, evolution_order="LO",interpolation=interpolation) - gamma_gg(k, Nf, evolve_type, evolution_order="LO",interpolation=interpolation)) * \
                d_element(j, k) * gamma_gq(k, Nf, evolve_type, evolution_order="LO",interpolation=interpolation)
        term3 = gamma_gq(j, Nf, evolve_type, evolution_order="LO") * conformal_anomaly_qq(j, k)
        term4 = - conformal_anomaly_gg(j, k) * gamma_gq(k, Nf, evolve_type, evolution_order="LO",interpolation=interpolation)
        term5 = (gamma_gg(j, Nf, evolve_type, evolution_order="LO",interpolation=interpolation) - gamma_qq(k,evolution_order="LO",interpolation=interpolation)) * conformal_anomaly_gq(j, k)

        result = term1 + term2 + term3 + term4 + term5
    else:
        raise ValueError(f"Currently unsupported evolution order {evolution_order}")
    return result

def gamma_gg_nd(j,k, Nf=3, evolve_type = "vector",evolution_order="LO",interpolation=True):
    """ Belistky (4.203)"""
    if evolution_order == "LO":
        return 0
    if isinstance(j, (int)) and  isinstance(k, (int)) and k >= j:
        return 0
    Nc = 3
    beta_0 = 2/3* Nf - 11/3 * Nc
    
    if evolution_order == "NLO":
        term1 = (gamma_gg(j, Nf, evolve_type, evolution_order="LO",interpolation=interpolation) - gamma_gg(k, Nf, evolve_type, evolution_order="LO",interpolation=interpolation)) * \
                (d_element(j, k) * (beta_0 - gamma_gg(k, Nf, evolve_type, "LO",interpolation=interpolation)) + conformal_anomaly_gg(j, k))
        term2 = - (gamma_gq(j, Nf, evolve_type, evolution_order="LO",interpolation=interpolation) - gamma_gq(k, Nf, evolve_type, evolution_order="LO",interpolation=interpolation)) * \
                d_element(j, k) * gamma_qg(k, Nf, evolve_type, evolution_order="LO",interpolation=interpolation)
        term3 = - conformal_anomaly_gq(j, k) * gamma_qg(k, Nf, evolve_type, evolution_order="LO",interpolation=interpolation)
        result = term1 + term2 + term3
    else:
        raise ValueError(f"Currently unsupported evolution order {evolution_order}")
    return result

def gamma_pm(j, Nf = 3, evolve_type = "vector",solution="+",interpolation=True):
    """ Compute the (+) and (-) eigenvalues of the LO evolution equation of the coupled singlet quark and gluon GPD
    Arguments:
    - j: conformal spin,
    - evolve_type: "vector" or "axial"
    - Nf: Number of active flavors (default Nf = 3 )
    - interpolation (bool, optional): Use interpolated values
    Returns:
    The eigenvalues (+) and (-) in terms of an array
    """
    # Check evolve_type
    hp.check_evolve_type(evolve_type)

    base = gamma_qq(j,evolution_order="LO",interpolation=interpolation)+gamma_gg(j,Nf,evolve_type,evolution_order="LO",interpolation=interpolation)
    root = mp.sqrt((gamma_qq(j,evolution_order="LO",interpolation=interpolation)-gamma_gg(j,Nf,evolve_type,evolution_order="LO",interpolation=interpolation))**2
                   +4*gamma_gq(j,Nf,evolve_type,evolution_order="LO",interpolation=interpolation)*gamma_qg(j,Nf,evolve_type,evolution_order="LO",interpolation=interpolation))

    if solution == "+":
        return (base + root)/2
    elif solution == "-":
        return (base - root)/2
    else:
        raise ValueError("Invalid solution evolve_type. Use '+' or '-'.")

def R_qq(j,Nf=3,evolve_type="vector",interpolation=True):
    Nc = 3
    c_a = Nc
    c_f = (Nc**2-1)/(2*Nc)
    beta_0 = 2/3* Nf - 11/3 * Nc
    beta_1 = 10/3 * c_a * Nf + 2 * c_f * Nf -34/3 * c_a**2
    term1 = gamma_qq(j,Nf,"singlet",evolve_type,evolution_order="NLO",interpolation=interpolation)
    term2 = - .5 * beta_1/beta_0 * gamma_qq(j,Nf,"singlet",evolve_type,evolution_order="LO",interpolation=interpolation)
    result = term1 + term2
    return result

def R_qg(j,Nf=3,evolve_type="vector",interpolation=True):

    Nc = 3
    c_a = Nc
    c_f = (Nc**2-1)/(2*Nc)
    beta_0 = 2/3* Nf - 11/3 * Nc
    beta_1 = 10/3 * c_a * Nf + 2 * c_f * Nf -34/3 * c_a**2
    term1 = gamma_qg(j,Nf,evolve_type,"NLO",interpolation=interpolation)
    term2 = - .5 * beta_1/beta_0 * gamma_qg(j,Nf,evolve_type,"LO",interpolation=interpolation)
    result = term1 + term2
    return result

def R_gq(j,Nf=3,evolve_type="vector",interpolation=True):

    Nc = 3
    c_a = Nc
    c_f = (Nc**2-1)/(2*Nc)
    beta_0 = 2/3* Nf - 11/3 * Nc
    beta_1 = 10/3 * c_a * Nf + 2 * c_f * Nf -34/3 * c_a**2
    term1 = gamma_gq(j,Nf,evolve_type,"NLO",interpolation=interpolation)
    term2 = - .5 * beta_1/beta_0 * gamma_gq(j,Nf,evolve_type,"LO",interpolation=interpolation)
    result = term1 + term2
    return result

def R_gg(j,Nf=3,evolve_type="vector",interpolation=True):

    Nc = 3
    c_a = Nc
    c_f = (Nc**2-1)/(2*Nc)
    beta_0 = 2/3* Nf - 11/3 * Nc
    beta_1 = 10/3 * c_a * Nf + 2 * c_f * Nf -34/3 * c_a**2
    term1 = gamma_gg(j,Nf,evolve_type,"NLO",interpolation=interpolation)
    term2 = - .5 * beta_1/beta_0 * gamma_gg(j,Nf,evolve_type,"LO",interpolation=interpolation)
    result = term1 + term2
    return result

evolve_moment_interpolation = {}
if cfg.INTERPOLATE_MOMENTS:
    for particle,moment_type, moment_label, evolution_order, error_type in product(
        ["quark","gluon"], cfg.MOMENTS, cfg.LABELS, cfg.ORDERS, cfg.ERRORS):
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
            "error_type": error_type,
        }
        selected_triples = [
            (eta, t, mu)
            for eta, t, mu in zip(cfg.ETA_ARRAY, cfg.T_ARRAY, cfg.MU_ARRAY)
        ]
        evolve_moment_interpolation[(particle,moment_type,moment_label, evolution_order, error_type)] = [
            hp.build_moment_interpolator(eta, t, mu, **params)
            for eta, t, mu in selected_triples
        ]

@hp.mpmath_vectorize
def evolve_conformal_moment(j,eta,t,mu,Nf=3,A0=1,particle="quark",moment_type="non_singlet_isovector",moment_label ="A", evolution_order = "LO", error_type = "central",interpolation=True):
    """
    Evolve the conformal moment F_{j}^{+-} from some input scale mu_in to some other scale mu. 

    Arguments:
    - j (float): conformal spin
    - eta (float): skewness parameter
    - t (float): Mandelstam t
    - mu (float): Resolution scale
    - Nf (int. optional): Number of active flavors (default Nf = 3)
    - A0 (float optional): Normalization factor (default A0 = 1)
    - particle (str. optional): quark or gluon
    - moment_type (str. optional): non_singlet_isovector, non_singlet_isoscalar, or singlet
    - moment_label (str. optional): A(Tilde) B(Tilde) depending on H(Tilde) or E(Tilde) GPD etc.
    - evolution_order (str. optional): LO, NLO
    - error_type (str. optional): Choose central, upper or lower value for input PDF parameters
    - interpolation (bool, optional): Use interpolated values for anomalous dimensions

    Returns:
    The value of the evolved conformal moment at scale mu
    """
    
    hp.check_particle_type(particle)
    hp.check_moment_type_label(moment_type,moment_label)
    hp.check_error_type(error_type)
    hp.check_evolution_order(evolution_order)
    if particle == "gluon" and moment_type != "singlet":
        raise ValueError("Gluon is only singlet")
    
    if cfg.INTERPOLATE_MOMENTS and isinstance(j,(complex,mp.mpc)):
        selected_triples = [
            (eta_, t_, mu_)
            for eta_, t_, mu_ in zip(cfg.ETA_ARRAY, cfg.T_ARRAY, cfg.MU_ARRAY)
        ]
        index = selected_triples.index((eta, t, mu))
        key = (particle,moment_type,moment_label,evolution_order,error_type)
        interp = evolve_moment_interpolation[key][index]
        return interp(j)

    # Set parameters
    Nc = 3
    c_a = Nc
    c_f = (Nc**2-1)/(2*Nc)
    beta_0 = 2/3* Nf - 11/3 * Nc
    beta_1 = 10/3 * c_a * Nf + 2 * c_f * Nf -34/3 * c_a**2
    # Extract fixed quantities
    alpha_s_in = get_alpha_s(evolution_order)
    alpha_s_evolved = evolve_alpha_s(mu,Nf,evolution_order)

    if moment_label in ["A","B"]:
        evolve_type = "vector"
    elif moment_label in ["Atilde","Btilde"]:
        evolve_type = "axial"

    if moment_type == "non_singlet_isovector":
        moment_in = non_singlet_isovector_moment(j,eta,t,moment_label,evolve_type,evolution_order,error_type)
    elif moment_type == "non_singlet_isoscalar":
        moment_in = non_singlet_isoscalar_moment(j,eta,t,moment_label,evolve_type,evolution_order,error_type)

    # moment_in, evolve_type = MOMENT_TO_FUNCTION.get((moment_type, moment_label))

    ga_qq = gamma_qq(j-1,Nf,moment_type,evolve_type,evolution_order="LO",interpolation=interpolation)

    if moment_type == "singlet":
        # Roots  of LO anomalous dimensions
        ga_p = gamma_pm(j-1,Nf,evolve_type,"+",interpolation=interpolation)
        ga_m = gamma_pm(j-1,Nf,evolve_type,"-",interpolation=interpolation)
        moment_in_p, error_p = singlet_moment(j,eta,t, Nf, moment_label, evolve_type,"+",evolution_order,error_type,interpolation=interpolation)
        moment_in_m, error_m = singlet_moment(j,eta,t, Nf, moment_label, evolve_type,"-",evolution_order,error_type,interpolation=interpolation)
        ga_gq = gamma_gq(j-1,Nf, evolve_type,"LO",interpolation=interpolation)
        ga_qg = gamma_qg(j-1,Nf, evolve_type,"LO",interpolation=interpolation)
        if evolution_order != "LO":
            ga_gg = gamma_gg(j-1,Nf,evolve_type,"LO",interpolation=interpolation)
            r_qq = R_qq(j-1,Nf,evolve_type,interpolation=interpolation)
            r_qg = R_qg(j-1,Nf,evolve_type,interpolation=interpolation)
            r_gq = R_gq(j-1,Nf,evolve_type,interpolation=interpolation)
            r_gg = R_gg(j-1,Nf,evolve_type,interpolation=interpolation) 

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

    def E_non_singlet_lo():
        return alpha_frac**(ga_qq/beta_0)

    def E_non_singlet_nlo(j):
        ga_qq = gamma_qq(j,Nf,moment_type,evolve_type,"LO",interpolation=interpolation)
        ga_qq_nlo = gamma_qq(j,Nf,moment_type,evolve_type,"NLO",interpolation=interpolation)
        result = alpha_frac**(ga_qq/beta_0) * (1 + (.5 * beta_1/beta_0**2 *  ga_qq - ga_qq_nlo/beta_0) * \
                                      (alpha_s_evolved - alpha_s_in)/(2*np.pi)
                                      )
        return result
    def B_non_singlet_nlo(k):
        gamma_term = (ga_qq - gamma_qq(k,Nf,moment_type,evolve_type,"LO",interpolation=interpolation) + beta_0)
        ga_nd = gamma_qq_nd(j-1,k,Nf,evolve_type,evolution_order,interpolation=interpolation)
        result = alpha_s_evolved/(2*np.pi) * ga_nd/gamma_term * (
            1 - alpha_frac**(gamma_term/beta_0)
        )
        if j - 1 == k:
            result += 1
        return result

    def EB_non_singlet_nlo(k):
        # Combinede function to call in fractional_finite_sum
        if moment_type == "non_singlet_isovector":
            moment_k  = non_singlet_isovector_moment(k,eta,t,moment_label,evolve_type,evolution_order,error_type)
        else:
            moment_k  = non_singlet_isoscalar_moment(k,eta,t,moment_label,evolve_type,evolution_order,error_type)
        non_diagonal_terms = eta**(j - k) * E_non_singlet_nlo(k-1) * B_non_singlet_nlo(k-1)
        non_diagonal_terms = non_diagonal_terms * moment_k
        return non_diagonal_terms
    
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

    def prf_T_nlo(k):
        ga_j_p, ga_j_m = ga_p, ga_m
        ga_k_p, ga_k_m = gamma_pm(k-1,Nf,evolve_type,"+",interpolation=interpolation), gamma_pm(k-1,Nf,evolve_type,"-",interpolation=interpolation)
        alpha_term = alpha_s_evolved/(2*mp.pi)
        ga_1 = ga_j_p - ga_k_p + beta_0
        ga_2 = ga_j_p - ga_k_m + beta_0
        ga_3 = ga_j_m - ga_k_p + beta_0
        ga_4 = ga_j_m - ga_k_m + beta_0
        ga_kk_jj = (ga_k_p - ga_k_m)*(ga_j_p - ga_j_m)
        prf_T_1 = - alpha_term/ga_1 * (1 - alpha_frac**(ga_1/beta_0))/ga_kk_jj
        prf_T_2 = - alpha_term/ga_2 * (1 - alpha_frac**(ga_2/beta_0))/ga_kk_jj
        prf_T_3 = - alpha_term/ga_3 * (1 - alpha_frac**(ga_3/beta_0))/ga_kk_jj
        prf_T_4 = - alpha_term/ga_4 * (1 - alpha_frac**(ga_4/beta_0))/ga_kk_jj

        return prf_T_1, prf_T_2, prf_T_3, prf_T_4

    # T1 and T3 go with "+" solution, T2 and T4 go with "-" solution
    def T_quark_nlo(k):
        # Note T = 0 for j=k
        ga_j_p, ga_j_m = ga_p, ga_m
    
        ga_k_p, ga_k_m = gamma_pm(k-1,Nf,evolve_type,"+",interpolation=interpolation), gamma_pm(k-1,Nf,evolve_type,"-",interpolation=interpolation)
        ga_qq_k = gamma_qq(k-1,evolution_order="LO",interpolation=interpolation)
        ga_gq_k = gamma_gq(k-1,Nf, evolve_type,"LO",interpolation=interpolation)
        ga_qq_nd = gamma_qq_nd(j-1,k-1,Nf,evolve_type,"NLO",interpolation=interpolation)
        ga_qg_nd = gamma_qg_nd(j-1,k-1,Nf,evolve_type,"NLO",interpolation=interpolation)
        ga_gq_nd = gamma_gq_nd(j-1,k-1,Nf,evolve_type,"NLO",interpolation=interpolation)
        ga_gg_nd = gamma_gg_nd(j-1,k-1,Nf,evolve_type,"NLO",interpolation=interpolation)

        prf_T_1, prf_T_2, prf_T_3, prf_T_4 = prf_T_nlo(k)

        moment_k_p, error_k_p = singlet_moment(k,eta,t, Nf, moment_label, evolve_type,"+",evolution_order,error_type,interpolation=interpolation)
        moment_k_m, error_k_m = singlet_moment(k,eta,t, Nf, moment_label, evolve_type,"-",evolution_order,error_type,interpolation=interpolation)

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
        sum_squared = (eta**(j-k) * plus_terms * error_k_p)**2 + (eta**(j-k) * minus_terms * error_k_m)**2
        quark_non_diagonal_errors = abs(mp.sqrt(sum_squared))

        return quark_non_diagonal_part, quark_non_diagonal_errors

    def T_gluon_nlo(k):
        # Note T = 0 for j=k
        ga_j_p, ga_j_m = ga_p, ga_m
        ga_k_p, ga_k_m = gamma_pm(k-1,Nf,evolve_type,"+",interpolation=interpolation), gamma_pm(k-1,Nf,evolve_type,"-",interpolation=interpolation)
        ga_qq_k = gamma_qq(k-1,evolution_order="LO",interpolation=interpolation)
        ga_gq_k = gamma_gq(k-1,Nf, evolve_type,"LO",interpolation=interpolation)
        ga_qq_nd = gamma_qq_nd(j-1,k-1,Nf,evolve_type,"NLO",interpolation=interpolation)
        ga_qg_nd = gamma_qg_nd(j-1,k-1,Nf,evolve_type,"NLO",interpolation=interpolation)
        ga_gq_nd = gamma_gq_nd(j-1,k-1,Nf,evolve_type,"NLO",interpolation=interpolation)
        ga_gg_nd = gamma_gg_nd(j-1,k-1,Nf,evolve_type,"NLO",interpolation=interpolation)

        prf_T_1, prf_T_2, prf_T_3, prf_T_4 = prf_T_nlo(k)

        moment_k_p, error_k_p = singlet_moment(k,eta,t, Nf, moment_label, evolve_type,"+",evolution_order,error_type,interpolation=interpolation)
        moment_k_m, error_k_m = singlet_moment(k,eta,t, Nf, moment_label, evolve_type,"-",evolution_order,error_type,interpolation=interpolation)

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
        sum_squared = (eta**(j-k) * plus_terms * error_k_p)**2 + (eta**(j-k) * minus_terms * error_k_m)**2
        gluon_non_diagonal_errors = abs(mp.sqrt(sum_squared))

        return gluon_non_diagonal_part, gluon_non_diagonal_errors

    if moment_type == "singlet":
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
                non_diagonal_terms = 0
                non_diagonal_errors = 0
                # Hard bypass for fractional_finite_sum
                j = int(np.ceil(j.real))
                if isinstance(j, (int, np.integer)) and eta != 0:
                    for k in range(2,j - 2 + 1):
                        non_diagonal_terms += T_quark_nlo(k)[0]
                        non_diagonal_errors += T_quark_nlo(k)[1]
                elif eta != 0:
                    non_diagonal_terms, non_diagonal_errors = fractional_finite_sum(T_quark_nlo,k_0=2,k_1=j - 2 + 1,n_tuple=2)
                error = diagonal_errors + non_diagonal_errors
                result += diagonal_terms + non_diagonal_terms + hp.error_sign(error,error_type)
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
                non_diagonal_terms = 0
                non_diagonal_errors = 0
                # Hard bypass for fractional_finite_sum
                j = int(np.ceil(j.real))
                if isinstance(j, (int, np.integer)) and eta != 0:
                    for k in range(2,j - 2 + 1):
                        non_diagonal_terms += T_gluon_nlo(k)[0]
                        non_diagonal_errors += T_gluon_nlo(k)[1]
                elif eta != 0:
                    non_diagonal_terms, non_diagonal_errors = fractional_finite_sum(T_gluon_nlo,k_0=2,k_1=j - 2 + 1,n_tuple=2)
                error = diagonal_errors + non_diagonal_errors
                result += diagonal_terms + non_diagonal_terms + hp.error_sign(error,error_type)

    elif moment_type in ["non_singlet_isovector","non_singlet_isoscalar"]: 
        if evolution_order == "LO":
            result = moment_in * E_non_singlet_lo()
        elif evolution_order == "NLO":
            result = moment_in * E_non_singlet_nlo(j-1)
            non_diagonal_terms = 0
            # Hard bypass for fractional_finite_sum
            j = int(np.ceil(j.real))
            if isinstance(j, (int, np.integer)) and eta != 0:
                for k in range(1,j - 2 + 1):
                    non_diagonal_terms += EB_non_singlet_nlo(k)
            elif eta != 0:
                non_diagonal_terms = fractional_finite_sum(EB_non_singlet_nlo,k_0=1,k_1=j - 2 + 1)
            result += non_diagonal_terms

    result *= A0
    # Return real value when called for real j
    if mp.im(result) == 0:
        return np.float64(mp.re(result))
    return result

def first_singlet_moment(eta, mu, particle="quark", gpd_label="H",evolution_order="LO", error_type="central",t0_only=None):
    """
    Returns t_values and corresponding first moment of singlet GPD.

    Parameters:
    - eta (float): Skewness parameter
    - mu (float): Resolution scale
    - particle (str. optional): quark or gluon
    - gpd_label (str. optional): A, B, Atilde...
    - error_type (str. optional): central, plus, minus
    - t0_only (bool optional): Only extract value at t=t0
    """
    hp.check_particle_type(particle)

    prefix = f"singlet_{particle}_GPD_{gpd_label}"
    FILE_PATH = os.path.join(cfg.GPD_PATH, "") 
    if not os.path.isdir(FILE_PATH):
        print(f"Directory {FILE_PATH} does not exist. Generate data for parameters (eta,mu)={eta,mu} using plot_gpds")
        return [], []
    
    if t0_only is not None:
        t0 = t0_only
        filename = hp.generate_filename(eta, t0, mu, prefix=prefix, error_type=error_type)
        full_path = FILE_PATH / filename
        if os.path.exists(full_path):
            x_values, y_values = hp.load_gpd_data(
                eta=eta, t=t0, mu=mu, particle=particle,
                gpd_type="singlet", gpd_label=gpd_label,
                evolution_order=evolution_order,
                error_type=error_type
            )
            if x_values is not None:
                if particle == "gluon":
                    # Gluon GPD is x g(x)
                    integrand = y_values/x_values
                else:
                    integrand = y_values
                moment = trapezoid(integrand, x_values)
                return [t0], [moment]
            else:
                print(f"Data for t=0 exists but couldn't be loaded: {filename}")
        else:
            print(f"No t={t0_only} file found: {filename}")
        return [], []

    t_vals = []
    moments = []

    for filename in os.listdir(FILE_PATH):
        if not filename.startswith(prefix):
            continue

        parsed = hp.parse_filename(filename, prefix=prefix)
        if parsed is None:
            continue

        eta_f, t, mu_f, err = parsed
        if eta == eta_f and mu == mu_f and err == error_type:
            x_values, y_values = hp.load_gpd_data(
                eta=eta, t=t, mu=mu, particle=particle,
                gpd_type="singlet", gpd_label=gpd_label,
                evolution_order=evolution_order,
                error_type=error_type
            )
            if x_values is not None:
                if particle == "gluon":
                    # Gluon GPD is x g(x)
                    integrand = y_values/x_values
                else:
                    integrand = y_values
                moment = trapezoid(integrand, x_values)
                t_vals.append(t)
                moments.append(moment)

    # Sort by t
    t_vals, moments = zip(*sorted(zip(t_vals, moments))) if t_vals else ([], [])
    return list(t_vals), list(moments)
    

def first_singlet_moment_dipole(eta,t,mu,Nf=3,particle="gluon",moment_label="Atilde",evolution_order="LO",error_type="central"):
    """
    Dipole parameters obtained from dipole_fit_first_singlet_moment and saved to singlet_particle_dipole_moment_eta_000_mu_error_type.csv

    Parameters:
    - eta (float): Skewness parameter
    - t (float): Mandelstam t
    - mu (float): Resolution scale
    - particle (str. optional): quark, gluon
    - moment_label (str. optional): Atilde,...
    - error_type (str. optional)
    """
    def dipole_form(t, A_D, m_D2): 
        return A_D / (1 - t / m_D2)**2
    
    def parse_csv_params(csv_path):
        params = {}
        with open(csv_path, 'r') as f:
            for line in f:
                label, value = line.strip().split(',')
                params[label] = float(value)
        return params
    def get_dipole_from_csv(t, csv_path):
        params = parse_csv_params(csv_path)
        if 'A_D' not in params or 'mD2' not in params:
            print("Required parameters 'A_D' or 'mD2' not found in CSV - abort")
            return None
        return dipole_form(t, params['A_D'], params['mD2'])
    
    if evolution_order != "LO":
        print("Warning: Currently no distinction on file system between LO and NLO")
    if Nf != 3:
        print("Warning: Currently no distinction on file system for Nf != 3")

    FILE_PATH = hp.generate_filename(eta,0,mu,cfg.MOMENTUM_SPACE_MOMENTS_PATH / f"singlet_{particle}_dipole_moment_{moment_label}" )
    result = get_dipole_from_csv(t,FILE_PATH)
    return result

def dipole_moment(n,eta,t,mu,Nf=3,particle="gluon",moment_type="non_singlet_isovector",moment_label="Atilde",evolution_order="LO",error_type="central"):
    def dipole_form(t, A_D, m_D2): 
        return A_D / (1 - t / m_D2)**2
    
    def parse_csv_params(csv_path, particle, moment_type, moment_label, n, evolution_order):
        csv_path = cfg.Path(csv_path)
        key = (particle, moment_type, moment_label, str(n), evolution_order)

        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
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
    
    if Nf != 3:
        print("Warning: Currently no distinction on file system for Nf != 3")
    
    prefix = "dipole_moments"
    file_path = hp.generate_filename(eta,0,mu,cfg.MOMENTUM_SPACE_MOMENTS_PATH / prefix,error_type )
    result = get_dipole_from_csv(t,file_path, particle, moment_type, moment_label, n, evolution_order)
    return result

def evolve_singlet_D(j,eta,t,mu,Nf=3,D0=1,particle="quark",moment_label="A",evolution_order="LO",error_type="central"):
    hp.check_particle_type(particle)
    hp.check_moment_type_label("singlet",moment_label)
    if j == 2:
        eta = 1 # Result is eta independent 
    term_1 = evolve_conformal_moment(j,eta,t,mu,Nf,1,particle,"singlet",moment_label,evolution_order,error_type)
    term_2 = evolve_conformal_moment(j,0,t,mu,Nf,1,particle,"singlet",moment_label,evolution_order,error_type)
    result = D0 * (term_1-term_2)/eta**2
    return result

def evolve_quark_non_singlet(j,eta,t,mu,Nf=3,A0=1,moment_type="non_singlet_isovector",moment_label = "A",evolution_order="LO",error_type="central"):
    result = evolve_conformal_moment(j,eta,t,mu,Nf,A0,"quark",moment_type,moment_label,evolution_order,error_type)
    return result

def evolve_quark_singlet(j,eta,t,mu,Nf=3,A0=1,moment_label = "A",evolution_order="LO",error_type="central"):
    result = evolve_conformal_moment(j,eta,t,mu,Nf,A0,"quark","singlet",moment_label,evolution_order,error_type)
    return result

def evolve_gluon_singlet(j,eta,t,mu,Nf=3,A0=1,moment_label = "A",evolution_order="LO",error_type="central"):
    result = evolve_conformal_moment(j,eta,t,mu,Nf,A0,"gluon","singlet",moment_label,evolution_order,error_type)
    return result

def evolve_quark_singlet_D(eta,t,mu,Nf=3,D0=1,moment_label = "A",evolution_order="LO",error_type="central"):
    result = evolve_singlet_D(eta,t,mu,Nf,D0,"quark",moment_label,evolution_order,error_type)
    return result

def evolve_gluon_singlet_D(j,eta,t,mu,Nf=3,D0=1,moment_label = "A",evolution_order="LO",error_type="central"):
    result = evolve_singlet_D(eta,t,mu,Nf,D0,"gluon",moment_label,evolution_order,error_type)
    return result

def fourier_transform_moment(j,eta,mu,b_vec,Nf=3,A0=1,particle="quark",moment_type="non_singlet_isovector", moment_label="A",evolution_order="LO", Delta_max = 5,num_points=100, error_type="central"):
    """
    Optimized calculation of Fourier transformed moments using trapezoidal rule.

    Parameters:
    - j (float): Conformal spin
    - eta (float): Skewness parameter
    - mu (float): RGE scale
    - b_vec: (b_x, b_y), the vector for which to compute the result
    - Nf (int, optional): Number of active flavors. Default is 3.
    - A0 (float, optional): Overall scale
    - particle (str. optional): "quark" or "gluon". Default is quark.
    - moment_type (str. optional): singlet, non_singlet_isovector or non_singlet_isoscalar. Default is non_singlet_isovector.
    - moment_label (str. optiona): Label of conformal moment, e.g. A
    - Delta_max (float, optional): maximum radius for the integration domain (limits the integration bounds)
    - num_points: number of points for discretizing the domain (adapt as needed)
    - error_type (str. optional): Whether to use central, plus or minus value of input PDF. Default is central.

    Returns:
    - The value of the Fourier transformed moment at (b_vec)
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
        t = -(Delta_x**2+Delta_y**2)
        exponent = -1j * (b_x * Delta_x + b_y * Delta_y)
        if j >= 2 or moment_type != "singlet":
            moment = evolve_conformal_moment(j,eta,t,mu,Nf,A0,particle,moment_type,moment_label,evolution_order,error_type)
        else:
            moment = first_singlet_moment_dipole(eta,t,mu,Nf,particle,moment_label,evolution_order,error_type)
        moment_re = np.vectorize(lambda x: float(mp.re(x)))(moment).astype(np.float64)
        result = moment_re * np.exp(exponent)
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
        moment = fourier_transform_moment(j,eta,mu,b_vec,Nf,particle,moment_type,moment_label,evolution_order,num_points=num_points,Delta_max=Delta_max)
        result = np.float64(mp.re(moment))*np.exp(exponent)
        return result

    # Compute the integrand for each pair of (Delta_x, Delta_y) values
    integrand_values = np.array(Parallel(n_jobs=-1)(delayed(integrand)(b_x, b_y, Delta_x, Delta_y)
                                                 for b_y in b_y_vals
                                                 for b_x in b_x_vals))
    integrand_values = integrand_values.reshape((num_points, num_points))

    integral_result = trapezoid(trapezoid(integrand_values, b_x_vals, axis=1), b_y_vals,axis=0)

    return integral_result.real

def fourier_transform_transverse_moment(j,eta,mu,b_vec,Nf=3,A0=1,particle="quark",moment_type="non_singlet_isovector",evolution_order="LO", Delta_max = 5,num_points=100, error_type="central"):
    """
    Optimized calculation of Fourier transformed moments for transversely polarized target using trapezoidal rule. 
    Automatically uses A_n and B_n moments with assumed nucleon mass of M_n = 0.93827 GeV

    Parameters:
    - j (float): Conformal spin
    - eta (float): Skewness parameter
    - mu (float): RGE scale
    - b_vec: (b_x, b_y), the vector for which to compute the result
    - Nf (int, optional): Number of active flavors. Default is 3.
    - A0 (float, optional): Overall scale
    - particle (str. optional): "quark" or "gluon". Default is quark.
    - moment_type (str. optional): singlet, non_singlet_isovector or non_singlet_isoscalar. Default is non_singlet_isovector.
    - Delta_max (float, optional): maximum radius for the integration domain (limits the integration bounds)
    - num_points: number of points for discretizing the domain (adapt as needed)
    - error_type (str. optional): Whether to use central, plus or minus value of input PDF. Default is central.

    Returns:
    - The value of the Fourier transformed moment at (b_vec)
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
        moment_1 = evolve_conformal_moment(j,eta,t,mu,Nf,A0,particle,moment_type,"A",evolution_order,error_type)
        moment_2 = 1j * Delta_y/(2*M_n) * evolve_conformal_moment(j,eta,t,mu,Nf,A0,particle,moment_type,"B",evolution_order,error_type)
        moment = moment_1 + moment_2
        result = np.float64(mp.re(moment))*np.exp(exponent)
        return result
    
    if moment_type == "singlet" and j < 2:
        print("Warning: Fourier transform for transverse singlet moments for j = 1 unsupported")

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
    hp.check_particle_type(particle)
    if moment_type not in ["singlet","non_singlet_isovector","non_singlet_isoscalar","u","d"]:
        raise ValueError(f"Wrong moment_type {moment_type}")
    hp.check_error_type(error_type)

    if moment_type in ["singlet","non_singlet_isovector","non_singlet_isoscalar"]:
        result = fourier_transform_moment(1,eta,mu,b_vec,Nf,particle,moment_type,moment_label="Atilde",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type=error_type)/2
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
    hp.check_particle_type(particle)
    if moment_type not in ["singlet","non_singlet_isovector","non_singlet_isoscalar","u","d"]:
        raise ValueError(f"Wrong moment_type {moment_type}")
    hp.check_error_type(error_type)

    if moment_type in ["singlet","non_singlet_isovector","non_singlet_isoscalar"]:
            term_1 = fourier_transform_moment(2,eta,mu,b_vec,Nf,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type="central")
            term_2 = fourier_transform_moment(1,eta,mu,b_vec,Nf,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type="central")
            moment = (term_1-term_2)/2
            if error_type != "central":      
                term_1_error = .5 * (fourier_transform_moment(2,eta,mu,b_vec,Nf,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type=error_type)
                                - term_1)
                term_2_error = .5 * (fourier_transform_moment(1,eta,mu,b_vec,Nf,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type=error_type)
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
    hp.check_particle_type(particle)
    if moment_type not in ["singlet","non_singlet_isovector","non_singlet_isoscalar","u","d"]:
        raise ValueError(f"Wrong moment_type {moment_type}")
    hp.check_error_type(error_type)

    if moment_type in ["singlet","non_singlet_isovector","non_singlet_isoscalar"]:
            term_1 = fourier_transform_moment(2,eta,mu,b_vec,Nf,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type="central")
            term_2 = fourier_transform_moment(2,eta,mu,b_vec,Nf,particle=particle,moment_type=moment_type,moment_label="B",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type="central")
            term_3 = fourier_transform_moment(1,eta,mu,b_vec,Nf,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type="central")
            moment = (term_1+term_2-term_3)/2
            if error_type != "central":      
                term_1_error = .5 * (fourier_transform_moment(2,eta,mu,b_vec,Nf,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type=error_type)
                                - term_1)
                term_2_error = .5 * (fourier_transform_moment(2,eta,mu,b_vec,Nf,particle=particle,moment_type=moment_type,moment_label="B",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type=error_type)
                                - term_2)
                term_3_error = .5 * (fourier_transform_moment(1,eta,mu,b_vec,Nf,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,Delta_max=Delta_max,num_points=num_points,error_type=error_type)
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
    hp.check_particle_type(particle)
    hp.check_parity(parity)
    if parity not in ["even", "odd","none"]:
        raise ValueError("Parity must be even, odd or none")
    
    # Precompute factors that do not change
    if particle == "quark":
        gamma_term = lambda j: 2.0**j * mp.gamma(1.5 + j) / (mp.gamma(0.5) * mp.gamma(j))
        sin_term = lambda j: mp.sin(mp.pi * j) / mp.pi
        def cal_P(x,eta):
            eta = 1e-6 if eta < 1e-6 else eta
            arg = (1 + x / eta)
            hyp = mp.hyp2f1(-j, j + 1, 2, 0.5 * arg)
            result = 1 / eta**j  * arg * hyp * gamma_term(j)
            return result
        def cal_Q(x,eta):
            hyp = mp.hyp2f1(0.5 * j, 0.5 * (j + 1), 1.5 + j, (eta / x)**2) 
            result = 1 / x**j * hyp * sin_term(j)
            return result
    else:   
        gamma_term = lambda j: 2.0**(j-1) * mp.gamma(1.5 + j) / (mp.gamma(0.5) * mp.gamma(j-1))
        sin_term =lambda j: mp.sin(mp.pi * (j+1))  / mp.pi 
        def cal_P(x,eta):
            eta = 1e-6 if eta < 1e-6 else eta
            arg = (1. + x / eta)
            hyp = mp.hyp2f1(-j, j + 1, 3, 0.5 * arg)
            result = 1 / eta**(j-1) * arg**2 * hyp * gamma_term(j)
            return result
        def cal_Q(x,eta):
            hyp = mp.hyp2f1(0.5 * (j-1), 0.5 * j, 1.5 + j, (eta / x)**2) 
            result = 1 / x**(j-1) * hyp * sin_term(j)
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
    hp.check_particle_type(particle)
    hp.check_moment_type_label(moment_type,moment_label)

    if moment_label in ["A","B"]:
        if particle == "quark" and moment_type in ["non_singlet_isovector","non_singlet_isoscalar"]:
            j_base, parity = .95, "none"
        elif particle == "quark" and moment_type == "singlet":
            j_base, parity = 1.7, "odd"
        elif particle == "gluon" and moment_type == "singlet":
            j_base, parity = 1.7, "even"
    elif moment_label == "Atilde":
        if particle == "quark" and moment_type in ["non_singlet_isovector","non_singlet_isoscalar"]:
            j_base, parity = .95, "none"
        if particle == "quark" and moment_type == "singlet":
            j_base, parity = 1.7, "even"
        if particle == "gluon" and moment_type == "singlet":
            j_base, parity = 1.7, "odd"
    else:
        raise ValueError(f"Wrong moment type {moment_type} and/or label {moment_label} for particle {particle}")
    
    return j_base, parity

def mellin_barnes_gpd(x, eta, t, mu, Nf=3, A0=1 ,particle = "quark", moment_type="non_singlet_isovector",moment_label="A",evolution_order="LO", error_type="central",real_imag ="real",j_max = 15, n_jobs=1):
    """
    Numerically evaluate the Mellin-Barnes integral parallel to the imaginary axis to obtain the corresponding GPD
    
    Parameters:
    - x (float): Parton x
    - eta (float): Skewness.
    - t (float): Mandelstam t
    - mu (float): Resolution scale
    - Nf (int 1<= Nf <=3, optional): Number of flavors
    - A0 (float. optional): Overall scale
    - particle (str,optional): particle species (quark or gluon)
    - moment_type (str,optional): singlet, non_singlet_isovector, non_singlet_isoscalar
    - moment_label (str,optional): A, Atilde, B
    - error_type (str,optional): value of input PDFs (central, plus, minus)
    - real_imag (str,optional): Choose to compute real part, imaginary part or both
    - j_max (float,optional): Integration range parallel to the imaginary axis
    - n_jobs (int,optional): Number of subregions, and thus processes, the integral is split into
    - n_k (int,optional): Number of sampling points within the interval [-j_max,j_max]
    Returns: 
    - The value of the Mellin-Barnes integral with real and imaginary part.
    Note:
    - For low x and/or eta it is recommended to divide the integration region
    """
    hp.check_particle_type(particle)
    hp.check_error_type(error_type)
    hp.check_evolution_order(evolution_order)
    hp.check_moment_type_label(moment_type,moment_label)

    j_base, parity = get_j_base(particle,moment_type,moment_label)

    # Integrand function which returns both real and imaginary parts
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
        pw_val = conformal_partial_wave(z, x, eta, particle, parity)

        if particle == "quark":
            if moment_type == "singlet":
                mom_val = evolve_quark_singlet(z,eta,t,mu,Nf,A0,moment_label,evolution_order,error_type)
            else:
                mom_val = evolve_quark_non_singlet(z,eta,t,mu,Nf,A0,moment_type,moment_label,evolution_order,error_type)
        else:
            # (-1) from shift in Sommerfeld-Watson transform
            mom_val = (-1) * evolve_gluon_singlet(z,eta,t,mu,Nf,A0,moment_label,evolution_order,error_type)
        result = -.5j * dz * pw_val * mom_val / sin_term
        # print(z,dz,sin_term,pw_val,mom_val)
        if real_imag == 'real':
            return np.float64(mp.re(result))
        elif real_imag == 'imag':
            return np.float64(mp.im(result))
        elif real_imag == 'both':
            return result
        else:
            raise ValueError("real_imag must be either 'real', 'imag', or 'both'")
    # print(integrand(.2,'real'))
 
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
            raise ValueError(f"Maximum number of iterations reached at (x,eta,t) = {x,eta,t} without finding a suitable bound. Increase initial value of j_max")

        # Check for rapid oscillations
        if abs(integrand(j_max,  "real") - integrand(j_max + 2,  "real")) > tolerance:
            while abs(integrand(j_max,  "real")) > tolerance and iterations < max_iterations:
                j_max += step_size
                iterations += 1

            if iterations == max_iterations:
                raise ValueError(f"Maximum number of iterations reached at (x,eta,t) = {x,eta,t} without finding a suitable bound. Increase initial value of j_max")
        if j_max > 250:
            print(f"Warning j_max={j_max} is large, adjust corresponding base value in get_j_base")
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
            # integral, error = quad(lambda k: integrand(k, 'real'), k_min, k_max, limit = 200)
            integral, _ = fixed_quad(lambda k: integrand(k, 'real'), k_min, k_max, n = 130)
            integral_50, _ = fixed_quad(lambda k: integrand(k, 'real'), k_min, k_max, n = 70)
            error = abs(integral-integral_50)
            # Use symmetry of the real part of the integrand
            integral *= 2
            error *= 2
            return integral, error
        elif real_imag == 'imag':
            # integral, error = quad(lambda k: integrand(k, 'imag'), k_min, k_max, limit = 100)
            integral, _ = fixed_quad(lambda k: integrand(k, 'imag'), k_min, k_max, n = 100)
            integral_50, _ = fixed_quad(lambda k: integrand(k, 'imag'), k_min, k_max, n = 50)
            error = abs(integral-integral_50)
            return integral, error
        elif real_imag == 'both':
            # integral, error = mp.quad(lambda k: integrand(k, 'both'), -k_max, k_max, error=True)
            integral, _ = fixed_quad(lambda k: integrand(k, 'both'), k_min, k_max, n = 100)
            integral_50, _ = fixed_quad(lambda k: integrand(k, 'both'), k_min, k_max, n = 50)
            real_integral, imag_integral = np.float64(mp.re(integral)), np.float64(mp.im(integral))
            error = abs(integral.real-integral_50.real) + 1j * abs(integral.imag-integral_50.imag)
            real_error, imag_error = np.float64(mp.re(error)), np.float64(mp.im(error))
            # real_integral, real_error = quad(lambda k: integrand(k, 'real'), k_min, k_max, limit = 200)
            # imag_integral, imag_error = quad(lambda k: integrand(k, 'imag'), k_min, k_max, limit = 200)
            return real_integral, real_error, imag_integral, imag_error
        else:
            raise ValueError("real_imag must be either 'real', 'imag', or 'both'") 

    # Dynamically determine integration bound
    j_max = find_integration_bound(integrand, j_max) 
    # print('jmax',j_max)
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
    # print(integral,error)
    # Check for the estimated error
    if np.abs(error) > 1e-3:
        print(f"Warning: Large error estimate for (x,eta,t)={x,eta,t}: {error}")
    return float(integral)

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

###################################
## Generate interpolation tables ##
###################################
def generate_moment_table(eta,t,mu,Nf,solution,particle,moment_type,moment_label, evolution_order, error_type,
                          im_j_max=100,step=.1):
    """
    Generate tables for interpolation of input and evolved moments. The real part of j is defined by the value in get_j_base
    """
    if moment_label in ["A","B"]:
        evolve_type = "vector"
    elif moment_label in ["Atilde","Btilde"]:
        evolve_type = "axial"

    def compute_moment(j):
        # For mu != 1 we interpolate the evolved moments
        if mu != 1 or (moment_type == "singlet" and solution not in ["+","-"]):
            return evolve_conformal_moment(j, eta, t,mu,Nf=Nf,A0=1,particle=particle,moment_type=moment_type,
                                           moment_label=moment_label,evolution_order=evolution_order,
                                           error_type=error_type)
        # For mu = 1 we use the corresponding input moments
        elif moment_type == "non_singlet_isovector":
            return non_singlet_isovector_moment(j,eta,t,
                                                moment_label=moment_label, evolve_type=evolve_type,
                                                evolution_order=evolution_order, error_type=error_type)
        elif moment_type == "non_singlet_isoscalar":
            return non_singlet_isoscalar_moment(j,eta,t,
                                                moment_label=moment_label, evolve_type=evolve_type,
                                                evolution_order=evolution_order, error_type=error_type)
        elif moment_type == "singlet":
            val = singlet_moment(j, eta, t, Nf=Nf,
                                  moment_label=moment_label, evolve_type=evolve_type,
                                  solution=solution, evolution_order=evolution_order, error_type=error_type)
            return val[0] if error_type == "central" else val[1]
        else:
            raise ValueError(f"Unknown moment_type {moment_type}")

    # Define output filename
    if mu != 1 or (moment_type == "singlet" and solution not in ["+","-"]) or moment_type != "singlet":
        prefix = cfg.INTERPOLATION_TABLE_PATH / f"{moment_type}_{particle}_moments_{moment_label}_{evolution_order}"
    elif moment_type == "singlet":
        prefix = cfg.INTERPOLATION_TABLE_PATH/ f"{moment_type}_{solution}_moments_{moment_label}_{evolution_order}"
    filename = hp.generate_filename(eta, t, mu, prefix, error_type)

    # Grids
    re_j_b, _ = get_j_base(particle=particle,moment_type=moment_type,moment_label=moment_label)
    # First value is lower value of nd evolution when resummed
    # Second value is real part of Mellin-Barnes integration variable
    # Third part is intermediate value since pchip needs >= 4 data points.
    # Fourth part is from shift in fractional finite sum:
    # k + j - k0 + 1 where k0 = 1 for non-singlet and 2 for singlet
    if moment_type != "singlet":
        re_j = [0.8, re_j_b,re_j_b + 0.4, re_j_b + 0.8]
    else:
        re_j = [1.8, re_j_b,re_j_b + 0.4, re_j_b + 0.8]
    im_j = np.arange(0.0, im_j_max + step, step)  # Im ≥ 0 only
    # Generate grid points
    grid_points = [
        (rj, ij)
        for rj in re_j
        for ij in im_j
    ]
    def compute_wrapper(rj, ij):
        return (rj, ij, compute_moment(complex(rj, ij)))

    with hp.tqdm_joblib(tqdm(total=len(grid_points))) as progress_bar:
        results = Parallel(n_jobs=-1)(
            delayed(compute_wrapper)(rj, ij) for rj, ij in grid_points
        )

    # Add mirrored points: f(j*) = f(j)* for Im(j) < 0
    mirrored = []
    for rj, ij, val in results[1:]:  # skip ij = 0 to avoid duplication
        if ij > 0:
            mirrored.append((rj, -ij, np.conj(val)))

    all_data = results + mirrored
    all_data.sort(key=lambda x: (x[0], x[1]))  # sort by Re(j), Im(j)

    # Write CSV
    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Re(j)", "Im(j)", "value"])
        for rj, ij, val in all_data:
            writer.writerow([rj, ij, val])

    print(f"Successfully wrote table to {filename}")

def generate_harmonic_table(indices,j_re_min=0, j_re_max=10, j_im_min=0, j_im_max=110,step=0.1, n_jobs=-1):
    def compute_values(j_re, j_im):
        j = complex(j_re, j_im)
        if isinstance(indices,int):
            val = harmonic_number(indices, j)
        else:
            val = nested_harmonic_number(indices, j, interpolation=False)
        row = [[j_re, j_im, val]]
        if j_im > 0:
            row.append([j_re, -j_im, np.conj(val)])
        return row

    re_vals = np.arange(j_re_min, j_re_max + step, step)
    im_vals = np.arange(j_im_min, j_im_max + step, step)

    if isinstance(indices,int):
        m1 = indices
        filename = cfg.ANOMALOUS_DIMENSIONS_PATH / f"harmonic_m1_{m1}.csv"
    elif len(indices) == 2:
        m1, m2 = indices
        filename = cfg.ANOMALOUS_DIMENSIONS_PATH / f"nested_harmonic_m1_{m1}_m2_{m2}.csv"
    elif len(indices) == 3:
        m1, m2, m3 = indices
        filename = cfg.ANOMALOUS_DIMENSIONS_PATH / f"nested_harmonic_m1_{m1}_m2_{m2}_m3_{m3}.csv"
    else:
        raise ValueError("Table generation currently only supports 3 nested harmonics")

    # Generate grid points
    grid = [(j_re, j_im) for j_re in re_vals for j_im in im_vals]
    with hp.tqdm_joblib(tqdm(total=len(grid))) as progress_bar:
        # Parallel computation over (j_re, j_im) pairs
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_values)(j_re, j_im)
            for j_re, j_im in grid
        )

    # Flatten the list of lists
    data = [row for sublist in results for row in sublist]
    data.sort(key=lambda x: (x[0], x[1]))  # Sort by Re(j), Im(j)

    # Write to CSV
    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Re(j)", "Im(j)", "harmonic_number"])
        writer.writerows(data)

    print(f"Successfully wrote table to {filename}")
    
def generate_nested_harmonic_table(m1, m2,
                                            j_re_min=1e-4, j_re_max=3, j_im_min=0, j_im_max=110,
                                            step=0.1, n_jobs=-1):
    def compute_values(j_re, j_im, m1, m2):
        j = complex(j_re, j_im)
        val = nested_harmonic_number([m1, m2], j, interpolation=False)
        row = [[j_re, j_im, val]]
        if j_im > 0:
            row.append([j_re, -j_im, np.conj(val)])
        return row

    re_vals = np.arange(j_re_min, j_re_max + step, step)
    im_vals = np.arange(j_im_min, j_im_max + step, step)

    filename = cfg.ANOMALOUS_DIMENSIONS_PATH / f"nested_harmonic_m1_{m1}_m2_{m2}.csv"

    # Generate grid points
    grid = [(j_re, j_im) for j_re in re_vals for j_im in im_vals]
    with hp.tqdm_joblib(tqdm(total=len(grid))) as progress_bar:
        # Parallel computation over (j_re, j_im) pairs
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_values)(j_re, j_im, m1, m2)
            for j_re, j_im in grid
        )

    # Flatten the list of lists
    data = [row for sublist in results for row in sublist]
    data.sort(key=lambda x: (x[0], x[1]))  # Sort by Re(j), Im(j)

    # Write to CSV
    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Re(j)", "Im(j)", "nested_harmonic_number"])
        writer.writerows(data)

    print(f"Successfully wrote table to {filename}")

def generate_anomalous_dimension_table(suffix,moment_type,evolve_type,Nf=3,evolution_order="NLO",
                                            j_re_min=1e-4, j_re_max=6, j_im_min=0, j_im_max=110,
                                            step=0.1, n_jobs=-1):
    def compute_values(j_re, j_im):
        j = mp.mpc(j_re,j_im)
        if suffix == "qq":
            val = gamma_qq(j,Nf,moment_type,evolve_type,evolution_order,interpolation=False)
        elif suffix == "qg":
            val = gamma_qg(j,Nf,evolve_type,evolution_order,interpolation=False)
        elif suffix == "gq":
            val = gamma_gq(j,Nf,evolve_type,evolution_order,interpolation=False)
        elif suffix == "gg":
            val = gamma_gg(j,Nf,evolve_type,evolution_order,interpolation=False)
        else:
            raise ValueError(f"Wrong suffix {suffix}")
        
        val = complex(val)
        row = [[j_re, j_im, val]]
        if j_im > 0:
            row.append([j_re, -j_im, np.conj(val)])
        return row

    if evolution_order == "LO":
        order = "lo"
    elif evolution_order == "NLO":
        order = "nlo"
    else:
        raise ValueError(f"Wrong evolution_order {evolution_order}")

    re_vals = np.arange(j_re_min, j_re_max + step, step)
    im_vals = np.arange(j_im_min, j_im_max + step, step)
    
    if moment_type != "singlet" and suffix == "qq":
        filename = cfg.ANOMALOUS_DIMENSIONS_PATH / f"gamma_{suffix}_non_singlet_{evolve_type}_{order}.csv"
    else:
        filename = cfg.ANOMALOUS_DIMENSIONS_PATH / f"gamma_{suffix}_{evolve_type}_{order}.csv"

    # Generate grid points
    grid = [(j_re, j_im) for j_re in re_vals for j_im in im_vals]
    with hp.tqdm_joblib(tqdm(total=len(grid))) as progress_bar:
        # Parallel computation over (j_re, j_im) pairs
        results = Parallel(n_jobs=n_jobs)(
            delayed(compute_values)(j_re, j_im)
            for j_re, j_im in grid
        )

    # Flatten the list of lists
    data = [row for sublist in results for row in sublist]
    data.sort(key=lambda x: (x[0], x[1]))  # Sort by Re(j), Im(j)

    # Write to CSV
    with open(filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Re(j)", "Im(j)", f"gamma_{suffix}"])
        writer.writerows(data)

    print(f"Successfully wrote table to {filename}")

################################
#### Additional Observables ####
################################

def spin_orbit_corelation(eta,t,mu,Nf =3, A0 = 1, particle="quark",moment_type="non_singlet_isovector",evolution_order="LO"):
    """ Returns the spin orbit correlation of moment_type including errors

    Parameters:
    - eta (float): Skewness parameter   
    - t (float): Mandelstam t
    - mu (float): The momentum scale of the process
    - Nf (int. optional): Number of active flavors
    - A0 (float, optional): Overall scale
    - moment_type (str. optional): The flavor dependence. Either non_singlet_isovector or non_singlet_isoscalar    
    """

    if moment_type not in ["singlet",
                           "non_singlet_isoscalar",
                           "non_singlet_isovector"]:
        raise ValueError(f"Wrong moment type {moment_type}")

    term_1 = evolve_conformal_moment(2,0,t,mu,Nf,A0,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,error_type="central")
    term_1_plus = evolve_conformal_moment(2,0,t,mu,Nf,A0,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,error_type="plus")
    term_1_minus = evolve_conformal_moment(2,0,t,mu,Nf,A0,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,error_type="minus")

    # if moment_type != "singlet":
    #     term_2 = evolve_conformal_moment(1,0,t,mu,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,error_type="central")
    #     term_2_plus = evolve_conformal_moment(1,0,t,mu,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,error_type="plus")
    #     term_2_minus = evolve_conformal_moment(1,0,t,mu,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,error_type="minus")
    # else:
    #     term_2 = first_singlet_moment(eta,mu,particle=particle,gpd_label="H",error_type="central",t0_only=t)
    #     term_2_plus = first_singlet_moment(eta,mu,particle=particle,gpd_label="H",error_type="plus",t0_only=t)
    #     term_2_minus = first_singlet_moment(eta,mu,particle=particle,gpd_label="H",error_type="minus",t0_only=t)

    term_2 = evolve_conformal_moment(1,0,t,mu,Nf,A0,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,error_type="central")
    term_2_plus = evolve_conformal_moment(1,0,t,mu,Nf,A0,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,error_type="plus")
    term_2_minus = evolve_conformal_moment(1,0,t,mu,Nf,A0,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,error_type="minus")

    result = (term_1 - term_2)/2
    error_plus = np.sqrt((term_1_plus-term_1)**2+(term_2_plus-term_2)**2)/2
    error_minus = np.sqrt((term_1_minus-term_1)**2+(term_2_minus-term_2)**2)/2

    return result, error_plus, error_minus

def total_spin(t,mu,Nf=3,A0=1,particle="quark",moment_type="non_singlet_isovector",evolution_order="LO"):
    """ Returns the total spin of moment_type including errors

    Parameters:
    - eta (float): Skewness parameter
    - t (float): Mandelstam t
    - Nf (int, optional): Number of active quarks
    - A0 (float, optional): Overall scale
    - mu (float): The momentum scale of the process
    - moment_type (str. optional): The flavor dependence. Either non_singlet_isovector or non_singlet_isoscalar    
    """
    if moment_type not in ["singlet",
                           "non_singlet_isoscalar",
                           "non_singlet_isovector"]:
        raise ValueError(f"Wrong moment type {moment_type}")

    term_1 = evolve_conformal_moment(2,0,t,mu,Nf,A0,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,error_type="central")
    term_2 = evolve_conformal_moment(2,0,t,mu,Nf,A0,particle=particle,moment_type=moment_type,moment_label="B",evolution_order=evolution_order,error_type="central")
    result = (term_1 + term_2)/2

    term_1_plus = evolve_conformal_moment(2,0,t,mu,Nf,A0,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,error_type="plus")
    term_2_plus = evolve_conformal_moment(2,0,t,mu,Nf,A0,particle=particle,moment_type=moment_type,moment_label="B",evolution_order=evolution_order,error_type="plus")
    error_plus = np.sqrt((term_1_plus-term_1)**2+(term_2_plus-term_2)**2)/2

    term_1_minus = evolve_conformal_moment(2,0,t,mu,Nf,A0,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,error_type="minus")
    term_2_minus = evolve_conformal_moment(2,0,t,mu,Nf,A0,particle=particle,moment_type=moment_type,moment_label="B",evolution_order=evolution_order,error_type="minus")
    error_minus = np.sqrt((term_1_minus-term_1)**2+(term_2_minus-term_2)**2)/2

    return result, error_plus, error_minus

def orbital_angular_momentum(eta,t,mu,Nf=3,A0=1, particle="quark",moment_type="non_singlet_isovector",evolution_order="LO"):
    """ Returns the orbital angular momentum of moment_type including errors

    Parameters:
    - eta (float): Skewness parameter
    - t (float): Mandelstam t
    - Nf (int, optional): Number of active quarks
    - A0 (float, optional): Overall scale
    - mu (float): The momentum scale of the process
    - moment_type (str. optional): The flavor dependence. Either non_singlet_isovector or non_singlet_isoscalar    
    """
    hp.check_particle_type(particle)
    if moment_type not in ["singlet",
                           "non_singlet_isoscalar",
                           "non_singlet_isovector"]:
        raise ValueError(f"Wrong moment type {moment_type}")

    term_1 = evolve_conformal_moment(2,eta,t,mu,Nf,A0,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,error_type="central")
    term_2 = evolve_conformal_moment(2,eta,t,mu,Nf,A0,particle=particle,moment_type=moment_type,moment_label="B",evolution_order=evolution_order,error_type="central")

    term_1_plus = evolve_conformal_moment(2,eta,t,mu,Nf,A0,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,error_type="plus")
    term_2_plus = evolve_conformal_moment(2,eta,t,mu,Nf,A0,particle=particle,moment_type=moment_type,moment_label="B",evolution_order=evolution_order,error_type="plus")

    term_1_minus = evolve_conformal_moment(2,eta,t,mu,Nf,A0,particle=particle,moment_type=moment_type,moment_label="A",evolution_order=evolution_order,error_type="minus")
    term_2_minus = evolve_conformal_moment(2,eta,t,mu,Nf,A0,particle=particle,moment_type=moment_type,moment_label="B",evolution_order=evolution_order,error_type="minus")

    term_3 = evolve_conformal_moment(1,eta,t,mu,Nf,A0,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,error_type="central")
    term_3_plus = evolve_conformal_moment(1,eta,t,mu,Nf,A0,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,error_type="plus")
    term_3_minus = evolve_conformal_moment(1,eta,t,mu,Nf,A0,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,error_type="minus")

    # if moment_type != "singlet":
    #     term_3 = evolve_conformal_moment(1,eta,t,mu,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,error_type="central")
    #     term_3_plus = evolve_conformal_moment(1,eta,t,mu,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,error_type="plus")
    #     term_3_minus = evolve_conformal_moment(1,eta,t,mu,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,error_type="minus")
    # else:
    #     term_3 = first_singlet_moment(eta,mu,particle=particle,gpd_label="Htilde",error_type="central",t0_only=t)
    #     term_3_plus = first_singlet_moment(eta,mu,particle=particle,gpd_label="Htilde",error_type="plus",t0_only=t)
    #     term_3_minus = first_singlet_moment(eta,mu,particle=particle,gpd_label="Htilde",error_type="minus",t0_only=t)

    result = (term_1 + term_2)/2 - term_3/2
    error_plus = np.sqrt((term_1_plus-term_1)**2+(term_2_plus-term_2)**2+(term_3-term_3_plus)**2)/2
    error_minus = np.sqrt((term_1_minus-term_1)**2+(term_2_minus-term_2)**2+(term_3-term_3_minus)**2)/2

    return result, error_plus, error_minus

def quark_gluon_helicity(eta,t,mu,Nf=3,A0=1, particle="quark",moment_type="non_singlet_isovector",evolution_order="LO"):
    """ Prints the quark helicity of moment_type including errors

    Parameters:
    - eta (float): Skewness parameter
    - t (float): Mandelstam t
    - mu (float): The momentum scale of the process
    - Nf (int, optional): Number of active quarks
    - A0 (float, optional): Overall scale
    - moment_type (str. optional): The flavor dependence. Either non_singlet_isovector or non_singlet_isoscalar    
    """
    hp.check_particle_type(particle)
    if moment_type not in ["singlet","non_singlet_isoscalar","non_singlet_isovector"]:
        raise ValueError(f"Wrong moment type {moment_type}")
    if particle == "gluon" and moment_type != "singlet":
        raise ValueError(f"Wrong moment_type {moment_type} for {particle}")

    # if particle == "quark":
    #     result = evolve_conformal_moment(1,eta,t,mu,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,error_type="central")/2

    #     term_1 = evolve_conformal_moment(1,eta,t,mu,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,error_type="plus")/2
    #     error_plus = abs(result - term_1)

    #     term_1 = evolve_conformal_moment(1,eta,t,mu,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,error_type="minus")/2
    #     error_minus = abs(result - term_1)
    # else:
    #     result = first_singlet_moment(eta=eta,mu=mu,particle="gluon",gpd_label="Htilde",error_type="central",t0_only=t)/2

    #     term_1 = first_singlet_moment(eta=eta,mu=mu,particle="gluon",gpd_label="Htilde",error_type="plus",t0_only=t)/2
    #     error_plus = abs(result - term_1)

    #     term_1 = first_singlet_moment(eta=eta,mu=mu,particle="gluon",gpd_label="Htilde",error_type="plus",t0_only=t)/2
    #     error_minus = abs(result - term_1)

    result = evolve_conformal_moment(1,eta,t,mu,Nf,A0,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,error_type="central")/2

    term_1 = evolve_conformal_moment(1,eta,t,mu,Nf,A0,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,error_type="plus")/2
    error_plus = abs(result - term_1)

    term_1 = evolve_conformal_moment(1,eta,t,mu,Nf,A0,particle=particle,moment_type=moment_type,moment_label="Atilde",evolution_order=evolution_order,error_type="minus")/2
    error_minus = abs(result - term_1)

    return result, error_plus, error_minus

def quark_helicity(eta,t,mu,Nf=3,A0=1, moment_type="non_singlet_isovector",evolution_order="LO"):
    result, error_plus, error_minus = quark_gluon_helicity(eta,t,mu,Nf,A0,particle="quark",moment_type=moment_type,evolution_order=evolution_order)
    return result, error_plus, error_minus

def gluon_helicity(eta,t,mu,Nf=3,A0=1,evolution_order="LO"):
    result, error_plus, error_minus = quark_gluon_helicity(t,mu,Nf,A0,particle="gluon",moment_type="singlet",evolution_order=evolution_order)
    return result, error_plus, error_minus

################################
####### Plot functions #########
################################


def plot_moment(n,eta,y_label,mu_in=2,t_max=3,Nf=3,A0=1,particle="quark",moment_type="non_singlet_isovector", moment_label="A",evolution_order="LO", n_t=50):
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
    - A0 (float, optional): Overall scale
    - particle (str., optional): Either quark or gluon
    - moment_type (str): The type of moment (e.g., "non_singlet_isovector").
    - moment_label (str): The label of the moment (e.g., "A").
    - n_t (int, optional): Number of points for t_fine (default is 50).
    - num_columns (int, optional): Number of columns for the grid layout (default is 3).
    """
    hp.check_particle_type(particle)
    hp.check_moment_type_label(moment_type,moment_label)
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
        return None  # Or handle accordingly
    
    # Compute results for the evolution functions
    def compute_results(j, eta, t_vals, mu, Nf=3, particle="quark", moment_type="non_singlet_isovector", moment_label="A"):
        """Compute central, plus, and minus results for a given evolution function."""
        if moment_type != "D":
            results = Parallel(n_jobs=-1)(
                delayed(lambda t: float(evolve_conformal_moment(j, eta, t, mu, Nf, A0, particle, moment_type, moment_label, evolution_order, "central").real))(t)
                for t in t_vals
            )
            results_plus = Parallel(n_jobs=-1)(
                delayed(lambda t: float(evolve_conformal_moment(j, eta, t, mu, Nf, A0, particle, moment_type, moment_label, evolution_order, "plus").real))(t)
                for t in t_vals
            )
            results_minus = Parallel(n_jobs=-1)(
                delayed(lambda t: float(evolve_conformal_moment(j, eta, t, mu, Nf, A0, particle, moment_type, moment_label, evolution_order, "minus").real))(t)
                for t in t_vals
            )
            return results, results_plus, results_minus
        else:
            results = Parallel(n_jobs=-1)(
                delayed(lambda t: float(evolve_singlet_D(j, eta, t, mu, Nf, A0, particle, moment_label, evolution_order, "central").real))(t)
                for t in t_vals
            )
            results_plus = Parallel(n_jobs=-1)(
                delayed(lambda t: float(evolve_singlet_D(j, eta, t, mu, Nf, A0, particle, moment_label, evolution_order, "plus").real))(t)
                for t in t_vals
            )
            results_minus = Parallel(n_jobs=-1)(
                delayed(lambda t: float(evolve_singlet_D(j, eta, t, mu, Nf, A0, particle, moment_label, evolution_order, "minus").real))(t)
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

    for pub_id, (color,mu) in cfg.PUBLICATION_MAPPING.items():
        if mu != mu_in:
            continue
        data, n_to_row_map = hp.load_lattice_moment_data(particle,moment_type, moment_label, pub_id)
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

def plot_moments_on_grid(eta, y_label, t_max=3, Nf=3,A0=1, particle="quark", moment_type="non_singlet_isovector", moment_label="A",evolution_order="LO", n_t=50, num_columns=3,D_term = False,set_y_lim=False,y_0 = -1, y_1 =1):
    """
    Plots conformal moments vs. available lattice data.

    Parameters:
    - eta (float): Skewness parameter
    - y_label (str.): Label on y-axis
    - t_max (float, optional): Maximum value of -t
    - Nf (float, optional): Number of active flavors
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
        data, n_to_row_map = hp.load_lattice_moment_data(particle,data_moment_type, data_moment_label, pub_id)

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
                delayed(lambda t: float(evolve_conformal_moment(j, eta, t, mu, Nf, A0, particle, moment_type, moment_label, evolution_order, "central").real))(t)
                for t in t_vals
            )
            results_plus = Parallel(n_jobs=-1)(
                delayed(lambda t: float(evolve_conformal_moment(j, eta, t, mu, Nf, A0, particle, moment_type, moment_label, evolution_order, "plus").real))(t)
                for t in t_vals
            )
            results_minus = Parallel(n_jobs=-1)(
                delayed(lambda t: float(evolve_conformal_moment(j, eta, t, mu, Nf, A0, particle, moment_type, moment_label, evolution_order, "minus").real))(t)
                for t in t_vals
            )
            return results, results_plus, results_minus
        else:
            results = Parallel(n_jobs=-1)(
                delayed(lambda t: float(evolve_singlet_D(j, eta, t, mu, Nf,A0, particle, moment_label, evolution_order, "central").real))(t)
                for t in t_vals
            )
            results_plus = Parallel(n_jobs=-1)(
                delayed(lambda t: float(evolve_singlet_D(j, eta, t, mu, Nf,A0, particle, moment_label, evolution_order, "plus").real))(t)
                for t in t_vals
            )
            results_minus = Parallel(n_jobs=-1)(
                delayed(lambda t: float(evolve_singlet_D(j, eta, t, mu, Nf,A0, particle, moment_label, evolution_order, "minus").real))(t)
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
    for pub_id, (color,mu) in cfg.PUBLICATION_MAPPING.items():
        data, n_to_row_map = hp.load_lattice_moment_data(particle,moment_type, data_moment_label, pub_id)
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
        evolve_moment_central, evolve_moment_plus, evolve_moment_minus = compute_results(n, t_fine)

        if publication_data:
            ax.plot(-t_fine, evolve_moment_central, color="blue", linewidth=2, label="This work")
        else:
            ax.plot(-t_fine, evolve_moment_central, color="blue", linewidth=2)
        ax.fill_between(-t_fine, evolve_moment_minus, evolve_moment_plus, color="blue", alpha=0.2)

        # Plot data from publications
        if publication_data:
            for pub_id, (color, mu) in cfg.PUBLICATION_MAPPING.items():
                data, n_to_row_map = hp.load_lattice_moment_data(particle,moment_type, data_moment_label, pub_id)
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
        ax_single.plot(-t_fine, evolve_moment_central, color="blue", linewidth=2)
        ax_single.fill_between(-t_fine, evolve_moment_minus, evolve_moment_plus, color="blue", alpha=0.2)

        # Plot data from publications
        if publication_data:
            for pub_id, (color, mu) in cfg.PUBLICATION_MAPPING.items():
                data, n_to_row_map = hp.load_lattice_moment_data(particle,moment_type, data_moment_label, pub_id)
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

    # Close the figure
    plt.close(fig)

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
            t_values, val_data, err_data = hp.load_Cz_data(particle,moment_type,pub[0],pub[1])
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

    FILE_PATH = cfg.PLOT_PATH / "Cz_over_t.pdf"
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
    hp.check_particle_type(particle)
    hp.check_moment_type_label(moment_type,moment_label)
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
    - n_b (int, optional): Number of points the interval [-b_max, b_max] is split into (default is 100).
    - interpolation (bool, optional): Interpolate data points on finer grid
    - n_int (int, optional): Number of points used for interpolation
    - vmin (float ,optioanl): Sets minimum value of colorbar
    - vmax (float, optional): Sets maximum value of colorbar
    - read_from_file (bool): Whether to load data from file system
    - write_to_file (bool): Whether to write data to file system
    """
    hp.check_particle_type(particle)

    if moment_type not in ["non_singlet_isovector", "non_singlet_isoscalar", "u", "d", "all","singlet"]:
        raise ValueError(f"Wrong moment_type {moment_type}")

    if write_to_file and read_from_file:
        raise ValueError("write_to_file and read_from_file can't simultaneously be True")

    FILE_PATH = cfg.PLOT_PATH / f"imp_param_transv_pol_moment_j_{j}_{moment_type}.pdf"

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
        READ_WRITE_PATH = cfg.IMPACT_PARAMETER_MOMENTS_PATH / f"imp_param_transv_pol_moment_j_{j}_{mom_type}"
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
                    fourier_transform_moment_values_flat = Parallel(n_jobs=-1)(delayed(fourier_transform_transverse_moment)(
                        j=j,eta=eta, mu=mu, b_vec=b_vec, Nf=Nf, A0=1,particle=particle,moment_type=mom_type, evolution_order=evolution_order, Delta_max=Delta_max, num_points=num_points, error_type="central") for b_vec in b_vecs)
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
        ax.set_title(rf"$\rho_{{{j},\perp}}^{{{title}}}$", fontsize=14)
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
        READ_WRITE_PATH = cfg.IMPACT_PARAMETER_MOMENTS_PATH / "imp_param_spin_orbit_" + mom_type 
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
                    file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"plus")
                    _, _, fourier_transform_moment_values_flat_plus = hp.read_ft_from_csv(file_name)
                    file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"minus")
                    _, _, fourier_transform_moment_values_flat_minus = hp.read_ft_from_csv(file_name)
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
        READ_WRITE_PATH = cfg.IMPACT_PARAMETER_MOMENTS_PATH + "imp_param_helicity_" + mom_type
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
                        file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"plus")
                        _, _, fourier_transform_moment_values_flat_plus = hp.read_ft_from_csv(file_name)
                        file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"minus")
                        _, _, fourier_transform_moment_values_flat_minus = hp.read_ft_from_csv(file_name)
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
    
    FILE_PATH = cfg.PLOT_PATH / f"imp_param_helicity_singlet_{particle}.pdf"
    READ_WRITE_PATH = cfg.IMPACT_PARAMETER_MOMENTS_PATH / "imp_param_helicity_" + "singlet_" + particle

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
        fourier_transform_moment_values_flat = Parallel(n_jobs=-1)(delayed(fourier_transform_quark_gluon_helicity)(
                    eta, mu, b_vec, Nf, particle,"singlet", evolution_order, Delta_max, num_points, "central") for b_vec in b_vecs)
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
    
    FILE_PATH = cfg.PLOT_PATH / f"imp_param_spin_orbit_singlet_{particle}.pdf"
    READ_WRITE_PATH = cfg.IMPACT_PARAMETER_MOMENTS_PATH / "imp_param_"  + "spin_orbit_singlet_" + particle

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
        fourier_transform_moment_values_flat = Parallel(n_jobs=-1)(delayed(fourier_transform_spin_orbit_correlation)(
                    eta, mu, b_vec, Nf, particle, "singlet", evolution_order, Delta_max, num_points, "central") for b_vec in b_vecs)
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
        READ_WRITE_PATH = cfg.IMPACT_PARAMETER_MOMENTS_PATH / "imp_param_oam_" + mom_type 
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
                        file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"plus")
                        _, _, fourier_transform_moment_values_flat_plus = hp.read_ft_from_csv(file_name)
                        file_name = hp.generate_filename(eta,0,mu,READ_WRITE_PATH,"minus")
                        _, _, fourier_transform_moment_values_flat_minus = hp.read_ft_from_csv(file_name)
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
            x_values, gpd_values = hp.load_lattice_gpd_data(eta,t,mu,particle,gpd_type,gpd_label,pub_id,error_type)
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
                x_val, results = hp.load_gpd_data(eta,t,mu,particle,gpd_type,gpd_label,evolution_order)
                if x_val is None:
                    print(f"No data for {gpd_type} {gpd_label} at (eta,t,mu) = {eta},{t},{mu} - abort ")
                    return 
                    #raise ValueError("No data found on system. Change write_to_file = True")
            else:
                results = Parallel(n_jobs=-1)(delayed(compute_result)(x,eta,t,mu) for x in x_val)

            # Error bar computations
            if error_bars:
                if read_from_file:
                    x_plus, results_plus = hp.load_gpd_data(eta,t,mu,particle,gpd_type,gpd_label,evolution_order,"plus")
                    x_minus,results_minus = hp.load_gpd_data(eta,t,mu,particle,gpd_type,gpd_label,evolution_order,"minus")
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

def plot_gpds(eta_array, t_array, mu_array, colors,A0=1, Nf=3, particle="quark",gpd_type="non_singlet_isovector",gpd_label="H",evolution_order="LO",sampling=True, n_init=os.cpu_count(), n_points=50, x_0=-1, x_1=1, y_0 = -1e-2, y_1 = 3, 
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

    # eta_array = [1e-3,0.1,1/3]
    # t_array = [-1e-3,-0.23,-0.69]
    # mu_array = [2,2,2]
    # colors = ["purple","orange","green"]

    if moment_type == "singlet":
        x_0 = 1e-2

    def compute_result(x, eta,t,mu,error_type="central"):
        return mellin_barnes_gpd(x, eta, t, mu, Nf,A0,particle,moment_type,moment_label, evolution_order=evolution_order, real_imag="real", error_type=error_type,n_jobs=1)

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
        x_values = np.sort(x_values)
        if read_from_file:
                x_values = None
                x_values, results = hp.load_gpd_data(eta,t,mu,particle,gpd_type,gpd_label,evolution_order)
                if x_values is None:
                    raise ValueError("No data found on system. Change write_to_file = True")
        else:
            # results = Parallel(n_jobs=-1)(delayed(compute_result)(x,eta,t,mu) for x in x_values)
            with hp.tqdm_joblib(tqdm(total=len(x_values))) as progress_bar:
                results = Parallel(n_jobs=-1)(delayed(compute_result)(x,eta,t,mu) for x in x_values)
        # Error bar computations
        if error_bars:
            if read_from_file:
                x_plus, results_plus = hp.load_gpd_data(eta,t,mu,particle,gpd_type,gpd_label,evolution_order,"plus")
                x_minus,results_minus = hp.load_gpd_data(eta,t,mu,particle,gpd_type,gpd_label,evolution_order,"minus")
            else:
                # results_plus = Parallel(n_jobs=-1)(delayed(compute_result)(x,eta,t,mu, error_type="plus") for x in x_values)
                with hp.tqdm_joblib(tqdm(total=len(x_values))) as progress_bar:
                    results_plus = Parallel(n_jobs=-1)(delayed(compute_result)(x,eta,t,mu, error_type="plus") for x in x_values)
                # results_minus = Parallel(n_jobs=-1)(delayed(compute_result)(x,eta,t,mu, error_type="minus") for x in x_values)
                with hp.tqdm_joblib(tqdm(total=len(x_values))) as progress_bar:
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
            hp.save_gpd_data(x_values,eta,t,mu,results,particle,gpd_type,gpd_label,evolution_order)
            hp.save_gpd_data(x_values,eta,t,mu,results_plus,particle,gpd_type,gpd_label,evolution_order,"plus")
            hp.save_gpd_data(x_values,eta,t,mu,results_minus,particle,gpd_type,gpd_label,evolution_order,"minus")

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

