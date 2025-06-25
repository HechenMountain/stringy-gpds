#####################################
######## Exact integrals for ########
## reggeized PDFs and their errors ##
#####################################
import numpy as np

from .mstw_pdf import MSTW_PDF,get_alpha_s
from .aac_pdf import AAC_PDF

from . import helpers as hp
from .core import mp


def pdf_regge(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,j,alpha_p,t):
    """
    Result of the integral of a Reggeized PDF of the form:

        \int dx x**(j - 1 - alpha_p * t) * pdf(x),

    where the PDF is parameterized as:

        pdf(x) = A_pdf * x**eta_1 * (1 - x)**eta_2 * (1 + epsilon * sqrt(x) + gamma_pdf * x)

    Parameters
    ----------
    A_pdf : float
        Normalization constant of the unpolarized input PDF.
    eta_1 : float
        Small-x parameter.
    eta_2 : float
        Large-x parameter
    epsilon : float
        sqrt(x) prefactor
    gamma_pdf : float
        Additional linear piece
    j : float
        Conformal spin.
    alpha_p : float
        Regge slope.
    t : float
        Mandelstam t.

    Returns
    -------
    float or array_like
        Value of the Reggeized moment.
    """
    frac_1 = epsilon*mp.gamma(eta_1+j-alpha_p*t -.5)/(mp.gamma(eta_1+eta_2+j-alpha_p*t+.5))
    frac_2 = (eta_1+eta_2-gamma_pdf+eta_1*gamma_pdf+j*(1+gamma_pdf)-(1+gamma_pdf)*alpha_p*t)*mp.gamma(eta_1+j-alpha_p*t-1)/mp.gamma(1+eta_1+eta_2+j-alpha_p*t)
    result = A_pdf*mp.gamma(1+eta_2)*(frac_1+frac_2)
    return result

def polarized_pdf_regge(
                A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
                Delta_A_pdf,alpha,gamma_pol, lambda_pol,
                j,alpha_p,t,evolution_order="nlo"
                ):
    """
    Result of the integral of a Reggeized polarized PDF of the form:

        \int dx x**(j - 1 - alpha_p * t) * polarized_pdf(x),

    where the polarized PDF is modeled as:

        polarized_pdf(x) = Delta_A_pdf * x**alpha * (1 + gamma_pol * x**lambda_pol) * pdf(x)

    and the unpolarized input PDF is:

        pdf(x) = A_pdf * x**eta_1 * (1 - x)**eta_2 * (1 + epsilon * sqrt(x) + gamma_pdf * x)

    The unpolarized input PDF is taken at its central value, without error variation.

    Parameters
    ----------
    A_pdf : float
        Normalization constant of the unpolarized input PDF.
    eta_1 : float
        Small-x exponent of the unpolarized PDF.
    eta_2 : float
        Large-x exponent of the unpolarized PDF.
    epsilon : float
        Coefficient of sqrt(x) in the unpolarized PDF.
    gamma_pdf : float
        Coefficient of x in the unpolarized PDF.
    Delta_A_pdf : float
        Normalization constant of the polarized input PDF.
    alpha : float
        Small-x exponent of the polarized PDF.
    gamma_pol : float
        Coefficient of x^lambda_pol in the polarized PDF.
    lambda_pol : float
        Exponent controlling intermediate-x behavior in the polarized PDF.
    j : float
        Conformal spin.
    alpha_p : float
        Regge slope.
    t : float
        Mandelstam t.
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"

    Returns
    -------
    float
        Value of the Reggeized polarized moment integral.
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

    if evolution_order == "lo":
        result = term1
    elif evolution_order == "nlo":
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

def pdf_regge_error(A_pdf,delta_A_pdf,eta_1,delta_eta_1,eta_2,delta_eta_2,epsilon,delta_epsilon,gamma_pdf,delta_gamma_pdf,j,alpha_p,t, error_type="central"):
    """
    Compute the error (added in quadrature) of the Reggeized PDF moment:

        f = ∫ dx x**(j - 1 - alpha_p * t) * pdf(x)

    The error is computed using Gaussian quadrature.

    The PDF is parameterized as:

        pdf(x) = A_pdf * x**eta_1 * (1 - x)**eta_2 * (1 + epsilon * sqrt(x) + gamma_pdf * x)

    Parameters
    ----------
    A_pdf : float
        Normalization constant of the input PDF.
    delta_A_pdf : float
        Error in A_pdf.
    eta_1 : float
        Small-x exponent of the input PDF.
    delta_eta_1 : float
        Error in eta_1.
    eta_2 : float
        Large-x exponent of the input PDF.
    delta_eta_2 : float
        Error in eta_2.
    epsilon : float
        Coefficient of sqrt(x) in the input PDF.
    delta_epsilon : float
        Error in epsilon.
    gamma_pdf : float
        Coefficient of x in the input PDF.
    delta_gamma_pdf : float
        Error in gamma_pdf.
    j : float
        Conformal spin.
    alpha_p : float
        Regge slope.
    t : float
        Mandelstam t.

    Returns
    -------
    float
        0 if the central value is selected; +/- error if `error_type` is "plus" or "minus".

    Note
    ----
    Gaussian quadrature overshoots the error significantly at low-x
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

    sum_squared = Delta_A_pdf**2+Delta_eta_1**2+Delta_eta_2**2+Delta_epsilon**2+Delta_gamma_pdf**2
    result = abs(mp.sqrt(sum_squared))
    return result


def polarized_pdf_regge_error(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
                                       Delta_A_pdf,err_Delta_A_pdf,alpha,err_alpha,gamma_pol,err_gamma_pol, lambda_pol,err_lambda_pol,
                                       j,alpha_p,t,evolution_order="nlo", error_type="central"):
    """
    Compute the error (added in quadrature) of the Reggeized **polarized** PDF moment:

        \int dx x**(j - 1 - alpha_p * t) * polarized_pdf(x)

    where the polarized PDF is of the form:

        polarized_pdf(x) = Delta_A_pdf * x**alpha * (1 + gamma_pol * x**lambda_pol) * pdf(x)

    and the unpolarized input PDF is:

        pdf(x) = A_pdf * x**eta_1 * (1 - x)**eta_2 * (1 + epsilon * sqrt(x) + gamma_pdf * x)

    The error is computed using Gaussian quadrature.

    Parameters
    ----------
    A_pdf : float
        Normalization constant of the unpolarized input PDF.
    eta_1 : float
        Small-x exponent of the unpolarized PDF.
    eta_2 : float
        Large-x exponent of the unpolarized PDF.
    epsilon : float
        Coefficient of sqrt(x) in the unpolarized PDF.
    gamma_pdf : float
        Coefficient of x in the unpolarized PDF.
    Delta_A_pdf : float
        Normalization constant of the polarized PDF.
    err_Delta_A_pdf : float
        Error in Delta_A_pdf.
    alpha : float
        Small-x exponent of the polarized PDF.
    err_alpha : float
        Error in alpha.
    gamma_pol : float
        Coefficient of x**lambda_pol in the polarized PDF.
    err_gamma_pol : float
        Error in gamma_pol.
    lambda_pol : float
        Exponent controlling intermediate-x behavior in the polarized PDF.
    err_lambda_pol : float
        Error in lambda_pol.
    j : float
        Conformal spin.
    alpha_p : float
        Regge slope.
    t : float
        Mandelstam variable t (typically < 0 in the physical region).
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central" Default is central

    Returns
    -------
    float
        0 if the central value is selected; ±error if `error_type` is "plus" or "minus".

    Notes
    -----
    - Gaussian quadrature overshoots the error significantly at low-x. So we manually make it smaller
      to reflect the correct error quoted in the input parametrization.
    """
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
            if evolution_order == "lo":
                term6 = 0
            elif evolution_order == "nlo":
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
            if evolution_order == "lo":
                term12 = 0
            elif evolution_order == "nlo":
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
            
            if evolution_order == "lo":
                term4 = 0
            elif evolution_order == "nlo":
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
            # Same for lo and nlo
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
    
    sum_squared = Delta_Delta_A_pdf**2+Delta_alpha**2+Delta_gamma_pol**2+Delta_lambda_pol**2
    result = abs(mp.sqrt(sum_squared))
    return result

def uv_pdf_regge(j,eta,alpha_p,t, evolution_order = "nlo", error_type="central"):
    """
    Result of the integral of the Reggeized uv(x) PDF using the given parameters and selected error type.

    Parameters
    ----------
    j : float
        Conformal spin.
    eta : float
        Skewness parameter (currently a placeholder, not used).
    alpha_p : float
        Regge slope.
    t : float or array_like
        Mandelstam t.
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"

    Returns
    -------
    float
        The value of the Reggeized integral of the uv PDF based on the selected parameters and error type.
    
    Note
    ----
    eta for the non-singlet sector is currently only a placeholder
    """
    # Check type
    hp.check_error_type(error_type)
    
    # Get the column index corresponding to the error_type
    error_col_index = hp.ERROR_MAP.get(error_type) 

    # Extracting parameter values
    A_pdf     = MSTW_PDF["A_u"][evolution_order][0]
    eta_1     = MSTW_PDF["eta_1"][evolution_order][0]
    eta_2     = MSTW_PDF["eta_2"][evolution_order][0]
    epsilon   = MSTW_PDF["epsilon_u"][evolution_order][0]
    gamma_pdf = MSTW_PDF["gamma_u"][evolution_order][0]

    pdf = pdf_regge(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,j,alpha_p,t)

    if error_type != "central":
    # Extracting errors
        delta_A_pdf  = MSTW_PDF["A_u"][evolution_order][error_col_index]
        delta_eta_1 = MSTW_PDF["eta_1"][evolution_order][error_col_index]
        delta_eta_2 = MSTW_PDF["eta_2"][evolution_order][error_col_index]
        delta_epsilon = MSTW_PDF["epsilon_u"][evolution_order][error_col_index]
        delta_gamma_pdf = MSTW_PDF["gamma_u"][evolution_order][error_col_index]

        
        pdf_error = pdf_regge_error(A_pdf,delta_A_pdf,eta_1,delta_eta_1,eta_2,delta_eta_2,epsilon,delta_epsilon,gamma_pdf,delta_gamma_pdf,j,alpha_p,t,error_type)
        return pdf, pdf_error
    else:
        return pdf, 0

def dv_pdf_regge(j,eta,alpha_p,t, evolution_order = "nlo", error_type="central"):
    """
    Result of the integral of the Reggeized dv(x) PDF using the given parameters and selected error type.

    Parameters
    ----------
    j : float
        Conformal spin.
    eta : float
        Skewness parameter (currently a placeholder, not used).
    alpha_p : float
        Regge slope.
    t : float or array_like
        Mandelstam t.
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"

    Returns
    -------
    float
        The value of the Reggeized integral of the dv PDF based on the selected parameters and error type.
    
    Note
    ----
    eta for the non-singlet sector is currently only a placeholder
    """
    # Check type
    hp.check_error_type(error_type)
    
    # Get the column index corresponding to the error_type
    error_col_index = hp.ERROR_MAP.get(error_type)

    A_pdf     = MSTW_PDF["A_d"][evolution_order][0]
    eta_1     = MSTW_PDF["eta_3"][evolution_order][0]
    eta_2     = MSTW_PDF["eta_2"][evolution_order][0] + MSTW_PDF["eta_4-eta_2"][evolution_order][0]  # eta_4 ≡ eta_2 + (eta_4 - eta_2)
    epsilon   = MSTW_PDF["epsilon_d"][evolution_order][0]
    gamma_pdf = MSTW_PDF["gamma_d"][evolution_order][0]

    pdf = pdf_regge(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,j,alpha_p,t)

    if error_type != "central":
        # Extracting errors
        delta_A_pdf  = MSTW_PDF["A_d"][evolution_order][error_col_index]
        delta_eta_1 = MSTW_PDF["eta_3"][evolution_order][error_col_index]
        delta_eta_2 = np.sign(MSTW_PDF["eta_4-eta_2"][evolution_order][error_col_index]) * np.sqrt(MSTW_PDF["eta_4-eta_2"][evolution_order][error_col_index]**2 + MSTW_PDF["eta_2"][evolution_order][error_col_index]**2)
        delta_epsilon = MSTW_PDF["epsilon_d"][evolution_order][error_col_index]
        delta_gamma_pdf = MSTW_PDF["gamma_d"][evolution_order][error_col_index]


        pdf_error = pdf_regge_error(A_pdf,delta_A_pdf,eta_1,delta_eta_1,eta_2,delta_eta_2,epsilon,delta_epsilon,gamma_pdf,delta_gamma_pdf,j,alpha_p,t,error_type)
        return pdf, pdf_error
    else:
        return pdf, 0

def sv_pdf_regge(j,eta,alpha_p,t, evolution_order = "nlo", error_type="central"):
    """
    Result of the integral of the Reggeized sv(x) PDF using the given parameters and selected error type.

    Parameters
    ----------
    j : float
        Conformal spin.
    eta : float
        Skewness parameter (currently a placeholder, not used).
    alpha_p : float
        Regge slope.
    t : float or array_like
        Mandelstam t.
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"

    Returns
    -------
    float
        The value of the Reggeized integral of the sv PDF based on the selected parameters and error type.
    
    Note
    ----
    eta dependence is treated separately and currently only a placeholder.
    """
    # eta_1 = delta_minus, eta_2 = eta_minus, epsilon = 0, gamma = 0
    def sv_pdf_regge(A_m,delta_m,eta_m,x_0,j,alpha_p,t):
        frac = mp.gamma(1+eta_m)*mp.gamma(j+delta_m-1-alpha_p*t)/(x_0*mp.gamma(1+delta_m+eta_m+j-alpha_p*t))
        result = -A_m*(j-1-delta_m*(x_0-1)-x_0*(eta_m+j-alpha_p*t)-alpha_p*t)*frac
        return result
    def sv_pdf_regge_error(A_m,delta_A_m,delta_m,delta_delta_m,eta_m,delta_eta_m,x_0,delta_x_0,j,alpha_p,t, error_type="central"):
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

        result = abs(mp.sqrt(Delta_A_m**2+Delta_delta_m**2+Delta_eta_m**2+Delta_x_0**2))
        return result
        
    # Check type
    hp.check_error_type(error_type)
    
    error_col_index = hp.ERROR_MAP.get(error_type)

    # Extracting parameter values
    A_m = MSTW_PDF["A_-"][evolution_order][0]
    delta_m = MSTW_PDF["delta_-"][evolution_order][0]
    eta_m = MSTW_PDF["eta_-"][evolution_order][0]
    x_0 = MSTW_PDF["x_0"][evolution_order][0]

    pdf = sv_pdf_regge(A_m,delta_m,eta_m,x_0,j,alpha_p,t)

    if error_type != "central":
    # Extracting errors
        delta_A_m  = MSTW_PDF["A_-"][evolution_order][error_col_index]
        delta_delta_m = MSTW_PDF["delta_-"][evolution_order][error_col_index]
        delta_eta_m = MSTW_PDF["eta_-"][evolution_order][error_col_index]
        delta_x_0 = MSTW_PDF["x_0"][evolution_order][error_col_index]

        pdf_error = sv_pdf_regge_error(A_m,delta_A_m,delta_m,delta_delta_m,eta_m,delta_eta_m,x_0,delta_x_0,j,alpha_p,t, error_type)
        return pdf, pdf_error
    else:
        return pdf, 0

def S_pdf_regge(j,eta,alpha_p,t, evolution_order = "nlo", error_type="central"):
    """
    Result of the integral of the Reggeized S(x) = 2 (ubar(x) + dbar(x)) + s(x) + sbar(x)  PDF using the given parameters and selected error type.

    Parameters
    ----------
    j : float
        Conformal spin.
    eta : float
        Skewness parameter (currently a placeholder, not used).
    alpha_p : float
        Regge slope.
    t : float or array_like
        Mandelstam t.
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"

    Returns
    -------
    float
        The value of the Reggeized integral of the S PDF based on the selected parameters and error type.
    
    Note
    ----
    eta dependence is treated separately and currently only a placeholder.
    """
    # Check type
    hp.check_error_type(error_type)
    
    error_col_index = hp.ERROR_MAP.get(error_type)

    A_pdf      = MSTW_PDF["A_S"][evolution_order][0]
    eta_1      = MSTW_PDF["delta_S"][evolution_order][0]
    eta_2      = MSTW_PDF["eta_S"][evolution_order][0]
    epsilon    = MSTW_PDF["epsilon_S"][evolution_order][0]
    gamma_pdf  = MSTW_PDF["gamma_S"][evolution_order][0]

    pdf = pdf_regge(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,j,alpha_p,t)

    if error_type != "central":
    # Extracting errors
        delta_A_pdf  = MSTW_PDF["A_S"][evolution_order][error_col_index]
        delta_eta_1 = MSTW_PDF["delta_S"][evolution_order][error_col_index]
        delta_eta_2 = MSTW_PDF["eta_S"][evolution_order][error_col_index]
        delta_epsilon = MSTW_PDF["epsilon_S"][evolution_order][error_col_index]
        delta_gamma_pdf = MSTW_PDF["gamma_S"][evolution_order][error_col_index]

        pdf_error = pdf_regge_error(A_pdf,delta_A_pdf,eta_1,delta_eta_1,eta_2,delta_eta_2,epsilon,delta_epsilon,gamma_pdf,delta_gamma_pdf,j,alpha_p,t,error_type)
        return pdf, pdf_error
    else:
        return pdf, 0

def s_plus_pdf_regge(j,eta,alpha_p,t, evolution_order = "nlo", error_type="central"):
    """
    Result of the integral of the Reggeized s_+(x) PDF using the given parameters and selected error type.

    Parameters
    ----------
    j : float
        Conformal spin.
    eta : float
        Skewness parameter (currently a placeholder, not used).
    alpha_p : float
        Regge slope.
    t : float or array_like
        Mandelstam t.
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"

    Returns
    -------
    float
        The value of the Reggeized integral of the s_+ PDF based on the selected parameters and error type.
    
    Note
    ----
    eta dependence is treated separately and currently only a placeholder.
    """
    # Check type
    hp.check_error_type(error_type)

    error_col_index = hp.ERROR_MAP.get(error_type)

    A_pdf      = MSTW_PDF["A_+"][evolution_order][0]
    eta_1      = MSTW_PDF["delta_S"][evolution_order][0]
    eta_2      = MSTW_PDF["eta_+"][evolution_order][0]
    epsilon    = MSTW_PDF["epsilon_S"][evolution_order][0]
    gamma_pdf  = MSTW_PDF["gamma_S"][evolution_order][0]

    pdf = pdf_regge(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,j,alpha_p,t)

    if error_type != "central":
        # Extracting errors
        delta_A_pdf      = MSTW_PDF["A_+"][evolution_order][error_col_index]
        delta_eta_1      = MSTW_PDF["delta_S"][evolution_order][error_col_index]
        delta_eta_2      = MSTW_PDF["eta_+"][evolution_order][error_col_index]
        delta_epsilon    = MSTW_PDF["epsilon_S"][evolution_order][error_col_index]
        delta_gamma_pdf  = MSTW_PDF["gamma_S"][evolution_order][error_col_index]

        pdf_error = pdf_regge_error(A_pdf,delta_A_pdf,eta_1,delta_eta_1,eta_2,delta_eta_2,epsilon,delta_epsilon,gamma_pdf,delta_gamma_pdf,j,alpha_p,t,error_type)
        return pdf, pdf_error
    else:
        return pdf, 0

def Delta_pdf_regge(j,eta,alpha_p,t, evolution_order = "nlo", error_type="central"):
    """
    Result of the integral of the Reggeized Delta(x) = ubar(x) - dbar(x) PDF using the given parameters and selected error type.

    Parameters
    ----------
    j : float
        Conformal spin.
    eta : float
        Skewness parameter (currently a placeholder, not used).
    alpha_p : float
        Regge slope.
    t : float or array_like
        Mandelstam t.
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"

    Returns
    -------
    float
        The value of the Reggeized integral of the Delta PDF based on the selected parameters and error type.
    
    Note
    ----
    eta dependence is treated separately and currently only a placeholder.
    """
    def Delta_pdf_regge(A_Delta,eta_Delta,eta_S,gamma_Delta,delta_Delta,j,alpha_p,t):
        frac_1 = (2+eta_Delta+eta_S+j-alpha_p*t)*(3+eta_Delta+eta_S+j-alpha_p*t)
        frac_2 = mp.gamma(3+eta_S)*mp.gamma(j+eta_Delta-1-alpha_p*t)/(mp.gamma(2+eta_Delta+eta_S+j-alpha_p*t))
        result = A_Delta*(1+((delta_Delta*(eta_Delta+j-alpha_p*t)+gamma_Delta*(3+eta_Delta+eta_S+j-alpha_p*t))*(eta_Delta+j-1+alpha_p*t))/frac_1)*frac_2
        return result
    def Delta_pdf_regge_error(A_Delta,delta_A_Delta,eta_Delta,delta_eta_Delta,eta_S,delta_eta_S,gamma_Delta,delta_gamma_Delta,delta_Delta,delta_delta_Delta,j,alpha_p,t, error_type="central"):
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
    
    # Get the column index corresponding to the error_type
    error_col_index = hp.ERROR_MAP.get(error_type) 

    A_Delta     = MSTW_PDF["A_Delta"][evolution_order][0]
    eta_Delta   = MSTW_PDF["eta_Delta"][evolution_order][0]
    eta_S       = MSTW_PDF["eta_S"][evolution_order][0]
    delta_Delta = MSTW_PDF["delta_Delta"][evolution_order][0]
    gamma_Delta = MSTW_PDF["gamma_Delta"][evolution_order][0]


    pdf = Delta_pdf_regge(A_Delta,eta_Delta,eta_S,gamma_Delta,delta_Delta,j,alpha_p,t)

    if error_type != "central":
        # Extracting errors
        delta_A_Delta      = MSTW_PDF["A_Delta"][evolution_order][error_col_index]
        delta_eta_Delta    = MSTW_PDF["eta_Delta"][evolution_order][error_col_index]
        delta_eta_S        = MSTW_PDF["eta_S"][evolution_order][error_col_index]
        delta_delta_Delta  = MSTW_PDF["delta_Delta"][evolution_order][error_col_index]
        delta_gamma_Delta  = MSTW_PDF["gamma_Delta"][evolution_order][error_col_index]

        pdf_error = Delta_pdf_regge_error(A_Delta,delta_A_Delta,eta_Delta,delta_eta_Delta,eta_S,delta_eta_S,gamma_Delta,delta_gamma_Delta,delta_Delta,delta_delta_Delta,j,alpha_p,t, error_type)
        return pdf, pdf_error
    else:
        return pdf, 0

def gluon_pdf_regge(j,eta,alpha_p,t, evolution_order = "nlo", error_type="central"):
    """
    Result of the integral of the Reggeized g(x) PDF using the given parameters and selected error type.

    Parameters
    ----------
    j : float
        Conformal spin.
    eta : float
        Skewness parameter (currently a placeholder, not used).
    alpha_p : float
        Regge slope.
    t : float or array_like
        Mandelstam t.
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"

    Returns
    -------
    float
        The value of the Reggeized integral of the g PDF based on the selected parameters and error type.
    
    Note
    ----
    eta dependence is treated separately and currently only a placeholder.
    """
    # Check type
    hp.check_error_type(error_type)
    
    # Get the column index corresponding to the error_type
    error_col_index = hp.ERROR_MAP.get(error_type) 

    A_pdf     = MSTW_PDF["A_g"][evolution_order][0]
    eta_1     = MSTW_PDF["delta_g"][evolution_order][0]
    eta_2     = MSTW_PDF["eta_g"][evolution_order][0]
    epsilon   = MSTW_PDF["epsilon_g"][evolution_order][0]
    gamma_pdf = MSTW_PDF["gamma_g"][evolution_order][0]


    pdf = pdf_regge(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,j,alpha_p,t)

    # Additional term at nlo and nnlo
    if evolution_order != "lo":
        # Get row index of entry
        A_pdf_prime     = MSTW_PDF["A_g'"][evolution_order][0]
        eta_1_prime     = MSTW_PDF["delta_g'"][evolution_order][0]
        eta_2_prime     = MSTW_PDF["eta_g'"][evolution_order][0]

        nlo_term = A_pdf_prime * (eta_1_prime + eta_2_prime + j - alpha_p * t) * mp.gamma(j - alpha_p *t + eta_1_prime-1)*mp.gamma(1+eta_2_prime)/\
                mp.gamma(j-alpha_p*t+eta_1_prime+eta_2_prime + 1)
        pdf += nlo_term
    if error_type != "central":
    # Extracting errors
        delta_A_pdf      = MSTW_PDF["A_g"][evolution_order][error_col_index]
        delta_eta_1      = MSTW_PDF["delta_g"][evolution_order][error_col_index]
        delta_eta_2      = MSTW_PDF["eta_g"][evolution_order][error_col_index]
        delta_epsilon    = MSTW_PDF["epsilon_g"][evolution_order][error_col_index]
        delta_gamma_pdf  = MSTW_PDF["gamma_g"][evolution_order][error_col_index]

        pdf_error = pdf_regge_error(A_pdf,delta_A_pdf,eta_1,delta_eta_1,eta_2,delta_eta_2,epsilon,delta_epsilon,gamma_pdf,delta_gamma_pdf,j,alpha_p,t,error_type)
        if evolution_order != "lo":
            delta_A_prime_pdf     = MSTW_PDF["A_g'"][evolution_order][error_col_index]
            delta_eta_1_prime     = MSTW_PDF["delta_g'"][evolution_order][error_col_index]
            delta_eta_2_prime     = MSTW_PDF["eta_g'"][evolution_order][error_col_index]

            dpdf_dA = mp.gamma(j - alpha_p *t + eta_1_prime-1)*mp.gamma(1+eta_2_prime)/\
                mp.gamma(j-alpha_p*t+eta_1_prime+eta_2_prime)
            dpdf_deta_1 = A_pdf_prime * \
                mp.gamma(eta_2_prime + 1) * \
                mp.gamma(eta_1_prime + j - alpha_p * t - 1) * \
                ( \
                    (eta_1_prime + eta_2_prime + j - alpha_p * t) * \
                        mp.digamma(eta_1_prime + j - alpha_p * t - 1) - \
                    (eta_1_prime + eta_2_prime + j - alpha_p * t) * \
                        mp.digamma(eta_1_prime + eta_2_prime + j - alpha_p * t + 1) + \
                    1 \
                ) / \
                mp.gamma(eta_1_prime + eta_2_prime + j - alpha_p * t + 1)
            dpdf_deta_2 = (A_pdf_prime * mp.gamma(1+eta_2_prime) * mp.gamma(j - alpha_p *t + eta_1_prime-1)/\
              mp.gamma(j-alpha_p*t+eta_1_prime+eta_2_prime + 1) * \
             (1 + (eta_1_prime + eta_2_prime + j -alpha_p * t) * (mp.digamma(eta_2_prime + 1) - mp.digamma(eta_1_prime + eta_2_prime + j -alpha_p * t + 1)))
            )
            Delta_A = dpdf_dA * delta_A_prime_pdf
            Delta_eta_1 = dpdf_deta_1 * delta_eta_1_prime
            Delta_eta_2 = dpdf_deta_2 * delta_eta_2_prime
            sum_squared = Delta_A**2+Delta_eta_1**2+Delta_eta_2**2
            result = abs(mp.sqrt(sum_squared))
            pdf_error += result
        return pdf, pdf_error
    else:
        return pdf, 0

def u_minus_d_pdf_regge(j,eta,t, evolution_order = "nlo", error_type="central"):
    """ Currently only experimental function that does not set ubar=dbar"""
    # Check type
    hp.check_error_type(error_type)
    # Value optmized for range -t < 5 GeV
    alpha_prime = 0.675606
    # Normalize to 1 at t = 0
    return 1.107*(uv_pdf_regge(j,eta,alpha_prime,t,evolution_order,error_type)
                    - dv_pdf_regge(j,eta,alpha_prime,t,evolution_order,error_type)
                    - Delta_pdf_regge(j,alpha_prime,t,evolution_order,error_type))

def u_plus_d_pdf_regge(j,eta,t, evolution_order = "nlo", error_type="central"):
    """ Currently only experimental function that does not set ubar=dbar"""
    # Check type
    hp.check_error_type(error_type)
    # Value optmized for range -t < 5 GeV
    alpha_prime = 0.949256
    # Normalize to 1 at t = 0
    return 0.973*(uv_pdf_regge(j,eta,alpha_prime,t,evolution_order,error_type)
                    + dv_pdf_regge(j,eta,alpha_prime,t,evolution_order,error_type)
                    + Delta_pdf_regge(j,alpha_prime,t,evolution_order,error_type))

def polarized_uv_pdf_regge(j,eta,alpha_p,t, evolution_order = "nlo", error_type="central"):
    """
    Result of the integral of the Reggeized polarized uv(x) PDF using the given parameters and selected error type.

    Parameters
    ----------
    j : float
        Conformal spin.
    eta : float
        Skewness parameter (currently a placeholder, not used).
    alpha_p : float
        Regge slope.
    t : float or array_like
        Mandelstam t.
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"

    Returns
    -------
    float
        The value of the Reggeized integral of the polarized uv PDF based on the selected parameters and error type.
    
    Note
    ----
    eta dependence is treated separately and currently only a placeholder.
    """
    # Check type
    hp.check_error_type(error_type)

    # Get the column index corresponding to the error_type
    error_col_index = hp.ERROR_MAP.get(error_type, 0)  # Default to 'central' if error_type is invalid

    A_pdf     = MSTW_PDF["A_u"][evolution_order][0]
    eta_1     = MSTW_PDF["eta_1"][evolution_order][0]
    eta_2     = MSTW_PDF["eta_2"][evolution_order][0]
    epsilon   = MSTW_PDF["epsilon_u"][evolution_order][0]
    gamma_pdf = MSTW_PDF["gamma_u"][evolution_order][0]

    delta_A_pdf = AAC_PDF["Delta_A_u"][evolution_order][0]
    alpha       = AAC_PDF["alpha_u"][evolution_order][0]
    gamma_pol   = AAC_PDF["Delta_gamma_u"][evolution_order][0]
    lambda_pol  = AAC_PDF["Delta_lambda_u"][evolution_order][0]

    pdf = polarized_pdf_regge(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
                                        delta_A_pdf,alpha,gamma_pol,lambda_pol,
                                        j,alpha_p,t,evolution_order)
    if error_type != "central":
        err_delta_A_pdf = AAC_PDF["Delta_A_u"][evolution_order][error_col_index]
        err_alpha       = AAC_PDF["alpha_u"][evolution_order][error_col_index]
        err_gamma_pol   = AAC_PDF["Delta_gamma_u"][evolution_order][error_col_index]
        err_lambda_pol  = AAC_PDF["Delta_lambda_u"][evolution_order][error_col_index]

        pdf_error = polarized_pdf_regge_error(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
                                        delta_A_pdf,err_delta_A_pdf,alpha,err_alpha,gamma_pol,err_gamma_pol,lambda_pol,err_lambda_pol,
                                        j,alpha_p,t,evolution_order,error_type)
        return pdf, pdf_error
    else:
        return pdf, 0

def polarized_dv_pdf_regge(j,eta,alpha_p,t, evolution_order = "nlo", error_type="central"):
    """
    Result of the integral of the Reggeized polarized dv(x) PDF using the given parameters and selected error type.

    Parameters
    ----------
    j : float
        Conformal spin.
    eta : float
        Skewness parameter (currently a placeholder, not used).
    alpha_p : float
        Regge slope.
    t : float or array_like
        Mandelstam t.
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"

    Returns
    -------
    float
        The value of the Reggeized integral of the polarized dv PDF based on the selected parameters and error type.
    
    Note
    ----
    eta dependence is treated separately and currently only a placeholder.
    """
    # Check type
    hp.check_error_type(error_type)

    # Get the column index corresponding to the error_type
    error_col_index = hp.ERROR_MAP.get(error_type, 0)  # Default to 'central' if error_type is invalid

    A_pdf     = MSTW_PDF["A_d"][evolution_order][0]
    eta_1     = MSTW_PDF["eta_3"][evolution_order][0]
    eta_2     = MSTW_PDF["eta_2"][evolution_order][0] + MSTW_PDF["eta_4-eta_2"][evolution_order][0]  # eta_4 ≡ eta_2 + (eta_4 - eta_2)
    epsilon   = MSTW_PDF["epsilon_d"][evolution_order][0]
    gamma_pdf = MSTW_PDF["gamma_d"][evolution_order][0]

    Delta_A_pdf = AAC_PDF["Delta_A_d"][evolution_order][0]
    alpha       = AAC_PDF["alpha_d"][evolution_order][0]
    gamma_pol   = AAC_PDF["Delta_gamma_d"][evolution_order][0]
    lambda_pol  = AAC_PDF["Delta_lambda_d"][evolution_order][0]

    pdf = polarized_pdf_regge(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
                                        Delta_A_pdf,alpha,gamma_pol,lambda_pol,
                                        j,alpha_p,t,evolution_order)
    if error_type != "central":
        err_delta_A_pdf = AAC_PDF["Delta_A_d"][evolution_order][error_col_index]
        err_alpha       = AAC_PDF["alpha_d"][evolution_order][error_col_index]
        err_gamma_pol   = AAC_PDF["Delta_gamma_d"][evolution_order][error_col_index]
        err_lambda_pol  = AAC_PDF["Delta_lambda_d"][evolution_order][error_col_index]

        pdf_error = polarized_pdf_regge_error(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
                                        Delta_A_pdf,err_delta_A_pdf,alpha,err_alpha,gamma_pol,err_gamma_pol,lambda_pol,err_lambda_pol,
                                        j,alpha_p,t,evolution_order,error_type)
        return pdf, pdf_error
    else:
        return pdf, 0

def polarized_S_pdf_regge(j,eta,alpha_p,t, evolution_order = "nlo", error_type="central"):
    """
    Result of the integral of the Reggeized polarized S(x) = 2 (ubar(x) + dbar(x)) + s(x) + sbar(x)  PDF using the given parameters and selected error type.

    Parameters
    ----------
    j : float
        Conformal spin.
    eta : float
        Skewness parameter (currently a placeholder, not used).
    alpha_p : float
        Regge slope.
    t : float or array_like
        Mandelstam t.
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"

    Returns
    -------
    float
        The value of the Reggeized integral of the polarized S PDF based on the selected parameters and error type.
    
    Note
    ----
    eta dependence is treated separately and currently only a placeholder.
    """
    # Check type
    hp.check_error_type(error_type)
    
    # Get the column index corresponding to the error_type
    error_col_index = hp.ERROR_MAP.get(error_type, 0)  # Default to 'central' if error_type is invalid

    A_pdf     = MSTW_PDF["A_S"][evolution_order][0]
    eta_1     = MSTW_PDF["delta_S"][evolution_order][0]
    eta_2     = MSTW_PDF["eta_S"][evolution_order][0]
    epsilon   = MSTW_PDF["epsilon_S"][evolution_order][0]
    gamma_pdf = MSTW_PDF["gamma_S"][evolution_order][0]

    delta_A_pdf = AAC_PDF["Delta_A_S"][evolution_order][0]
    alpha       = AAC_PDF["alpha_S"][evolution_order][0]
    gamma_pol   = AAC_PDF["Delta_gamma_S"][evolution_order][0]
    lambda_pol  = AAC_PDF["Delta_lambda_S"][evolution_order][0]

    pdf = polarized_pdf_regge(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
                                        delta_A_pdf,alpha,gamma_pol,lambda_pol,
                                        j,alpha_p,t,evolution_order)
    if error_type != "central":
        err_delta_A_pdf = AAC_PDF["Delta_A_S"][evolution_order][error_col_index]
        err_alpha       = AAC_PDF["alpha_S"][evolution_order][error_col_index]
        err_gamma_pol   = AAC_PDF["Delta_gamma_S"][evolution_order][error_col_index]
        err_lambda_pol  = AAC_PDF["Delta_lambda_S"][evolution_order][error_col_index]
        pdf_error = polarized_pdf_regge_error(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
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

def polarized_gluon_pdf_regge(j,eta,alpha_p,t, evolution_order = "nlo", error_type="central"):
    """
    Result of the integral of the Reggeized polarized g(x) PDF using the given parameters and selected error type.

    Parameters
    ----------
    j : float
        Conformal spin.
    eta : float
        Skewness parameter (currently a placeholder, not used).
    alpha_p : float
        Regge slope.
    t : float or array_like
        Mandelstam t.
    evolution_order : str, optional
        "lo", "nlo",... . Default is "nlo"
    error_type : str, optional
        Choose "central", upper ("plus") or lower ("minus") value for input PDF parameters. Default is "central"

    Returns
    -------
    float
        The value of the Reggeized integral of the polarized g PDF based on the selected parameters and error type.
    
    Note
    ----
    eta dependence is treated separately and currently only a placeholder.
    """
    # Check type
    hp.check_error_type(error_type)
    
    # Get the column index corresponding to the error_type
    error_col_index = hp.ERROR_MAP.get(error_type, 0)  # Default to 'central' if error_type is invalid

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

    pdf = polarized_pdf_regge(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
                                        delta_A_pdf,alpha,gamma_pol,lambda_pol,
                                        j,alpha_p,t,evolution_order)
    if evolution_order != "lo":
        # Additional gluon contribution at nlo and nnlo that is not of the lo form
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

        pdf_error = polarized_pdf_regge_error(A_pdf,eta_1,eta_2,epsilon,gamma_pdf,
                                        delta_A_pdf,err_delta_A_pdf,alpha,err_alpha,gamma_pol,err_gamma_pol,lambda_pol,err_lambda_pol,
                                        j,alpha_p,t,evolution_order,error_type)
        if evolution_order != "lo":
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
