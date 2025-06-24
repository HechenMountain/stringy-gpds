################################
###### Special functions, ######
#### Harmonic numbers, etc. ####
################################
import matplotlib.pyplot as plt
import numpy as np
from . import helpers as hp

from joblib import Parallel, delayed
from scipy.integrate import trapezoid, fixed_quad

# mpmath precision set in config
from . config import mp
    
def fractional_finite_sum(func,k_0=1,k_1=1,epsilon=.2,k_range=10,n_k=300,alternating_sum=False, n_tuple = 1, plot_integrand = False,
                          full_range=True,trap=False,n_jobs=1,error_est=True):
    """
    Computes a fractional finite sum of a one-parameter function using either Gauss-Legendre quadrature 
    (`trap=False`) or a trapezoidal rule (`trap=True`). 
    Can also handle tuples of functions and parallel evaluation if trapezoidal rule is used.

    Parameters
    ----------
    func : callable or tuple of callables
        The function(s) to be summed over. Must be callable with a single parameter.
    k_0 : float, optional
        Lower bound of the summation interval. Default is 1.
    k_1 : float, optional
        Upper bound of the summation interval. Default is 1.
    epsilon : float, optional
        Shift away from pole.
    k_range : int, optional
        Controls range of integration. Default is 10.
    n_k : int, optional
        Number of summation/integration points. Default is 300.
    alternating_sum : bool, optional
        Whether the sum has alternating sign. Default is False.
    n_tuple : int, optional
        Number of functions in the tuple (if `func` is a tuple). Default is 1.
    plot_integrand : bool, optional
        If True, plot the integrand. Default is False.
    full_range : bool, optional
        If True, compute over the full integration/summation range. Default is True.
        This compactifies using k = Tan(x)
    trap : bool, optional
        If True, use trapezoidal rule; otherwise, use fixed Gauss-Legendre quadrature. Default is False.
    n_jobs : int, optional
        Number of parallel jobs to use in trapezoidal evaluation. Default is 1.
    error_est : bool, optional
        If True and using Gauss quadrature, return integration error estimate. Default is True.

    Returns
    -------
    result : float or complex or tuple
        The evaluated fractional sum(s) for the input function(s). Type depends on input.
    
    Notes
    -----
    - When computing non-diagonal components in moment evolution, we may rely on the symmetry of the moment:
      conj(f(j)) = -f(j).
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
                    alt_sign = (-1)**(k_1 + 1 - mp.re(k_0))
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
            int_tmp, _ = fixed_quad(lambda k: integrand_quad(k), b1, b2, n = 150)
            if error_est:
                # Estimate error
                int_tmp_50, _ = fixed_quad(lambda k: integrand_quad(k), b1, b2, n = 80)
                err_tmp = abs(int_tmp - int_tmp_50)
                if err_tmp > 1e-2:
                    print(f"Warning: large error={err_tmp} detected in fractional_finite_sum")
            # Discard small imaginary part
            int_tmp = int_tmp.real if abs(int_tmp.imag) < 1e-6 else int_tmp 
            integral[i] += int_tmp
    else:
        # Trapezoidal rule
        k_vals = np.linspace(-k_range, k_range, n_k) 
        k_vals = k_vals - mp.im(k_0)

        k_vals_trig = mp.re(k_0) - epsilon + 1j * k_vals
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
            alt_sign = (-1)**(k_1 + 1 - mp.re(k_0))
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
                integral[i] += integral_real + 1j * integral_imag

    if n_tuple == 1:
        # Return a scalar for n_tuple = 1
        return integral[0]
    else:
        return integral
    
def polygamma_asymptotic(n, z, terms=5):
    """
    Asymptotic form of polygamma function.

    Parameters
    ----------
    n : int
        Index of polygamma function
    z : complex
        Argument of polygamma function
    term : int, optional
        Number of terms used in expansion. Default is 5
    
    Returns
    -------
    The asymptotically approximated value of the polygamma function.
    """
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

def harmonic_sum(l,j):
    """
    Compute the harmonic sum for integer j

    Parameters
    ----------
    l : int
        Index of harmonic numbers
    j : float or complex
        Argument of the harmonic number, here conformal spin.
    
    Returns
    -------
    float
        the value of the harmonic sum
    """
    if j < 0:
        print("Warning: sum called for negative iterator bound")
    sign = (-1 if l < 0 else 1)
    l_abs = abs(l)
    return sum((sign)**k/k**l_abs for k in range(1, j+1))

def harmonic_number(l,j,trap=False,k_range=3,n_k=100,plot_integrand=False):
    """
    Compute Harmonic number H(j) for complex or real conformal spin j.

    Parameters
    ----------
    l : int
        Index of harmonic numbers
    j : float or complex
        Argument of the harmonic number, here conformal spin.
    trap : bool, optional
        Whether to use trapezoidal rule for numerical integration in fractional_finite_sum. 
        Default is False.
    k_range : int, optional
        Controls the integration range for trapezoid
    n_k : int, optional
        Number of terms used for trapezoid integration Default is 100.

    Returns
    -------
    float or complex
        Value of the nested harmonic number H_{indices}(j).
    
    Note
    ----
    If abs(j) >= 25, the polygamma functions are approximated by their asymptotic
    expression for faster numerical convergence. Due to the appearance of (-1)**j 
    in some of the expressions below, the function diverges for Im(j) < 0
    and we use H_n(j*) = H_n(j)* instead
    """
    conjugate = False
    if j.imag < 0:
        j = mp.conj(j)
        conjugate = True

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
            result = (-1)**j * mp.lerchphi(-1, 1, 1 + j) - mp.log(2)
        else:
        # Approximate with asymptotic series expansion
        # Lerch transcendent [-1,1,j+1]
            lerch_phi = .5 * (polygamma_asymptotic(0,1+.5*j,terms=1) - polygamma_asymptotic(0,.5*(1+j),terms=1))
            result = (-1)**j * lerch_phi - mp.log(2)
    elif l < -1:
        m = abs(l)
        if abs(j) < 25:
            result = mp.polylog(m,-1) + (-1)**j * 2**(-m) * (
                mp.zeta(m,(j+1)/2) - mp.zeta(m,1+.5*j)
            )
        else: 
        # Approximate with asymptotic series expansion
            zeta_1 = polygamma_asymptotic(m-1,(j+1)/2,terms=1)
            zeta_2 = polygamma_asymptotic(m-1,1+.5*j,terms=1)
            result = mp.polylog(m,-1) + (-1)**j * 2**(-m) * (-1)**m / mp.factorial(m-1)* (
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
    
    if conjugate:
        return mp.conj(result)
    else:
        return result

# Generate interpolator
nested_harmonic_1_2_interpolation = hp.build_harmonic_interpolator([1,2])
nested_harmonic_1_m2_interpolation = hp.build_harmonic_interpolator([1,-2])
nested_harmonic_2_1_interpolation = hp.build_harmonic_interpolator([2,1])
# Pick interpolation
def nested_harmonic_interpolation(indices):
    """
    See documentation for nested_harmonic_number
    """
    indices = tuple(int(i) for i in indices)
    if indices == (1,2):
        return nested_harmonic_1_2_interpolation
    elif indices == (1,-2):
        return nested_harmonic_1_m2_interpolation
    elif indices == (2,1):
        return nested_harmonic_2_1_interpolation
    else:
        raise ValueError(f"Generated table for interpolation of indices = {indices} and include in module.")
    
def nested_harmonic_number(indices, j,interpolation=True,trap=False,n_k=100,k_range=10):
    """
    Compute nested harmonic sums H_{indices}(j) for complex or real conformal spin j.

    If `interpolation` is True, precomputed tabulated values are used to evaluate the sum.
    Otherwise, the sum is computed numerically with adjustable integration/summation strategy.

    Currently only supports strictly positive indices. If an intermediate index is negative, 
    the sum involves alternating and non-alternating components, which are not automatically handled.
    Use the supplied tables or generate then with Mathematica notebook NestedHarmonics.nb .

    Parameters
    ----------
    indices : array_like of int
        Indices of the nested harmonic sum (must be all > 0).
    j : float or complex
        Argument of the harmonic sum, here conformal spin.
    interpolation : bool, optional
        Whether to use precomputed interpolation tables. Default is True.
    trap : bool, optional
        Whether to use trapezoidal rule for numerical integration in fractional_finite_sum. 
        Default is False.
    n_k : int, optional
        Number of terms in the summation if computing explicitly. Default is 100.
    k_range : int, optional
        Controls the integration range for trapezoid

    Returns
    -------
    float or complex
        Value of the nested harmonic number H_{indices}(j).
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
            inner_sum = nested_harmonic_number(rest, i,interpolation=False,k_range=k_range,n_k=n_k,trap=trap)
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
            inner_sum = nested_harmonic_number(rest,i,interpolation=False,k_range=k_range,n_k=n_k,trap=trap)
            return factor * inner_sum
        result = fractional_finite_sum(func,k_1=j,alternating_sum=alternating_sum, n_k=n_k,k_range=k_range,trap=trap)
    return result
        
def d_weight(m,k,n):
    """
    Compute the weight between difference in harmonic sums.

    Note
    ----
    The variable N used in Nucl. Phys. B 889 (2014) 351-400 
    corresponds to j + 1 here.

    Parameters
    ----------
    m : int
        Power index of the harmonic weight.
    k : int
        Summation index.
    n : int
        Conformal spin j via N = j + 1.

    Returns
    -------
    float
        The value of the weight
    """
    result = 1/(n+k)**m
    return result 