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

# Conjugate twice to circumvent cut
def power_minus_1(j):
    if mp.im(j) < 0:
        result = mp.mpc(-1,0)**(mp.conj(j))
        return mp.conj(result)
    else:
        result = mp.mpc(-1,0)**j
        return result
    
def fractional_finite_sum(func,k_0=1,k_1=1,epsilon=.2,k_range=10,n_k=300,alternating_sum=False, n_tuple = 1, plot_integrand = False,
                          full_range=True,trap=False,n_jobs=1,error_est=True):
    """
    Computes the fractional finite sume of a one-parameter function by either using fixed_quad (trap = False) or a trapezoidal rule (trap = True).
    Able to handle tuples of functions as input.

    Note
    ----
    One needs to be careful with certain functions in alternating sums as they introduce factors of (-1)**k_1
    which has diverges for im(k_1) < 0. Using fixed_quad automatically computes an error estimate. If it is large,
    this might be the reason why. When computing the non-diagonal part of the moment evolution, we utilize that the
    evolved moment f(j) satisfies conj(f(j)) = - f(j).
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
    if j < 0:
        print("Warning: sum called for negative iterator bound")
    sign = (-1 if l < 0 else 1)
    l_abs = abs(l)
    return sum((sign)**k/k**l_abs for k in range(1, j+1))

def harmonic_number(l,j,k_range=3,n_k=300,plot_integrand=False,trap=False):
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