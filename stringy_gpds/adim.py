################################
##### Anomalous dimensions #####
################################

from . import config as cfg
from . import helpers as hp
from . import special as sp
# mpmath precision set in config
from .config import mp

# Generate the interpolator
gamma_qq_lo_interpolation = hp.build_gamma_interpolator("qq","non_singlet_isovector","vector",evolution_order="lo")
def gamma_qq_lo(j,interpolation=True):
    if interpolation:
        interp = gamma_qq_lo_interpolation 
        return interp(j)
    
    # Belitsky (4.152)
    result = - cfg.C_F * (-4*mp.digamma(j+2)+4*mp.digamma(1)+2/((j+1)*(j+2))+3)
    return result

# Generate the interpolators
gamma_qq_non_singlet_vector_nlo_interpolation = hp.build_gamma_interpolator("qq","non_singlet_isovector","vector",evolution_order="nlo")
gamma_qq_non_singlet_axial_nlo_interpolation = hp.build_gamma_interpolator("qq","non_singlet_isovector","axial",evolution_order="nlo")
gamma_qq_singlet_vector_nlo_interpolation = hp.build_gamma_interpolator("qq","singlet","vector",evolution_order="nlo")
gamma_qq_singlet_axial_nlo_interpolation = hp.build_gamma_interpolator("qq","singlet","axial",evolution_order="nlo")

# Pick correct interpolation
def gamma_qq_nlo_interpolation(moment_type,evolve_type):
    if moment_type != "singlet":
        return gamma_qq_non_singlet_vector_nlo_interpolation if evolve_type == "vector" else gamma_qq_non_singlet_axial_nlo_interpolation
    elif moment_type == "singlet":
        return gamma_qq_singlet_vector_nlo_interpolation if evolve_type == "vector" else gamma_qq_singlet_axial_nlo_interpolation
    else:
        raise ValueError(f"Wrong moment_type {moment_type}")
# @cfg.memory.cache
def gamma_qq_nlo(j,moment_type="non_singlet_isovector",evolve_type="vector",interpolation=True):
    if interpolation:
        # interp = hp.build_gamma_interpolator("qq",moment_type,evolve_type,evolution_order="nlo")
        interp = gamma_qq_nlo_interpolation(moment_type,evolve_type)
        result = interp(j)
        return result
    
    # Nucl.Phys.B 691 (2004) 129-181
    # Note that N there is j + 1 here
    # p is +: N -> N + 1 -> j + 2
    # m is -: N -> N - 1 -> j
    s_1 = sp.harmonic_number(1,j+1)
    s_2 = sp.harmonic_number(2,j+1)
    s_3_p = sp.harmonic_number(3,j+2)
    s_3_m = sp.harmonic_number(3,j)
    s_m3 = sp.harmonic_number(-3,j+1)
    s_1_p = sp.harmonic_number(1,j+2)
    s_1_m = sp.harmonic_number(1,j)
    s_2_p = sp.harmonic_number(2,j+2)
    s_2_pp = sp.harmonic_number(2,j+3)
    s_2_m = sp.harmonic_number(2,j)
    s_1_m2_p = sp.nested_harmonic_number([1,-2],j+2)
    s_1_m2_m = sp.nested_harmonic_number([1,-2],j)
    s_1_2_p = sp.nested_harmonic_number([1,2],j+2)
    s_1_2_m = sp.nested_harmonic_number([1,2],j)
    s_2_1_p = sp.nested_harmonic_number([2,1],j+2)
    s_2_1_m = sp.nested_harmonic_number([2,1],j)

    if moment_type == "singlet":
        s_1_mm = sp.harmonic_number(1,j-1)
        s_1_pp = sp.harmonic_number(1,j+3)
    # q + qbar
    # Nucl.Phys.B 688 (2004) 101-134
    # Note different beta function convention
    # so we reverse the sign
    # (3.5)
    term1 = - 4 * cfg.C_A * cfg.C_F * (2 * s_3_p - 17/24 - 2 * s_m3 - 28/3 * s_1
                            + 151/18 * (s_1_m + s_1_p) + 2 * (s_1_m2_m + s_1_m2_p) - 11/6 * (s_2_m + s_2_p)
                            )
    # Factor 2 because we insert cfg.T_F
    term2 = - 8 * cfg.C_F * cfg.T_F * cfg.N_F * (1/12 + 4/3 * s_1 - (11/9 * (s_1_m + s_1_p) - 1/3*(s_2_m + s_2_p)) )
    term3 = - 4 * cfg.C_F**2 * (4*s_m3 + 2*s_1 + 2 * s_2 -3/8 + (s_2_m + 2 * s_3_m)
                            - ((s_1_m + s_1_p) + 4 * (s_1_m2_m +s_1_m2_p) + 2 * (s_1_2_m + s_1_2_p) + 2 * (s_2_1_m + s_2_1_p) + (s_3_m + s_3_p))
    )
    
    if moment_type != "singlet":
        # q - qbar
        term1 += - 16 * cfg.C_F * (cfg.C_F - .5 * cfg.C_A) * (
            (s_2_m - s_2_p) - (s_3_m - s_3_p)
            - 2 * (s_1_m + s_1_p - 2 * s_1)
        )
    # Add pure singlet piece
    elif moment_type == "singlet":
        if evolve_type == "vector":
            # Nucl.Phys.B 691 (2004) 129-181
            # Note different beta function convention
            # so we reverse the sign
            term1 += - 8 * cfg.C_F * cfg.T_F * cfg.N_F * (
                20/9 * (s_1_mm - s_1_m) - (56/9*(s_1_p - s_1_pp) + 8/3 * ((s_2_p - s_2_pp)) )
                + (8*(s_1 - s_1_p)-4 * (s_2 - s_2_p)) - (2 * (s_1_m - s_1_p) + (s_2_m - s_2_p) + 2 * (s_3_m - s_3_p))
            )
        # Constraint at j = 0 is 0
        if evolve_type == "axial":
            # Nucl.Phys.B 889 (2014) 351-400
            # Note different beta function convention
            # so we reverse the sign
            # (4.4)
            eta = 1/((j+1)*(j+2))
            d02 = sp.d_weight(2,0,j + 1)
            d03 = sp.d_weight(3,0,j + 1)
            term1 += - 4 * cfg.C_F * cfg.N_F * (-5 * eta + 3 * eta**2 + 2 * eta**3 + 4 * d02 - 4 * d03)
    else:
        raise ValueError(f"Wrong moment_type {moment_type}")
    
    result = term1 + term2 + term3
    return result

def gamma_qq(j,moment_type="non_singlet_isovector",evolve_type="vector",evolution_order="nlo",interpolation=True):
    """
    Returns conformal qq singlet or non-singlet anomalous dimension for conformal spin-j

    Arguments:
    - j (float): conformal spin
    - moment_type (str. optional): non_singlet_isovector, non_singlet_isoscalar, singlet
    - evolve_type (str. optional): vector or axial
    - evolution_order (str. optional): lo, nlo or nnlo
    - interpolation (bool, optional): Use tabulated values for interpolation (only beyond lo)
    """
    if evolution_order == "lo":
        return gamma_qq_lo(j,False)
    elif evolution_order == "nlo":
        return gamma_qq_nlo(j,moment_type,evolve_type,interpolation)
    else:
        raise ValueError(f"Wrong evolution_order {evolution_order}")

# Generate the interpolator
gamma_qg_vector_lo_interpolation = hp.build_gamma_interpolator("qg","singlet","vector",evolution_order="lo")
gamma_qg_axial_lo_interpolation = hp.build_gamma_interpolator("qg","singlet","axial",evolution_order="lo")
# Pick the correct interpolation
def gamma_qg_lo_interpolation(evolve_type):
    return gamma_qg_vector_lo_interpolation if evolve_type == "vector" else gamma_qg_axial_lo_interpolation

def gamma_qg_lo(j,  evolve_type = "vector",interpolation=True):
    if interpolation:
        interp = gamma_qg_lo_interpolation(evolve_type)
        return interp(j)
    
    # Note additional factor of j/6 at lo (see (K.1) in 0504030)
    if j == 0:
        j = 1e-12
    if evolve_type == "vector":
        result = -24*cfg.N_F*cfg.T_F*(j**2+3*j+4)/(j*(j+1)*(j+2)*(j+3))
    elif evolve_type == "axial":
        result = -24*cfg.N_F*cfg.T_F/((j+1)*(j+2))
    else:
        raise ValueError("evolve_type must be axial or vector")
    # Match forward to Wilson anomalous dimension
    result*=j/6
    return result

# Generate the interpolator
gamma_qg_singlet_vector_nlo_interpolation = hp.build_gamma_interpolator("qg","singlet","vector",evolution_order="nlo")
gamma_qg_singlet_axial_nlo_interpolation = hp.build_gamma_interpolator("qg","singlet","axial",evolution_order="nlo")
# Pick the correct interpolation
def gamma_qg_nlo_interpolation(evolve_type):
    return gamma_qg_singlet_vector_nlo_interpolation if evolve_type == "vector" else gamma_qg_singlet_axial_nlo_interpolation

def gamma_qg_nlo(j,  evolve_type = "vector",interpolation=True):
    if interpolation:
        # interp = hp.build_gamma_interpolator("qg","singlet",evolve_type,"nlo")
        interp = gamma_qg_nlo_interpolation(evolve_type)
        result = interp(j)
        return result  
    if j == 0:
        j = 1e-12
    if evolve_type == "vector":
        s_1 = sp.harmonic_number(1,j+1)
        s_2 = sp.harmonic_number(2,j+1)
        s_1_1_1 = sp.nested_harmonic_number([1,1,1],j+1)
        s_1_1_1_m = sp.nested_harmonic_number([1,1,1],j)
        s_1_1_1_p = sp.nested_harmonic_number([1,1,1],j+2)
        s_1_m2 = sp.nested_harmonic_number([1,-2],j+1)
        s_1_m2_m = sp.nested_harmonic_number([1,-2],j)
        s_1_m2_p = sp.nested_harmonic_number([1,-2],j+2)
        s_1_2 = sp.nested_harmonic_number([1,2],j+1)
        s_1_2_m = sp.nested_harmonic_number([1,2],j)
        s_1_2_p = sp.nested_harmonic_number([1,2],j+2)
        s_2_1 = sp.nested_harmonic_number([2,1],j+1)
        s_2_1_m = sp.nested_harmonic_number([2,1],j)
        s_2_1_p = sp.nested_harmonic_number([2,1],j+2)
        s_1_m = sp.harmonic_number(1,j)
        s_1_mm = sp.harmonic_number(1,j-1)
        s_1_p = sp.harmonic_number(1,j+2)
        s_1_pp = sp.harmonic_number(1,j+3)
        s_2 = sp.harmonic_number(2,j+1)
        s_2_m = sp.harmonic_number(2,j)
        s_2_p = sp.harmonic_number(2,j+2)
        s_2_pp = sp.harmonic_number(2,j+3)
        s_1_1 = sp.nested_harmonic_number([1,1],j+1)
        s_1_1_p = sp.nested_harmonic_number([1,1],j+2)
        s_3 = sp.harmonic_number(3,j+1)
        s_3_m = sp.harmonic_number(3,j)
        s_3_p = sp.harmonic_number(3,j+2)

        s_1_1_pp =sp.nested_harmonic_number([1,1],j+3)
        s_1_m2_pp = sp.nested_harmonic_number([1,-2],j+3)
        s_1_1_1_pp = sp.nested_harmonic_number([1,1,1],j+3)
        s_1_2_pp = sp.nested_harmonic_number([1,2],j+3)
        s_2_1_pp = sp.nested_harmonic_number([2,1],j+3)
        s_3_pp = sp.harmonic_number(3,j+3)

        # Nucl.Phys.B 691 (2004) 129-181
        # Note different beta function convention
        # so we reverse the sign
        term1 = - 4 * cfg.C_A * cfg.N_F * (
            20/9 * (s_1_mm - s_1_m) - (2* (s_1_m - s_1_p) + (s_2_m - s_2_p) + 2 * (s_3_m - s_3_p))
            - (218/9*(s_1_p-s_1_pp) + 4 * (s_1_1_p-s_1_1_pp) + 44/3 * (s_2_p - s_2_pp))
            + (27*(s_1-s_1_p) + 4 * (s_1_1 - s_1_1_p) - 7 * (s_2 - s_2_p) - 2 * (s_3 - s_3_p))
            - 2 * ((s_1_m2_m + 4 * s_1_m2_p - 2 * s_1_m2_pp - 3 * s_1_m2) + (s_1_1_1_m + 4 * s_1_1_1_p - 2 * s_1_1_1_pp - 3 * s_1_1_1))
        )
        term2 = - 8 * cfg.C_F * cfg.T_F * cfg.N_F * (
            2 * (5*(s_1_p - s_1_pp ) + 2 * (s_1_1_p - s_1_1_pp) - 2 * (s_2_p - s_2_pp) + (s_3_p - s_3_pp))
            - (43/2 * (s_1-s_1_p) + 4 * (s_1_1 - s_1_1_p) - 7/2 * (s_2 - s_2_p))
            + (7 * (s_1_m - s_1_p) - 1.5 * (s_2_m - s_2_p))
            + 2 * ((s_1_1_1_m + 4 * s_1_1_1_p -2 * s_1_1_1_pp - 3 * s_1_1_1)
                    -(s_1_2_m + 4 * s_1_2_p -2 * s_1_2_pp - 3 * s_1_2)
                    -(s_2_1_m + 4 * s_2_1_p -2 * s_2_1_pp - 3 * s_2_1)
                    +.5 * (s_3_m + 4 * s_3_p -2 * s_3_pp - 3 * s_3))
        )
    # Constraint at j = 0 is 0
    elif evolve_type == "axial":
        # Nucl.Phys.B 889 (2014) 351-400
        # Note different beta function convention
        # so we reverse the sign
        d1 = sp.d_weight(1,1,j+1)
        d0 = sp.d_weight(1,0,j+1)
        d02 = sp.d_weight(2,0,j+1)
        d03 = sp.d_weight(3,0,j+1)
        d12 = sp.d_weight(2,1,j+1)
        d13 = sp.d_weight(3,1,j+1)
        Delta_pqg = (2 * d1 - d0)
        s_1 = sp.harmonic_number(1,j+1)
        s_2 = sp.harmonic_number(2,j+1)
        s_m2 = sp.harmonic_number(-2,j+1)
        s_1_1 = sp.nested_harmonic_number([1,1],j+1)
        # (4.5)
        term1 = - 8 * cfg.C_F * cfg.T_F * cfg.N_F * (
            2 * Delta_pqg * (s_1_1 - s_2) - 2 * (2 * d0 - d02 - 2 * d1) * s_1
            - 11 * d0 + 9/2 * d02 - d03 + 27/2 * d1 + 4 * d12 - 2 * d13
        )
        term2 = - 4 * cfg.C_A * cfg.N_F * (
            - 2 * Delta_pqg * (s_m2 + s_1_1) + 4 * (d0 - d1 - d12) * s_1
            + 12 * d0 - d02 - 2 * d03 - 11 * d1 - 12 * d12 - 12 * d13
        )
    else:
        raise ValueError("evolve_type must be axial or vector")
    result = term1 + term2
    return result

def gamma_qg(j,evolve_type="vector",evolution_order="nlo",interpolation=True):
    """
    Returns conformal qg singlet anomalous dimension for conformal spin-j

    Arguments:
    - j (float): conformal spin
    - evolve_type (str. optional): vector or axial
    - evolution_order (str. optional): lo, nlo or nnlo
    - interpolation (bool, optional): Use tabulated values for interpolation (only beyond lo)
    """
    if evolution_order == "lo":
        return gamma_qg_lo(j,evolve_type,False)
    elif evolution_order == "nlo":
        return gamma_qg_nlo(j,evolve_type,interpolation)
    else:
        raise ValueError(f"Wrong evolution_order {evolution_order}")
    
# Generate the interpolator
gamma_gq_vector_lo_interpolation = hp.build_gamma_interpolator("gq","singlet","vector",evolution_order="lo")
gamma_gq_axial_lo_interpolation = hp.build_gamma_interpolator("gq","singlet","axial",evolution_order="lo")
# Pick the correct interpolation
def gamma_gq_lo_interpolation(evolve_type):
    return gamma_gq_vector_lo_interpolation if evolve_type == "vector" else gamma_gq_axial_lo_interpolation
def gamma_gq_lo(j,evolve_type="vector",interpolation=True):
    if interpolation:
        interp = gamma_gq_lo_interpolation(evolve_type)
        return interp(j)
    
    if j == 0:
        j = 1e-12

    if evolve_type == "vector":
        result = -cfg.C_F*(j**2+3*j+4)/(3*(j+1)*(j+2))
    elif evolve_type == "axial":
        result = -cfg.C_F*j*(j+3)/(3*(j+1)*(j+2))
    else:
        raise ValueError("Type must be axial or vector")
    # Match forward to Wilson anomalous dimension
    result*=6/j
    return result

# Generate the interpolator
gamma_gq_singlet_vector_nlo_interpolation = hp.build_gamma_interpolator("gq","singlet","vector",evolution_order="nlo")
gamma_gq_singlet_axial_nlo_interpolation = hp.build_gamma_interpolator("gq","singlet","axial",evolution_order="nlo")
# Pick the correct interpolation
def gamma_gq_nlo_interpolation(evolve_type):
    return gamma_gq_singlet_vector_nlo_interpolation if evolve_type == "vector" else gamma_gq_singlet_axial_nlo_interpolation

# @cfg.memory.cache
def gamma_gq_nlo(j, evolve_type = "vector",interpolation=True):
    if interpolation:
        # interp = hp.build_gamma_interpolator("gq","singlet",evolve_type,evolution_order="nlo")
        interp = gamma_gq_nlo_interpolation(evolve_type)
        result = interp(j)
        return result

    if j == 0:
        j = 1e-12

    if evolve_type == "vector":
        s_1 = sp.harmonic_number(1,j+1)
        s_2 = sp.harmonic_number(2,j+1)
        s_1_1_1 = sp.nested_harmonic_number([1,1,1],j+1)
        s_1_1_1_m = sp.nested_harmonic_number([1,1,1],j)
        s_1_1_1_mm = sp.nested_harmonic_number([1,1,1],j-1)
        s_1_1_1_p = sp.nested_harmonic_number([1,1,1],j+2)
        s_1_m2 = sp.nested_harmonic_number([1,-2],j+1)
        s_1_m2_m = sp.nested_harmonic_number([1,-2],j)
        s_1_m2_mm = sp.nested_harmonic_number([1,-2],j-1)
        s_1_m2_p = sp.nested_harmonic_number([1,-2],j+2)
        s_1_2 = sp.nested_harmonic_number([1,2],j+1)
        s_1_2_m = sp.nested_harmonic_number([1,2],j)
        s_1_2_mm = sp.nested_harmonic_number([1,2],j-1)
        s_1_2_p = sp.nested_harmonic_number([1,2],j+2)
        s_2_1 = sp.nested_harmonic_number([2,1],j+1)
        s_2_1_m = sp.nested_harmonic_number([2,1],j)
        s_2_1_mm = sp.nested_harmonic_number([2,1],j-1)
        s_2_1_p = sp.nested_harmonic_number([2,1],j+2)
        s_1_m = sp.harmonic_number(1,j)
        s_1_mm = sp.harmonic_number(1,j-1)
        s_1_p = sp.harmonic_number(1,j+2)
        s_1_pp = sp.harmonic_number(1,j+3)
        s_2 = sp.harmonic_number(2,j+1)
        s_2_m = sp.harmonic_number(2,j)
        s_2_p = sp.harmonic_number(2,j+2)
        s_2_pp = sp.harmonic_number(2,j+3)
        s_1_1 = sp.nested_harmonic_number([1,1],j+1)
        s_1_1_m = sp.nested_harmonic_number([1,1],j)
        s_1_1_mm = sp.nested_harmonic_number([1,1],j-1)
        s_1_1_p = sp.nested_harmonic_number([1,1],j+2)
        s_3 = sp.harmonic_number(3,j+1)
        s_3_m = sp.harmonic_number(3,j)
        s_3_p = sp.harmonic_number(3,j+2)

        # Nucl.Phys.B 691 (2004) 129-181
        # Note different beta function convention
        # so we reverse the sign
        term1 = - 4 * cfg.C_A * cfg.C_F * (
            2* ((2 * s_1_1_1_mm - 4 * s_1_1_1_m - s_1_1_1_p + 3 *s_1_1_1)
                - (2 * s_1_m2_mm - 4 * s_1_m2_m - s_1_m2_p + 3 *s_1_m2) 
                - (2 * s_1_2_mm - 4 * s_1_2_m - s_1_2_p + 3 *s_1_2)
                - (2 * s_2_1_mm - 4 * s_2_1_m - s_2_1_p + 3 *s_2_1))
            + (2 * (s_1 - s_1_p) - 13 * (s_1_1 - s_1_1_p) - 7 * (s_2 - s_2_p) - 2 * (s_3 - s_3_p))
            + (s_1_mm - 2 * s_1_m + s_1_p) - 22/3 * (s_1_1_mm - 2 * s_1_1_m + s_1_1_p)
            + 4 * (7/9 * (s_1_m - s_1_p) + 3 * (s_2_m - s_2_p) + (s_3_m - s_3_p))
            + (44/9 * (s_1_p - s_1_pp) + 8/3 * (s_2_p - s_2_pp)) 
        )
        term2 = - 8 * cfg.C_F * cfg.T_F * cfg.N_F * (
            (4/3 * (s_1_1_mm - 2 * s_1_1_m + s_1_1_p) -20/9 * (s_1_mm - 2 * s_1_m + s_1_p) )
            - (4 * (s_1 - s_1_p) - 2 * (s_1_1 - s_1_1_p))
        )
        term3 = - 4 * cfg.C_F**2 *(
            3 * (2 * s_1_1_mm - 4 * s_1_1_m - s_1_1_p + 3 * s_1_1) - 2 * (2 * s_1_1_1_mm - 4 * s_1_1_1_m - s_1_1_1_p + 3 * s_1_1_1)
            - ((s_1 - s_1_p) - 2 * (s_1_1 - s_1_1_p) + 1.5 * (s_2 - s_2_p) - 3 * (s_3 - s_3_p))
            - (5/2 * (s_1_m - s_1_p) + 2 * (s_2_m - s_2_p) + 2 * (s_3_m - s_3_p))
        )
    # Constraint at j = 0 is 
    # 18 * cfg.C_F**2 - 142/3 * cfg.C_A * cfg.C_F + 8/3 * cfg.C_F * cfg.T_F * cfg.N_F
    elif evolve_type == "axial": 
        # Nucl.Phys.B 889 (2014) 351-400
        # Note different beta function convention
        # so we reverse the sign
        d1 = sp.d_weight(1,1,j+1)
        d0 = sp.d_weight(1,0,j+1)
        d02 = sp.d_weight(2,0,j+1)
        d03 = sp.d_weight(3,0,j+1)
        d12 = sp.d_weight(2,1,j+1)
        d13 = sp.d_weight(3,1,j+1)
        s_1 = sp.harmonic_number(1,j+1)
        s_2 = sp.harmonic_number(2,j+1)
        s_m2 = sp.harmonic_number(-2,j+1)
        s_1_1 = sp.nested_harmonic_number([1,1],j+1)
        Delta_pgq = (2 * d0 - d1)
        # (4.6)
        # Constrained by 
        term1 = - 16/9 * cfg.C_F * cfg.T_F * cfg.N_F * (
            3 * Delta_pgq * s_1 - 4 * d0 - d1 - 3 * d12
        )
        term2 = - 4 * cfg.C_F**2 * (
            - Delta_pgq * (2* s_1_1 - s_1) + 2 *(d1 + d12) * s_1
            -17/2 * d0 + 2 * d02 + 2 * d03 + 4 * d1 + .5 * d12 + d13
        )
        term3 = - 4 * cfg.C_A * cfg.C_F *(
            2* Delta_pgq * (s_1_1 - s_m2 - s_2) - (10/3 * d0 + 4 * d02 + 1/3 * d1) * s_1
            + 41/9 * d0 - 4 * d02 + 4 * d03 + 35/9 * d1 + 38/3 * d12 + 6 * d13
        )
    else:
        raise ValueError("Type must be axial or vector")
    result = term1 + term2 + term3
    return result

def gamma_gq(j,evolve_type="vector",evolution_order="nlo",interpolation=True):
    """
    Returns conformal gq singlet anomalous dimension for conformal spin-j

    Arguments:
    - j (float): conformal spin
    - evolve_type (str. optional): vector or axial
    - evolution_order (str. optional): lo, nlo or nnlo
    - interpolation (bool, optional): Use tabulated values for interpolation (only beyond lo)
    """
    if evolution_order == "lo":
        return gamma_gq_lo(j,evolve_type,False)
    elif evolution_order == "nlo":
        return gamma_gq_nlo(j,evolve_type,interpolation)
    else:
        raise ValueError(f"Wrong evolution_order {evolution_order}")
# Generate the interpolator
gamma_gg_vector_lo_interpolation = hp.build_gamma_interpolator("gg","singlet","vector",evolution_order="lo")
gamma_gg_axial_lo_interpolation = hp.build_gamma_interpolator("gg","singlet","axial",evolution_order="lo")
# Pick the correct interpolation
def gamma_gg_lo_interpolation(evolve_type):
    return gamma_gg_vector_lo_interpolation if evolve_type == "vector" else gamma_gg_axial_lo_interpolation

def gamma_gg_lo(j,evolve_type="vector",interpolation=True):
    if interpolation:
        interp = gamma_gg_lo_interpolation(evolve_type)
        return interp(j)

    if j == 0:
        j = 1e-12
    if evolve_type == "vector":
        result = -cfg.C_A*(-4*mp.digamma(j+2)+4*mp.digamma(1)+8*(j**2+3*j+3)/(j*(j+1)*(j+2)*(j+3))-cfg.BETA_0/cfg.C_A)
    elif evolve_type == "axial":
        result = -cfg.C_A*(-4*mp.digamma(j+2)+4*mp.digamma(1)+8/((j+1)*(j+2)) - cfg.BETA_0/cfg.C_A)
    else:
        raise ValueError("Type must be axial or vector")
    return result

# Generate the interpolator
gamma_gg_singlet_vector_nlo_interpolation = hp.build_gamma_interpolator("gg","singlet","vector",evolution_order="nlo")
gamma_gg_singlet_axial_nlo_interpolation = hp.build_gamma_interpolator("gg","singlet","axial",evolution_order="nlo")
# Pick the correct interpolation
def gamma_gg_nlo_interpolation(evolve_type):
    return gamma_gg_singlet_vector_nlo_interpolation if evolve_type == "vector" else gamma_gg_singlet_axial_nlo_interpolation
# @cfg.memory.cache
def gamma_gg_nlo(j,  evolve_type = "vector",interpolation=True):
    if interpolation:
        # interp = hp.build_gamma_interpolator("gg","singlet",evolve_type,evolution_order="nlo")
        interp = gamma_gg_nlo_interpolation(evolve_type)
        result = interp(j)
        return result
    if evolve_type == "vector":
        s_1 = sp.harmonic_number(1,j+1)
        s_1_m2 = sp.nested_harmonic_number([1,-2],j+1)
        s_1_m2_m = sp.nested_harmonic_number([1,-2],j)
        s_1_m2_mm = sp.nested_harmonic_number([1,-2],j-1)
        s_1_m2_p = sp.nested_harmonic_number([1,-2],j+2)
        s_1_2 = sp.nested_harmonic_number([1,2],j+1)
        s_1_2_m = sp.nested_harmonic_number([1,2],j)
        s_1_2_mm = sp.nested_harmonic_number([1,2],j-1)
        s_1_2_p = sp.nested_harmonic_number([1,2],j+2)
        s_2_1 = sp.nested_harmonic_number([2,1],j+1)
        s_2_1_m = sp.nested_harmonic_number([2,1],j)
        s_2_1_mm = sp.nested_harmonic_number([2,1],j-1)
        s_2_1_p = sp.nested_harmonic_number([2,1],j+2)
        s_1_m = sp.harmonic_number(1,j)
        s_1_mm = sp.harmonic_number(1,j-1)
        s_1_p = sp.harmonic_number(1,j+2)
        s_1_pp = sp.harmonic_number(1,j+3)
        s_2 = sp.harmonic_number(2,j+1)
        s_2_m = sp.harmonic_number(2,j)
        s_2_p = sp.harmonic_number(2,j+2)
        s_2_pp = sp.harmonic_number(2,j+3)
        s_3 = sp.harmonic_number(3,j+1)
        s_3_m = sp.harmonic_number(3,j)
        s_3_p = sp.harmonic_number(3,j+2)
        s_1_m2_pp = sp.nested_harmonic_number([1,-2],j+3)
        s_1_2_pp = sp.nested_harmonic_number([1,2],j+3)
        s_2_1_pp = sp.nested_harmonic_number([2,1],j+3)
        s_m3 = sp.harmonic_number(-3,j+1)
        s_3_pp = sp.harmonic_number(3,j+3)

        # Nucl.Phys.B 691 (2004) 129-181
        # Note different beta function convention
        # so we reverse the sign
        term1 = - 4 * cfg.C_A * cfg.N_F * (
            2/3 - 16/3 * s_1 -23/9 * (s_1_mm + s_1_pp) + 14/3 * (s_1_m + s_1_p) + 2/3 * (s_2_m - s_2_p)
        )
        term2 = - 4 * cfg.C_A**2 * (
            + 2 * s_m3 - 8/3  - 14/3 * s_1 + 2 * s_3 
            - 4 * ((s_1_m2_mm - 2 * s_1_m2_m - 2 * s_1_m2_p + s_1_m2_pp + 3 * s_1_m2)
                    + (s_1_2_mm - 2 * s_1_2_m - 2 * s_1_2_p + s_1_2_pp + 3 * s_1_2)
                    + (s_2_1_mm - 2 * s_2_1_m - 2 * s_2_1_p + s_2_1_pp + 3 * s_2_1)
                )
            + 8/3 * (s_2_p - s_2_pp) - 4  * (3 * (s_2_m - 3 * s_2_p + s_2_pp + s_2) -  (s_3_m - 3 * s_3_p + s_3_pp + s_3)) 
            + 109/18 * (s_1_m + s_1_p) + 61/3 * (s_2_m - s_2_p)
        )
        term3 = - 8 * cfg.C_F * cfg.T_F * cfg.N_F * (
            .5 + 2/3 * (s_1_mm - 13 * s_1_m - s_1_p - 5 * s_1_pp + 18 * s_1)
            + (3 * s_2_m - 5 * s_2_p + 2 * s_2) - 2 * (s_3_m - s_3_p)
        )
    # Constraint at j = 0 is 2 * cfg.BETA_1
    elif evolve_type == "axial":
        # Nucl.Phys.B 889 (2014) 351-400
        # Note different beta function convention
        # so we reverse the sign
        d02 = sp.d_weight(2,0,j+1)
        d03 = sp.d_weight(3,0,j+1)
        eta = 1/((j+1)*(j+2))
        s_1 = sp.harmonic_number(1,j+1)
        s_2 = sp.harmonic_number(2,j+1)
        s_m2 = sp.harmonic_number(-2,j+1)
        s_1_2 = sp.nested_harmonic_number([1,2],j+1)
        s_2_1 = sp.nested_harmonic_number([2,1],j+1)
        s_m3 = sp.harmonic_number(-3,j+1)
        s_3 = sp.harmonic_number(3,j+1)
        s_1_m2 = sp.nested_harmonic_number([1,-2],j+1)
        # (4.7)
        term1 = - 8 * cfg.C_F * cfg.T_F * cfg.N_F * (
            -.5 - 7 * eta + 5 * eta**2 + 2 * eta**3 + 6 * d02 - 4 * d03
        )
        term2 = - 4/3 * cfg.C_A * cfg.N_F *(
            10/3 * s_1 - 2 - 26/3 * eta + 2 * eta**2
        )
        term3 = -  4 * cfg.C_A**2 * (
            4 * (s_1_m2 + s_1_2 + s_2_1) - 2 * (s_3 +s_m3) - 67/9 * s_1 + 8/3
            - 8 * eta * (s_2 + s_m2) + 8 * (2 * eta + eta**2 - 2 * d02) * s_1
            + 901/18 * eta - 149/3 * eta**2 -24 * eta**3 - 32 * (d02 - d03)
        )
    else:
        raise ValueError("Type must be axial or vector")
    result = term1 + term2 + term3
    return result

def gamma_gg(j,evolve_type="vector",evolution_order="nlo",interpolation=True):
    """
    Returns conformal gg singlet anomalous dimension for conformal spin-j

    Arguments:
    - j (float): conformal spin
    - evolve_type (str. optional): vector or axial
    - evolution_order (str. optional): lo, nlo or nnlo
    - interpolation (bool, optional): Use tabulated values for interpolation (only beyond lo)
    """
    if evolution_order == "lo":
        return gamma_gg_lo(j,evolve_type,False)
    elif evolution_order == "nlo":
        return gamma_gg_nlo(j,evolve_type,interpolation)
    else:
        raise ValueError(f"Wrong evolution_order {evolution_order}")

def d_element(j,k):
    """ Belistky (4.204)"""
    if j == k:
        raise ValueError("j and k must be unequal")
    result = - .5 * (1 + sp.power_minus_1(j-k)) * (2 * k + 3)/((j - k)*(j + k + 3))
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
    
    
    result =  -cfg.C_F * nd_projector(j,k) * ((3 + 2 * k) / ((j - k) * (j + k + 3))) * (
        2 * digamma_A(j, k) + 
        (digamma_A(j, k) - mp.digamma(j + 2) + mp.digamma(1)) * ((j - k) * (j + k + 3)) / ((k + 1) * (k + 2))
        )
    return result

def conformal_anomaly_gq(j,k):
    """Belitsky (4.208). Equal for vector and axial """
    if j == k:
        raise ValueError("j and k must be unqual")
    
    
    result = -cfg.C_F * nd_projector(j,k) * (1 / 6) * ((3 + 2 * k) / ((k + 1) * (k + 2)))
    return result

def conformal_anomaly_gg(j,k):
    """Belitsky (4.209). Equal for vector and axial """
    if j == k:
        raise ValueError("j and k must be unqual")
    
    
    result = (
            -cfg.C_A * nd_projector(j,k) *
            ((3 + 2 * k) / ((j - k) * (j + k + 3))) *
            (
                2 * digamma_A(j,k) +
                (digamma_A(j,k) - mp.digamma(j + 2) + mp.digamma(1)) *
                ((mp.gamma(j + 4) * mp.gamma(k)) / (mp.gamma(j) * mp.gamma(k + 4)) - 1) +
                2 * (j - k) * (j + k + 3) * (mp.gamma(k) / mp.gamma(k + 4))
            )
        )
    return result

def gamma_qq_nd(j,k,  evolve_type = "vector",evolution_order="nlo",interpolation=True):
    """ Belistky (4.203)"""
    if evolution_order == "lo":
        return 0
    if isinstance(j, (int)) and  isinstance(k, (int)) and k >= j:
        return 0
    
    if evolution_order == "nlo":
        term1 = (gamma_qq(j,evolution_order="lo",interpolation=interpolation)-gamma_qq(k,evolution_order="lo",interpolation=interpolation))* \
                (d_element(j,k) * (cfg.BETA_0 - gamma_qq(k,evolution_order="lo",interpolation=interpolation)) + conformal_anomaly_qq(j,k))
        term2 = - (gamma_qg(j,evolve_type,"lo",interpolation=interpolation) - 
                   gamma_qg(k,evolve_type,evolution_order="lo",interpolation=interpolation)) * d_element(j,k) * gamma_gq(j,evolve_type,evolution_order="lo",interpolation=interpolation)
        term3 = gamma_qg(j,evolve_type,"lo",interpolation=interpolation) * conformal_anomaly_gq(j,k)
        result = term1 + term2 + term3
    else:
        raise ValueError(f"Currently unsupported evolution order {evolution_order}")
    return result

def gamma_qg_nd(j,k,  evolve_type = "vector",evolution_order="nlo",interpolation=True):
    """ Belistky (4.203)"""
    if evolution_order == "lo":
        return 0
    if isinstance(j, (int)) and  isinstance(k, (int)) and k >= j:
        return 0

    if evolution_order == "nlo":
        term1 = (gamma_qg(j,  evolve_type, "lo",interpolation=interpolation) - gamma_qg(k,  evolve_type, "lo",interpolation=interpolation)) * \
                d_element(j, k) * (cfg.BETA_0 - gamma_gg(k,  evolve_type, "lo",interpolation=interpolation))
        term2 = - (gamma_qq(j, evolution_order="lo",interpolation=interpolation) - gamma_qq(k, evolution_order="lo",interpolation=interpolation)) * \
                d_element(j, k) * gamma_qg(k,  evolve_type, "lo",interpolation=interpolation)
        term3 = gamma_qg(j,  evolve_type, evolution_order="lo") * conformal_anomaly_gg(j, k)
        term4 = - conformal_anomaly_qq(j, k) * gamma_qg(k,  evolve_type, evolution_order="lo",interpolation=interpolation)
        result = term1 + term2 + term3 + term4
    else:
        raise ValueError(f"Currently unsupported evolution order {evolution_order}")
    return result

def gamma_gq_nd(j,k,  evolve_type = "vector",evolution_order="nlo",interpolation=True):
    """ Belistky (4.203)"""
    if evolution_order == "lo":
        return 0
    if isinstance(j, (int)) and  isinstance(k, (int)) and k >= j:
        return 0

    if evolution_order == "nlo":
        term1 = (gamma_gq(j,  evolve_type, evolution_order="lo",interpolation=interpolation) - gamma_gq(k,  evolve_type, evolution_order="lo",interpolation=interpolation)) * \
                d_element(j, k) * (cfg.BETA_0 - gamma_qq(k, evolution_order="lo",interpolation=interpolation))
        term2 = - (gamma_gg(j,  evolve_type, evolution_order="lo",interpolation=interpolation) - gamma_gg(k,  evolve_type, evolution_order="lo",interpolation=interpolation)) * \
                d_element(j, k) * gamma_gq(k,  evolve_type, evolution_order="lo",interpolation=interpolation)
        term3 = gamma_gq(j,  evolve_type, evolution_order="lo") * conformal_anomaly_qq(j, k)
        term4 = - conformal_anomaly_gg(j, k) * gamma_gq(k,  evolve_type, evolution_order="lo",interpolation=interpolation)
        term5 = (gamma_gg(j,  evolve_type, evolution_order="lo",interpolation=interpolation) - gamma_qq(k,evolution_order="lo",interpolation=interpolation)) * conformal_anomaly_gq(j, k)

        result = term1 + term2 + term3 + term4 + term5
    else:
        raise ValueError(f"Currently unsupported evolution order {evolution_order}")
    return result

def gamma_gg_nd(j,k, evolve_type = "vector",evolution_order="nlo",interpolation=True):
    """ Belistky (4.203)"""
    if evolution_order == "lo":
        return 0
    if isinstance(j, (int)) and  isinstance(k, (int)) and k >= j:
        return 0

    if evolution_order == "nlo":
        term1 = (gamma_gg(j,  evolve_type, evolution_order="lo",interpolation=interpolation) - gamma_gg(k,  evolve_type, evolution_order="lo",interpolation=interpolation)) * \
                (d_element(j, k) * (cfg.BETA_0 - gamma_gg(k,  evolve_type, "lo",interpolation=interpolation)) + conformal_anomaly_gg(j, k))
        term2 = - (gamma_gq(j,  evolve_type, evolution_order="lo",interpolation=interpolation) - gamma_gq(k,  evolve_type, evolution_order="lo",interpolation=interpolation)) * \
                d_element(j, k) * gamma_qg(k,  evolve_type, evolution_order="lo",interpolation=interpolation)
        term3 = - conformal_anomaly_gq(j, k) * gamma_qg(k,  evolve_type, evolution_order="lo",interpolation=interpolation)
        result = term1 + term2 + term3
    else:
        raise ValueError(f"Currently unsupported evolution order {evolution_order}")
    return result

def gamma_pm(j,  evolve_type = "vector",solution="+",interpolation=True):
    """ Compute the (+) and (-) eigenvalues of the lo evolution equation of the coupled singlet quark and gluon GPD
    Arguments:
    - j: conformal spin,
    - evolve_type: "vector" or "axial"
    - interpolation (bool, optional): Use interpolated values
    Returns:
    The eigenvalues (+) and (-) in terms of an array
    """
    # Check evolve_type
    hp.check_evolve_type(evolve_type)

    base = gamma_qq(j,evolution_order="lo",interpolation=interpolation)+gamma_gg(j,evolve_type,evolution_order="lo",interpolation=interpolation)
    root = mp.sqrt((gamma_qq(j,evolution_order="lo",interpolation=interpolation)-gamma_gg(j,evolve_type,evolution_order="lo",interpolation=interpolation))**2
                   +4*gamma_gq(j,evolve_type,evolution_order="lo",interpolation=interpolation)*gamma_qg(j,evolve_type,evolution_order="lo",interpolation=interpolation))

    if solution == "+":
        result = (base + root)/2
    elif solution == "-":
        result = (base - root)/2
    else:
        raise ValueError("Invalid solution evolve_type. Use '+' or '-'.")
    return result

def R_qq(j,evolve_type="vector",interpolation=True):
    
    term1 = gamma_qq(j,"singlet",evolve_type,evolution_order="nlo",interpolation=interpolation)
    term2 = - .5 * cfg.BETA_1/cfg.BETA_0 * gamma_qq(j,"singlet",evolve_type,evolution_order="lo",interpolation=interpolation)
    result = term1 + term2
    return result

def R_qg(j,evolve_type="vector",interpolation=True):
    
    term1 = gamma_qg(j,evolve_type,"nlo",interpolation=interpolation)
    term2 = - .5 * cfg.BETA_1/cfg.BETA_0 * gamma_qg(j,evolve_type,"lo",interpolation=interpolation)
    result = term1 + term2
    return result

def R_gq(j,evolve_type="vector",interpolation=True):
    
    term1 = gamma_gq(j,evolve_type,"nlo",interpolation=interpolation)
    term2 = - .5 * cfg.BETA_1/cfg.BETA_0 * gamma_gq(j,evolve_type,"lo",interpolation=interpolation)
    result = term1 + term2
    return result

def R_gg(j,evolve_type="vector",interpolation=True):
    
    term1 = gamma_gg(j,evolve_type,"nlo",interpolation=interpolation)
    term2 = - .5 * cfg.BETA_1/cfg.BETA_0 * gamma_gg(j,evolve_type,"lo",interpolation=interpolation)
    result = term1 + term2
    return result