# Cleaner path structure
from pathlib import Path
# Set cache memory below
from joblib import Memory

# Set precision globally
from mpmath import mp
mp.dps = 16

####################################
####   Define directories for   ####
####    clear data handling     ####
####################################

# User base directory
USER_PATH = Path.home() / "stringy-gpds"
# Parent directory for data
DATA_PATH = USER_PATH / "data"
# Folder for generated plots
PLOT_PATH = USER_PATH / "plots"
# PDF location
PDF_PATH = USER_PATH / "pdfs"

# Additional subdirectories
IMPACT_PARAMETER_MOMENTS_PATH = DATA_PATH / "ImpactParameterMoments"
MOMENTUM_SPACE_MOMENTS_PATH = DATA_PATH / "MomentumSpaceMoments"
GPD_PATH = DATA_PATH / "GPDs" 
INTERPOLATION_TABLE_PATH = DATA_PATH / "InterpolationTables"

# Input PDF set: the unpolarized parameters (MSTW schema) are read from
# pdfs/<PDF_SET>.csv and the polarized factors (AAC schema) from pdfs/<POL_PDF_SET>.csv.
# Override with pdf_set / pol_pdf_set in user_config.py; new sets are generated with
# fit.fit_input_pdfs / fit.fit_polarized_input_pdfs (see README).
# Supplied sets: "JAM22" (default, mu0 = 2 GeV), "GUMP" (mu0 = 2 GeV, no strange),
# "MSTW_original" + "AAC_original" (mu0 = 1 GeV, set MU_INPUT = 1 as well)
PDF_SET = "JAM22"
POL_PDF_SET = None  # defaults to PDF_SET + "_POL"

# Input/reference scale (GeV) at which the PDF parametrizations
# and alpha_S(Q0^2) of the selected set are defined
MU_INPUT = 2

##################################
###########    Cache   ###########
##################################
#### Make sure to clear after ####
## parameters have been changed ##
##################################
# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent
# Cache directory
CACHE_PATH = PROJECT_ROOT / "cache"
memory = Memory(CACHE_PATH,verbose=0)
# Clear after changing parameters
# below using
# memory.clear()

#######################################
## Kinematics used for interpolation ##
##     Need to have equal length     ##
#######################################

# Interpolate evolved moments (recommended: True)
# run generate_moment_table first, 
# then set to True
INTERPOLATE_MOMENTS = True

# Compute non-diagonal evolution
# for analytically continued moments
# Can usually be neglected (recommended: False)
ND_EVOLVED_COMPLEX_MOMENT = False

# Standard choice
ETA_ARRAY = [0,0.33,0.1]
T_ARRAY = [-0.69,-0.69,-0.23]
MU_ARRAY = [2,2,2]

PARTICLES = ["quark","gluon"]
MOMENTS = ["singlet","non_singlet_isoscalar","non_singlet_isovector"]
LABELS = ["A","Atilde","B"]
ORDERS = ["nlo"]
ERRORS = ["central","plus","minus"]

########################################
#### Dictionaries and data handling ####
####       Change as required       ####
########################################

# Add some colors
saturated_pink  = "#ff1a99"
blue            = "#1f77b4"
orange          = "#ff7f0e"
green           = "#2ca02c"

# ArXiv ID and renormalization scale mu
PUBLICATION_MAPPING = {
    "2305.11117": ("cyan",2),
    "0705.4295": ("orange",2),
    "1908.10706": (saturated_pink,2),
    "2310.08484": ("darkblue",2),
    "2410.03539": ("green",2)
# Add more publication IDs and corresponding colors here
}
# Select which data to plot. Comment out as desired
GPD_PUBLICATION_MAPPING = {
    # publication ID, GPD type, GPD label, eta, t ,mu
    ("2008.10573","non_singlet_isovector","H",0.00, -0.69, 2.00): ("mediumturquoise","000_069_200"),
    ("2008.10573","non_singlet_isovector","H",0.33, -0.69, 2.00): ("green","033_069_200"),
    ("2008.12474","non_singlet_isovector","H",0.00, -0.39, 3.00): ("purple","000_039_300"),
    ("2312.10829","non_singlet_isovector","H",0.10, -0.23, 2.00): ("orange","010_023_200"),
    ("2008.10573","non_singlet_isovector","Htilde",0.00, -0.69, 2.00): ("mediumturquoise","000_069_200"),
    ("2008.10573","non_singlet_isovector","Htilde",0.33, -0.69, 2.00): ("green","033_069_200"),
    ("2112.07519","non_singlet_isovector","Htilde",0.00, -0.39, 3.00): ("purple","000_039_300"),
    ("2008.10573","non_singlet_isovector","E",0.00, -0.69, 2.00): ("mediumturquoise","000_069_200"),
    ("2008.10573","non_singlet_isovector","E",0.33, -0.69, 2.00): ("green","033_069_200"),
    ("2312.10829","non_singlet_isovector","E",0.10, -0.23, 2.00): ("orange","010_023_200"),
    ("2305.11117","non_singlet_isovector","E",0.00,-0.17,2.00): (blue,"000_017_200"),
    ("2310.13114","non_singlet_isovector","Htilde",0.00,-0.17,2.00): (blue,"000_017_200"),
    ("2305.11117","non_singlet_isovector","H",0.00,-0.17,2.00): (blue,"000_017_200"),
    ("2305.11117","non_singlet_isovector","E",0.00,-0.65,2.00): (green,"000_065_200"),
    ("2310.13114","non_singlet_isovector","Htilde",0.00,-0.65,2.00): (green,"000_065_200"),
    ("2305.11117","non_singlet_isovector","H",0.00,-0.65,2.00): (green,"000_065_200")
# Add more publication IDs and corresponding colors here
}

# Map GPDs to moment labels
GPD_LABEL_MAP ={"H": "A",
                "E": "B",
                "Htilde": "Atilde",
                "Etilde" : "Btilde"
                    }

# Invert map
INVERTED_GPD_LABEL_MAP = {v: k for k, v in GPD_LABEL_MAP.items()}

#####################
### QCD Paramters ###
#####################

N_C = 3
C_A = N_C
C_F = (N_C**2-1)/(2*N_C)
T_F = .5
N_F = 3

# Beta function
BETA_0 = 4/3 * T_F * N_F - 11/3 * N_C
BETA_1 = 20/3 * T_F * C_A * N_F + 4 * C_F * T_F * N_F -34/3 * C_A**2

# Proton mass
m_N = 0.93827

########################
### Model Parameters ###
########################

# Re-fit (2026-07-13) to form-factor (pseudo-)data using the JAM22 input PDFs
# (JAM22.csv / JAM22_POL.csv, mu0 = MU_INPUT = 2 GeV, strange resolved).
# Regenerate with fit.fit_non_singlet_slopes / fit_singlet_slopes_A /
# fit_singlet_slopes_Atilde after changing the input PDF set (see README).
# Singlet tuples are ordered (quark_A, quark_D, gluon_A, gluon_D).
REGGE_SLOPES = {
    "non_singlet_isovector": {
        "A": {
            "lo": 0.5657,
            "nlo": 0.5657
        },
        "B": {
            "lo": 1.1753,
            "nlo": 1.1753
        },
        "Atilde": {
            "lo": 0.5161,
            "nlo": 0.5161
        }
    },
    "non_singlet_isoscalar": {
        "A": {
            "lo": 0.8696,
            "nlo": 0.8696
        },
        "B": {
            "lo": 1.0409,
            "nlo": 1.0409
        },
        "Atilde": {
            "lo": 0.3595,
            "nlo": 0.3595
        }
    },
    "singlet": {
        "A": {
            "lo": (0.5713,1.4919,0.2875,3.7418),
            "nlo": (0.5713,1.4919,0.2880,3.7272)
        },
        "B": {
            "lo": (0.5713,1.4919,0.2875,3.7418),
            "nlo": (0.5713,1.4919,0.2880,3.7272)
        },
        "Atilde": {
            "lo": (0.6155,1.179,0.490,0.744),
            "nlo": (0.6207,1.179,0.490,0.744)
        }
    }
}

# The A normalizations of the singlet (and the isovector A) are fixed to 1: their forward
# limit is the momentum fraction / charge predicted by the input PDFs, so the momentum sum
# rule A_q(0) + A_g(0) = 1 holds exactly. Only the D normalizations are fitted.
MOMENT_NORMALIZATIONS = {
    "non_singlet_isovector": {
        "A": {
            "lo": 1,
            "nlo": 1
        },
        "B": {
            "lo": 3.6713,
            "nlo": 3.6713
        },
        "Atilde": {
            "lo": 1.3267,
            "nlo": 1.3267
        }
    },
    "non_singlet_isoscalar": {
        "A": {
            "lo": 0.9517,
            "nlo": 0.9517
        },
        "B": {
            "lo": -0.1171,
            "nlo": -0.1171
        },
        "Atilde": {
            "lo": 0.8953,
            "nlo": 0.8953
        }
    },
    "singlet": {
        "A": {
            "lo": (1,1.0649,1,0.5444),
            "nlo": (1,1.0649,1,0.5464)
        },
        "B": {
            "lo": (0,1.0649,0,0.5444),
            "nlo": (0,1.0649,0,0.5464)
        },
        "Atilde": {
            "lo": (1,1,1,1),
            "nlo": (1,1,1,1)
        }
    }
}

# Overwrite standard values with user input
CONFIG_PATH = USER_PATH / "user_config.py"
if CONFIG_PATH.exists():
    # Dictionary
    user_cfg = {}
    # Read as string
    exec(CONFIG_PATH.read_text(), user_cfg)
    # Overwrite
    # Interpolation
    INTERPOLATE_MOMENTS = user_cfg.get("interpolate_moments",INTERPOLATE_MOMENTS)
    ND_EVOLVED_COMPLEX_MOMENT =user_cfg.get("nd_evolved_complex_moment",ND_EVOLVED_COMPLEX_MOMENT)
    # Kinematics
    ETA_ARRAY = user_cfg.get("eta_array", ETA_ARRAY)
    T_ARRAY = user_cfg.get("t_array", T_ARRAY)
    MU_ARRAY = user_cfg.get("mu_array", MU_ARRAY)
    # Choice of moments
    PARTICLES = user_cfg.get("particles", PARTICLES)
    MOMENTS = user_cfg.get("moments", MOMENTS)
    LABELS = user_cfg.get("labels", LABELS)
    ORDERS = user_cfg.get("orders", ORDERS)
    ERRORS = user_cfg.get("errors", ERRORS)
    # (Lattice) data
    PUBLICATION_MAPPING = user_cfg.get("publication_mapping", PUBLICATION_MAPPING)
    GPD_PUBLICATION_MAPPING = user_cfg.get("gpd_publication_mapping", GPD_PUBLICATION_MAPPING)
    # Model parameters
    REGGE_SLOPES = user_cfg.get("regge_slopes", REGGE_SLOPES)
    MOMENT_NORMALIZATIONS = user_cfg.get("moment_normalizations", MOMENT_NORMALIZATIONS)
    # Input PDFs
    PDF_SET = user_cfg.get("pdf_set", PDF_SET)
    POL_PDF_SET = user_cfg.get("pol_pdf_set", POL_PDF_SET)
    MU_INPUT = user_cfg.get("mu_input", MU_INPUT)

# Resolve the input PDF parameter files (the names are historical: the unpolarized set
# follows the MSTW schema and the polarized one the AAC schema)
MSTW_PATH = PDF_PATH / f"{PDF_SET}.csv"
AAC_PATH = PDF_PATH / f"{POL_PDF_SET or PDF_SET + '_POL'}.csv"