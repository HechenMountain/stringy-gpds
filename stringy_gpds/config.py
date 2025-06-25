# Cleaner path structure
from pathlib import Path
# Set cache memory below
from joblib import Memory

# Set precision globally
import mpmath as mp
mp.dps = 16

####################################
####   Define directories for   ####
####    clear data handling     ####
####################################

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Parent directory for data
BASE_PATH = PROJECT_ROOT / "data"
# Folder for generated plots
PLOT_PATH = PROJECT_ROOT / "plots"
# PDF location
PDF_PATH = PROJECT_ROOT / "pdfs"

# Subdirectories for cleaner file handling
IMPACT_PARAMETER_MOMENTS_PATH = BASE_PATH / "ImpactParameterMoments"
MOMENTUM_SPACE_MOMENTS_PATH = BASE_PATH / "MomentumSpaceMoments"
GPD_PATH = BASE_PATH / "GPDs" 
INTERPOLATION_TABLE_PATH = BASE_PATH / "InterpolationTables"

MSTW_PATH = PDF_PATH / "MSTW.csv"
AAC_PATH = PDF_PATH / "AAC.csv"

##################################
###########    Cache   ###########
##################################
#### Make sure to clear after ####
## parameters have been changed ##
##################################
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

# If no lattice data:
ETA_ARRAY = [0,0.33,0.1]
T_ARRAY = [-0.69,-0.69,-0.23]
MU_ARRAY = [2,2,2]

PARTICLES = ["quark","gluon"]
MOMENTS = ["singlet","non_singlet_isoscalar","non_singlet_isovector"]
LABELS = ["A","Atilde"]
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
    # ("2008.10573","non_singlet_isovector","H",0.00, -0.69, 2.00): ("mediumturquoise","000_069_200"),
    # ("2008.10573","non_singlet_isovector","H",0.33, -0.69, 2.00): ("green","033_069_200"),
    # ("2008.12474","non_singlet_isovector","H",0.00, -0.39, 3.00): ("purple","000_039_300"),
    # ("2312.10829","non_singlet_isovector","H",0.10, -0.23, 2.00): ("orange","010_023_200"),
    # ("2008.10573","non_singlet_isovector","Htilde",0.00, -0.69, 2.00): ("mediumturquoise","000_069_200"),
    # ("2008.10573","non_singlet_isovector","Htilde",0.33, -0.69, 2.00): ("green","033_069_200"),
    # ("2112.07519","non_singlet_isovector","Htilde",0.00, -0.39, 3.00): ("purple","000_039_300"),
    # ("2008.10573","non_singlet_isovector","E",0.00, -0.69, 2.00): ("mediumturquoise","000_069_200"),
    # ("2008.10573","non_singlet_isovector","E",0.33, -0.69, 2.00): ("green","033_069_200"),
    # ("2312.10829","non_singlet_isovector","E",0.10, -0.23, 2.00): ("orange","010_023_200"),
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
                "Htilde": "Atilde"
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

########################
### Model Parameters ###
########################

REGGE_SLOPES = {
    "vector": {
        "non_singlet_isovector": {
            "A": {
                "lo": 0.6582,
                "nlo": 0.6345
            },
            "B": {
                "lo": 1.4581,
                "nlo": 1.3929
            }
        },
        "non_singlet_isoscalar": {
            "A": {
                "lo": 0.9426,
                "nlo": 0.9492
            },
            "B": {
                "lo": 1.1298,
                "nlo": 1.1368
            }
        },
        "singlet": {
            "A": {
                "lo": (0.5306,1.9267, 0.6552, 5.1354),
                "nlo": (0.5044,1.9657, 0.5584, 5.2081)
            },
            "B": {
                "lo": (0,0,0),
                "nlo": (0,0,0)
            }
        }
    },
    "axial": {
        "non_singlet_isovector": {
            "Atilde": {
                "lo": 0.4553,
                "nlo": 0.3140
            }
        },
        "non_singlet_isoscalar": {
            "Atilde": {
                "lo": 0.2974,
                "nlo": 0.3140
            }
        },
        "singlet": {
            "Atilde": {
                "lo": (0.8454, 1.179, 0.490, 0.744),
                "nlo": (0.7186, 1.179, 0.490, 0.744)
            },
        }
    }
}

MOMENT_NORMALIZATIONS = {
    "vector": {
        "non_singlet_isovector": {
            "A": {
                "lo": 1,
                "nlo": 1
            },
            "B": {
                "lo": 3.8319,
                "nlo": 3.8170
            }
        },
        "non_singlet_isoscalar": {
            "A": {
                "lo": 0.9879,
                "nlo": 0.9703
            },
            "B": {
                "lo": -0.1215,
                "nlo": -0.1194
            }
        },
        "singlet": {
            "A": {
                "lo": (1,1,1,1),
                "nlo": (1,1,1,1)
            },
            "B": {
                "lo": (1,1,1,1),
                "nlo": (1,1,1,1)
            }
        }
    },
    "axial": {
        "non_singlet_isovector": {
            "Atilde": {
                "lo": 1.0010,
                "nlo": 0.9890
            }
        },
        "non_singlet_isoscalar": {
            "Atilde": {
                "lo": 0.7129,
                "nlo": 0.7152
            }
        },
        "singlet": {
            "Atilde": {
                "lo": (1,1,1,1),
                "nlo": (1,1,1,1)
            },
        }
    }
}