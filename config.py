# Cleaner path structure
from pathlib import Path
# Set cache memory below
from joblib import Memory

# Set precision globally
import mpmath as mp
mp.dps = 16

########################################
#### Dictionaries and data handling ####
####       Change as required       ####
########################################

# Parent directory for data
BASE_PATH = Path("/mnt/c/Users/flori/Documents/PostDoc/Data/stringy-gpds")
# Folder for generated plots
PLOT_PATH = Path("/mnt/c/Users/flori/Documents/PostDoc/Plots/stringy-gpds")
# PDF location
PDF_PATH = Path("/mnt/c/Users/flori/Documents/PostDoc/Data/PDFs")

# Subdirectories for cleaner file handling
IMPACT_PARAMETER_MOMENTS_PATH = BASE_PATH / "ImpactParameterMoments"
MOMENTUM_SPACE_MOMENTS_PATH = BASE_PATH / "MomentumSpaceMoments"
GPD_PATH = BASE_PATH / "GPDs" 
ANOMALOUS_DIMENSIONS_PATH = BASE_PATH / "AnomalousDimensions"

MSTW_PATH = PDF_PATH / "MSTW.csv"
AAC_PATH = PDF_PATH / "AAC.csv"

# Add some colors
saturated_pink = (1.0, 0.1, 0.6)  

# ArXiv ID and renormalization scale mu
PUBLICATION_MAPPING = {
    "2305.11117": ("cyan",2),
    "0705.4295": ("orange",2),
    "1908.10706": (saturated_pink,2),
    "2310.08484": ("darkblue",2),
    "2410.03539": ("green",2)
# Add more publication IDs and corresponding colors here
}

GPD_PUBLICATION_MAPPING = {
    # publication ID, GPD type, GPD label, eta, t ,mu
    ("2008.10573","non_singlet_isovector","Htilde",0.00, -0.69, 2.00): ("mediumturquoise","000_069_200"),
    ("2008.10573","non_singlet_isovector","Htilde",0.33, -0.69, 2.00): ("green","033_069_200"),
    ("2112.07519","non_singlet_isovector","Htilde",0.00, -0.39, 3.00): ("purple","000_039_300"),
    ("2008.10573","non_singlet_isovector","E",0.00, -0.69, 2.00): ("mediumturquoise","000_069_200"),
    ("2008.10573","non_singlet_isovector","E",0.33, -0.69, 2.00): ("green","033_069_200"),
    ("2312.10829","non_singlet_isovector","E",0.10, -0.23, 2.00): ("orange","010_023_200"),
    # No data:
    # ("","non_singlet_isoscalar","E",0.00, -0.00, 2.00): ("purple","000_000_200"),
    # ("","non_singlet_isoscalar","E",0.33, -0.69, 2.00): ("green","033_069_200"),
    # ("","non_singlet_isoscalar","E",0.10, -0.23, 2.00): ("orange","010_023_200"),
    # ("","non_singlet_isoscalar","E",0.33, -0.69, 2.00): (saturated_pink,"000_039_200"),
# Add more publication IDs and corresponding colors here
}

GPD_LABEL_MAP ={"H": "A",
                "E": "B",
                "Htilde": "Atilde"
                    }

REGGE_SLOPES = {
    "vector": {
        "non_singlet_isovector": {
            "A": {
                "LO": 0.6582,
                "NLO": 0.6345
            },
            "B": {
                "LO": 1.4581,
                "NLO": 1.3929
            }
        },
        "non_singlet_isoscalar": {
            "A": {
                "LO": 0.9426,
                "NLO": 0.9492
            },
            "B": {
                "LO": 1.1298,
                "NLO": 1.1368
            }
        },
        "singlet": {
            "A": {
                "LO": (0.5306,1.9267, 0.6552, 5.1354),
                "NLO": (0.5044,1.9657, 0.5584, 5.2081)
            },
            "B": {
                "LO": (0,0,0),
                "NLO": (0,0,0)
            }
        }
    },
    "axial": {
        "non_singlet_isovector": {
            "Atilde": {
                "LO": 0.4553,
                "NLO": 0.3140
            }
        },
        "non_singlet_isoscalar": {
            "Atilde": {
                "LO": 0.2974,
                "NLO": 0.3140
            }
        },
        "singlet": {
            "Atilde": {
                "LO": (0.2974, 1.179, 0.490, 0.744),
                "NLO": (0.3140, 1.179, 0.490, 0.744)
            },
        }
    }
}

MOMENT_NORMALIZATIONS = {
    "vector": {
        "non_singlet_isovector": {
            "A": {
                "LO": 1,
                "NLO": 1
            },
            "B": {
                "LO": 3.8319,
                "NLO": 3.8170
            }
        },
        "non_singlet_isoscalar": {
            "A": {
                "LO": 0.9879,
                "NLO": 0.9703
            },
            "B": {
                "LO": -0.1215,
                "NLO": -0.1194
            }
        },
        "singlet": {
            "A": {
                # "LO": (0.7914,0.7539,1.3580,0.7738),
                # "NLO": (0.7829,0.7019, 1.3500, 0.7058),
                "LO": (1,1,1,1),
                "NLO": (1,1,1,1)
            },
            "B": {
                "LO": (1,1,1,1),
                "NLO": (1,1,1,1)
            }
        }
    },
    "axial": {
        "non_singlet_isovector": {
            "Atilde": {
                "LO": 1.0010,
                "NLO": 0.9890
            }
        },
        "non_singlet_isoscalar": {
            "Atilde": {
                "LO": 0.7129,
                "NLO": 0.7152
            }
        },
        "singlet": {
            "Atilde": {
                "LO": (1,1,1,1),
                "NLO": (1,1,1,1)
            },
        }
    }
}

####################
####   Cache    ####
####################

cache_dir = "/mnt/c/Users/flori/Documents/PostDoc/Jupyter/Python/cache"
memory = Memory(cache_dir, verbose = 0)