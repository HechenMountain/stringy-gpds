"""Validation of an input PDF re-fit.

1. Charges / sum rules from the moments at the input scale.
2. Forward-limit closure: mellin_barnes_gpd at (eta = 1e-5, t = -1e-4, mu = mu_input)
   must reproduce the input PDF combinations (cf. checks at core.py:~2050).

mellin_barnes_gpd only works at kinematics registered in user_config.py (the
gpd_errors/interpolation gates), so this script temporarily swaps in a validation
user_config with the forward kinematic triple and central errors only, and restores
the original afterwards.

Usage:
    python scripts/validate.py --pdf-set JAM22
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import _gpd_common as gc

parser = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--pdf-set", default="JAM22")
parser.add_argument("--mu-input", type=float, default=2)
args = parser.parse_args()

ORIG = gc.CONFIG_PATH.read_text()

VALIDATION_CFG = f"""
pdf_set = "{args.pdf_set}"
mu_input = {args.mu_input:g}
interpolate_moments = False
nd_evolved_complex_moment = False
eta_array = [1e-05]
t_array = [-0.0001]
mu_array = [{args.mu_input:g}]
particles = ["quark", "gluon"]
moments = ["non_singlet_isovector", "singlet"]
labels = ["A", "Atilde"]
orders = ["nlo"]
errors = ["central"]
"""

gc.CONFIG_PATH.write_text(VALIDATION_CFG)
try:
    import stringy_gpds.config as cfg
    assert not cfg.INTERPOLATE_MOMENTS
    from stringy_gpds.core import evolve_conformal_moment, mellin_barnes_gpd
    import stringy_gpds.unpolarized_pdf as unpol
    import stringy_gpds.polarized_pdf as pol

    EPS_T = -1e-6

    print(f"=== moments at input scale mu = {args.mu_input:g} (nlo) ===", flush=True)
    F1iv = evolve_conformal_moment(complex(1), 0, EPS_T, args.mu_input, particle="quark",
                                   moment_type="non_singlet_isovector", moment_label="A", evolution_order="nlo")
    print(f"F1 isovector(0)  = {complex(F1iv).real:.4f}   (target 1.0)")
    gA = evolve_conformal_moment(complex(1), 0, EPS_T, args.mu_input, particle="quark",
                                 moment_type="non_singlet_isovector", moment_label="Atilde", evolution_order="nlo")
    print(f"gA               = {complex(gA).real:.4f}   (target 1.2723)")
    Aq = evolve_conformal_moment(complex(2), 0, EPS_T, args.mu_input, particle="quark",
                                 moment_type="singlet", moment_label="A", evolution_order="nlo")
    Ag = evolve_conformal_moment(complex(2), 0, EPS_T, args.mu_input, particle="gluon",
                                 moment_type="singlet", moment_label="A", evolution_order="nlo")
    print(f"A_q(0) = {complex(Aq).real:.4f} (lattice 0.510), A_g(0) = {complex(Ag).real:.4f} (lattice 0.501), "
          f"sum = {complex(Aq).real + complex(Ag).real:.4f}", flush=True)

    print(f"\n=== forward-limit MB closure (eta = 1e-5, t = -1e-4, mu = {args.mu_input:g}, nlo) ===", flush=True)
    targets = [
        ("non_singlet_isovector", "A", "quark", lambda x: unpol.uv_minus_dv_pdf(x)),
        ("singlet", "A", "quark", lambda x: unpol.uv_plus_dv_plus_S_pdf(x)),
        ("singlet", "A", "gluon", lambda x: x * unpol.gluon_pdf(x)),
        ("non_singlet_isovector", "Atilde", "quark",
         lambda x: pol.polarized_uv_pdf(x) - pol.polarized_dv_pdf(x)),
    ]
    for mtype, label, particle, ref in targets:
        for x in [0.1, 0.3]:
            # n_jobs=1: loky fails to pickle gmpy2 mpq objects on this setup
            gpd = mellin_barnes_gpd(x, 1e-5, -1e-4, args.mu_input, particle=particle, moment_type=mtype,
                                    moment_label=label, evolution_order="nlo", j_max=40, n_jobs=1)
            print(f"{particle:6s} {mtype:22s} {label:6s} x={x:5.2f}: MB = {float(gpd):9.4f}   input pdf = {float(ref(x)):9.4f}",
                  flush=True)
finally:
    gc.CONFIG_PATH.write_text(ORIG)
    print("restored user_config.py")
