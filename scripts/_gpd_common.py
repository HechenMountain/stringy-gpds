"""Shared CLI flags and kinematics/config helpers for the H-GPD table pipeline
(generate_moment_tables.py, generate_interp_tables.py, generate_gpd_tables.py,
combine_gpd_tables.py, run_gpd_pipeline.py)
"""
import argparse
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

USER_PATH = Path.home() / "stringy-gpds"
CONFIG_PATH = USER_PATH / "user_config.py"

PARTICLES = ["quark", "gluon"]
MOMENT_TYPES = ["non_singlet_isovector", "non_singlet_isoscalar", "singlet"]
ORDERS = ["lo", "nlo"]
ERRORS = ["central", "plus", "minus"]
M_N = 0.93827

# GPD label -> moment label (mirrors config.py's GPD_LABEL_MAP)
GPD_LABEL_MAP = {"H": "A", "E": "B", "Htilde": "Atilde", "Etilde": "Btilde"}
# GPD label -> LaTeX symbol, for plot titles/axes only (not used in filenames/headers)
GPD_TEX_SYMBOL = {"H": "H", "E": "E", "Htilde": r"\widetilde{H}", "Etilde": r"\widetilde{E}"}


def add_common_args(parser):
    """Flags shared by every stage: which PDF set/label/channels/kinematics to run."""
    parser.add_argument("--pdf-set", default="JAM22",
        help="input PDF set; must match pdfs/<set>.csv (fit.fit_input_pdfs). "
             "If fit_output/slopes_<set>.json exists (see refit_slopes.py) its "
             "regge_slopes/moment_normalizations override config.py's; otherwise "
             "the set is assumed to already be configured there (true for JAM22).")
    parser.add_argument("--label", default="H", choices=sorted(GPD_LABEL_MAP),
        help="GPD label, mapped to a moment label via GPD_LABEL_MAP (default H -> A)")
    parser.add_argument("--particles", nargs="+", default=list(PARTICLES), choices=PARTICLES,
        help="particle species to include (default: both)")
    parser.add_argument("--moments", nargs="+", default=list(MOMENT_TYPES), choices=MOMENT_TYPES,
        help="moment/GPD types to include (default: all three)")
    parser.add_argument("--order", default="nlo", choices=ORDERS,
        help="evolution order (default: nlo)")
    parser.add_argument("--errors", nargs="+", default=list(ERRORS), choices=ERRORS,
        help="error bands to compute (default: all three)")
    parser.add_argument("--eta", nargs="+", type=float, required=True,
        help="skewness values, e.g. --eta 0 0.1 0.2 0.3")
    parser.add_argument("--mu2", nargs="+", type=float, required=True,
        help="output scales mu^2 [GeV^2], e.g. --mu2 2 10")
    t_group = parser.add_mutually_exclusive_group(required=True)
    t_group.add_argument("--t-min", action="store_true",
        help="use the minimal |t| at Delta_perp^2 = 0: t_min(eta) = -4 eta^2 m_N^2/(1-eta^2)")
    t_group.add_argument("--t", nargs="+", type=float,
        help="literal t values [GeV^2], one per --eta, same order")
    return parser


def add_moment_args(parser):
    """Extra flags for the moment-table stage: the conformal-spin j range."""
    g = parser.add_argument_group("moment table options")
    g.add_argument("--j-max", type=int, default=5, help="highest conformal spin j")
    g.add_argument("--j-min-singlet", type=int, default=2,
        help="lowest j for singlet channels (2 = the momentum moment, the label A default)")
    g.add_argument("--j-min-nonsinglet", type=int, default=1,
        help="lowest j for non-singlet channels")
    return parser


def add_gpd_args(parser):
    """Extra flags for the x-space GPD stage: how many x-points to sample."""
    g = parser.add_argument_group("GPD x-grid options")
    g.add_argument("--n-points-singlet", type=int, default=100,
        help="x-grid points for singlet channels, 0<x<1 (matches plot_gpds' n_points)")
    g.add_argument("--n-points-nonsinglet", type=int, default=200,
        help="x-grid points for non-singlet channels, -1<x<1")
    return parser


def current_pdf_set():
    m = re.search(r'pdf_set\s*=\s*"([^"]*)"', CONFIG_PATH.read_text())
    return m.group(1) if m else None


def set_pdf_set(name, mu_input=2):
    """Set (or insert) pdf_set = name in the live user_config.py."""
    text = CONFIG_PATH.read_text()
    if "pdf_set" in text:
        text = re.sub(r'pdf_set\s*=\s*"[^"]*"', f'pdf_set = "{name}"', text)
    else:
        text = f'pdf_set = "{name}"\nmu_input = {mu_input:g}\n\n' + text
    CONFIG_PATH.write_text(text)
    print(f"user_config.py: pdf_set = {name!r}")


def build_parser(description):
    return add_common_args(argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter))


def resolve_combos(particles, moment_types):
    """[(particle, moment_type), ...], skipping the unphysical non_singlet+gluon
    pairing (gluon only has a singlet channel; config.py's REGGE_SLOPES has no
    gluon entry for non_singlet_isovector/isoscalar)."""
    combos = []
    for particle in particles:
        for mtype in moment_types:
            if particle == "gluon" and mtype != "singlet":
                print(f"skip: gluon has no {mtype} channel (gluon is singlet-only)", flush=True)
                continue
            combos.append((particle, mtype))
    return combos


def t_min(eta):
    """Minimal |t| [GeV^2] at Delta_perp^2 = 0."""
    return 0.0 if eta == 0 else round(-4 * eta ** 2 * M_N ** 2 / (1 - eta ** 2), 2)


def resolve_t(args):
    """{eta: t} for every requested eta, via t_min(eta) or the literal --t list."""
    if args.t_min:
        return {eta: t_min(eta) for eta in args.eta}
    if len(args.t) != len(args.eta):
        raise SystemExit(f"--t must have one value per --eta ({len(args.eta)} eta, "
                          f"{len(args.t)} t given)")
    return dict(zip(args.eta, args.t))


def resolve_triples(args):
    """[(eta, t, mu), ...], eta outer / mu2 inner (matches the cfg.ETA_ARRAY/T_ARRAY/
    MU_ARRAY convention: the mu grid repeats for every eta)."""
    t_of = resolve_t(args)
    return [(eta, t_of[eta], math.sqrt(mu2)) for eta in args.eta for mu2 in args.mu2]


def code(v):
    """Zero-padded 3-digit filename code (matches helpers.generate_filename)."""
    return f"{abs(v):.2f}".replace(".", "").zfill(3)


def pdf_set_config_lines(pdf_set, moment_label, particles, moment_types, order, errors,
                          interpolate, triples=None):
    """user_config.py text selecting the PDF set and the requested channels. Omit
    triples (None) for the moment stage, which calls evolve_conformal_moment directly
    at integer j and so never looks up the eta/t/mu_array kinematics; pass it for any
    stage that needs registered (eta, t, mu) triples (interpolation tables, GPD x-space
    reconstruction). If fit_output/slopes_<pdf_set>.json exists it overrides config.py's
    regge_slopes/moment_normalizations (see refit_slopes.py)."""
    lines = [
        f'pdf_set = "{pdf_set}"',
        "mu_input = 2",
        f"interpolate_moments = {bool(interpolate)}",
        "nd_evolved_complex_moment = False",
    ]
    if triples is not None:
        lines += [
            "eta_array = " + repr([t[0] for t in triples]),
            "t_array = " + repr([t[1] for t in triples]),
            "mu_array = " + repr([t[2] for t in triples]),
        ]
    lines += [
        "particles = " + repr(list(particles)),
        "moments = " + repr(list(moment_types)),
        "labels = " + repr([moment_label]),
        "orders = " + repr([order]),
        "errors = " + repr(list(errors)),
    ]
    slopes_path = ROOT / "fit_output" / f"slopes_{pdf_set}.json"
    if slopes_path.exists():
        overrides = json.loads(slopes_path.read_text())
        lines.append("regge_slopes = " + repr(overrides["regge_slopes"]))
        lines.append("moment_normalizations = " + repr(overrides["moment_normalizations"]))
    return "\n".join(lines) + "\n"
