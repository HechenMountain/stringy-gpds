"""Re-fit Regge slopes and moment normalizations to form-factor (pseudo-)data for
one input PDF set.

Route (matches how the published config.py values were produced):
- fit_non_singlet_slopes: A, B (Dirac/Pauli dipoles), Atilde (axial dipoles)
  for isovector & isoscalar -> (norm, alpha') each.
- fit_singlet_slopes_A: independent quark/gluon A and D fits vs the lattice
  gravitational FFs (2310.08484). The A normalizations are held at 1
  (fix_norm_A=True), so the forward limit is the momentum fraction predicted by
  the input PDFs and A_q(0) + A_g(0) = 1 holds by the momentum sum rule; only
  the t-dependence is fitted. The D normalizations are fitted.
- fit_singlet_D_slopes: combined D fit of the evolved moments. Run for comparison
  only: with these PDFs it tends to pin its parameters at the bounds, so the
  separate D fits above are used for the config.
- fit_singlet_slopes_Atilde: polarized singlet quark slope (shape fit at j = 2
  vs the axial GA dipole) -> alpha' (first entry of the Atilde singlet tuple;
  the remaining entries are kept, no fit function in fit.py produces them).

Usage:
    python scripts/refit_slopes.py --pdf-set JAM22

Writes fit_output/slopes_<pdf-set>.json (consumed by the table-generation pipeline
via _gpd_common.pdf_set_config_lines) and prints ready-to-paste config.py dictionaries.

Without --pdf-set, refits whatever pdf_set is already live in user_config.py, in
place, with no swap/restore. With --pdf-set, the live user_config.py is temporarily
switched to that set and restored afterwards -- pass the set you already have active
if you don't want a restore.
"""
import argparse
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "fit_output"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import _gpd_common as gc

parser = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--pdf-set", default=None,
    help="pdf_set to refit (default: whatever is already live in user_config.py)")
args = parser.parse_args()

ORIG = gc.CONFIG_PATH.read_text() if args.pdf_set else None
if args.pdf_set:
    gc.set_pdf_set(args.pdf_set)

# previous Atilde singlet tuple entries 2-4 (not refit by fit.py, see docstring)
ATILDE_SINGLET_REST = {"lo": (1.179, 0.490, 0.744), "nlo": (1.179, 0.490, 0.744)}

try:
    from stringy_gpds.fit import (fit_non_singlet_slopes, fit_singlet_slopes_A,
                                  fit_singlet_D_slopes, fit_singlet_slopes_Atilde)
    from stringy_gpds.core import evolve_conformal_moment
    import stringy_gpds.config as cfg

    print(f"input PDF set: {cfg.MSTW_PATH.name} / {cfg.AAC_PATH.name}, mu0 = {cfg.MU_INPUT} GeV")

    results = {}
    for order in ["nlo", "lo"]:
        print(f"\n===================== {order} =====================")
        ns = fit_non_singlet_slopes(evolution_order=order, plot=False)
        sA = fit_singlet_slopes_A(evolution_order=order, plot=False, fix_norm_A=True)
        norm_Aq, alpha_ud = sA["quark_A"]
        norm_Ag, alpha_T = sA["gluon_A"]
        norm_Dq, alpha_s = sA["quark_D"]
        norm_Dg, alpha_S = sA["gluon_D"]

        # combined fit of the evolved D moments, for comparison only
        try:
            combined = fit_singlet_D_slopes(norm_Aq, alpha_T, norm_Ag, evolution_order=order,
                                            plot=False, alpha_prime_ud=alpha_ud)
            print(f"[combined D fit, not used] alpha'_s = {combined[0]:.4f}, norm_Dq = {combined[1]:.4f}, "
                  f"alpha'_S = {combined[2]:.4f}, norm_Dg = {combined[3]:.4f}")
        except Exception as exc:
            print(f"[combined D fit failed: {exc}]")

        _, alpha_At = fit_singlet_slopes_Atilde(evolution_order=order, plot=False)

        results[order] = {
            "non_singlet": {f"{k[0]}|{k[1]}": list(map(float, v)) for k, v in ns.items()},
            "singlet_A_slopes": [float(alpha_ud), float(alpha_s), float(alpha_T), float(alpha_S)],
            "singlet_A_norms": [float(norm_Aq), float(norm_Dq), float(norm_Ag), float(norm_Dg)],
            "singlet_Atilde_alpha": float(alpha_At),
        }

    # ---- assemble config dictionaries ----
    def g(order, mt, lbl, i):
        return results[order]["non_singlet"][f"{mt}|{lbl}"][i]

    def fmt(v):
        if isinstance(v, (list, tuple)):
            return "(" + ", ".join(f"{x:.4f}" for x in v) + ")"
        return f"{v:.4f}"

    slopes, norms = {}, {}
    for mt in ["non_singlet_isovector", "non_singlet_isoscalar"]:
        slopes[mt] = {lbl: {o: g(o, mt, lbl, 1) for o in ["lo", "nlo"]} for lbl in ["A", "B", "Atilde"]}
        norms[mt] = {lbl: {o: g(o, mt, lbl, 0) for o in ["lo", "nlo"]} for lbl in ["A", "B", "Atilde"]}

    singlet_slopes_AB = {o: list(results[o]["singlet_A_slopes"]) for o in ["lo", "nlo"]}
    singlet_norms_A = {o: list(results[o]["singlet_A_norms"]) for o in ["lo", "nlo"]}
    singlet_slopes_At = {o: [results[o]["singlet_Atilde_alpha"], *ATILDE_SINGLET_REST[o]]
                         for o in ["lo", "nlo"]}
    # B shares the D entries of A but zeroes the A normalizations
    singlet_norms_B = {o: [0, singlet_norms_A[o][1], 0, singlet_norms_A[o][3]] for o in ["lo", "nlo"]}

    slopes["singlet"] = {"A": singlet_slopes_AB, "B": singlet_slopes_AB, "Atilde": singlet_slopes_At}
    norms["singlet"] = {"A": singlet_norms_A, "B": singlet_norms_B,
                        "Atilde": {o: [1, 1, 1, 1] for o in ["lo", "nlo"]}}

    # ---- momentum sum-rule cross-check with the freshly assembled parameters ----
    cfg.REGGE_SLOPES = slopes
    cfg.MOMENT_NORMALIZATIONS = norms
    EPS_T = -1e-6
    Aq = complex(evolve_conformal_moment(complex(2), 0, EPS_T, cfg.MU_INPUT, particle="quark",
                 moment_type="singlet", moment_label="A", evolution_order="nlo")).real
    Ag = complex(evolve_conformal_moment(complex(2), 0, EPS_T, cfg.MU_INPUT, particle="gluon",
                 moment_type="singlet", moment_label="A", evolution_order="nlo")).real
    print(f"\n[sum rule] A_q(0) = {Aq:.4f}, A_g(0) = {Ag:.4f}, sum = {Aq + Ag:.4f}  (target 1.0000)")

    out = OUT / f"slopes_{cfg.PDF_SET}.json"
    out.write_text(json.dumps({
        "regge_slopes": slopes, "moment_normalizations": norms, "raw": results,
        "sum_rule": {"A_q": Aq, "A_g": Ag, "sum": Aq + Ag}}, indent=2))
    print("wrote", out)

    print("\n--- regge_slopes ---")
    for mt in slopes:
        for lbl in slopes[mt]:
            print(f"{mt:24s} {lbl:6s} lo = {fmt(slopes[mt][lbl]['lo'])}, nlo = {fmt(slopes[mt][lbl]['nlo'])}")
    print("\n--- moment_normalizations ---")
    for mt in norms:
        for lbl in norms[mt]:
            print(f"{mt:24s} {lbl:6s} lo = {fmt(norms[mt][lbl]['lo'])}, nlo = {fmt(norms[mt][lbl]['nlo'])}")
finally:
    if ORIG is not None:
        gc.CONFIG_PATH.write_text(ORIG)
        print("restored user_config.py")
