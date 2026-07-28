"""Conformal moment tables of a GPD (any label, any order) for one input PDF set,
over an arbitrary (eta, t, mu) grid.

Integer-j evolve_conformal_moment computes directly (the interpolation-table path is
gated on Im(j) != 0), so no interpolation tables are needed for this stage.

Usage:
    python scripts/generate_moment_tables.py --pdf-set JAM22 --label H \\
        --eta 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 --mu2 2 10 --t-min

Writes  data/moments/<SET>/moments_<label>_<moment_type>_<particle>_<order>_musq_<mu2>.csv
        (columns: eta, t, mu, j, F_central, F_plus, F_minus -- whichever of --errors requested)
and     plots/moments/<SET>/moment_<label>_<moment_type>_<particle>.pdf   (F_j vs eta, +/- band)

The set's Regge slopes / moment normalizations are configured via a temporary
user_config.py (see _gpd_common.pdf_set_config_lines), which is restored afterwards.
"""
import csv
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import _gpd_common as gc

parser = gc.build_parser(__doc__)
gc.add_moment_args(parser)
args = parser.parse_args()

MOMENT_LABEL = gc.GPD_LABEL_MAP[args.label]
COMBOS = [(particle, mtype, args.j_min_singlet if mtype == "singlet" else args.j_min_nonsinglet)
          for particle, mtype in gc.resolve_combos(args.particles, args.moments)]
T_OF = gc.resolve_t(args)

OUT_DATA = ROOT / "data" / "moments" / args.pdf_set
OUT_PLOT = ROOT / "plots" / "moments" / args.pdf_set
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_PLOT.mkdir(parents=True, exist_ok=True)

USETEX = shutil.which("latex") is not None
plt.rcParams.update({"text.usetex": USETEX, "font.family": "serif",
                     "font.size": 11, "axes.titlesize": 12})

# ---- configure the requested input PDF set via a temporary user_config.py ----
ORIG = gc.CONFIG_PATH.read_text()
gc.CONFIG_PATH.write_text(gc.pdf_set_config_lines(
    args.pdf_set, MOMENT_LABEL, args.particles, args.moments, args.order, args.errors,
    interpolate=False))

try:
    import stringy_gpds.config as cfg
    assert cfg.PDF_SET == args.pdf_set, cfg.PDF_SET
    from stringy_gpds.core import evolve_conformal_moment

    def moment(j, eta, t, mu, particle, moment_type, error_type):
        # Pass an int j: evolve_conformal_moment then takes its integer branch, which
        # exactly resums the non-diagonal (skewness) mixing and returns a real scalar
        # (a complex/tuple is only returned for non-integer j, used by the MB integral).
        val = evolve_conformal_moment(int(j), eta, t, mu, particle=particle,
                                      moment_type=moment_type, moment_label=MOMENT_LABEL,
                                      evolution_order=args.order, error_type=error_type)
        if isinstance(val, tuple):
            val = val[0]
        return float(complex(val).real)

    n_bad = 0
    for particle, mtype, jmin in COMBOS:
        js = list(range(jmin, args.j_max + 1))
        # curves[mu2][j][err] = [] over args.eta, for the plot
        curves = {}
        for mu2 in args.mu2:
            mu = math.sqrt(mu2)
            rows = []
            curves[mu2] = {j: {err: [] for err in args.errors} for j in js}
            for eta in args.eta:
                t = T_OF[eta]
                for j in js:
                    vals = {err: moment(j, eta, t, mu, particle, mtype, err) for err in args.errors}
                    rows.append([eta, t, mu, j] + [vals[err] for err in args.errors])
                    for err in args.errors:
                        curves[mu2][j][err].append(vals[err])
                    if not np.all(np.isfinite(list(vals.values()))):
                        n_bad += 1
                        print(f"  !! non-finite: {args.pdf_set} {mtype} {particle} "
                              f"eta={eta} t={t} mu^2={mu2} j={j} -> {vals}", flush=True)
            fname = OUT_DATA / (f"moments_{args.label}_{mtype}_{particle}_{args.order}_"
                                 f"musq_{mu2:02.0f}.csv")
            with open(fname, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["eta", "t", "mu", "j"] + [f"F_{err}" for err in args.errors])
                w.writerows(rows)
            print(f"wrote {fname}", flush=True)

        # ---- plot F_j vs eta, one panel per mu^2, +/- band per j (if available) ----
        fig, axes = plt.subplots(1, len(args.mu2), figsize=(6.2 * len(args.mu2), 4.6))
        cmap = plt.get_cmap("viridis")
        for ax, mu2 in zip(np.atleast_1d(axes), args.mu2):
            for k, j in enumerate(js):
                c = np.array(curves[mu2][j].get("central", curves[mu2][j][args.errors[0]]))
                col = cmap(k / max(1, len(js) - 1))
                ax.plot(args.eta, c, "-o", ms=3, color=col, label=rf"$j={j}$")
                if "plus" in curves[mu2][j] and "minus" in curves[mu2][j]:
                    p = np.array(curves[mu2][j]["plus"])
                    m = np.array(curves[mu2][j]["minus"])
                    ax.fill_between(args.eta, np.minimum(m, p), np.maximum(m, p), color=col, alpha=0.2)
            ax.set_xlabel(r"$\eta$")
            ax.set_ylabel(r"$F_j(\eta,\,t(\eta),\,\mu)$")
            ax.set_title(rf"$\mu^2={mu2:g}\;\mathrm{{GeV}}^2$")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9, frameon=False, ncol=2)
        pretty = mtype.replace("_", r"\_") if USETEX else mtype
        fig.suptitle(rf"{args.pdf_set}: {args.label} {pretty} {particle} ({args.order.upper()})", y=1.02)
        pdf = OUT_PLOT / f"moment_{args.label}_{mtype}_{particle}.pdf"
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {pdf}", flush=True)

    # ---- sanity check: isovector charge F_1^(u-d)(eta=0,t=0) ~ 1 for label H ----
    check_combo = ("quark", "non_singlet_isovector", args.j_min_nonsinglet)
    if (args.label == "H" and 0.0 in args.eta and T_OF[0.0] == 0.0
            and check_combo in COMBOS and 2 in args.mu2):
        f1 = moment(1, 0.0, 0.0, math.sqrt(2), "quark", "non_singlet_isovector", "central")
        print(f"\n[check] {args.pdf_set} F_1^(u-d)(eta=0, t=0, mu^2=2) = {f1:.4f}  "
              f"(isovector charge, target ~1.0)")
    if n_bad:
        print(f"[warning] {n_bad} non-finite moment value(s) encountered.", flush=True)
finally:
    gc.CONFIG_PATH.write_text(ORIG)
    print("restored user_config.py", flush=True)
