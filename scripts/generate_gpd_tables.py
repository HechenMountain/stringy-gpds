"""x-space reconstruction of a GPD (any label/order/channels) for one input PDF set,
over an arbitrary (eta, t, mu) grid. Requires the interpolation tables from
generate_interp_tables.py for the same set/label/kinematics.

Usage:
    python scripts/generate_gpd_tables.py --pdf-set JAM22 --label H \\
        --eta 0 0.1 0.2 --mu2 2 10 --t-min

For each (particle, moment_type) channel and each (eta, t, mu) triple the central GPD
is reconstructed with core.mellin_barnes_gpd on a dense x-grid matching plot_gpds'
n_points keyword (--n-points-nonsinglet/--n-points-singlet; singular endpoints x=0,
x=+/-1 avoided, +/-eta crossovers inserted). The plus/minus error curves are the
central curve scaled by the per-kinematic factor core.gpd_errors[...].

The PDF-dependent caches are purged before importing core so a
previous set's values are not reused. CSVs -> data/gpds/<SET>/, overview plots ->
plots/gpds/<SET>/.
"""
import os
import shutil
import sys
import time
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
gc.add_gpd_args(parser)
args = parser.parse_args()

MOMENT_LABEL = gc.GPD_LABEL_MAP[args.label]
COMBOS = gc.resolve_combos(args.particles, args.moments)
TRIPLES = gc.resolve_triples(args)
# central is always needed as the base value for the plus/minus scaling below, even
# if the user only asked to save a subset via --errors
BUILD_ERRORS = sorted(set(args.errors) | {"central"})

USETEX = shutil.which("latex") is not None
plt.rcParams.update({"text.usetex": USETEX, "font.family": "serif",
                     "font.size": 11, "axes.titlesize": 12})

OUT_DATA = ROOT / "data" / "gpds" / args.pdf_set
OUT_PLOT = ROOT / "plots" / "gpds" / args.pdf_set
OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_PLOT.mkdir(parents=True, exist_ok=True)


def x_grid(mtype, eta):
    """Dense x-grid a la plot_gpds' n_points; singlet on (1e-2,1), non-singlet on
    (-1,1). The singular endpoints x=0, x=+/-1 are dropped and the ERBL/DGLAP
    crossovers +/-eta inserted (full float precision, no rounding, so the grid
    spacing is preserved)."""
    if mtype == "singlet":
        xs = np.linspace(1e-2, 1.0, args.n_points_singlet)
        xs = xs[xs < 1.0 - 1e-9]                            # drop x=1
        cross = (round(eta, 10),) if eta > 0 else ()       # +eta only (x>0)
    else:
        xs = np.linspace(-1.0, 1.0, args.n_points_nonsinglet)
        xs = xs[(np.abs(xs) < 1.0 - 1e-9) & (xs != 0.0)]   # drop +/-1 and 0
        cross = tuple(e for e in (round(eta, 10), round(-eta, 10)) if e != 0)
    pts = set(xs.tolist()) | {c for c in cross if abs(c) < 1.0}
    return np.array(sorted(pts))


ORIG = gc.CONFIG_PATH.read_text()
gc.CONFIG_PATH.write_text(gc.pdf_set_config_lines(
    args.pdf_set, MOMENT_LABEL, args.particles, args.moments, args.order, BUILD_ERRORS,
    interpolate=True, triples=TRIPLES))

# Purge the PDF-dependent joblib caches (keyed without pdf_set) BEFORE importing
# stringy_gpds: importing the package runs __init__ -> core, whose module-level code
# builds gpd_errors and the moment interpolators. If a previous set's entries are still
# cached, they are (wrongly) reused for this set. Purging here, ahead of the import,
# forces a rebuild from this set's interpolation tables / parameters. Keep the
# PDF-independent gamma/harmonic interpolators. SGPD_SKIP_PURGE=1 reuses the cache
# (only safe for a same-set re-run).
CACHE = ROOT / "stringy_gpds" / "cache"
if os.environ.get("SGPD_SKIP_PURGE") != "1":
    for sub in (CACHE / "stringy_gpds" / "core" / "estimate_gpd_error",
                CACHE / "stringy_gpds" / "helpers" / "build_moment_interpolator"):
        shutil.rmtree(sub, ignore_errors=True)

try:
    import stringy_gpds.config as cfg
    assert cfg.PDF_SET == args.pdf_set and cfg.INTERPOLATE_MOMENTS

    t_imp = time.time()
    print(f"[{args.pdf_set}] importing core (builds gpd_errors over "
          f"{len(TRIPLES)} kinematics)...", flush=True)
    from stringy_gpds import core, helpers as hp
    print(f"[{args.pdf_set}] import complete in {time.time() - t_imp:.0f}s; "
          f"starting reconstruction", flush=True)

    for particle, mtype in COMBOS:
        store = {mu2: {} for mu2 in args.mu2}       # mu2 -> eta -> (xs, values)
        for eta, t, mu in TRIPLES:
            mu2 = round(mu ** 2)
            idx = TRIPLES.index((eta, t, mu))
            xs = x_grid(mtype, eta)
            t0 = time.time()
            c = np.array([float(core.mellin_barnes_gpd(x, eta, t, mu, 1, particle, mtype,
                                                       MOMENT_LABEL, args.order, "central", "real"))
                          for x in xs])
            values = {"central": c}
            if "plus" in args.errors:
                rel_p = core.gpd_errors[(particle, mtype, MOMENT_LABEL, args.order, "plus")][idx]
                values["plus"] = c * rel_p
            if "minus" in args.errors:
                rel_m = core.gpd_errors[(particle, mtype, MOMENT_LABEL, args.order, "minus")][idx]
                values["minus"] = c * rel_m
            for err in args.errors:
                hp.save_gpd_data(xs, eta, t, mu, values[err], particle, mtype, args.label,
                                 args.order, err)
                src = cfg.GPD_PATH / hp.generate_filename(
                    eta, t, mu, f"{mtype}_{particle}_GPD_{args.label}_{args.order}", err)
                shutil.copy(src, OUT_DATA / src.name)
            store[mu2][eta] = (xs, values)
            print(f"[{args.pdf_set}] {particle} {mtype} eta={eta} t={t} mu^2={mu2} "
                  f"({len(xs)} x-pts) {time.time() - t0:.1f}s", flush=True)

        # ---- overview plot: GPD(x) vs x, one panel per mu^2, curves coloured by eta ----
        try:
            symbol = gc.GPD_TEX_SYMBOL[args.label]
            fig, axes = plt.subplots(1, len(args.mu2), figsize=(6.4 * len(args.mu2), 4.8))
            cmap = plt.get_cmap("viridis")
            for ax, mu2 in zip(np.atleast_1d(axes), args.mu2):
                etas = sorted(store[mu2])
                for k, eta in enumerate(etas):
                    xs, values = store[mu2][eta]
                    col = cmap(k / max(1, len(etas) - 1))
                    ax.plot(xs, values["central"], "-", lw=1.3, color=col, label=rf"$\eta={eta:.2g}$")
                    if "plus" in values and "minus" in values:
                        ax.fill_between(xs, np.minimum(values["minus"], values["plus"]),
                                        np.maximum(values["minus"], values["plus"]),
                                        color=col, alpha=0.12)
                ax.axhline(0, color="0.7", lw=0.5)
                ax.set_xlabel(r"$x$")
                ax.set_ylabel(rf"${symbol}(x,\eta,t(\eta);\mu)$")
                ax.set_title(rf"$\mu^2={mu2:g}\;\mathrm{{GeV}}^2$")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8, frameon=False, ncol=2)
            fig.suptitle(rf"{args.pdf_set}: {particle} ${symbol}$ ({args.order.upper()})", y=1.02)
            pdf = OUT_PLOT / f"gpd_{args.label}_{mtype}_{particle}.pdf"
            fig.savefig(pdf, bbox_inches="tight")
            plt.close(fig)
            print(f"saved {pdf}", flush=True)
        except Exception as e:
            print(f"[{args.pdf_set}] PLOT FAILED for {particle} {mtype} (data saved): {e}", flush=True)
finally:
    gc.CONFIG_PATH.write_text(ORIG)
    print("restored user_config.py", flush=True)
