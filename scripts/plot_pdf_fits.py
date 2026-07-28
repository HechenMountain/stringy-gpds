"""Fit-quality plots (NLO only) of one input PDF set, on a linear and a log x-axis.

The fitted NLO curves are reconstructed from the parameter CSVs and compared with the
PDF data. Each PDF gets a main panel (x*f(x) with the data error bars) and a pull strip
below it that SHARES its x-axis; pull = (fit - data)/delta f is the normalized residual.

Labels and titles are rendered with LaTeX (text.usetex); the figures are written as PDF
to fit_output/. Run after fit_input_pdfs.py for the same set.

Usage:
    python scripts/plot_pdf_fits.py --pdf-set JAM22 --data PDFdata_JAM22.csv
"""
import argparse
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
OUT = ROOT / "fit_output"
import _gpd_common as gc

from stringy_gpds.fit import (load_pdf_data, read_pdf_parameter_csv, input_carriers,
                              _polarized_factor)

ORDER = "nlo"
KAPPA = 0.5  # strange completion for a data set without strange PDFs

# LaTeX rendering for all text; fall back to mathtext if the toolchain is unusable
USETEX = shutil.which("latex") is not None
plt.rcParams.update({
    "text.usetex": USETEX,
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
})


def unpolarized_curves(data, params):
    """{title: (data, error, NLO model)} for the unpolarized combinations present in data."""
    x = data["x"]
    c = input_carriers(params, x, ORDER)
    curves = {
        r"$x\,u_v(x)$": (data["uv"], data["uv_err"], c["uv"]),
        r"$x\,d_v(x)$": (data["dv"], data["dv_err"], c["dv"]),
    }
    if data["has_strange"]:
        S, S_err = data["S"], data["S_err"]
    else:
        S = (2 + KAPPA) * (data["ubar"] + data["dbar"])
        S_err = (2 + KAPPA) * np.hypot(data["ubar_err"], data["dbar_err"])
    curves[r"$x\,S(x)$"] = (S, S_err, c["S"])
    curves[r"$x\,\Delta(x)=x(\bar{d}-\bar{u})(x)$"] = (data["Delta"], data["Delta_err"], c["Delta"])
    curves[r"$x\,g(x)$"] = (data["g"], data["g_err"], c["g"])
    if data["has_strange"]:
        curves[r"$x\,(s+\bar{s})(x)$"] = (data["s_plus"], data["s_plus_err"], c["s_plus"])
        curves[r"$x\,(s-\bar{s})(x)$"] = (data["sv"], data["sv_err"], c["sv"])
    return curves


def polarized_curves(data, params, pol_params):
    """{title: (data, error, NLO model)} for the polarized combinations present in data."""
    x = data["x"]
    c = input_carriers(params, x, ORDER)

    def factor(row):
        p = [pol_params[f"{name}_{row}"][ORDER][0]
             for name in ("Delta_A", "alpha", "Delta_gamma", "Delta_lambda")]
        return _polarized_factor(x, *p, ORDER)

    curves = {
        r"$x\,\Delta u_v(x)$": (data["Duv"], data["Duv_err"], factor("u") * c["uv"]),
        r"$x\,\Delta d_v(x)$": (data["Ddv"], data["Ddv_err"], factor("d") * c["dv"]),
        r"$x\,\Delta\bar{u}(x)$": (data["Dubar"], data["Dubar_err"], factor("ubar") * c["ubar"]),
        r"$x\,\Delta g(x)$": (data["Dg"], data["Dg_err"], factor("g") * c["g"]),
        r"$x\,\Delta\bar{d}(x)$": (data["Ddbar"], data["Ddbar_err"], factor("dbar") * c["dbar"]),
    }
    if data["has_strange"]:
        curves[r"$x\,\Delta s(x)$"] = (data["Ds"], data["Ds_err"], factor("s") * c["s"])
        curves[r"$x\,\Delta\bar{s}(x)$"] = (data["Dsbar"], data["Dsbar_err"], factor("sbar") * c["sbar"])
        curves[r"$x\,\Delta(s+\bar{s})(x)$"] = (data["Ds_plus"], data["Ds_plus_err"],
                                                factor("s_plus") * c["s_plus"])
        DS, DS_err = data["DS"], data["DS_err"]
    else:
        DS = (2 + KAPPA) * (data["Dubar"] + data["Ddbar"])
        DS_err = (2 + KAPPA) * np.hypot(data["Dubar_err"], data["Ddbar_err"])
    curves[r"$x\,\Delta S(x)$"] = (DS, DS_err, factor("S") * c["S"])
    return curves


def plot(x, curves, title, xscale, path):
    n = len(curves)
    fig = plt.figure(figsize=(2.9 * n, 5.6))
    gs = fig.add_gridspec(2, n, height_ratios=[3, 1], hspace=0.0, wspace=0.34)
    for i, (name, (y, ye, model)) in enumerate(curves.items()):
        # main panel and its pull strip share the x-axis (one per PDF)
        ax = fig.add_subplot(gs[0, i])
        ax_pull = fig.add_subplot(gs[1, i], sharex=ax)
        plt.setp(ax.get_xticklabels(), visible=False)

        ax.errorbar(x, x * y, yerr=x * ye, fmt=".", ms=3, color="#1f77b4", label="data", zorder=2)
        ax.plot(x, x * model, "-", lw=1.6, color="#d62728", label="fit", zorder=3)
        ax.axhline(0, color="0.7", lw=0.5)
        ax.set_title(name)
        ax.set_xscale(xscale)
        if i == 0:
            ax.legend(fontsize=9, frameon=False, loc="best")
            ax.set_ylabel(r"$x\,f(x)$")

        pull = (model - y) / ye
        span = max(3.0, 1.1 * float(np.abs(pull).max()))
        ax_pull.axhspan(-1, 1, color="0.9", zorder=0)
        ax_pull.axhline(0, color="k", lw=0.5)
        ax_pull.plot(x, pull, ".", ms=3, color="#1f77b4")
        ax_pull.set_xscale(xscale)
        ax_pull.set_xlabel(r"$x$")
        ax_pull.set_ylim(-span, span)
        if i == 0:
            ax_pull.set_ylabel("pull")
        if xscale == "linear":
            ax.set_xlim(0, x.max())

    fig.suptitle(title, y=0.99)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def make_set(name, data_path, unpol_csv, pol_csv):
    data = load_pdf_data(data_path)
    params = read_pdf_parameter_csv(unpol_csv)
    pol_params = read_pdf_parameter_csv(pol_csv)
    x = data["x"]
    unpol = unpolarized_curves(data, params)
    pol = polarized_curves(data, params, pol_params)
    for xscale in ("linear", "log"):
        plot(x, unpol, rf"{name} unpolarized input PDFs (NLO)",
             xscale, OUT / f"fit_{name}_unpolarized_{xscale}.pdf")
        plot(x, pol, rf"{name} polarized input factors (NLO)",
             xscale, OUT / f"fit_{name}_polarized_{xscale}.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf-set", required=True, help="name of the fitted set, e.g. JAM22")
    parser.add_argument("--data", required=True,
        help="input PDF data CSV, relative to the repo root or an absolute path")
    parser.add_argument("--unpol-csv", default=None,
        help="fitted unpolarized parameter CSV (default: the live pdfs/<pdf-set>.csv)")
    parser.add_argument("--pol-csv", default=None,
        help="fitted polarized parameter CSV (default: the live pdfs/<pdf-set>_POL.csv)")
    args = parser.parse_args()

    if not USETEX:
        print("warning: latex not found, falling back to mathtext")
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    unpol_csv = Path(args.unpol_csv) if args.unpol_csv else gc.USER_PATH / "pdfs" / f"{args.pdf_set}.csv"
    pol_csv = Path(args.pol_csv) if args.pol_csv else gc.USER_PATH / "pdfs" / f"{args.pdf_set}_POL.csv"
    make_set(args.pdf_set, data_path, unpol_csv, pol_csv)


if __name__ == "__main__":
    main()
