"""Fit an input PDF data CSV (unpolarized + polarized) to the stringy_gpds input
parametrization and activate the result as the live pdf_set.

Runs stringy_gpds.fit.fit_input_pdfs / fit_polarized_input_pdfs on --data and writes
pdfs/<pdf-set>.csv + pdfs/<pdf-set>_POL.csv into the live ~/stringy-gpds/pdfs.

The fit itself is run with --base-pdf-set active (default: whatever pdf_set is currently live)
alpha_S(Q0^2) is inherited from it (same --mu-input scale). 
Afterwards the live user_config.py is switched to pdf_set = <pdf-set>.

Usage:
    python scripts/fit_input_pdfs.py --pdf-set JAM22 --data PDFdata_JAM22.csv
"""
import argparse
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import _gpd_common as gc

OUT = ROOT / "fit_output"

parser = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--pdf-set", required=True, help="name of the set being fit, e.g. JAM22")
parser.add_argument("--data", required=True,
    help="input PDF data CSV, relative to the repo root or an absolute path")
parser.add_argument("--base-pdf-set", default=None,
    help="pdf_set to have active while fitting, so alpha_S(Q0^2) is inherited from it "
         "(default: whatever pdf_set is currently live in user_config.py)")
parser.add_argument("--mu-input", type=float, default=2, help="input scale [GeV]")
args = parser.parse_args()

DATA = Path(args.data)
if not DATA.is_absolute():
    DATA = ROOT / DATA


def save_open_figures(prefix):
    for i, num in enumerate(plt.get_fignums()):
        path = OUT / f"{prefix}_{i}.png"
        plt.figure(num).savefig(path, dpi=140)
        print("saved", path)
    plt.close("all")


def main():
    base = args.base_pdf_set or gc.current_pdf_set()
    if base is None:
        raise SystemExit("no pdf_set is currently configured in user_config.py; "
                          "pass --base-pdf-set explicitly")
    gc.set_pdf_set(base, args.mu_input)

    from stringy_gpds import fit
    import stringy_gpds.config as cfg
    print(f"active input set: {cfg.MSTW_PATH.name} / {cfg.AAC_PATH.name}, mu0 = {cfg.MU_INPUT} GeV")

    print("\n=== unpolarized ===")
    fit.fit_input_pdfs(DATA, pdf_set=args.pdf_set)
    save_open_figures(f"fit_{args.pdf_set}_unpolarized")

    print("\n=== polarized ===")
    fit.fit_polarized_input_pdfs(DATA, pdf_set=args.pdf_set)
    save_open_figures(f"fit_{args.pdf_set}_polarized")

    for name in (f"{args.pdf_set}.csv", f"{args.pdf_set}_POL.csv"):
        shutil.copy(cfg.PDF_PATH / name, ROOT / "cfg" / "pdfs" / name)
        shutil.copy(cfg.PDF_PATH / name, OUT / name)
    print("mirrored", args.pdf_set, "csv files to cfg/pdfs and fit_output")

    gc.set_pdf_set(args.pdf_set, args.mu_input)


if __name__ == "__main__":
    main()
