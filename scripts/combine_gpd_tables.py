"""Combine the per-eta x-space GPD CSVs (data/gpds/<SET>/) into one table per
(particle, moment_type, mu^2) with every requested eta side by side:

    "x","H(x,0,0.00)","H(x,0,0.00)_plus","H(x,0,0.00)_minus",
        "H(x,0.1,-0.04)","H(x,0.1,-0.04)_plus","H(x,0.1,-0.04)_minus", ...

t = t_min(eta) or the literal --t value is filled in per eta column group; the
symbol is --label (e.g. Htilde(x,...)). 

Pure reformat of already-generated data:
requires generate_gpd_tables.py to have been run for the same --pdf-set/--label/
--order/--particles/--moments/--eta/--mu2/--t(-min)/--errors.

Usage:
    python scripts/combine_gpd_tables.py --pdf-set JAM22 --label H \\
        --eta 0 0.1 0.2 --mu2 2 10 --t-min

Writes data/gpds/<SET>/combined/<moment_type>_<particle>_GPD_<label>_<order>_musq_<mu2>.csv
"""
import csv
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import _gpd_common as gc

args = gc.build_parser(__doc__).parse_args()

COMBOS = gc.resolve_combos(args.particles, args.moments)
T_OF = gc.resolve_t(args)
ROUND = 10   # decimals for matching x across per-eta files (grids are bit-identical
             # for shared points since they come from the same np.linspace call)
SUFFIX = {"central": "", "plus": "_plus", "minus": "_minus"}


def gpd_path(set_dir, mtype, particle, eta, mu, error):
    fn = (f"{mtype}_{particle}_GPD_{args.label}_{args.order}_"
          f"{gc.code(eta)}_{gc.code(T_OF[eta])}_{gc.code(mu)}{SUFFIX[error]}.csv")
    return set_dir / fn


def load(path):
    """{rounded_x: value}, or {} if the file is missing."""
    if not path.exists():
        return {}
    a = np.atleast_2d(np.loadtxt(path, delimiter=","))
    return {round(float(x), ROUND): float(v) for x, v in a}


set_dir = ROOT / "data" / "gpds" / args.pdf_set
out_dir = set_dir / "combined"
out_dir.mkdir(parents=True, exist_ok=True)

for particle, mtype in COMBOS:
    for mu2 in args.mu2:
        mu = math.sqrt(mu2)
        # per_eta[eta][error] = {x: value}
        per_eta = {eta: {err: load(gpd_path(set_dir, mtype, particle, eta, mu, err))
                          for err in args.errors}
                   for eta in args.eta}

        all_x = sorted({x for eta in args.eta for x in per_eta[eta][args.errors[0]]})

        # "H(x,xi,t)" contains literal commas once xi/t are filled in, so each header
        # cell needs CSV quoting; csv.writer's default QUOTE_MINIMAL dialect does
        # this automatically.
        header = ["x"]
        for eta in args.eta:
            label = f"{args.label}(x,{eta:g},{T_OF[eta]:.2f})"
            header += [f"{label}{SUFFIX[err]}" for err in args.errors]

        out = out_dir / f"{mtype}_{particle}_GPD_{args.label}_{args.order}_musq_{mu2:02.0f}.csv"
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            for x in all_x:
                row = [f"{x:.18e}"]
                for eta in args.eta:
                    for err in args.errors:
                        v = per_eta[eta][err].get(x)
                        row.append(f"{v:.18e}" if v is not None else "")
                w.writerow(row)

        n_cells = len(all_x) * len(args.eta)
        n_blank = sum(1 for x in all_x for eta in args.eta
                      if x not in per_eta[eta][args.errors[0]])
        print(f"[{args.pdf_set}] {mtype:22s} {particle:6s} mu^2={mu2:g}  "
              f"{len(all_x)} x-rows, {n_blank}/{n_cells} blank eta-cells  -> {out}",
              flush=True)
