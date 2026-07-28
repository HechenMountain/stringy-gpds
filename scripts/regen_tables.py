"""Regenerate the moment interpolation tables for the currently live input PDF set,
over the standard kinematics already configured in user_config.py (eta/t/mu_array).

- Quarantines the moment tables of the set being replaced into
  InterpolationTables/stale_<previous pdf_set>/ (the anomalous-dimension gamma_*
  and harmonic tables are PDF-independent and stay).
- Temporarily sets interpolate_moments = False, regenerates
  the requested channels, then restores the flag.

Run AFTER user_config.py has been updated with the new PDF set and the
corresponding slopes/normalizations (see refit_slopes.py).

Usage:
    python scripts/regen_tables.py
    python scripts/regen_tables.py --particles quark --moments singlet --labels A
"""
import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import _gpd_common as gc

parser = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--particles", nargs="+", default=list(gc.PARTICLES), choices=gc.PARTICLES)
parser.add_argument("--moments", nargs="+", default=list(gc.MOMENT_TYPES), choices=gc.MOMENT_TYPES)
parser.add_argument("--labels", nargs="+", default=["A", "Atilde", "B"], choices=["A", "Atilde", "B"])
parser.add_argument("--orders", nargs="+", default=["nlo"], choices=gc.ORDERS)
parser.add_argument("--stale-dir", default=None,
    help="quarantine directory name (default: stale_<pdf_set being replaced>)")
args = parser.parse_args()

TABLES = gc.USER_PATH / "data" / "InterpolationTables"
STALE_DIR = args.stale_dir or f"stale_{gc.current_pdf_set()}"

# ---- 1. toggle interpolate_moments off ----
uc_text = gc.CONFIG_PATH.read_text()
if "interpolate_moments = True" in uc_text:
    gc.CONFIG_PATH.write_text(uc_text.replace("interpolate_moments = True", "interpolate_moments = False"))

try:
    # ---- 2. quarantine the moment tables of the previous set ----
    stale = TABLES / STALE_DIR
    stale.mkdir(exist_ok=True)
    moved = 0
    for f in TABLES.glob("*_moments_*.csv"):
        shutil.move(str(f), str(stale / f.name))
        moved += 1
    print(f"quarantined {moved} stale moment tables -> {stale}")

    # ---- 3. regenerate ----
    import stringy_gpds.config as cfg
    assert not cfg.INTERPOLATE_MOMENTS
    from stringy_gpds.tabgen import generate_moment_table
    import stringy_gpds.helpers as hp

    triples = list(zip(cfg.ETA_ARRAY, cfg.T_ARRAY, cfg.MU_ARRAY))
    combos = gc.resolve_combos(args.particles, args.moments)

    n_total = len(triples) * len(combos) * len(args.labels) * len(args.orders)
    i = 0
    for order in args.orders:
        for particle, mtype in combos:
            for label in args.labels:
                for eta, t, mu in triples:
                    i += 1
                    prefix = TABLES / f"{mtype}_{particle}_moments_{label}_{order}"
                    fname = hp.generate_filename(eta, t, mu, prefix, "central")
                    if Path(fname).exists():
                        print(f"[{i}/{n_total}] skip (exists): {Path(fname).name}", flush=True)
                        continue
                    print(f"[{i}/{n_total}] {particle} {mtype} {label} {order} eta={eta} t={t} mu={mu}", flush=True)
                    try:
                        generate_moment_table(eta, t, mu, ".", particle, mtype, label,
                                              order, "central")
                    except Exception as e:
                        print(f"  FAILED: {e}", flush=True)
finally:
    # ---- 4. restore flag ----
    uc_text = gc.CONFIG_PATH.read_text()
    if "interpolate_moments = False" in uc_text:
        gc.CONFIG_PATH.write_text(uc_text.replace("interpolate_moments = False", "interpolate_moments = True"))
    print("restored interpolate_moments = True")
