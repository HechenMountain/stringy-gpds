"""Moment interpolation tables (any label/order/channels) for one input PDF set, at
an arbitrary (eta, t, mu) grid. Required before generate_gpd_tables.py, which
reconstructs x-space GPDs from the interpolated (non-integer j) moments.

Usage:
    python scripts/generate_interp_tables.py --pdf-set JAM22 --label H \\
        --eta 0 0.1 0.2 --mu2 2 10 --t-min

Writes into ~/stringy-gpds/data/InterpolationTables/ via tabgen.generate_moment_table
(skips any table that already exists). The filenames are NOT tagged by PDF set, so
different sets must be generated sequentially with a cleanup in between (e.g. move
the previous set's <moment_type>_<particle>_moments_<label>_<order>_*.csv aside).
Restores user_config.py afterwards.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import _gpd_common as gc

args = gc.build_parser(__doc__).parse_args()

MOMENT_LABEL = gc.GPD_LABEL_MAP[args.label]
COMBOS = gc.resolve_combos(args.particles, args.moments)
TRIPLES = gc.resolve_triples(args)

ORIG = gc.CONFIG_PATH.read_text()
gc.CONFIG_PATH.write_text(gc.pdf_set_config_lines(
    args.pdf_set, MOMENT_LABEL, args.particles, args.moments, args.order, ["central"],
    interpolate=False, triples=TRIPLES))
try:
    import stringy_gpds.config as cfg
    assert cfg.PDF_SET == args.pdf_set, cfg.PDF_SET
    assert not cfg.INTERPOLATE_MOMENTS
    from stringy_gpds.tabgen import generate_moment_table
    import stringy_gpds.helpers as hp

    N = len(TRIPLES) * len(COMBOS)
    print(f"{args.pdf_set}: generating up to {N} interpolation tables "
          f"({len(COMBOS)} channels x {len(TRIPLES)} kinematics)", flush=True)
    i = 0
    for particle, mtype in COMBOS:
        for eta, t, mu in TRIPLES:
            i += 1
            prefix = cfg.INTERPOLATION_TABLE_PATH / f"{mtype}_{particle}_moments_{MOMENT_LABEL}_{args.order}"
            fname = hp.generate_filename(eta, t, mu, prefix, "central")
            if Path(fname).exists():
                print(f"[{i}/{N}] skip (exists): {Path(fname).name}", flush=True)
                continue
            t0 = time.time()
            try:
                generate_moment_table(eta, t, mu, ".", particle, mtype, MOMENT_LABEL,
                                      args.order, "central")
                print(f"[{i}/{N}] {particle} {mtype} eta={eta} t={t} mu={mu:.4f}  "
                      f"{time.time() - t0:.1f}s", flush=True)
            except Exception as e:
                print(f"[{i}/{N}] FAILED {particle} {mtype} eta={eta} t={t} mu={mu:.4f}: {e}", flush=True)
finally:
    gc.CONFIG_PATH.write_text(ORIG)
    print("restored user_config.py", flush=True)
