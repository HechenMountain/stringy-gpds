"""Run the whole H-GPD table pipeline -- moments -> interpolation tables -> x-space
GPDs -> combined tables -- for one input PDF set and one kinematic grid, using the
same flags as the individual stage scripts (each stage also runs standalone).

Usage:
    python scripts/run_gpd_pipeline.py --pdf-set JAM22 --label H \\
        --particles quark gluon --moments singlet non_singlet_isovector \\
        --eta 0 0.1 0.2 --mu2 2 10 --t-min

Use --stages to run only part of the pipeline, e.g. to checkpoint after the cheap
moment tables before committing to the heavy GPD reconstruction:

    python scripts/run_gpd_pipeline.py --stages moments ...
    python scripts/run_gpd_pipeline.py --stages interp gpds combine ...

Each stage runs as its own subprocess so user_config.py is safely swapped.
"""
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import _gpd_common as gc

STAGES = ["moments", "interp", "gpds", "combine"]
STAGE_SCRIPT = {
    "moments": "generate_moment_tables.py",
    "interp": "generate_interp_tables.py",
    "gpds": "generate_gpd_tables.py",
    "combine": "combine_gpd_tables.py",
}
# common flags forwarded to every stage; stage-specific flags forwarded only to the
# stage that understands them
COMMON_FLAGS = ["pdf_set", "label", "particles", "moments", "order", "errors",
                "eta", "mu2", "t_min", "t"]
STAGE_FLAGS = {
    "moments": ["j_max", "j_min_singlet", "j_min_nonsinglet"],
    "interp": [],
    "gpds": ["n_points_singlet", "n_points_nonsinglet"],
    "combine": [],
}


def flag_args(args, names):
    """Re-serialize the requested argparse attributes as CLI flags."""
    out = []
    for name in names:
        value = getattr(args, name)
        if value is None:
            continue
        flag = "--" + name.replace("_", "-")
        if isinstance(value, bool):
            if value:
                out.append(flag)
        elif isinstance(value, list):
            out += [flag] + [str(v) for v in value]
        else:
            out += [flag, str(value)]
    return out


def main():
    parser = gc.build_parser(__doc__)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=STAGES,
        help="pipeline stages to run, in order (default: all four)")
    gc.add_moment_args(parser)
    gc.add_gpd_args(parser)
    args = parser.parse_args()

    for stage in [s for s in STAGES if s in args.stages]:
        script = SCRIPTS / STAGE_SCRIPT[stage]
        argv = flag_args(args, COMMON_FLAGS + STAGE_FLAGS[stage])
        print(f"\n===== stage: {stage} ({script.name}) =====", flush=True)
        subprocess.run([sys.executable, str(script), *argv], check=True)


if __name__ == "__main__":
    main()
