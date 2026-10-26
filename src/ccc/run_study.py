import argparse
import sys

from ccc import tracking


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ccc.run_study")
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 21)))
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--n-initial", type=int)
    ap.add_argument("--profiles", choices=("handwritten", "copula"), default="handwritten")
    ap.add_argument("--locale", default="en_US")
    ap.add_argument("--tracking-dir", default="mlruns")
    ap.add_argument("--tracking-uri")
    ap.add_argument("--min-recovery", type=float, default=tracking.DEFAULT_MIN_RECOVERY)
    ap.add_argument("--alpha", type=float, default=tracking.DEFAULT_ALPHA)
    ap.add_argument("--sdv-scores", action="store_true")
    args = ap.parse_args(argv)
    result = tracking.run_simulation_study(
        args.scenario,
        args.seeds,
        args.days,
        args.n_initial,
        profile_source=args.profiles,
        locale=args.locale,
        tracking_dir=args.tracking_dir,
        tracking_uri=args.tracking_uri,
        min_recovery=args.min_recovery,
        alpha=args.alpha,
        with_sdv_scores=args.sdv_scores,
    )
    print(tracking.format_summary(result["summary"]))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
