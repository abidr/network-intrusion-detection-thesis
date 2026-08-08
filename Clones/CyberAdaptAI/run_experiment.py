"""CLI entry point for the CyberAdaptAI reimplementation.

    python run_experiment.py --dataset cic-unsw-2024 --batch-size 2000 --out results/

Ablations over the paper's ambiguities:

    --weight-rule eq8            Algorithm 1 verbatim (no diversity term at all)
    --weight-rule eq3            Eq 3 verbatim (no temporal smoothing)
    --weight-rule hybrid         default: Eq-3 target smoothed by Eq-8 gamma
    --diversity q_statistic      Eq 4 verbatim (wrong sign -- rewards redundancy)
    --scaler global              the paper's likely reading of Eq 1 (leaks the future)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapters import cic_unsw_2024, unsw_nb15
from cyberadaptai import Config, CyberAdaptAI
from cyberadaptai.config import DiversityMeasure, RetrainScope, ScalerFit, WeightRule


def main() -> int:
    p = argparse.ArgumentParser(
        description="CyberAdaptAI / AdaptEnsembleNet reimplementation (NOT the authors' code)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dataset", default="cic-unsw-2024",
        choices=["cic-unsw-2024", "unsw-nb15"],
        help="unsw-nb15 = the ORIGINAL dataset the paper uses (Table 4)",
    )
    p.add_argument(
        "--task", default=None, choices=["binary", "multiclass"],
        help="unsw-nb15 only. Default binary -- the paper's Table 4 reports a single P/R/F1 "
             "triple, which is the attack-class score, so binary is the comparable setting.",
    )
    p.add_argument("--data-root", default=None, help="directory holding the dataset CSVs")
    p.add_argument("--form", default=None, choices=["presplit", "full"],
                   help="unsw-nb15 distribution form; auto-detected when omitted")
    p.add_argument("--subsample", type=int, default=None, help="use only the first N rows")

    p.add_argument("--batch-size", type=int, default=2000, help="paper: 1500-2000")
    p.add_argument("--alpha", type=float, default=0.6, help="Eq 3 accuracy/diversity trade-off")
    p.add_argument("--gamma", type=float, default=0.8, help="Eq 8 EWMA smoothing")
    p.add_argument("--adwin-delta", type=float, default=0.002, help="Eq 7 confidence")

    p.add_argument("--weight-rule", default="hybrid", choices=[r.value for r in WeightRule])
    p.add_argument("--diversity", default="disagreement", choices=[d.value for d in DiversityMeasure])
    p.add_argument("--scaler", default="first_batch", choices=[s.value for s in ScalerFit])
    p.add_argument("--retrain-scope", default="window", choices=[r.value for r in RetrainScope],
                   help="data used by a drift retrain: 'window' (section 4.1 literal) or "
                        "'cumulative' (keep the original training data too)")

    p.add_argument("--no-dedup", action="store_true", help="skip duplicate removal (leakage!)")
    p.add_argument(
        "--shuffle", action="store_true",
        help="shuffle rows before streaming. DIAGNOSTIC ONLY: destroys any temporal ordering, "
             "so the 'stream' becomes IID and concept drift is meaningless by construction. "
             "Use to test whether a reported score depends on IID splitting.",
    )
    p.add_argument("--train-file-only", action="store_true",
                   help="unsw-nb15 presplit: use UNSW_NB15_training-set.csv alone instead of "
                        "concatenating both official files")
    p.add_argument("--keep-id", action="store_true",
                   help="unsw-nb15: retain the `id` column. LEAKAGE -- id alone predicts the "
                        "label at ~0.79 accuracy. Diagnostic only.")
    p.add_argument("--no-shap", action="store_true")
    p.add_argument("--no-hoeffding", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="directory for results.json / CSVs")
    p.add_argument("--quiet", action="store_true")

    args = p.parse_args()

    config = Config(
        batch_size=args.batch_size,
        alpha=args.alpha,
        gamma=args.gamma,
        adwin_delta=args.adwin_delta,
        weight_rule=WeightRule(args.weight_rule),
        diversity_measure=DiversityMeasure(args.diversity),
        scaler_fit=ScalerFit(args.scaler),
        retrain_scope=RetrainScope(args.retrain_scope),
        drop_duplicates=not args.no_dedup,
        enable_shap=not args.no_shap,
        use_hoeffding_tree=not args.no_hoeffding,
        seed=args.seed,
        verbose=not args.quiet,
    )

    print("=" * 62)
    print("CyberAdaptAI (reimplementation from paper text -- NOT author code)")
    print("Uddamari & Sammulal 2025, IJECE 12(9) 11-31")
    print("=" * 62)

    task = args.task or ("binary" if args.dataset == "unsw-nb15" else "multiclass")

    try:
        if args.dataset == "unsw-nb15":
            X, y = unsw_nb15.load(
                root=args.data_root, task=task, form=args.form,
                subsample=args.subsample, seed=args.seed, keep_id=args.keep_id,
                combine_presplit=not args.train_file_only,
            )
            summary = unsw_nb15.class_summary(y, task=task)
            print(f"\nDataset: ORIGINAL UNSW-NB15 (the paper's dataset), task={task}")
        else:
            if args.task == "binary":
                print("\nERROR: --task binary is only implemented for --dataset unsw-nb15")
                return 1
            X, y = cic_unsw_2024.load(
                root=args.data_root, subsample=args.subsample, seed=args.seed
            )
            summary = cic_unsw_2024.class_summary(y)
            print("\nDataset: CIC-UNSW-NB15-2024 (NOT the paper's dataset), task=multiclass")
    except (FileNotFoundError, ValueError) as e:
        print(f"\nERROR: {e}")
        return 1

    print(f"Loaded {len(X):,} rows x {X.shape[1]} features")
    print(summary.to_string(index=False))

    if args.shuffle:
        import numpy as np

        order = np.random.default_rng(args.seed).permutation(len(X))
        X, y = X.iloc[order].reset_index(drop=True), y[order]
        print("\nWARNING: rows shuffled (--shuffle). The stream is now IID; any drift the "
              "detector reports is an artifact. Diagnostic use only.")
    print()

    framework = CyberAdaptAI(config)
    results = framework.run(X, y)

    if args.out:
        framework.save(results, args.out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
