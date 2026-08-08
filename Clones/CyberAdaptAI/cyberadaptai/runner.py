"""Algorithm 1 -- CyberAdaptAI: Adaptive Ensemble Learning for Streaming Intrusion Detection.

    1. For each batch B_t:
    2.   Preprocess B_t: clean, encode, normalize
    3.   For each instance x_i in B_t:
    4.     For each h_m in H, compute y_hat_i^(m) = h_m(x_i)
    5.     Compute final prediction (Eq 2)
    6.     Store prediction
    7.   Evaluate y_hat_i against true y_i
    8.   Update error estimates E_m^t
    9.   Update weights (Eq 8)
   10.   If drift detected (ADWIN on error stream):
   11.     Reset or retrain underperforming classifiers
   12.   Partially fit h_m on B_t if online-capable
   13. Return predictions, updated weights

Steps 3-6 are vectorised over the batch rather than looped per instance; the result is
identical and the per-instance loop is only there in the paper to describe semantics.
"""

import json
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .base_learners import HoeffdingTreeLearner, XGBoostLearner, build_pool
from .config import Config, RetrainScope, ScalerFit
from .drift import DriftDetector
from .ensemble import AdaptEnsembleNet
from .explain import explain_xgboost, shap_available
from .metrics import aggregate, compute_metrics, confusion, per_class_report
from .preprocessing import StreamPreprocessor
from .stream import BatchStream


class CyberAdaptAI:
    """The full framework: preprocessing + AdaptEnsembleNet + drift handling + explainability."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.preprocessor = StreamPreprocessor(self.config)
        self.ensemble: Optional[AdaptEnsembleNet] = None
        self.drift_detector: Optional[DriftDetector] = None
        self.batch_log: List[Dict] = []
        self.drift_events: List[Dict] = []
        self.notes: List[str] = []

    def _log(self, msg: str) -> None:
        if self.config.verbose:
            print(msg, flush=True)

    # --- setup ------------------------------------------------------------------
    def _prepare(self, X: pd.DataFrame, y: np.ndarray):
        X = StreamPreprocessor.encode_categoricals(X)
        X, y_series, n_dupes = StreamPreprocessor.clean(
            X, pd.Series(y), drop_duplicates=self.config.drop_duplicates
        )
        y = y_series.to_numpy()
        self._log(f"  duplicates removed : {n_dupes:,} ({n_dupes / (len(y) + n_dupes):.1%})")
        self.notes.append(f"duplicates_removed={n_dupes}")
        return X, y

    def _fit_scaler(self, train_stream: BatchStream, full_X: pd.DataFrame) -> None:
        mode = self.config.scaler_fit
        if mode == ScalerFit.GLOBAL:
            # Reproduces the paper's most likely reading. Leaks future batch statistics.
            self.preprocessor.fit(full_X)
            self.notes.append("scaler=global (LEAKY - fitted on all batches incl. test)")
        elif mode == ScalerFit.FIRST_BATCH:
            first = next(iter(train_stream))
            self.preprocessor.fit(first.X)
            self.notes.append("scaler=first_batch (leakage-safe)")
        else:  # RUNNING
            first = next(iter(train_stream))
            self.preprocessor.fit(first.X)
            self.notes.append("scaler=running (leakage-safe, updates per batch)")

    # --- Algorithm 1 ------------------------------------------------------------
    def run(self, X: pd.DataFrame, y: np.ndarray) -> Dict:
        cfg = self.config
        self._log(cfg.describe())
        self._log("\n[1/4] Preparing data (section 3.2)")
        X, y = self._prepare(X, y)

        stream = BatchStream(X, y, batch_size=cfg.batch_size)
        train_stream, test_stream = stream.split_sequential_holdout(cfg.train_test_split)
        classes = stream.all_classes()
        self._log(
            f"  {len(X):,} rows -> {len(stream)} batches of {cfg.batch_size} "
            f"({len(train_stream)} train / {len(test_stream)} test), {len(classes)} classes"
        )

        self._fit_scaler(train_stream, X)

        # --- warm start: fit the pool on the training segment ---
        self._log("\n[2/4] Warm-starting the base learner pool (section 4.1)")
        pool = build_pool(cfg)
        ht = next((l for l in pool if isinstance(l, HoeffdingTreeLearner)), None)
        if ht is not None and ht.is_fallback:
            msg = "  NOTE: river not installed -> Hoeffding Tree replaced by SGD online learner"
            self._log(msg)
            self.notes.append("hoeffding_tree=SGD-fallback (river not installed)")

        self.ensemble = AdaptEnsembleNet(pool, cfg, classes)
        self.drift_detector = DriftDetector(
            delta=cfg.adwin_delta,
            max_window=cfg.adwin_max_window,
            accuracy_window=cfg.accuracy_window,
            accuracy_drop_threshold=cfg.accuracy_drop_threshold,
            accuracy_drop_batches=cfg.accuracy_drop_batches,
        )

        X_train_scaled = self.preprocessor.transform(train_stream.X)
        t0 = time.perf_counter()
        self.ensemble.fit(X_train_scaled, train_stream.y)
        self._log(f"  pool fitted on {len(train_stream.y):,} rows in {time.perf_counter() - t0:.1f}s")
        self._log(f"  members: {', '.join(self.ensemble.learner_names)}")

        # --- streaming evaluation over the test segment ---
        self._log(f"\n[3/4] Streaming {len(test_stream)} test batches (Algorithm 1)")
        all_true, all_pred = [], []

        # Section 4.1: on drift, "weaker classifiers are retrained with new data from the
        # current window" -- the window, not the single triggering batch. Retraining on one
        # batch also tends to drop rare classes entirely, which is what produced the
        # non-contiguous-label crash in XGBoost.
        window_X: deque = deque(maxlen=cfg.accuracy_window)
        window_y: deque = deque(maxlen=cfg.accuracy_window)

        for batch in test_stream:                                        # line 1
            Xb = self.preprocessor.transform(batch.X)                     # line 2
            window_X.append(Xb)
            window_y.append(batch.y)

            t_start = time.perf_counter()
            member_preds = self.ensemble.member_predictions(Xb)           # line 4
            y_pred = self.ensemble.vote(member_preds)                     # line 5
            latency = time.perf_counter() - t_start                       # Eq 16

            all_true.append(batch.y)                                      # line 6
            all_pred.append(y_pred)

            proba = self.ensemble.vote_scores(Xb)
            m = compute_metrics(batch.y, y_pred, proba, classes, latency_s=latency)  # line 7

            record = self.ensemble.update_weights(member_preds, batch.y)  # lines 8-9

            errors = (y_pred != batch.y).astype(float)
            drifted = self.drift_detector.update(errors, m["accuracy"])   # line 10

            retrained: List[str] = []
            if drifted:                                                   # line 11
                if cfg.retrain_scope == RetrainScope.CUMULATIVE:
                    Xw = pd.concat([X_train_scaled] + list(window_X), axis=0)
                    yw = np.concatenate([train_stream.y] + list(window_y))
                else:
                    Xw = pd.concat(list(window_X), axis=0) if len(window_X) > 1 else Xb
                    yw = np.concatenate(list(window_y)) if len(window_y) > 1 else batch.y
                retrained = self.ensemble.retrain_underperformers(
                    Xw, yw, accuracies=np.array(record["accuracies"])
                )
                self.drift_events.append({
                    "batch": batch.index,
                    "reason": self.drift_detector.last_reason,
                    "accuracy": m["accuracy"],
                    "retrained": retrained,
                })
                self._log(
                    f"  batch {batch.index:>3}  DRIFT ({self.drift_detector.last_reason}) "
                    f"acc={m['accuracy']:.4f}  retrained={retrained or 'none'}"
                )

            self.ensemble.partial_fit(Xb, batch.y)                        # line 12

            m.update({
                "batch": batch.index,
                "drift": bool(drifted),
                "drift_reason": self.drift_detector.last_reason,
                "retrained": retrained,
                "weights": record["weights"],
                "member_accuracies": record["accuracies"],
                "member_diversity": record["diversity"],
            })
            self.batch_log.append(m)

        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)

        # --- explainability ---
        self._log("\n[4/4] Explainability (section 3.5)")
        shap_out = None
        if cfg.enable_shap:
            xgb = next((l for l in self.ensemble.pool if isinstance(l, XGBoostLearner)), None)
            last_X = self.preprocessor.transform(list(test_stream)[-1].X)
            shap_out = explain_xgboost(xgb, last_X, cfg.shap_sample_size, cfg.seed)
            if shap_out:
                self._log(f"  backend: {shap_out['backend']}")
                self._log(f"  TreeSHAP on {shap_out['n_explained']} rows")
                self._log(f"  top features: {', '.join(shap_out['top_features'][:5])}")
                self._log(f"  Eq 9 additivity residual: {shap_out['additivity_residual']}")
                self.notes.append(f"shap_backend={shap_out['backend']}")
            else:
                self._log("  SKIPPED: no working TreeSHAP backend")
                self.notes.append("shap=SKIPPED")

        results = {
            "config": {
                "batch_size": cfg.batch_size,
                "weight_rule": cfg.weight_rule.value,
                "diversity_measure": cfg.diversity_measure.value,
                "alpha": cfg.alpha,
                "gamma": cfg.gamma,
                "adwin_delta": cfg.adwin_delta,
                "scaler_fit": cfg.scaler_fit.value,
                "retrain_scope": cfg.retrain_scope.value,
                "seed": cfg.seed,
            },
            "notes": self.notes,
            "members": self.ensemble.learner_names,
            "overall": compute_metrics(y_true, y_pred, classes=classes),
            "streaming_aggregate": aggregate(self.batch_log),
            "per_class": per_class_report(y_true, y_pred, classes),
            "confusion_matrix": confusion(y_true, y_pred, classes).tolist(),
            "classes": classes,
            "final_weights": dict(zip(self.ensemble.learner_names, self.ensemble.weights.tolist())),
            "weight_movement": self._weight_movement(),
            "n_drift_events": len(self.drift_events),
            "drift_events": self.drift_events,
            "shap": shap_out,
        }
        self._print_summary(results)
        return results

    # --- diagnostics ------------------------------------------------------------
    def _weight_movement(self) -> Dict:
        """How much did the adaptive weights actually move?

        This is the honest test of the paper's central claim. If the per-member standard
        deviation across batches is ~0 and the weights sit near 1/M, the "dynamic" ensemble is
        a uniform static ensemble in disguise -- the same collapse this thesis measured in
        two_stage_stageB_des.ipynb (weights 0.333 +/- 0.02).
        """
        if not self.batch_log:
            return {}
        W = np.array([b["weights"] for b in self.batch_log])
        names = self.ensemble.learner_names
        uniform = 1.0 / W.shape[1]
        return {
            "per_member_mean": dict(zip(names, W.mean(axis=0).round(4).tolist())),
            "per_member_std": dict(zip(names, W.std(axis=0).round(4).tolist())),
            "max_abs_deviation_from_uniform": float(np.abs(W - uniform).max()),
            "mean_abs_deviation_from_uniform": float(np.abs(W - uniform).mean()),
            "uniform_weight": uniform,
        }

    def _print_summary(self, r: Dict) -> None:
        o = r["overall"]
        wm = r["weight_movement"]
        self._log("\n" + "=" * 62)
        self._log("RESULTS (pooled over the test stream)")
        self._log("=" * 62)
        self._log(f"  Accuracy           {o['accuracy']:.4f}")
        if "f1_binary" in o:
            # Binary task -- this triple is what the paper's Table 4 reports.
            self._log(f"  Precision (attack) {o['precision_binary']:.4f}   <- comparable to Table 4")
            self._log(f"  Recall    (attack) {o['recall_binary']:.4f}   <- comparable to Table 4")
            self._log(f"  F1        (attack) {o['f1_binary']:.4f}   <- comparable to Table 4")
            self._log(f"  Macro F1           {o['f1_macro']:.4f}")
        else:
            self._log(f"  Macro F1           {o['f1_macro']:.4f}   <- the meaningful metric here")
            self._log(f"  Macro precision    {o['precision_macro']:.4f}")
            self._log(f"  Macro recall       {o['recall_macro']:.4f}")
            self._log(f"  Weighted F1        {o['f1_weighted']:.4f}")
        lat = r["streaming_aggregate"].get("latency_per_instance_ms")
        if lat is not None:
            self._log(f"  Latency/instance   {lat:.4f} ms   (Eq 16)")
        self._log(f"  Drift events       {r['n_drift_events']}")
        self._log("\n  Final adaptive weights:")
        for name, w in r["final_weights"].items():
            self._log(f"    {name:<32} {w:.4f}")
        if wm:
            self._log(
                f"\n  Weight drift from uniform ({wm['uniform_weight']:.4f}): "
                f"mean |dev| {wm['mean_abs_deviation_from_uniform']:.4f}, "
                f"max |dev| {wm['max_abs_deviation_from_uniform']:.4f}"
            )
            if wm["mean_abs_deviation_from_uniform"] < 0.02:
                self._log("  -> weights barely moved: this is a STATIC ensemble in disguise.")
        self._log("\n  Per-class F1:")
        for c, s in r["per_class"].items():
            self._log(f"    class {c}: F1 {s['f1']:.4f}  (P {s['precision']:.4f} "
                      f"R {s['recall']:.4f}  n={s['support']})")
        if r["notes"]:
            self._log("\n  Notes: " + "; ".join(r["notes"]))
        self._log("=" * 62)

    def save(self, results: Dict, out_dir: str) -> None:
        """Write results. Uses UTF-8 files -- the terminal here is cp1252."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        with open(out / "results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        pd.DataFrame([
            {k: v for k, v in b.items() if not isinstance(v, list)} for b in self.batch_log
        ]).to_csv(out / "batch_metrics.csv", index=False, encoding="utf-8")

        names = self.ensemble.learner_names
        pd.DataFrame(
            [b["weights"] for b in self.batch_log],
            columns=names,
        ).assign(batch=[b["batch"] for b in self.batch_log]).to_csv(
            out / "weights.csv", index=False, encoding="utf-8"
        )

        self._log(f"\nSaved to {out.resolve()}")
        self._log("  results.json | batch_metrics.csv | weights.csv")
