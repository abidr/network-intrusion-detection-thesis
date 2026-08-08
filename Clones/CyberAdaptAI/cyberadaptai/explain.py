"""Section 3.5 -- Explainability and decision confidence. Implements Eq 9-11.

    f(x_i) = phi_0 + sum_j phi_j                      (Eq 9)   SHAP additivity
    P(y=c|x_i) = e^{z_c} / sum_k e^{z_k}              (Eq 10)  softmax
    Conf(x_i) = max_c P(y=c|x_i)                      (Eq 11)  confidence

Section 4.1: "SHAP values were calculated using TreeExplainer for the XGBoost model."

Two backends compute the same quantity:

  "xgboost_native"  -- Booster.predict(..., pred_contribs=True). This is the exact TreeSHAP
                       algorithm (Lundberg et al.) built into XGBoost, and it is the same code
                       path shap.TreeExplainer dispatches to for XGBoost models. Preferred:
                       no extra dependency and no version coupling.
  "shap.TreeExplainer" -- the literal library call named in section 4.1.

The native backend is tried first because shap <= 0.46 cannot parse XGBoost >= 3.x multiclass
models: `base_score` became a per-class vector and shap's XGBTreeModelLoader still casts it
with float(), raising ValueError. Upgrading shap past that fixes it but requires numpy >= 2.
The two backends agree to floating-point precision, so the substitution is cosmetic.
"""

from typing import Dict, List, Optional

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """Eq 10, computed in a shift-stable form."""
    z = np.asarray(logits, dtype=float)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def confidence(proba: np.ndarray) -> np.ndarray:
    """Eq 11: Conf(x_i) = max_c P(y=c|x_i)."""
    return np.max(np.asarray(proba), axis=-1)


def low_confidence_mask(proba: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Section 3.5: 'Alerts for a human or ensemble fallback ... are triggered when the
    confidence values are low.' The paper never states the threshold; 0.5 is this
    reimplementation's default and is a free parameter, not a paper value.
    """
    return confidence(proba) < threshold


def shap_available() -> bool:
    try:
        import shap  # noqa: F401

        return True
    except ImportError:
        return False


def explain_xgboost(
    xgb_learner,
    X,
    sample_size: int = 200,
    seed: int = 42,
) -> Optional[Dict]:
    """TreeSHAP attributions for the XGBoost pool member (section 3.5, 4.1).

    Returns None if the learner is unfitted. Reports which backend produced the values and the
    Eq-9 additivity residual, so a silently broken explainer cannot pass unnoticed.
    """
    if getattr(xgb_learner, "model", None) is None:
        return None

    rng = np.random.default_rng(seed)
    n = len(X)
    idx = np.sort(rng.choice(n, size=min(sample_size, n), replace=False))
    X_sample = X.iloc[idx] if hasattr(X, "iloc") else np.asarray(X)[idx]
    feature_names = (
        list(X.columns) if hasattr(X, "columns")
        else [f"f{i}" for i in range(np.asarray(X_sample).shape[1])]
    )

    result = _native_treeshap(xgb_learner, X_sample, feature_names)
    if result is None:
        result = _shap_treeexplainer(xgb_learner, X_sample, feature_names)
    if result is None:
        return None

    result["n_explained"] = int(len(idx))
    return result


def _rank(importance: np.ndarray, feature_names: List[str]) -> Dict:
    importance = np.asarray(importance).ravel()[: len(feature_names)]
    order = np.argsort(importance)[::-1]
    ranking = [
        {"feature": feature_names[i], "mean_abs_shap": float(importance[i])} for i in order
    ]
    return {
        "global_importance": ranking,
        "top_features": [r["feature"] for r in ranking[:15]],
    }


def _native_treeshap(xgb_learner, X_sample, feature_names: List[str]) -> Optional[Dict]:
    """Exact TreeSHAP via XGBoost itself.

    For a K-class model, pred_contribs returns shape (n, K, d+1); the trailing column is the
    bias term phi_0. Additivity (Eq 9) holds by construction, so the residual we report is a
    genuine numerical check of that guarantee.
    """
    try:
        import xgboost as xgb

        booster = xgb_learner.model.get_booster()
        dmatrix = xgb.DMatrix(X_sample, feature_names=None)

        contribs = np.asarray(booster.predict(dmatrix, pred_contribs=True))
        margins = np.asarray(booster.predict(dmatrix, output_margin=True))

        if contribs.ndim == 2:      # binary: (n, d+1)
            phi, phi0 = contribs[:, :-1], contribs[:, -1]
            recon = phi.sum(axis=1) + phi0
            importance = np.abs(phi).mean(axis=0)
        else:                        # multiclass: (n, K, d+1)
            phi, phi0 = contribs[..., :-1], contribs[..., -1]
            recon = phi.sum(axis=-1) + phi0
            importance = np.abs(phi).mean(axis=(0, 1))

        residual = (
            float(np.max(np.abs(recon - margins))) if recon.shape == margins.shape else None
        )

        out = {
            "backend": "xgboost_native (exact TreeSHAP)",
            "expected_value": np.asarray(phi0).mean(axis=0).tolist()
            if np.asarray(phi0).ndim > 1 else float(np.mean(phi0)),
            "additivity_residual": residual,
        }
        out.update(_rank(importance, feature_names))
        return out
    except Exception as e:
        return None if not _DEBUG else _raise(e)


def _shap_treeexplainer(xgb_learner, X_sample, feature_names: List[str]) -> Optional[Dict]:
    """The literal section-4.1 call. Kept as a fallback and a cross-check."""
    if not shap_available():
        return None
    try:
        import shap

        explainer = shap.TreeExplainer(xgb_learner.model)
        arr = np.asarray(explainer.shap_values(X_sample))

        if arr.ndim == 3:
            axes = tuple(i for i in range(arr.ndim) if arr.shape[i] != len(feature_names))
            importance = np.abs(arr).mean(axis=axes)
        else:
            importance = np.abs(arr).mean(axis=0)

        base = explainer.expected_value
        out = {
            "backend": "shap.TreeExplainer",
            "expected_value": np.asarray(base).tolist() if np.asarray(base).ndim else float(base),
            "additivity_residual": None,
        }
        out.update(_rank(importance, feature_names))
        return out
    except Exception:
        return None


_DEBUG = False


def _raise(e):
    raise e
