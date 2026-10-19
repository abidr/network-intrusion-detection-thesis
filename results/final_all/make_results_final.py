"""Collect results/final_all/*.json into results_final.md (two tables, nothing else)."""
import json
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
ORDER = [("full", "CIC-UNSW-NB15-2024 (full release)"),
         ("curated", "CIC-UNSW-NB15-2024 (curated)"),
         ("unsw", "UNSW-NB15"),
         ("cicids2017", "CIC-IDS2017"),
         ("toniot", "TON-IoT")]
R = {k: json.loads((HERE / (k + ".json")).read_text(encoding="utf-8"))
     for k, _ in ORDER if (HERE / (k + ".json")).exists()}

L = ["# Results - final model (`final_model_v7_1_xai.ipynb`)", "",
     "| Dataset | Accuracy | Macro-P | Macro-R | Macro-F1 | Wtd-P | Wtd-R | Wtd-F1 | ROC-AUC | SELV winner |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for k, nm in ORDER:
    if k not in R: continue
    m = R[k]["metrics"]
    L.append("| %s | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %.4f | %s @ t=%.1f |"
             % (nm, m["accuracy"], m["macro_P"], m["macro_R"], m["macro_F1"], m["wtd_P"],
                m["wtd_R"], m["wtd_F1"], m["roc_auc"], R[k]["selv_winner"], R[k]["threshold"]))

L += ["", "## XAI verification", "",
      "| Dataset | Additivity max residual | shap vs xgboost max diff | Decisions reproduced | Gate top SHAP group | Stage-B top SHAP group |",
      "|---|---|---|---|---|---|"]
for k, nm in ORDER:
    if k not in R: continue
    x = R[k]["xai"]
    L.append("| %s | %.1e | %.1e | %.2f%% | %s (%.1f%%) | %s (%.1f%%) |"
             % (nm, x["additivity_max"], x["shap_vs_xgboost_max"], 100 * x["reproduction"],
                x["gate_top"][0], 100 * x["gate_top"][1], x["stageB_top"][0], 100 * x["stageB_top"][1]))

(ROOT / "results_final.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("\n".join(L))
