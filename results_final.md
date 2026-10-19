# Results - final model (`final_model_v7_1_xai.ipynb`)

| Dataset | Accuracy | Macro-P | Macro-R | Macro-F1 | Wtd-P | Wtd-R | Wtd-F1 | ROC-AUC | SELV winner |
|---|---|---|---|---|---|---|---|---|---|
| CIC-UNSW-NB15-2024 (full release) | 0.9867 | 0.7126 | 0.6819 | 0.6873 | 0.9880 | 0.9867 | 0.9872 | 0.9644 | HYBRID W x OLA-cw k=30 p=3 @ t=0.7 |
| CIC-UNSW-NB15-2024 (curated) | 0.9489 | 0.7823 | 0.7229 | 0.7421 | 0.9498 | 0.9489 | 0.9486 | 0.9765 | DYN OLA-cw k=30 p=8 @ t=0.7 |
| UNSW-NB15 | 0.8259 | 0.7008 | 0.6590 | 0.6211 | 0.8550 | 0.8259 | 0.8329 | 0.9318 | HYBRID W x OLA-cw k=100 p=3 @ t=0.5 |
| CIC-IDS2017 | 0.9991 | 0.9849 | 0.9823 | 0.9835 | 0.9990 | 0.9991 | 0.9990 | 0.9929 | DYN OLA-cw k=100 p=1 @ t=0.7 |
| TON-IoT | 0.9809 | 0.9546 | 0.9673 | 0.9604 | 0.9815 | 0.9809 | 0.9811 | 0.9955 | DYN OLA-cw k=100 p=3 @ t=0.6 |

## XAI verification

| Dataset | Additivity max residual | shap vs xgboost max diff | Decisions reproduced | Gate top SHAP group | Stage-B top SHAP group |
|---|---|---|---|---|---|
| CIC-UNSW-NB15-2024 (full release) | 2.1e-05 | 0.0e+00 | 100.00% | SIZE (70.5%) | SVC (40.7%) |
| CIC-UNSW-NB15-2024 (curated) | 1.6e-05 | 0.0e+00 | 100.00% | SIZE (70.2%) | SVC (42.0%) |
| UNSW-NB15 | 1.7e-05 | 0.0e+00 | 100.00% | TIME (53.1%) | SIZE (44.5%) |
| CIC-IDS2017 | 1.8e-05 | 0.0e+00 | 100.00% | SIZE (50.3%) | SIZE (43.9%) |
| TON-IoT | 1.8e-05 | 0.0e+00 | 100.00% | SIZE (41.2%) | SIZE (52.1%) |
