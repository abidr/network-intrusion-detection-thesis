# figures/

Artifacts for the plain-language report (`../v7_1_reject_stages_report.md` / `.docx`).

| File | What it is |
|---|---|
| `v71b_architecture.png` | Figure 1 — the full v7.1b architecture |
| `make_arch_diagram.py` | regenerates the figure (matplotlib only) |
| `md2docx.py` | renders the Markdown report to `../v7_1_reject_stages_report.docx` |

Rebuild both, in this order, with the `ai4beg` env:

    python figures/make_arch_diagram.py
    python figures/md2docx.py

`md2docx.py` needs `python-docx`. Edit the **Markdown** file, never the `.docx` —
the `.docx` is generated and any hand edit is lost on the next run.

The figure departs from a plain pipeline drawing in one way: the combination rule
(box 6) is drawn as a **choice** between Static / Dynamic / Hybrid, because these
are one nested family and the winner is selected per dataset on the SELV split.

## XAI figures (`../results/xai_treeshap/`)

Regenerate with `python figures/make_xai_figures.py` (reads `xai_treeshap.json`, matplotlib only).
Group colours are fixed across all five: SVC blue, TIME orange, SIZE aqua, FLAG yellow.

| File | What it shows |
|---|---|
| `xai_fig1_shap_vs_gain.png` | SHAP vs gain-importance group shares, gate and Stage B |
| `xai_fig2_per_family.png` | per-family share of attribution by group |
| `xai_fig3_overlap_pairs.png` | which group pushes confused rows toward the wrong family (95% CI) |
| `xai_fig4_faithfulness.png` | top-k SHAP deletion vs random-k deletion |
| `xai_fig5_stability.png` | explanation stability under v7.3 random noise vs pre-registered thresholds |
