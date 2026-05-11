# 12 Inference Method Variants Used in the Confound Audit

This document enumerates every inference-method variant audited in the
manuscript, the prior network and edge-combination strategy each uses,
the tissue context, any post-processing, and the hyperparameters that
were tuned vs. left at their published defaults.  The same canonical
labels appear in the manuscript's Methods table and in every figure
legend.

| # | Label (as in figures)                | Family / source                     | Prior network        | Edge combination | Tissue context  | Post-processing               | Notes                                                  |
|--:|--------------------------------------|-------------------------------------|----------------------|------------------|-----------------|-------------------------------|--------------------------------------------------------|
|  1 | FM-GRN (no prior)                    | scGPT attention                     | none                 | raw              | generic         | none                          | Baseline: raw foundation-model attention edges.       |
|  2 | Regulatory prior only                | Curated regulatory baseline         | OmniPath-Reg         | prior-only       | generic         | none                          | No FM contribution; isolates the prior's intrinsic structure. |
|  3 | OmniPath                             | scGPT + OmniPath full               | OmniPath full        | union            | generic         | none                          | Full OmniPath signalling layer.                       |
|  4 | OmniPath (relaxed)                   | scGPT + OmniPath relaxed            | OmniPath             | relaxed          | generic         | none                          | Relaxed edge filter; lower confidence threshold.      |
|  5 | OmniPath-Relaxed + Immune/HPN        | scGPT + OmniPath, immune HPN        | OmniPath             | relaxed          | immune (HPN)    | none                          | "HPN" = head-processing network, the immune-tissue   |
|  6 | OmniPath-Relaxed + Immune/HPN (cal.) | as #5 + logistic calibration        | OmniPath             | relaxed          | immune (HPN)    | logistic calibration          | tissue-specific post-hoc edge re-weighting layer.    |
|  7 | DoRothEA-Intersect + Immune/HPN      | scGPT + OmniPath ∩ DoRothEA         | OmniPath ∩ DoRothEA  | intersection     | immune (HPN)    | none                          | Strictest prior overlap; reduces false positives.    |
|  8 | DoRothEA-Union + Immune/HPN          | scGPT + OmniPath ∪ DoRothEA         | OmniPath ∪ DoRothEA  | union            | immune (HPN)    | none                          | Most permissive prior union.                          |
|  9 | DoRothEA-Union + Immune/HPN (cal.)   | as #8 + logistic calibration        | OmniPath ∪ DoRothEA  | union            | immune (HPN)    | logistic calibration          |                                                       |
| 10 | DoRothEA-Union + Immune/HPN (scaled) | as #8 + score scaling               | OmniPath ∪ DoRothEA  | union            | immune (HPN)    | scaled (range normalisation)  |                                                       |
| 11 | DoRothEA-Union + Immune/HPN (sc-L)   | as #8 + large-dim score scaling     | OmniPath ∪ DoRothEA  | union            | immune (HPN)    | scaled-L (large-dim)          |                                                       |
| 12 | InterCell + Immune/HPN               | scGPT + InterCell                   | OmniPath InterCell   | union            | immune (HPN)    | none                          | Cell-cell communication layer; ligand-receptor focus. |

### HPN — Head-Processing Network

"HPN" (head-processing network) refers to the post-hoc filtering and
re-weighting layer applied to raw scGPT attention edges in the
immune-tissue context.  It is a small fully-connected network that takes
as input the multi-head attention statistics for a candidate edge and
produces a single tissue-conditioned score.  Methods labelled
"+ Immune/HPN" use this layer; methods without that suffix use the raw
attention score directly.

### Candidate-universe construction

All 12 variants share the same candidate-universe construction:

  * 76 TFs (intersection of (a) the scGPT vocabulary, (b) the human-TF
    catalogue of Lambert et al., 2018, and (c) TFs with non-trivial
    representation across all three Tabula Sapiens tissues).
  * 108 targets (top-N expressed genes in the union of the three
    tissues, intersected with the scGPT vocabulary).
  * ≈8000 directed (TF -> target) edges per tissue, after removing
    TF=target pairs and edges with zero overlap in all priors.

### Why these 12 variants?

The variant grid is built to cross-cut **prior coverage** (none -> permissive
-> strict), **tissue conditioning** (generic vs. immune/HPN), and
**post-processing** (raw vs. calibrated vs. scaled).  These three axes
were chosen because they are the levers most often exposed to users by
existing tools (OmniPath/DoRothEA/InterCell/scGPT pipelines) and the
levers most likely to interact with the three confound classes audited.
We deliberately did not include the dozens of further hyperparameter
combinations that affect inference quality but not the audit story.

### Hyperparameters (prespecified vs. tuned)

| Hyperparameter                          | Source        | Value                                       | Tuned? |
|-----------------------------------------|---------------|---------------------------------------------|--------|
| scGPT model version                     | published     | scGPT-Human v1                              | no     |
| Attention layer for edge extraction     | published     | last layer, mean across heads               | no     |
| HPN architecture                        | this work     | 3-layer MLP, hidden=64                      | tuned (Sec. 3.4) |
| HPN training data                       | published     | Immune Human, scIB benchmark                | no     |
| DoRothEA confidence cut                 | Garcia-Alonso 2019 | level A-B                              | no     |
| OmniPath edge filter (strict / relaxed) | Türei 2016    | curation_effort >= 2 / >= 1                 | no     |
| Logistic calibration training data      | this work     | held-out 20% of immune edges                | tuned  |
| Score-scaling normaliser                | this work     | per-method robust z-score                   | tuned  |
| Top-k cut for audit                     | this work     | {100, 250, 500, 1000, 2000}                 | swept  |
| Distance threshold for proximity        | this work     | {0.5, 1, 5, 10} Mb                          | swept  |
| Topological null replicate count        | this work     | 2000 (with optional GPD tail)               | tuned  |
| ASI blacklist threshold                 | this work     | 0.5 (calibrated on simulated null, A.5)     | tuned  |
| BH q cut                                | standard      | 0.10                                        | no     |
