# Companion code — *Birth-Death Stochastic Conic Particle Gradient Descent for global optimization on the space of measures*

Companion Jupyter notebooks for the numerical experiments of Section 4 of

> **Birth-Death Stochastic Conic Particle Gradient Descent for global optimization on the space of measures**
> Yohann De Castro, Sébastien Gadat, Clément Marteau (2026, JMLR submission).

This repository contains everything needed to reproduce — qualitatively at a low budget, or exactly at the paper budget — the two numerical experiments of Section 4: density estimation on a 2D Gaussian Mixture Model and a two-layer ReLU regression on California Housing.

## Quick start

```bash
# 1. Clone
git clone <repo-url>
cd <repo-name>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch JupyterLab
jupyter lab
```

Open [`section4_companion.ipynb`](section4_companion.ipynb) and run all cells. The light demo finishes in about **2–3 minutes** on a recent laptop CPU; embedded paper figures load instantly from the saved `results_*` directories.

## Notebook layout

| Notebook | Purpose | Budget |
|---|---|---|
| **[`section4_companion.ipynb`](section4_companion.ipynb)** | One-stop companion to **Section 4 of the paper**. Mirrors §4.1, §4.2 and §4.3 with explicit `main_jmlr.tex` line cross-references, runs a light-budget demo, and displays the paper's exact figures and tables. | ~2-3 min |
| [`experiments_fastpart_gmm_fixed_covariance.ipynb`](experiments_fastpart_gmm_fixed_covariance.ipynb) | Full-budget GMM experiment of §4.1. Generates the saved `results_gmm_fixed_cov_*` directories. | ~10-20 min |
| [`big_experiment_fastpart_gmm_fixed_covariance.ipynb`](big_experiment_fastpart_gmm_fixed_covariance.ipynb) | Long-running GMM variant ($T_{\mathrm{sto}}=10^{6}$, $K=25$, $R=30$). | hours |
| [`experiments_fastpart_nn_california.ipynb`](experiments_fastpart_nn_california.ipynb) | Full-budget California-Housing experiment of §4.2. Generates the saved `results_nn_california_*` directories. | ~30-60 min |

## Section 4 cross-references inside `section4_companion.ipynb`

Each markdown cell of the companion notebook flags the corresponding line range in `main_jmlr.tex`:

| Section | `main_jmlr.tex` |
|---|---|
| §4 — Numerical experiments | line 1173 |
| §4.1 — GMM with fixed covariance | line 1179 |
| §4.2 — Two-layer NN on California Housing | line 1200 |
| §4.3 — Experimental Results and Discussion | line 1227 |
| §4.3.1 — Dynamics of the Birth-Death process | line 1237 |
| §4.3.2 — Convergence and generalisation | line 1258 |
| §4.3.3 — Spatial distribution of GMM particles | line 1274 |
| §4.3.4 — Performance Analysis and Experimental Parameters | line 1321 |

## Reproducing the paper figures and tables

`section4_companion.ipynb` pulls the paper's exact figures from two timestamped directories:

- `results_gmm_fixed_cov_20260305_115544/` — source of `fig3*` (loss / BD events / tv) and `fig4*` (final positions) used in Figures `fig:loss_vs_time`, `fig:bd_events`, `fig:gmm_positions` and Table `tab:gmm_results`.
- `results_nn_california_20260303_171130/` — source of `fig1*` (no-BD baselines), `fig2*` (BD curves) for Figures `fig:loss_vs_time` and `fig:bd_events`, and Table `tab:nn_results`. The directory also contains `fig3*_dual_cert_*`, `fig4*_weights_*` and `fig5*_pos_*` panels that are not in the paper — they are shown in the companion notebook as pedagogical diagnostics.

To regenerate either snapshot at the full paper budget, run the corresponding experiment notebook end-to-end; each one writes a new `results_*` directory next to itself with a fresh timestamp.

## Methods at a glance

All four variants minimise the BLASSO objective $J(\nu)=\tfrac{1}{2}\|y-\Phi(\nu)\|_{\mathbb{H}}^{2}+\kappa\|\nu\|_{\mathrm{TV}}$ over the space of finite positive measures:

| Variant | Gradient | Birth-Death |
|---|---|---|
| Full-Batch CPGD | Exact, $\mathcal{O}(p\,n)$ per step | — |
| Stochastic CPGD (Fast Spawn) | Mini-batch, $\mathcal{O}(p\,B)$ per step | — |
| Full-Batch CPGD + BD | Exact | Death rule $s_j > \tau_{\mathrm{death}}$, birth rule $\min_k J'_\nu(\hat t^{(k)}) < \tau_{\mathrm{birth}}$ |
| Fast Spawn & Prune (paper, **FS&P**) | Mini-batch | Same BD rules with the stochastic birth threshold $\tau_{\mathrm{birth}}\sqrt{\log m_k / m_k}$ |

The exact rules (and their derivation from a second-order Taylor expansion of $J$) are reproduced in the **math appendix** at the end of `section4_companion.ipynb`.

## Hardware

All experiments and figures in the paper were produced on a standard laptop CPU (Apple Silicon, no GPU required). PyTorch will use a CUDA GPU automatically if one is available.

## Requirements

- Python ≥ 3.10
- `torch`, `numpy`, `matplotlib`, `scikit-learn`, `jaxtyping`
- `jupyterlab` (for interactive use)

See [`requirements.txt`](requirements.txt) for pinned versions.

## Citing

If you use this code in academic work, please cite the paper:

```bibtex
@article{decastro2026fastpart2,
  title  = {Birth-Death Stochastic Conic Particle Gradient Descent for global optimization on the space of measures},
  author = {De Castro, Yohann and Gadat, S{\'e}bastien and Marteau, Cl{\'e}ment},
  journal= {Journal of Machine Learning Research},
  year   = {2026},
  note   = {Submitted}
}
```

## License

MIT — see [`LICENSE`](LICENSE).
