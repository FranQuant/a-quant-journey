# Teaching and reproduction guide

1. [01 — Understand financial market data](01_understand_the_data.ipynb) — Observed sessions, missing prices, returns and volatility.
2. [02 — Explore candidate features](02_construct_candidate_features.ipynb) — Trailing indicators, train-fitted PCA and clustering.
3. [03 — Test predictive value](03_test_predictive_value.ipynb) — Seven fixed SPY classifiers and a training-majority baseline.
4. [04 — From predictions to portfolio accounting](04_predictions_to_portfolio_accounting.ipynb) — Next-close positions, fees and a self-financing ledger.
5. [05 — Portfolio simulation, reconciliation and analytics](05_vectorbt_portfolio_reconciliation.ipynb) — Independent VectorBT fills, reconciliation and QuantStats analytics.

The SMA, RSI, MACD, principal components and clusters in 02 are exploratory. Notebook 03 builds its three fixed SPY predictors independently. Notebook 04 consumes the dated predictions. Notebook 05 consumes those predictions and compares its independent simulation with the canonical ledger and the same run’s saved equity and summaries.

## Environments and execution

The tested numerical environments use CPython 3.11.15 on macOS arm64. The root README sets up the base environment and kernel. Install the optional environment separately when reproducing the five-notebook run:

```bash
python3.11 -m venv .venv-vectorbt
.venv-vectorbt/bin/python -m pip install -r requirements-vectorbt.lock
.venv-vectorbt/bin/python -m ipykernel install --user --name nb05-native
.venv/bin/python scripts/execute_teaching_notebooks.py --all
```

By default, the runner uses the current Git commit. During development, use `--candidate` to test uncommitted source changes:

```bash
.venv/bin/python scripts/execute_teaching_notebooks.py --candidate --all
```

The runner prints a fresh directory under `outputs/teaching_runs/`. Its `source.json` identifies each source file in the snapshot by SHA-256 and records the checkout revision. Fresh kernels execute in the snapshot, reading the local input through the bounded loaders. Selecting 05 by filename also runs 03 and 04. For 01–04 only, omit `--all`. The options `--kernel` and `--analytics-kernel` select registered kernels.

Open the notebooks in `executed/` with JupyterLab. To use the interactive dashboard there, trust your own verified output.

For optional local HTML exports, add `--html` to either command and open the files in the run’s `html/` directory in a modern browser. Export requires Pandoc (tested with 3.9.0.2) and `jupyter-nbconvert` (7.17.0) on PATH. Pandoc renders equations as accessible MathML. The dashboard embeds Plotly JavaScript for offline hover and zoom.

The runner supplies 05 with `AQJ_ARTIFACT_DIR` pointing to its own `outputs/track_e_multifamily/`. For manual replay, set that variable to the exact directory of the identified run before starting the analytics kernel.

## Research boundaries

December 28, 2021 is the authorized research observation cutoff. Training and validation retain only labels available strictly before their respective boundaries. The 1,503 training examples precede the 500 previously examined validation predictions. Reserved observations were excluded from the producing model-development and reported evaluation workflows. An earlier unrestricted dataset-integrity regression accessed the full historical dataset. The historical manifest field `reserved_test_access` describes the producing workflow. Use bounded test selections for reproduction.

A feature at close $t$ sets a position at the next observed close. Its first price change runs from that close to the following observed close. Both engines start with USD 1,000 and use fractional SPY shares, zero or five basis points per fill, and terminal marking without a forced sale. They credit no separate dividends or splits. Validation accuracy measures direction; economic outcomes depend on exposure, move size and trading costs.

## Numerical reproducibility

The recorded NumPy 2.2.6 extension links to Accelerate and SciPy 1.13.1 BLAS links to bundled OpenBLAS. The 02/03 runs emitted matrix-operation warnings of undetermined cause. Earlier OpenBLAS comparisons agreed on aligned PCA outputs within tolerance and matched KMeans assignments. For 03, cross-backend prediction agreement applies to an earlier configuration. The final cap-500 MLP predictions were verified on the project backend.

## Scientific identities

Historical producer: `5dd73eda0f2f740365af8cf357696645114da94e`. Protected pre-refinement checkout: `55d5904dd67c029c91005d2643187e34e40a84e2`. Release `v0.2.0` identifies commit `55fb109f70a63b38734b26c38ecf36641ebfd902`. Each run records its own source identity.

The historical `data_sha256` is a declared expected input identity. The bounded run does not hash the full historical file. Prediction and scientific output digests below are calculated from produced bytes. Compare them before interpreting portfolio results.

| Artifact | Reference SHA-256 |
|---|---|
| `portfolio_equity_curves.csv` | `2fb9534ac0e2ce1a2a014d03af5cadc057cb4b1881cf11d3731bc5e81b0122e0` |
| `portfolio_provenance.json` | `0c9bf4e47598c78c13683b7114b3d78297c5ee67926cea9017fa95a777ef8602` |
| `portfolio_summary.csv` | `c74afb98cf2f78278f94873e68c8bcea885adbd89dc383ab6f244eaf3f368ab7` |
| `provenance.json` | `03af81e4f977ef9c471547bbb371e9fd9182cdd2da7ce801504c4039551ec0ad` |
| `validation_metrics.csv` | `80e2cad21692eb265c0d870bc88ede18154274742e35b144c4fa6ec1b8ccd3e8` |
| `validation_predictions.csv` | `0f47e3fa4662e17c01bd8f1c9d475471f4f261ca84d564f342f98309c840440e` |
