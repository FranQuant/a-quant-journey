# Teaching Guide

Follow [Getting Started](../../README.md#getting-started) from the repository root. The runner uses the committed notebooks and Python helpers, links the [included market data](../../data/raw/tpq_eod_2014_2023.csv), and starts a fresh kernel for each notebook. It prints a new directory under `outputs/teaching_runs/` for each run. Open the copies in its `executed/` folder to see the figures and tables.

## Reading the notebooks

1. [Understand the data](01_understand_the_data.ipynb) examines how missing observations affect daily returns, volatility and cross-asset correlations. The teaching analysis uses observations through December 28, 2021.
2. [Construct candidate features](02_construct_candidate_features.ipynb) explores technical indicators, PCA and clustering. Learned transformations are fitted on the training period only.
3. [Test predictive value](03_test_predictive_value.ipynb) compares seven fixed classifiers using the same three point-in-time SPY features. Each model predicts the same chronological validation dates. The benchmark always predicts the most frequent training class.
4. [From predictions to portfolio accounting](04_predictions_to_portfolio_accounting.ipynb) uses Notebook 03’s dated predictions to hold SPY or cash. It compares the resulting portfolios with cash and SPY buy-and-hold over matched dates. Selecting Notebook 04 also runs Notebook 03.

## Interpreting the results

Training precedes validation. Each period excludes examples whose outcomes would still be unknown at its end. The validation dates were examined during development, so the results provide no fresh, untouched holdout estimate.

A signal uses information available after the close. A position change takes effect at the next observed close and earns its first price return from that close to the following observed close. The backtest compares zero trading costs with an illustrative charge of five basis points per dollar traded. Portfolio returns exclude dividends and splits.
