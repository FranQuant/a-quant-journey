# Teaching Guide

The four notebooks form a reproducible research sequence, from market-data exploration to hypothetical portfolio accounting. Start with the [project README](../../README.md) for installation and execution instructions.

## Research sequence

1. [Understand the data](01_understand_the_data.ipynb) — inspect closes, missing observations, returns, volatility and correlations.
2. [Construct candidate features](02_construct_candidate_features.ipynb) — explore technical indicators, PCA and clustering; fit learned transformations on training data only.
3. [Test predictive value](03_test_predictive_value.ipynb) — compare seven fixed classifiers with a training-majority baseline using chronological train/validation splits.
4. [Predictions to portfolio accounting](04_predictions_to_portfolio_accounting.ipynb) — translate predictions into hypothetical SPY long/cash positions and compare with cash and SPY buy-and-hold.

## Execution and outputs

From the repository root, after installing `requirements.txt` and registering the Jupyter kernel:

```sh
.venv/bin/python scripts/execute_teaching_notebooks.py
```

The runner uses the included `data/raw/tpq_eod_2014_2023.csv`, runs each notebook in a fresh kernel, and saves executed notebooks and results under `outputs/teaching_runs/`. Notebook 04 consumes Notebook 03's predictions from the same run. The analysis reads market observations only through December 28, 2021.

## Research assumptions and limitations

- Features use information available after close `t`; hypothetical SPY entry is at the next observed close, with valuation at the following observed close—not the next market open.
- The experiment uses chronological validation and avoids fitting transformations on validation observations. This validation period was previously used during research development; it is **not** an untouched out-of-sample test.
- Portfolio accounting illustrates zero costs and 5 basis points per traded notional. Dividends, splits, taxes and market impact are excluded; results are price-return proxies, not certified total returns.
- The notebooks are educational research examples, **not evidence of deployable investment alpha**. Google Colab execution has not been validated.
