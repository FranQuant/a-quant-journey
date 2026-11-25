# Corrected four-notebook experiment

The four teaching notebooks reconstruct the historical workflow and its
methodological correction.

## Learning path

1. [Multi-asset EDA](01_understand_the_data.ipynb) — bounded closes, missingness,
   returns, distributions, normality examples, volatility and correlation.
2. [Indicators, PCA and clustering](02_construct_candidate_features.ipynb) —
   selected historical indicators and exploratory cross-asset structure; learned
   transforms fit on the training segment only.
3. [Fixed predictive comparison](03_test_predictive_value.ipynb) — common causal
   SPY dataset and chronological Track E train/validation protocol; seven fixed
   classifier families plus a training-majority baseline.
4. [Portfolio accounting](04_predictions_to_portfolio_accounting.ipynb) —
   Notebook 03 predictions mapped to SPY long/cash positions using the next
   observed close mapper and canonical ledger; matched cash/SPY benchmarks and
   zero-cost/illustrative 5 bps cases.

## Boundaries and execution

The data-dependent notebooks use the existing TPQ snapshot
`data/raw/tpq_eod_2014_2023.csv` (SHA-256
`771c40d3f8d19d1cf524764296bfa2943f8ca9c63514b99bb8d95c9b84d1e407`) through
2021-12-28 only. The multi-column reader scans the date field through the
cutoff, then reads selected columns and the bounded prefix. Notebook 03 uses
accepted Track E dates and strict maturity purges; it does not construct or
score reserved-test examples. The same three causal features feed all seven
models. There is no tuning, search, or validation-based deployment selection.

The forecast is a next-close **price-return proxy**: features are known after
close t, the hypothetical entry is the next observed close, and the position is
valued at the following observed close. It is not next-open execution, total-
return certification, or investable alpha. The validation period is prior
development evidence, not a fresh independent holdout. Notebook 04 excludes
dividends and splits and uses zero costs and an illustrative proportional cost
of 5 bps per traded notional.

## Local setup

From the repository root, use Python 3.11.15 and a new virtual environment.
The teaching requirements pin the direct dependencies. Git is required to
load the committed notebooks and helpers.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --sys-prefix --name a-quant-journey
```

The snapshot is included at the path above. The runner
links that file into a fresh run directory. The committed notebook loaders
bound market observations through 2021-12-28. The runner does not download
the file or verify its full-file digest.

## Running the sequence

```sh
.venv/bin/python scripts/execute_teaching_notebooks.py
```

Each invocation prints its commit and unique directory under
`outputs/teaching_runs/`. Each notebook starts a fresh kernel in that
directory, using notebook sources and project helpers from the same Git
commit. The runner accepts optional notebook filenames, executes selections
in teaching order, and includes 03 whenever 04 is requested.

| Location within the run directory | Contents |
|---|---|
| `executed/` | Executed notebook copies, including figures and tables |
| `outputs/track_e_multifamily/` from 03 | `validation_metrics.csv`, `validation_predictions.csv`, `provenance.json` |
| Same directory from 04 | `portfolio_equity_curves.csv`, `portfolio_summary.csv`, `portfolio_provenance.json` |

Notebook 04 checks the prediction file digest against the matching provenance
and verifies alignment with the validation targets. Both notebooks use the
same run directory. Existing checkout outputs remain available separately.
A failed run may contain partial results. Rerunning creates a new directory.
Keep these local artifacts private, including the dataset link.

For the data-independent tests in a clean teaching checkout, run:

```sh
.venv/bin/python -m pytest -q --deselect=tests/test_tpq.py::TestTPQ::test_historical_dataset
```

The excluded historical regression reads the full snapshot. Local execution
is the documented workflow. Colab execution has yet to be validated.
