# NYC Airbnb Pricing Lab

This project predicts nightly Airbnb listing prices for New York City using the
`AB_NYC_2019.csv` dataset. The main workflow cleans the data, engineers
listing-level features, tunes several regression models with Optuna, compares
held-out dollar-scale metrics, and analyzes model errors.

## Repository Map

- `pricing_lab/`: shared Python package for cleaning, feature engineering,
  metrics, tuning, diagnostics, and model definitions.
- `notebooks/`: final notebook workflow for data checks, training, analysis,
  and optional XGBoost price-band plots.
- `tests/`: lightweight `unittest` checks for preprocessing audits,
  diagnostics, and lite-mode sampling.
- `scripts/generate_eda_plots.py`: optional script that regenerates the curated
  EDA images in `airbnb_plots/`.
- `airbnb_plots/`: selected EDA figures kept in the repo for review.
- `legacy/`: earlier exploratory notebook/script work kept for reference, not
  the recommended workflow.
- `artifacts/`: generated model files and analysis outputs. This directory is
  intentionally ignored by git.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate ml-optimized
python -m ipykernel install --user --name ml-optimized --display-name "ml-optimized"
```

Use the `ml-optimized` kernel for the notebooks.

## Recommended Review Order

1. `notebooks/01_data_and_split.ipynb`
   - Loads the CSV through `pricing_lab.data`.
   - Shows the shared cleaned train/test split and log-price target.
2. `notebooks/02_train_and_compare.ipynb`
   - Trains or loads the configured base models and ensembles.
   - Writes fitted model artifacts to `artifacts/`.
3. `notebooks/03_analyze.ipynb`
   - Loads model artifacts, compares metrics, plots residuals, and reports
     feature importance and price-cap diagnostics.
4. `notebooks/04_xgboost_price_band_plots.ipynb`
   - Optional focused plots for XGBoost prediction gaps by price band.

## Training Modes

Use `lite` for a quick representative smoke run, `sample` for fast iteration,
and `full` for heavier tuning.

Notebook training:

```python
MODE = "lite"  # lite | sample | full
```

CLI training:

```bash
python -m pricing_lab.run_all --mode lite
python -m pricing_lab.run_all --mode sample
python -m pricing_lab.run_all --mode full
python -m pricing_lab.run_all --mode lite --lite-train-rows 400
```

Optional trial overrides include `--n-trials-elastic`, `--n-trials-knn`,
`--n-trials-xgb`, `--n-trials-svm`, `--n-trials-nn`, `--n-trials-rf`, and
`--n-trials-ensemble-weights`.

## Tests

`pytest` is not required for this repo. Use the standard-library test runner:

```bash
conda run -n ml-optimized python -B -m unittest discover -s tests -v
```

Fast CLI smoke check:

```bash
conda run -n ml-optimized python -B -m pricing_lab.run_all --mode lite \
  --lite-train-rows 40 \
  --n-trials-elastic 1 --n-trials-knn 1 --n-trials-xgb 1 \
  --n-trials-svm 1 --n-trials-nn 1 --n-trials-rf 1 \
  --n-trials-ensemble-weights 1
```

## Regenerating EDA Plots

The checked-in `airbnb_plots/` images are review artifacts. To regenerate them:

```bash
conda run -n ml-optimized python scripts/generate_eda_plots.py
```

The script uses `pricing_lab.data.clean_listings_dataframe` as the shared
cleaning source of truth, then preserves display-only columns such as host name
and last review date for plotting.

## Notes for Review

- The target is `log1p(price)` during modeling; reported MAE/RMSE/R2 are mapped
  back to the original dollar scale.
- The preprocessing intentionally applies a global IQR cap to model the
  mainstream listing market. `compute_price_cap_audit` and `03_analyze.ipynb`
  quantify which rows are excluded.
- Bytecode, notebook checkpoints, pytest caches, and generated model artifacts
  are ignored so the repo stays focused on source, notebooks, tests, data, and
  selected review figures.

## Fun Facts

- "Michael" manages 383 listings, more than many boutique hotel chains.
- Manhattan dominates with about 42% of NYC Airbnb listings.
- Manhattan is the priciest borough at about $135/night median.
- Manhattan has the highest share of entire-home listings.
- About 20% of listings get zero reviews per month.
- Entire homes cost about 3.3x more than shared rooms in the cleaned data.
- Price vs reviews correlation is about -0.02, so more reviews does not imply a
  pricier listing.
- Peak estimated review activity is May 2019.
