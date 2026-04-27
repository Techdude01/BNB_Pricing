## Environment Setup (Conda)

```bash
conda env create -f environment.yml
conda activate ml-optimized
python -m ipykernel install --user --name ml-optimized --display-name "ml-optimized"
```

Use the `ml-optimized` kernel when running:
- `notebooks/01_data_and_split.ipynb`
- `notebooks/02_train_and_compare.ipynb`
- `notebooks/03_analyze.ipynb`

Model artifacts are written to `artifacts/` by default from notebook 02.

## Training Modes

Use `sample` for fast iteration and `full` for heavier tuning.

- Notebook (`notebooks/02_train_and_compare.ipynb`)
  - Set `MODE = "sample"` or `MODE = "full"` in the config cell.
  - `sample` uses fewer trials and a 35% training subset.
  - `full` uses full training data and default config trial counts.
  - Granular retraining flags control each model independently:
    - Base: `RETRAIN_ELASTICNET`, `RETRAIN_KNN`, `RETRAIN_XGBOOST`, `RETRAIN_RANDOM_FOREST`, `RETRAIN_SVM`, `RETRAIN_NEURAL_NETWORK`
    - Ensemble: `RETRAIN_VOTING_EQUAL`, `RETRAIN_VOTING_WEIGHTED`, `RETRAIN_STACKING`
  - Ensembles are resolved after base models, using selected base candidates.
- CLI (`pricing_lab/run_all.py`)
  - Fast: `python -m pricing_lab.run_all --mode sample`
  - Heavy: `python -m pricing_lab.run_all --mode full`
  - Optional overrides still work: `--n-trials-elastic`, `--n-trials-knn`, `--n-trials-xgb`

## Analysis Notebook

Use `notebooks/03_analyze.ipynb` for post-training diagnostics and explainability.

- Expected run order:
  - Run `notebooks/02_train_and_compare.ipynb` first to create `artifacts/*.joblib`.
  - Then run `notebooks/03_analyze.ipynb` with `RUN_TRAINING=False` for fast analysis-only reruns.
- Included outputs:
  - Actual vs Predicted plots (per model)
  - Residuals vs Predicted plots (per model)
  - Residual density overlay
  - Residual histograms
  - MAE/RMSE bars and R² bar
  - Absolute error quantile curves
  - SHAP bar/beeswarm/dependence (XGBoost)
  - Permutation importance for all models
  - Optional model-agnostic SHAP for selected non-tree models
- Exports:
  - `artifacts/shap_feature_importance_xgboost.csv`
  - `artifacts/permutation_importance_all_models.csv`
  - Optional plot images in `artifacts/plots/` when `SAVE_PLOTS=True`

## Fun Facts

- 'Michael' manages 383 listings, more than many boutique hotel chains.
- Manhattan dominates with 42% of all NYC Airbnb listings
- Manhattan is the priciest borough at $135/night median
- Manhattan has the highest share of entire-home listings (58%)
- 20% of all listings get zero reviews per month; among active ones, the median is ~0.72/month
- The most common minimum stay is 1 night; 26% of hosts allow single-night bookings
- 37% of listings show 0 available days, so listed but fully blocked out
- Entire homes cost ~3.3× more than shared rooms ($150 vs $45 median nightly)
- Price vs reviews correlation is −0.02 meaning more reviews does NOT mean a pricier listing
- Peak estimated review activity is May 2019
- Most-reviewed host overall is Michael
- 45,876 total clean listings across NYC
