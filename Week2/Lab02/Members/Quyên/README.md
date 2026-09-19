# House Price Experiments

## Structure

- `notebooks/`: one notebook per model and one comparison notebook.
- `src/`: shared preprocessing, evaluation and W&B tracking utilities.
- `experiments/`: immutable run folders containing config, metrics, model and predictions.
- `data/`: train/test CSV files.

## Run order

1. Run `gradient_boosting_regression.ipynb`.
2. Run `svr_regression.ipynb`.
3. Run `mlp_regression.ipynb`.
4. Run `pytorch_mlp_regression.ipynb`.
5. Run `logistic_regression_classification.ipynb` for high/low price classification.
6. Run `model_comparison.ipynb` to compare all regression metrics and the classification metrics separately.

Run `wandb login` before executing the notebooks. The model notebooks set `WANDB_MODE=online` and sync each run to the W&B cloud while also keeping local artifacts.
