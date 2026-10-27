# MLP approximation of a 2D Gaussian

This project implements a PyTorch MLP to approximate the bivariate Gaussian density specified in the assignment. It includes independent train/validation/test splits, a custom training loop, validation-based early stopping, metrics, plots, CSV logs, and model checkpoints.

## Environment

- Python 3.10+
- PyTorch
- NumPy, pandas, Matplotlib, scikit-learn

## Install

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Run

From this folder:

```bash
python main.py
```

The script selects CUDA when available; otherwise it uses CPU. It generates 1,200 train samples, 300 validation samples, and 100 independent test samples. Data generation is fixed with `data_seed=2026`; model training is repeated with seeds 42, 123, and 2026 while keeping the architecture and training settings fixed.

## Main configuration

Edit `CONFIG` and `SEEDS` in `main.py` to change the experiment. The default model has two hidden layers with 64 neurons each, `tanh` activations, Adam, learning rate 0.001, batch size 64, up to 1,000 epochs, and early-stopping patience 100.

## Outputs

- `data/`: train, validation, and test CSV files
- `results/seed_results.csv`: test metrics and training time for each seed
- `results/summary_mean_std.csv`: mean and sample standard deviation (`ddof=1`) across seeds
- `results/loss_history_*.csv`: train/validation losses by epoch
- `results/predictions_seed_*.csv`: test predictions and errors
- `figures/`: data scatter, input/target distributions, actual-vs-predicted, error distributions, Gaussian surfaces, loss curves
- `checkpoints/`: model weights for each seed

## Metrics

- MSE: mean squared error
- MAE: mean absolute error
- RMSE: square root of MSE
- `train_seconds`: wall-clock time for training (not including data generation and plotting)

The test set is for final evaluation only. Do not use test metrics to choose hyperparameters. Choose a configuration using validation loss, then report test metrics. This script saves each seed's checkpoint and summarizes the three repeated runs.

## Notes

- The covariance matrix follows the assignment: `[[0.3, -0.5], [-0.5, 2.0]]`.
- The data split remains fixed across the three model seeds to make the runs comparable.
- Results are generated locally when you run the script; the ZIP does not contain fabricated metric values.
