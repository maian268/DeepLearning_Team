import json
import os
import pickle
from datetime import datetime
from pathlib import Path


def start_experiment(root_dir, model_name, config):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(root_dir) / model_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return run_id, run_dir


def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as model_file:
        pickle.dump(model, model_file)


def log_wandb(run_dir, project, model_name, config, metrics, files):
    import wandb

    mode = os.getenv("WANDB_MODE", "online")
    run = wandb.init(
        project=project,
        name=f"{model_name}_{run_dir.name}",
        job_type="train",
        dir=str(run_dir),
        mode=mode,
        config={**config, **metrics},
    )
    wandb.log(metrics)
    artifact = wandb.Artifact(
        name=f"{model_name}-{run_dir.name}",
        type="model",
        metadata=config,
    )
    for file_path in files:
        artifact.add_file(str(file_path))
    run.log_artifact(artifact)
    run.finish()
    return mode
