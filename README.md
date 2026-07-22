# Preference-Conditioned Reinforcement Learning for Energy-Aware Multi-Objective Flexible Job Shop Scheduling

This repository contains an implementation of Preference-Conditioned Reinforcement Learning for solving a multi-objective Flexible Job Shop Scheduling Problem (MOFJSP) considering on-site photovoltaic (PV) generation, approximated using pvlib-python (https://github.com/pvlib/pvlib-python), and dynamic electricity prices obtained from SMARD (https://www.smard.de).

## Setup

The project environment is defined in `.devcontainer/env.yaml` (conda/mamba). Either open the repository in the provided dev container (`.devcontainer/`), or create the environment manually:

```bash
micromamba create -n mofjsp -f .devcontainer/env.yaml
micromamba activate mofjsp
```

### Weights & Biases (optional)

Create a `.env` file in the project root with:

```
WANDB_API_KEY=<your_api_key>
WANDB_ENTITY=<your_wandb_entity>
```

Logging can also be disabled per run via the `wandb_mode` config option.

## Data generation

Problem instances are generated from a data-generation config (paths are relative to `config/`):

```bash
python -m src.data_generator.instance_factory -fp data_generation/green_fjsp/config_GreenFJSP_10x10x10.yaml
```

This produces a pickled instance file under `data/instances/green_fjsp/`.

### Converting instances for publication

To export a generated `.pkl` instance file into portable, tool-independent `instances.json` and `energy_profiles.csv` files (used for the public data release):

```bash
python convert_data_csv.py <input.pkl> <output_dir>
```

## Training

```bash
python -m src.agents.train_RL -fp training/green_fjsp/config_train_GreenFJSP_10x10x10.yaml
```

Optional flags: `-s/--seed`, `-wbn/--wandb_project`, `-cd/--config_description`, `-mn/--saved_model_name`, `-en/--experiment_name`.

## Evaluation / testing

```bash
python -m src.agents.test_RL -fp testing/green_fjsp/config_test_GreenFJSP_10x10x10.yaml
```

Add `-plot` to generate Gantt charts during evaluation.

## NSGA-II baseline

```bash
python -m src.agents.metaheuristic.nsga2
```

## Ablation study

`run_ablation.py` trains a single ablation variant (network architecture x preference sampler), intended to be invoked as a SLURM array job:

```bash
sbatch --array=0-5 ablation_slurm.sh
```

or directly:

```bash
python run_ablation.py --variant_id 0 --config training/green_fjsp/config_train_GreenFJSP_10x10x10.yaml
```

See the variant table at the top of `run_ablation.py` for the architecture/sampler combination behind each `--variant_id`.

## Data availability

Large generated artifacts (raw problem instances, trained model checkpoints, WandB run logs) are not included in this repository. Regenerate them with the commands above, or obtain the published dataset from: `[TODO: add external data hosting link, e.g. Zenodo]`.
