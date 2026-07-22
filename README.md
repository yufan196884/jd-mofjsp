# Preference-Conditioned Reinforcement Learning for Energy-Aware Multi-Objective Flexible Job Shop Scheduling

This repository contains an implementation of Preference-Conditioned Reinforcement Learning for solving a multi-objective Flexible Job Shop Scheduling Problem (MOFJSP) considering on-site photovoltaic (PV) generation, approximated using pvlib-python (https://github.com/pvlib/pvlib-python), and dynamic electricity prices obtained from SMARD (https://www.smard.de).


### Weights & Biases (optional)

Create a `.env` file in the project root with:

```
WANDB_API_KEY=<your_api_key>
WANDB_ENTITY=<your_wandb_entity>
```

Logging can also be disabled per run via the `wandb_mode` config option.

## Data generation

Problem instances are generated from a data-generation config

```bash
python -m src.data_generator.instance_factory -fp data_generation/green_fjsp/config_GreenFJSP_10x10x10.yaml
```

This produces a pickled instance file under `data/instances/green_fjsp/`.


## Training

```bash
python -m src.agents.train_RL -fp training/green_fjsp/config_train_GreenFJSP_10x10x10.yaml
```

## Evaluation / testing

```bash
python -m src.agents.test_RL -fp testing/green_fjsp/config_test_GreenFJSP_10x10x10.yaml
```


## Data availability

Large generated artifacts (raw problem instances, trained model checkpoints, WandB run logs) are not included in this repository. Regenerate them with the commands above.
