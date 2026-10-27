
import random
import numpy as np
import yaml

from src.data_generator.instance_factory import (
    generate_instances_from_config,
)

config_path = (
    "config/data_generation/green_fjsp/"
    "config_GreenFJSP_5x5.yaml"
)

with open(config_path, encoding="utf-8") as f:
    config = yaml.safe_load(f)

config["num_instances"] = 2
config["write_to_file"] = False

seed = int(config.get("seed", 0))
random.seed(seed)
np.random.seed(seed)

instances = generate_instances_from_config(config)

assert len(instances) == 2

for idx, instance in enumerate(instances):
    tasks = instance["tasks"]
    machines = instance["machines"]
    solar = instance["solar_profile"]
    prices = instance["energy_prices"]

    assert len(tasks) > 0
    assert len(machines) > 0
    assert len(solar) > 0
    assert len(prices) > 0

    for task in tasks:
        assert len(task.machines) > 0
        assert len(task.runtimes) == len(machines)
        assert len(task.machine_running_energy) == len(machines)

        for m in task.machines:
            assert task.runtimes[int(m)] > 0

    print(f"Instance {idx}")
    print("  Operations:", len(tasks))
    print("  Machines:", len(machines))
    print("  Solar length:", len(solar))
    print("  Prices length:", len(prices))
    print("  Start offset:", instance["start_offset"])

print("MOFJSP generator smoke test passed.")
