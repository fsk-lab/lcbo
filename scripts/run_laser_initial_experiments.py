import time
from pathlib import Path
import json
from moses.datatypes import LCGradient
from lcbo.data.config import HPLCExperimentConfig
from lcbo.routines import run_experiment


if __name__ == "__main__":

    config_dir = Path.cwd() / "configs"
    data_dir = Path.cwd() / "results" / "laser_experiment" / "initial_experiments"

    with open(config_dir / "thermo_experiment_config.json", "r") as f:
        experiment_config = HPLCExperimentConfig.from_dict(json.load(f))

    instrument = experiment_config.get_instrument()

    initial_gradients = [
        {"proportions_b": [0.2, 0.4, 1.0, 1.0], "time_steps": [0.0, 5.0, 15.0, 20.0]},
        {"proportions_b": [0.1, 0.3, 0.8, 1.0], "time_steps": [0.0, 3.0, 12.0, 25.0]},
        {"proportions_b": [0.5, 0.6, 0.9, 1.0], "time_steps": [0.0, 1.0, 15.0, 21.0]},
        {"proportions_b": [0.8, 0.9, 1.0, 1.0], "time_steps": [0.0, 8.0, 16.0, 24.0]}
    ]

    initial_experiments = []
    for grad_details in initial_gradients:

        time_stamp = time.strftime("%Y-%m-%d_%H-%M")

        sample = {
            "sample_name": f"Laser_Crude_{time_stamp}",
            "folder": "Felix/LCBO/Laser",
            "sample_position": "R:D9",
            "injection_volume": 2.0,
            "sample_solvent": "Acetonitrile"
        }

        eluent_program = LCGradient(
            name="Laser_Gradient",
            time_steps=grad_details["time_steps"],
            proportions_b=grad_details["proportions_b"],
            flow_rate=0.4,
            eluents=("Water (0.1% Formic Acid)", "Acetonitrile (5% THF, 0.1% Formic Acid)"),
            equilibration_time=2.0,
        )

        _ = run_experiment(
            instrument=instrument,
            config=experiment_config,
            gradient=eluent_program,
            sample_settings=sample,
            targets=None,
            data_dir=data_dir / sample["sample_name"]
        )

        initial_experiments.append(str(data_dir / sample["sample_name"]))

    instrument.standby()

    with open(data_dir / "initial_experiments.json", "w") as f:
        json.dump(initial_experiments, f, indent=4)

