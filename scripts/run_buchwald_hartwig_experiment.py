import time
from pathlib import Path
import json
from moses.datatypes import Compound, LCGradient
from lcbo.data.config import HPLCExperimentConfig
from lcbo.routines import run_experiment


if __name__ == "__main__":

    data_dir = Path.cwd() / "results" / "buchwald_hartwig_experiment"

    time_stamp = time.strftime("%Y-%m-%d_%H-%M")

    sample = {
        "sample_name": f"BH_Crude_{time_stamp}",
        "folder": "Felix/LCBO/Buchwald-Hartwig",
        "sample_position": "R:D6",
        "injection_volume": 1.0,
        "sample_solvent": "Acetonitrile"
    }

    eluent_program = LCGradient(
        name="Buchwald_Hartwig_Gradient",
        time_steps=[0.0, 2.0, 12.0, 20.0],
        proportions_b=[0.4, 0.4, 1.0, 1.0],
        flow_rate=0.4,
        eluents=("Water (0.1% Formic Acid)", "Acetonitrile (5% THF, 0.1% Formic Acid)"),
        equilibration_time=2.0,
    )

    config_dir = Path.cwd() / "configs"

    with open(config_dir / "thermo_experiment_config.json", "r") as f:
        experiment_config = HPLCExperimentConfig.from_dict(json.load(f))

    instrument = experiment_config.get_instrument()

    result = run_experiment(
        instrument=instrument,
        config=experiment_config,
        gradient=eluent_program,
        sample_settings=sample,
        targets=None,
        data_dir=data_dir / sample["sample_name"]
    )

    instrument.standby()

