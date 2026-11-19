from pathlib import Path
import json
from moses.datatypes import LCGradient


if __name__ == "__main__":

    instrument = "Thermo"  # "Agilent"
    sample_name = "7-Compound-Mixture"
    injection_volume = 2.0

    data_dir = Path.cwd() / "results" / "cross_platform_reproduction"

    gradient = LCGradient(
        name="Test_Gradient",
        time_steps=[0.0, 3.0, 5.0, 8.2, 8.8, 10.5, 12.0, 14.0],
        proportions_b=[0.2, 0.2, 0.3, 0.43, 0.48, 0.56, 0.8, 0.2],
        flow_rate=0.4,
        eluents=("Water (0.1% Formic Acid)", "Acetonitrile (0.1% Formic Acid)"),
        equilibration_time=3.0,
    )

    if instrument == "Thermo":
        from moses.chromeleon import ThermoHPLCMSController

        with open(Path.cwd() / "configs" / "thermo_instrument_config.json", "r") as f:
            config = json.load(f)
        controller = ThermoHPLCMSController(config)

        sample_position = "G:F1"
        results_folder = "Felix/LCBO"

    elif instrument == "Agilent":
        from moses.agilent import AgilentHPLCMSController

        with open(Path.cwd() / "configs" / "agilent_lcms_config.json", "r") as f:
            config = json.load(f)
        controller = AgilentHPLCMSController(config)

        sample_position = "P1-A4"
        results_folder = r"C:\CDSProjects\Results\LCBO"

    else:
        raise ValueError(f"Unsupported instrument: {instrument}")

    controller.set_acquisition_method(gradient=gradient, ms_settings={"mode": "positive_negative"})

    data = controller.start_run(
        folder=results_folder,
        sample_name=sample_name,
        sample_position=sample_position,
        injection_volume=injection_volume
    )

    data_dir.mkdir(parents=True, exist_ok=True)
    data.save(data_dir / f"{sample_name}_{instrument}.npz")





