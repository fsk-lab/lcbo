import time
from pathlib import Path
import json

from moses.datatypes import Compound
from lcbo.data.config import HPLCExperimentConfig, HPLCOptimizationConfig
from lcbo.routines import run_optimization_campaign


if __name__ == "__main__":

    config_dir = Path.cwd() / "configs"
    data_dir = Path.cwd() / "results" / "known_mixture_optimization"

    time_stamp = time.strftime("%Y-%m-%d_%H-%M")
    campaign_name = f"Optimization_Campaign_{time_stamp}"

    sample_settings = {
        "sample_name": "Sample7C",
        "folder": f"Felix/LCBO/{campaign_name}",
        "sample_position": "G:F1",
        "injection_volume": 2.0,
        "sample_solvent": "Acetonitrile"
    }

    compounds = [
        Compound.from_smiles(name="Terpyridine", smiles="C1(C2=NC(C3=NC=CC=C3)=CC=C2)=CC=CC=N1"),
        Compound.from_smiles(name="2-Bromo-5-chloropyridine", smiles="C1=CC(=NC=C1Cl)Br"),
        Compound.from_smiles(name="4-Bromo-2-nitrobenzoic acid", smiles="O=C(O)C1=CC(=C(C=C1)Br)[N+](=O)[O-]"),
        Compound.from_smiles(name="1-(p-Toluenesulfonyl)pyrrole-2-boronic acid MIDA ester", smiles="CN1CC(=O)OB(OC(=O)C1)c2cccn2S(=O)(=O)c3ccc(C)cc3"),
        Compound.from_smiles(name="3,6-Dimethylcarbazole", smiles="CC1=CC2=C(C=C1)NC3=C2C=C(C)C=C3"),
        Compound.from_smiles(name="1,3,5-Trimethoxybenzene", smiles="COC1=CC(=CC(=C1)OC)OC"),
        Compound.from_smiles(name="3-Chloro-benzo[B]thiophene-2-carboxylic acid", smiles="O=C(O)C1=CC2=C(S1)C=C(C=C2)Cl"),
    ]


    with open(config_dir / "eluent_optimization_config.json", "r") as f:
        optimization_config = HPLCOptimizationConfig.from_dict(json.load(f))
    optimization_config.working_dir = data_dir / campaign_name
    optimization_config.num_compounds = len(compounds)

    with open(optimization_config.experiment_config, "r") as f:
        experiment_config = HPLCExperimentConfig.from_dict(json.load(f))
    experiment_config.working_dir = optimization_config.working_dir

    instrument = experiment_config.get_instrument()

    try:
        campaign = run_optimization_campaign(
            simulation=False,
            sample_settings=sample_settings,
            instrument=instrument,
            optimization_config=optimization_config,
            experiment_config=experiment_config,
            analytes=compounds,
            data_dir=optimization_config.working_dir
        )

        with open(optimization_config.working_dir / "campaign_summary.json", "w") as f:
            json.dump(campaign.to_dict(), f, indent=4)

    finally:
        instrument.standby()