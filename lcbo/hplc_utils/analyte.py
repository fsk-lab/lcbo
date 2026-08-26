from typing import Optional
from pathlib import Path
import json


class Analyte(object):
    """
    A class representing an analyte with specific properties relevant to HPLC retention modeling.
    """
    def __init__(self, log_k0: float, area: float, S_1: float, S_2: Optional[float] = None):
        self.log_k0 = log_k0
        self.area = area
        self.S_1 = S_1
        self.S_2 = S_2

    def to_dict(self) -> dict[str, Optional[float]]:
        return {
            "log_k0": self.log_k0,
            "area": self.area,
            "S_1": self.S_1,
            "S_2": self.S_2,
        }


class AnalyteSet(list[Analyte]):
    """
    A class representing a set of chemical compounds.
    """
    def to_dicts(self) -> list[dict[str, float]]:
        return [compound.to_dict() for compound in self]

    @classmethod
    def from_dicts(cls, dicts: list[dict[str, float]]) -> "AnalyteSet":
        compound_set = cls()
        for compound_dict in dicts:
            compound = Analyte(
                log_k0=compound_dict["log_k0"],
                area=compound_dict["area"],
                S_1=compound_dict["S_1"],
                S_2=compound_dict.get("S_2"),
            )
            compound_set.append(compound)
        return compound_set

    def to_json(self, file: Path):
        with open(file, "w") as f:
            json.dump(self.to_dicts(), f, indent=4)

    @classmethod
    def from_json(cls, file: Path) -> "AnalyteSet":
        with open(file, "r") as f:
            dicts = json.load(f)
        return cls.from_dicts(dicts)

    @property
    def log_k0(self) -> list[float]:
        return [compound.log_k0 for compound in self]

    @property
    def area(self) -> list[float]:
        return [compound.area for compound in self]

    @property
    def S_1(self) -> list[float]:
        return [compound.S_1 for compound in self]

    @property
    def S_2(self) -> list[float]:
        return [compound.S_2 for compound in self]

    def __hash__(self):
        return hash(tuple((compound.log_k0, compound.area, compound.S_1, compound.S_2) for compound in self))
