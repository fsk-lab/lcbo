from typing import Optional
from pathlib import Path
import matplotlib.pyplot as plt
from moses.datatypes import Chromatogram, LCGradient, PeakList
from moses import graphy


def plot_chromatogram(
        chrom: Chromatogram,
        file: str | Path,
        peaks: Optional[PeakList] = None,
        gradient: Optional[LCGradient] = None,
) -> None:
    """
    Plot a chromatogram using `moses.graphy`, and optionally overlay detected peaks and the eluent program.

    Args:
        chrom (moses.datatypes.Chromatogram): The chromatogram data to plot.
        file (str | Path): The file path to save the plot.
        peaks (Optional[moses.datatypes.PeakList]): Optional list of detected peaks to overlay on the chromatogram.
        gradient (Optional[moses.datatypes.LCGradient]): Optional eluent gradient to overlay on the chromatogram.
    """
    fig, ax = graphy.subplots()
    ax.plot(peaks, chrom) if peaks is not None else ax.plot(chrom)
    if gradient is not None:
        ax2 = ax.twinx()
        ax2.plot(gradient, zorder=-10, linestyle="--")
    plt.savefig(file, dpi=300, transparent=True)
    plt.close()
