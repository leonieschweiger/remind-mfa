"""Overlay comparison plots (multiple named series on a shared region-subplot grid).

Unlike :mod:`remind_mfa.common.parameter_plots`, which auto-plots one input parameter per
page, this module draws an explicit list of caller-supplied series (each already reduced to a
common ``(x, r)`` shape) on top of each other. It is used to compare reconstructed future-MFA
flows against the original input parameters. The low-level styling primitives are shared with
``parameter_plots``.
"""

import logging
import math
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import flodym as fd

from remind_mfa.common.parameter_plots import _COLORS, _LINESTYLES  # noqa: F401 (reused palette)

logger = logging.getLogger(__name__)


class ComparisonPlotsExporter:
    """Multi-series overlay plotter writing to a single multi-page PDF.

    Intended to be used as a context manager::

        with ComparisonPlotsExporter(path, last_hist_year=2019) as exp:
            exp.add_page("Production", "...", series)

    Each entry of ``series`` is a dict with keys ``label``, ``array`` (a
    :class:`flodym.FlodymArray`), ``x_letter`` (``"t"`` or ``"h"``), ``color`` and
    ``linestyle``. Arrays are reduced to ``(x_letter, region_dim)`` for plotting; any leftover
    dimensions are summed over.
    """

    def __init__(self, output_path: str, last_hist_year: Optional[float] = None):
        self.output_path = output_path
        self.last_hist_year = last_hist_year
        self._pdf: Optional[PdfPages] = None

    def __enter__(self) -> "ComparisonPlotsExporter":
        self._pdf = PdfPages(self.output_path)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._pdf is not None:
            self._pdf.close()
            self._pdf = None

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _region_items(series: list, region_dim: str) -> list:
        for s in series:
            if region_dim in s["array"].dims.letters:
                return list(s["array"].dims[region_dim].items)
        return []

    @staticmethod
    def _reduce(arr: fd.FlodymArray, x_letter: str, region_dim: str) -> Optional[fd.FlodymArray]:
        """Reduce an array to a subset of {x_letter, region_dim}, summing over anything else."""
        keep = tuple(l for l in (x_letter, region_dim) if l in arr.dims.letters)
        if x_letter not in keep:
            return None
        if set(arr.dims.letters) != set(keep):
            arr = arr.sum_to(keep)
        return arr

    # ── public API ───────────────────────────────────────────────────────

    def add_page(
        self,
        title: str,
        description: str,
        series: list,
        region_dim: str = "r",
        page_split: Optional[str] = None,
    ):
        """Add one (or, with ``page_split``, several) PDF page(s) overlaying ``series``.

        ``page_split`` names a dimension (e.g. ``"p"``) present on some series arrays; one page
        is produced per item of that dimension, with every series sliced to it first.
        """
        if self._pdf is None:
            raise RuntimeError("ComparisonPlotsExporter must be used as a context manager.")

        page_values = [None]
        if page_split is not None:
            for s in series:
                if page_split in s["array"].dims.letters:
                    page_values = list(s["array"].dims[page_split].items)
                    break

        for page_value in page_values:
            page_series = []
            for s in series:
                arr = s["array"]
                if page_value is not None and page_split in arr.dims.letters:
                    arr = arr[{page_split: page_value}]
                page_series.append({**s, "array": arr})
            page_title = title
            if page_value is not None:
                page_title = f"{title}  —  {page_split}: {page_value}"
            self._draw_page(page_title, description, page_series, region_dim)

    # ── drawing ──────────────────────────────────────────────────────────

    def _draw_page(self, title: str, description: str, series: list, region_dim: str):
        regions = self._region_items(series, region_dim)
        n_region = len(regions) if regions else 1
        ncols = min(4, n_region)
        nrows = math.ceil(n_region / ncols)

        # shared x-limits across all series
        x_all = []
        for s in series:
            arr = self._reduce(s["array"], s["x_letter"], region_dim)
            if arr is not None:
                x_all += list(arr.dims[s["x_letter"]].items)
        shared_xlim = (min(x_all), max(x_all)) if x_all else None

        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(16, 4 * nrows + 1.5), squeeze=False
        )
        fig.suptitle(f"{title}\n{description}", fontsize=9, y=1.0, va="top")

        cells = regions if regions else [None]
        for idx, region in enumerate(cells):
            ax = axes[idx // ncols, idx % ncols]
            if region is not None:
                ax.set_title(str(region), fontsize=8)

            any_negative = False
            for s in series:
                arr = self._reduce(s["array"], s["x_letter"], region_dim)
                if arr is None:
                    continue
                if region is not None and region_dim in arr.dims.letters:
                    arr = arr[{region_dim: region}]
                    arr = self._reduce(arr, s["x_letter"], region_dim)
                x_vals = np.array(arr.dims[s["x_letter"]].items)
                y_vals = np.asarray(arr.values).ravel()
                any_negative = any_negative or bool(np.any(y_vals < 0))
                ax.plot(
                    x_vals,
                    y_vals,
                    color=s["color"],
                    linestyle=s["linestyle"],
                    linewidth=1.4,
                    alpha=0.9,
                    label=s["label"],
                )

            if any_negative:
                ax.axhline(y=0, color="black", linestyle="-", alpha=0.4, linewidth=0.8)
            if self.last_hist_year is not None:
                ax.axvline(
                    x=self.last_hist_year, color="gray", linestyle=":", alpha=0.6, linewidth=0.8
                )
            if shared_xlim:
                ax.set_xlim(*shared_xlim)
            ax.tick_params(labelsize=7)
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=6, loc="best")

        for idx in range(n_region, nrows * ncols):
            axes[idx // ncols, idx % ncols].set_visible(False)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        self._pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
