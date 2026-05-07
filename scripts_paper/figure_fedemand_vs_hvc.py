"""Figure: REMIND ue_chemicals demand vs. HVC=>polymerization flow, indexed to 2020.

For each region one subplot with lines (all indexed to BASE_YEAR=1):
  - HVC_input => polymerization (model, solid)
  - ue_chemicals from f_fedemand SSP2 (dashed)
  - Methanol, Ammonia (IEA Petrochem, dotted)
  - Ethylene+Propylene+BTX sum (IEA Petrochem, dashdot)
  - Fertilizer (MAgPIE SSP2.rcp45, longdashdot)
"""

import pathlib
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from constants import (
    COLOR_PALETTE,
    COLORS_REMIND,
    LAST_HISTORICAL_YEAR_PLASTICS,
    PATH_PLASTICS,
    REGION_DISPLAY_NAMES,
)

RUN_PLASTICS = "model_plastics_SSP2_h12_2026-05-07--10-11-54"
INPUT_DIR = pathlib.Path("data/plastics/input")
OUTPUT_PATH = pathlib.Path(__file__).with_name("figure_fedemand_vs_hvc.png")
BASE_YEAR = 2020


# ── helpers ───────────────────────────────────────────────────────────────────
def read_iea(filename: str) -> pd.DataFrame:
    """Read IEA Petrochem cs4r (format: X{year},region,value,val)."""
    df = pd.read_csv(
        INPUT_DIR / filename,
        comment="*",
        header=None,
        names=["year", "region", "var", "value"],
    )
    df["year"] = df["year"].str.replace("X", "", regex=False).astype(int)
    return df[["year", "region", "value"]]


def index_to_base(df: pd.DataFrame, val_col: str = "value") -> pd.Series:
    """Return a Series of values divided by their BASE_YEAR value per region.

    If BASE_YEAR is not present in the data, the base value is linearly
    interpolated from the two nearest surrounding years.
    """
    if BASE_YEAR in df["year"].values:
        base = df[df["year"] == BASE_YEAR].set_index("region")[val_col]
    else:
        # interpolate per region between the two nearest years
        lo = df[df["year"] <= BASE_YEAR].groupby("region").last().reset_index()
        hi = df[df["year"] >= BASE_YEAR].groupby("region").first().reset_index()
        merged = lo.merge(hi, on="region", suffixes=("_lo", "_hi"))
        merged["base"] = np.where(
            merged["year_lo"] == merged["year_hi"],
            merged[f"{val_col}_lo"],
            merged[f"{val_col}_lo"] + (merged[f"{val_col}_hi"] - merged[f"{val_col}_lo"])
            * (BASE_YEAR - merged["year_lo"])
            / (merged["year_hi"] - merged["year_lo"]),
        )
        base = merged.set_index("region")["base"]
    return df.apply(
        lambda row: row[val_col] / base[row["region"]]
        if row["region"] in base and base[row["region"]] != 0
        else np.nan,
        axis=1,
    )


# ── load model flow ───────────────────────────────────────────────────────────
pickle_path = PATH_PLASTICS / f"{RUN_PLASTICS}.pickle"
with pickle_path.open("rb") as fh:
    mfa = pickle.load(fh).future_mfa

flow = mfa.flows["HVC_input => polymerization"].sum_to(("t", "r"))
flow_df = flow.to_df().reset_index()
time_col   = next(c for c in flow_df.columns if str(c).strip().lower() in ("time", "t"))
region_col = next(c for c in flow_df.columns if str(c).strip().lower() in ("region", "r"))
val_col    = next(c for c in flow_df.columns if str(c).strip().lower() == "value")
flow_df = flow_df.rename(columns={time_col: "year", region_col: "region", val_col: "value"})
flow_df["year"] = flow_df["year"].astype(int)
flow_df["idx"] = index_to_base(flow_df)

# ── load f_fedemand (SSP2, ue_chemicals) ─────────────────────────────────────
fed_df = pd.read_csv(
    INPUT_DIR / "f_fedemand.cs4r",
    comment="*",
    header=None,
    names=["year", "region", "scenario", "variable", "value"],
)
fed_df = fed_df[(fed_df["scenario"] == "SSP2") & (fed_df["variable"] == "ue_chemicals")].copy()
fed_df["year"] = fed_df["year"].astype(int)
fed_df["idx"] = index_to_base(fed_df)

# ── load IEA Petrochem files ──────────────────────────────────────────────────
methanol_df  = read_iea("leonie_IEA_Petrochem_production5type_Methanol.cs4r")
ammonia_df   = read_iea("leonie_IEA_Petrochem_production5type_Ammonia.cs4r")
ethylene_df  = read_iea("leonie_IEA_Petrochem_production5type_Ethylene.cs4r")
propylene_df = read_iea("leonie_IEA_Petrochem_production5type_Propylene.cs4r")
btx_df       = read_iea("leonie_IEA_Petrochem_production5type_BTX.cs4r")

epb_df = (
    pd.concat([ethylene_df, propylene_df, btx_df])
    .groupby(["year", "region"], as_index=False)["value"]
    .sum()
)

for df in (methanol_df, ammonia_df, epb_df):
    df["idx"] = index_to_base(df)

# ── load Fertilizer (wide format, SSP2.rcp45) ────────────────────────────────
fert_raw = pd.read_csv(INPUT_DIR / "leonie_MAgPIEReport_fertilizer.cs4r", comment="*", header=0)
fert_raw.columns = fert_raw.columns.str.strip()
fert_df = fert_raw[["dummy", "dummy.1", "SSP2.rcp45"]].copy()
fert_df.columns = ["year_raw", "region", "value"]
fert_df["year"] = fert_df["year_raw"].str.replace("y", "", regex=False).astype(int)
fert_df = fert_df[["year", "region", "value"]]
fert_df["idx"] = index_to_base(fert_df)

# ── series catalogue: (label, dataframe, line_kwargs, marker_symbol) ─────────
SERIES = [
    ("HVC→polymerization (model)",      flow_df,     {"width": 2.5},                        "circle"),
    ("ue_chemicals (f_fedemand SSP2)",   fed_df,      {"width": 2,   "dash": "dash"},        "circle-open"),
    ("Methanol (IEA)",                   methanol_df, {"width": 1.5, "dash": "dot"},         "diamond-open"),
    ("Ammonia (IEA)",                    ammonia_df,  {"width": 1.5, "dash": "dot"},         "square-open"),
    ("Ethylene+Propylene+BTX (IEA)",     epb_df,      {"width": 1.5, "dash": "dashdot"},     "triangle-up-open"),
    ("Fertilizer (MAgPIE SSP2.rcp45)",  fert_df,     {"width": 1.5, "dash": "longdashdot"}, "star-open"),
]

# All series use fixed colours (series 0 = HVC model, same across all panels)
SERIES_COLORS = ["#000000", "#444444", "#e6194b", "#4363d8", "#f58231", "#3cb44b"]

# ── build figure ──────────────────────────────────────────────────────────────
regions = sorted(flow_df["region"].unique())
n_cols = 3
n_rows = -(-len(regions) // n_cols)

fig = make_subplots(
    rows=n_rows,
    cols=n_cols,
    shared_yaxes=False,
    horizontal_spacing=0.08,
    vertical_spacing=0.12,
    subplot_titles=[REGION_DISPLAY_NAMES.get(r, r) for r in regions],
)

legend_added = [False] * len(SERIES)

for i, region in enumerate(regions):
    row = i // n_cols + 1
    col = i % n_cols + 1
    region_color = COLORS_REMIND.get(region, COLOR_PALETTE[i % len(COLOR_PALETTE)])

    for s_idx, (label, df, line_style, marker_sym) in enumerate(SERIES):
        color = SERIES_COLORS[s_idx]
        rdf = df[df["region"] == region].sort_values("year")
        if rdf.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=rdf["year"],
                y=rdf["idx"],
                mode="lines+markers",
                name=label,
                legendgroup=label,
                showlegend=not legend_added[s_idx],
                line={"color": color, **line_style},
                marker={"symbol": marker_sym, "size": 4, "color": color},
            ),
            row=row, col=col,
        )
        legend_added[s_idx] = True

    fig.add_vline(x=LAST_HISTORICAL_YEAR_PLASTICS, line_dash="dot",
                  line_color="gray", line_width=1, row=row, col=col)
    fig.add_hline(y=1.0, line_dash="dot", line_color="lightgray",
                  line_width=1, row=row, col=col)

    fig.update_xaxes(title_text="Year", title_standoff=4, range=[2010, 2100], row=row, col=col)
    fig.update_yaxes(title_text=f"Index ({BASE_YEAR}=1)", title_standoff=4, row=row, col=col)

fig.update_layout(
    template="plotly_white",
    width=1200,
    height=400 * n_rows,
    title_text=(
        "<b>REMIND chemical demand vs. REMIND-MFA plastics demand</b>"
        f"<br><sup>All indexed to {BASE_YEAR} = 1</sup>"
    ),
    title_x=0.5,
    font={"size": 11},
    legend={"x": 1.01, "y": 1, "xanchor": "left", "tracegroupgap": 4},
    margin={"l": 60, "r": 210, "t": 90, "b": 60},
)

fig.write_image(
    OUTPUT_PATH,
    width=fig.layout.width,
    height=fig.layout.height,
    scale=3,
)
print(f"Saved figure to {OUTPUT_PATH}")
fig.show()
