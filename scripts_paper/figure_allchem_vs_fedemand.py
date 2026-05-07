"""Figure: p37_AllChem_Flow × mat2ue conversion factors vs. f_fedemand ue_chemicals.

Each region subplot shows:
  - Stacked area: individual chemical flows (Gt) × mat2ue ($/kg) = T$
  - Line: f_fedemand SSP2 ue_chemicals (T$ = $tn)

Unit algebra: flow (Gt) × factor ($/kg) = flow × 1e12 kg × factor ($/kg) × 1e-12 T$/$ = flow × factor [T$]

mat2ue is only available from 2020 onward; for historic AllChem_Flow years (2005-2020)
the 2020 conversion factor is used as the nearest available value.
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

INPUT_DIR = pathlib.Path("data/plastics/input")
OUTPUT_PATH = pathlib.Path(__file__).with_name("figure_allchem_vs_fedemand.png")

# ── load p37_AllChem_Flow ────────────────────────────────────────────────────
flow_raw = pd.read_csv(
    INPUT_DIR / "p37_AllChem_Flow_Value_2005_2020.cs4r",
    comment="*",
    header=None,
    names=["year", "region", "flow", "value_gt"],
)
flow_raw["year"] = flow_raw["year"].astype(int)

# ── load p37_mat2ue_chemicals ────────────────────────────────────────────────
mat_raw = pd.read_csv(
    INPUT_DIR / "p37_mat2ue_chemicals.cs4r",
    comment="*",
    header=None,
    names=["year", "region", "flow", "ue", "factor"],
)
mat_raw["year"] = mat_raw["year"].astype(int)

# For each (region, flow) keep only the 2020 factor (earliest available,
# used as proxy for all historic years)
mat_2020 = mat_raw[mat_raw["year"] == 2020][["region", "flow", "factor"]].copy()

# OtherChem has no mat2ue entry — add it with a constant factor of 1
regions_list = flow_raw["region"].unique()
other_rows = pd.DataFrame({"region": regions_list, "flow": "OtherChem", "factor": 1.0})
mat_2020 = pd.concat([mat_2020, other_rows], ignore_index=True)

# Also extend mat_future with factor=1 for OtherChem for all future years
other_future = (
    mat_raw[mat_raw["year"] > 2020][["year", "region"]]
    .drop_duplicates()
    .assign(flow="OtherChem", factor=1.0)
)
mat_future_ext = pd.concat([mat_raw[mat_raw["year"] > 2020], other_future], ignore_index=True)

# ── load f_fedemand (SSP2, ue_chemicals) ─────────────────────────────────────
fed_df = pd.read_csv(
    INPUT_DIR / "f_fedemand.cs4r",
    comment="*",
    header=None,
    names=["year", "region", "scenario", "variable", "value"],
)
fed_df = fed_df[(fed_df["scenario"] == "SSP2") & (fed_df["variable"] == "ue_chemicals")].copy()
fed_df["year"] = fed_df["year"].astype(int)

# ── build a combined dataset covering 2005-2100 ───────────────────────────────
# For years <= 2020: actual flow × mat2ue(2020)  [T$]
hist = flow_raw.merge(mat_2020, on=["region", "flow"], how="inner")
hist["value_tbn"] = hist["value_gt"] * hist["factor"]
hist = hist[["year", "region", "flow", "value_tbn"]]

# ue_share per (region, flow) = mat2ue(2020) * flow(2020) / ue_chemicals(2020)
# This is just hist[year==2020] value / ue_chemicals(2020)
fed_2020 = fed_df[(fed_df["scenario"] == "SSP2") & (fed_df["variable"] == "ue_chemicals") & (fed_df["year"] == 2020)].set_index("region")["value"]
hist_2020 = hist[hist["year"] == 2020].copy()
hist_2020["ue_share"] = hist_2020.apply(lambda r: r["value_tbn"] / fed_2020[r["region"]], axis=1)
ue_share = hist_2020[["region", "flow", "ue_share"]]

# For years > 2020: mat2ue(t) × flow(t) = ue_chemicals(t) × ue_share  (mat2ue cancels)
# sum over all flows = ue_chemicals(t) × sum(ue_share) = ue_chemicals(t) × 1
fed_future = fed_df[fed_df["year"] > 2020][["year", "region", "value"]].copy()
fut = fed_future.merge(ue_share, on="region")
fut["value_tbn"] = fut["value"] * fut["ue_share"]
fut = fut[["year", "region", "flow", "value_tbn"]]

chem_df = pd.concat([hist, fut], ignore_index=True).sort_values(["region", "flow", "year"])

# ── display names and colours for each flow ───────────────────────────────────
FLOW_DISPLAY = {
    "hvc":       "HVC",
    "ammoFinal": "Ammonia (final)",
    "fertilizer":"Fertilizer",
    "methFinal": "Methanol (final)",
    "OtherChem": "Other chemicals",
}
FLOW_COLORS = {
    "hvc":       "#4363d8",
    "ammoFinal": "#e6194b",
    "fertilizer":"#3cb44b",
    "methFinal": "#f58231",
    "OtherChem": "#aaaaaa",
}
FLOWS_ORDERED = ["hvc", "ammoFinal", "fertilizer", "methFinal", "OtherChem"]

# ── build figure ──────────────────────────────────────────────────────────────
regions = sorted(fed_df["region"].unique())
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

flow_legend_added = set()
fed_legend_added = False

for i, region in enumerate(regions):
    row = i // n_cols + 1
    col = i % n_cols + 1

    # stacked area — one trace per flow, each region
    for flow_name in FLOWS_ORDERED:
        sub = chem_df[(chem_df["region"] == region) & (chem_df["flow"] == flow_name)].sort_values("year")
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["year"],
                y=sub["value_tbn"],
                name=FLOW_DISPLAY.get(flow_name, flow_name),
                legendgroup=flow_name,
                showlegend=flow_name not in flow_legend_added,
                mode="lines",
                stackgroup="chem",
                fillcolor=FLOW_COLORS[flow_name],
                line={"color": FLOW_COLORS[flow_name], "width": 0.5},
            ),
            row=row, col=col,
        )
        flow_legend_added.add(flow_name)

    # f_fedemand line
    fdf = fed_df[fed_df["region"] == region].sort_values("year")
    fig.add_trace(
        go.Scatter(
            x=fdf["year"],
            y=fdf["value"],
            name="ue_chemicals (f_fedemand SSP2)",
            legendgroup="fedemand",
            showlegend=not fed_legend_added,
            mode="lines",
            line={"color": "#000000", "width": 2, "dash": "dash"},
        ),
        row=row, col=col,
    )
    fed_legend_added = True

    fig.add_vline(x=LAST_HISTORICAL_YEAR_PLASTICS, line_dash="dot",
                  line_color="gray", line_width=1, row=row, col=col)

    fig.update_xaxes(title_text="Year", title_standoff=4, range=[2005, 2100], row=row, col=col)
    fig.update_yaxes(title_text="T$ (2017)", title_standoff=4, row=row, col=col)

fig.update_layout(
    template="plotly_white",
    width=1200,
    height=400 * n_rows,
    title_text=(
        "<b>Chemical material value flows (T$)</b>"
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
