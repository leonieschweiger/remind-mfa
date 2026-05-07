"""Figure: Global plastics demand — REMIND-MFA model vs. literature validation sources.

Reproduces the comparison from PlasticsVisualizer.compare_demand() as a standalone
publication-quality script in the style of figure_1.py.
"""

import pathlib
import pickle

import pandas as pd
import plotly.graph_objects as go

from constants import (
    COLOR_PALETTE,
    LAST_HISTORICAL_YEAR_PLASTICS,
    PATH_PLASTICS,
)

RUN_PLASTICS = "model_plastics_SSP2_h12_2026-05-07--10-11-54"
VALIDATION_CSV = pathlib.Path("data/plastics/input/validation.csv")
OUTPUT_PATH = pathlib.Path(__file__).with_name("figure_demand_validation.png")

# ── colours ──────────────────────────────────────────────────────────────────
MODEL_COLOR = "#000000"          # black for the model line
SOURCE_COLORS = COLOR_PALETTE    # cycle through palette for literature sources

# ── load model output ─────────────────────────────────────────────────────────
pickle_path = PATH_PLASTICS / f"{RUN_PLASTICS}.pickle"
with pickle_path.open("rb") as fh:
    mfa = pickle.load(fh).future_mfa

demand_model = (
    mfa.stocks["in_use"]
    .inflow
    .sum_to(("t",))          # global total, time only
)
demand_df = demand_model.to_df().reset_index()

# normalise column names (FlodymArray.to_df may use "Time" or "t")
time_col = next(c for c in demand_df.columns if str(c).strip().lower() in ("time", "t"))
val_col  = next(c for c in demand_df.columns if str(c).strip().lower() == "value")

demand_df = demand_df.rename(columns={time_col: "year", val_col: "value_t"})
demand_df["value_Mt"] = demand_df["value_t"] / 1e6   # t → Mt

# ── load validation data ──────────────────────────────────────────────────────
val_df = pd.read_csv(VALIDATION_CSV, sep=";")
val_df["year"]  = pd.to_numeric(val_df["year"],  errors="coerce")
val_df["value"] = pd.to_numeric(val_df["value"], errors="coerce")  # already in Mt

# ── build figure ──────────────────────────────────────────────────────────────
fig = go.Figure()

# model line
fig.add_trace(
    go.Scatter(
        x=demand_df["year"],
        y=demand_df["value_Mt"],
        mode="lines",
        name="REMIND-MFA",
        line={"color": MODEL_COLOR, "width": 2},
    )
)

# historical / future split marker
fig.add_vline(
    x=LAST_HISTORICAL_YEAR_PLASTICS,
    line_dash="dash",
    line_color="black",
    line_width=1,
)
fig.add_annotation(
    x=LAST_HISTORICAL_YEAR_PLASTICS,
    y=1.02,
    xref="x",
    yref="paper",
    text="end of<br>historical data",
    showarrow=False,
    font={"size": 10, "color": "black"},
    xanchor="left",
)

# literature sources (markers only — they are sparse point estimates)
sources = val_df["source"].unique()
for i, source in enumerate(sources):
    src_df = val_df[val_df["source"] == source].sort_values("year")
    color  = SOURCE_COLORS[i % len(SOURCE_COLORS)]
    fig.add_trace(
        go.Scatter(
            x=src_df["year"],
            y=src_df["value"],
            mode="markers+lines",
            name=source,
            line={"color": color, "dash": "dot", "width": 1.5},
            marker={"color": color, "size": 7, "symbol": "circle"},
        )
    )

# ── layout ────────────────────────────────────────────────────────────────────
fig.update_layout(
    template="plotly_white",
    width=960,
    height=540,
    xaxis={"title": "Year", "range": [1950, 2100]},
    yaxis={"title": "Global plastics demand [Mt/yr]"},
    legend={"title": "Source", "x": 0.02, "y": 0.98, "xanchor": "left", "yanchor": "top"},
    font={"size": 13},
    margin={"l": 60, "r": 30, "t": 30, "b": 60},
)

# ── export & show ─────────────────────────────────────────────────────────────
fig.write_image(
    OUTPUT_PATH,
    width=fig.layout.width,
    height=fig.layout.height,
    scale=3,
)
print(f"Saved figure to {OUTPUT_PATH}")
fig.show()
