"""Figure: HVC input to polymerization flow by region."""

import pathlib
import pickle

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from constants import (
    LAST_HISTORICAL_YEAR_PLASTICS,
    PATH_PLASTICS,
    REGION_DISPLAY_NAMES,
)

RUN_PLASTICS = "model_plastics_SSP2_h12_2026-05-07--10-11-54"
FLOW_NAME = "HVC_input => polymerization"
INPUT_DIR = pathlib.Path("data/plastics/input")
OUTPUT_PATH = pathlib.Path(__file__).with_name("figure_hvc_polymerization.png")

# ── load model output ─────────────────────────────────────────────────────────
pickle_path = PATH_PLASTICS / f"{RUN_PLASTICS}.pickle"
with pickle_path.open("rb") as fh:
    mfa = pickle.load(fh).future_mfa

flow = mfa.flows[FLOW_NAME].sum_to(("t", "r"))  # sum over 'e' (element)
flow_df = (flow / 1e6).to_df().reset_index()     # t → Mt

time_col   = next(c for c in flow_df.columns if str(c).strip().lower() in ("time", "t"))
region_col = next(c for c in flow_df.columns if str(c).strip().lower() in ("region", "r"))
val_col    = next(c for c in flow_df.columns if str(c).strip().lower() == "value")
flow_df = flow_df.rename(columns={time_col: "year", region_col: "region", val_col: "value"})
flow_df["year"] = flow_df["year"].astype(int)

# ── compute HVC flow from REMIND: ue_chemicals * ue_share_hvc / mat2ue_hvc ───
# f_fedemand SSP2 ue_chemicals
fed_df = pd.read_csv(
    INPUT_DIR / "f_fedemand.cs4r",
    comment="*", header=None,
    names=["year", "region", "scenario", "variable", "value"],
)
fed_df = fed_df[(fed_df["scenario"] == "SSP2") & (fed_df["variable"] == "ue_chemicals")].copy()
fed_df["year"] = fed_df["year"].astype(int)

# mat2ue for hvc, all years
mat_raw = pd.read_csv(
    INPUT_DIR / "p37_mat2ue_chemicals.cs4r",
    comment="*", header=None,
    names=["year", "region", "flow", "ue", "factor"],
)
mat_raw["year"] = mat_raw["year"].astype(int)
mat_hvc = mat_raw[mat_raw["flow"] == "hvc"][["year", "region", "factor"]].copy()

# AllChem_Flow in 2020 for hvc (to compute ue_share)
flow_raw = pd.read_csv(
    INPUT_DIR / "p37_AllChem_Flow_Value_2005_2020.cs4r",
    comment="*", header=None,
    names=["year", "region", "flow", "value_gt"],
)
flow_raw["year"] = flow_raw["year"].astype(int)
hvc_2020 = flow_raw[(flow_raw["year"] == 2020) & (flow_raw["flow"] == "hvc")][["region", "value_gt"]]
mat_hvc_2020 = mat_hvc[mat_hvc["year"] == 2020][["region", "factor"]].rename(columns={"factor": "mat2ue_2020"})
fed_2020 = fed_df[fed_df["year"] == 2020][["region", "value"]].rename(columns={"value": "ue_2020"})

ue_share_hvc = (
    hvc_2020
    .merge(mat_hvc_2020, on="region")
    .merge(fed_2020, on="region")
)
ue_share_hvc["ue_share"] = ue_share_hvc["value_gt"] * ue_share_hvc["mat2ue_2020"] / ue_share_hvc["ue_2020"]
ue_share_hvc = ue_share_hvc[["region", "ue_share"]]

# REMIND HVC flow = ue_chemicals * ue_share / mat2ue  [Gt → Mt: ×1000]
remind_hvc = (
    fed_df[["year", "region", "value"]]
    .merge(ue_share_hvc, on="region")
    .merge(mat_hvc, on=["year", "region"])
)
remind_hvc["value_mt"] = remind_hvc["value"] * remind_hvc["ue_share"] / remind_hvc["factor"] * 1000  # Gt → Mt

# ── load IEA Petrochem: Ethylene + Propylene + BTX sum ───────────────────────
def read_iea(filename: str) -> pd.DataFrame:
    df = pd.read_csv(
        INPUT_DIR / filename,
        comment="*", header=None,
        names=["year", "region", "var", "value_mt"],
    )
    df["year"] = df["year"].str.replace("X", "", regex=False).astype(int)
    return df[["year", "region", "value_mt"]]

epb_df = (
    pd.concat([
        read_iea("leonie_IEA_Petrochem_production5type_Ethylene.cs4r"),
        read_iea("leonie_IEA_Petrochem_production5type_Propylene.cs4r"),
        read_iea("leonie_IEA_Petrochem_production5type_BTX.cs4r"),
    ])
    .groupby(["year", "region"], as_index=False)["value_mt"]
    .sum()
)

regions = sorted(flow_df["region"].unique())
n_cols = 3
n_rows = -(-len(regions) // n_cols)  # ceiling division

fig = make_subplots(
    rows=n_rows,
    cols=n_cols,
    shared_yaxes=False,
    horizontal_spacing=0.08,
    vertical_spacing=0.10,
    subplot_titles=[REGION_DISPLAY_NAMES.get(r, r) for r in regions],
)

COLOR_MODEL  = "#000000"   # black  – REMIND-MFA model flow
COLOR_REMIND = "#e6194b"   # red    – REMIND implied HVC
COLOR_EPB    = "#4363d8"   # blue   – IEA Ethylene+Propylene+BTX

model_legend_added = False
remind_legend_added = False
epb_legend_added = False

for i, region in enumerate(regions):
    row = i // n_cols + 1
    col = i % n_cols + 1

    region_df = flow_df[flow_df["region"] == region].sort_values("year")

    fig.add_trace(
        go.Scatter(
            x=region_df["year"],
            y=region_df["value"]/0.88, # 0.88 of total HVC goes into plastics globally according to Levi and Cullen
            mode="lines",
            name="HVC demand (REMIND-MFA)",
            legendgroup="model",
            showlegend=not model_legend_added,
            line={"color": COLOR_MODEL, "width": 2},
        ),
        row=row,
        col=col,
    )
    model_legend_added = True

    remind_df = remind_hvc[remind_hvc["region"] == region].sort_values("year")
    fig.add_trace(
        go.Scatter(
            x=remind_df["year"],
            y=remind_df["value_mt"],
            mode="lines+markers",
            name="HVC flow (REMIND: ue_chem × ue_share / mat2ue)",
            legendgroup="remind",
            showlegend=not remind_legend_added,
            line={"color": COLOR_REMIND, "width": 2, "dash": "dash"},
            marker={"symbol": "circle-open", "size": 5},
        ),
        row=row,
        col=col,
    )
    remind_legend_added = True

    epb_region = epb_df[epb_df["region"] == region].sort_values("year")
    fig.add_trace(
        go.Scatter(
            x=epb_region["year"],
            y=epb_region["value_mt"],
            mode="lines+markers",
            name="Ethylene+Propylene+BTX (IEA)",
            legendgroup="epb",
            showlegend=not epb_legend_added,
            line={"color": COLOR_EPB, "width": 2, "dash": "dot"},
            marker={"symbol": "diamond-open", "size": 5},
        ),
        row=row,
        col=col,
    )
    epb_legend_added = True

    fig.add_vline(
        x=LAST_HISTORICAL_YEAR_PLASTICS,
        line_dash="dash",
        line_color="gray",
        line_width=1,
        row=row,
        col=col,
    )

    fig.update_xaxes(title_text="Year", title_standoff=4, range=[1950, 2100], row=row, col=col)
    fig.update_yaxes(title_text="Flow [Mt/yr]", title_standoff=4, row=row, col=col)

fig.update_layout(
    template="plotly_white",
    width=1100,
    height=380 * n_rows,
    title_text=f"HVC demand by region",
    title_x=0.5,
    font={"size": 12},
    margin={"l": 60, "r": 200, "t": 80, "b": 60},
    legend={"x": 1.01, "y": 1, "xanchor": "left"},
)

fig.write_image(
    OUTPUT_PATH,
    width=fig.layout.width,
    height=fig.layout.height,
    scale=3,
)
print(f"Saved figure to {OUTPUT_PATH}")
fig.show()
