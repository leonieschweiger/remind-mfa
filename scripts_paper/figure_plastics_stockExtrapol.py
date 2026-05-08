import pickle
import math
import pathlib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from constants import (
    COLOR_PALETTE,
    RUN_PLASTICS,
    PATH_PLASTICS,
    LAST_HISTORICAL_YEAR_PLASTICS,
    RUN_STEEL,
    PATH_STEEL,
    LAST_HISTORICAL_YEAR_STEEL,
    RUN_CEMENT,
    PATH_CEMENT,
    LAST_HISTORICAL_YEAR_CEMENT,
    REGION_DISPLAY_NAMES,
)

MAT = "plastics"

if MAT == "cement":
    RUN_MAT = RUN_CEMENT
    PATH_MAT = PATH_CEMENT
    LAST_HISTORICAL_YEAR_MAT = LAST_HISTORICAL_YEAR_CEMENT
elif MAT == "plastics":
    RUN_MAT = RUN_PLASTICS
    PATH_MAT = PATH_PLASTICS
    LAST_HISTORICAL_YEAR_MAT = LAST_HISTORICAL_YEAR_PLASTICS
    x_lower = 1950
    y_lower = 0
    y_upper = 3.5
elif MAT == "steel":
    RUN_MAT = RUN_STEEL
    PATH_MAT = PATH_STEEL
    LAST_HISTORICAL_YEAR_MAT = LAST_HISTORICAL_YEAR_STEEL
    x_lower = 1920
    y_lower = 4
    y_upper = 12


def _get_column_name(df, target_name: str) -> str:
    for column in df.columns:
        if str(column).strip().lower() == target_name.lower():
            return column
    raise KeyError(f"Could not find column '{target_name}' in dataframe columns {list(df.columns)}")

seen_regions = set()
region_colors = {}
def _get_region_color(region: str) -> str:
    if region not in region_colors:
        region_colors[region] = COLOR_PALETTE[len(region_colors) % len(COLOR_PALETTE)]
    return region_colors[region]


fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=False,
    vertical_spacing=0.17,
    subplot_titles=("a) Over GDP per capita", "b) Over time"),
)

pickle_path = PATH_MAT / f"{RUN_MAT}.pickle"
with pickle_path.open("rb") as file_handle:
    model = pickle.load(file_handle)
    mfa = model.future_mfa

stock = mfa.stocks["in_use"].stock.sum_to(("t", "r"))
stock_pc = stock / mfa.parameters["population"]
stock_pc = stock_pc[{"t": mfa.dims["h"]}].to_df().reset_index()
gdppc = mfa.parameters["gdppc"][{"t": mfa.dims["h"]}].to_df().reset_index()

time_col_stock = _get_column_name(stock_pc, "Historic Time")
region_col_stock = _get_column_name(stock_pc, "Region")
value_col_stock = _get_column_name(stock_pc, "value")

time_col_gdppc = _get_column_name(gdppc, "Historic Time")
region_col_gdppc = _get_column_name(gdppc, "Region")
value_col_gdppc = _get_column_name(gdppc, "value")

stock_gdppc_df = stock_pc.merge(
        gdppc,
        left_on=[time_col_stock, region_col_stock],
        right_on=[time_col_gdppc, region_col_gdppc],
        suffixes=("_stock", "_gdppc"),
    )

for region, region_df in stock_gdppc_df.groupby(region_col_stock):
    region_df = region_df.sort_values(time_col_stock)
    region_code = str(region)
    legend_label = REGION_DISPLAY_NAMES.get(region_code, region_code)
    show_legend = region_code not in seen_regions
    region_color = _get_region_color(region_code)

    fig.add_trace(
        go.Scatter(
            x=region_df[f"{value_col_gdppc}_gdppc"],
            y=region_df[f"{value_col_stock}_stock"],
            mode="lines",
            name=legend_label,
            legendgroup=region_code,
            showlegend=show_legend,
            line={"color": region_color, "width": 3},
        ),
        row=1,
        col=1,
    )
    seen_regions.add(region_code)

    fig.add_trace(
        go.Scatter(
            x=region_df[time_col_stock],
            y=region_df[f"{value_col_stock}_stock"],
            mode="lines",
            name=legend_label,
            legendgroup=region_code,
            showlegend=False,
            line={"color": region_color, "width": 3},
        ),
        row=2,
        col=1,
    )

stock_handler = model.stock_handler
sat_level = model.sector_specific_sat_level
time_factor = model.time_factor

if not hasattr(stock_handler, "pure_regression"):
    raise AttributeError(
        "The loaded pickle does not contain stock_handler.pure_regression. "
        "Please rerun the model with the updated stock_extrapolation code and regenerate the pickle."
    )

def _to_region_df(stock_pc_array):
    region_series = (stock_pc_array * sat_level * time_factor).sum_to(("t", "r"))
    return region_series.to_df().reset_index()

pure_df = _to_region_df(stock_handler.pure_regression)
fitted_df = _to_region_df(stock_handler.fitted_regression)
smoothed_df = _to_region_df(stock_handler.stocks_pc)
gdppc_full_df = mfa.parameters["gdppc"].to_df().reset_index()

time_col_extrap = _get_column_name(pure_df, "Time")
value_col_extrap = _get_column_name(pure_df, "value")
region_col_extrap = _get_column_name(fitted_df, "Region")
time_col_gdppc_full = _get_column_name(gdppc_full_df, "Time")
region_col_gdppc_full = _get_column_name(gdppc_full_df, "Region")
value_col_gdppc_full = _get_column_name(gdppc_full_df, "value")

# Show only future years for regional adaptation and transition smoothing.
fitted_df = fitted_df[fitted_df[time_col_extrap] >= LAST_HISTORICAL_YEAR_MAT].copy()
smoothed_df = smoothed_df[smoothed_df[time_col_extrap] >= LAST_HISTORICAL_YEAR_MAT].copy()

def _merge_extrap_with_gdppc(extrap_df):
    return extrap_df.merge(
        gdppc_full_df,
        left_on=[time_col_extrap, region_col_extrap],
        right_on=[time_col_gdppc_full, region_col_gdppc_full],
        suffixes=("_stock", "_gdppc"),
    )

pure_gdppc_df = _merge_extrap_with_gdppc(pure_df)
fitted_gdppc_df = _merge_extrap_with_gdppc(fitted_df)
smoothed_gdppc_df = _merge_extrap_with_gdppc(smoothed_df)

for region, region_df in pure_gdppc_df.groupby(region_col_extrap):
    region_df = region_df.sort_values(time_col_extrap)
    region_code = str(region)
    legend_label = REGION_DISPLAY_NAMES.get(region_code, region_code)
    show_legend = region_code not in seen_regions
    region_color = _get_region_color(region_code)

    fig.add_trace(
        go.Scatter(
            x=region_df[f"{value_col_gdppc_full}_gdppc"],
            y=region_df[f"{value_col_extrap}_stock"],
            mode="lines",
            name=legend_label,
            legendgroup=region_code,
            showlegend=False,
            line={"color": region_color, "width": 0.5, 'dash':'dot'},
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=region_df[time_col_extrap],
            y=region_df[f"{value_col_extrap}_stock"],
            mode="lines",
            name=legend_label,
            legendgroup=region_code,
            showlegend=False,
            line={"color": region_color, "width": 0.5, 'dash':'dot'},
        ),
        row=2,
        col=1,
    )

for region, region_df in fitted_gdppc_df.groupby(region_col_extrap):
    region_df = region_df.sort_values(time_col_extrap)
    region_code = str(region)
    legend_label = REGION_DISPLAY_NAMES.get(region_code, region_code)
    show_legend = region_code not in seen_regions
    region_color = _get_region_color(region_code)

    fig.add_trace(
        go.Scatter(
            x=region_df[f"{value_col_gdppc_full}_gdppc"],
            y=region_df[f"{value_col_extrap}_stock"],
            mode="lines",
            name=legend_label,
            legendgroup=region_code,
            showlegend=False,
            line={"color": region_color, "width": 1, 'dash':'dash'},
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=region_df[time_col_extrap],
            y=region_df[f"{value_col_extrap}_stock"],
            mode="lines",
            name=legend_label,
            legendgroup=region_code,
            showlegend=False,
            line={"color": region_color, "width": 1, 'dash':'dash'},
        ),
        row=2,
        col=1,
    )

for region, region_df in smoothed_gdppc_df.groupby(region_col_extrap):
    region_df = region_df.sort_values(time_col_extrap)
    region_code = str(region)
    legend_label = REGION_DISPLAY_NAMES.get(region_code, region_code)
    show_legend = region_code not in seen_regions
    region_color = _get_region_color(region_code)

    fig.add_trace(
        go.Scatter(
            x=region_df[f"{value_col_gdppc_full}_gdppc"],
            y=region_df[f"{value_col_extrap}_stock"],
            mode="lines",
            name=legend_label,
            legendgroup=region_code,
            showlegend=False,
            line={"color": region_color, "width": 1},
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=region_df[time_col_extrap],
            y=region_df[f"{value_col_extrap}_stock"],
            mode="lines",
            name=legend_label,
            legendgroup=region_code,
            showlegend=False,
            line={"color": region_color, "width": 1},
        ),
        row=2,
        col=1,
    )

fig.add_shape(
    type="line",
    x0=LAST_HISTORICAL_YEAR_MAT,
    x1=LAST_HISTORICAL_YEAR_MAT,
    y0=y_lower,
    y1=y_upper,
    xref="x2",
    yref="y2",
    line={"dash": "dash", "color": "black"},
)
fig.update_xaxes(
    title_text="GDP per capita [USD 2017]",
    title_standoff=4,
    type="log",
    range=[math.log10(3000), math.log10(120000)],
    row=1,
    col=1,
)
fig.update_xaxes(title_text="Year", title_standoff=4, range=[x_lower, 2100], row=2, col=1)
fig.update_yaxes(
    title_text="In-use stock per capita [t]",
    title_standoff=4,
    range=[y_lower, y_upper],
    row=1,
    col=1,
)
fig.update_yaxes(
    title_text="In-use stock per capita [t]",
    title_standoff=4,
    range=[y_lower, y_upper],
    row=2,
    col=1,
)

figure_height = 750
figure_width = 750

within_group_spacing = 0.055 * 900 / figure_height
title_to_group_spacing = 0.06 * 900 / figure_height

top_hist_group_top_y = 0.8
top_extrap_group_top_y = 0.5

bottom_hist_group_top_y = 0.97
bottom_extrap_group_top_y = 0.55

hist_group_x = 0.02
extrap_group_x = 0.65

fig.update_layout(
        height=figure_height,
        width=figure_width,
        template="plotly_white",
    )

# Save a high-resolution static copy while preserving on-figure relative sizing.
output_path = pathlib.Path(__file__).with_name(f"figure_plastics_stockExtrapol.png")
fig.write_image(
    output_path,
    width=fig.layout.width,
    height=fig.layout.height,
    scale=3,
)

fig.show()

#######################################################################################
# plot effect of timefactor

fig = make_subplots(
    rows=2,
    cols=1,
    vertical_spacing=0.17,
    subplot_titles=("a) without timefactor", "b) with timefactor"),
)

stock_pc = mfa.stocks["in_use"].stock.sum_to(("t", "r", "g")) / mfa.parameters["population"]
stock_time_factor = (stock_pc[{"t": mfa.dims["h"]}] / time_factor[{"t": mfa.dims["h"]}]).sum_to(("h", "r"))
stock_time_factor_df = stock_time_factor.to_df().reset_index()
stock_df = stock_gdppc_df.merge(
        stock_time_factor_df,
        left_on=[time_col_stock, region_col_stock],
        right_on=[time_col_stock, region_col_stock],
    )

for region, region_df in stock_df.groupby(region_col_stock):
    region_df = region_df.sort_values(time_col_stock)
    region_code = str(region)
    legend_label = REGION_DISPLAY_NAMES.get(region_code, region_code)
    show_legend = region_code not in seen_regions
    region_color = _get_region_color(region_code)

    fig.add_trace(
        go.Scatter(
            x=region_df[f"{value_col_stock}_gdppc"],
            y=region_df[f"{value_col_stock}_stock"],
            mode="lines",
            name=legend_label,
            legendgroup=region_code,
            line={"color": region_color},
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=region_df[f"{value_col_stock}_gdppc"],
            y=region_df[f"{value_col_stock}"],
            mode="lines",
            name=legend_label,
            legendgroup=region_code,
            showlegend=False,
            line={"color": region_color},
        ),
        row=2,
        col=1,
    )

fig.update_xaxes(
    title_text="GDP per capita [USD 2017]",
    title_standoff=4,
    type="log",
    range=[math.log10(3000), math.log10(120000)],
    row=1,
    col=1,
)
fig.update_xaxes(
    title_text="GDP per capita [USD 2017]",
    title_standoff=4,
    type="log",
    range=[math.log10(3000), math.log10(120000)],
    row=2,
    col=1,
)
fig.update_yaxes(title_text="In-use stock per capita [t]", title_standoff=4, row=1, col=1)
fig.update_yaxes(title_text="In-use stock per capita divided by time factor[t]", title_standoff=4, row=2, col=1)

fig.update_layout(
        height=figure_height,
        width=figure_width,
        template="plotly_white",
    )

# Save a high-resolution static copy while preserving on-figure relative sizing.
output_path = pathlib.Path(__file__).with_name(f"figure_plastics_timeFactor.png")
fig.write_image(
    output_path,
    width=fig.layout.width,
    height=fig.layout.height,
    scale=3,
)

fig.show()