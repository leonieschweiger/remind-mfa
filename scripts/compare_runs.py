import pickle
import flodym as fd
import flodym.export as fde
import pathlib
import questionary

DIRECTORY = "data/steel/output/export/pickle" 
FLOW_NAME = "forming => ip_market" #"ip_market => exports" #"imports => ip_market"
IS_STOCK = False
RUNS = ["model_steel_SSP2_h12_2026-08-12--13-35-36", "model_steel_SSP2_h12_2026-08-12--10-06-12"] 
# RUNS = None
LABELS = ["new", "old"]
SUBPLOT_DIM_LETTER = "r"  # e.g. "r" for region, or None for no subplots

# When True, ignore FLOW_NAME/IS_STOCK and instead compare sector splits
# (consumption = in_use stock inflow, turned into shares over the good/sector dim).
COMPARE_SECTOR_SPLITS = False
GOOD_LETTER = "g"  # good/sector dim letter: "g" for steel/plastics, "s" for cement

DIRECTORY = pathlib.Path(DIRECTORY)

if not RUNS:
    # Interactively choose files from all model_*.pickle files in the directory
    available_files = sorted(DIRECTORY.glob("model_*.pickle"))

    if not available_files:
        raise FileNotFoundError(f"No model_*.pickle files found in: {DIRECTORY}")

    run_file_names = questionary.checkbox(
        "Select run files to compare:",
        choices=[file.name for file in available_files],
        validate=lambda selected: True if selected else "Select at least one file.",
    ).ask()

    if not run_file_names:
        raise ValueError("No files selected. Aborting comparison.")

else:
    run_file_names = [f"{run_name}.pickle" for run_name in RUNS]

run_file_paths = [DIRECTORY / file_name for file_name in run_file_names]
if not LABELS:
    LABELS = [pathlib.Path(f).stem for f in run_file_names]

if RUNS is not None and len(RUNS) != len(run_file_names):
    raise ValueError("run_names must have the same length as selected files")


new_dim = fd.Dimension(letter="X", name="Run", items=LABELS)

mfas = []
for pickle_path in run_file_paths:
    with pickle_path.open("rb") as file_handle:
        mfas.append(pickle.load(file_handle).future_mfa)

if COMPARE_SECTOR_SPLITS:
    # Sector split = consumption (in_use inflow) as shares over the good/sector dim.
    # Keep consumption absolute until region aggregation, then compute shares last:
    # summing per-region share fractions over regions is not the global share.
    consumptions = [
        mfa.stocks["in_use"].inflow.sum_to(("t", "r", GOOD_LETTER)) for mfa in mfas
    ]
    stacked = fd.flodym_array_stack(consumptions, dimension=new_dim)  # (t, X, r, g)

    # Global: subplot per sector, one line per run.
    global_shares = stacked.sum_to(("t", "X", GOOD_LETTER)).get_shares_over((GOOD_LETTER,))
    plotter = fde.PlotlyArrayPlotter(
        array=global_shares,
        title="Sector splits across runs (global)",
        intra_line_dim="t",
        linecolor_dim="X",
        subplot_dim=GOOD_LETTER,
        ylabel="Share",
    )
    fig = plotter.plot()
    fig.show()

    # Per region: PlotlyArrayPlotter allows only 3 dims, so slice one figure per region.
    for region in stacked.dims["r"].items:
        region_shares = stacked[{"r": region}].get_shares_over((GOOD_LETTER,))  # (t, X, g)
        plotter = fde.PlotlyArrayPlotter(
            array=region_shares,
            title=f"Sector splits across runs ({region})",
            intra_line_dim="t",
            linecolor_dim="X",
            subplot_dim=GOOD_LETTER,
            ylabel="Share",
        )
        fig = plotter.plot()
        fig.show()

else:
    if IS_STOCK:
        arrays = [mfa.stocks[FLOW_NAME].stock for mfa in mfas]
    else:
        arrays = [mfa.flows[FLOW_NAME] for mfa in mfas]
    comparison_array = fd.flodym_array_stack(arrays, dimension=new_dim)

    if SUBPLOT_DIM_LETTER:
        plotter = fde.PlotlyArrayPlotter(
            array=comparison_array.sum_to(("t", "X", SUBPLOT_DIM_LETTER)),
            title=f"Comparison of {FLOW_NAME} across runs",
            intra_line_dim="t",
            linecolor_dim="X",
            subplot_dim="r",
        )
        fig = plotter.plot()
        fig.show()

    plotter = fde.PlotlyArrayPlotter(
        array=comparison_array.sum_to(("t", "X")),
        title=f"Comparison of {FLOW_NAME} across runs",
        intra_line_dim="t",
        linecolor_dim="X",
    )
    fig = plotter.plot()
    fig.show()
