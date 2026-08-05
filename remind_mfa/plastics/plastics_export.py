import flodym as fd
import pandas as pd
from typing import TYPE_CHECKING

from remind_mfa.common.common_export import CommonDataExporter, IamcVariable
from remind_mfa.common.comparison_plots import ComparisonPlotsExporter
from remind_mfa.common.parameter_plots import _COLORS

if TYPE_CHECKING:
    from remind_mfa.plastics.plastics_model import PlasticsModel


class PlasticsDataExporter(CommonDataExporter):

    def export_custom(self, model: "PlasticsModel"):
        if self.cfg.csv.do_export:
            self.export_eol_data_by_region_and_year(mfa=model.future_mfa)
            self.export_use_data_by_region_and_year(mfa=model.future_mfa)
            self.export_recycling_data_by_region_and_year(mfa=model.future_mfa)
            self.export_stock_extrapolation(model=model)
        if self.cfg.flow_comparison_plots.do_export:
            self.export_flow_parameter_comparison(model=model)
            self.export_historic_supply_vs_trade(model=model)

    def export_stock_extrapolation(self, model: "PlasticsModel"):
        model.stock_handler.pure_parameters.to_df().to_csv(
            self.export_path("csv", "stock_extrapolation_parameters.csv")
        )
        model.stock_handler.bound_list.bound_list[0].upper_bound.to_df().to_csv(
            self.export_path("csv", "stock_extrapolation_saturationLevel.csv")
        )

    def export_eol_data_by_region_and_year(self, mfa: fd.MFASystem):
        eol_data = (
            mfa.flows["eol => collected"]
            + mfa.flows["waste_market => collected"]
            - mfa.flows["collected => waste_market"]
        )
        df = eol_data.sum_to(("t", "r", "m")).to_df(index=True)
        df.to_csv(self.export_path("csv", "eol_by_region_year.csv"), index=True)

    def export_use_data_by_region_and_year(self, mfa: fd.MFASystem):
        df = mfa.stocks["in_use"].inflow.sum_to(("t", "r")).to_df(index=True)
        df.to_csv(self.export_path("csv", "use_by_region_year.csv"), index=True)

    def export_recycling_data_by_region_and_year(self, mfa: fd.MFASystem):
        recl_data = mfa.flows["collected => reclmech"] + mfa.flows["collected => reclchem"]
        df = recl_data.sum_to(("t", "r", "m")).to_df(index=True)
        df.to_csv(self.export_path("csv", "recycling_by_region_year.csv"), index=True)

    # ── flow-vs-parameter comparison plots ───────────────────────────────

    @staticmethod
    def _series(label: str, array: fd.FlodymArray, x_letter: str, color_idx: int, linestyle: str):
        """Build a ComparisonPlotsExporter series entry with a palette colour."""
        return {
            "label": label,
            "array": array,
            "x_letter": x_letter,
            "color": _COLORS[color_idx % len(_COLORS)],
            "linestyle": linestyle,
        }

    def material_type_indicator(self, model: "PlasticsModel") -> fd.FlodymArray:
        """(p, m) indicator: 1 where material m belongs to polymer type p, else 0.

        Derived from the nonzero pattern of ``sector_polymer_split`` (each material maps to
        exactly one polymer type), so no material→type mapping needs to be hardcoded.
        """
        split = model.parameters["sector_polymer_split"].sum_to(("p", "m"))
        return fd.Parameter(dims=split.dims, values=(split.values > 0).astype(float))

    @staticmethod
    def _lift_to_type(arr_trm: fd.FlodymArray, indicator: fd.FlodymArray) -> fd.FlodymArray:
        """Aggregate a material-resolved array (…, m) up to polymer type: (…, p)."""
        return (arr_trm * indicator).sum_over("m")

    def export_flow_parameter_comparison(self, model: "PlasticsModel"):
        """Overlay reconstructed future-MFA production/trade flows on the original input params."""
        flw = model.future_mfa.flows
        prm = model.historic_parameters  # pre-extrapolation snapshot = original inputs
        indicator = self.material_type_indicator(model)
        last_hist_year = float(model.dims["h"].items[-1])

        with ComparisonPlotsExporter(
            output_path=self.export_path("flow_comparison_plots", "flow_parameter_comparison.pdf"),
            last_hist_year=last_hist_year,
        ) as exp:
            # ---- primary production ----
            future_prod = (
                flw["polymerization => primary_market"] + flw["reclmech => primary_market"]
            )
            exp.add_page(
                title="Primary production: input parameter vs future MFA (region totals)",
                description=(
                    "Input 'production' vs future MFA (polymerization + reclmech) => primary_market, "
                    "summed over element/material/type."
                ),
                series=[
                    self._series("production (input)", prm["production"].sum_to(("h", "r")), "h", 0, "--"),
                    self._series(
                        "polymerization+reclmech (future)", future_prod.sum_to(("t", "r")), "t", 1, "-"
                    ),
                    self._series(
                        "polymerization (future)",
                        flw["polymerization => primary_market"].sum_to(("t", "r")),
                        "t", 2, ":",
                    ),
                    self._series(
                        "reclmech (future)",
                        flw["reclmech => primary_market"].sum_to(("t", "r")),
                        "t", 3, ":",
                    ),
                ],
            )
            exp.add_page(
                title="Primary production: input parameter vs future MFA (by polymer type)",
                description="Input 'production' vs future (polymerization + reclmech), lifted to polymer type.",
                series=[
                    self._series("production (input)", prm["production"], "h", 0, "--"),
                    self._series(
                        "polymerization+reclmech (future)",
                        self._lift_to_type(future_prod.sum_to(("t", "r", "m")), indicator),
                        "t", 1, "-",
                    ),
                ],
                page_split="p",
            )

            # ---- trade ----
            trade_specs = [
                ("primary", "primary_market => exports", "imports => primary_market",
                 "primary_his_exports", "primary_his_imports"),
                ("final", "good_market => exports", "imports => good_market",
                 "final_his_exports", "final_his_imports"),
                ("waste", "waste_market => exports", "imports => waste_market",
                 "waste_his_exports", "waste_his_imports"),
            ]
            for cat, exp_flow, imp_flow, exp_par, imp_par in trade_specs:
                # exports use colour 0, imports colour 1; input = dashed, future = solid
                exp.add_page(
                    title=f"{cat} trade: input parameter vs future MFA (region totals)",
                    description="Historic trade parameter vs future MFA trade flow, summed over element/material/type.",
                    series=[
                        self._series(f"{cat} exports (input)", prm[exp_par].sum_to(("h", "r")), "h", 0, "--"),
                        self._series(f"{cat} exports (future)", flw[exp_flow].sum_to(("t", "r")), "t", 0, "-"),
                        self._series(f"{cat} imports (input)", prm[imp_par].sum_to(("h", "r")), "h", 1, "--"),
                        self._series(f"{cat} imports (future)", flw[imp_flow].sum_to(("t", "r")), "t", 1, "-"),
                    ],
                )
                exp.add_page(
                    title=f"{cat} trade: input parameter vs future MFA (by polymer type)",
                    description="Historic trade parameter vs future MFA trade flow, lifted to polymer type.",
                    series=[
                        self._series(f"{cat} exports (input)", prm[exp_par].sum_to(("h", "r", "p")), "h", 0, "--"),
                        self._series(
                            f"{cat} exports (future)",
                            self._lift_to_type(flw[exp_flow].sum_to(("t", "r", "m")), indicator),
                            "t", 0, "-",
                        ),
                        self._series(f"{cat} imports (input)", prm[imp_par].sum_to(("h", "r", "p")), "h", 1, "--"),
                        self._series(
                            f"{cat} imports (future)",
                            self._lift_to_type(flw[imp_flow].sum_to(("t", "r", "m")), indicator),
                            "t", 1, "-",
                        ),
                    ],
                    page_split="p",
                )

    def export_historic_supply_vs_trade(self, model: "PlasticsModel"):
        """Historic input params only: production vs primary exports/imports by region and type.

        The ``net = production + imports - exports`` line goes negative exactly where historic
        exports exceed domestic supply — i.e. where ``scale_trade_exports_to_supply`` in the
        historic MFA has to cap exports to keep fabrication inflow non-negative.
        """
        prm = model.historic_parameters
        last_hist_year = float(model.dims["h"].items[-1])

        production = prm["production"]  # (h, r, p)
        exports = prm["primary_his_exports"].sum_over("m")  # (h, r, p)
        imports = prm["primary_his_imports"].sum_over("m")  # (h, r, p)
        net = production + imports - exports

        with ComparisonPlotsExporter(
            output_path=self.export_path("flow_comparison_plots", "historic_supply_vs_trade.pdf"),
            last_hist_year=last_hist_year,
        ) as exp:
            exp.add_page(
                title="Historic primary supply vs trade (by polymer type)",
                description=(
                    "net = production + imports - exports. Where net < 0, fabrication inflow would go "
                    "negative unless historic exports are scaled down to supply."
                ),
                series=[
                    self._series("production", production, "h", 0, "-"),
                    self._series("primary exports", exports, "h", 1, "-"),
                    self._series("primary imports", imports, "h", 2, "-"),
                    self._series("net (prod + imp - exp)", net, "h", 3, "--"),
                ],
                page_split="p",
            )

    def iamc_variables(self) -> list[IamcVariable]:
        return [
            # production
            IamcVariable(
                variable_name="Production|Chemicals|Plastics|Primary",  # PRISMA nomenclature
                calculation_function=lambda mfa: (
                    mfa.flows["polymerization => primary_market"].sum_to(("t", "r"))
                    - mfa.flows["reclchem => HVC_input"]
                ),
                unit="t/yr",
            ),
            IamcVariable(
                variable_name="Production|Chemicals|Plastics|Secondary",  # PRISMA nomenclature
                calculation_function=lambda mfa: (
                    mfa.flows["reclmech => primary_market"] + mfa.flows["reclchem => HVC_input"]
                ).sum_to(("t", "r")),
                unit="t/yr",
            ),
            # demand by good
            IamcVariable(
                variable_name="Material Demand|Chemicals|Plastics",  # PRISMA nomenclature
                calculation_function=lambda mfa: mfa.stocks["in_use"].inflow.sum_to(
                    ("t", "r", "g")
                ),
                unit="t/yr",
                split_name="Good",
            ),
            # demand per capita
            IamcVariable(
                variable_name="Material Demand|Chemicals|Plastics|Per Capita",
                calculation_function=lambda mfa: (
                    mfa.stocks["in_use"].inflow / mfa.parameters["population"]
                ).sum_to(("t", "r")),
                unit="t/cap/yr",
                region_weight="Population",
            ),
            # trade
            IamcVariable(
                variable_name="Import|Industry|Chemicals|Plastics|Primary Forms",  # CIRCOMOD nomenclature (further differentiated by stage)
                calculation_function=lambda mfa: mfa.flows["imports => primary_market"].sum_to(
                    ("t", "r")
                ),
                unit="t/yr",
            ),
            IamcVariable(
                variable_name="Export|Industry|Chemicals|Plastics|Primary Forms",  # CIRCOMOD nomenclature (further differentiated by stage)
                calculation_function=lambda mfa: mfa.flows["primary_market => exports"].sum_to(
                    ("t", "r")
                ),
                unit="t/yr",
            ),
            IamcVariable(
                variable_name="Import|Industry|Chemicals|Plastics|Goods",  # CIRCOMOD nomenclature (further differentiated by stage)
                calculation_function=lambda mfa: mfa.flows["imports => good_market"].sum_to(
                    ("t", "r", "g")
                ),
                unit="t/yr",
                split_name="Good",
            ),
            IamcVariable(
                variable_name="Export|Industry|Chemicals|Plastics|Goods",  # CIRCOMOD nomenclature (further differentiated by stage)
                calculation_function=lambda mfa: mfa.flows["good_market => exports"].sum_to(
                    ("t", "r", "g")
                ),
                unit="t/yr",
                split_name="Good",
            ),
        ]

    def iamc_aggregates(self) -> list[str]:
        # Primary + Secondary are separate specs (no `per`), so their parent must be
        # aggregated explicitly. "Material Demand|Chemicals|Plastics" is handled
        # automatically via its `per="Good"` split.
        return ["Production|Chemicals|Plastics"]
