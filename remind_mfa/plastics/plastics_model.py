import numpy as np
import flodym as fd

from .plastics_mfa_system import PlasticsMFASystemFuture
from .plastics_mfa_system_historic import PlasticsMFASystemHistoric
from .plastics_export import PlasticsDataExporter
from .plastics_visualization import PlasticsVisualizer
from .plastics_definition import get_plastics_definition
from .plastics_mappings import PlasticsDimensionFiles, PlasticsDisplayNames
from remind_mfa.plastics.plastics_definition import scenario_parameters as plastics_scn_prm_def
from remind_mfa.plastics.plastics_config import PlasticsCfg
from remind_mfa.common.common_model import CommonModel
from remind_mfa.common.assumptions_doc import add_assumption_doc
from remind_mfa.common.stock_extrapolation import StockExtrapolation
from remind_mfa.common.data_transformations import Bound, BoundList


class PlasticsModel(CommonModel):

    ConfigCls = PlasticsCfg
    DimensionFilesCls = PlasticsDimensionFiles
    DataExporterCls = PlasticsDataExporter
    VisualizerCls = PlasticsVisualizer
    DisplayNamesCls = PlasticsDisplayNames
    HistoricMFASystemCls = PlasticsMFASystemHistoric
    FutureMFASystemCls = PlasticsMFASystemFuture
    get_definition = staticmethod(get_plastics_definition)
    custom_scn_prm_def = plastics_scn_prm_def

    # TODO: unify, then delete
    end_use_good_letter: str = "g"
    historic_stock_name: str = "in_use_historic"
    stock_projection_saturation_level: int = 3 #TODO replace this first guess

    def modify_parameters(self):
        # copy/rename for use in common model
        self.parameters["sector_split_limit"] = self.parameters["sector_split"]
        # cast lifetime mean to correct dimensions for use in common model
        self.parameters["lifetime_mean"] = fd.Parameter(
            dims=self.dims["t", "g"],
            values=self.parameters["lifetime_mean"].cast_to(self.dims["t", "g"]).values,
        )
        # Conversion Mt -> t
        # TODO: move to mrmfa
        self.parameters["primary_his_imports"][...] *= 1e6
        self.parameters["primary_his_exports"][...] *= 1e6
        self.parameters["final_his_imports"][...] *= 1e6
        self.parameters["final_his_exports"][...] *= 1e6
        self.parameters["waste_his_imports"][...] *= 1e6
        self.parameters["waste_his_exports"][...] *= 1e6
        self.parameters["consumption"][...] *= 1e6

    def transfer_historic_parameters(self):
        # get material split of stock inflow from historic MFA to be extrapolated by ParameterExtrapolation for use in future MFA
        self.parameters["material_shares_use_inflow"] = self.historic_mfa.parameters["material_shares_use_inflow"]

        indep_fit_dim_letters = ("g",)
        arr = fd.FlodymArray(
            dims=self.dims[indep_fit_dim_letters],
        )
        # get saturation levels
        growth_rate_bound_gdp = Bound(
                var_name="x1_growth_rate",
                lower_bound=fd.FlodymArray.full_like(arr, 0.),
                upper_bound=fd.FlodymArray.full_like(arr, np.inf),
            )
        growth_rate_bound_time = Bound(
            var_name="x2_growth_rate",
            lower_bound=fd.FlodymArray.full_like(arr, 0.),
            upper_bound=fd.FlodymArray.full_like(arr, np.inf),
        )
        bound_list_obj = BoundList(
            target_dims=self.dims[self.end_use_good_letter,],
            bound_list=[growth_rate_bound_gdp, growth_rate_bound_time],
        )
        historic_stocks = self.historic_mfa.stocks["in_use_historic"].stock
        self.stock_handler_0 = StockExtrapolation(
            cfg=self.cfg.model_switches,
            historic_stocks=historic_stocks,
            dims=self.dims,
            parameters=self.parameters,
            target_dim_letters="all",
            indep_fit_dim_letters=(self.end_use_good_letter,),
            bound_list=bound_list_obj,
        )
        self.stock_handler_0.set_dims(indep_fit_dim_letters)
        self.stock_handler_0.init_arrays()
        self.stock_handler_0.calc_arrays_from_parameters_dict()
        self.stock_handler_0.set_predictor()
        self.stock_handler_0.get_pure_regression()
        sat_idx = self.stock_handler_0.extrapolation.prm_names.index('saturation_level')
        sat_levels = self.stock_handler_0.extrapolation._fit_prms[:, sat_idx]

