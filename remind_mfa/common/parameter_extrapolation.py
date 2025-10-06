import flodym as fd
from abc import ABC, abstractmethod
from typing import Optional, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from remind_mfa.common.common_cfg import GeneralCfg
from remind_mfa.common.assumptions_doc import add_assumption_doc


class ParameterExtrapolation(ABC):
    """Base class from which new parameter extrapolations can be implemented."""

    @abstractmethod
    def fill_future_values(self, old_param: fd.Parameter, new_param: fd.Parameter) -> fd.Parameter:
        """Sets values of new_param based on extrapolation method."""
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a description of the extrapopation."""
        raise NotImplementedError

    def extrapolate(self, parameter: fd.Parameter, extended_time: fd.Dimension) -> fd.Parameter:
        """Extrapolate parameter to extended time dimension, fill with future values using extrapolation method."""

        new_param = self.initialize_empty_parameter(parameter, extended_time)
        new_param = self.fill_future_values(parameter, new_param)
        # overwrite historic values with old parameter values
        new_param[{"t": parameter.dims["h"]}] = parameter

        return new_param

    @staticmethod
    def initialize_empty_parameter(
        parameter: fd.Parameter, extended_time: fd.Dimension
    ) -> fd.Parameter:
        """Initialize a new parameter with extended time dimension."""

        if not "h" in parameter.dims.letters:
            raise ValueError(
                f"Parameter {parameter.name} does not have historic time dimension 'h'"
            )
        if not "t" == extended_time.letter:
            raise ValueError(f"New time dimension does not have letter 't'")

        new_dims = parameter.dims.replace("h", extended_time)
        new_param = fd.Parameter(dims=new_dims, name=parameter.name)
        return new_param


class ConstantExtrapolation(ParameterExtrapolation):
    """Keep parameter constant at last observed value."""

    def fill_future_values(
        self, old_param: fd.Parameter, new_param: fd.FlodymArray
    ) -> fd.Parameter:
        add_assumption_doc(
            type="model switch",
            name=f"Keep {old_param.name} constant",
            description=self.description,
        )

        # get last historic value
        last_historic_year = old_param.dims["h"].items[-1]
        last_value = old_param[{"h": last_historic_year}]

        # set values to last historic value
        new_param[...] = last_value.cast_to(new_param.dims)

        return new_param

    @property
    def description(self) -> str:
        return "Parameter is kept constant into the future at last observed value."

class LinearToTargetExtrapolation(ParameterExtrapolation):
    """
    Extrapolate linearly from the last observed value to a target value in a target year,
    then keep it constant afterwards.
    """

    def __init__(self, target_value: float, target_year: int):
        self._target_value = target_value
        self._target_year = target_year

    def fill_future_values(
        self, old_param: fd.Parameter, new_param: fd.FlodymArray
    ) -> fd.Parameter:
        add_assumption_doc(
            type="model switch",
            name=f"Linear extrapolation of {old_param.name} to target",
            description=self.description,
        )

        # Get last historic year + value
        last_historic_year = old_param.dims["h"].items[-1]
        last_value = old_param[{"h": last_historic_year}]

        # Iterate through future years
        for year in new_param.dims["t"].items:
            if year <= last_historic_year:
                continue  # skip historic
            elif year < self._target_year:
                # Linear interpolation between last value and target
                frac = (year - last_historic_year) / (self._target_year - last_historic_year)
                value = (last_value + frac * (self._target_value - last_value)).copy()
            elif year >= self._target_year:
                # After (and including) target year: hold constant
                value = last_value.copy() * 0 + self._target_value

            new_param[{"t": year}] = value

        return new_param

    @property
    def description(self) -> str:
        return (
            f"Parameter is linearly extrapolated to {self._target_value} in "
            f"{self._target_year} and then kept constant."
        )
    
class ParameterExtrapolationManager:
    """Manager for applying parameter extrapolations."""

    def __init__(
        self,
        cfg: "GeneralCfg",
        extended_time: fd.Dimension,
    ):
        self.parameter_extrapolation_classes = cfg.model_switches.parameter_extrapolation_classes
        self.extended_time = extended_time

    def apply_prm_extrapolation(
        self,
        parameters: Dict[str, fd.Parameter],
    ) -> Dict[str, fd.Parameter]:
        """Apply extrapolation to parameters. Only those listed in parameter_extrapolation in config model switches are adjusted."""

        modified_parameters = parameters.copy()

        if self.parameter_extrapolation_classes is None:
            return modified_parameters

        for param_name, extrapolation_class in self.parameter_extrapolation_classes.items():
            if param_name not in modified_parameters:
                raise ValueError(f"Parameter '{param_name}' not found in parameters.")

            modified_parameters[param_name] = extrapolation_class().extrapolate(
                modified_parameters[param_name], self.extended_time
            )

        return modified_parameters
